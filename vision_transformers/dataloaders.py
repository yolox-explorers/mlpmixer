import os
# from absl import logging
import pandas as pd
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms
from tensorflow.keras.utils import to_categorical
from vision_transformers.yolact_edge.ye_inference import YE_Inference
from vision_transformers.data_transforms import get_image_only_aug_transform


class Mlpmixer_Dataset(Dataset):
    def __init__(self, args, subset, aug_transforms, model):
        self.args = args
        self.data = pd.read_pickle(args.image_catalog_path)
        self.data = self.data[self.data['station']==args.station]
        self.aug_transforms = aug_transforms # albumentations
        self.model = model # Instance Segmentation model

        if args.time == "night":
            self.data = self.data.loc[self.data["time_of_day"] == "night"]
        else:
            self.data = self.data.loc[self.data["time_of_day"] != "night"]
        
        print(f"\t\tStation: {args.station}\n\
            Time of day: {self.args.time}\n\
            Data size: {self.data.shape}")

        assert args.train_dataset in ['train', 'train+val', 'train+val+test'], \
            "train_dataset only accept 'train', 'train+val', 'train+val+test'"
        
        # Splits dataset according to train, train+val or train+val+test
        if args.train_dataset == 'train':
            if subset == 'train':
                self.data = self.data.loc[self.data['train_val_test'] == 'train']
            elif subset == 'val':
                self.data = self.data.loc[self.data['train_val_test'] == 'val']
        elif args.train_dataset == 'train+val':
            if subset == 'train':
                self.data = self.data.loc[self.data['train_val_test'] != 'test']
            elif subset == 'val':
                self.data = self.data.loc[self.data['train_val_test'] == 'test']            
        elif args.train_dataset == 'train+val+test':
            if subset == 'train':
                self.data = self.data
            elif subset == 'val':
                self.data = self.data.loc[self.data['train_val_test'] == 'test'] 

        if args.image_sample_n > 0:
            self.data = self.data.sample(n = args['train']['image_sample_n'])

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):

        img_fpath = self.data.iloc[idx]['img_fpath']
        img = cv2.imread(img_fpath)
        resize_factor = self.args.image_resize_factor
        img_height = img.shape[0]
        img_new_height = int(img_height * resize_factor)
        img_width = img.shape[1]
        img_new_width = int(img_width * resize_factor)
        image = cv2.resize(img, (img_new_width, img_new_height))

        # Performs augmentation for training
        if self.aug_transforms:
            image = self.aug_transforms(image = image)['image']

        # Mask, bbox generation by YE
        p, _ = self.model.predict(image)
        label = self.data.iloc[idx]['pass_fail']
        label = 1 if label == 'on' else 0

        if p[0] is not None:
            cropped_img = p[0][0]
            # saves cropped images if save folder path is given 
            if self.args.save_cropped_images:
                assert os.path.exists(self.args.cropped_images_folder), "save cropped images folder does not exist"
                savepath = os.path.join(self.args.cropped_images_folder, os.path.basename(img_fpath))
                cv2.imwrite(savepath, cropped_img.cpu().numpy()*255)
                print(f"Image {os.path.basename(img_fpath)} saved to {savepath}")

            cropped_img = cropped_img.permute(2, 0, 1)

            # Resize again for cropped mask ROI
            image_ROI = transforms.Resize((self.args.crop_size, self.args.crop_size))(cropped_img)

            return image_ROI, label
        
        else:
            image_ROI = torch.zeros((3, self.args.crop_size, self.args.crop_size), device = torch.device('cuda'))
            
            return image_ROI, label

class DataLoader(nn.Module):
    def __init__(self, args):
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.args = args

    def dataloader(self):
        """Generates dataloader for train, val and test dataset

        Returns:
            data_loader_train: dataloader for model training
            data_loader_val: dataloader for model validation during model training
            data_loader_test: dataloader for model validation
        """

        # Initialise YE model
        model_inference = YE_Inference(
            weights=self.args.ye_weights_path,
            disable_tensorrt=True,
            YE_inference_only=True,
            method=self.args.model_name,
            station=self.args.station,
            crop_option=self.args.crop_option,
            crop_margin_factor=self.args.crop_margin_factor,
            crop_option_a_retain_background=self.args.crop_option_a_retain_background,
            crop_option_b_alpha=self.args.crop_option_b_alpha,
            crop_option_c_vertical=self.args.crop_option_c_vertical,
            crop_option_c_horizontal=self.args.crop_option_c_horizontal
            )

        # numpy to JAX format
        dataset_train = Mlpmixer_Dataset(self.args, subset = 'train', \
            aug_transforms = get_image_only_aug_transform(self.args) if self.args.augment else None, 
            model = model_inference)
        dataset_val = Mlpmixer_Dataset(self.args, subset = 'val', \
            aug_transforms = None, model = model_inference)

        print(f"len(dataset (train)): {len(dataset_train)}")
        print(f"len(dataset (val)): {len(dataset_val)}")

        def collate_fn(batch):
            batch = list(filter(lambda x: x is not None, batch))
            return torch.utils.data.dataloader.default_collate(batch)

        data_loader_train = torch.utils.data.DataLoader(
            dataset_train,
            batch_size=self.args.train_batch_size,
            shuffle=self.args.train_shuffle,
            num_workers=self.args.train_num_workers,
            collate_fn = collate_fn)

        data_loader_val = torch.utils.data.DataLoader(
            dataset_val,
            batch_size=self.args.eval_batch_size,
            shuffle=self.args.test_shuffle,
            num_workers=self.args.test_num_workers,
            collate_fn = collate_fn)

        try:
            torch.multiprocessing.set_start_method('fork',force=True)
        except RuntimeError:
            pass

        return data_loader_train, data_loader_val


