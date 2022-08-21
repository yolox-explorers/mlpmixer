import os
import argparse
# from absl import logging
import logging
import flax
from flax.training import checkpoints as flax_checkpoints
import jax
import ast
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import tqdm
import random
import sys
import subprocess
import albumentations as A
import cv2
import torch
import torch.nn as nn
import torch.multiprocessing as mp
from datetime import datetime
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from tensorflow.keras.utils import to_categorical
from vision_transformers.yolact_edge.ye_inference import YE_Inference
from vision_transformers.vit_jax import checkpoint
from vision_transformers.vit_jax import input_pipeline
from vision_transformers.vit_jax import utils
from vision_transformers.vit_jax import models
from vision_transformers.vit_jax import momentum_clip
from vision_transformers.vit_jax import train
from vision_transformers.vit_jax.configs import common as common_config
from vision_transformers.vit_jax.configs import models as models_config
from vision_transformers.mlpmixer_helper import labelnames, make_label_getter, show_img, show_img_grid, get_accuracy, get_accuracy_mixer
from vision_transformers.data_transforms import get_image_only_aug_transform

parser = argparse.ArgumentParser(description='MLPmixer training Script')

parser.add_argument('--num_classes', default = 2, type = int)
parser.add_argument('--model_name', default = 'Mixer-B_16', type = str, choices = ['ViT-B_32', 'Mixer-B_16'],\
    help = "Model name either 'ViT-B_32' or 'Mixer-B_16'")
parser.add_argument('--model_path', \
    default="/home/user/MLPmixer/vision_transformers/model_weights/imagenet21k_Mixer-B_16.npz", 
    type=str, help='file path for model weights')
parser.add_argument('--epochs', default = 30, type = int,
                    help = "number of epochs for training")
parser.add_argument('--validation_epochs', default=1, type=int,
                    help='number of epochs before validation on validation set')
parser.add_argument('--patience', default = 20, type = int,
                    help = "Early stopping steps before training stops")
parser.add_argument('--total_steps', default = 0, type = int,
                    help = "number of steps for training. Set as 0 to use all training images for training")
parser.add_argument('--warmup_steps', default = 5, type = int)
parser.add_argument('--decay_type', default = 'cosine', type = str)
parser.add_argument('--grad_norm_clip', default = 1.0, type = float)
parser.add_argument('--accum_steps', default = 8, type = int,\
    help = "This controls in how many forward passes the batch is split. 8 works well with a TPU runtime that has 8 devices. \
        64 should work on a GPU. You can of course also adjust the batch_size above, but that would require you to adjust the learning rate accordingly.")
parser.add_argument('--base_lr', default = 0.03, type = float)
parser.add_argument('--image_catalog_path', default = '/home/user/MLPmixer/data/extracted/clean_img_data.pkl', type = str)
parser.add_argument('--model_save', default = True, type = ast.literal_eval, dest = 'model_save')
parser.add_argument('--model_save_dir', \
    default = "/home/user/MLPmixer/vision_transformers/model_weights/checkpoint_weights", type = str)
parser.add_argument('--seed', default = 42, type = int)
parser.add_argument('--station', default = 'slope', type = str, choices = ['slope', 'park'])
parser.add_argument('--time', default = 'day', type = str, choices = ['day', 'night'])
parser.add_argument('--image_resize_factor', default = 0.1809, type = float,\
    help = "Scale factor to resize training images to feed to YE model for inference")
parser.add_argument('--train_batch_size', default = 8, type = int)
parser.add_argument('--train_shuffle', default = True, \
    type = ast.literal_eval, dest='train_shuffle')
parser.add_argument('--train_num_workers', default = 0, type = int)
parser.add_argument('--test_batch_size', default = 8, type = int)
parser.add_argument('--test_shuffle', default = False, \
    type = ast.literal_eval, dest = 'test_shuffle')
parser.add_argument('--test_num_workers', default = 0, type = int)
parser.add_argument('--image_sample_n', default = 0, type = int,\
    help = "number of images to sample for training. default as 0 for full training dataset")
parser.add_argument('--train_dataset', default = 'train', type = str, \
    choices = ['train', 'train+val', 'train+val+test'], 
    help = "'train' for 80 percent training, 'train+val' for 90 percent training, 'train+val+test' for 100 percent training")
parser.add_argument('--train_mode', default = True, \
    type = ast.literal_eval, dest = 'train_mode', \
    help = "Set to 'True' to enter training mode, 'False' to disable model training")

parser.add_argument('--num_gpus', default=1, type=int,
                    help='Number of GPUs used in training')

# YE generated mask options
parser.add_argument('--ye_weights_path', type = str, help = 'filepath for ye_weights')
parser.add_argument('--ye_model_config', type = str)
parser.add_argument('--crop_size', default = 224, type = int, \
    help = "size of cropped masks from YE model to feed to MLP/ViT model for training & inference")
parser.add_argument('--save_cropped_images', default = False, \
    type = ast.literal_eval, dest = "save_cropped_images")
parser.add_argument('--cropped_images_folder', 
    default = "/home/user/MLPmixer/vision_transformers/ye_images/", \
    type = str, help = 'folder to save cropped masks generated by YE model')
parser.add_argument('--crop_option', default = "A", type = str)
parser.add_argument('--crop_margin_factor', default = 0.1, type = float)
parser.add_argument('--crop_option_a_retain_background', default = False, \
    type = ast.literal_eval, dest = 'crop_option_a_retain_background')
parser.add_argument('--crop_option_b_alpha', default = 0, type = float)
parser.add_argument('--crop_option_c_vertical', default = 0.5, type = float)
parser.add_argument('--crop_option_c_horizontal', default = 0.05, type = float)

# augmentation options
parser.add_argument('--augment', type = ast.literal_eval, \
    default = False, dest = 'augment', \
    help = "If set to 'True', augmented images will be used for model training. Default set as 'False'")
parser.add_argument('--rotate', default = 0, type = int, help = 'degrees of rotation') # default no rotation. # 3, 5, 8
parser.add_argument('--brightness', default = 0, type = float, help = 'percentage change to brightness') # default no adjustment to brightness. # 0.3, 0.5, 0.8
parser.add_argument('--contrast', default = 0, type = float, help = 'percentage change to contrast') # default no adjustment to contrast. # 0.3, 0.5, 0.8
parser.add_argument('--shift', default = 0, type = float, help = 'percentage shift') # default no shift. # 0.3, 0.5, 0.8
parser.add_argument('--shear', default = 0, type = int, help = "angle of shear") # default no shear # 3, 5, 8
parser.add_argument('--zoom_min', default = 1.0, type = float, help = "minimum zoom factor") # default no zoom
parser.add_argument('--zoom_max', default = 1.0, type = float, help = "maximum zoom factor") # default no zoom

parser.set_defaults(keep_latest=False)

# CUDA_TENSORS_ON_WORKER = True

# Load dataset
class config():
    def __init__(self, args):
        self.args = args

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
            image_ROI = transforms.Resize((self.args.crop_size, self.args.crop_size))(cropped_img).permute(2, 1, 0)
            # print("Before resize:", cropped_img.shape, "After resize:", image_ROI.shape)
            
            data = {"image": image_ROI, "label": label}
            return data
        
        else:
            image_ROI = torch.zeros((self.args.crop_size, self.args.crop_size,3), \
                device = torch.device('cuda'))
            data = {"image": image_ROI, "label": label}
            # print("No car mask predicted")
            return data

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
            batch_size=self.args.test_batch_size,
            shuffle=self.args.test_shuffle,
            num_workers=self.args.test_num_workers,
            collate_fn = collate_fn)

        try:
            torch.multiprocessing.set_start_method('fork',force=True)
        except RuntimeError:
            pass

        return data_loader_train, data_loader_val


class MLPmixer_train():
    def __init__(self):
        pass

    def train(self, args, ds_train, ds_val, vit_apply_repl, params):
        # Model finetuning
        '''
        100 Steps take approximately 15 minutes in the TPU runtime.
        This controls in how many forward passes the batch is split. 8 works well with
        a TPU runtime that has 8 devices. 64 should work on a GPU. You can of course
        also adjust the batch_size above, but that would require you to adjust the
        learning rate accordingly.
        '''
        print(f"ds_train: {len(ds_train)}")
        print(f"ds_val: {len(ds_val)}")

        # Controlling number of steps or batches of training images per epoch for training
        if args.total_steps != 0:
            total_train_steps = args.total_steps 
        else:
            total_train_steps = len(ds_train)
        total_val_steps = len(ds_val)

        # Check out train.make_update_fn in the editor on the right side for details.
        train_lr_fn = utils.create_learning_rate_schedule(total_train_steps, args.base_lr, args.decay_type, args.warmup_steps)
        train_update_fn_repl = train.make_update_fn(apply_fn=model.apply, accum_steps=args.accum_steps, lr_fn=train_lr_fn)

        # We use a momentum optimizer that uses half precision for state to save
        # memory. It also implements the gradient clipping.
        train_opt = momentum_clip.Optimizer(grad_norm_clip=args.grad_norm_clip).create(params)
        train_opt_repl = flax.jax_utils.replicate(train_opt)

        # Initialize PRNGs for dropout.
        train_update_rng_repl = flax.jax_utils.replicate(jax.random.PRNGKey(0))

        train_labels = []
        train_predicted = []
        train_losses = []
        train_lrs = []
        
        val_lr_fn = utils.create_learning_rate_schedule(total_val_steps, args.base_lr, args.decay_type, args.warmup_steps)
        val_update_fn_repl = train.make_update_fn(apply_fn=model.apply, accum_steps=args.accum_steps, lr_fn=val_lr_fn)
        val_opt = momentum_clip.Optimizer(grad_norm_clip=args.grad_norm_clip).create(params)
        val_opt_repl = flax.jax_utils.replicate(val_opt)
        val_update_rng_repl = flax.jax_utils.replicate(jax.random.PRNGKey(0))     
        val_labels = []
        val_losses = []
        val_lrs = []
        best_val_loss = 1e4
        best_epoch = 0

        now = datetime.now().strftime("%d%m%Y-%H%M%S") # current date and time

        # Completes in ~20 min on the TPU runtime .
        for epoch in range(args.epochs):
            train_loss = 0
            train_lr = 0
            for step, batch in zip(tqdm.trange(1, total_train_steps), ds_train):
                batch['image'] = batch['image'].cpu().numpy()
                batch['label'] = batch['label'].cpu().numpy()
                train_labels.extend(batch['label'])

                # Changing shape of batch['image'] to folloow (1, train_batch_size, width, height, channels)
                if len(batch['image'].shape) == 3:
                    batch['image'] = np.expand_dims(np.expand_dims(batch['image'], 0), 0)
                elif len(batch['image'].shape) == 4:
                    batch['image'] = np.expand_dims(batch['image'], 0)

                # changing shape of labels to follow (1, train_batch_size, num_class)
                batch['label'] = np.expand_dims(to_categorical(batch['label'], num_classes=2), 0)
                
                train_opt_repl, train_loss_repl, train_update_rng_repl = train_update_fn_repl(
                    train_opt_repl, flax.jax_utils.replicate(step), batch, train_update_rng_repl)

                train_loss += train_loss_repl[0]
                train_lr += train_lr_fn(step)

            train_losses.append(np.asarray(train_loss/total_train_steps).item())
            train_lrs.append(np.asarray(train_lr/total_train_steps).item())

            # Val loss
            val_loss = 0
            val_lr = 0
            for step, batch in zip(tqdm.trange(1, total_val_steps), ds_val):
                batch['image'] = batch['image'].cpu().numpy()
                batch['label'] = batch['label'].cpu().numpy()
                val_labels.append(batch['label'])

                if len(batch['image'].shape) == 3:
                    batch['image'] = np.expand_dims(np.expand_dims(batch['image'], 0), 0)
                elif len(batch['image'].shape) == 4:
                    batch['image'] = np.expand_dims(batch['image'], 0)

                batch['label'] = np.expand_dims(to_categorical(batch['label'], num_classes=2), 0)

                # val_opt_repl, val_loss_repl, val_update_rng_repl = val_update_fn_repl(
                #     val_opt_repl, flax.jax_utils.replicate(step), batch, val_update_rng_repl)

                _, val_loss_repl, _ = val_update_fn_repl(
                    train_opt_repl, flax.jax_utils.replicate(step), batch, train_update_rng_repl)

                val_loss += val_loss_repl[0]
                val_lr += val_lr_fn(step)
            
            if np.asarray(val_loss/total_val_steps).item() < best_val_loss:
                best_val_loss = np.asarray(val_loss/total_val_steps).item()
                best_epoch = epoch + 1

                if args.model_save:
                    save_path = args.model_save_dir + '/' + now
                    checkpoint_path = flax_checkpoints.save_checkpoint(
                        save_path, (flax.jax_utils.unreplicate(train_opt_repl), epoch + 1), epoch + 1)
                    print('Stored checkpoint at step %d to "%s"', epoch + 1, checkpoint_path)

            val_losses.append(np.asarray(val_loss/total_val_steps).item())
            val_lrs.append(np.asarray(val_lr/total_val_steps).item())

            print(f"Epoch {epoch + 1}/{args.epochs}: Training loss is {train_losses[epoch]}, lrs is {train_lrs[epoch]}, \
                 Val loss is {val_losses[epoch]}, lrs is {val_lrs[epoch]}")
    
            # Evaluation on val dataset
            if (epoch+1) % args.validation_epochs == 0:
                if args.train_dataset != 'train+val+test':
                    print(f"Model accuracy on validation set is: {get_accuracy_mixer(train_opt_repl.target, vit_apply_repl, ds_val)}")

        print(f"Best val loss: {best_val_loss}, Best epoch: {best_epoch}")
        print(f"Training loss: {train_losses}")
        print(f"Val loss: {val_losses}")    

        return train_opt_repl.target


if __name__ == "__main__":
    args = parser.parse_args()
    logger = logging.getLogger('mlpmixer')
    print(jax.local_devices())

    # Downloads model weights for first time into docker container
    if args.model_name.startswith('ViT'):
        args.model_path = "/home/user/MLPmixer/vision_transformers/model_weights/imagenet21k_ViT-B_32.npz"
        if not os.path.exists(args.model_path):
            subprocess.run(["gsutil", "cp", "gs://vit_models/imagenet21k/ViT-B_32.npz", args.model_path])
    elif args.model_name.startswith('Mixer'):
        args.model_path = "/home/user/MLPmixer/vision_transformers/model_weights/imagenet21k_Mixer-B_16.npz"
        if not os.path.exists(args.model_path):
            subprocess.run(["gsutil", "cp", "gs://mixer_models/imagenet21k/Mixer-B_16.npz", args.model_path])

    assert os.path.exists(args.model_path), f"{args.model_name} weights does not exist"

    # asserts that station is within list
    # Checks if YE weights exists
    if args.ye_weights_path is None:
        if args.station == 'slope':
            args.ye_weights_path = "/home/user/MLPmixer/vision_transformers/model_weights/slope_yolact_edge_mobilenetv_100_46258_zoom_5.pth"
        elif args.station == 'park' & args.time == 'day':
            args.ye_weights_path = "/home/user/MLPmixer/vision_transformers/model_weights/park_day_yolact_edge_mobilenetv2_100_41208_zoom_5.pth"
        elif args.station == 'park' & args.time == 'night':
            args.ye_weights_path = "/home/user/MLPmixer/vision_transformers/model_weights/park_night_yolact_edge_mobilenetv2_100_41309_zoom_5.pth"
    assert os.path.exists(args.ye_weights_path), f"{args.ye_weights_path} does not exist"

    if './vision_transformers' not in sys.path:
        sys.path.append('./vision_transformers')

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    print('train_mode', args.train_mode, type(args.train_mode))
    print('train_shuffle', args.train_shuffle, type(args.train_shuffle))
    print('test_shuffle', args.test_shuffle, type(args.test_shuffle))
    print('augment', args.augment, type(args.augment))

    # Performs check on augmentation inputs:
    if args.train_mode:
        if args.augment:
            print(f'Augmentation Flag is set to True. Training {args.model_name} using augment images')
            assert args.zoom_max >= args.zoom_min, "Max zoom factor must be larger than min zoom factor"

    # Loads datasets
    dataloader = DataLoader(args)
    ds_train, ds_val = dataloader.dataloader()
        
    batch = next(iter(ds_train))
    batch['image'] = batch['image'].cpu().numpy()
    batch['image'] = np.expand_dims(batch['image'], 0)
    batch['label'] = batch['label'].cpu().numpy()

    # Load Pre-train model
    model_config = models_config.MODEL_CONFIGS[args.model_name]
    print("Model config:")
    print(model_config)

    # Load model definition & initialize random parameters. This also compiles the model to XLA (takes some minutes the first time).
    if args.model_name.startswith('Mixer'):
        model = models.MlpMixer(num_classes=args.num_classes, **model_config)
    else:
        model = models.VisionTransformer(num_classes=args.num_classes, **model_config)

    variables = jax.jit(lambda: model.init(
        jax.random.PRNGKey(0),
        # Discard the "num_local_devices" dimension of the batch for initialization.
        batch['image'][0,:1],
        train=args.train_mode, # default as False for eval and inference
    ), backend='cpu')()

    '''
    Load and convert pretrained checkpoint. 
    This involves loading the actual pre-trained model results, but then also also
    modifying the parameters a bit, e.g. changing the final layers, and resizing the positional embeddings.
    For details, refer to the code and to the methods of the paper.
    '''
    params = checkpoint.load_pretrained(
        pretrained_path=f'{args.model_path}',
        init_params=variables['params'],
        model_config=model_config,
    )

    # Evaluate
    ''''
    So far, all our data is in the host memory. Let's now replicate the arrays
    into the devices.
    This will make every array in the pytree params become a ShardedDeviceArray
    that has the same data replicated across all local devices.
    For TPU it replicates the params in every core.
    For a single GPU this simply moves the data onto the device.
    For CPU it simply creates a copy.
    '''
    params_repl = flax.jax_utils.replicate(params)
    print('params.cls:', type(params['head']['bias']).__name__,
        params['head']['bias'].shape)
    print('params_repl.cls:', type(params_repl['head']['bias']).__name__,
        params_repl['head']['bias'].shape)

    # Then map the call to our model's forward pass onto all available devices.
    vit_apply_repl = jax.pmap(lambda params, inputs: model.apply(
        dict(params=params), inputs, train=args.train_mode))

    # Gets performance without fine-tuning
    # print(f"Val set accuracy using non-finetuned model: \
    #     {get_accuracy_mixer(params_repl, vit_apply_repl, ds_val)}")

    # Model finetuning
    if args.train_mode:
        mixer = MLPmixer_train()
        opt_repl_target = mixer.train(args, ds_train, ds_val, vit_apply_repl, params)

        # Evaluation on val dataset
        if args.train_dataset != 'train+val+test':
            print(f"Model accuracy on validation set is: {get_accuracy_mixer(opt_repl_target, vit_apply_repl, ds_val)}")