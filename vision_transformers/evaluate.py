from __future__ import absolute_import, division, print_function
import ast
import logging
import argparse
from random import choices
from vision_transformers.yolact_edge.ye_inference import YE_Inference
import jax
import cv2
import ast
import os
import sys
import numpy as np
import pandas as pd
import torch
import tqdm
import matplotlib.pyplot as plt
from datetime import datetime
from torchvision import transforms
from vision_transformers.models.modeling import MlpMixer, CONFIGS
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

parser = argparse.ArgumentParser(description='MLPmixer training Script')

parser.add_argument('--station', default = 'slope', required = True, type = str, choices = ['slope', 'park'])
parser.add_argument('--time', default = 'day', required = True, type = str, choices = ['day', 'night'])
parser.add_argument('--image_catalog_path', required=True, type = str,\
    default = '/home/user/MLPmixer/data/extracted/clean_img_data.pkl')
parser.add_argument('--image_resize_factor', default = 0.1809, type = float,\
    help = "Scale factor to resize training images to feed to YE model for inference")
parser.add_argument('--image_output_folder', default = '/home/user/MLPmixer/vision_transformers/eval', \
    required=True, type = str, \
    help = "Folder to save evaluation results")
parser.add_argument('--val_dataset', required=True, default = 'val', type = str, \
    choices = ['val', 'test'], 
    help = "'val' for validation dataset. 'test' for test dataset")

# MLPmixer parameters
parser.add_argument('--patch_size', default = 16, type = int, help = 'width, height of each divided grid of image')
parser.add_argument('--num_gpus', default=1, type=int, help='Number of GPUs used in training')
parser.add_argument("--name", default="mlpmixer",
                    help="Name of this run. Used for monitoring.")
parser.add_argument("--model_type", choices=["Mixer-B_16", "Mixer-L_16", "Mixer-B_16-21k", "Mixer-L_16-21k"],
                    default="Mixer-B_16", help="Which model to use.")
parser.add_argument("--mixer_fpath", type=str, required=True,\
                    default="/home/user/MLPmixer/vision_transformers/model_weights/checkpoint_weights/21072022-091609/mlpmixer_epoch_1_50.pth",
                    help="fpath for trained ViT or Mixer models.")
parser.add_argument('--disable_tensorrt', default=False, dest='disable_tensorrt', type = ast.literal_eval)
parser.add_argument('--fp16', default = False, type = ast.literal_eval,
                    help="Whether to use 16-bit float precision instead of 32-bit")
parser.add_argument('--fp16_opt_level', type=str, default='O2',
                    help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                            "See details at https://nvidia.github.io/apex/amp.html")
parser.add_argument('--img_size', default=224, type=int)
parser.add_argument('--num_classes', default=2, type=int)

# YE generated mask options
parser.add_argument('--ye_weights_path', type = str, required=True, help = 'filepath for ye_weights')
parser.add_argument('--ye_model_config', default="yolact_edge_mobilenetv2_custom_config", type = str)
parser.add_argument('--crop_size', default = 224, type = int, \
    help = "size of cropped masks from YE model to feed to MLP/ViT model for training & inference")
parser.add_argument('--crop_option', default = "A", type = str)
parser.add_argument('--crop_margin_factor', default = 0.1, type = float)
parser.add_argument('--crop_option_a_retain_background', default = False, \
    type = ast.literal_eval, dest = 'crop_option_a_retain_background')
parser.add_argument('--crop_option_b_alpha', default = 0, type = float)
parser.add_argument('--crop_option_c_vertical', default = 0.5, type = float)
parser.add_argument('--crop_option_c_horizontal', default = 0.05, type = float)

parser.set_defaults(keep_latest=False)


def setup(args):
    # Prepare model
    config = CONFIGS[args.model_type]

    model = MlpMixer(config, args.img_size, num_classes=args.num_classes, patch_size=args.patch_size, zero_head=True)
    # model.load_from(np.load(args.mixer_fpath))
    model.load_state_dict(torch.load(args.mixer_fpath))
    model.to(args.device)
    num_params = count_parameters(model)

    logger.info("{}".format(config))
    logger.info("Training parameters %s", args)
    logger.info("Total Parameter: \t%2.1fM" % num_params)
    return model


def count_parameters(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params/1000000


def evaluate(args):

    data = pd.read_pickle(args.image_catalog_path)

    # filter by station and time of day
    data = data.loc[(data['time_of_day']==args.time) & (data['station']==args.station)]

    # filter by validation or test set
    data = data.loc[data['train_val_test']==args.val_dataset]

    ye_model = YE_Inference(
        weights= args.ye_weights_path,
        disable_tensorrt=True,
        YE_inference_only=True,
        method= args.model_type,
        station= args.station,
        crop_option= args.crop_option,
        crop_margin_factor= args.crop_margin_factor,
        crop_option_a_retain_background= args.crop_option_a_retain_background,
        crop_option_b_alpha= args.crop_option_b_alpha,
        crop_option_c_vertical= args.crop_option_c_vertical,
        crop_option_c_horizontal= args.crop_option_c_horizontal
        )

    # Model & Tokenizer Setup
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mixer_model = setup(args)
    mixer_model.eval()

    now = datetime.now().strftime("%d%m%Y-%H%M%S")
    TP_folder = os.path.join(args.image_output_folder, args.station, args.time, args.val_dataset, str(now), 'TP')
    TN_folder = os.path.join(args.image_output_folder, args.station, args.time, args.val_dataset, str(now), 'TN')
    FP_folder = os.path.join(args.image_output_folder, args.station, args.time, args.val_dataset, str(now), 'FP')
    FN_folder = os.path.join(args.image_output_folder, args.station, args.time, args.val_dataset, str(now), 'FN')

    for folder in [TP_folder, TN_folder, FP_folder, FN_folder]:
        if not os.path.exists(folder):
            os.makedirs(folder)

    true_labels = []
    pred_labels = []

    logger.info("Performing evaluation now ........")
    for i in range(data.shape[0]):
        filepath = data.iloc[i]["img_fpath"]
        label = data.iloc[i]["pass_fail"]

        frame = cv2.imread(filepath)
        img_new_width = int(frame.shape[1] * args.image_resize_factor)
        img_new_height = int(frame.shape[0] * args.image_resize_factor)
        frame = cv2.resize(frame, (img_new_width, img_new_height))
        p, _ = ye_model.predict(frame)

        if p[0] is not None:
            cropped_img = p[0][0]
            cropped_img = cropped_img.permute(2, 0, 1)
            image_ROI = transforms.Resize((args.crop_size, args.crop_size))(cropped_img)
            logits = mixer_model(torch.unsqueeze(image_ROI, 0))
            preds = "on" if torch.argmax(logits[0], dim = -1)==1 else "off"
            
            true_labels.append(label)
            pred_labels.append(preds)

            # logger.info(f"Actual label: {label}, Predicted: {preds}")

            frame = cv2.resize(frame, (400,300))
            if preds==1:
                frame = cv2.putText(frame, 'ON', (50, 50), fontFace=cv2.FONT_HERSHEY_SIMPLEX,\
                fontScale=1, color=(0, 255, 0), thickness=2)
            elif preds==0:
                frame = cv2.putText(frame, 'OFF', (50, 50), fontFace=cv2.FONT_HERSHEY_SIMPLEX,\
                fontScale=1, color=(0, 0, 255), thickness=2)               
            if (label=="on") & (preds=="on"):
                cv2.imwrite(os.path.join(TP_folder, os.path.basename(filepath)), frame)
            if (label=="off") & (preds=="off"):
                cv2.imwrite(os.path.join(TN_folder, os.path.basename(filepath)), frame)
            if (label=="on") & (preds=="off"):
                cv2.imwrite(os.path.join(FN_folder, os.path.basename(filepath)), frame)            
            if (label=="off") & (preds=="on"):
                cv2.imwrite(os.path.join(FP_folder, os.path.basename(filepath)), frame)
        
        else:
            continue
    # logger.info(f"Ground Truth: {true_labels}")
    # logger.info(f"Predicted: {pred_labels}")
    report = classification_report(true_labels, pred_labels, output_dict=True)
    df = pd.DataFrame(report).transpose()
    df.to_csv(os.path.join(args.image_output_folder, args.station, args.time, args.val_dataset, str(now), "classification_report.csv"))
    
    cm = confusion_matrix(true_labels, pred_labels, labels=["off", "on"])
    disp = ConfusionMatrixDisplay(
        confusion_matrix = cm,
        display_labels=["off", "on"])
    disp.plot()
    plt.show()
    plt.savefig(os.path.join(args.image_output_folder, args.station, args.time, args.val_dataset, str(now), "confusion_matrix.jpg"))


if __name__ == "__main__":
    args = parser.parse_args()
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Setup logging

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
    logger = logging.getLogger('mlpmixer')
    logger.info(jax.local_devices())


    # checks if mixer model weights exist
    assert os.path.exists(args.mixer_fpath), f"{args.mixer_fpath} weights does not exist"

    # asserts that station is within list
    # Checks if YE weights exists
    if args.ye_weights_path is None:
        if args.station == 'slope':
            args.ye_weights_path = "/home/user/MLPmixer/vision_transformers/model_weights/slope_yolact_edge_mobilenetv2_100_57873_zoom_5.pth"
        elif args.station == 'park' & args.time == 'day':
            args.ye_weights_path = "/home/user/MLPmixer/vision_transformers/model_weights/park_day_yolact_edge_mobilenetv2_100_41208_zoom_5.pth"
        elif args.station == 'park' & args.time == 'night':
            args.ye_weights_path = "/home/user/MLPmixer/vision_transformers/model_weights/park_night_yolact_edge_mobilenetv2_100_41309_zoom_5.pth"
    assert os.path.exists(args.ye_weights_path), f"{args.ye_weights_path} does not exist"

    if './vision_transformers' not in sys.path:
        sys.path.append('./vision_transformers')

    # Inference
    evaluate(args)

