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
import torch
import tqdm
from torchvision import transforms
from vision_transformers.models.modeling import MlpMixer, CONFIGS

parser = argparse.ArgumentParser(description='MLPmixer training Script')

parser.add_argument('--station', default = 'slope', required = True, type = str, choices = ['slope', 'park'])
parser.add_argument('--time', default = 'day', required = True, type = str, choices = ['day', 'night'])
parser.add_argument('--image_resize_factor', default = 0.1809, type = float,\
    help = "Scale factor to resize training images to feed to YE model for inference")
parser.add_argument('--source', default=1, required=True, type = int, choices=[1,2], \
    help="1 for images, 2 for video")
parser.add_argument('--image_source_folder', default = '', type = str)
parser.add_argument('--image_output_folder', default = '', type = str)
parser.add_argument('--output_pred_img', default=False, dest='output_pred_img', type = ast.literal_eval)
parser.add_argument('--show_window', default=True, dest='show_window', type = ast.literal_eval)
parser.add_argument('--fps_summary', default=False, dest='fps_summary', type = ast.literal_eval)
parser.add_argument('--video_fpath', default='', dest='video_fpath')
parser.add_argument('--video_out_fpath', default=None, type=str, dest='video_out_fpath', required=False) # show_window must be True for video_out_fpath to take effect. Please use MP4 file extension

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
parser.add_argument('--ye_weights_path', required=True, type = str, help = 'filepath for ye_weights')
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


def inference(args):
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

    if args.source==1:
        image_list = []
        assert os.path.exists(os.path.join(os.getcwd(), args.image_source_folder)), "Image folder does not exist"
        for _, _, files in os.walk(os.path.join(os.getcwd(), args.image_source_folder)):
            for file in files:
                image_list.append(os.path.join(os.getcwd(), args.image_source_folder, file))

        for image in image_list:
            frame = cv2.imread(image)
            img_new_width = int(frame.shape[1] * args.image_resize_factor)
            img_new_height = int(frame.shape[0] * args.image_resize_factor)
            frame = cv2.resize(frame, (img_new_width, img_new_height))
            p, _ = ye_model.predict(frame)

            if p[0] is not None:
                cropped_img = p[0][0]
                cropped_img = cropped_img.permute(2, 0, 1)
                image_ROI = transforms.Resize((args.crop_size, args.crop_size))(cropped_img)
                logits = mixer_model(torch.unsqueeze(image_ROI, 0))
                preds = torch.argmax(logits[0], dim = -1)
                on_off = 'ON' if preds == 1 else 'OFF'
                print(on_off)

                frame = cv2.resize(frame, (400,300))
                if on_off == 'ON':
                    frame = cv2.putText(frame, 'ON', (50, 50), fontFace=cv2.FONT_HERSHEY_SIMPLEX,\
                    fontScale=1, color=(0, 255, 0), thickness=2)
                elif on_off=='OFF':
                    frame = cv2.putText(frame, 'OFF', (50, 50), fontFace=cv2.FONT_HERSHEY_SIMPLEX,\
                    fontScale=1, color=(0, 0, 255), thickness=2)                    
                if args.image_output_folder:
                    cv2.imwrite(os.path.join(args.image_output_folder, os.path.basename(image)), frame)


    elif args.source==2:
        video_capture = cv2.VideoCapture(args.video_fpath)
        assert os.path.exists(os.path.join(os.getcwd(), args.video_out_fpath)), "Video file does not exist"
        fourcc = cv2.VideoWriter_fourcc(*'MP4V') # or DIVX
        video_writer = cv2.VideoWriter(args.video_out_fpath, fourcc, 30, (400,300)) 

        while True:
            ret, frame = video_capture.read()
            if not ret:
                print("No frame captured from camera/video file")
                break
            img_new_width = int(frame.shape[1] * args.image_resize_factor)
            img_new_height = int(frame.shape[0] * args.image_resize_factor)
            frame = cv2.resize(frame, (img_new_width, img_new_height))
            p, _ = ye_model.predict(frame)

            if p[0] is not None:
                cropped_img = p[0][0]
                cropped_img = cropped_img.permute(2, 0, 1)
                image_ROI = transforms.Resize((args.crop_size, args.crop_size))(cropped_img)
            
                logits = mixer_model(image_ROI)
                preds = torch.argmax(logits, dim = -1)

                if args.show_window:
                    if cv2.getWindowProperty("inference", cv2.WND_PROP_AUTOSIZE) >= 0:
                        frame = cv2.resize(p[0] if args.output_pred_img else frame, (400,300))
                        frame = cv2.putText(frame, preds, (50, 50), font=cv2.FONT_HERSHEY_SIMPLEX,\
                            fontScale=1, color = (255, 0, 0), thickness=2)
                        cv2.imshow("inference", frame)
                        if args.video_out_fpath:
                            video_writer.write(frame)
                    else:
                        break         
        video_capture.release()
        cv2.destroyAllWindows()    


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
            args.ye_weights_path = "/home/user/MLPmixer/vision_transformers/model_weights/slope_yolact_edge_mobilenetv_100_57873_zoom_5.pth"
        elif args.station == 'park' & args.time == 'day':
            args.ye_weights_path = "/home/user/MLPmixer/vision_transformers/model_weights/park_day_yolact_edge_mobilenetv2_100_41208_zoom_5.pth"
        elif args.station == 'park' & args.time == 'night':
            args.ye_weights_path = "/home/user/MLPmixer/vision_transformers/model_weights/park_night_yolact_edge_mobilenetv2_100_41309_zoom_5.pth"
    assert os.path.exists(args.ye_weights_path), f"{args.ye_weights_path} does not exist"

    if './vision_transformers' not in sys.path:
        sys.path.append('./vision_transformers')


    # Inference
    inference(args)

