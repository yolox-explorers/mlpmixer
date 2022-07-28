import torch
import torch.backends.cudnn as cudnn
from collections import defaultdict
from vision_transformers.yolact_edge.data.config import cfg, set_cfg
from vision_transformers.yolact_edge.yolact import Yolact
from vision_transformers.yolact_edge.utils.augmentations import FastBaseTransform, BaseTransform
from vision_transformers.yolact_edge.utils import timer
from vision_transformers.yolact_edge.layers.output_utils import postprocess, undo_image_transformation
from vision_transformers.yolact_edge.data import COLORS, set_dataset
from vision_transformers.yolact_edge.utils.tensorrt import convert_to_tensorrt
import numpy as np
from operator import itemgetter
from datetime import datetime
from vision_transformers.data_transforms import get_image_only_transform
import os
import cv2
from time import perf_counter
from os.path import exists
from argparse import Namespace
import logging
try:
    from torch2trt import torch2trt, TRTModule
except:
    pass

class YE_Inference(object):

    def __init__(self,
                weights,
                disable_tensorrt=False,
                image_resize_factor=None,
                station="slope",
                YE_inference_only=False,
                method="Mixer-B_16",
                y_offset=0,
                y_offset_frac=0.1,
                crop_option="A",
                crop_margin_factor=0.1,
                crop_option_a_retain_background=False,
                crop_option_b_alpha=0.0,
                crop_option_c_vertical=0.5,
                crop_option_c_horizontal=0.15,
                mixer_model_path=None,
                debug=False,
                output_pred_img=False
                ):

        if not exists(weights):
            raise ValueError(f"YE Weight file does not exist: {weights}")


        if method in ["cnn", 'ViT-B_32', 'Mixer-B_16']:
            if crop_option in ("B", "C"):
                if station == "park":
                    raise ValueError(f"Park station can be specified with crop option {crop_option} (only slope station allowed)")

            if not YE_inference_only:
                if not os.path.exists(mixer_model_path):
                    raise ValueError("Mixer weights cannot be found")

        if debug:
            logging.basicConfig()
            logger_list = ["yolact.eval", "yolact.model.load"]
            for logger_name in logger_list:
                logger = logging.getLogger(logger_name)
                logger.setLevel(logging.DEBUG)

        print("Configuring YOLACT edge...")
        self.color_cache = defaultdict(lambda: {})


        self.model_config="yolact_edge_mobilenetv2_custom_config"
        self.dataset="custom_dataset"
        self.calib_images="./data/calib_images"
        self.config_ovr={}
        self.args_ovr={}

        # CHANGE - CM inconsistent issue
        # self.ye_height_width = 550 # height and width are the same
        self.image_resize_factor = image_resize_factor

        self.use_fp32_databinding = False
        self.station = station
        self.YE_inference_only = YE_inference_only
        self.method = method
        self.scores_sorted = True
        self.y_offset = y_offset
        self.y_offset_frac = y_offset_frac
        self.crop_option = crop_option
        self.crop_margin_factor = crop_margin_factor
        self.crop_option_a_retain_background = crop_option_a_retain_background
        self.crop_option_b_alpha = crop_option_b_alpha
        self.crop_option_c_vertical = crop_option_c_vertical
        self.crop_option_c_horizontal = crop_option_c_horizontal
        self.debug=debug
        self.output_pred_img = output_pred_img
        # cnn_pretrained_model = "mobilenetv2"

        global cfg
        set_cfg(self.model_config)
        cfg.dataset.calib_images = self.calib_images
        cfg.replace(cfg.copy(self.config_ovr))

        global args

        args={
            'trained_model': weights, # CHANGE HERE
            'top_k': 100,
            'cuda': True,
            'fast_nms': True,
            'display_masks': True,
            'display_bboxes': True,
            'display_text': False,
            'display_scores': False,
            'display': False,
            'shuffle': False,
            'ap_data_file': 'results/ap_data.pkl',
            'resume': False,
            'max_images': -1,
            'eval_stride': 5,
            'output_coco_json': False,
            'bbox_det_file': 'results/bbox_detections.json',
            'mask_det_file': 'results/mask_detections.json',
            'config': None,
            'output_web_json': False,
            'web_det_path': 'web/dets/',
            'no_bar': False,
            'display_lincomb': False,
            'benchmark': False,
            'fast_eval': False,
            'deterministic': False,
            'no_sort': False,
            'seed': None,
            'mask_proto_debug': False,
            'crop': True,
            'image': None,
            'images': None,
            'video': None,
            'video_multiframe': 1,
            'score_threshold': 0.05,
            'dataset': None,
            'detect': False,
            'yolact_transfer': False, # unused
            'coco_transfer': False, # unused
            'drop_weights': None,
            'calib_images': None,
            'trt_batch_size': 1,
            'disable_tensorrt': disable_tensorrt,
            'use_fp16_tensorrt': True,
            'use_tensorrt_safe_mode':True,
            'no_hash': False # unused
        }

        args = Namespace(**args)

        args.dataset = self.dataset
        set_dataset(args.dataset)
        for item in self.args_ovr:
            if item in args:
                args[item] = self.args_ovr[item]

        with torch.no_grad():
            if torch.cuda.is_available():
                cudnn.fastest = True
                cudnn.deterministic = True
                cudnn.benchmark = False
                torch.set_default_tensor_type('torch.cuda.FloatTensor')
            else:
                print("CUDA missing... Exiting...")
                exit(1)

            print("Loading YOLACT edge model...")
            net = Yolact(training=False)
            net.load_weights(weights, args=args)
            net.eval()
            convert_to_tensorrt(net, cfg, args, transform=BaseTransform())
            net = net.cuda()
            self.net = net

            # CHANGE HERE
            net.detect.use_fast_nms = args.fast_nms
            cfg.mask_proto_debug = args.mask_proto_debug

            print("Model ready for inference...")


    @property
    def weights(self):
        return self._weights

    @weights.setter
    def weights(self, value):
        if not exists(value):
            raise ValueError(f"YE Weight file does not exist: {value}")
        self._weights = value

    @property
    def disable_tensorrt(self):
        return self._disable_tensorrt

    @disable_tensorrt.setter
    def disable_tensorrt(self, value):
        if not isinstance(value, bool):
            raise ValueError(f"Invalid disable_tensorrt: {value}")
        self._disable_tensorrt = value

    @property
    def image_resize_factor(self):
        return self._image_resize_factor

    @image_resize_factor.setter
    def image_resize_factor(self, value):
        if value is not None and not (0 <= value <= 1 ):
            raise ValueError(f"Invalid image_resize_factor: {value}")
        self._image_resize_factor = value

    @property
    def station(self):
        return self._station

    @station.setter
    def station(self, value):
        if value not in ("slope", "park"):
            raise ValueError(f"Invalid station: {value}")
        self._station = value

    @property
    def YE_inference_only(self):
        return self._YE_inference_only

    @YE_inference_only.setter
    def YE_inference_only(self, value):
        if not isinstance(value, bool):
            raise ValueError(f"Invalid YE_inference_only: {value}")
        self._YE_inference_only = value

    @property
    def method(self):
        return self._method

    @method.setter
    def method(self, value):
        if value not in ("ViT-B_32", "Mixer-B_16"):
            raise ValueError(f"Invalid method: {value}")
        self._method = value

    @property
    def y_offset(self):
        return self._y_offset

    @y_offset.setter
    def y_offset(self, value):
        if not isinstance(value, int):
            raise ValueError(f"Invalid y_offset: {value}")
        self._y_offset = value

    @property
    def y_offset_frac(self):
        return self._y_offset_frac

    @y_offset_frac.setter
    def y_offset_frac(self, value):
        if not (0 <= value <= 1 ):
            raise ValueError(f"Invalid y_offset_frac: {value}")
        self._y_offset_frac = value

    @property
    def crop_option(self):
        return self._crop_option

    @crop_option.setter
    def crop_option(self, value):
        if value not in ("A", "B", "C"):
            raise ValueError(f"Invalid crop_option: {value}")
        self._crop_option = value


    @property
    def crop_margin_factor(self):
        return self._crop_margin_factor

    @crop_margin_factor.setter
    def crop_margin_factor(self, value):
        if not (0 <= value <= 1 ):
            raise ValueError(f"Invalid crop_margin_factor: {value}")
        self._crop_margin_factor = value

    @property
    def crop_option_a_retain_background(self):
        return self._crop_option_a_retain_background

    @crop_option_a_retain_background.setter
    def crop_option_a_retain_background(self, value):
        if not isinstance(value, bool):
            raise ValueError(f"Invalid crop_option_a_retain_background: {value}")
        self._crop_option_a_retain_background = value


    @property
    def crop_option_b_alpha(self):
        return self._crop_option_b_alpha

    @crop_option_b_alpha.setter
    def crop_option_b_alpha(self, value):
        if not (0 <= value <= 1 ):
            raise ValueError(f"Invalid crop_option_b_alpha: {value}")
        self._crop_option_b_alpha = value

    @property
    def crop_option_c_vertical(self):
        return self._crop_option_c_vertical

    @crop_option_c_vertical.setter
    def crop_option_c_vertical(self, value):
        if not (0 <= value <= 1 ):
            raise ValueError(f"Invalid crop_option_c_vertical: {value}")
        self._crop_option_c_vertical = value

    @property
    def crop_option_c_horizontal(self):
        return self._crop_option_c_horizontal

    @crop_option_c_horizontal.setter
    def crop_option_c_horizontal(self, value):
        if not (0 <= value <= 1 ):
            raise ValueError(f"Invalid crop_option_c_horizontal: {value}")
        self._crop_option_c_horizontal = value

    @property
    def mixer_model_path(self):
        return self._mixer_model_path

    @mixer_model_path.setter
    def mixer_model_path(self, value):
        if not exists(value):
            raise ValueError(f"Mixer Weight file does not exist: {value}")
        self._mixer_model_path = value

    @property
    def debug(self):
        return self._debug

    @debug.setter
    def debug(self, value):
        if not isinstance(value, bool):
            raise ValueError(f"Invalid debug: {value}")
        self._debug = value

    @property
    def output_pred_img(self):
        return self._output_pred_img

    @output_pred_img.setter
    def output_pred_img(self, value):
        if not isinstance(value, bool):
            raise ValueError(f"Invalid output_pred_img: {value}")
        self._output_pred_img = value

    def __str__(self):
        return str(vars(self) )

    def prep_output(self, dets_out, img, h, w, save_pred_img, ts, undo_transform=True, class_color=False, mask_alpha=0.45):
        """
        Note: If undo_transform=False then im_h and im_w are allowed to be None.
        """
        if undo_transform:
            img_numpy = undo_image_transformation(img, w, h)
            img_gpu = torch.Tensor(img_numpy).cuda()
        else:
            img_gpu = img / 255.0
            h, w, _ = img.shape

        with timer.env('Postprocess'):
            t = postprocess(dets_out, w, h, visualize_lincomb=args.display_lincomb,
                            crop_masks=args.crop,
                            score_threshold=args.score_threshold)
            torch.cuda.synchronize()

        with timer.env('Copy'):
            if cfg.eval_mask_branch:
                # Masks are drawn on the GPU, so don't copy
                masks = t[3][:args.top_k]
            classes, scores, boxes = [
                x[:args.top_k].cpu().detach().numpy() for x in t[:3]]

        num_dets_to_consider = min(args.top_k, classes.shape[0])
        for j in range(num_dets_to_consider):
            if scores[j] < args.score_threshold:
                num_dets_to_consider = j
                break

        if num_dets_to_consider == 0:
            # No detections found so just output the original image
            if self.output_pred_img:
                img_numpy = (img_gpu * 255).byte().cpu().numpy()
                if save_pred_img:
                    filepath = str(ts) + ".jpg"
                    cv2.imwrite(filepath, img_numpy)
                    print(filepath, "saved.")
                return None, img_numpy
            return None, None

        if self.output_pred_img:
            # Filter for 1 MB and 1 to 9 Obs
            num_obstacles = 1 if self.station == "park" else 9

            car_index =  np.where(classes == 0)[0][:1]
            obstacle_indices =  np.where(classes == 1)[0][:num_obstacles]

            retained_indices = np.append(car_index, obstacle_indices)

            masks = masks[retained_indices]
            classes = classes[retained_indices]
            scores = scores[retained_indices]
            boxes = boxes[retained_indices]

            num_dets_to_consider = len(masks)

            # Quick and dirty lambda for selecting the color for a particular index
            # Also keeps track of a per-gpu color cache for maximum speed
            def get_color(j, on_gpu=None):
                color_idx = (
                    classes[j] * 5 if class_color else j * 5) % len(COLORS)

                if on_gpu is not None and color_idx in self.color_cache[on_gpu]:
                    return self.color_cache[on_gpu][color_idx]
                else:
                    color = COLORS[color_idx]
                    if not undo_transform:
                        # The image might come in as RGB or BRG, depending
                        color = (color[2], color[1], color[0])
                    if on_gpu is not None:
                        color = torch.Tensor(color).to(on_gpu).float() / 255.
                        self.color_cache[on_gpu][color_idx] = color
                    return color

            # First, draw the masks on the GPU where we can do it really fast
            # Beware: very fast but possibly unintelligible mask-drawing code ahead
            # I wish I had access to OpenGL or Vulkan but alas, I guess Pytorch tensor operations will have to suffice
            if args.display_masks and cfg.eval_mask_branch:
                # After this, mask is of size [num_dets, h, w, 1]
                masks_cp =  masks.clone().detach()
                masks_cp = masks_cp[:num_dets_to_consider, :, :, None]

                # Prepare the RGB images for each mask given their color (size [num_dets, h, w, 1])
                colors = torch.cat([get_color(j, on_gpu=img_gpu.device.index).view(
                    1, 1, 1, 3) for j in range(num_dets_to_consider)], dim=0)
                masks_color = masks_cp.repeat(1, 1, 1, 3) * colors * mask_alpha

                # This is 1 everywhere except for 1-mask_alpha where the mask is
                inv_alph_masks = masks_cp * (-mask_alpha) + 1

                # I did the math for this on pen and paper. This whole block should be equivalent to:
                #    for j in range(num_dets_to_consider):
                #        img_gpu = img_gpu * inv_alph_masks[j] + masks_color[j]
                masks_color_summand = masks_color[0]
                if num_dets_to_consider > 1:
                    inv_alph_cumul = inv_alph_masks[:(
                        num_dets_to_consider-1)].cumprod(dim=0)
                    masks_color_cumul = masks_color[1:] * inv_alph_cumul
                    masks_color_summand += masks_color_cumul.sum(dim=0)

                img_gpu_cp =  img_gpu.clone().detach()
                img_gpu_cp = img_gpu_cp * \
                    inv_alph_masks.prod(dim=0) + masks_color_summand

            # Then draw the stuff that needs to be done on the cpu
            # Note, make sure this is a uint8 tensor or opencv will not anti alias text for whatever reason
            img_numpy = (img_gpu_cp * 255).byte().cpu().numpy()

            if args.display_text or args.display_bboxes:
                for j in reversed(range(num_dets_to_consider)):
                    x1, y1, x2, y2 = boxes[j, :]
                    color = get_color(j)
                    score = scores[j]

                    if args.display_bboxes:
                        cv2.rectangle(img_numpy, (x1, y1), (x2, y2), color, 1)

                    if args.display_text:
                        _class = cfg.dataset.class_names[classes[j]]
                        text_str = '%s: %.2f' % (
                            _class, score) if args.display_scores else _class

                        font_face = cv2.FONT_HERSHEY_DUPLEX
                        font_scale = 0.6
                        font_thickness = 1

                        text_w, text_h = cv2.getTextSize(
                            text_str, font_face, font_scale, font_thickness)[0]

                        text_pt = (x1, y1 - 3)
                        text_color = [255, 255, 255]

                        cv2.rectangle(img_numpy, (x1, y1),
                                    (x1 + text_w, y1 - text_h - 4), color, -1)
                        cv2.putText(img_numpy, text_str, text_pt, font_face,
                                    font_scale, text_color, font_thickness, cv2.LINE_AA)

            if save_pred_img:
                filepath = str(ts) + ".jpg"
                cv2.imwrite(filepath, img_numpy)
                print(filepath, "saved.")
                
        if  (0 not in classes) or (1 not in classes):
            if self.output_pred_img:
                return None, img_numpy
            else:
                return None, None

        if self.method == "rules":
            if self.output_pred_img:
                return (img_gpu, classes, scores, boxes, masks), img_numpy
            else:
                return (img_gpu, classes, scores, boxes, masks), None

        elif self.method in ["cnn", 'ViT-B_32', 'Mixer-B_16']:
            num_obstacles = 1 if self.station == "park" else 9

            car_index =  np.where(classes == 0)[0][:1]
            obstacle_indices =  np.where(classes == 1)[0][:num_obstacles]

            car_mask = masks[car_index][0]
            obstacle_masks = masks[obstacle_indices]

            if self.crop_option == "A":
                combined_obstacle_masks = obstacle_masks.sum(dim=0)

                combined_all_masks = car_mask + combined_obstacle_masks
                combined_all_masks[combined_all_masks > 0] = 1

                all_masks_coordinates = combined_all_masks.nonzero(as_tuple=False)
                # Below is to cater for case when there is no 1s in the masks
                if all_masks_coordinates.shape[0] == 0:
                    if self.output_pred_img:
                        return None, img_numpy
                    else:
                        return None, None

                all_y_pos, all_x_pos = all_masks_coordinates[:, 0], all_masks_coordinates[:, 1]
                y_pos_min, y_pos_max, x_pos_min, x_pos_max = all_y_pos.min(), all_y_pos.max(), all_x_pos.min(), all_x_pos.max()
                height = y_pos_max - y_pos_min
                width = x_pos_max - x_pos_min

                if not self.crop_option_a_retain_background:
                    combined_all_masks = combined_all_masks.unsqueeze(dim=-1)

                    img_gpu = img_gpu * combined_all_masks

                row_min = y_pos_min - self.crop_margin_factor * height
                row_max = y_pos_max + self.crop_margin_factor * height + 1
                col_min = x_pos_min - self.crop_margin_factor * width
                col_max = x_pos_max + self.crop_margin_factor * width + 1

                # prevent negative index on the min row and min col by using max function wuth 1 operand as 0 (meaning min can only be 0 or more)
                img_gpu = img_gpu[
                    max(row_min, torch.tensor(0)).type(torch.IntTensor) : row_max.type(torch.IntTensor),
                    max(col_min, torch.tensor(0)).type(torch.IntTensor) : col_max.type(torch.IntTensor)]

            elif self.crop_option == "B":
                car_box = boxes[car_index][0]

                car_top_left_x = car_box[0]
                car_top_left_y = car_box[1]
                car_bottom_right_x = car_box[2]
                car_bottom_right_y = car_box[3]

                car_height = car_bottom_right_y - car_top_left_y
                front_wheel_height = 1/3 * car_height
                front_wheel_min_y = car_bottom_right_y - front_wheel_height
                front_wheel_min_y = front_wheel_min_y.astype(np.int)

                front_wheel_mask = car_mask.clone().detach()

                front_wheel_mask[:front_wheel_min_y, :] = 0

                obstacle_boxes = boxes[obstacle_indices]
                obstacle_masks_list = []
                for i in range(obstacle_boxes.shape[0]):
                    obstacle_masks_list.append( {"top_left_y": obstacle_boxes[i][1], "index": obstacle_indices[i], "mask": obstacle_masks[i]})

                obstacle_masks_list = sorted(obstacle_masks_list, key=itemgetter("top_left_y"), reverse=True) 

                front_wheel_intersects_obstacle = False
                for i in range(len(obstacle_masks_list) ):
                    obstacle_mask = obstacle_masks_list[i]["mask"]

                    combined_masks = front_wheel_mask + obstacle_mask

                    if ((combined_masks) > 1).sum() > 1:  # overlap
                        combined_masks = car_mask + obstacle_mask

                        combined_masks[combined_masks > 0] = 1

                        all_masks_coordinates = combined_masks.nonzero(as_tuple=False)
                        # Below is to cater for case when there is no 1s in the masks
                        if all_masks_coordinates.shape[0] == 0:
                            if self.output_pred_img:
                                return None, img_numpy
                            else:
                                return None, None

                        all_y_pos, all_x_pos = all_masks_coordinates[:, 0], all_masks_coordinates[:, 1]
                        y_pos_min, y_pos_max, x_pos_min, x_pos_max = all_y_pos.min(), all_y_pos.max(), all_x_pos.min(), all_x_pos.max()
                        height = y_pos_max - y_pos_min
                        width = x_pos_max - x_pos_min

                        combined_masks = combined_masks.unsqueeze(dim=-1)
                        img_gpu = img_gpu * combined_masks

                        crop_y_min= (y_pos_min-self.crop_margin_factor*height).type(torch.IntTensor)
                        crop_y_max = (y_pos_max + self.crop_margin_factor * height + 1).type(torch.IntTensor)
                        crop_x_min = (x_pos_min-self.crop_margin_factor*width).type(torch.IntTensor)
                        crop_x_max = (x_pos_max + self.crop_margin_factor * width + 1).type(torch.IntTensor)

                        img_gpu = img_gpu[max(crop_y_min, torch.tensor(0).type(torch.IntTensor) ) :crop_y_max, max(crop_x_min, torch.tensor(0).type(torch.IntTensor) ) :crop_x_max]

                        mask_alpha = self.crop_option_b_alpha

                        obstacle_indices = obstacle_masks_list[i]["index"]

                        front_wheel_intersects_obstacle = True
                        break

                if not front_wheel_intersects_obstacle:
                    if self.output_pred_img:
                        return None, img_numpy
                    else:
                        return None, None

            elif self.crop_option == "C":
                obstacle_boxes = boxes[obstacle_indices]
                obstacle_boxes_list = []
                for i in range(obstacle_boxes.shape[0]):
                    obstacle_boxes_list.append(obstacle_boxes[i])

                obstacle_boxes_list = sorted(obstacle_boxes_list, key=itemgetter(1), reverse=True) 

                car_mask_coordinates = car_mask.nonzero(as_tuple=False)
                # Below is to cater for case when there is no 1s in the masks
                if car_mask_coordinates.shape[0] == 0:
                    if self.output_pred_img:
                        return None, img_numpy
                    else:
                        return None, None

                car_mask_bottommost_coordinates = car_mask_coordinates[car_mask_coordinates.argmax(dim=0)[0]:, :]

                top_left_x, top_left_y, bottom_right_x, bottom_right_y = 0, 0, 0, 0

                any_car_mask_bottommost_coordinate_near_obstacle = False
                try:
                    for i in range(len(obstacle_boxes_list) ):
                        top_left_x = obstacle_boxes_list[i][0]
                        top_left_y = obstacle_boxes_list[i][1]
                        bottom_right_x = obstacle_boxes_list[i][2]
                        bottom_right_y = obstacle_boxes_list[i][3]

                        obstacle_height = bottom_right_y - top_left_y
                        obstacle_width = bottom_right_x - top_left_x

                        top_left_x -= obstacle_width * self.crop_option_c_horizontal
                        bottom_right_x += obstacle_width * self.crop_option_c_horizontal

                        if i == 0:
                            if len(obstacle_boxes_list) == 1:
                                height_below_obstacle = obstacle_height * ((self.crop_option_c_vertical + 0.05) * 120 / 14)
                            else:
                                height_below_obstacle = (top_left_y - obstacle_boxes_list[i+1][3]) * (self.crop_option_c_vertical + 0.05)
                        else:
                            height_below_obstacle = (obstacle_boxes_list[i-1][1] - bottom_right_y) * self.crop_option_c_vertical

                        bottom_right_y += height_below_obstacle

                        for j in range(car_mask_bottommost_coordinates.shape[0]):

                            if (top_left_y <= car_mask_bottommost_coordinates[j][0] <= bottom_right_y) and (top_left_x <= car_mask_bottommost_coordinates[j][1] <= bottom_right_x):
                                raise StopIteration

                except StopIteration:
                    any_car_mask_bottommost_coordinate_near_obstacle = True

                
                if any_car_mask_bottommost_coordinate_near_obstacle:
                    if i == len(obstacle_boxes_list) - 1:
                        if len(obstacle_boxes_list) == 1:
                            height_above_obstacle = obstacle_height * ((self.crop_option_c_vertical - 0.05) * 120 / 14)
                        else:
                            height_above_obstacle = (obstacle_boxes_list[i-1][1] - bottom_right_y) * (self.crop_option_c_vertical - 0.05)
                    else:
                        height_above_obstacle = (top_left_y - obstacle_boxes_list[i+1][3]) * self.crop_option_c_vertical

                    top_left_y -= height_above_obstacle

                    img_gpu = img_gpu[ max(top_left_y.astype(np.int), 0) : bottom_right_y.astype(np.int) + 1, max(top_left_x.astype(np.int), 0) : bottom_right_x.astype(np.int) + 1]

                else:
                    if self.output_pred_img:
                        return None, img_numpy
                    else:
                        return None, None

            else:
                raise ValueError("Invalid crop option:", self.crop_option)

            retained_indices = np.append(car_index, obstacle_indices)

            masks = masks[retained_indices]
            classes = classes[retained_indices]
            scores = scores[retained_indices]
            boxes = boxes[retained_indices]

            num_dets_to_consider = len(masks)


            # Quick and dirty lambda for selecting the color for a particular index
            # Also keeps track of a per-gpu color cache for maximum speed
            def get_color(j, on_gpu=None):
                color_idx = (
                    classes[j] * 5 if class_color else j * 5) % len(COLORS)

                if on_gpu is not None and color_idx in self.color_cache[on_gpu]:
                    return self.color_cache[on_gpu][color_idx]
                else:
                    color = COLORS[color_idx]
                    if not undo_transform:
                        # The image might come in as RGB or BRG, depending
                        color = (color[2], color[1], color[0])
                    if on_gpu is not None:
                        color = torch.Tensor(color).to(on_gpu).float() / 255.
                        self.color_cache[on_gpu][color_idx] = color
                    return color

            # First, draw the masks on the GPU where we can do it really fast
            # Beware: very fast but possibly unintelligible mask-drawing code ahead
            # I wish I had access to OpenGL or Vulkan but alas, I guess Pytorch tensor operations will have to suffice
            # if args.display_masks and cfg.eval_mask_branch:
            if self.crop_option == "B" and self.crop_option_b_alpha > 0:
                # After this, mask is of size [num_dets, h, w, 1]
                masks_B =  masks.clone().detach()
                masks_B = masks_B[:num_dets_to_consider, :, :, None]

                # Prepare the RGB images for each mask given their color (size [num_dets, h, w, 1])
                colors = torch.cat([get_color(j, on_gpu=img_gpu.device.index).view(
                    1, 1, 1, 3) for j in range(num_dets_to_consider)], dim=0)
                masks_color = masks_B.repeat(1, 1, 1, 3) * colors * mask_alpha

                # This is 1 everywhere except for 1-mask_alpha where the mask is
                inv_alph_masks = masks_B * (-mask_alpha) + 1

                # I did the math for this on pen and paper. This whole block should be equivalent to:
                #    for j in range(num_dets_to_consider):
                #        img_gpu = img_gpu * inv_alph_masks[j] + masks_color[j]
                masks_color_summand = masks_color[0]
                if num_dets_to_consider > 1:
                    inv_alph_cumul = inv_alph_masks[:(
                        num_dets_to_consider-1)].cumprod(dim=0)
                    masks_color_cumul = masks_color[1:] * inv_alph_cumul
                    masks_color_summand += masks_color_cumul.sum(dim=0)

                img_gpu = img_gpu * \
                    inv_alph_masks.prod(dim=0)[ max(crop_y_min, torch.tensor(0).type(torch.IntTensor) ) : crop_y_max, max(crop_x_min, torch.tensor(0).type(torch.IntTensor) ) :crop_x_max] + \
                        masks_color_summand[ max(crop_y_min, torch.tensor(0).type(torch.IntTensor) ) :crop_y_max, max(crop_x_min, torch.tensor(0).type(torch.IntTensor) ) :crop_x_max]

            if self.output_pred_img:
                return (img_gpu, classes, scores, boxes, masks), img_numpy
            else:
                return (img_gpu, classes, scores, boxes, masks), None

    def predict(self, img, save_pred_img=False):
        if img is None:
           raise ValueError("'img' argument must not be None")

        ts = datetime.timestamp(datetime.now())

        if self.debug:
            debug_out = {}
        else:
            debug_out = None

        if self.debug: start_time = perf_counter() 

        if self.image_resize_factor is not None:
            # CHANGE - CM inconsistent issue
            img_org_shape = img.shape
            img_new_height = int(img.shape[0] * self.image_resize_factor)
            img_new_width = int(img.shape[1] * self.image_resize_factor)
            img = cv2.resize(img, (img_new_width, img_new_height))

            if self.debug:
                debug_out["img_resized_from"] = img_org_shape
                debug_out["img_resized_to"] = img.shape

        frame = torch.Tensor(img).cuda().float() # aiap: height, width, 3 (unlike T.ToTensor(), torch.Tensor does not change shape)
        batch = FastBaseTransform()(frame.unsqueeze(0))

        extras = {"backbone": "full", "interrupt": False,
                  "keep_statistics": False, "moving_statistics": None}

        with torch.no_grad():
            preds = self.net(batch, extras=extras)["pred_outs"]

            out = self.prep_output(
                preds, frame, None, None, save_pred_img, ts, undo_transform=False)

            if self.YE_inference_only:
                if self.debug:
                    debug_out["msg"] = "Either # of car or # of obstacles is 0" if out[0] is None else "OK"
                    stop_time = perf_counter()
                    sec_taken = stop_time - start_time
                    debug_out["sec_taken"] = sec_taken
                    debug_out["fps"] = 1 / sec_taken

                return out, debug_out

            if out[0] is None:
                if self.debug:
                    debug_out["msg"] = "Either # of car or # of obstacles is 0"
                    stop_time = perf_counter()
                    sec_taken = stop_time - start_time
                    debug_out["sec_taken"] = sec_taken
                    debug_out["fps"] = 1 / sec_taken

                return "OFF", ts, debug_out, out[1], None, None, None, None

            (img_gpu, classes, scores, boxes, masks) = out[0]
            img_numpy = out[1]

            output = None

            if self.method == "rules":
                # Get index of car with highest score
                # Get indices of obstacles with top 1 (park) or top 9 (slope) highest score

                num_obstacles = 1 if self.station == "park" else 9

                if self.scores_sorted:
                    car_index =  np.where(classes == 0)[0][:1]
                    obstacle_indices =  np.where(classes == 1)[0][:num_obstacles]
                else:
                    cars_scores = scores * (1 - classes)
                    car_index = np.argpartition(cars_scores, -1)[-1:]

                    obstacle_scores = scores * classes
                    obstacle_indices = np.argpartition(obstacle_scores, -num_obstacles)[-num_obstacles:]

                car_mask = masks[car_index]
                obstacle_masks = masks[obstacle_indices]

                car_max_y = car_mask[0].nonzero(as_tuple=False)[-1][0]
                
                # ASSUMPTION: all top "num_obstacles" obstacle masks do not intersect
                combined_obstacle_masks = obstacle_masks.sum(dim=0)

                all_points_at_car_max_y = car_mask[0][car_max_y] + combined_obstacle_masks[car_max_y]

                is_interect = (all_points_at_car_max_y > 1)
                num_intersection_points = is_interect.sum()

                if (self.y_offset or self.y_offset_frac):
                    if self.y_offset:
                        car_y = torch.narrow(car_mask[0].nonzero(as_tuple=False), 1,0,1).unique()[-self.y_offset: ]
                        wheel_mask = car_mask[0][car_y, :]
                        obs_mask = combined_obstacle_masks[car_y, :]
                        is_interect = wheel_mask + obs_mask
                        num_intersection_points = (is_interect > 1).sum()

                    if self.y_offset_frac:
                        car_min_y = car_mask[0].nonzero(as_tuple=False)[0][0]
                        car_height = car_max_y - car_min_y
                        front_wheel_height = 1/3 * car_height
                        wheel_min = (car_max_y - torch.round(self.y_offset_frac * front_wheel_height))
                        car_y = torch.range(start = wheel_min, end = car_max_y, step = 1).type(torch.int64)
                        wheel_mask = car_mask[0][car_y, :]
                        obs_mask = combined_obstacle_masks[car_y, :]
                        is_interect = wheel_mask + obs_mask
                        num_intersection_points = (is_interect > 1).sum()

                output = "ON" if num_intersection_points > 0 else "OFF"

            elif self.method in ["cnn", 'ViT-B_32', 'Mixer-B_16']:
                # convert img_gpu to 3 x H x W (model takes in this)
                img_gpu_transformed = img_gpu.permute(2, 0, 1)
                
                # then transform to 224x224
                transforms = get_image_only_transform(train=False)

                img_gpu_transformed = transforms(img_gpu_transformed)

                if not args.disable_tensorrt:
                    if self.use_fp32_databinding:
                        preds = self.cnn_model(img_gpu_transformed.cuda().unsqueeze(0))
                    else:
                        preds = self.cnn_model(img_gpu_transformed.cuda().half().unsqueeze(0))
                else:
                    preds = self.cnn_model(img_gpu_transformed.unsqueeze(0))

                y_pred = (preds > 0.0).type(torch.float) # using logits
                output = "ON" if y_pred == 1 else "OFF"

                if self.output_pred_img:
                    if save_pred_img:
                        img_numpy_transformed = (img_gpu_transformed.permute(1, 2, 0) * 255).byte().cpu().numpy()
                        filepath = str(ts) + "_cropped.jpg"
                        cv2.imwrite(filepath, img_numpy_transformed)
                        print(filepath, "saved.")

            else:
                raise ValueError(f"Invalid method: {self.method}")

        if self.debug:
            debug_out["msg"] = "OK"
            stop_time = perf_counter()
            sec_taken = stop_time - start_time
            debug_out["sec_taken"] = sec_taken
            debug_out["fps"] = 1 / sec_taken
            
        if self.output_pred_img:
            return output, ts, debug_out, img_numpy, classes, scores, boxes, masks
        else:
            return output, ts, debug_out, img_numpy, None, None, None, None
