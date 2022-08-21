""" Contains functions used to sanitize and prepare the output of Yolact. """


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2

from vision_transformers.yolact_edge.data import cfg, mask_type, MEANS, STD, activation_func
from vision_transformers.yolact_edge.utils.augmentations import Resize
from vision_transformers.yolact_edge.utils import timer
from .box_utils import crop, sanitize_coordinates

def run_rules(masks, classes):
    try:
        print("\nmasks:")
        print("classes", classes)
        print("no. of masks: ", len(masks), masks.unique())

        motorcycle_list, obstacle_list = [], []
        for idx, obj_class in enumerate(classes):
            if obj_class == 0:
                print("class index for MB:", idx, "obj:", obj_class, masks[idx], masks[idx].shape)
                motorcycle_list.append(masks[idx])
            elif obj_class == 1:
                print("class index for obs:", idx, "obj:", obj_class, masks[idx], masks[idx].shape)
                obstacle_list.append(masks[idx])
        # print("MB list, O list: ", motorcycle_list, obstacle_list)    

        if len(motorcycle_list) != 0 and len(obstacle_list) != 0:
            MB_y_list, O_points_list = [], []
            for idx, mask in enumerate(motorcycle_list):
                MB_yx = np.where(mask.cpu() == 1)
                MB_y, MB_x = MB_yx
                print("MB_yx", MB_yx, "no. of pixel combi:", len(MB_x))
                MB_y_list.append(MB_y) # assuming the highest confidence MB is the target and comes first in list

            for idx, obstacle in enumerate(obstacle_list):
                O_yx = np.where(obstacle.cpu() == 1)
                O_y, O_x = O_yx
                print(idx+1, "O_yx", O_yx, "no. of pixel combi:", len(O_x))
                O_points_list.append(list(zip(O_x, O_y)))


            max_MB_y = MB_y_list[0].max() # assuming the highest confidence MB is the target and comes first in list
            # alternative to getting max y is using bottom most
            # MB_contours = np.array([[MB_x[i], MB_y[i]] for i in range(len(MB_x))],dtype=np.int32)
            # max_MB_xy = tuple(MB_contours[MB_contours[:,1].argmax()])
            # max_MB_y = int(max_MB_xy[1])

            winner = np.argwhere(MB_y == np.amax(MB_y))
            max_y_indices = winner.flatten().tolist()
            print("All max Ys index:", max_y_indices, max_MB_y) # if you want it as a list

            MB_bottom_points = [(MB_x[i], max_MB_y) for i in max_y_indices]
            print("MB_bottom_points", MB_bottom_points)

            for point in MB_bottom_points:
                # print("check obs list: ", obstacle_list[point[0], point[1]])
                # print("check O points list: ", O_points_list[point[0], point[1]])
                # print("check Opoints: ", O_points[point[0], point[1]])
                for idx, O_points in enumerate(O_points_list): # for obstacle points in each obstacle mask
                    # if masks[o_i][point[0], point[1]] == 0: # TODO: amend this method if its faster
                    if point not in O_points:
                        output = "OFF"
                        print(f"point not in obs {idx+1}")
                    elif point in O_points:
                        output = "ON"
                        print(f"point in obs {idx+1}")
                        break
            print("Output label: ", output)
        else:
            output = "NONE"
            print("Either motorcycle or obstacle(s) detection missing.")
    except IndexError:
        output = "NONE2"
        print("INDEXERROR", output)

    return output

def create_new_save_path(output: str, save_path: str):
    """Creates a new save path using output label and input save path so that 
    image can be saved to respective output label folders.

    Args:
        output (str): output labels, accepts either "ON", "OFF", or "NONE"
        save_path (str): input save path

    Returns:
        save_path (str): new save path containing output label
    """
    output_class = ["ON", "OFF", "NONE"]
    if output == output_class[0]:
        save_path = "\\".join(save_path.rsplit("\\")[:-1]) +"\\"+ output_class[0] +"\\"+ save_path.rsplit("\\")[-1]
    elif output == output_class[1]:
        save_path = "\\".join(save_path.rsplit("\\")[:-1]) +"\\"+ output_class[1] +"\\"+ save_path.rsplit("\\")[-1]
    elif output == output_class[2]:
        save_path = "\\".join(save_path.rsplit("\\")[:-1]) +"\\"+ output_class[2] +"\\"+ save_path.rsplit("\\")[-1]
    return save_path

def postprocess(
    det_output, w, h, batch_idx=0, interpolation_mode='bilinear',
    visualize_lincomb=False, crop_masks=True, score_threshold=0,
    save_path=None
):
    """
    Postprocesses the output of Yolact on testing mode into a format that makes sense,
    accounting for all the possible configuration settings.

    Args:
        - det_output: The lost of dicts that Detect outputs.
        - w: The real with of the image.
        - h: The real height of the image.
        - batch_idx: If you have multiple images for this batch, the image's index in the batch.
        - interpolation_mode: Can be 'nearest' | 'area' | 'bilinear' (see torch.nn.functional.interpolate)
        - save_path: save path for output image, defaults to None.

    Returns 4 torch Tensors (in the following order):
        - classes [num_det]: The class idx for each detection.
        - scores  [num_det]: The confidence score for each detection.
        - boxes   [num_det, 4]: The bounding box for each detection in absolute point form.
        - masks   [num_det, h, w]: Full image masks for each detection.
    """
    
    dets = det_output[batch_idx]
    
    if dets is None:
        return [torch.Tensor()] * 4 # Warning, this is 4 copies of the same thing

    if score_threshold > 0:
        keep = dets['score'] > score_threshold

        for k in dets:
            if k != 'proto':
                if cfg.use_tensorrt_safe_mode:
                    dets[k] = torch.index_select(dets[k], 0, torch.nonzero(keep, as_tuple=True)[0])
                else:
                    dets[k] = dets[k][keep]
        
        if dets['score'].size(0) == 0:
            return [torch.Tensor()] * 4

    # im_w and im_h when it concerns bboxes. This is a workaround hack for preserve_aspect_ratio
    b_w, b_h = (w, h)

    # Undo the padding introduced with preserve_aspect_ratio
    if cfg.preserve_aspect_ratio:
        r_w, r_h = Resize.faster_rcnn_scale(w, h, cfg.min_size, cfg.max_size)

        # Get rid of any detections whose centers are outside the image
        boxes = dets['box']
        boxes = center_size(boxes)
        s_w, s_h = (r_w/cfg.max_size, r_h/cfg.max_size)
        
        not_outside = ((boxes[:, 0] > s_w) + (boxes[:, 1] > s_h)) < 1 # not (a or b)
        for k in dets:
            if k != 'proto':
                dets[k] = dets[k][not_outside]

        # A hack to scale the bboxes to the right size
        b_w, b_h = (cfg.max_size / r_w * w, cfg.max_size / r_h * h)
    
    # Actually extract everything from dets now
    classes = dets['class']
    boxes   = dets['box']
    scores  = dets['score']
    masks   = dets['mask']

    if cfg.mask_type == mask_type.lincomb and cfg.eval_mask_branch:
        # At this points masks is only the coefficients
        proto_data = dets['proto']
        
        # Test flag, do not upvote
        if cfg.mask_proto_debug:
            np.save('scripts/proto.npy', proto_data.cpu().numpy())
        
        if visualize_lincomb:
            display_lincomb(proto_data, masks)

        masks = proto_data @ masks.t()
        masks = cfg.mask_proto_mask_activation(masks)

        # Crop masks before upsampling because you know why
        if crop_masks:
            masks = crop(masks, boxes)

        # Permute into the correct output shape [num_dets, proto_h, proto_w]
        masks = masks.permute(2, 0, 1).contiguous()

        # Scale masks up to the full image
        if cfg.preserve_aspect_ratio:
            # Undo padding
            masks = masks[:, :int(r_h/cfg.max_size*proto_data.size(1)), :int(r_w/cfg.max_size*proto_data.size(2))]
        
        masks = F.interpolate(masks.unsqueeze(0), (h, w), mode=interpolation_mode, align_corners=False).squeeze(0)

        # print("masks pp1", masks, masks.shape)

        # Binarize the masks
        masks.gt_(0.5)

        # xxx = np.where(masks.cpu() == 1)
        # print("xxxx", len(xxx), xxx)
        # print("masks pp", masks, masks.shape, len(masks[masks == 1] ), xxx, len(xxx[0]), len(xxx[1]), len(xxx[2])) 

        # custom changes 1 - start
        if save_path is not None:
            output = run_rules(masks, classes)
            save_path = create_new_save_path(
                output = output,
                save_path = save_path
            )
        # custom changes 1 - end
    
    boxes[:, 0], boxes[:, 2] = sanitize_coordinates(boxes[:, 0], boxes[:, 2], b_w, cast=False)
    boxes[:, 1], boxes[:, 3] = sanitize_coordinates(boxes[:, 1], boxes[:, 3], b_h, cast=False)
    boxes = boxes.long()

    if cfg.mask_type == mask_type.direct and cfg.eval_mask_branch:
        # Upscale masks
        full_masks = torch.zeros(masks.size(0), h, w)

        for jdx in range(masks.size(0)):
            x1, y1, x2, y2 = boxes[jdx, :]

            mask_w = x2 - x1
            mask_h = y2 - y1

            # Just in case
            if mask_w * mask_h <= 0 or mask_w < 0:
                continue
            
            mask = masks[jdx, :].view(1, 1, cfg.mask_size, cfg.mask_size)
            mask = F.interpolate(mask, (mask_h, mask_w), mode=interpolation_mode, align_corners=False)
            mask = mask.gt(0.5).float()
            full_masks[jdx, y1:y2, x1:x2] = mask
        
        masks = full_masks

    # custom changes 2 - start
    if save_path is not None:
        return classes, scores, boxes, masks, output, save_path
    else:
        return classes, scores, boxes, masks
    # custom changes 2 - end

    


def undo_image_transformation(img, w, h):
    """
    Takes a transformed image tensor and returns a numpy ndarray that is untransformed.
    Arguments w and h are the original height and width of the image.
    """
    img_numpy = img.permute(1, 2, 0).cpu().numpy()
    img_numpy = img_numpy[:, :, (2, 1, 0)] # To BRG

    if cfg.backbone.transform.normalize:
        img_numpy = (img_numpy * np.array(STD) + np.array(MEANS)) / 255.0
    elif cfg.backbone.transform.subtract_means:
        img_numpy = (img_numpy / 255.0 + np.array(MEANS) / 255.0).astype(np.float32)
        
    img_numpy = img_numpy[:, :, (2, 1, 0)] # To RGB
    img_numpy = np.clip(img_numpy, 0, 1)

    if cfg.preserve_aspect_ratio:
        # Undo padding
        r_w, r_h = Resize.faster_rcnn_scale(w, h, cfg.min_size, cfg.max_size)
        img_numpy = img_numpy[:r_h, :r_w]

        # Undo resizing
        img_numpy = cv2.resize(img_numpy, (w,h))

    else:
        return cv2.resize(img_numpy, (w,h))


def display_lincomb(proto_data, masks):
    out_masks = torch.matmul(proto_data, masks.t())
    # out_masks = cfg.mask_proto_mask_activation(out_masks)

    for kdx in range(1):
        jdx = kdx + 0
        import matplotlib.pyplot as plt
        coeffs = masks[jdx, :].cpu().numpy()
        idx = np.argsort(-np.abs(coeffs))
        # plt.bar(list(range(idx.shape[0])), coeffs[idx])
        # plt.show()
        
        coeffs_sort = coeffs[idx]
        arr_h, arr_w = (4,8)
        proto_h, proto_w, _ = proto_data.size()
        arr_img = np.zeros([proto_h*arr_h, proto_w*arr_w])
        arr_run = np.zeros([proto_h*arr_h, proto_w*arr_w])
        test = torch.sum(proto_data, -1).cpu().numpy()

        for y in range(arr_h):
            for x in range(arr_w):
                i = arr_w * y + x

                if i == 0:
                    running_total = proto_data[:, :, idx[i]].cpu().numpy() * coeffs_sort[i]
                else:
                    running_total += proto_data[:, :, idx[i]].cpu().numpy() * coeffs_sort[i]

                running_total_nonlin = running_total
                if cfg.mask_proto_mask_activation == activation_func.sigmoid:
                    running_total_nonlin = (1/(1+np.exp(-running_total_nonlin)))

                arr_img[y*proto_h:(y+1)*proto_h, x*proto_w:(x+1)*proto_w] = (proto_data[:, :, idx[i]] / torch.max(proto_data[:, :, idx[i]])).cpu().numpy() * coeffs_sort[i]
                arr_run[y*proto_h:(y+1)*proto_h, x*proto_w:(x+1)*proto_w] = (running_total_nonlin > 0.5).astype(np.float)
        plt.imshow(arr_img)
        plt.show()
        # plt.imshow(arr_run)
        # plt.show()
        # plt.imshow(test)
        # plt.show()
        plt.imshow(out_masks[:, :, jdx].cpu().numpy())
        plt.show()
