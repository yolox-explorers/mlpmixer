# Introduction

## What this project is about

This is an academic project where we trained and evaluate a 2 stage Instance Segmentation - MLPmixer model. Instance segmentation model used is Yolact-Edge model while mixer model used is `Mixer-B_16` (*training codes for YE model is not provided in this repo*).  

Our custom dataset comprises of 6000+ images of drivers undergoing circuit tests. YE model will generate masks of the vehicle as well as the test station obstacles to go through. These generated mask images are passed to MLP mixer model for training. 

The weights of the models can be found over here:
- slope station:
- park day station:
- park night station:

*Note: The purpose of this repo is to demonstrate that MLPmixer model can be coupled with other computer vision models to form a multi-stage model system. It is not the intention of the authors to demonstrate training of instance segmentation model.*

**Yolact Edge Model**

**MLP Mixer Model**


# Before you proceed

**What was done**:
- Docker image & container was created to run the training, inference and evaluation.
- Testing was done on a GPU based machine. Docker environment was built with cuda toolkit installed for running training/inference.
- For local training, use `mlpmixer-training-gpu-trt-local.Dockerfile`. First check your machine's CUDA GPU driver (**including plx**) version using `nvidia-smi`.
  - update docker image's CUDA toolkit version (**line 95**) to a version that matches your CUDA GPU driver version.
  - Update TensorRT and cuDNN versions (**line 73, 77**)  that match your updated CUDA toolkit version
    - TensorRT: https://docs.nvidia.com/deeplearning/tensorrt/release-notes/tensorrt-8.html#tensorrt-8
    - cudnn: https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/
  - MLPmixer requires installation of JAX dependencies, with min. version of 0.3.7. 
  - TensorRT, torch2trt are installed in the docker environment
- JAX requires CUDA 11.1 or newer and NVidia driver version which is at least as new as your CUDA toolkit's corresponding driver version (https://github.com/google/jax#pip-installation-gpu-cuda). 
- A dockerfile `mlpmixer-training-gpu.Dockerfile` has been created that does not use tensorrt or jax for training. This can be used as well if Tensorrt (for inference) is not required, or if you are not intending to run the jax mixer model.


## Project Repo Tree

```
project_repo
    ├── README.md           <- The top-level README containing the basic
    │                       guide for using the repository.
    ├── .dockerignore       <- File for specifying files or directories
    │                       to be ignored by Docker contexts.
    ├── .gitignore          <- File for specifying files or directories
    │                       to be ignored by Git.
    ├── assets/             <- Folder containing images for README.md
    ├── data/               <- Folder containing images files for model training
    │   ├── cleaned         <- Folder containing raw zipped folders for 
    |   |                   training image
    │   └── extracted       <- Folder containing extracted unzipped folders for 
    |                       training images, and image metadata stored in .pkl file
    └── vision_transformers/<- Folder containing the source code and
        |                   packages for the project repository.
        |── model_weights/  <- Folder containing training weights for Mixer
        |                   and YE weights for mask generation for 3 Mixer training
        |── vit_jax/        <- model files for Mixer, ViT (jax)
        |── models/         <- Model files for Mixer (non-jax)
        |── utils/          <- Model support files for Mixer (non-jax)
        |── eval/           <- Folder to save evaluation results
        |── yolact_edge/    <- Model files for Yolact Edge instance segmentation model
        ├── docker/
        │   ├── mlpmixer-training-gpu-trt.Dockerfile         
        |   |               ^- Dockerfile for creating Docker image with tensortrt installed
        │   └── mlpmixer-training-gpu.Dockerfile       
        |                   ^- Dockerfile for creating Docker image with no tensortrt or jax installed
        |── data_transforms.py
        |                   ^- Image transformation and augmentation for YE
        |── dataloaders.py  <- Dataloaders for mixer model (non-jax) training
        |── model.py        <- mixer model (non-jax) scripts
        |── train.py        <- mixer model (non-jax) training scripts
        |── inference.py    <- python script for inference for non-jax mixer model
        |── evaluate.py     <- python script for evaluation of results or non-jax mixer model
        |── mlpmixer_helper.py
        |                   ^- helper functions for mlpmixer (jax)
        |── mlpmixer_train.py
        |                   ^- model training script for mlpmixer (jax)
        └── requirements.yml
                            ^- Yaml file to set up conda environment for training, eval, inference
 

```

# Docker environment
- To build docker image **locally**: 
`docker build --network=host -t <your_dockerhub_username>/mlpmixer-local:v1 -f "<local_project_dir>/vision_transformers/docker/mlpmixer-training-gpu-trt-local.Dockerfile" .`
- To run container for **first time**:
`docker container run --gpus all -it -v "<local_project_dir>/vision_transformers":/home/user/MLPmixer/vision_transformers --name mlpmixer-local <your_dockerhub_username>/mlpmixer-local:v1 bash`
  - "-v" will map files in specified local directory to the specified container directory
  - "--gpus all" will allow docker container to use GPUs on your local machine
  - "--name" gives name for your container
- Get docker container id from running `docker ps`
- Copy training image files from local machine to container
`docker cp "<local_project_dir>/data/cleaned" mlpmixer-local:/home/user/MLPmixer/data`
- In docker container bash terminal, unzip image folders and generate image pickle file.
`python -m src.prep_data data_prep.img_zip_dir=/home/user/MLPmixer/data/cleaned data_prep.img_save_dir=/home/user/MLPmixer/data/extracted`
- (Optional) Copy out extracted images and pickle file to local computer. 
`docker cp mlpmixer-local:/home/user/MLPmixer/data/extracted "<local_project_dir>\data"`
- (Optional) If **another** fresh container is created, you can copy the extracted images from local computer into the new container
`docker cp "<local_project_dir>\data\extracted" mlpmixer-local:/home/user/MLPmixer/data`
- To stop container, run `docker stop mlpmixer-local`
- To restart container, run `docker start mlpmixer-local`
- After container has been started, enter container's bash terminal by running `docker exec -it mlpmixer-local bash`
- To copy training images into docker environment, run `docker cp "<local_project_dir>\data\extracted" <container_id>:/home/user/MLPmixer/data`

# Dataloader
- https://discuss.pytorch.org/t/not-using-multiprocessing-but-getting-cuda-error-re-forked-subprocess/54610/10
- CUDA library requests "spawn" while Dataloader requests "fork". Set `train_num_workers` and `test_num_workers` for train and val to 0 to disable multiprocessing. Use num_workers=0 to disable multiprocessing. (RuntimeError: Cannot re-initialize CUDA in forked subprocess. To use CUDA with multiprocessing, you must use the 'spawn' start method). 
- `train_batch_size` and `test_batch_size` must be in multiples of `accum_steps`, i.e. if `accum_steps` is set to 8, `train_batch_size` and `test_batch_size` can be set as 8, 16, 24 etc.
- Adjust `train_batch_size`, `test_batch_size` and `accum_steps` according to GPU memory availability error messages.


# Local Training
## Jax

For jax version of the model training, it can only be done on devices with Nvidia CUDA drive version >=11.1. It uses `mlpmixer_train.py`, `mlpmixer_helper.py` and mixer model files from `vit_jax` folder. (**Note: we are unable to output saved weights from the jax model training script.**)

- To carry out training, execute `python -m vision_transformers.mlpmixer_train`
- Flags to know:
  - train_shuffle: Set as `True` to shuffle train images. Default as False
  - test_shuffle: Set as `False` to **not shuffle** train images. Default as True.
  - train_mode: Set as `True` to train model. Default as False
  - augment: Set as `True` to augment images for training. Default as False
  - total_steps: Set as 0 to use all training images for training.
  - train_dataset: use "train" for 80% data for training, "train+val" for 90% training, "train+val+test" for 100% training.
  - epochs: number of epochs to train
  - ye_model_config: set to `yolact_edge_mobilenetv2_custom_config_slope_champion_3_zoom_5` as for slope station and `yolact_edge_mobilenetv2_custom_config_park_champion_3_zoom_5` for park station

Sample codes for Slope (augment)
```
python -m vision_transformers.mlpmixer_train --station=slope --time=day --train_batch_size=8 --test_batch_size=8 --epochs=30 --validation_epochs=5 --total_steps=0 --train_mode=True --image_resize_factor=0.1809 --train_num_workers=0 --train_shuffle=True --test_num_workers=0 --test_shuffle=False --patience=20 --seed=42 --ye_model_config=yolact_edge_mobilenetv2_custom_config_slope_champion_3_zoom_5 --augment=True --shift=0.05 --rotate=8 --shear=8 --brightness=0.8 --contrast=0.8 --zoom_min=0.8 --zoom_max=1.2 --crop_option=A --crop_option_a_retain_background=False --train_dataset="train" --model_save=True --save_cropped_images=False
```

Sample codes for Slope (non-augment)
```
python -m vision_transformers.mlpmixer_train --station=slope --time=day --train_batch_size=8 --test_batch_size=8 --epochs=30 --validation_epochs=5 --total_steps=50 --train_mode=True --image_resize_factor=0.1809 --train_num_workers=0 --train_shuffle=True --test_num_workers=0 --test_shuffle --test_shuffle=False --patience=20 --seed=42 --ye_model_config=yolact_edge_mobilenetv2_custom_config_slope_champion_3_zoom_5 --train_dataset=train
```

## Pytorch (non-jax)
For pytorch (non jax) mixer model, it uses `train.py`, `model.py`, `dataloaders.py` and mixer model files from `models` folder. This training script is able to output model weights for later inference and evaluation.

Slope station - 80% training data
```
python -m vision_transformers.train --station=slope --time=day --pretrained_dir="/home/user/MLPmixer/vision_transformers/model_weights/imagenet21k_Mixer-B_16.npz" --train_batch_size=8 --eval_batch_size=8 --epochs=1 --eval_every=25 --train_mode=True --image_resize_factor=0.1809 --seed=42 --ye_model_config=yolact_edge_mobilenetv2_custom_config_slope_champion_3_zoom_5 --augment=True --shift=0.05 --rotate=8 --shear=8 --brightness=0.8 --contrast=0.8 --zoom_min=0.8 --zoom_max=1.2 --crop_option=A --crop_option_a_retain_background=False --train_dataset="train" --save_cropped_images=False
```


park day station - 80% training data
```
python -m vision_transformers.train --station=park --time=day --pretrained_dir="/home/user/MLPmixer/vision_transformers/model_weights/imagenet21k_Mixer-B_16.npz" --train_batch_size=8 --eval_batch_size=8 --epochs=1 --eval_every=25 --train_mode=True --image_resize_factor=0.1809 --seed=42 --ye_model_config=yolact_edge_mobilenetv2_custom_config_park_champion_3_zoom_5 --augment=True --shift=0.05 --rotate=8 --shear=8 --brightness=0.8 --contrast=0.8 --zoom_min=0.8 --zoom_max=1.2 --crop_option=A --crop_option_a_retain_background=False --train_dataset="train" --save_cropped_images=False
```

park night station - 80% training data
```
python -m vision_transformers.train --station=park --time=night --pretrained_dir="/home/user/MLPmixer/vision_transformers/model_weights/imagenet21k_Mixer-B_16.npz" --train_batch_size=8 --eval_batch_size=8 --epochs=1 --eval_every=25 --train_mode=True --image_resize_factor=0.1809 --seed=42 --ye_model_config=yolact_edge_mobilenetv2_custom_config_park_champion_3_zoom_5 --augment=True --shift=0.05 --rotate=8 --shear=8 --brightness=0.8 --contrast=0.8 --zoom_min=0.8 --zoom_max=1.2 --crop_option=A --crop_option_a_retain_background=False --train_dataset="train" --save_cropped_images=False
```



# Inference

```
python -m vision_transformers.inference --station=slope --time=day --image_resize_factor=0.1809 --show_window=True --image_source_folder="/home/user/MLPmixer/data/extracted/20220309_1542721/slope_off_sunny_part2_1/images" --source=1 --image_output_folder="/home/user/MLPmixer/vision_transformers/ye_images/" --ye_weights_path="/home/user/MLPmixer/vision_transformers/model_weights/slope_yolact_edge_mobilenetv2_100_46258_zoom_5.pth" --ye_model_config="yolact_edge_mobilenetv2_custom_config" --mixer_fpath="/home/user/MLPmixer/vision_transformers/model_weights/slope_mlpmixer_epoch_1_50.pth" --disable_tensorrt=True
```


# Eval

Mixer model training was done for `slope`, `park day` and `park night` stations. For each training, **50** epochs were performed and evaluation was done every **25** batches of training images to get the validation loss. The model with the lowest validation loss was saved and used for evaluation.

slope station:  35 epochs, 7825 training batches
park day:      27 epochs, 5450 training batches
park night:    39 epochs, 7925 training batches


### slope station - 80% training data, 10% validation data
Eval Command:
```
python -m vision_transformers.evaluate --station=slope --time=day --image_catalog_path=/home/user/MLPmixer/data/extracted/clean_img_data.pkl --image_resize_factor=0.1809 --image_output_folder="/home/user/MLPmixer/vision_transformers/eval" --ye_weights_path="/home/user/MLPmixer/vision_transformers/model_weights/slope_yolact_edge_mobilenetv2_100_46258_zoom_5.pth" --ye_model_config="yolact_edge_mobilenetv2_custom_config" --mixer_fpath=/home/user/MLPmixer/vision_transformers/model_weights/checkpoint_weights/<path_to_weights> --val_dataset=val
```

**Evaluation Results (slope day)**

val set -   weighted f1 score [0.8014], accuracy [0.8072]

<img src="https://github.com/yolox-explorers/mlpmixer/blob/main/assets/slope_val_confusion_matrix.jpg" alt="slope validation cm" width="640" height="480"/>


test set -   weighted f1 score [0.6936], accuracy [0.7074]

<img src="https://github.com/yolox-explorers/mlpmixer/blob/main/assets/slope_val_confusion_matrix.jpg" alt="slope test cm" width="640" height="480"/>


### park day station - 80% training data, 10% validation data
Eval Command:
```
python -m vision_transformers.evaluate --station=park --time=day --image_catalog_path=/home/user/MLPmixer/data/extracted/clean_img_data.pkl --image_resize_factor=0.1809 --image_output_folder="/home/user/MLPmixer/vision_transformers/eval" --ye_weights_path="/home/user/MLPmixer/vision_transformers/model_weights/park_day_yolact_edge_mobilenetv2_100_41208_zoom_5.pth" --ye_model_config="yolact_edge_mobilenetv2_custom_config" --mixer_fpath=/home/user/MLPmixer/vision_transformers/model_weights/checkpoint_weights/<path_to_weights> --val_dataset=val
```

**Evaluation Results (Park day):**

val set -   weighted f1 score [0.9668], accuracy [0.9667]
<img src="https://github.com/yolox-explorers/mlpmixer/blob/main/assets/park_day_val_confusion_matrix.jpg" alt="park day validation cm" width="640" height="480"/>

test set -   weighted f1 score [0.9783], accuracy [0.9783]
<img src="https://github.com/yolox-explorers/mlpmixer/blob/main/assets/park_day_test_confusion_matrix.jpg" alt="park day test cm" width="640" height="480"/>
      

### park night station - 80% training data, 10% validation data
Eval Command:
```
python -m vision_transformers.evaluate --station=park --time=night --image_catalog_path=/home/user/MLPmixer/data/extracted/clean_img_data.pkl --image_resize_factor=0.1809 --image_output_folder="/home/user/MLPmixer/vision_transformers/eval" --ye_weights_path="/home/user/MLPmixer/vision_transformers/model_weights/park_night_yolact_edge_mobilenetv_100_41309_zoom_5.pth" --ye_model_config="yolact_edge_mobilenetv2_custom_config" --mixer_fpath=/home/user/MLPmixer/vision_transformers/model_weights/checkpoint_weights/<path_to_weights> --val_dataset=val
```

**Evaluation Results (Park Night):**

val set -   weighted f1 score [0.9750], accuracy [0.9750]
<img src="https://github.com/yolox-explorers/mlpmixer/blob/main/assets/park_night_val_confusion_matrix.jpg" alt="park night validation cm" width="640" height="480"/>

test set -   weighted f1 score [0.9604], accuracy [0.9604]
<img src="https://github.com/yolox-explorers/mlpmixer/blob/main/assets/park_night_test_confusion_matrix.jpg" alt="park night test cm" width="640" height="480"/>


# Credits
Credits to the following team mates for contributing to repo!
- Mr Tan Choon Meng
- Ms Joy Lin
- Mr Ooi Zi Wei Matt