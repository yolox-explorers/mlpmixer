from torchvision import transforms as tv_T
import albumentations as A

def get_image_only_transform(train):
    """Performs several transforms using torchvision (Resizes to 224x224, and random horizontal flip for training images)

    Args:
        train (bool): For training data only, performs random horizontal flip is set to True.

    Returns:
        Compose class: Returns a list of transformations using torchvision
    """
    transforms = []
    transforms.append(tv_T.Resize( (224, 224) )) # 224 for resnet50 and mobilenetv2
    if train:
        transforms.append(tv_T.RandomHorizontalFlip(0.5))
    
    return tv_T.Compose(transforms)

# Albumentation
def get_image_only_aug_transform(args):
    """Performs augmentations on training images using albumentation. List of augmentations include: rotate, shear, shift (translate), zoom, brightness and contrast.

    Args:
        args: list of arguments from pipelines.yml which includes augmentation parameters e.g. angle of rotation, brightness limit, contrast limit, shear degree, zoom scale, translation percentage.

    Returns:
        Compose: A transform function that will perform a list of image augmentation.
    """
    transforms = []
    if args.shift != 0:
        shift = A.Affine(translate_percent = (-args.shift, args.shift), p = 0.5) # shift = translate
        transforms.extend([shift])
    if args.rotate != 0:
        rotation = A.Affine(rotate = (-args.rotate, args.rotate), p = 0.5)
        transforms.extend([rotation])
    if args.brightness != 0 or args.contrast != 0:
        brightness_contrast = A.RandomBrightnessContrast(brightness_limit=(-args.brightness, args.brightness), \
            contrast_limit = 0, p = 0.5)
        transforms.extend([brightness_contrast])
    if args.shear != 0:
        shear = A.Affine(shear = (-args.shear, args.shear), p = 0.5)
        transforms.extend([shear])
    if args.zoom_min != 1 or args.zoom_max != 1:
        zoom = A.Affine(scale = (args.zoom_min, args.zoom_max), p = 0.5)
        transforms.extend([zoom])
    # transforms.extend([rotation, brightness, contrast, shear, shift, zoom])
    return A.Compose(transforms) 