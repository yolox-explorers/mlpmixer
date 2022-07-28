from turtle import clear
import matplotlib.pyplot as plt
from vision_transformers.vit_jax import input_pipeline
import jax
import numpy as np
import tqdm
import torch
import tensorflow as tf
from vision_transformers.vit_jax import checkpoint
from vision_transformers.vit_jax import models
from vision_transformers.vit_jax.configs import models as models_config
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, confusion_matrix

labelnames = dict(
    # https://www.cs.toronto.edu/~kriz/cifar.html
    cifar10=('airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'),
    # https://www.cs.toronto.edu/~kriz/cifar.html
    cifar100=('apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'computer_keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm')
)
def make_label_getter(dataset):
    """Returns a function converting label indices to names."""
    def getter(label):
        if dataset in labelnames:
            return labelnames[dataset][label]
        return f'label={label}'
    return getter

def show_img(img, ax=None, title=None):
    """Shows a single image."""
    if ax is None:
      ax = plt.gca()
    ax.imshow(img[...])
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
      ax.set_title(title)

def show_img_grid(imgs, titles):
    """Shows a grid of images."""
    n = int(np.ceil(len(imgs)**.5))
    _, axs = plt.subplots(n, n, figsize=(3 * n, 3 * n))
    for i, (img, title) in enumerate(zip(imgs, titles)):
        img = (img + 1) / 2  # Denormalize
        show_img(img, axs[i // n][i % n], title)

def get_accuracy_original(params_repl, vit_apply_repl, ds_test):
    """Returns accuracy evaluated on the test set."""
    good = total = 0
    steps = input_pipeline.get_dataset_info('cifar10', 'test')['num_examples'] // 512
    for _, batch in zip(tqdm.trange(steps), ds_test.as_numpy_iterator()):
        predicted = vit_apply_repl(params_repl, batch['image'])
        # print(predicted)
        # print(batch['label'])
        is_same = predicted.argmax(axis=-1) == batch['label'].argmax(axis=-1)
        good += is_same.sum()
        total += len(is_same.flatten())
    return good / total


def get_accuracy_cifar10(params_repl, vit_apply_repl, ds_test):
    """Returns accuracy evaluated on the test set."""
    good = total = 0
    steps = len(ds_test)
    actual_labels = []
    predicted_labels = []
    for _, batch in zip(tqdm.trange(1, steps), ds_test):
        batch['image'] = batch['image'].cpu().numpy()
        batch['label'] = batch['label'].cpu().numpy()
        actual_labels.extend(batch['label'].argmax(axis = 2)[0])

        if len(batch['image'].shape) == 3:
            batch['image'] = np.expand_dims(np.expand_dims(batch['image'], 0), 0)

        elif len(batch['image'].shape) == 4:
            batch['image'] = np.expand_dims(batch['image'], 0)
        
        predicted = vit_apply_repl(params_repl, batch['image'])
        print(f"Predicted: {predicted}")
        predicted = predicted.argmax(axis=2)
        predicted_labels.extend(predicted[0])

        # if batch['label'].shape != predicted.shape:
        #     is_same = predicted.argmax(axis=-1) == batch['label']
        # else:
        #     is_same = predicted.argmax(axis=-1) == batch['label'].argmax(axis=-1)

    print(confusion_matrix(actual_labels, predicted_labels))
    print(classification_report(actual_labels, predicted_labels))
    
    good = (np.array(actual_labels) == np.array(predicted_labels)).sum()
    total = len(actual_labels)

    return good / total

def get_accuracy_mixer(params_repl, vit_apply_repl, ds_test):
    """Returns accuracy evaluated on the test set."""
    good = total = 0
    steps = len(ds_test)
    actual_labels = []
    predicted_labels = []
    for _, batch in zip(tqdm.trange(1, steps), ds_test):
        batch['image'] = batch['image'].cpu().numpy()
        batch['label'] = batch['label'].cpu().numpy()
        actual_labels.extend(batch['label'])

        # should be (1, batch_size, width, height, channels)
        if len(batch['image'].shape) == 3:  #(width, height, channels)
            batch['image'] = np.expand_dims(np.expand_dims(batch['image'], 0), 0)

        elif len(batch['image'].shape) == 4:    #(batch_size, width, height, channels)
            batch['image'] = np.expand_dims(batch['image'], 0)
        
        predicted = vit_apply_repl(params_repl, batch['image'])
        predicted = predicted.argmax(axis=-1)
        predicted_labels.extend(predicted[0])

        # if batch['label'].shape != predicted.shape:
        #     is_same = predicted.argmax(axis=-1) == batch['label']
        # else:
        #     is_same = predicted.argmax(axis=-1) == batch['label'].argmax(axis=-1)

    print(confusion_matrix(actual_labels, predicted_labels))
    print(classification_report(actual_labels, predicted_labels))
    
    good = (np.array(actual_labels) == np.array(predicted_labels)).sum()
    total = len(actual_labels)

    return good / total

def get_accuracy(params_repl, vit_apply_repl, ds_test):
    """Returns accuracy evaluated on the test set."""
    good = total = 0
    steps = len(ds_test)
    actual_labels = []
    predicted_labels = []
    for _, batch in zip(tqdm.trange(1, steps), ds_test):
        batch['image'] = batch['image'].cpu().numpy()
        batch['label'] = batch['label'].cpu().numpy()
        actual_labels.extend(batch['label'])

        if len(batch['image'].shape) == 3:
            batch['image'] = np.expand_dims(np.expand_dims(batch['image'], 0), 0)

        elif len(batch['image'].shape) == 4:
            batch['image'] = np.expand_dims(batch['image'], 0)
        
        predicted = vit_apply_repl(params_repl, batch['image'])
        predicted = predicted.argmax(axis=-1)[0]
        predicted_labels.extend(predicted.item())

        # if batch['label'].shape != predicted.shape:
        #     is_same = predicted.argmax(axis=-1) == batch['label']
        # else:
        #     is_same = predicted.argmax(axis=-1) == batch['label'].argmax(axis=-1)

    print(confusion_matrix(actual_labels, predicted_labels))
    print(classification_report(actual_labels, predicted_labels))
    
    good = (np.array(actual_labels) == np.array(predicted_labels)).sum()
    total = len(actual_labels)

    return good / total