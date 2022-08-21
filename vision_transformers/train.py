from __future__ import absolute_import, division, print_function
import ast
import logging
import argparse
import jax
import os
import sys
import subprocess
import random
import numpy as np
import torch
import torch.distributed as dist
from datetime import timedelta, datetime
from tqdm import tqdm
from tensorflow.keras.utils import to_categorical
from torch.utils.tensorboard import SummaryWriter
from vision_transformers.dataloaders import DataLoader
from vision_transformers.models.modeling import MlpMixer, CONFIGS
from vision_transformers.utils.scheduler import WarmupLinearSchedule, WarmupCosineSchedule
from vision_transformers.utils.data_utils import get_loader
from vision_transformers.utils.dist_utils import get_world_size
from sklearn.metrics import classification_report, f1_score

parser = argparse.ArgumentParser(description='MLPmixer training Script')

parser.add_argument('--num_classes', default = 2, type = int)
parser.add_argument('--model_name', default = 'Mixer-B_16', type = str, choices = ['ViT-B_32', 'Mixer-B_16'],\
    help = "Model name either 'ViT-B_32' or 'Mixer-B_16'")
parser.add_argument('--epochs', default = 30, type = int,
                    help = "number of epochs for training")
parser.add_argument('--patience', default = 20, type = int,
                    help = "Early stopping steps before training stops")
parser.add_argument('--image_catalog_path', default = '/home/user/MLPmixer/data/extracted/clean_img_data.pkl', type = str)
parser.add_argument('--station', default = 'slope', required = True, type = str, choices = ['slope', 'park'])
parser.add_argument('--time', default = 'day', required = True, type = str, choices = ['day', 'night'])
parser.add_argument('--image_resize_factor', default = 0.1809, type = float,\
    help = "Scale factor to resize training images to feed to YE model for inference")
parser.add_argument('--train_shuffle', default = True, \
    type = ast.literal_eval, dest='train_shuffle')
parser.add_argument('--train_num_workers', default = 0, type = int)
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


# MLPmixer parameters
parser.add_argument('--patch_size', default = 16, type = int, help = 'width, height of each divided grid of image')
parser.add_argument('--num_gpus', default=1, type=int, help='Number of GPUs used in training')
parser.add_argument("--name", default="mlpmixer",
                    help="Name of this run. Used for monitoring.")
parser.add_argument("--model_type", choices=["Mixer-B_16", "Mixer-L_16",
                                                "Mixer-B_16-21k", "Mixer-L_16-21k"],
                    default="Mixer-B_16",
                    help="Which model to use.")
parser.add_argument("--pretrained_dir", type=str, required=True,\
                    help="Where to search for pretrained ViT models.")
parser.add_argument("--output_dir", type=str, \
                    default="/home/user/MLPmixer/vision_transformers/model_weights/checkpoint_weights",
                    help="The output directory where checkpoints will be written.")

parser.add_argument("--train_batch_size", default=8, type=int,
                    help="Total batch size for training.")
parser.add_argument("--eval_batch_size", default=8, type=int,
                    help="Total batch size for eval.")

parser.add_argument("--learning_rate", default=3e-2, type=float,
                    help="The initial learning rate for SGD.")
parser.add_argument("--weight_decay", default=0, type=float,
                    help="Weight deay if we apply some.")
parser.add_argument("--eval_every", default=2, type=int,
                    help="Run prediction on validation set every so many steps."
                            "Will always run one evaluation at the end of training.")
parser.add_argument("--decay_type", choices=["cosine", "linear"], default="cosine",
                    help="How to decay the learning rate.")
parser.add_argument("--warmup_steps", default=500, type=int,
                    help="Step of training to perform learning rate warmup for.")
parser.add_argument("--max_grad_norm", default=1.0, type=float,
                    help="Max gradient norm.")

parser.add_argument("--local_rank", type=int, default=-1,
                    help="local_rank for distributed training on gpus")
parser.add_argument('--seed', type=int, default=42,
                    help="random seed for initialization")
parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                    help="Number of updates steps to accumulate before performing a backward/update pass.")
parser.add_argument('--fp16', action='store_true',
                    help="Whether to use 16-bit float precision instead of 32-bit")
parser.add_argument('--fp16_opt_level', type=str, default='O2',
                    help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                            "See details at https://nvidia.github.io/apex/amp.html")
parser.add_argument('--loss_scale', type=float, default=0,
                    help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                            "0 (default value): dynamic loss scaling.\n"
                            "Positive power of 2: static loss scaling value.\n")

# YE generated mask options
parser.add_argument('--ye_weights_path', type = str, help = 'filepath for ye_weights')
parser.add_argument('--ye_model_config', default="yolact_edge_mobilenetv2_custom_config", type = str)
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

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def save_model(args, model, now, epoch, global_step):
    model_to_save = model.module if hasattr(model, 'module') else model
    model_checkpoint = os.path.join(args.output_dir, args.station, args.time, str(now), "%s_epoch_%d_%d.pth" % (args.name, epoch, global_step))
    torch.save(model_to_save.state_dict(), model_checkpoint)
    logger.info("Saved model checkpoint to [DIR: %s]", args.output_dir)


def setup(args):
    # Prepare model
    config = CONFIGS[args.model_type]

    model = MlpMixer(config, args.img_size, num_classes=args.num_classes, patch_size=args.patch_size, zero_head=True)
    model.load_from(np.load(args.pretrained_dir))
    model.to(args.device)
    num_params = count_parameters(model)

    logger.info("{}".format(config))
    logger.info("Training parameters %s", args)
    logger.info("Total Parameter: \t%2.1fM" % num_params)
    return model


def count_parameters(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params/1000000


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def valid(args, model, writer, test_loader, global_step):
    # Validation!
    eval_losses = AverageMeter()

    logger.info("***** Running Validation *****")
    logger.info("  Num steps = %d", len(test_loader))
    logger.info("  Batch size = %d", args.eval_batch_size)

    model.eval()
    all_preds, all_label = [], []
    epoch_iterator = tqdm(test_loader,
                          desc="Validating... (loss=X.X)",
                          bar_format="{l_bar}{r_bar}",
                          dynamic_ncols=True,
                          disable=args.local_rank not in [-1, 0])
    loss_fct = torch.nn.CrossEntropyLoss()

    for step, batch in enumerate(epoch_iterator):
        batch = tuple(t.to(args.device) for t in batch)
        x, y = batch
        
        with torch.no_grad():
            logits = model(x)
            eval_loss = loss_fct(logits, y)
            eval_losses.update(eval_loss.item())
            preds = torch.argmax(logits, dim=-1)

        if len(all_preds) == 0:
            all_preds.append(preds.detach().cpu().numpy())
            all_label.append(y.detach().cpu().numpy())
        else:
            all_preds[0] = np.append(
                all_preds[0], preds.detach().cpu().numpy(), axis=0
            )
            all_label[0] = np.append(
                all_label[0], y.detach().cpu().numpy(), axis=0
            )
        epoch_iterator.set_description("Validating... (loss=%2.5f)" % eval_losses.val)

    all_preds, all_label = all_preds[0], all_label[0]
    logger.info(f"Predicted: {all_preds}")
    logger.info(f"Actual: {all_label}")
    accuracy = simple_accuracy(all_preds, all_label)

    logger.info("\n")
    logger.info("Validation Results")
    logger.info(f"Classification Report: {classification_report(all_label, all_preds)}")
    logger.info("Valid Loss: %2.5f" % eval_losses.avg)
    logger.info("Valid Accuracy: %2.5f" % accuracy)
    logger.info(f"f1-score: {f1_score(all_label, all_preds)}")

    writer.add_scalar("test/accuracy", scalar_value=accuracy, global_step=global_step)
    return accuracy, eval_losses.avg


def train(args, model, train_loader, test_loader):
    """ Train the model """

    now = datetime.now().strftime("%d%m%Y-%H%M%S")
    if args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=os.path.join(args.output_dir, args.station, args.time, now, "logs", args.name))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    # Prepare dataset (for CIFAR100)
    # train_loader, test_loader = get_loader(args)

    # Prepare optimizer and scheduler
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=args.learning_rate,
                                momentum=0.9,
                                weight_decay=args.weight_decay)

    # Controlling number of steps or batches of training images per epoch for training
    # t_total = args.num_steps
    t_total = args.epochs* len(train_loader)


    if args.decay_type == "cosine":
        scheduler = WarmupCosineSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
    else:
        scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)

    if args.fp16:
        model, optimizer = amp.initialize(models=model,
                                          optimizers=optimizer,
                                          opt_level=args.fp16_opt_level)
        amp._amp_state.loss_scalers[0]._loss_scale = 2**20

    # Distributed training
    if args.local_rank != -1:
        model = DDP(model, message_size=250000000, gradient_predivide_factor=get_world_size())

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Total optimization steps = %d", t_total)
    logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps * (
                    torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)

    model.zero_grad()
    set_seed(args)  # Added here for reproducibility (even between python 2 and 3)
    losses = AverageMeter()
    global_step, best_epoch, best_step, best_acc, best_val_loss = 0, 0, 0, 0, 10

    while True:
        model.train()
        epoch_iterator = tqdm(train_loader,
                              desc="Training (X / X Steps) (loss=X.X)",
                              bar_format="{l_bar}{r_bar}",
                              dynamic_ncols=True,
                              disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            batch = tuple(t.to(args.device) for t in batch)
            x, y = batch
            loss = model(x, y)

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                losses.update(loss.item()*args.gradient_accumulation_steps)
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scheduler.step()
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

                epoch_iterator.set_description(
                    "Training (%d / %d Steps) (loss=%2.5f)" % (global_step, t_total, losses.val)
                )
                if args.local_rank in [-1, 0]:
                    writer.add_scalar("train/loss", scalar_value=losses.val, global_step=global_step)
                    writer.add_scalar("train/lr", scalar_value=scheduler.get_lr()[0], global_step=global_step)
                if global_step % args.eval_every == 0 and args.local_rank in [-1, 0]:
                    accuracy, val_loss = valid(args, model, writer, test_loader, global_step)

                    if val_loss < best_val_loss:
                        epoch = int(np.floor(global_step/len(train_loader))) + 1
                        save_model(args, model, now, epoch, global_step)
                        best_val_loss = val_loss
                        best_acc = accuracy
                        best_epoch = epoch
                        best_step = global_step
                    # if best_acc < accuracy:
                    #     save_model(args, model, now)
                    #     best_acc = accuracy
                    # model.train()

                if global_step % t_total == 0:
                    break
        losses.reset()
        if global_step % t_total == 0:
            break

    if args.local_rank in [-1, 0]:
        writer.close()
    logger.info(f"Best epoch: {best_epoch}. \tBest step: {best_step}. \t Best Accuracy: {best_acc}. \t Best Val Loss: {best_val_loss}. \t")

if __name__ == "__main__":
    args = parser.parse_args()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl',
                                             timeout=timedelta(minutes=60))
        args.n_gpu = 1

    args.device = device
    args.img_size = 224
    if args.fp16:
        from apex import amp
    if args.local_rank != -1:
        from apex.parallel import DistributedDataParallel as DDP

    # Setup logging

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger = logging.getLogger('mlpmixer')
    logger.info(jax.local_devices())
    logger.info(f"Device: {args.device}")
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s" %
                   (args.local_rank, args.device, args.n_gpu, bool(args.local_rank != -1), args.fp16))

    # Downloads model weights for first time into docker container
    if not os.path.exists(args.pretrained_dir):
        subprocess.run(["gsutil", "cp", "gs://mixer_models/imagenet21k/Mixer-B_16.npz", args.pretrained_dir])

    assert os.path.exists(args.pretrained_dir), f"{args.model_name} weights does not exist"

    # asserts that station is within list
    # Checks if YE weights exists
    if args.ye_weights_path is None:
        if args.station == 'slope':
            args.ye_weights_path = "/home/user/MLPmixer/vision_transformers/model_weights/slope_yolact_edge_mobilenetv_100_46258_zoom_5.pth"
        elif (args.station == 'park') & (args.time == 'day'):
            args.ye_weights_path = "/home/user/MLPmixer/vision_transformers/model_weights/park_day_yolact_edge_mobilenetv2_100_41208_zoom_5.pth"
        elif (args.station == 'park') & (args.time == 'night'):
            args.ye_weights_path = "/home/user/MLPmixer/vision_transformers/model_weights/park_night_yolact_edge_mobilenetv2_100_41309_zoom_5.pth"
    assert os.path.exists(args.ye_weights_path), f"{args.ye_weights_path} does not exist"

    if './vision_transformers' not in sys.path:
        sys.path.append('./vision_transformers')

    logger.info(f"'train_mode', {args.train_mode}, {type(args.train_mode)}")
    logger.info(f"'train_shuffle', {args.train_shuffle}, {type(args.train_shuffle)}")
    logger.info(f"'test_shuffle', {args.test_shuffle}, {type(args.test_shuffle)}")
    logger.info(f"'augment', {args.augment}, {type(args.augment)}")

    # Set seed
    set_seed(args)

    # Performs check on augmentation inputs:
    if args.train_mode:
        if args.augment:
            logger.info(f'Augmentation Flag is set to True. Training {args.model_name} using augment images')
            assert args.zoom_max >= args.zoom_min, "Max zoom factor must be larger than min zoom factor"

    # Loads datasets
    dataloader = DataLoader(args)
    ds_train, ds_val = dataloader.dataloader()

    # Model & Tokenizer Setup
    model = setup(args)

    # Training
    if args.train_mode:
        train(args, model, ds_train, ds_val)
