import argparse
import os
import logging
import sys
import itertools
from numpy.lib.type_check import imag

import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR

from vision.utils.misc import str2bool, Timer, freeze_net_layers, store_labels
from vision.yolof.mobiledet_yolof import create_mobiledet_yolof
from vision.datasets.voc_dataset import VOCDataset
from vision.yolof.config import yolof_config
from vision.yolof.data_preprocessing import TrainAugmentation 

from vision.utils.misc import convert_image_to_rgb
from vision.yolof.uniform_loss import UniformLoss
from vision.yolof.yolof import MatchAnchor

parser = argparse.ArgumentParser(
    description='Single Shot MultiBox Detector Training With Pytorch')

parser.add_argument("--dataset_type", default="voc", type=str,
                    help='Specify dataset type. Currently support voc and open_images.')

parser.add_argument('--datasets', nargs='+', help='Dataset directory path')
parser.add_argument('--validation_dataset', help='Dataset directory path')
parser.add_argument('--balance_data', action='store_true',
                    help="Balance training data by down-sampling more frequent labels.")


parser.add_argument('--net', default="mbd-yolof",
                    help="The network architecture, it can be mb1-ssd, mb1-lite-ssd, mb2-ssd-lite, mb3-large-ssd-lite, mb3-small-ssd-lite or vgg16-ssd.")
parser.add_argument('--freeze_base_net', action='store_true',
                    help="Freeze base net layers.")
parser.add_argument('--freeze_net', action='store_true',
                    help="Freeze all the layers except the prediction head.")

parser.add_argument('--mb2_width_mult', default=1.0, type=float,
                    help='Width Multiplifier for MobilenetV2')

# Params for SGD
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='Momentum value for optim')
parser.add_argument('--weight_decay', default=5e-4, type=float,
                    help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float,
                    help='Gamma update for SGD')
parser.add_argument('--base_net_lr', default=None, type=float,
                    help='initial learning rate for base net.')
parser.add_argument('--extra_layers_lr', default=None, type=float,
                    help='initial learning rate for the layers not in base net and prediction heads.')


# Params for loading pretrained basenet or checkpoints.
parser.add_argument('--base_net',
                    help='Pretrained base model')
parser.add_argument('--pretrained_ssd', help='Pre-trained base model')
parser.add_argument('--resume', default=None, type=str,
                    help='Checkpoint state_dict file to resume training from')

# Scheduler
parser.add_argument('--scheduler', default="multi-step", type=str,
                    help="Scheduler for SGD. It can one of multi-step and cosine")

# Params for Multi-step Scheduler
parser.add_argument('--milestones', default="80,100", type=str,
                    help="milestones for MultiStepLR")

# Params for Cosine Annealing
parser.add_argument('--t_max', default=120, type=float,
                    help='T_max value for Cosine Annealing Scheduler.')

# Train params
parser.add_argument('--batch_size', default=32, type=int,
                    help='Batch size for training')
parser.add_argument('--num_epochs', default=120, type=int,
                    help='the number epochs')
parser.add_argument('--num_workers', default=4, type=int,
                    help='Number of workers used in dataloading')
parser.add_argument('--validation_epochs', default=5, type=int,
                    help='the number epochs')
parser.add_argument('--debug_steps', default=100, type=int,
                    help='Set the debug log output frequency.')
parser.add_argument('--use_cuda', default=True, type=str2bool,
                    help='Use CUDA to train model')

parser.add_argument('--checkpoint_folder', default='models/',
                    help='Directory for saving checkpoint models')


logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
args = parser.parse_args()

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() and args.use_cuda else "cpu")

if args.use_cuda and torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    logging.info("Use Cuda.")


def train(loader, net, criterion, optimizer, device, debug_steps=100, epoch=-1):
    net.train(True)
    running_loss = 0.0
    running_regression_loss = 0.0
    running_classification_loss = 0.0
    for i, data in enumerate(loader):
        images, boxes, labels = data
        # print(f"Batch Sizes:{len(images)}")
        # print(f"Batch Sizes:{boxes.shape}")  ## torch.Size([32, 400, 4])
        # print(f"Batch Sizes:{labels.shape}") ## torch.Size([32, 400])
        images = images.to(device)
        boxes = boxes.to(device)
        labels = labels.to(device)
        # print(f"labels' dtype:{labels.dtype}")

        # anchors = [config.anchors for _ in range(len(images))]   

        optimizer.zero_grad()
        pred_logits, pred_anchor_deltas = net(images) 
        ## pred_logits[0].shape: [32, 400, 20] 
        ## pred_anchor_deltas[0].shape: [32, 400, 4] 
        # print(pred_anchor_deltas[0].device)  ##cuda:0
        # print(pred_logits[0].device)   ##cuda:0

        loss_cls, loss_box_reg = criterion(pred_logits[0], pred_anchor_deltas[0], labels, boxes) 
        loss = loss_box_reg + loss_cls
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_regression_loss += loss_box_reg.item()
        running_classification_loss += loss_cls.item()
        if i and i % debug_steps == 0:
            avg_loss = running_loss / debug_steps
            avg_reg_loss = running_regression_loss / debug_steps
            avg_clf_loss = running_classification_loss / debug_steps
            logging.info(
                f"Epoch: {epoch}, Step: {i}, " +
                f"Average Loss: {avg_loss:.4f}, " +
                f"Average Regression Loss {avg_reg_loss:.4f}, " +
                f"Average Classification Loss: {avg_clf_loss:.4f}"
            )
            running_loss = 0.0
            running_regression_loss = 0.0
            running_classification_loss = 0.0
            model_path = os.path.join(args.checkpoint_folder, f"{args.net}-Epoch-{epoch}-Loss-{avg_loss}.pth")
            net.save(model_path)
            logging.info(f"Saved model {model_path}")



if __name__ == "__main__":
    timer = Timer()

    logging.info(args)
    if args.net == 'mbd-yolof':
        create_net = create_mobiledet_yolof
        config = yolof_config
    else:
        logging.fatal("The net type is wrong.")
        parser.print_help(sys.stderr)
        sys.exit(1)

    train_transform = TrainAugmentation(config.image_size, config.image_mean, config.image_std)
    target_transform = MatchAnchor(config.anchors, config.center_variance, config.size_variance)

    logging.info("Prepare training datasets.")
    for dataset_path in args.datasets:
        if args.dataset_type == 'voc':
            train_dataset = VOCDataset(dataset_path, transform=train_transform, target_transform=target_transform)
            label_file = os.path.join(args.checkpoint_folder, "voc-model-labels.txt")
            store_labels(label_file, train_dataset.class_names)
            num_classes = len(train_dataset.class_names)
        else:
            raise ValueError(f"Dataset type {args.dataset_type} is not supported.")
    logging.info(f"Stored labels into file {label_file}.")
    logging.info("Train dataset size: {}".format(len(train_dataset)))
    train_loader = DataLoader(train_dataset, args.batch_size,num_workers=args.num_workers,shuffle=True)

    logging.info("Build network.")
    net = create_net(num_classes) 
    min_loss = -10000.0
    last_epoch = -1
    base_net_lr = args.base_net_lr if args.base_net_lr is not None else args.lr
    extra_layers_lr = args.extra_layers_lr if args.extra_layers_lr is not None else args.lr

    if args.freeze_base_net:
        logging.info("Freeze base net.")
        freeze_net_layers(net.backbone)

        params = itertools.chain(net.encoder.parameters(),
                                 net.decoder.parameters())
        params = [
            {'params': itertools.chain(
                net.encoder.parameters(),
                net.decoder.parameters()
            ), 'lr': extra_layers_lr},
        ]
    else:
        params = [
            {'params': net.backbone.parameters(), 'lr': base_net_lr},
            {'params': itertools.chain(
                net.encoder.parameters(),
            ), 'lr': extra_layers_lr},
            {'params': itertools.chain(
                net.decoder.parameters()
            )}
        ]


    timer.start("Load Model")
    net.to(DEVICE)
    criterion = UniformLoss(config.anchors, DEVICE).cuda()
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum,
                                weight_decay=args.weight_decay)
    logging.info(f"Learning rate: {args.lr}, Base net learning rate: {base_net_lr}, "
                 + f"Extra Layers learning rate: {extra_layers_lr}.")

    if args.scheduler == 'multi-step':
        logging.info("Uses MultiStepLR scheduler.")
        milestones = [int(v.strip()) for v in args.milestones.split(",")]
        scheduler = MultiStepLR(optimizer, milestones=milestones,
                                                     gamma=0.1, last_epoch=last_epoch)
    elif args.scheduler == 'cosine':
        logging.info("Uses CosineAnnealingLR scheduler.")
        scheduler = CosineAnnealingLR(optimizer, args.t_max, last_epoch=last_epoch)
    else:
        logging.fatal(f"Unsupported Scheduler: {args.scheduler}.")
        parser.print_help(sys.stderr)
        sys.exit(1)

    logging.info(f"Start training from epoch {last_epoch + 1}.")
    for epoch in range(last_epoch + 1, args.num_epochs):
        scheduler.step()
        train(train_loader, net, criterion, optimizer,
              device=DEVICE, debug_steps=args.debug_steps, epoch=epoch)
        






