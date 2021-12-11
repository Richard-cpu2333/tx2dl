import argparse
import os
import logging
import sys
import itertools
# from apex.amp.scaler import axpby_check_overflow_python
import numpy as np
from numpy import dtype
from numpy import core
from numpy.core.fromnumeric import shape
from numpy.lib.type_check import imag
# from apex import amp

import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR
from torch.utils.data.dataset import ConcatDataset

from vision.utils.misc import str2bool, Timer, freeze_net_layers, store_labels
from vision.yolof.mobiledet_yolof import create_mobilenetv1_yolof, create_mobilenetv2_yolof_lite, create_mobilenetv3_large_yolof_lite, create_mobilenetv3_small_yolof_lite, create_mobiledet_yolof, create_efficientnet_yolof
from vision.datasets.voc_dataset import VOCDataset
import vision.yolof.config.yolof_config as config
from vision.yolof.data_preprocessing import PredictionTransform, TrainAugmentation, TestTransform

from vision.yolof.uniform_loss import criterion
from vision.yolof.uniform_matcher import UniformMatcher 

parser = argparse.ArgumentParser(
    description='YOLOF Detector Training With Pytorch')

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
parser.add_argument('--pretrained_yolof', help='Pre-trained base model')
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
parser.add_argument('--num_epochs', default=200, type=int,
                    help='the number epochs')
parser.add_argument('--num_workers', default=6, type=int,
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

DEVICE = torch.device("cuda:0" if torch.cuda.is_available()
                      and args.use_cuda else "cpu")

if args.use_cuda and torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    logging.info("Use Cuda.")


'''
        # print(type(boxes[0])) ## len=batch_size  <class 'numpy.ndarray'>
        # print(labels[0]) ## <class 'numpy.ndarray'>
        # images = torch.cat([torch.unsqueeze(images[i], dim=0) for i in range(len(images))]).to(device)
        # images = torch.cat(torch.unsqueeze(images, dim=1).to(device) ## images.shape: torch.Size([2, 3, 300, 300]) if batch_size==2
        images = images.to(device)
        # images = images.half().to(device)
        # print(f"Image's shape: {images.shape}")
        # boxes = torch.from_numpy(np.concatenate(boxes, axis=0))
        # print(f"boxes' shape is: {boxes.shape}")

        # labels = torch.from_numpy(np.concatenate(labels, axis=0))
        # print(f"labels' shape is: {labels.shape}")
        
        pred_class_logits, pred_anchor_deltas = net(images)
        pred_anchor_deltas = pred_anchor_deltas[0]               ## torch.Size([2, 400, 4]), if batch=2
        pred_class_logits = pred_class_logits[0]                ## torch.Size([2, 400, 21]), if batch=2
        # print(pred_class_logits)
        # print(pred_anchor_deltas)
        tt1 = MatchAnchor(pred_anchor_deltas)
        best_target_per_anchor_index, ignore_idx = tt1(boxes, labels, None, None)

        tt2 = MatchAnchor(config.anchors)
        gt_boxes, gt_labels = tt2(boxes, labels, best_target_per_anchor_index, ignore_idx)
'''      
def train(loader, net, criterion, optimizer, device, debug_steps=100, epoch=-1):
    net.train(True)
    running_loss = 0.0
    running_regression_loss = 0.0
    running_classification_loss = 0.0
    for i, data in enumerate(loader):
        images, boxes, labels = data
        images = images.to(device)
        
        gt_boxes = [torch.from_numpy(boxes[i]) for i in range(images.shape[0])]
        gt_labels = [torch.from_numpy(labels[i]) for i in range(images.shape[0])]

        pred_class_logits, pred_anchor_deltas = net(images)

        anchors = [config.anchors[None] for i in range(images.shape[0])]
        matcher = UniformMatcher(4)
        indices = matcher(pred_anchor_deltas[0], anchors, gt_boxes, gt_labels)
        loss_cls, loss_box_reg = criterion(indices, gt_boxes, gt_labels, anchors, pred_class_logits[0], pred_anchor_deltas[0])
        add_loss = loss_box_reg + loss_cls

        if not torch.isnan(add_loss):
            # optimizer.zero_grad()
            # with amp.scale_loss(loss, opt) as scaled_loss:
            #     scaled_loss.backward()
            # # loss.backward()
            # optimizer.step()
            optimizer.zero_grad()
            add_loss.backward()
            optimizer.step()

        running_loss += add_loss.item()
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


def test(loader, net, criterion, device):
    net.eval()
    running_loss = 0.0
    running_regression_loss = 0.0
    running_classification_loss = 0.0
    num = 0
    for _, data in enumerate(loader):
        images, boxes, labels = data
        images = images.to(device)
        
        gt_boxes = [torch.from_numpy(boxes[i]) for i in range(images.shape[0])]
        gt_labels = [torch.from_numpy(labels[i]) for i in range(images.shape[0])]
        num += 1
        with torch.no_grad():
            pred_class_logits, pred_anchor_deltas = net(images)

            anchors = [config.anchors[None] for i in range(images.shape[0])]
            matcher = UniformMatcher(4)
            indices = matcher(pred_anchor_deltas[0], anchors, gt_boxes, gt_labels)
            loss_cls, loss_box_reg = criterion(indices, gt_boxes, gt_labels, anchors, pred_class_logits[0], pred_anchor_deltas[0])
            loss = loss_box_reg + loss_cls
        running_loss += loss.item()
        running_regression_loss += loss_box_reg.item()
        running_classification_loss += loss_cls.item()
    return running_loss / num, running_regression_loss / num, running_classification_loss / num



def my_collate(batch):
    # images = [item[0] for item in batch]
    images = torch.cat([torch.unsqueeze(item[0], dim=0) for item in batch])
    # print(len(images))
    target = [item[1] for item in batch]
    # print(target[0].shape)
    label = [item[2] for item in batch]
    # print(label[0].shape)
    return [images, target, label]

if __name__ == "__main__":
    timer = Timer()

    logging.info(args)
    if args.net == 'mb1-yolof':
        create_net = create_mobilenetv1_yolof
        config = config
    elif args.net == 'mb2-yolof-lite':
        def create_net(num): return create_mobilenetv2_yolof_lite(
            num, width_mult=args.mb2_width_mult)
        config = config
    elif args.net == 'mb3-large-yolof-lite':
        def create_net(num): return create_mobilenetv3_large_yolof_lite(num)
        config = config
    elif args.net == 'mb3-small-yolof-lite':
        def create_net(num): return create_mobilenetv3_small_yolof_lite(num)
        config = config
    elif args.net == 'mbd-yolof':
        create_net = create_mobiledet_yolof
        config = config
    elif args.net == 'ef-yolof':
        create_net = create_efficientnet_yolof
        config = config
    else:
        logging.fatal("The net type is wrong.")
        parser.print_help(sys.stderr)
        sys.exit(1)

    train_transform = TrainAugmentation(
        config.image_size, config.image_mean, config.image_std)
    # target_transform = MatchAnchor(
    #     config.anchors, config.center_variance, config.size_variance)
    test_transform = TestTransform(
        config.image_size, config.image_mean, config.image_std)

    logging.info("Prepare training datasets.")
    datasets = []
    for dataset_path in args.datasets:
        if args.dataset_type == 'voc':
            dataset = VOCDataset(
                dataset_path, transform=train_transform, target_transform=None)
            label_file = os.path.join(
                args.checkpoint_folder, "voc-model-labels.txt")
            store_labels(label_file, dataset.class_names)
            num_classes = len(dataset.class_names)
        else:
            raise ValueError(
                f"Dataset type {args.dataset_type} is not supported.")
        datasets.append(dataset)
    logging.info(f"Stored labels into file {label_file}.")
    train_dataset = ConcatDataset(datasets)
    logging.info("Train dataset size: {}".format(len(dataset)))
    # train_loader = DataLoader(dataset, args.batch_size, num_workers=args.num_workers, shuffle=True)
    train_loader = DataLoader(dataset, args.batch_size, collate_fn=my_collate, num_workers=args.num_workers, shuffle=True)

    logging.info("Prepare Validation datasets.")
    if args.dataset_type == 'voc':
        val_dataset = VOCDataset(args.validation_dataset, transform=test_transform,
                                 target_transform=None, is_test=True)
        logging.info(val_dataset)
    elif args.dataset_type == 'coco':
        pass
    logging.info(f"validation dataset size: {len(val_dataset)}")

    val_loader = DataLoader(val_dataset, args.batch_size, collate_fn=my_collate, num_workers=args.num_workers, shuffle=False)

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
    if args.resume:
        logging.info(f"Resume from the model {args.resume}")
        net.load(args.resume)
    elif args.base_net:
        logging.info(f"Init from base net {args.base_net}")
        net.init_from_base_net(args.base_net)
    elif args.pretrained_yolof:
        logging.info(f"Init from pretrained yolof {args.pretrained_yolof}")
        net.init_from_pretrained_yolof(args.pretrained_yolof)
    logging.info(
        f'Took {timer.end("Load Model"):.2f} seconds to load the model')

    net.to(DEVICE)

    # criterion = UniformLoss()

    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # net, opt = amp.initialize(net, optimizer, opt_level='O1') 
    logging.info(f"Learning rate: {args.lr}, Base net learning rate: {base_net_lr}, "
                 + f"Extra Layers learning rate: {extra_layers_lr}.")

    if args.scheduler == 'multi-step':
        logging.info("Uses MultiStepLR scheduler.")
        milestones = [int(v.strip()) for v in args.milestones.split(",")]
        scheduler = MultiStepLR(optimizer, milestones=milestones,
                                gamma=0.1, last_epoch=last_epoch)
    elif args.scheduler == 'cosine':
        logging.info("Uses CosineAnnealingLR scheduler.")
        scheduler = CosineAnnealingLR(
            optimizer, args.t_max, last_epoch=last_epoch)
    else:
        logging.fatal(f"Unsupported Scheduler: {args.scheduler}.")
        parser.print_help(sys.stderr)
        sys.exit(1)

    logging.info(f"Start training from epoch {last_epoch + 1}.")
    for epoch in range(last_epoch + 1, args.num_epochs):
        scheduler.step()
        train(train_loader, net, criterion, optimizer,
              device=DEVICE, debug_steps=args.debug_steps, epoch=epoch)

        if epoch % args.validation_epochs == 0 or epoch == args.num_epochs - 1:
            val_loss, val_reg_loss, val_cls_loss = test(
                val_loader, net, criterion, DEVICE)
            logging.info(
                f"Epoch: {epoch}, " +
                f"Validation Loss: {val_loss:.4f}, " +
                f"Validation Regression Loss {val_reg_loss:.4f}, " +
                f"Validation Classification Loss: {val_cls_loss:.4f}"
            )
            model_path = os.path.join(
                args.checkpoint_folder, f"{args.net}-Epoch-{epoch}-Loss-{val_loss}.pth")
            net.save(model_path)
            logging.info(f"Saved model {model_path}")
