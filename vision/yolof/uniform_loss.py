from os import posix_fallocate
import torch
from torch.functional import Tensor
import torch.nn as nn
import torch.nn.functional as F
from fvcore.nn import sigmoid_focal_loss_jit, giou_loss
import torch.distributed as dist
from vision.utils import box_utils
from vision.utils.box_utils import assign_anchors, iou_of
from . import comm
from vision.yolof.box_regression import YOLOFBox2BoxTransform
import math



def cat(tensors, dim: int = 0):
    """
    Efficient version of torch.cat that avoids a copy if there is only a single element in a list
    """
    assert isinstance(tensors, (list, tuple))
    if len(tensors) == 1:
        return tensors[0]
    return torch.cat(tensors, dim)


class UniformLoss(nn.Module):
    def __init__(self, anchors, neg_pos_ratio=3, neg_ignore_thresh=0.7, pos_ignore_thresh=0.3):
        super(UniformLoss, self).__init__()
        self.neg_pos_ratio = neg_pos_ratio
        self.neg_ignore_thresh = neg_ignore_thresh
        self.pos_ignore_thresh = pos_ignore_thresh
        self.anchors = anchors
        '''
        ## pred_class_logits.shape: [32, 600, 21] 
        ## pred_anchor_deltas.shape: [32, 600, 4] 
        ## labels.shape: []
        '''

    def forward(self, pos_labels, pos_anchors, picked_labels, pred_anchor_deltas, pred_class_logits):
        N = pred_class_logits.shape[0]
        NUM_CLASSES = pred_class_logits.shape[2]

        # pred_class_logits = pred_class_logits.view(-1, NUM_CLASSES)
        predicted_boxes = box_utils.apply_deltas(pred_anchor_deltas, self.anchors)
        predicted_boxes = predicted_boxes.reshape(N, -1, 4)
        # ignore_idx = picked_labels > self.neg_ignore_thresh
        # pos_labels[ignore_idx] = -1
        # pos_ignore_idx = pos_labels < self.pos_ignore_thresh
        # pos_labels[pos_ignore_idx] = -1
        # mask1 = pos_labels >= 0

        with torch.no_grad():
            # derived from cross_entropy=sum(log(p))    
            loss = -F.log_softmax(pred_class_logits, dim=2)[:, :, 0]

            # mask_predict = box_utils.hard_negative_mining(loss, picked_labels, self.neg_pos_ratio)
            mask = box_utils.hard_negative_mining(loss, pos_labels, self.neg_pos_ratio)

        classification_loss = F.cross_entropy(pred_class_logits[mask, :].reshape(-1, NUM_CLASSES), pos_labels[mask], size_average=False)

        # print(f"confidence info:{confidence.dtype}") ## torch.size([12800,21]), torch.float32
        # print(f"labels info:{labels.dtype}") ## torch.Size([32, 400]), torch.int64
        pos_mask = pos_labels > 0
        predicted_boxes = predicted_boxes[pos_mask, :].reshape(-1, 4)
        pos_anchors = pos_anchors[pos_mask, :].reshape(-1, 4)
        smooth_l1_loss = F.smooth_l1_loss(predicted_boxes, pos_anchors, size_average=False)
        num_pos = pos_anchors.size(0)
        return classification_loss/num_pos, smooth_l1_loss/num_pos

