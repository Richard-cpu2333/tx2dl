import torch
from torch.functional import Tensor
import torch.nn as nn
import torch.nn.functional as F
from fvcore.nn import sigmoid_focal_loss_jit, giou_loss
import torch.distributed as dist
from vision.utils.box_utils import iou_of
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
    def __init__(self, batch_anchors, device):
        super(UniformLoss, self).__init__()
        '''
        ## pred_class_logits.shape: [32, 400, 21] 
        ## pred_anchor_deltas.shape: [32, 400, 4] 
        ## labels.shape: []

        '''

        self.anchors = batch_anchors
        self.device = device


    def forward(self, confidence, pred_anchor_deltas, labels, gt_boxes):
        N = confidence.shape[0]
        NUM_CLASSES = confidence.shape[2]

        # print(f"labels.shape:{labels.shape}")  ## torch.Size([32, 400])
        # print(f"boxes.shape:{gt_boxes.shape}")  ## torch.Size([32, 400, 4])
        # pred_class_logits = pred_class_logits.view(-1, NUM_CLASSES)
        pred_anchor_deltas = pred_anchor_deltas.view(-1, 4)
        

        all_anchors = [self.anchors for _ in range(N)]
        all_anchors = cat(all_anchors).to(self.device)  ## torch.Size([12800, 4])
        
        box2box_transform = YOLOFBox2BoxTransform(weights=(1.0, 1.0, 1.0, 1.0))
        pred_boxes = box2box_transform.apply_deltas(
            pred_anchor_deltas, all_anchors)
        pred_boxes = pred_boxes.reshape(N, -1, 4)

        # pos_mask = labels > 0
        # pred_boxes = pred_boxes[pos_mask, :].reshape(-1, 4)

        with torch.no_grad():
            # derived from cross_entropy=sum(log(p))
            loss = -F.log_softmax(confidence, dim=2)[:, :, 0]
            pos_mask = labels > 0

            num_pos = pos_mask.long().sum(dim=1, keepdim=True)
            num_neg = num_pos * 3
            loss[pos_mask] = -math.inf
            _, indexes = loss.sort(dim=1, descending=True)
            _, orders = indexes.sort(dim=1)
            neg_mask = orders < num_neg
            mask = pos_mask | neg_mask

        confidence = confidence[mask, :]

        #loss_cls = sigmoid_focal_loss_jit(
        #     pred_class_logits[mask],
        #     labels[mask],
        #     alpha=2.0,
        #     gamma=0.25,
        #     reduction="sum",
        # )
        # print(f"confidence info:{confidence.dtype}") ## torch.size([12800,21]), torch.float32
        # print(f"labels info:{labels.dtype}") ## torch.Size([32, 400]), torch.int64
        classification_loss = F.cross_entropy(confidence.reshape(-1, NUM_CLASSES), labels[mask], size_average=False)
        pos_mask = labels > 0
        predicted_locations = pred_boxes[pos_mask, :].reshape(-1, 4)
        gt_locations = gt_boxes[pos_mask, :].reshape(-1, 4)
        smooth_l1_loss = F.smooth_l1_loss(predicted_locations, gt_locations, size_average=False)
        num_pos = gt_locations.size(0)
        return classification_loss/num_pos, smooth_l1_loss/num_pos

