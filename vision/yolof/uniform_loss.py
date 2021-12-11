from os import posix_fallocate
import torch
from torch.functional import Tensor
import torch.nn as nn
import torch.nn.functional as F
from fvcore.nn import sigmoid_focal_loss_jit, giou_loss
import torch.distributed as dist
from . import comm
from vision.yolof.box_regression import YOLOFBox2BoxTransform
import math
from torchvision.ops.boxes import box_iou


def cat(tensors, dim: int = 0):
    """
    Efficient version of torch.cat that avoids a copy if there is only a single element in a list
    """
    assert isinstance(tensors, (list, tuple))
    if len(tensors) == 1:
        return tensors[0]
    return torch.cat(tensors, dim)


num_classes = 4

def criterion(indices, gt_boxes, gt_labels, anchors, pred_class_logits, pred_anchor_deltas):
    # pred_class_logits = torch.cat(pred_class_logits, dim=1).view(-1, num_classes)
    # pred_anchor_deltas = torch.cat(pred_anchor_deltas, dim=1).view(-1, 4)
    pred_class_logits = pred_class_logits.view(-1, num_classes)
    pred_anchor_deltas = pred_anchor_deltas.view(-1, 4)
    N = len(anchors)
    gt_boxes = [gt_boxes[i] for i in range(N)]
    
    # list[Tensor(R, 4)], one for each image
    all_anchors = torch.cat([anchors[i][None] for i in range(N)]).view(-1, 4)
    # Boxes(Tensor(N*R, 4))
    trans = YOLOFBox2BoxTransform()
    predicted_boxes = trans.apply_deltas(pred_anchor_deltas, all_anchors)
    predicted_boxes = predicted_boxes.reshape(N, -1, 4).to("cpu")

        # We obtain positive anchors by choosing gt boxes' k nearest anchors
        # and leave the rest to be negative anchors. However, there may
        # exist negative anchors that have similar distances with the chosen
        # positives. These negatives may cause ambiguity for model training
        # if we just set them as negatives. Given that we want the model's
        # predict boxes on negative anchors to have low IoU with gt boxes,
        # we set a threshold on the IoU between predicted boxes and gt boxes
        # instead of the IoU between anchor boxes and gt boxes.
    anchors = [anchors_i.squeeze(0) for anchors_i in anchors]

    ious = []
    pos_ious = []
    for i in range(N):
        src_idx, tgt_idx = indices[i]
        iou = box_iou(predicted_boxes[i, ...], gt_boxes[i])
        if iou.numel() == 0:
            max_iou = iou.new_full((iou.size(0),), 0)
        else:
            max_iou = iou.max(dim=1)[0]
        a_iou = box_iou(anchors[i], gt_boxes[i])
        if a_iou.numel() == 0:
            pos_iou = a_iou.new_full((0,), 0)
        else:
            pos_iou = a_iou[src_idx, tgt_idx]
        ious.append(max_iou)
        pos_ious.append(pos_iou)
    ious = torch.cat(ious)

    import vision.yolof.config.yolof_config as config
    ignore_idx = ious > config.MODEL_YOLOF_NEG_IGNORE_THRESHOLD ## self.neg_ignore_thresh
    pos_ious = torch.cat(pos_ious)
    pos_ignore_idx = pos_ious < config.MODEL_YOLOF_POS_IGNORE_THRESHOLD ## self.pos_ignore_thresh

    src_idx = torch.cat([src + idx * anchors[0].shape[0] for idx, (src, _) in enumerate(indices)])
    gt_classes = torch.full(pred_class_logits.shape[:1], num_classes, dtype=torch.int64, device=pred_class_logits.device)
    gt_classes[ignore_idx] = -1
    target_classes_o = torch.cat([t[J] for t, (_, J) in zip(gt_labels, indices)]).to(pred_class_logits.device)
    target_classes_o[pos_ignore_idx] = -1
    gt_classes[src_idx] = target_classes_o

    valid_idxs = gt_classes >= 0
    foreground_idxs = (gt_classes >= 0) & (gt_classes != num_classes)
    num_foreground = foreground_idxs.sum()

    gt_classes_target = torch.zeros_like(pred_class_logits)
    gt_classes_target[foreground_idxs, gt_classes[foreground_idxs]] = 1


    # cls loss
    loss_cls = sigmoid_focal_loss_jit(
        pred_class_logits[valid_idxs],
        gt_classes_target[valid_idxs],
        alpha=config.MODEL_YOLOF_LOSSES_FOCAL_LOSS_ALPHA, ## self.focal_loss_alpha
        gamma=config.MODEL_YOLOF_LOSSES_FOCAL_LOSS_GAMMA, ## self.focal_loss_gamma
        reduction="sum",
    )
    # reg loss

    target_boxes = torch.cat(
        [t[i] for t, (_, i) in zip(gt_boxes, indices)],
        dim=0)
    target_boxes = target_boxes[~pos_ignore_idx]
    matched_predicted_boxes = predicted_boxes.reshape(-1, 4)[
        src_idx[~pos_ignore_idx]]
    loss_box_reg = giou_loss(
        matched_predicted_boxes, target_boxes, reduction="sum")

    return loss_cls / max(1, num_foreground), loss_box_reg / max(1, num_foreground)


'''
class UniformLoss(nn.Module):
    def __init__(self, anchors, neg_pos_ratio=3, neg_ignore_thresh=0.7, pos_ignore_thresh=0.3):
        super(UniformLoss, self).__init__()
        self.neg_pos_ratio = neg_pos_ratio
        self.neg_ignore_thresh = neg_ignore_thresh
        self.pos_ignore_thresh = pos_ignore_thresh
        self.anchors = anchors

    def forward(self, pos_labels, pos_anchors, ignore_idx, pred_anchor_deltas, pred_class_logits):
        N = pred_class_logits.shape[0]
        NUM_CLASSES = pred_class_logits.shape[2]
        # gt_classes = torch.full(pred_class_logits.shape[:1],
        #                         self.num_classes,
        #                         dtype=torch.int64,
        #                         device=pred_class_logits.device)
        # gt_classes[ignore_idx] = -1


        # pred_class_logits = pred_class_logits.view(-1, NUM_CLASSES)
        trans = YOLOFBox2BoxTransform()
        predicted_boxes = trans.apply_deltas(pred_anchor_deltas, self.anchors)
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
        # print(f"$$$$$$$$$${predicted_boxes.shape}$$$$$$$$$$$$$44")
        pos_anchors = pos_anchors[pos_mask, :].reshape(-1, 4)
        # print(f"$$$$$$$$$${pos_anchors.shape}$$$$$$$$$$$$$44")
        smooth_l1_loss = F.smooth_l1_loss(predicted_boxes, pos_anchors, size_average=False)
        num_pos = pos_anchors.size(0)
    
        return classification_loss/num_pos, smooth_l1_loss/num_pos

'''