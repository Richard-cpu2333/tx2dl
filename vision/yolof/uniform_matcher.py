import numpy as np
import torch
from torch import nn
import itertools 
import math

from torch._C import device
from vision.utils import box_utils

def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


class UniformMatcher(nn.Module):

    def __init__(self, match_times: int = 4):
        super().__init__()
        self.match_times = match_times

    @torch.no_grad()
    # def forward(self, pred_boxes, anchors, targets):
    def forward(self, pred_boxes, anchors, gt_boxes, gt_labels):
        bs, num_queries = pred_boxes.shape[:2]

        # We flatten to compute the cost matrices in a batch
        # [batch_size * num_anchors, 4]
        out_bbox = pred_boxes.flatten(0, 1)
        anchors = anchors.flatten(0, 1)

        # Also concat the target boxes
        # tgt_bbox = torch.cat([v.gt_boxes.tensor for v in targets])
        # Compute the L1 cost between boxes
        # Note that we use anchors and predict boxes both
        cost_bbox = torch.cdist(
            box_xyxy_to_cxcywh(out_bbox), box_xyxy_to_cxcywh(gt_boxes), p=1)
        cost_bbox_anchors = torch.cdist(
            box_xyxy_to_cxcywh(anchors), box_xyxy_to_cxcywh(gt_boxes), p=1)

        # Final cost matrix
        C = cost_bbox
        C = C.view(bs, num_queries, -1).cpu()
        C1 = cost_bbox_anchors
        C1 = C1.view(bs, num_queries, -1).cpu()

        sizes = [len(v) for v in gt_boxes]
        all_indices_list = [[] for _ in range(bs)]
        # positive indices when matching predict boxes and gt boxes
        indices = [
            tuple(
                torch.topk(
                    c[i],
                    k=self.match_times,
                    dim=0,
                    largest=False)[1].numpy().tolist()
            )
            for i, c in enumerate(C.split(sizes, -1))
        ]
        # positive indices when matching anchor boxes and gt boxes
        indices1 = [
            tuple(
                torch.topk(
                    c[i],
                    k=self.match_times,
                    dim=0,
                    largest=False)[1].numpy().tolist())
            for i, c in enumerate(C1.split(sizes, -1))]

        # concat the indices according to image ids
        for img_id, (idx, idx1) in enumerate(zip(indices, indices1)):
            img_idx_i = [
                np.array(idx_ + idx1_)
                for (idx_, idx1_) in zip(idx, idx1)
            ]
            img_idx_j = [
                np.array(list(range(len(idx_))) + list(range(len(idx1_))))
                for (idx_, idx1_) in zip(idx, idx1)
            ]
            all_indices_list[img_id] = [*zip(img_idx_i, img_idx_j)]

        # re-organize the positive indices
        all_indices = []
        for img_id in range(bs):
            all_idx_i = []
            all_idx_j = []
            for idx_list in all_indices_list[img_id]:
                idx_i, idx_j = idx_list
                all_idx_i.append(idx_i)
                all_idx_j.append(idx_j)
            all_idx_i = np.hstack(all_idx_i)
            all_idx_j = np.hstack(all_idx_j)
            all_indices.append((all_idx_i, all_idx_j))
        return [
            (torch.as_tensor(i, dtype=torch.int64),
             torch.as_tensor(j, dtype=torch.int64))
            for i, j in all_indices
        ]

'''
############################################
## To generate 6 anchors without params
############################################

def generate_anchors(clamp=True) -> torch.Tensor:
    anchors = []
    scale = 300 / 32 ## cfg.image_size / cfg.shrinkage
    for j, i in itertools.product(range(10), repeat=2): ## cfg.feature_map_size = 10
        x_center = (i + 0.5) / scale
        y_center = (j + 0.5) / scale

        size = 60 ## cfg.box_sizes[0]
        h = w = size / 300 ## cfg.image_size
        anchors.append([
            x_center,
            y_center,
            w,
            h
        ])

        size = 111 
        h = w = size / 300 ## cfg.image_size
        anchors.append([
            x_center,
            y_center,
            w,
            h
        ])

        size = 162 
        h = w = size / 300 ## cfg.image_size
        anchors.append([
            x_center,
            y_center,
            w,
            h
        ])

        size = 213
        h = w = size / 300 ## cfg.image_size
        anchors.append([
            x_center,
            y_center,
            w,
            h
        ])

        size = 264 
        h = w = size / 300 ## cfg.image_size
        anchors.append([
            x_center,
            y_center,
            w,
            h
        ])

        size = 315 ## math.sqrt(cfg.box_sizes[1] * cfg.box_sizes[0])
        h = w = size / 300 ## cfg.image_size
        anchors.append([
            x_center,
            y_center,
            w,
            h
        ])

    anchors = torch.tensor(anchors)
    if clamp:
        torch.clamp(anchors, 0.0, 1.0, out=anchors)
    return anchors
'''

########################
## To generate 4 anchors
########################
def generate_anchors(clamp=True) -> torch.Tensor:
    anchors = []
    scale = 300 / 32 ## cfg.image_size / cfg.shrinkage
    for j, i in itertools.product(range(10), repeat=2): ## cfg.feature_map_size = 10
        x_center = (i + 0.5) / scale
        y_center = (j + 0.5) / scale

        # small sized square box 32
        size = 111 ## cfg.box_sizes[0]
        h = w = size / 300 ## cfg.image_size
        anchors.append([
            x_center,
            y_center,
            w,
            h
        ])

        # small sized square box 64
        size = 162 ## cfg.box_sizes[1] * cfg.box_sizes[0]
        h = w = size / 300 ## cfg.image_size
        anchors.append([
            x_center,
            y_center,
            w,
            h
        ])

        # small sized square box 100
        size = 213 ## cfg.box_sizes[1] * cfg.box_sizes[0]
        h = w = size / 300 ## cfg.image_size
        anchors.append([
            x_center,
            y_center,
            w,
            h
        ])

        # small sized square box 300
        size = 264 ## cfg.box_sizes[1] * cfg.box_sizes[0]
        h = w = size / 300 ## cfg.image_size
        anchors.append([
            x_center,
            y_center,
            w,
            h
        ])

    anchors = torch.tensor(anchors)
    if clamp:
        torch.clamp(anchors, 0.0, 1.0, out=anchors)
    return anchors
    

class MatchAnchor(object):
    def __init__(self, center_form_anchors):
        self.center_form_anchors = center_form_anchors
        self.corner_form_anchors = box_utils.center_form_to_corner_form(center_form_anchors)

    def __call__(self, gt_boxes, gt_labels):
        batch_size = len(gt_boxes)
        masked_anchor_boxes = []
        masked_anchor_labels = []
        masked_pred_boxes = []
        masked_pred_labels = []

        if type(gt_boxes[0]) is np.ndarray:
            if type(gt_labels[0]) is np.ndarray:
                if(self.center_form_anchors.device.type == "cpu"):
                    for i in range(batch_size):
                        gt_box = torch.from_numpy(gt_boxes[i])
                        gt_box = box_utils.corner_form_to_center_form(gt_box)
                        gt_label = torch.from_numpy(gt_labels[i])
                        boxes, labels = box_utils.assign_anchors(gt_box, gt_label, self.center_form_anchors)
                        pred_anchor_deltas = box_utils.get_deltas(self.corner_form_anchors, boxes)
                        masked_anchor_boxes.append(torch.unsqueeze(pred_anchor_deltas, 0))
                        masked_anchor_labels.append(torch.unsqueeze(labels, 0))
                    boxes = torch.cat(masked_anchor_boxes, dim=0)
                    labels = torch.cat(masked_anchor_labels, dim=0)
                    boxes = boxes.to("cuda")
                    labels = labels.to("cuda")
                    # print(boxes.shape, labels.shape)
                else:
                    self.center_form_anchors = self.center_form_anchors.to("cpu")
                    for i in range(batch_size):
                        gt_box = torch.from_numpy(gt_boxes[i])
                        gt_box = box_utils.corner_form_to_center_form(gt_box)
                        gt_label = torch.from_numpy(gt_labels[i])
                        boxes, labels = box_utils.assign_priors(gt_box, gt_label, self.center_form_anchors[i], 0.5)

                        masked_pred_boxes.append(torch.unsqueeze(boxes, 0))
                        masked_pred_labels.append(torch.unsqueeze(labels, 0))
                    boxes = torch.cat(masked_pred_boxes, dim=0)
                    labels = torch.cat(masked_pred_labels, dim=0)
                    boxes = boxes.to("cuda")
                    labels = labels.to("cuda")
                    # print(boxes.shape, labels.shape)
            else:
                print(f"FUCK YOU!")
                exit(1)
        else:
            print("FUCK YOU!")
            exit(1)
                
        # locations = box_utils.convert_boxes_to_locations(boxes, self.center_form_anchors, self.center_variance, self.size_variance)
        # return locations, labels
        return boxes, labels