import torch
from torch import Tensor, nn
# import torch.distributed as dist
import itertools
import copy
import logging
import numpy as np
from typing import Dict, List, Tuple


from .box_regression import YOLOFBox2BoxTransform
from .uniform_matcher import UniformMatcher
from vision.utils import box_utils

__all__ = ["YOLOF", "MatchPrior"]

logger = logging.getLogger(__name__)


class YOLOF(nn.Module):
    def __init__(
            self,
            backbone,
            encoder,
            decoder,
    ):

        super().__init__()

        self.backbone = backbone
        self.encoder = encoder
        self.decoder = decoder

        # Anchors
        # self.generate_anchor = generate_anchors
        self.box2box_transform = YOLOFBox2BoxTransform(weights=(1_0, 1_0, 1_0, 1_0), add_ctr_clamp=True, ctr_clamp=32)
        self.anchor_matcher = UniformMatcher(4)


    # def forward(self, batched_inputs: Tuple[Dict[str, Tensor]]):
    def forward(self, batched_inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        num_images = len(batched_inputs)
        # print(num_images)
        # images = self.preprocess_image(batched_inputs)
        features = self.backbone(batched_inputs)

        # anchors_image = self.generate_anchors(config)
        # anchors = [copy.deepcopy(anchors_image) for _ in range(num_images)]
        pred_logits, pred_anchor_deltas = self.decoder(self.encoder(features))

        return pred_logits, pred_anchor_deltas

    def load(self, model):
        self.load_state_dict(torch.load(model, map_location=lambda storage, loc: storage))

    def save(self, model_path):
        torch.save(self.state_dict(), model_path)


    def init_from_base_net(self, model):
        self.backbone.load_state_dict(torch.load(model, map_location=lambda storage, loc: storage), strict=True)
        self.encoder.apply(_xavier_init_)
        self.decoder.apply(_xavier_init_)

    def init_from_pretrained_yolof(self, model):
        state_dict = torch.load(model, map_location=lambda storage, loc: storage)
        state_dict = {k: v for k, v in state_dict.items() if not (k.startswith("encoder") or k.startswith("decoder"))}
        model_dict = self.state_dict()
        model_dict.update(state_dict)
        self.load_state_dict(model_dict)

    def init(self):
        self.backbone.apply(_xavier_init_)
        self.encoder.apply(_xavier_init_)
        self.decoder.apply(_xavier_init_)





class MatchAnchor(object):
    def __init__(self, center_form_anchors, center_variance, size_variance):
        self.center_form_anchors = center_form_anchors
        # print(f"center_form_anchors'device: {self.center_form_anchors.device}")
        self.corner_form_anchors = box_utils.center_form_to_corner_form(center_form_anchors)
        self.center_variance = center_variance
        self.size_variance = size_variance

    def __call__(self, gt_boxes, gt_labels):
        if type(gt_boxes) is np.ndarray:
            gt_boxes = torch.from_numpy(gt_boxes)
            gt_boxes = box_utils.corner_form_to_center_form(gt_boxes)
        if type(gt_labels) is np.ndarray:
            gt_labels = torch.from_numpy(gt_labels)

        boxes, labels = box_utils.assign_anchors(gt_boxes, gt_labels, self.center_form_anchors)
        locations = box_utils.convert_boxes_to_locations(boxes, self.center_form_anchors, self.center_variance, self.size_variance)
        return locations, labels


def _xavier_init_(m: nn.Module):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight)