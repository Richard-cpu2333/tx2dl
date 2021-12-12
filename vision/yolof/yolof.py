import torch
from torch import Tensor, nn
import torch.nn.functional as F
import torch.distributed as dist
import itertools
import copy
import logging
import numpy as np
from typing import Dict, List, Tuple

from vision.nn.mobilenetv3 import test


from .box_regression import YOLOFBox2BoxTransform
from vision.utils import box_utils

logger = logging.getLogger(__name__)


class YOLOF(nn.Module):
    def __init__(self, backbone, encoder, decoder, is_test=False, config=None, device=None):

        super(YOLOF, self).__init__()

        self.backbone = backbone
        self.encoder = encoder
        self.decoder = decoder
        self.is_test = is_test
        self.device = device

        if device:
            self.device = device
        else:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if is_test:
            self.config = config

    def forward(self, batched_inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        num_images = len(batched_inputs)
        features = self.backbone(batched_inputs)

        pred_logits, pred_anchors = self.decoder(self.encoder(features))

        if self.is_test:
            pred_logits = pred_logits[0]
            pred_anchor_deltas = pred_anchors[0]
            N = pred_logits.shape[0]
            NUM_CLASSES = pred_logits.shape[2]
            pred_anchor_deltas =  pred_anchor_deltas.view(-1, 4)
            pred_anchor_deltas = pred_anchor_deltas.reshape(N, -1, 4)
            pred_logits = F.softmax(pred_logits, dim=2)

            
            from vision.yolof.config import yolof_config as config

            anchors = config.anchors.view(-1, 4)
            box2box_transform = YOLOFBox2BoxTransform(weights=(1.0, 1.0, 1.0, 1.0))
            pred_boxes = box2box_transform.apply_deltas(pred_anchor_deltas, anchors)
            pred_anchors = box_utils.center_form_to_corner_form(pred_boxes)
            return pred_logits, pred_anchors

        return pred_logits, pred_anchors

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


def _xavier_init_(m: nn.Module):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight)