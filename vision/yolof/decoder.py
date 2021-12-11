import math
from typing import Tuple

import torch
import torch.nn as nn

######################################################################
### decoder with Depth-wise Conv
######################################################################

def conv_dw(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
        nn.BatchNorm2d(inp),
        nn.ReLU(inplace=True),

        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True),
        )



######################################################################
### decoder with IBN
######################################################################

# from yolof import get_activation, get_norm
from vision.nn.mobilenet_v2 import InvertedResidual
block = InvertedResidual
interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]



class Decoder(nn.Module):
    """
    Head Decoder for YOLOF.

    This module contains two types of components:
        - A classification head with two 3x3 convolutions and one
            classification 3x3 convolution
        - A regression head with four 3x3 convolutions, one regression 3x3
          convolution, and one implicit objectness 3x3 convolution
    """

    def __init__(self, num_classes=21):
        super(Decoder, self).__init__()
        # fmt: off
        self.in_channels = 512
        self.num_classes = num_classes
        self.num_anchors = 4
        self.cls_num_convs = 2
        self.reg_num_convs = 4
        self.prior_prob = 0.01
        # fmt: on

        self.INF = 1e8
        # init
        self._init_layers()
        self._init_weight()

    def _init_layers(self):
        cls_subnet = []
        bbox_subnet = []
        for i in range(self.cls_num_convs):
            cls_subnet.append(
                conv_dw(self.in_channels,
                        self.in_channels,
                        stride=1))
            cls_subnet.append(nn.BatchNorm2d(self.in_channels))
            cls_subnet.append(nn.LeakyReLU())
        for i in range(self.reg_num_convs):
            bbox_subnet.append(
                conv_dw(self.in_channels,
                        self.in_channels,
                        stride=1))
            bbox_subnet.append(nn.BatchNorm2d(self.in_channels))
            bbox_subnet.append(nn.LeakyReLU())
        self.cls_subnet = nn.Sequential(*cls_subnet)
        self.bbox_subnet = nn.Sequential(*bbox_subnet)
        self.cls_score = nn.Conv2d(self.in_channels,
                                   self.num_anchors * self.num_classes, # 4 for 
                                   kernel_size=3,
                                   stride=1,
                                   padding=1)
        self.bbox_pred = conv_dw(self.in_channels, self.num_anchors * 4, stride=1)
        self.object_pred = conv_dw(self.in_channels, self.num_anchors, stride=1)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            if isinstance(m, (nn.GroupNorm, nn.BatchNorm2d, nn.SyncBatchNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Use prior in model initialization to improve stability
        bias_value = -math.log((1 - self.prior_prob) / self.prior_prob)
        torch.nn.init.constant_(self.cls_score.bias, bias_value)

    def forward(self,
                feature: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        cls_score = self.cls_score(self.cls_subnet(feature))
        N, _, H, W = cls_score.shape
        cls_score = cls_score.view(N, -1, self.num_classes, H, W)

        reg_feat = self.bbox_subnet(feature)
        bbox_reg = self.bbox_pred(reg_feat)
        objectness = self.object_pred(reg_feat)

        # implicit objectness
        objectness = objectness.view(N, -1, 1, H, W)
        normalized_cls_score = cls_score + objectness - torch.log(
            1. + torch.clamp(cls_score.exp(), max=self.INF) + torch.clamp(
                objectness.exp(), max=self.INF))
        normalized_cls_score = normalized_cls_score.view(N, -1, H, W)

        normalized_cls_score = [permute_to_N_HWA_K(normalized_cls_score, self.num_classes)]
        bbox_reg = [permute_to_N_HWA_K(bbox_reg, 4)]

        return normalized_cls_score, bbox_reg

def permute_to_N_HWA_K(tensor, K: int):
    """
    Transpose/reshape a tensor from (N, (Ai x K), H, W) to (N, (HxWxAi), K)
    """
    assert tensor.dim() == 4, tensor.shape
    N, _, H, W = tensor.shape
    tensor = tensor.view(N, -1, K, H, W)
    tensor = tensor.permute(0, 3, 4, 1, 2)
    tensor = tensor.reshape(N, -1, K)  # Size=(N,HWA,K)
    return tensor