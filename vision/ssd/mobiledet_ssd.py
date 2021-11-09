import torch
import torch.nn as nn
from ..nn.mobiledet_gpu import MobileDetGPU

from .ssd import SSD
from .predictor import Predictor
from .config import mobilenetv1_ssd_config as config

class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, use_batch_norm=True, onnx_compatible=False):
        super(InvertedResidual, self).__init__()
        ReLU = nn.ReLU if onnx_compatible else nn.ReLU6

        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expand_ratio == 1:
            if use_batch_norm:
                self.conv = nn.Sequential(
                    # dw
                    nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                    nn.BatchNorm2d(hidden_dim),
                    ReLU(inplace=True),
                    # pw-linear
                    nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                    nn.BatchNorm2d(oup),
                )
            else:
                self.conv = nn.Sequential(
                    # dw
                    nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                    ReLU(inplace=True),
                    # pw-linear
                    nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                )
        else:
            if use_batch_norm:
                self.conv = nn.Sequential(
                    # pw
                    nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                    nn.BatchNorm2d(hidden_dim),
                    ReLU(inplace=True),
                    # dw
                    nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                    nn.BatchNorm2d(hidden_dim),
                    ReLU(inplace=True),
                    # pw-linear
                    nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                    nn.BatchNorm2d(oup),
                )
            else:
                self.conv = nn.Sequential(
                    # pw
                    nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                    ReLU(inplace=True),
                    # dw
                    nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                    ReLU(inplace=True),
                    # pw-linear
                    nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

def SeperableConv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, onnx_compatible=False):
    """Replace Conv2d with a depthwise Conv2d and Pointwise Conv2d.
    """
    ReLU = nn.ReLU if onnx_compatible else nn.ReLU6
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size,
               groups=in_channels, stride=stride, padding=padding),
        nn.BatchNorm2d(in_channels),
        ReLU(),
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1),
    )

def create_mobiledet_ssd(num_classes, is_test=False):
    base_net = MobileDetGPU().model

    source_layer_indexes = [
        20,
        22,
    ]

    extras = nn.ModuleList([
        InvertedResidual(128, 128, stride=2, expand_ratio=0.2),
        InvertedResidual(128, 128, stride=2, expand_ratio=0.25),
        InvertedResidual(128, 128, stride=2, expand_ratio=0.5),
        InvertedResidual(128, 64, stride=2, expand_ratio=0.25)
    ])

    regression_headers = nn.ModuleList([
        SeperableConv2d(in_channels=128, out_channels=6 * 4,
                        kernel_size=3, padding=1, onnx_compatible=False),
        SeperableConv2d(in_channels=128, out_channels=6 * 4, kernel_size=3, padding=1, onnx_compatible=False),
        SeperableConv2d(in_channels=128, out_channels=6 * 4, kernel_size=3, padding=1, onnx_compatible=False),
        SeperableConv2d(in_channels=128, out_channels=6 * 4, kernel_size=3, padding=1, onnx_compatible=False),
        SeperableConv2d(in_channels=128, out_channels=6 * 4, kernel_size=3, padding=1, onnx_compatible=False),
        nn.Conv2d(in_channels=64, out_channels=6 * 4, kernel_size=1),
    ])

    classification_headers = nn.ModuleList([
        SeperableConv2d(in_channels=128, out_channels=6 * num_classes, kernel_size=3, padding=1),
        SeperableConv2d(in_channels=128, out_channels=6 * num_classes, kernel_size=3, padding=1),
        SeperableConv2d(in_channels=128, out_channels=6 * num_classes, kernel_size=3, padding=1),
        SeperableConv2d(in_channels=128, out_channels=6 * num_classes, kernel_size=3, padding=1),
        SeperableConv2d(in_channels=128, out_channels=6 * num_classes, kernel_size=3, padding=1),
        nn.Conv2d(in_channels=64, out_channels=6 * num_classes, kernel_size=1),
    ])

    return SSD(num_classes, base_net, source_layer_indexes,
               extras, classification_headers, regression_headers, is_test=is_test, config=config)


def create_mobiledet_ssd_predictor(net, candidate_size=200, nms_method=None, sigma=0.5, device=None):
    predictor = Predictor(net, config.image_size, config.image_mean,
                          config.image_std,
                          nms_method=nms_method,
                          iou_threshold=config.iou_threshold,
                          candidate_size=candidate_size,
                          sigma=sigma,
                          device=device)
    return predictor
