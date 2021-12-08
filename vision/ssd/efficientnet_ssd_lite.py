import torch
from torch.nn import Conv2d, Sequential, ModuleList, BatchNorm2d
from torch import nn
from ..nn.mobilenet_v2 import InvertedResidual
from torchvision.models import efficientnet_b0, efficientnet_b1, efficientnet_b2, efficientnet_b3, efficientnet_b4, efficientnet_b5, efficientnet_b6, efficientnet_b7

from .ssd import SSD, GraphPath
from .predictor import Predictor
from .config import mobilenetv1_ssd_config as config


def SeperableConv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, onnx_compatible=False):
    """Replace Conv2d with a depthwise Conv2d and Pointwise Conv2d.
    """
    ReLU = nn.ReLU if onnx_compatible else nn.ReLU6
    return Sequential(
        Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size,
               groups=in_channels, stride=stride, padding=padding),
        BatchNorm2d(in_channels),
        ReLU(),
        Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1),
    )


def create_efficientnet_ssd_lite_b0(num_classes, width_mult=1.0, use_batch_norm=True, onnx_compatible=False, is_test=False):
    base_net = efficientnet_b0().features

    source_layer_indexes = [
        5,
        19,
    ]
    extras = ModuleList([
        InvertedResidual(1280, 512, stride=2, expand_ratio=0.2),
        InvertedResidual(512, 256, stride=2, expand_ratio=0.25),
        InvertedResidual(256, 256, stride=2, expand_ratio=0.5),
        InvertedResidual(256, 64, stride=2, expand_ratio=0.25)
    ])

    regression_headers = ModuleList([
        SeperableConv2d(in_channels=80, out_channels=6 * 4,
                        kernel_size=3, padding=1, onnx_compatible=False),
        SeperableConv2d(in_channels=1280, out_channels=6 * 4, kernel_size=3, padding=1, onnx_compatible=False),
        SeperableConv2d(in_channels=512, out_channels=6 * 4, kernel_size=3, padding=1, onnx_compatible=False),
        SeperableConv2d(in_channels=256, out_channels=6 * 4, kernel_size=3, padding=1, onnx_compatible=False),
        SeperableConv2d(in_channels=256, out_channels=6 * 4, kernel_size=3, padding=1, onnx_compatible=False),
        Conv2d(in_channels=64, out_channels=6 * 4, kernel_size=1),
    ])

    classification_headers = ModuleList([
        SeperableConv2d(in_channels=80, out_channels=6 * num_classes, kernel_size=3, padding=1),
        SeperableConv2d(in_channels=1280, out_channels=6 * num_classes, kernel_size=3, padding=1),
        SeperableConv2d(in_channels=512, out_channels=6 * num_classes, kernel_size=3, padding=1),
        SeperableConv2d(in_channels=256, out_channels=6 * num_classes, kernel_size=3, padding=1),
        SeperableConv2d(in_channels=256, out_channels=6 * num_classes, kernel_size=3, padding=1),
        Conv2d(in_channels=64, out_channels=6 * num_classes, kernel_size=1),
    ])

    return SSD(num_classes, base_net, source_layer_indexes,
               extras, classification_headers, regression_headers, is_test=is_test, config=config)

def create_efficientnet_ssd_lite_b1(num_classes, width_mult=1.0, use_batch_norm=True, onnx_compatible=False, is_test=False):
    base_net = efficientnet_b1().features

    source_layer_indexes = [
        5, ## b0: 5 b1: 5 
        7, ## b0: 19 b2: 7
    ]
    extras = ModuleList([
        InvertedResidual(1280, 512, stride=2, expand_ratio=0.2),
        InvertedResidual(512, 256, stride=2, expand_ratio=0.25),
        InvertedResidual(256, 256, stride=2, expand_ratio=0.5),
        InvertedResidual(256, 64, stride=2, expand_ratio=0.25)
    ])

    regression_headers = ModuleList([
        SeperableConv2d(in_channels=80, out_channels=6 * 4,
                        kernel_size=3, padding=1, onnx_compatible=False),
        SeperableConv2d(in_channels=192, out_channels=6 * 4, kernel_size=3, padding=1, onnx_compatible=False),
        SeperableConv2d(in_channels=512, out_channels=6 * 4, kernel_size=3, padding=1, onnx_compatible=False),
        SeperableConv2d(in_channels=256, out_channels=6 * 4, kernel_size=3, padding=1, onnx_compatible=False),
        SeperableConv2d(in_channels=256, out_channels=6 * 4, kernel_size=3, padding=1, onnx_compatible=False),
        Conv2d(in_channels=64, out_channels=6 * 4, kernel_size=1),
    ])

    classification_headers = ModuleList([
        SeperableConv2d(in_channels=80, out_channels=6 * num_classes, kernel_size=3, padding=1),
        SeperableConv2d(in_channels=192, out_channels=6 * num_classes, kernel_size=3, padding=1),
        SeperableConv2d(in_channels=512, out_channels=6 * num_classes, kernel_size=3, padding=1),
        SeperableConv2d(in_channels=256, out_channels=6 * num_classes, kernel_size=3, padding=1),
        SeperableConv2d(in_channels=256, out_channels=6 * num_classes, kernel_size=3, padding=1),
        Conv2d(in_channels=64, out_channels=6 * num_classes, kernel_size=1),
    ])

    return SSD(num_classes, base_net, source_layer_indexes,
               extras, classification_headers, regression_headers, is_test=is_test, config=config)

def create_efficientnet_ssd_lite_b2(num_classes, width_mult=1.0, use_batch_norm=True, onnx_compatible=False, is_test=False):
    base_net = efficientnet_b2().features

    source_layer_indexes = [
        5, ## b0: 5 b1: 5 
        9, ## b0: 19 b2: 7
    ]
    extras = ModuleList([
        InvertedResidual(1408, 512, stride=2, expand_ratio=0.2),
        InvertedResidual(512, 256, stride=2, expand_ratio=0.25),
        InvertedResidual(256, 256, stride=2, expand_ratio=0.5),
        InvertedResidual(256, 64, stride=2, expand_ratio=0.25)
    ])

    regression_headers = ModuleList([
        SeperableConv2d(in_channels=88, out_channels=6 * 4,
                        kernel_size=3, padding=1, onnx_compatible=False),
        SeperableConv2d(in_channels=1408, out_channels=6 * 4, kernel_size=3, padding=1, onnx_compatible=False),
        SeperableConv2d(in_channels=512, out_channels=6 * 4, kernel_size=3, padding=1, onnx_compatible=False),
        SeperableConv2d(in_channels=256, out_channels=6 * 4, kernel_size=3, padding=1, onnx_compatible=False),
        SeperableConv2d(in_channels=256, out_channels=6 * 4, kernel_size=3, padding=1, onnx_compatible=False),
        Conv2d(in_channels=64, out_channels=6 * 4, kernel_size=1),
    ])

    classification_headers = ModuleList([
        SeperableConv2d(in_channels=88, out_channels=6 * num_classes, kernel_size=3, padding=1),
        SeperableConv2d(in_channels=1408, out_channels=6 * num_classes, kernel_size=3, padding=1),
        SeperableConv2d(in_channels=512, out_channels=6 * num_classes, kernel_size=3, padding=1),
        SeperableConv2d(in_channels=256, out_channels=6 * num_classes, kernel_size=3, padding=1),
        SeperableConv2d(in_channels=256, out_channels=6 * num_classes, kernel_size=3, padding=1),
        Conv2d(in_channels=64, out_channels=6 * num_classes, kernel_size=1),
    ])

    return SSD(num_classes, base_net, source_layer_indexes,
               extras, classification_headers, regression_headers, is_test=is_test, config=config)

def create_efficientnet_ssd_lite_b3(num_classes, width_mult=1.0, use_batch_norm=True, onnx_compatible=False, is_test=False):
    base_net = efficientnet_b3().features

    source_layer_indexes = [
        5, ## b0: 5 b1: 5 
        9, ## b0: 19 b2: 7
    ]
    extras = ModuleList([
        InvertedResidual(1536, 512, stride=2, expand_ratio=0.2),
        InvertedResidual(512, 256, stride=2, expand_ratio=0.25),
        InvertedResidual(256, 256, stride=2, expand_ratio=0.5),
        InvertedResidual(256, 64, stride=2, expand_ratio=0.25)
    ])

    regression_headers = ModuleList([
        SeperableConv2d(in_channels=96, out_channels=6 * 4,
                        kernel_size=3, padding=1, onnx_compatible=False),
        SeperableConv2d(in_channels=1536, out_channels=6 * 4, kernel_size=3, padding=1, onnx_compatible=False),
        SeperableConv2d(in_channels=512, out_channels=6 * 4, kernel_size=3, padding=1, onnx_compatible=False),
        SeperableConv2d(in_channels=256, out_channels=6 * 4, kernel_size=3, padding=1, onnx_compatible=False),
        SeperableConv2d(in_channels=256, out_channels=6 * 4, kernel_size=3, padding=1, onnx_compatible=False),
        Conv2d(in_channels=64, out_channels=6 * 4, kernel_size=1),
    ])

    classification_headers = ModuleList([
        SeperableConv2d(in_channels=96, out_channels=6 * num_classes, kernel_size=3, padding=1),
        SeperableConv2d(in_channels=1536, out_channels=6 * num_classes, kernel_size=3, padding=1),
        SeperableConv2d(in_channels=512, out_channels=6 * num_classes, kernel_size=3, padding=1),
        SeperableConv2d(in_channels=256, out_channels=6 * num_classes, kernel_size=3, padding=1),
        SeperableConv2d(in_channels=256, out_channels=6 * num_classes, kernel_size=3, padding=1),
        Conv2d(in_channels=64, out_channels=6 * num_classes, kernel_size=1),
    ])

    return SSD(num_classes, base_net, source_layer_indexes,
               extras, classification_headers, regression_headers, is_test=is_test, config=config)

def create_efficientnet_ssd_lite_predictor(net, candidate_size=200, nms_method=None, sigma=0.5, device=None):
    predictor = Predictor(net, config.image_size, config.image_mean,
                          config.image_std,
                          nms_method=nms_method,
                          iou_threshold=config.iou_threshold,
                          candidate_size=candidate_size,
                          sigma=sigma,
                          device=device)
    return predictor
