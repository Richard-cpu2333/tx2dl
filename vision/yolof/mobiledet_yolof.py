# from torch.nn import Conv2d, Sequential, ModuleList, ReLU
from vision.nn.mobiledet_gpu import MobileDetGPU
from vision.nn.mobilenet import MobileNetV1
from vision.nn.mobilenet_v2 import MobileNetV2
from vision.nn.mobilenetv3 import MobileNetV3_Large, MobileNetV3_Small
from torchvision.models import efficientnet_b0, efficientnet_b1, efficientnet_b2, efficientnet_b3, efficientnet_b4, efficientnet_b5, efficientnet_b6, efficientnet_b7

from .encoder import DilatedEncoder
from .decoder import Decoder
from .yolof import YOLOF
from .predictor import Predictor
from .config import yolof_config as config


def create_mobiledet_yolof(num_classes, is_test=False):
    backbone = MobileDetGPU().model
    encoder = DilatedEncoder(384)
    decoder = Decoder(num_classes)

    return YOLOF(backbone, encoder, decoder)
    # return YOLOF(backbone, encoder, decoder, num_classes,pos_ignore_thresh=0.15,neg_ignore_thresh=0.7,focal_loss_alpha=0.25,focal_loss_gamma=2.0,box_reg_loss_type="giou",test_score_thresh=0.05,test_topk_candidates=1000,test_nms_thresh=0.6,max_detections_per_image=100,vis_period=0,input_format="BGR")


def create_mobilenetv1_yolof(num_classes, is_test=False):
    backbone = MobileNetV1().model
    encoder = DilatedEncoder(1024)
    decoder = Decoder(num_classes)

    return YOLOF(backbone, encoder, decoder)

def create_mobilenetv1_yolof_lite(num_classes, is_test=False):
    backbone = MobileNetV1().model
    encoder = DilatedEncoder(1024)
    decoder = Decoder(num_classes)

    return YOLOF(backbone, encoder, decoder)


def create_mobilenetv2_yolof_lite(num_classes, width_mult=1.0, use_batch_norm=True, onnx_compatible=False, is_test=False):
    backbone = MobileNetV2(width_mult=width_mult, use_batch_norm=use_batch_norm,
                           onnx_compatible=onnx_compatible).features
    encoder = DilatedEncoder(1280)
    decoder = Decoder(num_classes)

    return YOLOF(backbone, encoder, decoder)

def create_mobilenetv3_large_yolof_lite(num_classes, is_test=False):
    backbone = MobileNetV3_Large().features
    encoder = DilatedEncoder(960)
    decoder = Decoder(num_classes)

    return YOLOF(backbone, encoder, decoder)

def create_mobilenetv3_small_yolof_lite(num_classes, is_test=False):
    backbone = MobileNetV3_Small().features
    encoder = DilatedEncoder(576)
    decoder = Decoder(num_classes)

    return YOLOF(backbone, encoder, decoder)

def create_efficient_yolof(num_classes, is_test=False):
    backbone = efficientnet_b0().features
    encoder = DilatedEncoder(1280)
    decoder = Decoder(num_classes)

    return YOLOF(backbone, encoder, decoder)

def create_mobiledet_yolof_predictor(net, candidate_size=200, nms_method=None, sigma=0.5, device=None):
    predictor = Predictor(net, config.image_size, config.image_mean,
                          config.image_std,
                          nms_method=nms_method,
                          iou_threshold=config.iou_threshold,
                          candidate_size=candidate_size,
                          sigma=sigma,
                          device=device)

    return predictor