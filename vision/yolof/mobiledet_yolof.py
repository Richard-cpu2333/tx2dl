import torch
# from torch.nn import Conv2d, Sequential, ModuleList, ReLU
from .mobiledet_gpu import MobileDetGPU
from .encoder import DilatedEncoder
from .decoder import Decoder
from .yolof import YOLOF

# from .predictor import Predictor
# from config import yolof_config as config


def create_mobiledet_yolof(num_classes = 20, is_test=False):
    backbone = MobileDetGPU().model
    encoder = DilatedEncoder()
    decoder = Decoder(num_classes)

    return YOLOF(backbone, encoder, decoder)
    # return YOLOF(backbone, encoder, decoder, num_classes,pos_ignore_thresh=0.15,neg_ignore_thresh=0.7,focal_loss_alpha=0.25,focal_loss_gamma=2.0,box_reg_loss_type="giou",test_score_thresh=0.05,test_topk_candidates=1000,test_nms_thresh=0.6,max_detections_per_image=100,vis_period=0,input_format="BGR")

# def create_mobiledet_yolof_predictor():
#     predictor = Predictor()

#     return predictor