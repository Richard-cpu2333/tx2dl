## pip install ptflops
## https://github.com/sovrasov/flops-counter.pytorch
## https://github.com/albanie/convnet-burden
## https://github.com/Swall0w/torchstat


import torchvision.models as models
import torch
from ptflops import get_model_complexity_info
from vision.yolof.mobiledet_yolof import create_mobilenetv2_yolof_lite, create_mobilenetv3_small_yolof_lite, create_efficient_yolof, create_mobiledet_yolof, efficientnet_b0
from vision.ssd.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite
from vision.ssd.mobilenetv3_ssd_lite import create_mobilenetv3_small_ssd_lite

with torch.cuda.device(0):
    # net = create_mobilenetv2_yolof_lite(4)
    # net = create_mobilenetv3_small_yolof_lite(4)
    net = create_efficient_yolof(4)
    # net = create_mobiledet_yolof(4)
    # net = create_mobilenetv2_ssd_lite(4)
    # net = create_mobilenetv3_small_ssd_lite(4)
    # net = create_mobilenetv3_small_ssd_lite(4)

    macs, params = get_model_complexity_info(net, (3, 300, 300), as_strings=True,
                                           print_per_layer_stat=True, verbose=True)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))