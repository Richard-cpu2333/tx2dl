from torchstat import stat
import torch
from vision.yolof.mobiledet_yolof import create_mobiledet_yolof, create_efficientnet_yolof, create_mobilenetv2_yolof_lite,create_mobilenetv3_large_yolof_lite,create_mobilenetv3_small_yolof_lite,create_mobilenetv1_yolof,create_mobiledet_yolof_predictor

num_classes = 4

net =  create_efficientnet_yolof(num_classes)
# net = mobiledet_yolof.create_mobiledet_yolof(num_classes)
# net = create_mobilenetv3_large_yolof_lite(num_classes)
# net = create_mobilenetv2_yolof_lite(num_classes)

x = torch.randn([1,3,300,300])
stat(net, (3,300,300))