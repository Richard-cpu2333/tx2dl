

from torchstat import stat
import torchvision.models as models
from vision.ssd.efficientnet_ssd_lite import create_efficientnet_ssd_lite
from vision.ssd.mobiledet_ssd_lite import create_mobiledet_ssd_lite
from vision.ssd.squeezenet_ssd_lite import create_squeezenet_ssd_lite
from vision.yolof.mobiledet_yolof import create_mobilenetv2_yolof_lite, create_mobilenetv3_small_yolof_lite, create_efficientnet_yolof, create_mobiledet_yolof
from vision.ssd.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite
from vision.ssd.mobilenetv3_ssd_lite import create_mobilenetv3_small_ssd_lite, create_mobilenetv3_large_ssd_lite

# model = create_mobilenetv3_small_ssd_lite(4)
# model = create_mobilenetv3_large_ssd_lite(4)
# model = create_mobilenetv2_ssd_lite(4)
# model = create_mobilenetv2_yolof_lite(4)
# model = create_mobiledet_ssd_lite(4)
model = create_efficientnet_ssd_lite(4)
# model = create_squeezenet_ssd_lite(4)
# model = create_efficient_yolof(4)
# model = create_mobiledet_yolof(4)
# model = create_mobilenetv3_small_yolof_lite(4)
stat(model, (3, 300, 300))