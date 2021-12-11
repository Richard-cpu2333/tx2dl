import numpy as np
from vision.utils.box_utils import generate_anchors

# YOLOF ignore thresholds
MODEL_YOLOF_POS_IGNORE_THRESHOLD = 0.15
MODEL_YOLOF_NEG_IGNORE_THRESHOLD = 0.7

# YOLOF losses
MODEL_YOLOF_LOSSES_FOCAL_LOSS_GAMMA = 2.0
MODEL_YOLOF_LOSSES_FOCAL_LOSS_ALPHA = 0.25
MODEL_YOLOF_LOSSES_BBOX_REG_LOSS_TYPE = "giou"

# YOLOF test
MODEL_YOLOF_SCORE_THRESH_TEST = 0.05
MODEL_YOLOF_TOPK_CANDIDATES_TEST = 1000
MODEL_YOLOF_NMS_THRESH_TEST = 0.6
MODEL_YOLOF_DETECTIONS_PER_IMAGE = 100

image_size = 300 # [1242, 375]
image_mean = np.array([127, 127, 127])  # RGB layout
image_std = 128.0
iou_threshold = 0.45
center_variance = 0.1
size_variance = 0.2
feature_map_size = 10
shrinkage = 32
box_sizes = [105, 150]
aspect_ratios = [2,3]
anchors = generate_anchors()
