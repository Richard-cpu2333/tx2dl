import numpy as np
from vision.yolof.uniform_matcher import generate_anchors


# -----------------------------------------------------------------------------
# Solver
# -----------------------------------------------------------------------------
SOLVER_BACKBONE_MULTIPLIER = 0.334
SOLVER_WEIGHT_DECAY_NORM = 0.

# -----------------------------------------------------------------------------
# JitterCrop Transformation
# -----------------------------------------------------------------------------
INPUT_JITTER_CROPENABLED = False
INPUT_JITTER_CROP_JITTER_RATIO = 0.3

# -----------------------------------------------------------------------------
# Resize Transformation
# -----------------------------------------------------------------------------
INPUT_RESIZE_ENABLED = False
INPUT_RESIZE_SHAPE = (640, 640)
INPUT_RESIZE_SCALE_JITTER = (0.8, 1.2)
INPUT_RESIZE_TEST_SHAPE = (608, 608)

# -----------------------------------------------------------------------------
# Distortion Transformation
# -----------------------------------------------------------------------------
INPUT_DISTORTION_ENABLED = False
INPUT_DISTORTION_HUE = 0.1
INPUT_DISTORTION_SATURATION = 1_5
INPUT_DISTORTION_EXPOSURE = 1_5

# -----------------------------------------------------------------------------
# Shift Transformation
# -----------------------------------------------------------------------------
INPUT_SHIFT_SHIFT_PIXELS = 32

# -----------------------------------------------------------------------------
# Mosaic Transformation
# -----------------------------------------------------------------------------
INPUT_MOSAIC_ENABLED = False
INPUT_MOSAIC_POOL_CAPACITY = 1000
INPUT_MOSAIC_NUM_IMAGES = 4
INPUT_MOSAIC_MIN_OFFSET = 0.2
INPUT_MOSAIC_MOSAIC_WIDTH = 640
INPUT_MOSAIC_MOSAIC_HEIGHT = 640

# -----------------------------------------------------------------------------
# Anchor generator options
# -----------------------------------------------------------------------------
MODEL_ANCHOR_GENERATOR_SIZES = [[32, 64, 128, 256, 512]]
MODEL_ANCHOR_GENERATOR_ASPECT_RATIOS = [[1_0]]
MODEL_ANCHOR_GENERATOR_OFFSET = 0.0

# -----------------------------------------------------------------------------
# BACKBONE
# -----------------------------------------------------------------------------
# MobileDet

# -----------------------------------------------------------------------------
# YOLOF
# -----------------------------------------------------------------------------

# YOLOF Encoder parameters
# Note that the list of dilations must be consistent with number of blocks


# YOLOF Decoder parameters


# YOLOF box2box transform

# YOLOF Uniform Matcher

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


# # MODEL:
#   BACKBONE:
#     NAME: "build_resnet_backbone"
#   RESNETS:
#     OUT_FEATURES: ["res5"]
# # DATASETS:
#   TRAIN: ("coco_2017_train",)
#   TEST: ("coco_2017_val",)
# # DATALOADER:
#   NUM_WORKERS: 8
# # SOLVER:
#   IMS_PER_BATCH: 64
#   BASE_LR: 0.12
#   WARMUP_FACTOR: 0.00066667
#   WARMUP_ITERS: 1500
#   STEPS: (15000, 20000)
#   MAX_ITER: 22500
#   CHECKPOINT_PERIOD: 2500
# # INPUT:
#   MIN_SIZE_TRAIN: (800,)

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
