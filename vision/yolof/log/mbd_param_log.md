```
(base) oem@richard:~/Documents/rqp_proj/tx2dl$ python measure.py 
Warning: module Mish is treated as a zero-op.
Warning: module Tucker is treated as a zero-op.
Warning: module Fused_IBN is treated as a zero-op.
Warning: module Inverted_Bottleneck is treated as a zero-op.
Warning: module Bottleneck is treated as a zero-op.
Warning: module DilatedEncoder is treated as a zero-op.
Warning: module Decoder is treated as a zero-op.
Warning: module YOLOF is treated as a zero-op.
YOLOF(
  29.718 M, 100.000% Params, 6.384 GMac, 100.000% MACs, 
  (backbone): Sequential(
    11.705 M, 39.386% Params, 4.582 GMac, 71.773% MACs, 
    (0): Conv2d(0.001 M, 0.003% Params, 0.02 GMac, 0.316% MACs, 3, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
    (1): BatchNorm2d(0.0 M, 0.000% Params, 0.001 GMac, 0.023% MACs, 32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): Mish(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, )
    (3): Tucker(
      0.001 M, 0.005% Params, 0.032 GMac, 0.500% MACs, 
      (conv1): Conv2d(0.0 M, 0.001% Params, 0.006 GMac, 0.093% MACs, 32, 8, kernel_size=(1, 1), stride=(1, 1))
      (bn1): BatchNorm2d(0.0 M, 0.000% Params, 0.0 GMac, 0.006% MACs, 8, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (act1): Mish(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, )
      (conv2): Conv2d(0.001 M, 0.003% Params, 0.02 GMac, 0.309% MACs, 8, 12, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (bn2): BatchNorm2d(0.0 M, 0.000% Params, 0.001 GMac, 0.008% MACs, 12, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (act2): Mish(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, )
      (conv3): Conv2d(0.0 M, 0.001% Params, 0.005 GMac, 0.073% MACs, 12, 16, kernel_size=(1, 1), stride=(1, 1))
      (bn3): BatchNorm2d(0.0 M, 0.000% Params, 0.001 GMac, 0.011% MACs, 16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (act3): Mish(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, )
    )
    (4): Fused_IBN(
      0.023 M, 0.077% Params, 0.129 GMac, 2.027% MACs, 
      (conv1): Conv2d(0.019 M, 0.062% Params, 0.104 GMac, 1.635% MACs, 16, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
      (bn1): BatchNorm2d(0.0 M, 0.001% Params, 0.001 GMac, 0.023% MACs, 128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (act1): Mish(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, )
      (conv2): Conv2d(0.004 M, 0.014% Params, 0.023 GMac, 0.364% MACs, 128, 32, kernel_size=(1, 1), stride=(1, 1))
      (bn2): BatchNorm2d(0.0 M, 0.000% Params, 0.0 GMac, 0.006% MACs, 32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (act2): Mish(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, )
    )
    (5): Tucker(
      0.001 M, 0.004% Params, 0.007 GMac, 0.109% MACs, 
      (conv1): Conv2d(0.0 M, 0.001% Params, 0.001 GMac, 0.023% MACs, 32, 8, kernel_size=(1, 1), stride=(1, 1))
      (bn1): BatchNorm2d(0.0 M, 0.000% Params, 0.0 GMac, 0.001% MACs, 8, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (act1): Mish(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, )
      (conv2): Conv2d(0.001 M, 0.002% Params, 0.003 GMac, 0.051% MACs, 8, 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (bn2): BatchNorm2d(0.0 M, 0.000% Params, 0.0 GMac, 0.001% MACs, 8, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (act2): Mish(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, )
      (conv3): Conv2d(0.0 M, 0.001% Params, 0.002 GMac, 0.025% MACs, 8, 32, kernel_size=(1, 1), stride=(1, 1))
      (bn3): BatchNorm2d(0.0 M, 0.000% Params, 0.0 GMac, 0.006% MACs, 32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (act3): Mish(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, )
    )
    (6): Tucker(
      0.001 M, 0.004% Params, 0.007 GMac, 0.109% MACs, 
      (conv1): Conv2d(0.0 M, 0.001% Params, 0.001 GMac, 0.023% MACs, 32, 8, kernel_size=(1, 1), stride=(1, 1))
      (bn1): BatchNorm2d(0.0 M, 0.000% Params, 0.0 GMac, 0.001% MACs, 8, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (act1): Mish(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, )
      (conv2): Conv2d(0.001 M, 0.002% Params, 0.003 GMac, 0.051% MACs, 8, 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (bn2): BatchNorm2d(0.0 M, 0.000% Params, 0.0 GMac, 0.001% MACs, 8, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (act2): Mish(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, )
      (conv3): Conv2d(0.0 M, 0.001% Params, 0.002 GMac, 0.025% MACs, 8, 32, kernel_size=(1, 1), stride=(1, 1))
      (bn3): BatchNorm2d(0.0 M, 0.000% Params, 0.0 GMac, 0.006% MACs, 32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (act3): Mish(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, )
    )
    (7): Tucker(
      0.001 M, 0.004% Params, 0.007 GMac, 0.109% MACs, 
      (conv1): Conv2d(0.0 M, 0.001% Params, 0.001 GMac, 0.023% MACs, 32, 8, kernel_size=(1, 1), stride=(1, 1))
      (bn1): BatchNorm2d(0.0 M, 0.000% Params, 0.0 GMac, 0.001% MACs, 8, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (act1): Mish(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, )
      (conv2): Conv2d(0.001 M, 0.002% Params, 0.003 GMac, 0.051% MACs, 8, 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (bn2): BatchNorm2d(0.0 M, 0.000% Params, 0.0 GMac, 0.001% MACs, 8, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (act2): Mish(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, )
      (conv3): Conv2d(0.0 M, 0.001% Params, 0.002 GMac, 0.025% MACs, 8, 32, kernel_size=(1, 1), stride=(1, 1))
      (bn3): BatchNorm2d(0.0 M, 0.000% Params, 0.0 GMac, 0.006% MACs, 32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (act3): Mish(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, )
    )
    (8): Fused_IBN(
      0.091 M, 0.306% Params, 0.132 GMac, 2.060% MACs, 
      (conv1): Conv2d(0.074 M, 0.249% Params, 0.107 GMac, 1.673% MACs, 32, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
      (bn1): BatchNorm2d(0.001 M, 0.002% Params, 0.001 GMac, 0.012% MACs, 256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (act1): Mish(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, )
      (conv2): Conv2d(0.016 M, 0.055% Params, 0.024 GMac, 0.372% MACs, 256, 64, kernel_size=(1, 1), stride=(1, 1))
      (bn2): BatchNorm2d(0.0 M, 0.000% Params, 0.0 GMac, 0.003% MACs, 64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (act2): Mish(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, )
    )
    (9): Fused_IBN(
      0.329 M, 1.108% Params, 0.476 GMac, 7.451% MACs, 
      (conv1): Conv2d(0.295 M, 0.994% Params, 0.427 GMac, 6.682% MACs, 64, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (bn1): BatchNorm2d(0.001 M, 0.003% Params, 0.001 GMac, 0.023% MACs, 512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (act1): Mish(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, )
      (conv2): Conv2d(0.033 M, 0.110% Params, 0.047 GMac, 0.743% MACs, 512, 64, kernel_size=(1, 1), stride=(1, 1))
      (bn2): BatchNorm2d(0.0 M, 0.000% Params, 0.0 GMac, 0.003% MACs, 64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (act2): Mish(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, )
    )
    (10): Fused_IBN(
      0.329 M, 1.108% Params, 0.476 GMac, 7.451% MACs, 
      (conv1): Conv2d(0.295 M, 0.994% Params, 0.427 GMac, 6.682% MACs, 64, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (bn1): BatchNorm2d(0.001 M, 0.003% Params, 0.001 GMac, 0.023% MACs, 512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (act1): Mish(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, )
      (conv2): Conv2d(0.033 M, 0.110% Params, 0.047 GMac, 0.743% MACs, 512, 64, kernel_size=(1, 1), stride=(1, 1))
      (bn2): BatchNorm2d(0.0 M, 0.000% Params, 0.0 GMac, 0.003% MACs, 64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (act2): Mish(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, )
    )
    (11): Fused_IBN(
      0.165 M, 0.555% Params, 0.238 GMac, 3.728% MACs, 
      (conv1): Conv2d(0.148 M, 0.497% Params, 0.213 GMac, 3.341% MACs, 64, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (bn1): BatchNorm2d(0.001 M, 0.002% Params, 0.001 GMac, 0.012% MACs, 256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (act1): Mish(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, )
      (conv2): Conv2d(0.016 M, 0.055% Params, 0.024 GMac, 0.372% MACs, 256, 64, kernel_size=(1, 1), stride=(1, 1))
      (bn2): BatchNorm2d(0.0 M, 0.000% Params, 0.0 GMac, 0.003% MACs, 64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (act2): Mish(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, )
    )
    (12): Fused_IBN(
      0.362 M, 1.219% Params, 0.131 GMac, 2.049% MACs, 
      (conv1): Conv2d(0.295 M, 0.994% Params, 0.107 GMac, 1.671% MACs, 64, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
      (bn1): BatchNorm2d(0.001 M, 0.003% Params, 0.0 GMac, 0.006% MACs, 512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (act1): Mish(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, )
      (conv2): Conv2d(0.066 M, 0.221% Params, 0.024 GMac, 0.371% MACs, 512, 128, kernel_size=(1, 1), stride=(1, 1))
      (bn2): BatchNorm2d(0.0 M, 0.001% Params, 0.0 GMac, 0.001% MACs, 128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (act2): Mish(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, )
    )
    (13): Fused_IBN(
      0.657 M, 2.212% Params, 0.237 GMac, 3.717% MACs, 
      (conv1): Conv2d(0.59 M, 1.986% Params, 0.213 GMac, 3.338% MACs, 128, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (bn1): BatchNorm2d(0.001 M, 0.003% Params, 0.0 GMac, 0.006% MACs, 512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (act1): Mish(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, )
      (conv2): Conv2d(0.066 M, 0.221% Params, 0.024 GMac, 0.371% MACs, 512, 128, kernel_size=(1, 1), stride=(1, 1))
      (bn2): BatchNorm2d(0.0 M, 0.001% Params, 0.0 GMac, 0.001% MACs, 128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (act2): Mish(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, )
    )
    (14): Fused_IBN(
      0.657 M, 2.212% Params, 0.237 GMac, 3.717% MACs, 
      (conv1): Conv2d(0.59 M, 1.986% Params, 0.213 GMac, 3.338% MACs, 128, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (bn1): BatchNorm2d(0.001 M, 0.003% Params, 0.0 GMac, 0.006% MACs, 512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (act1): Mish(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, )
      (conv2): Conv2d(0.066 M, 0.221% Params, 0.024 GMac, 0.371% MACs, 512, 128, kernel_size=(1, 1), stride=(1, 1))
      (bn2): BatchNorm2d(0.0 M, 0.001% Params, 0.0 GMac, 0.001% MACs, 128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (act2): Mish(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, )
    )
    (15): Fused_IBN(
      0.657 M, 2.212% Params, 0.237 GMac, 3.717% MACs, 
      (conv1): Conv2d(0.59 M, 1.986% Params, 0.213 GMac, 3.338% MACs, 128, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (bn1): BatchNorm2d(0.001 M, 0.003% Params, 0.0 GMac, 0.006% MACs, 512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (act1): Mish(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, )
      (conv2): Conv2d(0.066 M, 0.221% Params, 0.024 GMac, 0.371% MACs, 512, 128, kernel_size=(1, 1), stride=(1, 1))
      (bn2): BatchNorm2d(0.0 M, 0.001% Params, 0.0 GMac, 0.001% MACs, 128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (act2): Mish(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, )
    )
    (16): Fused_IBN(
      1.314 M, 4.422% Params, 0.474 GMac, 7.432% MACs, 
      (conv1): Conv2d(1.181 M, 3.973% Params, 0.426 GMac, 6.677% MACs, 128, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (bn1): BatchNorm2d(0.002 M, 0.007% Params, 0.001 GMac, 0.012% MACs, 1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (act1): Mish(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, )
      (conv2): Conv2d(0.131 M, 0.441% Params, 0.047 GMac, 0.742% MACs, 1024, 128, kernel_size=(1, 1), stride=(1, 1))
      (bn2): BatchNorm2d(0.0 M, 0.001% Params, 0.0 GMac, 0.001% MACs, 128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (act2): Mish(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, )
    )
    (17): Fused_IBN(
      1.314 M, 4.422% Params, 0.474 GMac, 7.432% MACs, 
      (conv1): Conv2d(1.181 M, 3.973% Params, 0.426 GMac, 6.677% MACs, 128, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (bn1): BatchNorm2d(0.002 M, 0.007% Params, 0.001 GMac, 0.012% MACs, 1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (act1): Mish(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, )
      (conv2): Conv2d(0.131 M, 0.441% Params, 0.047 GMac, 0.742% MACs, 1024, 128, kernel_size=(1, 1), stride=(1, 1))
      (bn2): BatchNorm2d(0.0 M, 0.001% Params, 0.0 GMac, 0.001% MACs, 128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (act2): Mish(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, )
    )
    (18): Fused_IBN(
      1.314 M, 4.422% Params, 0.474 GMac, 7.432% MACs, 
      (conv1): Conv2d(1.181 M, 3.973% Params, 0.426 GMac, 6.677% MACs, 128, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (bn1): BatchNorm2d(0.002 M, 0.007% Params, 0.001 GMac, 0.012% MACs, 1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (act1): Mish(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, )
      (conv2): Conv2d(0.131 M, 0.441% Params, 0.047 GMac, 0.742% MACs, 1024, 128, kernel_size=(1, 1), stride=(1, 1))
      (bn2): BatchNorm2d(0.0 M, 0.001% Params, 0.0 GMac, 0.001% MACs, 128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (act2): Mish(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, )
    )
    (19): Fused_IBN(
      1.314 M, 4.422% Params, 0.474 GMac, 7.432% MACs, 
      (conv1): Conv2d(1.181 M, 3.973% Params, 0.426 GMac, 6.677% MACs, 128, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (bn1): BatchNorm2d(0.002 M, 0.007% Params, 0.001 GMac, 0.012% MACs, 1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (act1): Mish(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, )
      (conv2): Conv2d(0.131 M, 0.441% Params, 0.047 GMac, 0.742% MACs, 1024, 128, kernel_size=(1, 1), stride=(1, 1))
      (bn2): BatchNorm2d(0.0 M, 0.001% Params, 0.0 GMac, 0.001% MACs, 128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (act2): Mish(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, )
    )
    (20): Fused_IBN(
      0.657 M, 2.212% Params, 0.066 GMac, 1.030% MACs, 
      (conv1): Conv2d(0.59 M, 1.986% Params, 0.059 GMac, 0.925% MACs, 128, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
      (bn1): BatchNorm2d(0.001 M, 0.003% Params, 0.0 GMac, 0.002% MACs, 512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (act1): Mish(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, )
      (conv2): Conv2d(0.066 M, 0.221% Params, 0.007 GMac, 0.103% MACs, 512, 128, kernel_size=(1, 1), stride=(1, 1))
      (bn2): BatchNorm2d(0.0 M, 0.001% Params, 0.0 GMac, 0.000% MACs, 128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (act2): Mish(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, )
    )
    (21): Fused_IBN(
      0.657 M, 2.212% Params, 0.066 GMac, 1.030% MACs, 
      (conv1): Conv2d(0.59 M, 1.986% Params, 0.059 GMac, 0.925% MACs, 128, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (bn1): BatchNorm2d(0.001 M, 0.003% Params, 0.0 GMac, 0.002% MACs, 512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (act1): Mish(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, )
      (conv2): Conv2d(0.066 M, 0.221% Params, 0.007 GMac, 0.103% MACs, 512, 128, kernel_size=(1, 1), stride=(1, 1))
      (bn2): BatchNorm2d(0.0 M, 0.001% Params, 0.0 GMac, 0.000% MACs, 128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (act2): Mish(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, )
    )
    (22): Fused_IBN(
      0.657 M, 2.212% Params, 0.066 GMac, 1.030% MACs, 
      (conv1): Conv2d(0.59 M, 1.986% Params, 0.059 GMac, 0.925% MACs, 128, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (bn1): BatchNorm2d(0.001 M, 0.003% Params, 0.0 GMac, 0.002% MACs, 512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (act1): Mish(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, )
      (conv2): Conv2d(0.066 M, 0.221% Params, 0.007 GMac, 0.103% MACs, 512, 128, kernel_size=(1, 1), stride=(1, 1))
      (bn2): BatchNorm2d(0.0 M, 0.001% Params, 0.0 GMac, 0.000% MACs, 128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (act2): Mish(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, )
    )
    (23): Fused_IBN(
      0.657 M, 2.212% Params, 0.066 GMac, 1.030% MACs, 
      (conv1): Conv2d(0.59 M, 1.986% Params, 0.059 GMac, 0.925% MACs, 128, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (bn1): BatchNorm2d(0.001 M, 0.003% Params, 0.0 GMac, 0.002% MACs, 512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (act1): Mish(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, )
      (conv2): Conv2d(0.066 M, 0.221% Params, 0.007 GMac, 0.103% MACs, 512, 128, kernel_size=(1, 1), stride=(1, 1))
      (bn2): BatchNorm2d(0.0 M, 0.001% Params, 0.0 GMac, 0.000% MACs, 128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (act2): Mish(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, )
    )
    (24): Inverted_Bottleneck(
      0.541 M, 1.820% Params, 0.054 GMac, 0.847% MACs, 
      (conv1): Conv2d(0.132 M, 0.445% Params, 0.013 GMac, 0.207% MACs, 128, 1024, kernel_size=(1, 1), stride=(1, 1))
      (bn1): BatchNorm2d(0.002 M, 0.007% Params, 0.0 GMac, 0.003% MACs, 1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (act1): Mish(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, )
      (conv2): Conv2d(0.01 M, 0.034% Params, 0.001 GMac, 0.016% MACs, 1024, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=1024)
      (bn2): BatchNorm2d(0.002 M, 0.007% Params, 0.0 GMac, 0.003% MACs, 1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (act2): Mish(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, )
      (conv3): Conv2d(0.394 M, 1.324% Params, 0.039 GMac, 0.617% MACs, 1024, 384, kernel_size=(1, 1), stride=(1, 1))
      (bn3): BatchNorm2d(0.001 M, 0.003% Params, 0.0 GMac, 0.001% MACs, 384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (act3): Mish(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, )
    )
  )
  (encoder): DilatedEncoder(
    3.682 M, 12.391% Params, 0.369 GMac, 5.773% MACs, 
    (lateral_conv): Conv2d(0.197 M, 0.663% Params, 0.02 GMac, 0.309% MACs, 384, 512, kernel_size=(1, 1), stride=(1, 1))
    (lateral_norm): BatchNorm2d(0.001 M, 0.003% Params, 0.0 GMac, 0.002% MACs, 512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (fpn_conv): Conv2d(2.36 M, 7.941% Params, 0.236 GMac, 3.697% MACs, 512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (fpn_norm): BatchNorm2d(0.001 M, 0.003% Params, 0.0 GMac, 0.002% MACs, 512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (dilated_encoder_blocks): Sequential(
      1.123 M, 3.780% Params, 0.113 GMac, 1.764% MACs, 
      (0): Bottleneck(
        0.281 M, 0.945% Params, 0.028 GMac, 0.441% MACs, 
        (conv1): Sequential(
          0.066 M, 0.222% Params, 0.007 GMac, 0.103% MACs, 
          (0): Conv2d(0.066 M, 0.221% Params, 0.007 GMac, 0.103% MACs, 512, 128, kernel_size=(1, 1), stride=(1, 1))
          (1): BatchNorm2d(0.0 M, 0.001% Params, 0.0 GMac, 0.000% MACs, 128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): LeakyReLU(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, negative_slope=0.01)
        )
        (conv2): Sequential(
          0.148 M, 0.497% Params, 0.015 GMac, 0.232% MACs, 
          (0): Conv2d(0.148 M, 0.497% Params, 0.015 GMac, 0.231% MACs, 128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(2, 2), dilation=(2, 2))
          (1): BatchNorm2d(0.0 M, 0.001% Params, 0.0 GMac, 0.000% MACs, 128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): LeakyReLU(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, negative_slope=0.01)
        )
        (conv3): Sequential(
          0.067 M, 0.226% Params, 0.007 GMac, 0.106% MACs, 
          (0): Conv2d(0.066 M, 0.222% Params, 0.007 GMac, 0.103% MACs, 128, 512, kernel_size=(1, 1), stride=(1, 1))
          (1): BatchNorm2d(0.001 M, 0.003% Params, 0.0 GMac, 0.002% MACs, 512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): LeakyReLU(0.0 M, 0.000% Params, 0.0 GMac, 0.001% MACs, negative_slope=0.01)
        )
      )
      (1): Bottleneck(
        0.281 M, 0.945% Params, 0.028 GMac, 0.441% MACs, 
        (conv1): Sequential(
          0.066 M, 0.222% Params, 0.007 GMac, 0.103% MACs, 
          (0): Conv2d(0.066 M, 0.221% Params, 0.007 GMac, 0.103% MACs, 512, 128, kernel_size=(1, 1), stride=(1, 1))
          (1): BatchNorm2d(0.0 M, 0.001% Params, 0.0 GMac, 0.000% MACs, 128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): LeakyReLU(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, negative_slope=0.01)
        )
        (conv2): Sequential(
          0.148 M, 0.497% Params, 0.015 GMac, 0.232% MACs, 
          (0): Conv2d(0.148 M, 0.497% Params, 0.015 GMac, 0.231% MACs, 128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(4, 4), dilation=(4, 4))
          (1): BatchNorm2d(0.0 M, 0.001% Params, 0.0 GMac, 0.000% MACs, 128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): LeakyReLU(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, negative_slope=0.01)
        )
        (conv3): Sequential(
          0.067 M, 0.226% Params, 0.007 GMac, 0.106% MACs, 
          (0): Conv2d(0.066 M, 0.222% Params, 0.007 GMac, 0.103% MACs, 128, 512, kernel_size=(1, 1), stride=(1, 1))
          (1): BatchNorm2d(0.001 M, 0.003% Params, 0.0 GMac, 0.002% MACs, 512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): LeakyReLU(0.0 M, 0.000% Params, 0.0 GMac, 0.001% MACs, negative_slope=0.01)
        )
      )
      (2): Bottleneck(
        0.281 M, 0.945% Params, 0.028 GMac, 0.441% MACs, 
        (conv1): Sequential(
          0.066 M, 0.222% Params, 0.007 GMac, 0.103% MACs, 
          (0): Conv2d(0.066 M, 0.221% Params, 0.007 GMac, 0.103% MACs, 512, 128, kernel_size=(1, 1), stride=(1, 1))
          (1): BatchNorm2d(0.0 M, 0.001% Params, 0.0 GMac, 0.000% MACs, 128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): LeakyReLU(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, negative_slope=0.01)
        )
        (conv2): Sequential(
          0.148 M, 0.497% Params, 0.015 GMac, 0.232% MACs, 
          (0): Conv2d(0.148 M, 0.497% Params, 0.015 GMac, 0.231% MACs, 128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(6, 6), dilation=(6, 6))
          (1): BatchNorm2d(0.0 M, 0.001% Params, 0.0 GMac, 0.000% MACs, 128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): LeakyReLU(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, negative_slope=0.01)
        )
        (conv3): Sequential(
          0.067 M, 0.226% Params, 0.007 GMac, 0.106% MACs, 
          (0): Conv2d(0.066 M, 0.222% Params, 0.007 GMac, 0.103% MACs, 128, 512, kernel_size=(1, 1), stride=(1, 1))
          (1): BatchNorm2d(0.001 M, 0.003% Params, 0.0 GMac, 0.002% MACs, 512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): LeakyReLU(0.0 M, 0.000% Params, 0.0 GMac, 0.001% MACs, negative_slope=0.01)
        )
      )
      (3): Bottleneck(
        0.281 M, 0.945% Params, 0.028 GMac, 0.441% MACs, 
        (conv1): Sequential(
          0.066 M, 0.222% Params, 0.007 GMac, 0.103% MACs, 
          (0): Conv2d(0.066 M, 0.221% Params, 0.007 GMac, 0.103% MACs, 512, 128, kernel_size=(1, 1), stride=(1, 1))
          (1): BatchNorm2d(0.0 M, 0.001% Params, 0.0 GMac, 0.000% MACs, 128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): LeakyReLU(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, negative_slope=0.01)
        )
        (conv2): Sequential(
          0.148 M, 0.497% Params, 0.015 GMac, 0.232% MACs, 
          (0): Conv2d(0.148 M, 0.497% Params, 0.015 GMac, 0.231% MACs, 128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(8, 8), dilation=(8, 8))
          (1): BatchNorm2d(0.0 M, 0.001% Params, 0.0 GMac, 0.000% MACs, 128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): LeakyReLU(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, negative_slope=0.01)
        )
        (conv3): Sequential(
          0.067 M, 0.226% Params, 0.007 GMac, 0.106% MACs, 
          (0): Conv2d(0.066 M, 0.222% Params, 0.007 GMac, 0.103% MACs, 128, 512, kernel_size=(1, 1), stride=(1, 1))
          (1): BatchNorm2d(0.001 M, 0.003% Params, 0.0 GMac, 0.002% MACs, 512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): LeakyReLU(0.0 M, 0.000% Params, 0.0 GMac, 0.001% MACs, negative_slope=0.01)
        )
      )
    )
  )
  (decoder): Decoder(
    14.331 M, 48.223% Params, 1.433 GMac, 22.454% MACs, 
    (cls_subnet): Sequential(
      4.722 M, 15.888% Params, 0.472 GMac, 7.398% MACs, 
      (0): Conv2d(2.36 M, 7.941% Params, 0.236 GMac, 3.697% MACs, 512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (1): BatchNorm2d(0.001 M, 0.003% Params, 0.0 GMac, 0.002% MACs, 512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): LeakyReLU(0.0 M, 0.000% Params, 0.0 GMac, 0.001% MACs, negative_slope=0.01)
      (3): Conv2d(2.36 M, 7.941% Params, 0.236 GMac, 3.697% MACs, 512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (4): BatchNorm2d(0.001 M, 0.003% Params, 0.0 GMac, 0.002% MACs, 512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (5): LeakyReLU(0.0 M, 0.000% Params, 0.0 GMac, 0.001% MACs, negative_slope=0.01)
    )
    (bbox_subnet): Sequential(
      9.443 M, 31.777% Params, 0.945 GMac, 14.796% MACs, 
      (0): Conv2d(2.36 M, 7.941% Params, 0.236 GMac, 3.697% MACs, 512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (1): BatchNorm2d(0.001 M, 0.003% Params, 0.0 GMac, 0.002% MACs, 512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): LeakyReLU(0.0 M, 0.000% Params, 0.0 GMac, 0.001% MACs, negative_slope=0.01)
      (3): Conv2d(2.36 M, 7.941% Params, 0.236 GMac, 3.697% MACs, 512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (4): BatchNorm2d(0.001 M, 0.003% Params, 0.0 GMac, 0.002% MACs, 512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (5): LeakyReLU(0.0 M, 0.000% Params, 0.0 GMac, 0.001% MACs, negative_slope=0.01)
      (6): Conv2d(2.36 M, 7.941% Params, 0.236 GMac, 3.697% MACs, 512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (7): BatchNorm2d(0.001 M, 0.003% Params, 0.0 GMac, 0.002% MACs, 512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (8): LeakyReLU(0.0 M, 0.000% Params, 0.0 GMac, 0.001% MACs, negative_slope=0.01)
      (9): Conv2d(2.36 M, 7.941% Params, 0.236 GMac, 3.697% MACs, 512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (10): BatchNorm2d(0.001 M, 0.003% Params, 0.0 GMac, 0.002% MACs, 512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (11): LeakyReLU(0.0 M, 0.000% Params, 0.0 GMac, 0.001% MACs, negative_slope=0.01)
    )
    (cls_score): Conv2d(0.074 M, 0.248% Params, 0.007 GMac, 0.116% MACs, 512, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (bbox_pred): Conv2d(0.074 M, 0.248% Params, 0.007 GMac, 0.116% MACs, 512, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (object_pred): Conv2d(0.018 M, 0.062% Params, 0.002 GMac, 0.029% MACs, 512, 4, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  )
)
Computational complexity:       6.38 GMac
Number of parameters:           29.72 M
```

