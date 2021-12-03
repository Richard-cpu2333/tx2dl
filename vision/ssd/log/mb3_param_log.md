```
(base) oem@richard:~/Documents/rqp_proj/tx2dl$ python measure.py 
Warning: module hswish is treated as a zero-op.
Warning: module hsigmoid is treated as a zero-op.
Warning: module SeModule is treated as a zero-op.
Warning: module Block is treated as a zero-op.
Warning: module SSD is treated as a zero-op.
SSD(
  1.082 M, 100.000% Params, 0.144 GMac, 100.000% MACs, 
  (base_net): Sequential(
    0.488 M, 45.061% Params, 0.117 GMac, 81.018% MACs, 
    (0): Conv2d(0.0 M, 0.040% Params, 0.01 GMac, 6.734% MACs, 3, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
    (1): BatchNorm2d(0.0 M, 0.003% Params, 0.001 GMac, 0.499% MACs, 16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): hswish(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, )
    (3): Block(
      0.001 M, 0.085% Params, 0.01 GMac, 6.672% MACs, 
      (se): SeModule(
        0.0 M, 0.016% Params, 0.0 GMac, 0.062% MACs, 
        (se): Sequential(
          0.0 M, 0.016% Params, 0.0 GMac, 0.062% MACs, 
          (0): AdaptiveAvgPool2d(0.0 M, 0.000% Params, 0.0 GMac, 0.062% MACs, output_size=1)
          (1): Conv2d(0.0 M, 0.006% Params, 0.0 GMac, 0.000% MACs, 16, 4, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (2): BatchNorm2d(0.0 M, 0.001% Params, 0.0 GMac, 0.000% MACs, 4, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (3): ReLU(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, inplace=True)
          (4): Conv2d(0.0 M, 0.006% Params, 0.0 GMac, 0.000% MACs, 4, 16, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (5): BatchNorm2d(0.0 M, 0.003% Params, 0.0 GMac, 0.000% MACs, 16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (6): hsigmoid(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, )
        )
      )
      (conv1): Conv2d(0.0 M, 0.024% Params, 0.006 GMac, 3.990% MACs, 16, 16, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn1): BatchNorm2d(0.0 M, 0.003% Params, 0.001 GMac, 0.499% MACs, 16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (nolinear1): ReLU(0.0 M, 0.000% Params, 0.0 GMac, 0.312% MACs, inplace=True)
      (conv2): Conv2d(0.0 M, 0.013% Params, 0.001 GMac, 0.561% MACs, 16, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=16, bias=False)
      (bn2): BatchNorm2d(0.0 M, 0.003% Params, 0.0 GMac, 0.125% MACs, 16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (nolinear2): ReLU(0.0 M, 0.000% Params, 0.0 GMac, 0.312% MACs, inplace=True)
      (conv3): Conv2d(0.0 M, 0.024% Params, 0.001 GMac, 0.998% MACs, 16, 16, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn3): BatchNorm2d(0.0 M, 0.003% Params, 0.0 GMac, 0.125% MACs, 16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (shortcut): Sequential(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, )
    )
    (4): Block(
      0.004 M, 0.357% Params, 0.012 GMac, 7.972% MACs, 
      (conv1): Conv2d(0.001 M, 0.106% Params, 0.006 GMac, 4.489% MACs, 16, 72, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn1): BatchNorm2d(0.0 M, 0.013% Params, 0.001 GMac, 0.561% MACs, 72, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (nolinear1): ReLU(0.0 M, 0.000% Params, 0.001 GMac, 0.353% MACs, inplace=True)
      (conv2): Conv2d(0.001 M, 0.060% Params, 0.001 GMac, 0.648% MACs, 72, 72, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=72, bias=False)
      (bn2): BatchNorm2d(0.0 M, 0.013% Params, 0.0 GMac, 0.144% MACs, 72, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (nolinear2): ReLU(0.0 M, 0.000% Params, 0.001 GMac, 0.353% MACs, inplace=True)
      (conv3): Conv2d(0.002 M, 0.160% Params, 0.002 GMac, 1.729% MACs, 72, 24, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn3): BatchNorm2d(0.0 M, 0.004% Params, 0.0 GMac, 0.048% MACs, 24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (shortcut): Sequential(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, )
    )
    (5): Block(
      0.005 M, 0.500% Params, 0.008 GMac, 5.594% MACs, 
      (conv1): Conv2d(0.002 M, 0.195% Params, 0.003 GMac, 2.113% MACs, 24, 88, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn1): BatchNorm2d(0.0 M, 0.016% Params, 0.0 GMac, 0.176% MACs, 88, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (nolinear1): ReLU(0.0 M, 0.000% Params, 0.0 GMac, 0.176% MACs, inplace=True)
      (conv2): Conv2d(0.001 M, 0.073% Params, 0.001 GMac, 0.792% MACs, 88, 88, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=88, bias=False)
      (bn2): BatchNorm2d(0.0 M, 0.016% Params, 0.0 GMac, 0.176% MACs, 88, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (nolinear2): ReLU(0.0 M, 0.000% Params, 0.0 GMac, 0.176% MACs, inplace=True)
      (conv3): Conv2d(0.002 M, 0.195% Params, 0.003 GMac, 2.113% MACs, 88, 24, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn3): BatchNorm2d(0.0 M, 0.004% Params, 0.0 GMac, 0.048% MACs, 24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (shortcut): Sequential(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, )
    )
    (6): Block(
      0.01 M, 0.916% Params, 0.006 GMac, 4.136% MACs, 
      (se): SeModule(
        0.001 M, 0.083% Params, 0.0 GMac, 0.011% MACs, 
        (se): Sequential(
          0.001 M, 0.083% Params, 0.0 GMac, 0.011% MACs, 
          (0): AdaptiveAvgPool2d(0.0 M, 0.000% Params, 0.0 GMac, 0.010% MACs, output_size=1)
          (1): Conv2d(0.0 M, 0.037% Params, 0.0 GMac, 0.000% MACs, 40, 10, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (2): BatchNorm2d(0.0 M, 0.002% Params, 0.0 GMac, 0.000% MACs, 10, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (3): ReLU(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, inplace=True)
          (4): Conv2d(0.0 M, 0.037% Params, 0.0 GMac, 0.000% MACs, 10, 40, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (5): BatchNorm2d(0.0 M, 0.007% Params, 0.0 GMac, 0.000% MACs, 40, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (6): hsigmoid(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, )
        )
      )
      (conv1): Conv2d(0.002 M, 0.213% Params, 0.003 GMac, 2.305% MACs, 24, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn1): BatchNorm2d(0.0 M, 0.018% Params, 0.0 GMac, 0.192% MACs, 96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (nolinear1): hswish(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, )
      (conv2): Conv2d(0.002 M, 0.222% Params, 0.001 GMac, 0.600% MACs, 96, 96, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), groups=96, bias=False)
      (bn2): BatchNorm2d(0.0 M, 0.018% Params, 0.0 GMac, 0.048% MACs, 96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (nolinear2): hswish(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, )
      (conv3): Conv2d(0.004 M, 0.355% Params, 0.001 GMac, 0.960% MACs, 96, 40, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn3): BatchNorm2d(0.0 M, 0.007% Params, 0.0 GMac, 0.020% MACs, 40, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (shortcut): Sequential(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, )
    )
    (7): Block(
      0.027 M, 2.508% Params, 0.009 GMac, 6.573% MACs, 
      (se): SeModule(
        0.001 M, 0.083% Params, 0.0 GMac, 0.011% MACs, 
        (se): Sequential(
          0.001 M, 0.083% Params, 0.0 GMac, 0.011% MACs, 
          (0): AdaptiveAvgPool2d(0.0 M, 0.000% Params, 0.0 GMac, 0.010% MACs, output_size=1)
          (1): Conv2d(0.0 M, 0.037% Params, 0.0 GMac, 0.000% MACs, 40, 10, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (2): BatchNorm2d(0.0 M, 0.002% Params, 0.0 GMac, 0.000% MACs, 10, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (3): ReLU(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, inplace=True)
          (4): Conv2d(0.0 M, 0.037% Params, 0.0 GMac, 0.000% MACs, 10, 40, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (5): BatchNorm2d(0.0 M, 0.007% Params, 0.0 GMac, 0.000% MACs, 40, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (6): hsigmoid(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, )
        )
      )
      (conv1): Conv2d(0.01 M, 0.887% Params, 0.003 GMac, 2.401% MACs, 40, 240, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn1): BatchNorm2d(0.0 M, 0.044% Params, 0.0 GMac, 0.120% MACs, 240, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (nolinear1): hswish(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, )
      (conv2): Conv2d(0.006 M, 0.554% Params, 0.002 GMac, 1.501% MACs, 240, 240, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=240, bias=False)
      (bn2): BatchNorm2d(0.0 M, 0.044% Params, 0.0 GMac, 0.120% MACs, 240, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (nolinear2): hswish(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, )
      (conv3): Conv2d(0.01 M, 0.887% Params, 0.003 GMac, 2.401% MACs, 240, 40, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn3): BatchNorm2d(0.0 M, 0.007% Params, 0.0 GMac, 0.020% MACs, 40, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (shortcut): Sequential(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, )
    )
    (8): Block(
      0.027 M, 2.508% Params, 0.009 GMac, 6.573% MACs, 
      (se): SeModule(
        0.001 M, 0.083% Params, 0.0 GMac, 0.011% MACs, 
        (se): Sequential(
          0.001 M, 0.083% Params, 0.0 GMac, 0.011% MACs, 
          (0): AdaptiveAvgPool2d(0.0 M, 0.000% Params, 0.0 GMac, 0.010% MACs, output_size=1)
          (1): Conv2d(0.0 M, 0.037% Params, 0.0 GMac, 0.000% MACs, 40, 10, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (2): BatchNorm2d(0.0 M, 0.002% Params, 0.0 GMac, 0.000% MACs, 10, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (3): ReLU(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, inplace=True)
          (4): Conv2d(0.0 M, 0.037% Params, 0.0 GMac, 0.000% MACs, 10, 40, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (5): BatchNorm2d(0.0 M, 0.007% Params, 0.0 GMac, 0.000% MACs, 40, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (6): hsigmoid(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, )
        )
      )
      (conv1): Conv2d(0.01 M, 0.887% Params, 0.003 GMac, 2.401% MACs, 40, 240, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn1): BatchNorm2d(0.0 M, 0.044% Params, 0.0 GMac, 0.120% MACs, 240, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (nolinear1): hswish(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, )
      (conv2): Conv2d(0.006 M, 0.554% Params, 0.002 GMac, 1.501% MACs, 240, 240, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=240, bias=False)
      (bn2): BatchNorm2d(0.0 M, 0.044% Params, 0.0 GMac, 0.120% MACs, 240, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (nolinear2): hswish(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, )
      (conv3): Conv2d(0.01 M, 0.887% Params, 0.003 GMac, 2.401% MACs, 240, 40, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn3): BatchNorm2d(0.0 M, 0.007% Params, 0.0 GMac, 0.020% MACs, 40, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (shortcut): Sequential(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, )
    )
    (9): Block(
      0.017 M, 1.610% Params, 0.006 GMac, 4.052% MACs, 
      (se): SeModule(
        0.001 M, 0.118% Params, 0.0 GMac, 0.013% MACs, 
        (se): Sequential(
          0.001 M, 0.118% Params, 0.0 GMac, 0.013% MACs, 
          (0): AdaptiveAvgPool2d(0.0 M, 0.000% Params, 0.0 GMac, 0.012% MACs, output_size=1)
          (1): Conv2d(0.001 M, 0.053% Params, 0.0 GMac, 0.000% MACs, 48, 12, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (2): BatchNorm2d(0.0 M, 0.002% Params, 0.0 GMac, 0.000% MACs, 12, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (3): ReLU(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, inplace=True)
          (4): Conv2d(0.001 M, 0.053% Params, 0.0 GMac, 0.000% MACs, 12, 48, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (5): BatchNorm2d(0.0 M, 0.009% Params, 0.0 GMac, 0.000% MACs, 48, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (6): hsigmoid(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, )
        )
      )
      (conv1): Conv2d(0.005 M, 0.444% Params, 0.002 GMac, 1.200% MACs, 40, 120, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn1): BatchNorm2d(0.0 M, 0.022% Params, 0.0 GMac, 0.060% MACs, 120, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (nolinear1): hswish(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, )
      (conv2): Conv2d(0.003 M, 0.277% Params, 0.001 GMac, 0.750% MACs, 120, 120, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=120, bias=False)
      (bn2): BatchNorm2d(0.0 M, 0.022% Params, 0.0 GMac, 0.060% MACs, 120, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (nolinear2): hswish(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, )
      (conv3): Conv2d(0.006 M, 0.532% Params, 0.002 GMac, 1.441% MACs, 120, 48, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn3): BatchNorm2d(0.0 M, 0.009% Params, 0.0 GMac, 0.024% MACs, 48, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (shortcut): Sequential(
        0.002 M, 0.186% Params, 0.001 GMac, 0.504% MACs, 
        (0): Conv2d(0.002 M, 0.177% Params, 0.001 GMac, 0.480% MACs, 40, 48, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (1): BatchNorm2d(0.0 M, 0.009% Params, 0.0 GMac, 0.024% MACs, 48, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (10): Block(
      0.019 M, 1.790% Params, 0.007 GMac, 4.539% MACs, 
      (se): SeModule(
        0.001 M, 0.118% Params, 0.0 GMac, 0.013% MACs, 
        (se): Sequential(
          0.001 M, 0.118% Params, 0.0 GMac, 0.013% MACs, 
          (0): AdaptiveAvgPool2d(0.0 M, 0.000% Params, 0.0 GMac, 0.012% MACs, output_size=1)
          (1): Conv2d(0.001 M, 0.053% Params, 0.0 GMac, 0.000% MACs, 48, 12, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (2): BatchNorm2d(0.0 M, 0.002% Params, 0.0 GMac, 0.000% MACs, 12, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (3): ReLU(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, inplace=True)
          (4): Conv2d(0.001 M, 0.053% Params, 0.0 GMac, 0.000% MACs, 12, 48, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (5): BatchNorm2d(0.0 M, 0.009% Params, 0.0 GMac, 0.000% MACs, 48, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (6): hsigmoid(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, )
        )
      )
      (conv1): Conv2d(0.007 M, 0.639% Params, 0.002 GMac, 1.729% MACs, 48, 144, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn1): BatchNorm2d(0.0 M, 0.027% Params, 0.0 GMac, 0.072% MACs, 144, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (nolinear1): hswish(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, )
      (conv2): Conv2d(0.004 M, 0.333% Params, 0.001 GMac, 0.900% MACs, 144, 144, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=144, bias=False)
      (bn2): BatchNorm2d(0.0 M, 0.027% Params, 0.0 GMac, 0.072% MACs, 144, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (nolinear2): hswish(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, )
      (conv3): Conv2d(0.007 M, 0.639% Params, 0.002 GMac, 1.729% MACs, 144, 48, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn3): BatchNorm2d(0.0 M, 0.009% Params, 0.0 GMac, 0.024% MACs, 48, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (shortcut): Sequential(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, )
    )
    (11): Block(
      0.055 M, 5.070% Params, 0.009 GMac, 6.079% MACs, 
      (se): SeModule(
        0.005 M, 0.448% Params, 0.0 GMac, 0.010% MACs, 
        (se): Sequential(
          0.005 M, 0.448% Params, 0.0 GMac, 0.010% MACs, 
          (0): AdaptiveAvgPool2d(0.0 M, 0.000% Params, 0.0 GMac, 0.007% MACs, output_size=1)
          (1): Conv2d(0.002 M, 0.213% Params, 0.0 GMac, 0.002% MACs, 96, 24, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (2): BatchNorm2d(0.0 M, 0.004% Params, 0.0 GMac, 0.000% MACs, 24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (3): ReLU(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, inplace=True)
          (4): Conv2d(0.002 M, 0.213% Params, 0.0 GMac, 0.002% MACs, 24, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (5): BatchNorm2d(0.0 M, 0.018% Params, 0.0 GMac, 0.000% MACs, 96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (6): hsigmoid(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, )
        )
      )
      (conv1): Conv2d(0.014 M, 1.277% Params, 0.005 GMac, 3.457% MACs, 48, 288, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn1): BatchNorm2d(0.001 M, 0.053% Params, 0.0 GMac, 0.144% MACs, 288, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (nolinear1): hswish(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, )
      (conv2): Conv2d(0.007 M, 0.665% Params, 0.001 GMac, 0.499% MACs, 288, 288, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), groups=288, bias=False)
      (bn2): BatchNorm2d(0.001 M, 0.053% Params, 0.0 GMac, 0.040% MACs, 288, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (nolinear2): hswish(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, )
      (conv3): Conv2d(0.028 M, 2.555% Params, 0.003 GMac, 1.915% MACs, 288, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn3): BatchNorm2d(0.0 M, 0.018% Params, 0.0 GMac, 0.013% MACs, 96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (shortcut): Sequential(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, )
    )
    (12): Block(
      0.132 M, 12.229% Params, 0.013 GMac, 8.842% MACs, 
      (se): SeModule(
        0.005 M, 0.448% Params, 0.0 GMac, 0.010% MACs, 
        (se): Sequential(
          0.005 M, 0.448% Params, 0.0 GMac, 0.010% MACs, 
          (0): AdaptiveAvgPool2d(0.0 M, 0.000% Params, 0.0 GMac, 0.007% MACs, output_size=1)
          (1): Conv2d(0.002 M, 0.213% Params, 0.0 GMac, 0.002% MACs, 96, 24, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (2): BatchNorm2d(0.0 M, 0.004% Params, 0.0 GMac, 0.000% MACs, 24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (3): ReLU(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, inplace=True)
          (4): Conv2d(0.002 M, 0.213% Params, 0.0 GMac, 0.002% MACs, 24, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (5): BatchNorm2d(0.0 M, 0.018% Params, 0.0 GMac, 0.000% MACs, 96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (6): hsigmoid(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, )
        )
      )
      (conv1): Conv2d(0.055 M, 5.110% Params, 0.006 GMac, 3.831% MACs, 96, 576, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn1): BatchNorm2d(0.001 M, 0.106% Params, 0.0 GMac, 0.080% MACs, 576, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (nolinear1): hswish(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, )
      (conv2): Conv2d(0.014 M, 1.331% Params, 0.001 GMac, 0.998% MACs, 576, 576, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=576, bias=False)
      (bn2): BatchNorm2d(0.001 M, 0.106% Params, 0.0 GMac, 0.080% MACs, 576, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (nolinear2): hswish(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, )
      (conv3): Conv2d(0.055 M, 5.110% Params, 0.006 GMac, 3.831% MACs, 576, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn3): BatchNorm2d(0.0 M, 0.018% Params, 0.0 GMac, 0.013% MACs, 96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (shortcut): Sequential(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, )
    )
    (13): Block(
      0.132 M, 12.229% Params, 0.013 GMac, 8.842% MACs, 
      (se): SeModule(
        0.005 M, 0.448% Params, 0.0 GMac, 0.010% MACs, 
        (se): Sequential(
          0.005 M, 0.448% Params, 0.0 GMac, 0.010% MACs, 
          (0): AdaptiveAvgPool2d(0.0 M, 0.000% Params, 0.0 GMac, 0.007% MACs, output_size=1)
          (1): Conv2d(0.002 M, 0.213% Params, 0.0 GMac, 0.002% MACs, 96, 24, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (2): BatchNorm2d(0.0 M, 0.004% Params, 0.0 GMac, 0.000% MACs, 24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (3): ReLU(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, inplace=True)
          (4): Conv2d(0.002 M, 0.213% Params, 0.0 GMac, 0.002% MACs, 24, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (5): BatchNorm2d(0.0 M, 0.018% Params, 0.0 GMac, 0.000% MACs, 96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (6): hsigmoid(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, )
        )
      )
      (conv1): Conv2d(0.055 M, 5.110% Params, 0.006 GMac, 3.831% MACs, 96, 576, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn1): BatchNorm2d(0.001 M, 0.106% Params, 0.0 GMac, 0.080% MACs, 576, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (nolinear1): hswish(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, )
      (conv2): Conv2d(0.014 M, 1.331% Params, 0.001 GMac, 0.998% MACs, 576, 576, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=576, bias=False)
      (bn2): BatchNorm2d(0.001 M, 0.106% Params, 0.0 GMac, 0.080% MACs, 576, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (nolinear2): hswish(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, )
      (conv3): Conv2d(0.055 M, 5.110% Params, 0.006 GMac, 3.831% MACs, 576, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn3): BatchNorm2d(0.0 M, 0.018% Params, 0.0 GMac, 0.013% MACs, 96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (shortcut): Sequential(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, )
    )
    (14): Conv2d(0.055 M, 5.110% Params, 0.006 GMac, 3.831% MACs, 96, 576, kernel_size=(1, 1), stride=(1, 1), bias=False)
    (15): BatchNorm2d(0.001 M, 0.106% Params, 0.0 GMac, 0.080% MACs, 576, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (16): hswish(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, )
  )
  (extras): ModuleList(
    0.473 M, 43.664% Params, 0.021 GMac, 14.295% MACs, 
    (0): Block(
      0.283 M, 26.140% Params, 0.018 GMac, 12.588% MACs, 
      (conv1): Conv2d(0.147 M, 13.626% Params, 0.015 GMac, 10.216% MACs, 576, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn1): BatchNorm2d(0.001 M, 0.047% Params, 0.0 GMac, 0.035% MACs, 256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (nolinear1): hswish(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, )
      (conv2): Conv2d(0.002 M, 0.213% Params, 0.0 GMac, 0.040% MACs, 256, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=256, bias=False)
      (bn2): BatchNorm2d(0.001 M, 0.047% Params, 0.0 GMac, 0.009% MACs, 256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (nolinear2): hswish(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, )
      (conv3): Conv2d(0.131 M, 12.112% Params, 0.003 GMac, 2.270% MACs, 256, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn3): BatchNorm2d(0.001 M, 0.095% Params, 0.0 GMac, 0.018% MACs, 512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (shortcut): Sequential(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, )
    )
    (1): Block(
      0.1 M, 9.285% Params, 0.002 GMac, 1.356% MACs, 
      (conv1): Conv2d(0.066 M, 6.056% Params, 0.002 GMac, 1.135% MACs, 512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn1): BatchNorm2d(0.0 M, 0.024% Params, 0.0 GMac, 0.004% MACs, 128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (nolinear1): hswish(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, )
      (conv2): Conv2d(0.001 M, 0.106% Params, 0.0 GMac, 0.007% MACs, 128, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=128, bias=False)
      (bn2): BatchNorm2d(0.0 M, 0.024% Params, 0.0 GMac, 0.002% MACs, 128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (nolinear2): hswish(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, )
      (conv3): Conv2d(0.033 M, 3.028% Params, 0.0 GMac, 0.204% MACs, 128, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn3): BatchNorm2d(0.001 M, 0.047% Params, 0.0 GMac, 0.003% MACs, 256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (shortcut): Sequential(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, )
    )
    (2): Block(
      0.068 M, 6.257% Params, 0.0 GMac, 0.302% MACs, 
      (conv1): Conv2d(0.033 M, 3.028% Params, 0.0 GMac, 0.204% MACs, 256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn1): BatchNorm2d(0.0 M, 0.024% Params, 0.0 GMac, 0.002% MACs, 128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (nolinear1): hswish(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, )
      (conv2): Conv2d(0.001 M, 0.106% Params, 0.0 GMac, 0.003% MACs, 128, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=128, bias=False)
      (bn2): BatchNorm2d(0.0 M, 0.024% Params, 0.0 GMac, 0.001% MACs, 128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (nolinear2): hswish(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, )
      (conv3): Conv2d(0.033 M, 3.028% Params, 0.0 GMac, 0.091% MACs, 128, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn3): BatchNorm2d(0.001 M, 0.047% Params, 0.0 GMac, 0.001% MACs, 256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (shortcut): Sequential(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, )
    )
    (3): Block(
      0.021 M, 1.981% Params, 0.0 GMac, 0.049% MACs, 
      (conv1): Conv2d(0.016 M, 1.514% Params, 0.0 GMac, 0.045% MACs, 256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn1): BatchNorm2d(0.0 M, 0.012% Params, 0.0 GMac, 0.000% MACs, 64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (nolinear1): hswish(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, )
      (conv2): Conv2d(0.001 M, 0.053% Params, 0.0 GMac, 0.000% MACs, 64, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=64, bias=False)
      (bn2): BatchNorm2d(0.0 M, 0.012% Params, 0.0 GMac, 0.000% MACs, 64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (nolinear2): hswish(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, )
      (conv3): Conv2d(0.004 M, 0.379% Params, 0.0 GMac, 0.003% MACs, 64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn3): BatchNorm2d(0.0 M, 0.012% Params, 0.0 GMac, 0.000% MACs, 64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (shortcut): Sequential(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, )
    )
  )
  (classification_headers): ModuleList(
    0.061 M, 5.638% Params, 0.003 GMac, 2.343% MACs, 
    (0): Sequential(
      0.002 M, 0.162% Params, 0.001 GMac, 0.450% MACs, 
      (0): Conv2d(0.0 M, 0.044% Params, 0.0 GMac, 0.120% MACs, 48, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=48)
      (1): BatchNorm2d(0.0 M, 0.009% Params, 0.0 GMac, 0.024% MACs, 48, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU6(0.0 M, 0.000% Params, 0.0 GMac, 0.012% MACs, )
      (3): Conv2d(0.001 M, 0.109% Params, 0.0 GMac, 0.294% MACs, 48, 24, kernel_size=(1, 1), stride=(1, 1))
    )
    (1): Sequential(
      0.021 M, 1.918% Params, 0.002 GMac, 1.478% MACs, 
      (0): Conv2d(0.006 M, 0.532% Params, 0.001 GMac, 0.399% MACs, 576, 576, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=576)
      (1): BatchNorm2d(0.001 M, 0.106% Params, 0.0 GMac, 0.080% MACs, 576, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU6(0.0 M, 0.000% Params, 0.0 GMac, 0.040% MACs, )
      (3): Conv2d(0.014 M, 1.280% Params, 0.001 GMac, 0.959% MACs, 576, 24, kernel_size=(1, 1), stride=(1, 1))
    )
    (2): Sequential(
      0.018 M, 1.705% Params, 0.0 GMac, 0.329% MACs, 
      (0): Conv2d(0.005 M, 0.473% Params, 0.0 GMac, 0.089% MACs, 512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=512)
      (1): BatchNorm2d(0.001 M, 0.095% Params, 0.0 GMac, 0.018% MACs, 512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU6(0.0 M, 0.000% Params, 0.0 GMac, 0.009% MACs, )
      (3): Conv2d(0.012 M, 1.138% Params, 0.0 GMac, 0.213% MACs, 512, 24, kernel_size=(1, 1), stride=(1, 1))
    )
    (3): Sequential(
      0.009 M, 0.854% Params, 0.0 GMac, 0.059% MACs, 
      (0): Conv2d(0.003 M, 0.237% Params, 0.0 GMac, 0.016% MACs, 256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=256)
      (1): BatchNorm2d(0.001 M, 0.047% Params, 0.0 GMac, 0.003% MACs, 256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU6(0.0 M, 0.000% Params, 0.0 GMac, 0.002% MACs, )
      (3): Conv2d(0.006 M, 0.570% Params, 0.0 GMac, 0.038% MACs, 256, 24, kernel_size=(1, 1), stride=(1, 1))
    )
    (4): Sequential(
      0.009 M, 0.854% Params, 0.0 GMac, 0.026% MACs, 
      (0): Conv2d(0.003 M, 0.237% Params, 0.0 GMac, 0.007% MACs, 256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=256)
      (1): BatchNorm2d(0.001 M, 0.047% Params, 0.0 GMac, 0.001% MACs, 256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU6(0.0 M, 0.000% Params, 0.0 GMac, 0.001% MACs, )
      (3): Conv2d(0.006 M, 0.570% Params, 0.0 GMac, 0.017% MACs, 256, 24, kernel_size=(1, 1), stride=(1, 1))
    )
    (5): Conv2d(0.002 M, 0.144% Params, 0.0 GMac, 0.001% MACs, 64, 24, kernel_size=(1, 1), stride=(1, 1))
  )
  (regression_headers): ModuleList(
    0.061 M, 5.638% Params, 0.003 GMac, 2.343% MACs, 
    (0): Sequential(
      0.002 M, 0.162% Params, 0.001 GMac, 0.450% MACs, 
      (0): Conv2d(0.0 M, 0.044% Params, 0.0 GMac, 0.120% MACs, 48, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=48)
      (1): BatchNorm2d(0.0 M, 0.009% Params, 0.0 GMac, 0.024% MACs, 48, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU6(0.0 M, 0.000% Params, 0.0 GMac, 0.012% MACs, )
      (3): Conv2d(0.001 M, 0.109% Params, 0.0 GMac, 0.294% MACs, 48, 24, kernel_size=(1, 1), stride=(1, 1))
    )
    (1): Sequential(
      0.021 M, 1.918% Params, 0.002 GMac, 1.478% MACs, 
      (0): Conv2d(0.006 M, 0.532% Params, 0.001 GMac, 0.399% MACs, 576, 576, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=576)
      (1): BatchNorm2d(0.001 M, 0.106% Params, 0.0 GMac, 0.080% MACs, 576, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU6(0.0 M, 0.000% Params, 0.0 GMac, 0.040% MACs, )
      (3): Conv2d(0.014 M, 1.280% Params, 0.001 GMac, 0.959% MACs, 576, 24, kernel_size=(1, 1), stride=(1, 1))
    )
    (2): Sequential(
      0.018 M, 1.705% Params, 0.0 GMac, 0.329% MACs, 
      (0): Conv2d(0.005 M, 0.473% Params, 0.0 GMac, 0.089% MACs, 512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=512)
      (1): BatchNorm2d(0.001 M, 0.095% Params, 0.0 GMac, 0.018% MACs, 512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU6(0.0 M, 0.000% Params, 0.0 GMac, 0.009% MACs, )
      (3): Conv2d(0.012 M, 1.138% Params, 0.0 GMac, 0.213% MACs, 512, 24, kernel_size=(1, 1), stride=(1, 1))
    )
    (3): Sequential(
      0.009 M, 0.854% Params, 0.0 GMac, 0.059% MACs, 
      (0): Conv2d(0.003 M, 0.237% Params, 0.0 GMac, 0.016% MACs, 256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=256)
      (1): BatchNorm2d(0.001 M, 0.047% Params, 0.0 GMac, 0.003% MACs, 256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU6(0.0 M, 0.000% Params, 0.0 GMac, 0.002% MACs, )
      (3): Conv2d(0.006 M, 0.570% Params, 0.0 GMac, 0.038% MACs, 256, 24, kernel_size=(1, 1), stride=(1, 1))
    )
    (4): Sequential(
      0.009 M, 0.854% Params, 0.0 GMac, 0.026% MACs, 
      (0): Conv2d(0.003 M, 0.237% Params, 0.0 GMac, 0.007% MACs, 256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=256)
      (1): BatchNorm2d(0.001 M, 0.047% Params, 0.0 GMac, 0.001% MACs, 256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU6(0.0 M, 0.000% Params, 0.0 GMac, 0.001% MACs, )
      (3): Conv2d(0.006 M, 0.570% Params, 0.0 GMac, 0.017% MACs, 256, 24, kernel_size=(1, 1), stride=(1, 1))
    )
    (5): Conv2d(0.002 M, 0.144% Params, 0.0 GMac, 0.001% MACs, 64, 24, kernel_size=(1, 1), stride=(1, 1))
  )
  (source_layer_add_ons): ModuleList(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, )
)
Computational complexity:       0.14 GMac
Number of parameters:           1.08 M
```

