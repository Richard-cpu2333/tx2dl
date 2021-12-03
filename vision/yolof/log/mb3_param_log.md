```
(base) oem@richard:~/Documents/rqp_proj/tx2dl$ python measure.py 
Warning: module hswish is treated as a zero-op.
Warning: module hsigmoid is treated as a zero-op.
Warning: module SeModule is treated as a zero-op.
Warning: module Block is treated as a zero-op.
Warning: module Bottleneck is treated as a zero-op.
Warning: module DilatedEncoder is treated as a zero-op.
Warning: module Decoder is treated as a zero-op.
Warning: module YOLOF is treated as a zero-op.
YOLOF(
  18.599 M, 100.000% Params, 1.929 GMac, 100.000% MACs, 
  (backbone): Sequential(
    0.488 M, 2.622% Params, 0.117 GMac, 6.063% MACs, 
    (0): Conv2d(0.0 M, 0.002% Params, 0.01 GMac, 0.504% MACs, 3, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
    (1): BatchNorm2d(0.0 M, 0.000% Params, 0.001 GMac, 0.037% MACs, 16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): hswish(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, )
    (3): Block(
      0.001 M, 0.005% Params, 0.01 GMac, 0.499% MACs, 
      (se): SeModule(
        0.0 M, 0.001% Params, 0.0 GMac, 0.005% MACs, 
        (se): Sequential(
          0.0 M, 0.001% Params, 0.0 GMac, 0.005% MACs, 
          (0): AdaptiveAvgPool2d(0.0 M, 0.000% Params, 0.0 GMac, 0.005% MACs, output_size=1)
          (1): Conv2d(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, 16, 4, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (2): BatchNorm2d(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, 4, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (3): ReLU(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, inplace=True)
          (4): Conv2d(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, 4, 16, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (5): BatchNorm2d(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, 16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (6): hsigmoid(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, )
        )
      )
      (conv1): Conv2d(0.0 M, 0.001% Params, 0.006 GMac, 0.299% MACs, 16, 16, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn1): BatchNorm2d(0.0 M, 0.000% Params, 0.001 GMac, 0.037% MACs, 16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (nolinear1): ReLU(0.0 M, 0.000% Params, 0.0 GMac, 0.023% MACs, inplace=True)
      (conv2): Conv2d(0.0 M, 0.001% Params, 0.001 GMac, 0.042% MACs, 16, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=16, bias=False)
      (bn2): BatchNorm2d(0.0 M, 0.000% Params, 0.0 GMac, 0.009% MACs, 16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (nolinear2): ReLU(0.0 M, 0.000% Params, 0.0 GMac, 0.023% MACs, inplace=True)
      (conv3): Conv2d(0.0 M, 0.001% Params, 0.001 GMac, 0.075% MACs, 16, 16, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn3): BatchNorm2d(0.0 M, 0.000% Params, 0.0 GMac, 0.009% MACs, 16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (shortcut): Sequential(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, )
    )
    (4): Block(
      0.004 M, 0.021% Params, 0.012 GMac, 0.597% MACs, 
      (conv1): Conv2d(0.001 M, 0.006% Params, 0.006 GMac, 0.336% MACs, 16, 72, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn1): BatchNorm2d(0.0 M, 0.001% Params, 0.001 GMac, 0.042% MACs, 72, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (nolinear1): ReLU(0.0 M, 0.000% Params, 0.001 GMac, 0.026% MACs, inplace=True)
      (conv2): Conv2d(0.001 M, 0.003% Params, 0.001 GMac, 0.049% MACs, 72, 72, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=72, bias=False)
      (bn2): BatchNorm2d(0.0 M, 0.001% Params, 0.0 GMac, 0.011% MACs, 72, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (nolinear2): ReLU(0.0 M, 0.000% Params, 0.001 GMac, 0.026% MACs, inplace=True)
      (conv3): Conv2d(0.002 M, 0.009% Params, 0.002 GMac, 0.129% MACs, 72, 24, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn3): BatchNorm2d(0.0 M, 0.000% Params, 0.0 GMac, 0.004% MACs, 24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (shortcut): Sequential(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, )
    )
    (5): Block(
      0.005 M, 0.029% Params, 0.008 GMac, 0.419% MACs, 
      (conv1): Conv2d(0.002 M, 0.011% Params, 0.003 GMac, 0.158% MACs, 24, 88, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn1): BatchNorm2d(0.0 M, 0.001% Params, 0.0 GMac, 0.013% MACs, 88, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (nolinear1): ReLU(0.0 M, 0.000% Params, 0.0 GMac, 0.013% MACs, inplace=True)
      (conv2): Conv2d(0.001 M, 0.004% Params, 0.001 GMac, 0.059% MACs, 88, 88, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=88, bias=False)
      (bn2): BatchNorm2d(0.0 M, 0.001% Params, 0.0 GMac, 0.013% MACs, 88, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (nolinear2): ReLU(0.0 M, 0.000% Params, 0.0 GMac, 0.013% MACs, inplace=True)
      (conv3): Conv2d(0.002 M, 0.011% Params, 0.003 GMac, 0.158% MACs, 88, 24, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn3): BatchNorm2d(0.0 M, 0.000% Params, 0.0 GMac, 0.004% MACs, 24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (shortcut): Sequential(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, )
    )
    (6): Block(
      0.01 M, 0.053% Params, 0.006 GMac, 0.310% MACs, 
      (se): SeModule(
        0.001 M, 0.005% Params, 0.0 GMac, 0.001% MACs, 
        (se): Sequential(
          0.001 M, 0.005% Params, 0.0 GMac, 0.001% MACs, 
          (0): AdaptiveAvgPool2d(0.0 M, 0.000% Params, 0.0 GMac, 0.001% MACs, output_size=1)
          (1): Conv2d(0.0 M, 0.002% Params, 0.0 GMac, 0.000% MACs, 40, 10, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (2): BatchNorm2d(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, 10, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (3): ReLU(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, inplace=True)
          (4): Conv2d(0.0 M, 0.002% Params, 0.0 GMac, 0.000% MACs, 10, 40, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (5): BatchNorm2d(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, 40, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (6): hsigmoid(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, )
        )
      )
      (conv1): Conv2d(0.002 M, 0.012% Params, 0.003 GMac, 0.172% MACs, 24, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn1): BatchNorm2d(0.0 M, 0.001% Params, 0.0 GMac, 0.014% MACs, 96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (nolinear1): hswish(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, )
      (conv2): Conv2d(0.002 M, 0.013% Params, 0.001 GMac, 0.045% MACs, 96, 96, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), groups=96, bias=False)
      (bn2): BatchNorm2d(0.0 M, 0.001% Params, 0.0 GMac, 0.004% MACs, 96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (nolinear2): hswish(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, )
      (conv3): Conv2d(0.004 M, 0.021% Params, 0.001 GMac, 0.072% MACs, 96, 40, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn3): BatchNorm2d(0.0 M, 0.000% Params, 0.0 GMac, 0.001% MACs, 40, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (shortcut): Sequential(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, )
    )
    (7): Block(
      0.027 M, 0.146% Params, 0.009 GMac, 0.492% MACs, 
      (se): SeModule(
        0.001 M, 0.005% Params, 0.0 GMac, 0.001% MACs, 
        (se): Sequential(
          0.001 M, 0.005% Params, 0.0 GMac, 0.001% MACs, 
          (0): AdaptiveAvgPool2d(0.0 M, 0.000% Params, 0.0 GMac, 0.001% MACs, output_size=1)
          (1): Conv2d(0.0 M, 0.002% Params, 0.0 GMac, 0.000% MACs, 40, 10, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (2): BatchNorm2d(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, 10, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (3): ReLU(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, inplace=True)
          (4): Conv2d(0.0 M, 0.002% Params, 0.0 GMac, 0.000% MACs, 10, 40, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (5): BatchNorm2d(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, 40, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (6): hsigmoid(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, )
        )
      )
      (conv1): Conv2d(0.01 M, 0.052% Params, 0.003 GMac, 0.180% MACs, 40, 240, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn1): BatchNorm2d(0.0 M, 0.003% Params, 0.0 GMac, 0.009% MACs, 240, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (nolinear1): hswish(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, )
      (conv2): Conv2d(0.006 M, 0.032% Params, 0.002 GMac, 0.112% MACs, 240, 240, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=240, bias=False)
      (bn2): BatchNorm2d(0.0 M, 0.003% Params, 0.0 GMac, 0.009% MACs, 240, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (nolinear2): hswish(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, )
      (conv3): Conv2d(0.01 M, 0.052% Params, 0.003 GMac, 0.180% MACs, 240, 40, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn3): BatchNorm2d(0.0 M, 0.000% Params, 0.0 GMac, 0.001% MACs, 40, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (shortcut): Sequential(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, )
    )
    (8): Block(
      0.027 M, 0.146% Params, 0.009 GMac, 0.492% MACs, 
      (se): SeModule(
        0.001 M, 0.005% Params, 0.0 GMac, 0.001% MACs, 
        (se): Sequential(
          0.001 M, 0.005% Params, 0.0 GMac, 0.001% MACs, 
          (0): AdaptiveAvgPool2d(0.0 M, 0.000% Params, 0.0 GMac, 0.001% MACs, output_size=1)
          (1): Conv2d(0.0 M, 0.002% Params, 0.0 GMac, 0.000% MACs, 40, 10, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (2): BatchNorm2d(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, 10, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (3): ReLU(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, inplace=True)
          (4): Conv2d(0.0 M, 0.002% Params, 0.0 GMac, 0.000% MACs, 10, 40, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (5): BatchNorm2d(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, 40, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (6): hsigmoid(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, )
        )
      )
      (conv1): Conv2d(0.01 M, 0.052% Params, 0.003 GMac, 0.180% MACs, 40, 240, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn1): BatchNorm2d(0.0 M, 0.003% Params, 0.0 GMac, 0.009% MACs, 240, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (nolinear1): hswish(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, )
      (conv2): Conv2d(0.006 M, 0.032% Params, 0.002 GMac, 0.112% MACs, 240, 240, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=240, bias=False)
      (bn2): BatchNorm2d(0.0 M, 0.003% Params, 0.0 GMac, 0.009% MACs, 240, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (nolinear2): hswish(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, )
      (conv3): Conv2d(0.01 M, 0.052% Params, 0.003 GMac, 0.180% MACs, 240, 40, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn3): BatchNorm2d(0.0 M, 0.000% Params, 0.0 GMac, 0.001% MACs, 40, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (shortcut): Sequential(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, )
    )
    (9): Block(
      0.017 M, 0.094% Params, 0.006 GMac, 0.303% MACs, 
      (se): SeModule(
        0.001 M, 0.007% Params, 0.0 GMac, 0.001% MACs, 
        (se): Sequential(
          0.001 M, 0.007% Params, 0.0 GMac, 0.001% MACs, 
          (0): AdaptiveAvgPool2d(0.0 M, 0.000% Params, 0.0 GMac, 0.001% MACs, output_size=1)
          (1): Conv2d(0.001 M, 0.003% Params, 0.0 GMac, 0.000% MACs, 48, 12, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (2): BatchNorm2d(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, 12, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (3): ReLU(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, inplace=True)
          (4): Conv2d(0.001 M, 0.003% Params, 0.0 GMac, 0.000% MACs, 12, 48, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (5): BatchNorm2d(0.0 M, 0.001% Params, 0.0 GMac, 0.000% MACs, 48, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (6): hsigmoid(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, )
        )
      )
      (conv1): Conv2d(0.005 M, 0.026% Params, 0.002 GMac, 0.090% MACs, 40, 120, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn1): BatchNorm2d(0.0 M, 0.001% Params, 0.0 GMac, 0.004% MACs, 120, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (nolinear1): hswish(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, )
      (conv2): Conv2d(0.003 M, 0.016% Params, 0.001 GMac, 0.056% MACs, 120, 120, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=120, bias=False)
      (bn2): BatchNorm2d(0.0 M, 0.001% Params, 0.0 GMac, 0.004% MACs, 120, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (nolinear2): hswish(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, )
      (conv3): Conv2d(0.006 M, 0.031% Params, 0.002 GMac, 0.108% MACs, 120, 48, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn3): BatchNorm2d(0.0 M, 0.001% Params, 0.0 GMac, 0.002% MACs, 48, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (shortcut): Sequential(
        0.002 M, 0.011% Params, 0.001 GMac, 0.038% MACs, 
        (0): Conv2d(0.002 M, 0.010% Params, 0.001 GMac, 0.036% MACs, 40, 48, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (1): BatchNorm2d(0.0 M, 0.001% Params, 0.0 GMac, 0.002% MACs, 48, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (10): Block(
      0.019 M, 0.104% Params, 0.007 GMac, 0.340% MACs, 
      (se): SeModule(
        0.001 M, 0.007% Params, 0.0 GMac, 0.001% MACs, 
        (se): Sequential(
          0.001 M, 0.007% Params, 0.0 GMac, 0.001% MACs, 
          (0): AdaptiveAvgPool2d(0.0 M, 0.000% Params, 0.0 GMac, 0.001% MACs, output_size=1)
          (1): Conv2d(0.001 M, 0.003% Params, 0.0 GMac, 0.000% MACs, 48, 12, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (2): BatchNorm2d(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, 12, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (3): ReLU(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, inplace=True)
          (4): Conv2d(0.001 M, 0.003% Params, 0.0 GMac, 0.000% MACs, 12, 48, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (5): BatchNorm2d(0.0 M, 0.001% Params, 0.0 GMac, 0.000% MACs, 48, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (6): hsigmoid(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, )
        )
      )
      (conv1): Conv2d(0.007 M, 0.037% Params, 0.002 GMac, 0.129% MACs, 48, 144, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn1): BatchNorm2d(0.0 M, 0.002% Params, 0.0 GMac, 0.005% MACs, 144, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (nolinear1): hswish(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, )
      (conv2): Conv2d(0.004 M, 0.019% Params, 0.001 GMac, 0.067% MACs, 144, 144, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=144, bias=False)
      (bn2): BatchNorm2d(0.0 M, 0.002% Params, 0.0 GMac, 0.005% MACs, 144, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (nolinear2): hswish(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, )
      (conv3): Conv2d(0.007 M, 0.037% Params, 0.002 GMac, 0.129% MACs, 144, 48, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn3): BatchNorm2d(0.0 M, 0.001% Params, 0.0 GMac, 0.002% MACs, 48, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (shortcut): Sequential(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, )
    )
    (11): Block(
      0.055 M, 0.295% Params, 0.009 GMac, 0.455% MACs, 
      (se): SeModule(
        0.005 M, 0.026% Params, 0.0 GMac, 0.001% MACs, 
        (se): Sequential(
          0.005 M, 0.026% Params, 0.0 GMac, 0.001% MACs, 
          (0): AdaptiveAvgPool2d(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, output_size=1)
          (1): Conv2d(0.002 M, 0.012% Params, 0.0 GMac, 0.000% MACs, 96, 24, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (2): BatchNorm2d(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, 24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (3): ReLU(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, inplace=True)
          (4): Conv2d(0.002 M, 0.012% Params, 0.0 GMac, 0.000% MACs, 24, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (5): BatchNorm2d(0.0 M, 0.001% Params, 0.0 GMac, 0.000% MACs, 96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (6): hsigmoid(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, )
        )
      )
      (conv1): Conv2d(0.014 M, 0.074% Params, 0.005 GMac, 0.259% MACs, 48, 288, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn1): BatchNorm2d(0.001 M, 0.003% Params, 0.0 GMac, 0.011% MACs, 288, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (nolinear1): hswish(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, )
      (conv2): Conv2d(0.007 M, 0.039% Params, 0.001 GMac, 0.037% MACs, 288, 288, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), groups=288, bias=False)
      (bn2): BatchNorm2d(0.001 M, 0.003% Params, 0.0 GMac, 0.003% MACs, 288, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (nolinear2): hswish(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, )
      (conv3): Conv2d(0.028 M, 0.149% Params, 0.003 GMac, 0.143% MACs, 288, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn3): BatchNorm2d(0.0 M, 0.001% Params, 0.0 GMac, 0.001% MACs, 96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (shortcut): Sequential(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, )
    )
    (12): Block(
      0.132 M, 0.712% Params, 0.013 GMac, 0.662% MACs, 
      (se): SeModule(
        0.005 M, 0.026% Params, 0.0 GMac, 0.001% MACs, 
        (se): Sequential(
          0.005 M, 0.026% Params, 0.0 GMac, 0.001% MACs, 
          (0): AdaptiveAvgPool2d(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, output_size=1)
          (1): Conv2d(0.002 M, 0.012% Params, 0.0 GMac, 0.000% MACs, 96, 24, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (2): BatchNorm2d(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, 24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (3): ReLU(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, inplace=True)
          (4): Conv2d(0.002 M, 0.012% Params, 0.0 GMac, 0.000% MACs, 24, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (5): BatchNorm2d(0.0 M, 0.001% Params, 0.0 GMac, 0.000% MACs, 96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (6): hsigmoid(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, )
        )
      )
      (conv1): Conv2d(0.055 M, 0.297% Params, 0.006 GMac, 0.287% MACs, 96, 576, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn1): BatchNorm2d(0.001 M, 0.006% Params, 0.0 GMac, 0.006% MACs, 576, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (nolinear1): hswish(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, )
      (conv2): Conv2d(0.014 M, 0.077% Params, 0.001 GMac, 0.075% MACs, 576, 576, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=576, bias=False)
      (bn2): BatchNorm2d(0.001 M, 0.006% Params, 0.0 GMac, 0.006% MACs, 576, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (nolinear2): hswish(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, )
      (conv3): Conv2d(0.055 M, 0.297% Params, 0.006 GMac, 0.287% MACs, 576, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn3): BatchNorm2d(0.0 M, 0.001% Params, 0.0 GMac, 0.001% MACs, 96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (shortcut): Sequential(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, )
    )
    (13): Block(
      0.132 M, 0.712% Params, 0.013 GMac, 0.662% MACs, 
      (se): SeModule(
        0.005 M, 0.026% Params, 0.0 GMac, 0.001% MACs, 
        (se): Sequential(
          0.005 M, 0.026% Params, 0.0 GMac, 0.001% MACs, 
          (0): AdaptiveAvgPool2d(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, output_size=1)
          (1): Conv2d(0.002 M, 0.012% Params, 0.0 GMac, 0.000% MACs, 96, 24, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (2): BatchNorm2d(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, 24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (3): ReLU(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, inplace=True)
          (4): Conv2d(0.002 M, 0.012% Params, 0.0 GMac, 0.000% MACs, 24, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (5): BatchNorm2d(0.0 M, 0.001% Params, 0.0 GMac, 0.000% MACs, 96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (6): hsigmoid(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, )
        )
      )
      (conv1): Conv2d(0.055 M, 0.297% Params, 0.006 GMac, 0.287% MACs, 96, 576, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn1): BatchNorm2d(0.001 M, 0.006% Params, 0.0 GMac, 0.006% MACs, 576, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (nolinear1): hswish(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, )
      (conv2): Conv2d(0.014 M, 0.077% Params, 0.001 GMac, 0.075% MACs, 576, 576, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=576, bias=False)
      (bn2): BatchNorm2d(0.001 M, 0.006% Params, 0.0 GMac, 0.006% MACs, 576, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (nolinear2): hswish(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, )
      (conv3): Conv2d(0.055 M, 0.297% Params, 0.006 GMac, 0.287% MACs, 576, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (bn3): BatchNorm2d(0.0 M, 0.001% Params, 0.0 GMac, 0.001% MACs, 96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (shortcut): Sequential(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, )
    )
    (14): Conv2d(0.055 M, 0.297% Params, 0.006 GMac, 0.287% MACs, 96, 576, kernel_size=(1, 1), stride=(1, 1), bias=False)
    (15): BatchNorm2d(0.001 M, 0.006% Params, 0.0 GMac, 0.006% MACs, 576, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (16): hswish(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, )
  )
  (encoder): DilatedEncoder(
    3.781 M, 20.327% Params, 0.378 GMac, 19.618% MACs, 
    (lateral_conv): Conv2d(0.295 M, 1.588% Params, 0.03 GMac, 1.532% MACs, 576, 512, kernel_size=(1, 1), stride=(1, 1))
    (lateral_norm): BatchNorm2d(0.001 M, 0.006% Params, 0.0 GMac, 0.005% MACs, 512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (fpn_conv): Conv2d(2.36 M, 12.688% Params, 0.236 GMac, 12.235% MACs, 512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (fpn_norm): BatchNorm2d(0.001 M, 0.006% Params, 0.0 GMac, 0.005% MACs, 512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (dilated_encoder_blocks): Sequential(
      1.123 M, 6.040% Params, 0.113 GMac, 5.840% MACs, 
      (0): Bottleneck(
        0.281 M, 1.510% Params, 0.028 GMac, 1.460% MACs, 
        (conv1): Sequential(
          0.066 M, 0.354% Params, 0.007 GMac, 0.342% MACs, 
          (0): Conv2d(0.066 M, 0.353% Params, 0.007 GMac, 0.340% MACs, 512, 128, kernel_size=(1, 1), stride=(1, 1))
          (1): BatchNorm2d(0.0 M, 0.001% Params, 0.0 GMac, 0.001% MACs, 128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): LeakyReLU(0.0 M, 0.000% Params, 0.0 GMac, 0.001% MACs, negative_slope=0.01)
        )
        (conv2): Sequential(
          0.148 M, 0.795% Params, 0.015 GMac, 0.767% MACs, 
          (0): Conv2d(0.148 M, 0.793% Params, 0.015 GMac, 0.765% MACs, 128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(2, 2), dilation=(2, 2))
          (1): BatchNorm2d(0.0 M, 0.001% Params, 0.0 GMac, 0.001% MACs, 128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): LeakyReLU(0.0 M, 0.000% Params, 0.0 GMac, 0.001% MACs, negative_slope=0.01)
        )
        (conv3): Sequential(
          0.067 M, 0.361% Params, 0.007 GMac, 0.350% MACs, 
          (0): Conv2d(0.066 M, 0.355% Params, 0.007 GMac, 0.342% MACs, 128, 512, kernel_size=(1, 1), stride=(1, 1))
          (1): BatchNorm2d(0.001 M, 0.006% Params, 0.0 GMac, 0.005% MACs, 512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): LeakyReLU(0.0 M, 0.000% Params, 0.0 GMac, 0.003% MACs, negative_slope=0.01)
        )
      )
      (1): Bottleneck(
        0.281 M, 1.510% Params, 0.028 GMac, 1.460% MACs, 
        (conv1): Sequential(
          0.066 M, 0.354% Params, 0.007 GMac, 0.342% MACs, 
          (0): Conv2d(0.066 M, 0.353% Params, 0.007 GMac, 0.340% MACs, 512, 128, kernel_size=(1, 1), stride=(1, 1))
          (1): BatchNorm2d(0.0 M, 0.001% Params, 0.0 GMac, 0.001% MACs, 128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): LeakyReLU(0.0 M, 0.000% Params, 0.0 GMac, 0.001% MACs, negative_slope=0.01)
        )
        (conv2): Sequential(
          0.148 M, 0.795% Params, 0.015 GMac, 0.767% MACs, 
          (0): Conv2d(0.148 M, 0.793% Params, 0.015 GMac, 0.765% MACs, 128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(4, 4), dilation=(4, 4))
          (1): BatchNorm2d(0.0 M, 0.001% Params, 0.0 GMac, 0.001% MACs, 128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): LeakyReLU(0.0 M, 0.000% Params, 0.0 GMac, 0.001% MACs, negative_slope=0.01)
        )
        (conv3): Sequential(
          0.067 M, 0.361% Params, 0.007 GMac, 0.350% MACs, 
          (0): Conv2d(0.066 M, 0.355% Params, 0.007 GMac, 0.342% MACs, 128, 512, kernel_size=(1, 1), stride=(1, 1))
          (1): BatchNorm2d(0.001 M, 0.006% Params, 0.0 GMac, 0.005% MACs, 512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): LeakyReLU(0.0 M, 0.000% Params, 0.0 GMac, 0.003% MACs, negative_slope=0.01)
        )
      )
      (2): Bottleneck(
        0.281 M, 1.510% Params, 0.028 GMac, 1.460% MACs, 
        (conv1): Sequential(
          0.066 M, 0.354% Params, 0.007 GMac, 0.342% MACs, 
          (0): Conv2d(0.066 M, 0.353% Params, 0.007 GMac, 0.340% MACs, 512, 128, kernel_size=(1, 1), stride=(1, 1))
          (1): BatchNorm2d(0.0 M, 0.001% Params, 0.0 GMac, 0.001% MACs, 128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): LeakyReLU(0.0 M, 0.000% Params, 0.0 GMac, 0.001% MACs, negative_slope=0.01)
        )
        (conv2): Sequential(
          0.148 M, 0.795% Params, 0.015 GMac, 0.767% MACs, 
          (0): Conv2d(0.148 M, 0.793% Params, 0.015 GMac, 0.765% MACs, 128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(6, 6), dilation=(6, 6))
          (1): BatchNorm2d(0.0 M, 0.001% Params, 0.0 GMac, 0.001% MACs, 128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): LeakyReLU(0.0 M, 0.000% Params, 0.0 GMac, 0.001% MACs, negative_slope=0.01)
        )
        (conv3): Sequential(
          0.067 M, 0.361% Params, 0.007 GMac, 0.350% MACs, 
          (0): Conv2d(0.066 M, 0.355% Params, 0.007 GMac, 0.342% MACs, 128, 512, kernel_size=(1, 1), stride=(1, 1))
          (1): BatchNorm2d(0.001 M, 0.006% Params, 0.0 GMac, 0.005% MACs, 512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): LeakyReLU(0.0 M, 0.000% Params, 0.0 GMac, 0.003% MACs, negative_slope=0.01)
        )
      )
      (3): Bottleneck(
        0.281 M, 1.510% Params, 0.028 GMac, 1.460% MACs, 
        (conv1): Sequential(
          0.066 M, 0.354% Params, 0.007 GMac, 0.342% MACs, 
          (0): Conv2d(0.066 M, 0.353% Params, 0.007 GMac, 0.340% MACs, 512, 128, kernel_size=(1, 1), stride=(1, 1))
          (1): BatchNorm2d(0.0 M, 0.001% Params, 0.0 GMac, 0.001% MACs, 128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): LeakyReLU(0.0 M, 0.000% Params, 0.0 GMac, 0.001% MACs, negative_slope=0.01)
        )
        (conv2): Sequential(
          0.148 M, 0.795% Params, 0.015 GMac, 0.767% MACs, 
          (0): Conv2d(0.148 M, 0.793% Params, 0.015 GMac, 0.765% MACs, 128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(8, 8), dilation=(8, 8))
          (1): BatchNorm2d(0.0 M, 0.001% Params, 0.0 GMac, 0.001% MACs, 128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): LeakyReLU(0.0 M, 0.000% Params, 0.0 GMac, 0.001% MACs, negative_slope=0.01)
        )
        (conv3): Sequential(
          0.067 M, 0.361% Params, 0.007 GMac, 0.350% MACs, 
          (0): Conv2d(0.066 M, 0.355% Params, 0.007 GMac, 0.342% MACs, 128, 512, kernel_size=(1, 1), stride=(1, 1))
          (1): BatchNorm2d(0.001 M, 0.006% Params, 0.0 GMac, 0.005% MACs, 512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): LeakyReLU(0.0 M, 0.000% Params, 0.0 GMac, 0.003% MACs, negative_slope=0.01)
        )
      )
    )
  )
  (decoder): Decoder(
    14.331 M, 77.051% Params, 1.433 GMac, 74.319% MACs, 
    (cls_subnet): Sequential(
      4.722 M, 25.386% Params, 0.472 GMac, 24.486% MACs, 
      (0): Conv2d(2.36 M, 12.688% Params, 0.236 GMac, 12.235% MACs, 512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (1): BatchNorm2d(0.001 M, 0.006% Params, 0.0 GMac, 0.005% MACs, 512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): LeakyReLU(0.0 M, 0.000% Params, 0.0 GMac, 0.003% MACs, negative_slope=0.01)
      (3): Conv2d(2.36 M, 12.688% Params, 0.236 GMac, 12.235% MACs, 512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (4): BatchNorm2d(0.001 M, 0.006% Params, 0.0 GMac, 0.005% MACs, 512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (5): LeakyReLU(0.0 M, 0.000% Params, 0.0 GMac, 0.003% MACs, negative_slope=0.01)
    )
    (bbox_subnet): Sequential(
      9.443 M, 50.773% Params, 0.945 GMac, 48.972% MACs, 
      (0): Conv2d(2.36 M, 12.688% Params, 0.236 GMac, 12.235% MACs, 512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (1): BatchNorm2d(0.001 M, 0.006% Params, 0.0 GMac, 0.005% MACs, 512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): LeakyReLU(0.0 M, 0.000% Params, 0.0 GMac, 0.003% MACs, negative_slope=0.01)
      (3): Conv2d(2.36 M, 12.688% Params, 0.236 GMac, 12.235% MACs, 512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (4): BatchNorm2d(0.001 M, 0.006% Params, 0.0 GMac, 0.005% MACs, 512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (5): LeakyReLU(0.0 M, 0.000% Params, 0.0 GMac, 0.003% MACs, negative_slope=0.01)
      (6): Conv2d(2.36 M, 12.688% Params, 0.236 GMac, 12.235% MACs, 512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (7): BatchNorm2d(0.001 M, 0.006% Params, 0.0 GMac, 0.005% MACs, 512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (8): LeakyReLU(0.0 M, 0.000% Params, 0.0 GMac, 0.003% MACs, negative_slope=0.01)
      (9): Conv2d(2.36 M, 12.688% Params, 0.236 GMac, 12.235% MACs, 512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (10): BatchNorm2d(0.001 M, 0.006% Params, 0.0 GMac, 0.005% MACs, 512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (11): LeakyReLU(0.0 M, 0.000% Params, 0.0 GMac, 0.003% MACs, negative_slope=0.01)
    )
    (cls_score): Conv2d(0.074 M, 0.396% Params, 0.007 GMac, 0.382% MACs, 512, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (bbox_pred): Conv2d(0.074 M, 0.396% Params, 0.007 GMac, 0.382% MACs, 512, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (object_pred): Conv2d(0.018 M, 0.099% Params, 0.002 GMac, 0.096% MACs, 512, 4, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  )
)
Computational complexity:       1.93 GMac
Number of parameters:           18.6 M
```

