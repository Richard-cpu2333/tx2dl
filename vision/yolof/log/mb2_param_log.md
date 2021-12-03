```
(base) oem@richard:~/Documents/rqp_proj/tx2dl$ python measure.py 
Warning: module InvertedResidual is treated as a zero-op.
Warning: module Bottleneck is treated as a zero-op.
Warning: module DilatedEncoder is treated as a zero-op.
Warning: module Decoder is treated as a zero-op.
Warning: module YOLOF is treated as a zero-op.
YOLOF(
  20.696 M, 100.000% Params, 2.446 GMac, 100.000% MACs, 
  (backbone): Sequential(
    2.224 M, 10.746% Params, 0.599 GMac, 24.470% MACs, 
    (0): Sequential(
      0.001 M, 0.004% Params, 0.022 GMac, 0.883% MACs, 
      (0): Conv2d(0.001 M, 0.004% Params, 0.019 GMac, 0.795% MACs, 3, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
      (1): BatchNorm2d(0.0 M, 0.000% Params, 0.001 GMac, 0.059% MACs, 32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU6(0.0 M, 0.000% Params, 0.001 GMac, 0.029% MACs, inplace=True)
    )
    (1): InvertedResidual(
      0.001 M, 0.004% Params, 0.021 GMac, 0.853% MACs, 
      (conv): Sequential(
        0.001 M, 0.004% Params, 0.021 GMac, 0.853% MACs, 
        (0): Conv2d(0.0 M, 0.001% Params, 0.006 GMac, 0.265% MACs, 32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=32, bias=False)
        (1): BatchNorm2d(0.0 M, 0.000% Params, 0.001 GMac, 0.059% MACs, 32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU6(0.0 M, 0.000% Params, 0.001 GMac, 0.029% MACs, inplace=True)
        (3): Conv2d(0.001 M, 0.002% Params, 0.012 GMac, 0.471% MACs, 32, 16, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (4): BatchNorm2d(0.0 M, 0.000% Params, 0.001 GMac, 0.029% MACs, 16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (2): InvertedResidual(
      0.005 M, 0.025% Params, 0.061 GMac, 2.483% MACs, 
      (conv): Sequential(
        0.005 M, 0.025% Params, 0.061 GMac, 2.483% MACs, 
        (0): Conv2d(0.002 M, 0.007% Params, 0.035 GMac, 1.413% MACs, 16, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (1): BatchNorm2d(0.0 M, 0.001% Params, 0.004 GMac, 0.177% MACs, 96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU6(0.0 M, 0.000% Params, 0.002 GMac, 0.088% MACs, inplace=True)
        (3): Conv2d(0.001 M, 0.004% Params, 0.005 GMac, 0.199% MACs, 96, 96, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=96, bias=False)
        (4): BatchNorm2d(0.0 M, 0.001% Params, 0.001 GMac, 0.044% MACs, 96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (5): ReLU6(0.0 M, 0.000% Params, 0.001 GMac, 0.022% MACs, inplace=True)
        (6): Conv2d(0.002 M, 0.011% Params, 0.013 GMac, 0.530% MACs, 96, 24, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (7): BatchNorm2d(0.0 M, 0.000% Params, 0.0 GMac, 0.011% MACs, 24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (3): InvertedResidual(
      0.009 M, 0.043% Params, 0.051 GMac, 2.097% MACs, 
      (conv): Sequential(
        0.009 M, 0.043% Params, 0.051 GMac, 2.097% MACs, 
        (0): Conv2d(0.003 M, 0.017% Params, 0.019 GMac, 0.795% MACs, 24, 144, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (1): BatchNorm2d(0.0 M, 0.001% Params, 0.002 GMac, 0.066% MACs, 144, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU6(0.0 M, 0.000% Params, 0.001 GMac, 0.033% MACs, inplace=True)
        (3): Conv2d(0.001 M, 0.006% Params, 0.007 GMac, 0.298% MACs, 144, 144, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=144, bias=False)
        (4): BatchNorm2d(0.0 M, 0.001% Params, 0.002 GMac, 0.066% MACs, 144, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (5): ReLU6(0.0 M, 0.000% Params, 0.001 GMac, 0.033% MACs, inplace=True)
        (6): Conv2d(0.003 M, 0.017% Params, 0.019 GMac, 0.795% MACs, 144, 24, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (7): BatchNorm2d(0.0 M, 0.000% Params, 0.0 GMac, 0.011% MACs, 24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (4): InvertedResidual(
      0.01 M, 0.048% Params, 0.031 GMac, 1.272% MACs, 
      (conv): Sequential(
        0.01 M, 0.048% Params, 0.031 GMac, 1.272% MACs, 
        (0): Conv2d(0.003 M, 0.017% Params, 0.019 GMac, 0.795% MACs, 24, 144, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (1): BatchNorm2d(0.0 M, 0.001% Params, 0.002 GMac, 0.066% MACs, 144, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU6(0.0 M, 0.000% Params, 0.001 GMac, 0.033% MACs, inplace=True)
        (3): Conv2d(0.001 M, 0.006% Params, 0.002 GMac, 0.076% MACs, 144, 144, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=144, bias=False)
        (4): BatchNorm2d(0.0 M, 0.001% Params, 0.0 GMac, 0.017% MACs, 144, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (5): ReLU6(0.0 M, 0.000% Params, 0.0 GMac, 0.008% MACs, inplace=True)
        (6): Conv2d(0.005 M, 0.022% Params, 0.007 GMac, 0.272% MACs, 144, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (7): BatchNorm2d(0.0 M, 0.000% Params, 0.0 GMac, 0.004% MACs, 32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (5): InvertedResidual(
      0.015 M, 0.072% Params, 0.022 GMac, 0.899% MACs, 
      (conv): Sequential(
        0.015 M, 0.072% Params, 0.022 GMac, 0.899% MACs, 
        (0): Conv2d(0.006 M, 0.030% Params, 0.009 GMac, 0.363% MACs, 32, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (1): BatchNorm2d(0.0 M, 0.002% Params, 0.001 GMac, 0.023% MACs, 192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU6(0.0 M, 0.000% Params, 0.0 GMac, 0.011% MACs, inplace=True)
        (3): Conv2d(0.002 M, 0.008% Params, 0.002 GMac, 0.102% MACs, 192, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=192, bias=False)
        (4): BatchNorm2d(0.0 M, 0.002% Params, 0.001 GMac, 0.023% MACs, 192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (5): ReLU6(0.0 M, 0.000% Params, 0.0 GMac, 0.011% MACs, inplace=True)
        (6): Conv2d(0.006 M, 0.030% Params, 0.009 GMac, 0.363% MACs, 192, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (7): BatchNorm2d(0.0 M, 0.000% Params, 0.0 GMac, 0.004% MACs, 32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (6): InvertedResidual(
      0.015 M, 0.072% Params, 0.022 GMac, 0.899% MACs, 
      (conv): Sequential(
        0.015 M, 0.072% Params, 0.022 GMac, 0.899% MACs, 
        (0): Conv2d(0.006 M, 0.030% Params, 0.009 GMac, 0.363% MACs, 32, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (1): BatchNorm2d(0.0 M, 0.002% Params, 0.001 GMac, 0.023% MACs, 192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU6(0.0 M, 0.000% Params, 0.0 GMac, 0.011% MACs, inplace=True)
        (3): Conv2d(0.002 M, 0.008% Params, 0.002 GMac, 0.102% MACs, 192, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=192, bias=False)
        (4): BatchNorm2d(0.0 M, 0.002% Params, 0.001 GMac, 0.023% MACs, 192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (5): ReLU6(0.0 M, 0.000% Params, 0.0 GMac, 0.011% MACs, inplace=True)
        (6): Conv2d(0.006 M, 0.030% Params, 0.009 GMac, 0.363% MACs, 192, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (7): BatchNorm2d(0.0 M, 0.000% Params, 0.0 GMac, 0.004% MACs, 32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (7): InvertedResidual(
      0.021 M, 0.102% Params, 0.015 GMac, 0.614% MACs, 
      (conv): Sequential(
        0.021 M, 0.102% Params, 0.015 GMac, 0.614% MACs, 
        (0): Conv2d(0.006 M, 0.030% Params, 0.009 GMac, 0.363% MACs, 32, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (1): BatchNorm2d(0.0 M, 0.002% Params, 0.001 GMac, 0.023% MACs, 192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU6(0.0 M, 0.000% Params, 0.0 GMac, 0.011% MACs, inplace=True)
        (3): Conv2d(0.002 M, 0.008% Params, 0.001 GMac, 0.025% MACs, 192, 192, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=192, bias=False)
        (4): BatchNorm2d(0.0 M, 0.002% Params, 0.0 GMac, 0.006% MACs, 192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (5): ReLU6(0.0 M, 0.000% Params, 0.0 GMac, 0.003% MACs, inplace=True)
        (6): Conv2d(0.012 M, 0.059% Params, 0.004 GMac, 0.181% MACs, 192, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (7): BatchNorm2d(0.0 M, 0.001% Params, 0.0 GMac, 0.002% MACs, 64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (8): InvertedResidual(
      0.054 M, 0.262% Params, 0.02 GMac, 0.812% MACs, 
      (conv): Sequential(
        0.054 M, 0.262% Params, 0.02 GMac, 0.812% MACs, 
        (0): Conv2d(0.025 M, 0.119% Params, 0.009 GMac, 0.363% MACs, 64, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (1): BatchNorm2d(0.001 M, 0.004% Params, 0.0 GMac, 0.011% MACs, 384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU6(0.0 M, 0.000% Params, 0.0 GMac, 0.006% MACs, inplace=True)
        (3): Conv2d(0.003 M, 0.017% Params, 0.001 GMac, 0.051% MACs, 384, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=384, bias=False)
        (4): BatchNorm2d(0.001 M, 0.004% Params, 0.0 GMac, 0.011% MACs, 384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (5): ReLU6(0.0 M, 0.000% Params, 0.0 GMac, 0.006% MACs, inplace=True)
        (6): Conv2d(0.025 M, 0.119% Params, 0.009 GMac, 0.363% MACs, 384, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (7): BatchNorm2d(0.0 M, 0.001% Params, 0.0 GMac, 0.002% MACs, 64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (9): InvertedResidual(
      0.054 M, 0.262% Params, 0.02 GMac, 0.812% MACs, 
      (conv): Sequential(
        0.054 M, 0.262% Params, 0.02 GMac, 0.812% MACs, 
        (0): Conv2d(0.025 M, 0.119% Params, 0.009 GMac, 0.363% MACs, 64, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (1): BatchNorm2d(0.001 M, 0.004% Params, 0.0 GMac, 0.011% MACs, 384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU6(0.0 M, 0.000% Params, 0.0 GMac, 0.006% MACs, inplace=True)
        (3): Conv2d(0.003 M, 0.017% Params, 0.001 GMac, 0.051% MACs, 384, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=384, bias=False)
        (4): BatchNorm2d(0.001 M, 0.004% Params, 0.0 GMac, 0.011% MACs, 384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (5): ReLU6(0.0 M, 0.000% Params, 0.0 GMac, 0.006% MACs, inplace=True)
        (6): Conv2d(0.025 M, 0.119% Params, 0.009 GMac, 0.363% MACs, 384, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (7): BatchNorm2d(0.0 M, 0.001% Params, 0.0 GMac, 0.002% MACs, 64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (10): InvertedResidual(
      0.054 M, 0.262% Params, 0.02 GMac, 0.812% MACs, 
      (conv): Sequential(
        0.054 M, 0.262% Params, 0.02 GMac, 0.812% MACs, 
        (0): Conv2d(0.025 M, 0.119% Params, 0.009 GMac, 0.363% MACs, 64, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (1): BatchNorm2d(0.001 M, 0.004% Params, 0.0 GMac, 0.011% MACs, 384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU6(0.0 M, 0.000% Params, 0.0 GMac, 0.006% MACs, inplace=True)
        (3): Conv2d(0.003 M, 0.017% Params, 0.001 GMac, 0.051% MACs, 384, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=384, bias=False)
        (4): BatchNorm2d(0.001 M, 0.004% Params, 0.0 GMac, 0.011% MACs, 384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (5): ReLU6(0.0 M, 0.000% Params, 0.0 GMac, 0.006% MACs, inplace=True)
        (6): Conv2d(0.025 M, 0.119% Params, 0.009 GMac, 0.363% MACs, 384, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (7): BatchNorm2d(0.0 M, 0.001% Params, 0.0 GMac, 0.002% MACs, 64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (11): InvertedResidual(
      0.067 M, 0.322% Params, 0.024 GMac, 0.994% MACs, 
      (conv): Sequential(
        0.067 M, 0.322% Params, 0.024 GMac, 0.994% MACs, 
        (0): Conv2d(0.025 M, 0.119% Params, 0.009 GMac, 0.363% MACs, 64, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (1): BatchNorm2d(0.001 M, 0.004% Params, 0.0 GMac, 0.011% MACs, 384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU6(0.0 M, 0.000% Params, 0.0 GMac, 0.006% MACs, inplace=True)
        (3): Conv2d(0.003 M, 0.017% Params, 0.001 GMac, 0.051% MACs, 384, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=384, bias=False)
        (4): BatchNorm2d(0.001 M, 0.004% Params, 0.0 GMac, 0.011% MACs, 384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (5): ReLU6(0.0 M, 0.000% Params, 0.0 GMac, 0.006% MACs, inplace=True)
        (6): Conv2d(0.037 M, 0.178% Params, 0.013 GMac, 0.544% MACs, 384, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (7): BatchNorm2d(0.0 M, 0.001% Params, 0.0 GMac, 0.003% MACs, 96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (12): InvertedResidual(
      0.118 M, 0.571% Params, 0.043 GMac, 1.762% MACs, 
      (conv): Sequential(
        0.118 M, 0.571% Params, 0.043 GMac, 1.762% MACs, 
        (0): Conv2d(0.055 M, 0.267% Params, 0.02 GMac, 0.816% MACs, 96, 576, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (1): BatchNorm2d(0.001 M, 0.006% Params, 0.0 GMac, 0.017% MACs, 576, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU6(0.0 M, 0.000% Params, 0.0 GMac, 0.008% MACs, inplace=True)
        (3): Conv2d(0.005 M, 0.025% Params, 0.002 GMac, 0.076% MACs, 576, 576, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=576, bias=False)
        (4): BatchNorm2d(0.001 M, 0.006% Params, 0.0 GMac, 0.017% MACs, 576, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (5): ReLU6(0.0 M, 0.000% Params, 0.0 GMac, 0.008% MACs, inplace=True)
        (6): Conv2d(0.055 M, 0.267% Params, 0.02 GMac, 0.816% MACs, 576, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (7): BatchNorm2d(0.0 M, 0.001% Params, 0.0 GMac, 0.003% MACs, 96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (13): InvertedResidual(
      0.118 M, 0.571% Params, 0.043 GMac, 1.762% MACs, 
      (conv): Sequential(
        0.118 M, 0.571% Params, 0.043 GMac, 1.762% MACs, 
        (0): Conv2d(0.055 M, 0.267% Params, 0.02 GMac, 0.816% MACs, 96, 576, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (1): BatchNorm2d(0.001 M, 0.006% Params, 0.0 GMac, 0.017% MACs, 576, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU6(0.0 M, 0.000% Params, 0.0 GMac, 0.008% MACs, inplace=True)
        (3): Conv2d(0.005 M, 0.025% Params, 0.002 GMac, 0.076% MACs, 576, 576, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=576, bias=False)
        (4): BatchNorm2d(0.001 M, 0.006% Params, 0.0 GMac, 0.017% MACs, 576, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (5): ReLU6(0.0 M, 0.000% Params, 0.0 GMac, 0.008% MACs, inplace=True)
        (6): Conv2d(0.055 M, 0.267% Params, 0.02 GMac, 0.816% MACs, 576, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (7): BatchNorm2d(0.0 M, 0.001% Params, 0.0 GMac, 0.003% MACs, 96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (14): InvertedResidual(
      0.155 M, 0.750% Params, 0.031 GMac, 1.248% MACs, 
      (conv): Sequential(
        0.155 M, 0.750% Params, 0.031 GMac, 1.248% MACs, 
        (0): Conv2d(0.055 M, 0.267% Params, 0.02 GMac, 0.816% MACs, 96, 576, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (1): BatchNorm2d(0.001 M, 0.006% Params, 0.0 GMac, 0.017% MACs, 576, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU6(0.0 M, 0.000% Params, 0.0 GMac, 0.008% MACs, inplace=True)
        (3): Conv2d(0.005 M, 0.025% Params, 0.001 GMac, 0.021% MACs, 576, 576, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=576, bias=False)
        (4): BatchNorm2d(0.001 M, 0.006% Params, 0.0 GMac, 0.005% MACs, 576, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (5): ReLU6(0.0 M, 0.000% Params, 0.0 GMac, 0.002% MACs, inplace=True)
        (6): Conv2d(0.092 M, 0.445% Params, 0.009 GMac, 0.377% MACs, 576, 160, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (7): BatchNorm2d(0.0 M, 0.002% Params, 0.0 GMac, 0.001% MACs, 160, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (15): InvertedResidual(
      0.32 M, 1.546% Params, 0.032 GMac, 1.316% MACs, 
      (conv): Sequential(
        0.32 M, 1.546% Params, 0.032 GMac, 1.316% MACs, 
        (0): Conv2d(0.154 M, 0.742% Params, 0.015 GMac, 0.628% MACs, 160, 960, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (1): BatchNorm2d(0.002 M, 0.009% Params, 0.0 GMac, 0.008% MACs, 960, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU6(0.0 M, 0.000% Params, 0.0 GMac, 0.004% MACs, inplace=True)
        (3): Conv2d(0.009 M, 0.042% Params, 0.001 GMac, 0.035% MACs, 960, 960, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=960, bias=False)
        (4): BatchNorm2d(0.002 M, 0.009% Params, 0.0 GMac, 0.008% MACs, 960, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (5): ReLU6(0.0 M, 0.000% Params, 0.0 GMac, 0.004% MACs, inplace=True)
        (6): Conv2d(0.154 M, 0.742% Params, 0.015 GMac, 0.628% MACs, 960, 160, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (7): BatchNorm2d(0.0 M, 0.002% Params, 0.0 GMac, 0.001% MACs, 160, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (16): InvertedResidual(
      0.32 M, 1.546% Params, 0.032 GMac, 1.316% MACs, 
      (conv): Sequential(
        0.32 M, 1.546% Params, 0.032 GMac, 1.316% MACs, 
        (0): Conv2d(0.154 M, 0.742% Params, 0.015 GMac, 0.628% MACs, 160, 960, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (1): BatchNorm2d(0.002 M, 0.009% Params, 0.0 GMac, 0.008% MACs, 960, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU6(0.0 M, 0.000% Params, 0.0 GMac, 0.004% MACs, inplace=True)
        (3): Conv2d(0.009 M, 0.042% Params, 0.001 GMac, 0.035% MACs, 960, 960, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=960, bias=False)
        (4): BatchNorm2d(0.002 M, 0.009% Params, 0.0 GMac, 0.008% MACs, 960, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (5): ReLU6(0.0 M, 0.000% Params, 0.0 GMac, 0.004% MACs, inplace=True)
        (6): Conv2d(0.154 M, 0.742% Params, 0.015 GMac, 0.628% MACs, 960, 160, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (7): BatchNorm2d(0.0 M, 0.002% Params, 0.0 GMac, 0.001% MACs, 160, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (17): InvertedResidual(
      0.474 M, 2.290% Params, 0.048 GMac, 1.945% MACs, 
      (conv): Sequential(
        0.474 M, 2.290% Params, 0.048 GMac, 1.945% MACs, 
        (0): Conv2d(0.154 M, 0.742% Params, 0.015 GMac, 0.628% MACs, 160, 960, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (1): BatchNorm2d(0.002 M, 0.009% Params, 0.0 GMac, 0.008% MACs, 960, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU6(0.0 M, 0.000% Params, 0.0 GMac, 0.004% MACs, inplace=True)
        (3): Conv2d(0.009 M, 0.042% Params, 0.001 GMac, 0.035% MACs, 960, 960, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=960, bias=False)
        (4): BatchNorm2d(0.002 M, 0.009% Params, 0.0 GMac, 0.008% MACs, 960, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (5): ReLU6(0.0 M, 0.000% Params, 0.0 GMac, 0.004% MACs, inplace=True)
        (6): Conv2d(0.307 M, 1.484% Params, 0.031 GMac, 1.256% MACs, 960, 320, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (7): BatchNorm2d(0.001 M, 0.003% Params, 0.0 GMac, 0.003% MACs, 320, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (18): Sequential(
      0.412 M, 1.992% Params, 0.041 GMac, 1.690% MACs, 
      (0): Conv2d(0.41 M, 1.979% Params, 0.041 GMac, 1.674% MACs, 320, 1280, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (1): BatchNorm2d(0.003 M, 0.012% Params, 0.0 GMac, 0.010% MACs, 1280, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU6(0.0 M, 0.000% Params, 0.0 GMac, 0.005% MACs, inplace=True)
    )
  )
  (encoder): DilatedEncoder(
    4.141 M, 20.009% Params, 0.414 GMac, 16.939% MACs, 
    (lateral_conv): Conv2d(0.656 M, 3.169% Params, 0.066 GMac, 2.681% MACs, 1280, 512, kernel_size=(1, 1), stride=(1, 1))
    (lateral_norm): BatchNorm2d(0.001 M, 0.005% Params, 0.0 GMac, 0.004% MACs, 512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (fpn_conv): Conv2d(2.36 M, 11.402% Params, 0.236 GMac, 9.646% MACs, 512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (fpn_norm): BatchNorm2d(0.001 M, 0.005% Params, 0.0 GMac, 0.004% MACs, 512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (dilated_encoder_blocks): Sequential(
      1.123 M, 5.428% Params, 0.113 GMac, 4.604% MACs, 
      (0): Bottleneck(
        0.281 M, 1.357% Params, 0.028 GMac, 1.151% MACs, 
        (conv1): Sequential(
          0.066 M, 0.319% Params, 0.007 GMac, 0.270% MACs, 
          (0): Conv2d(0.066 M, 0.317% Params, 0.007 GMac, 0.268% MACs, 512, 128, kernel_size=(1, 1), stride=(1, 1))
          (1): BatchNorm2d(0.0 M, 0.001% Params, 0.0 GMac, 0.001% MACs, 128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): LeakyReLU(0.0 M, 0.000% Params, 0.0 GMac, 0.001% MACs, negative_slope=0.01)
        )
        (conv2): Sequential(
          0.148 M, 0.714% Params, 0.015 GMac, 0.605% MACs, 
          (0): Conv2d(0.148 M, 0.713% Params, 0.015 GMac, 0.603% MACs, 128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(2, 2), dilation=(2, 2))
          (1): BatchNorm2d(0.0 M, 0.001% Params, 0.0 GMac, 0.001% MACs, 128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): LeakyReLU(0.0 M, 0.000% Params, 0.0 GMac, 0.001% MACs, negative_slope=0.01)
        )
        (conv3): Sequential(
          0.067 M, 0.324% Params, 0.007 GMac, 0.276% MACs, 
          (0): Conv2d(0.066 M, 0.319% Params, 0.007 GMac, 0.270% MACs, 128, 512, kernel_size=(1, 1), stride=(1, 1))
          (1): BatchNorm2d(0.001 M, 0.005% Params, 0.0 GMac, 0.004% MACs, 512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): LeakyReLU(0.0 M, 0.000% Params, 0.0 GMac, 0.002% MACs, negative_slope=0.01)
        )
      )
      (1): Bottleneck(
        0.281 M, 1.357% Params, 0.028 GMac, 1.151% MACs, 
        (conv1): Sequential(
          0.066 M, 0.319% Params, 0.007 GMac, 0.270% MACs, 
          (0): Conv2d(0.066 M, 0.317% Params, 0.007 GMac, 0.268% MACs, 512, 128, kernel_size=(1, 1), stride=(1, 1))
          (1): BatchNorm2d(0.0 M, 0.001% Params, 0.0 GMac, 0.001% MACs, 128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): LeakyReLU(0.0 M, 0.000% Params, 0.0 GMac, 0.001% MACs, negative_slope=0.01)
        )
        (conv2): Sequential(
          0.148 M, 0.714% Params, 0.015 GMac, 0.605% MACs, 
          (0): Conv2d(0.148 M, 0.713% Params, 0.015 GMac, 0.603% MACs, 128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(4, 4), dilation=(4, 4))
          (1): BatchNorm2d(0.0 M, 0.001% Params, 0.0 GMac, 0.001% MACs, 128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): LeakyReLU(0.0 M, 0.000% Params, 0.0 GMac, 0.001% MACs, negative_slope=0.01)
        )
        (conv3): Sequential(
          0.067 M, 0.324% Params, 0.007 GMac, 0.276% MACs, 
          (0): Conv2d(0.066 M, 0.319% Params, 0.007 GMac, 0.270% MACs, 128, 512, kernel_size=(1, 1), stride=(1, 1))
          (1): BatchNorm2d(0.001 M, 0.005% Params, 0.0 GMac, 0.004% MACs, 512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): LeakyReLU(0.0 M, 0.000% Params, 0.0 GMac, 0.002% MACs, negative_slope=0.01)
        )
      )
      (2): Bottleneck(
        0.281 M, 1.357% Params, 0.028 GMac, 1.151% MACs, 
        (conv1): Sequential(
          0.066 M, 0.319% Params, 0.007 GMac, 0.270% MACs, 
          (0): Conv2d(0.066 M, 0.317% Params, 0.007 GMac, 0.268% MACs, 512, 128, kernel_size=(1, 1), stride=(1, 1))
          (1): BatchNorm2d(0.0 M, 0.001% Params, 0.0 GMac, 0.001% MACs, 128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): LeakyReLU(0.0 M, 0.000% Params, 0.0 GMac, 0.001% MACs, negative_slope=0.01)
        )
        (conv2): Sequential(
          0.148 M, 0.714% Params, 0.015 GMac, 0.605% MACs, 
          (0): Conv2d(0.148 M, 0.713% Params, 0.015 GMac, 0.603% MACs, 128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(6, 6), dilation=(6, 6))
          (1): BatchNorm2d(0.0 M, 0.001% Params, 0.0 GMac, 0.001% MACs, 128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): LeakyReLU(0.0 M, 0.000% Params, 0.0 GMac, 0.001% MACs, negative_slope=0.01)
        )
        (conv3): Sequential(
          0.067 M, 0.324% Params, 0.007 GMac, 0.276% MACs, 
          (0): Conv2d(0.066 M, 0.319% Params, 0.007 GMac, 0.270% MACs, 128, 512, kernel_size=(1, 1), stride=(1, 1))
          (1): BatchNorm2d(0.001 M, 0.005% Params, 0.0 GMac, 0.004% MACs, 512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): LeakyReLU(0.0 M, 0.000% Params, 0.0 GMac, 0.002% MACs, negative_slope=0.01)
        )
      )
      (3): Bottleneck(
        0.281 M, 1.357% Params, 0.028 GMac, 1.151% MACs, 
        (conv1): Sequential(
          0.066 M, 0.319% Params, 0.007 GMac, 0.270% MACs, 
          (0): Conv2d(0.066 M, 0.317% Params, 0.007 GMac, 0.268% MACs, 512, 128, kernel_size=(1, 1), stride=(1, 1))
          (1): BatchNorm2d(0.0 M, 0.001% Params, 0.0 GMac, 0.001% MACs, 128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): LeakyReLU(0.0 M, 0.000% Params, 0.0 GMac, 0.001% MACs, negative_slope=0.01)
        )
        (conv2): Sequential(
          0.148 M, 0.714% Params, 0.015 GMac, 0.605% MACs, 
          (0): Conv2d(0.148 M, 0.713% Params, 0.015 GMac, 0.603% MACs, 128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(8, 8), dilation=(8, 8))
          (1): BatchNorm2d(0.0 M, 0.001% Params, 0.0 GMac, 0.001% MACs, 128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): LeakyReLU(0.0 M, 0.000% Params, 0.0 GMac, 0.001% MACs, negative_slope=0.01)
        )
        (conv3): Sequential(
          0.067 M, 0.324% Params, 0.007 GMac, 0.276% MACs, 
          (0): Conv2d(0.066 M, 0.319% Params, 0.007 GMac, 0.270% MACs, 128, 512, kernel_size=(1, 1), stride=(1, 1))
          (1): BatchNorm2d(0.001 M, 0.005% Params, 0.0 GMac, 0.004% MACs, 512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): LeakyReLU(0.0 M, 0.000% Params, 0.0 GMac, 0.002% MACs, negative_slope=0.01)
        )
      )
    )
  )
  (decoder): Decoder(
    14.331 M, 69.245% Params, 1.433 GMac, 58.591% MACs, 
    (cls_subnet): Sequential(
      4.722 M, 22.815% Params, 0.472 GMac, 19.304% MACs, 
      (0): Conv2d(2.36 M, 11.402% Params, 0.236 GMac, 9.646% MACs, 512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (1): BatchNorm2d(0.001 M, 0.005% Params, 0.0 GMac, 0.004% MACs, 512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): LeakyReLU(0.0 M, 0.000% Params, 0.0 GMac, 0.002% MACs, negative_slope=0.01)
      (3): Conv2d(2.36 M, 11.402% Params, 0.236 GMac, 9.646% MACs, 512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (4): BatchNorm2d(0.001 M, 0.005% Params, 0.0 GMac, 0.004% MACs, 512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (5): LeakyReLU(0.0 M, 0.000% Params, 0.0 GMac, 0.002% MACs, negative_slope=0.01)
    )
    (bbox_subnet): Sequential(
      9.443 M, 45.629% Params, 0.945 GMac, 38.608% MACs, 
      (0): Conv2d(2.36 M, 11.402% Params, 0.236 GMac, 9.646% MACs, 512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (1): BatchNorm2d(0.001 M, 0.005% Params, 0.0 GMac, 0.004% MACs, 512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): LeakyReLU(0.0 M, 0.000% Params, 0.0 GMac, 0.002% MACs, negative_slope=0.01)
      (3): Conv2d(2.36 M, 11.402% Params, 0.236 GMac, 9.646% MACs, 512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (4): BatchNorm2d(0.001 M, 0.005% Params, 0.0 GMac, 0.004% MACs, 512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (5): LeakyReLU(0.0 M, 0.000% Params, 0.0 GMac, 0.002% MACs, negative_slope=0.01)
      (6): Conv2d(2.36 M, 11.402% Params, 0.236 GMac, 9.646% MACs, 512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (7): BatchNorm2d(0.001 M, 0.005% Params, 0.0 GMac, 0.004% MACs, 512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (8): LeakyReLU(0.0 M, 0.000% Params, 0.0 GMac, 0.002% MACs, negative_slope=0.01)
      (9): Conv2d(2.36 M, 11.402% Params, 0.236 GMac, 9.646% MACs, 512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (10): BatchNorm2d(0.001 M, 0.005% Params, 0.0 GMac, 0.004% MACs, 512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (11): LeakyReLU(0.0 M, 0.000% Params, 0.0 GMac, 0.002% MACs, negative_slope=0.01)
    )
    (cls_score): Conv2d(0.074 M, 0.356% Params, 0.007 GMac, 0.301% MACs, 512, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (bbox_pred): Conv2d(0.074 M, 0.356% Params, 0.007 GMac, 0.301% MACs, 512, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (object_pred): Conv2d(0.018 M, 0.089% Params, 0.002 GMac, 0.075% MACs, 512, 4, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  )
)
Computational complexity:       2.45 GMac
Number of parameters:           20.7 M
```

