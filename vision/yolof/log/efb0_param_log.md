```
(base) oem@richard:~/Documents/rqp_proj/tx2dl$ python measure.py 
Warning: module SiLU is treated as a zero-op.
Warning: module ConvNormActivation is treated as a zero-op.
Warning: module Sigmoid is treated as a zero-op.
Warning: module SqueezeExcitation is treated as a zero-op.
Warning: module StochasticDepth is treated as a zero-op.
Warning: module MBConv is treated as a zero-op.
Warning: module Bottleneck is treated as a zero-op.
Warning: module DilatedEncoder is treated as a zero-op.
Warning: module Decoder is treated as a zero-op.
Warning: module YOLOF is treated as a zero-op.
YOLOF(
  22.48 M, 100.000% Params, 2.605 GMac, 100.000% MACs, 
  (backbone): Sequential(
    4.008 M, 17.828% Params, 0.758 GMac, 29.076% MACs, 
    (0): ConvNormActivation(
      0.001 M, 0.004% Params, 0.021 GMac, 0.801% MACs, 
      (0): Conv2d(0.001 M, 0.004% Params, 0.019 GMac, 0.746% MACs, 3, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
      (1): BatchNorm2d(0.0 M, 0.000% Params, 0.001 GMac, 0.055% MACs, 32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): SiLU(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, inplace=True)
    )
    (1): Sequential(
      0.001 M, 0.006% Params, 0.021 GMac, 0.801% MACs, 
      (0): MBConv(
        0.001 M, 0.006% Params, 0.021 GMac, 0.801% MACs, 
        (block): Sequential(
          0.001 M, 0.006% Params, 0.021 GMac, 0.801% MACs, 
          (0): ConvNormActivation(
            0.0 M, 0.002% Params, 0.008 GMac, 0.304% MACs, 
            (0): Conv2d(0.0 M, 0.001% Params, 0.006 GMac, 0.249% MACs, 32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=32, bias=False)
            (1): BatchNorm2d(0.0 M, 0.000% Params, 0.001 GMac, 0.055% MACs, 32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): SiLU(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, inplace=True)
          )
          (1): SqueezeExcitation(
            0.001 M, 0.002% Params, 0.001 GMac, 0.028% MACs, 
            (avgpool): AdaptiveAvgPool2d(0.0 M, 0.000% Params, 0.001 GMac, 0.028% MACs, output_size=1)
            (fc1): Conv2d(0.0 M, 0.001% Params, 0.0 GMac, 0.000% MACs, 32, 8, kernel_size=(1, 1), stride=(1, 1))
            (fc2): Conv2d(0.0 M, 0.001% Params, 0.0 GMac, 0.000% MACs, 8, 32, kernel_size=(1, 1), stride=(1, 1))
            (activation): SiLU(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, inplace=True)
            (scale_activation): Sigmoid(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, )
          )
          (2): ConvNormActivation(
            0.001 M, 0.002% Params, 0.012 GMac, 0.470% MACs, 
            (0): Conv2d(0.001 M, 0.002% Params, 0.012 GMac, 0.442% MACs, 32, 16, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (1): BatchNorm2d(0.0 M, 0.000% Params, 0.001 GMac, 0.028% MACs, 16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (stochastic_depth): StochasticDepth(p=0.0, mode=row)
      )
    )
    (2): Sequential(
      0.017 M, 0.074% Params, 0.109 GMac, 4.187% MACs, 
      (0): MBConv(
        0.006 M, 0.027% Params, 0.059 GMac, 2.249% MACs, 
        (block): Sequential(
          0.006 M, 0.027% Params, 0.059 GMac, 2.249% MACs, 
          (0): ConvNormActivation(
            0.002 M, 0.008% Params, 0.039 GMac, 1.492% MACs, 
            (0): Conv2d(0.002 M, 0.007% Params, 0.035 GMac, 1.327% MACs, 16, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (1): BatchNorm2d(0.0 M, 0.001% Params, 0.004 GMac, 0.166% MACs, 96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): SiLU(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, inplace=True)
          )
          (1): ConvNormActivation(
            0.001 M, 0.005% Params, 0.006 GMac, 0.228% MACs, 
            (0): Conv2d(0.001 M, 0.004% Params, 0.005 GMac, 0.187% MACs, 96, 96, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=96, bias=False)
            (1): BatchNorm2d(0.0 M, 0.001% Params, 0.001 GMac, 0.041% MACs, 96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): SiLU(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, inplace=True)
          )
          (2): SqueezeExcitation(
            0.001 M, 0.004% Params, 0.001 GMac, 0.021% MACs, 
            (avgpool): AdaptiveAvgPool2d(0.0 M, 0.000% Params, 0.001 GMac, 0.021% MACs, output_size=1)
            (fc1): Conv2d(0.0 M, 0.002% Params, 0.0 GMac, 0.000% MACs, 96, 4, kernel_size=(1, 1), stride=(1, 1))
            (fc2): Conv2d(0.0 M, 0.002% Params, 0.0 GMac, 0.000% MACs, 4, 96, kernel_size=(1, 1), stride=(1, 1))
            (activation): SiLU(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, inplace=True)
            (scale_activation): Sigmoid(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, )
          )
          (3): ConvNormActivation(
            0.002 M, 0.010% Params, 0.013 GMac, 0.508% MACs, 
            (0): Conv2d(0.002 M, 0.010% Params, 0.013 GMac, 0.497% MACs, 96, 24, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (1): BatchNorm2d(0.0 M, 0.000% Params, 0.0 GMac, 0.010% MACs, 24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (stochastic_depth): StochasticDepth(p=0.0125, mode=row)
      )
      (1): MBConv(
        0.011 M, 0.048% Params, 0.05 GMac, 1.938% MACs, 
        (block): Sequential(
          0.011 M, 0.048% Params, 0.05 GMac, 1.938% MACs, 
          (0): ConvNormActivation(
            0.004 M, 0.017% Params, 0.021 GMac, 0.808% MACs, 
            (0): Conv2d(0.003 M, 0.015% Params, 0.019 GMac, 0.746% MACs, 24, 144, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (1): BatchNorm2d(0.0 M, 0.001% Params, 0.002 GMac, 0.062% MACs, 144, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): SiLU(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, inplace=True)
          )
          (1): ConvNormActivation(
            0.002 M, 0.007% Params, 0.009 GMac, 0.342% MACs, 
            (0): Conv2d(0.001 M, 0.006% Params, 0.007 GMac, 0.280% MACs, 144, 144, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=144, bias=False)
            (1): BatchNorm2d(0.0 M, 0.001% Params, 0.002 GMac, 0.062% MACs, 144, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): SiLU(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, inplace=True)
          )
          (2): SqueezeExcitation(
            0.002 M, 0.008% Params, 0.001 GMac, 0.031% MACs, 
            (avgpool): AdaptiveAvgPool2d(0.0 M, 0.000% Params, 0.001 GMac, 0.031% MACs, output_size=1)
            (fc1): Conv2d(0.001 M, 0.004% Params, 0.0 GMac, 0.000% MACs, 144, 6, kernel_size=(1, 1), stride=(1, 1))
            (fc2): Conv2d(0.001 M, 0.004% Params, 0.0 GMac, 0.000% MACs, 6, 144, kernel_size=(1, 1), stride=(1, 1))
            (activation): SiLU(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, inplace=True)
            (scale_activation): Sigmoid(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, )
          )
          (3): ConvNormActivation(
            0.004 M, 0.016% Params, 0.02 GMac, 0.757% MACs, 
            (0): Conv2d(0.003 M, 0.015% Params, 0.019 GMac, 0.746% MACs, 144, 24, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (1): BatchNorm2d(0.0 M, 0.000% Params, 0.0 GMac, 0.010% MACs, 24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (stochastic_depth): StochasticDepth(p=0.025, mode=row)
      )
    )
    (3): Sequential(
      0.047 M, 0.207% Params, 0.074 GMac, 2.823% MACs, 
      (0): MBConv(
        0.015 M, 0.068% Params, 0.035 GMac, 1.356% MACs, 
        (block): Sequential(
          0.015 M, 0.068% Params, 0.035 GMac, 1.356% MACs, 
          (0): ConvNormActivation(
            0.004 M, 0.017% Params, 0.021 GMac, 0.808% MACs, 
            (0): Conv2d(0.003 M, 0.015% Params, 0.019 GMac, 0.746% MACs, 24, 144, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (1): BatchNorm2d(0.0 M, 0.001% Params, 0.002 GMac, 0.062% MACs, 144, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): SiLU(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, inplace=True)
          )
          (1): ConvNormActivation(
            0.004 M, 0.017% Params, 0.006 GMac, 0.215% MACs, 
            (0): Conv2d(0.004 M, 0.016% Params, 0.005 GMac, 0.200% MACs, 144, 144, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), groups=144, bias=False)
            (1): BatchNorm2d(0.0 M, 0.001% Params, 0.0 GMac, 0.016% MACs, 144, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): SiLU(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, inplace=True)
          )
          (2): SqueezeExcitation(
            0.002 M, 0.008% Params, 0.0 GMac, 0.008% MACs, 
            (avgpool): AdaptiveAvgPool2d(0.0 M, 0.000% Params, 0.0 GMac, 0.008% MACs, output_size=1)
            (fc1): Conv2d(0.001 M, 0.004% Params, 0.0 GMac, 0.000% MACs, 144, 6, kernel_size=(1, 1), stride=(1, 1))
            (fc2): Conv2d(0.001 M, 0.004% Params, 0.0 GMac, 0.000% MACs, 6, 144, kernel_size=(1, 1), stride=(1, 1))
            (activation): SiLU(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, inplace=True)
            (scale_activation): Sigmoid(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, )
          )
          (3): ConvNormActivation(
            0.006 M, 0.026% Params, 0.008 GMac, 0.324% MACs, 
            (0): Conv2d(0.006 M, 0.026% Params, 0.008 GMac, 0.319% MACs, 144, 40, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (1): BatchNorm2d(0.0 M, 0.000% Params, 0.0 GMac, 0.004% MACs, 40, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (stochastic_depth): StochasticDepth(p=0.037500000000000006, mode=row)
      )
      (1): MBConv(
        0.031 M, 0.139% Params, 0.038 GMac, 1.468% MACs, 
        (block): Sequential(
          0.031 M, 0.139% Params, 0.038 GMac, 1.468% MACs, 
          (0): ConvNormActivation(
            0.01 M, 0.045% Params, 0.015 GMac, 0.559% MACs, 
            (0): Conv2d(0.01 M, 0.043% Params, 0.014 GMac, 0.532% MACs, 40, 240, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (1): BatchNorm2d(0.0 M, 0.002% Params, 0.001 GMac, 0.027% MACs, 240, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): SiLU(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, inplace=True)
          )
          (1): ConvNormActivation(
            0.006 M, 0.029% Params, 0.009 GMac, 0.359% MACs, 
            (0): Conv2d(0.006 M, 0.027% Params, 0.009 GMac, 0.333% MACs, 240, 240, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=240, bias=False)
            (1): BatchNorm2d(0.0 M, 0.002% Params, 0.001 GMac, 0.027% MACs, 240, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): SiLU(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, inplace=True)
          )
          (2): SqueezeExcitation(
            0.005 M, 0.022% Params, 0.0 GMac, 0.013% MACs, 
            (avgpool): AdaptiveAvgPool2d(0.0 M, 0.000% Params, 0.0 GMac, 0.013% MACs, output_size=1)
            (fc1): Conv2d(0.002 M, 0.011% Params, 0.0 GMac, 0.000% MACs, 240, 10, kernel_size=(1, 1), stride=(1, 1))
            (fc2): Conv2d(0.003 M, 0.012% Params, 0.0 GMac, 0.000% MACs, 10, 240, kernel_size=(1, 1), stride=(1, 1))
            (activation): SiLU(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, inplace=True)
            (scale_activation): Sigmoid(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, )
          )
          (3): ConvNormActivation(
            0.01 M, 0.043% Params, 0.014 GMac, 0.537% MACs, 
            (0): Conv2d(0.01 M, 0.043% Params, 0.014 GMac, 0.532% MACs, 240, 40, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (1): BatchNorm2d(0.0 M, 0.000% Params, 0.0 GMac, 0.004% MACs, 40, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (stochastic_depth): StochasticDepth(p=0.05, mode=row)
      )
    )
    (4): Sequential(
      0.243 M, 1.081% Params, 0.083 GMac, 3.188% MACs, 
      (0): MBConv(
        0.037 M, 0.165% Params, 0.023 GMac, 0.867% MACs, 
        (block): Sequential(
          0.037 M, 0.165% Params, 0.023 GMac, 0.867% MACs, 
          (0): ConvNormActivation(
            0.01 M, 0.045% Params, 0.015 GMac, 0.559% MACs, 
            (0): Conv2d(0.01 M, 0.043% Params, 0.014 GMac, 0.532% MACs, 40, 240, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (1): BatchNorm2d(0.0 M, 0.002% Params, 0.001 GMac, 0.027% MACs, 240, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): SiLU(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, inplace=True)
          )
          (1): ConvNormActivation(
            0.003 M, 0.012% Params, 0.001 GMac, 0.037% MACs, 
            (0): Conv2d(0.002 M, 0.010% Params, 0.001 GMac, 0.030% MACs, 240, 240, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=240, bias=False)
            (1): BatchNorm2d(0.0 M, 0.002% Params, 0.0 GMac, 0.007% MACs, 240, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): SiLU(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, inplace=True)
          )
          (2): SqueezeExcitation(
            0.005 M, 0.022% Params, 0.0 GMac, 0.004% MACs, 
            (avgpool): AdaptiveAvgPool2d(0.0 M, 0.000% Params, 0.0 GMac, 0.003% MACs, output_size=1)
            (fc1): Conv2d(0.002 M, 0.011% Params, 0.0 GMac, 0.000% MACs, 240, 10, kernel_size=(1, 1), stride=(1, 1))
            (fc2): Conv2d(0.003 M, 0.012% Params, 0.0 GMac, 0.000% MACs, 10, 240, kernel_size=(1, 1), stride=(1, 1))
            (activation): SiLU(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, inplace=True)
            (scale_activation): Sigmoid(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, )
          )
          (3): ConvNormActivation(
            0.019 M, 0.086% Params, 0.007 GMac, 0.268% MACs, 
            (0): Conv2d(0.019 M, 0.085% Params, 0.007 GMac, 0.266% MACs, 240, 80, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (1): BatchNorm2d(0.0 M, 0.001% Params, 0.0 GMac, 0.002% MACs, 80, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (stochastic_depth): StochasticDepth(p=0.0625, mode=row)
      )
      (1): MBConv(
        0.103 M, 0.458% Params, 0.03 GMac, 1.160% MACs, 
        (block): Sequential(
          0.103 M, 0.458% Params, 0.03 GMac, 1.160% MACs, 
          (0): ConvNormActivation(
            0.039 M, 0.175% Params, 0.014 GMac, 0.545% MACs, 
            (0): Conv2d(0.038 M, 0.171% Params, 0.014 GMac, 0.532% MACs, 80, 480, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (1): BatchNorm2d(0.001 M, 0.004% Params, 0.0 GMac, 0.013% MACs, 480, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): SiLU(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, inplace=True)
          )
          (1): ConvNormActivation(
            0.005 M, 0.023% Params, 0.002 GMac, 0.073% MACs, 
            (0): Conv2d(0.004 M, 0.019% Params, 0.002 GMac, 0.060% MACs, 480, 480, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=480, bias=False)
            (1): BatchNorm2d(0.001 M, 0.004% Params, 0.0 GMac, 0.013% MACs, 480, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): SiLU(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, inplace=True)
          )
          (2): SqueezeExcitation(
            0.02 M, 0.088% Params, 0.0 GMac, 0.007% MACs, 
            (avgpool): AdaptiveAvgPool2d(0.0 M, 0.000% Params, 0.0 GMac, 0.007% MACs, output_size=1)
            (fc1): Conv2d(0.01 M, 0.043% Params, 0.0 GMac, 0.000% MACs, 480, 20, kernel_size=(1, 1), stride=(1, 1))
            (fc2): Conv2d(0.01 M, 0.045% Params, 0.0 GMac, 0.000% MACs, 20, 480, kernel_size=(1, 1), stride=(1, 1))
            (activation): SiLU(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, inplace=True)
            (scale_activation): Sigmoid(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, )
          )
          (3): ConvNormActivation(
            0.039 M, 0.172% Params, 0.014 GMac, 0.534% MACs, 
            (0): Conv2d(0.038 M, 0.171% Params, 0.014 GMac, 0.532% MACs, 480, 80, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (1): BatchNorm2d(0.0 M, 0.001% Params, 0.0 GMac, 0.002% MACs, 80, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (stochastic_depth): StochasticDepth(p=0.07500000000000001, mode=row)
      )
      (2): MBConv(
        0.103 M, 0.458% Params, 0.03 GMac, 1.160% MACs, 
        (block): Sequential(
          0.103 M, 0.458% Params, 0.03 GMac, 1.160% MACs, 
          (0): ConvNormActivation(
            0.039 M, 0.175% Params, 0.014 GMac, 0.545% MACs, 
            (0): Conv2d(0.038 M, 0.171% Params, 0.014 GMac, 0.532% MACs, 80, 480, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (1): BatchNorm2d(0.001 M, 0.004% Params, 0.0 GMac, 0.013% MACs, 480, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): SiLU(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, inplace=True)
          )
          (1): ConvNormActivation(
            0.005 M, 0.023% Params, 0.002 GMac, 0.073% MACs, 
            (0): Conv2d(0.004 M, 0.019% Params, 0.002 GMac, 0.060% MACs, 480, 480, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=480, bias=False)
            (1): BatchNorm2d(0.001 M, 0.004% Params, 0.0 GMac, 0.013% MACs, 480, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): SiLU(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, inplace=True)
          )
          (2): SqueezeExcitation(
            0.02 M, 0.088% Params, 0.0 GMac, 0.007% MACs, 
            (avgpool): AdaptiveAvgPool2d(0.0 M, 0.000% Params, 0.0 GMac, 0.007% MACs, output_size=1)
            (fc1): Conv2d(0.01 M, 0.043% Params, 0.0 GMac, 0.000% MACs, 480, 20, kernel_size=(1, 1), stride=(1, 1))
            (fc2): Conv2d(0.01 M, 0.045% Params, 0.0 GMac, 0.000% MACs, 20, 480, kernel_size=(1, 1), stride=(1, 1))
            (activation): SiLU(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, inplace=True)
            (scale_activation): Sigmoid(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, )
          )
          (3): ConvNormActivation(
            0.039 M, 0.172% Params, 0.014 GMac, 0.534% MACs, 
            (0): Conv2d(0.038 M, 0.171% Params, 0.014 GMac, 0.532% MACs, 480, 80, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (1): BatchNorm2d(0.0 M, 0.001% Params, 0.0 GMac, 0.002% MACs, 80, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (stochastic_depth): StochasticDepth(p=0.08750000000000001, mode=row)
      )
    )
    (5): Sequential(
      0.543 M, 2.416% Params, 0.162 GMac, 6.220% MACs, 
      (0): MBConv(
        0.126 M, 0.561% Params, 0.039 GMac, 1.480% MACs, 
        (block): Sequential(
          0.126 M, 0.561% Params, 0.039 GMac, 1.480% MACs, 
          (0): ConvNormActivation(
            0.039 M, 0.175% Params, 0.014 GMac, 0.545% MACs, 
            (0): Conv2d(0.038 M, 0.171% Params, 0.014 GMac, 0.532% MACs, 80, 480, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (1): BatchNorm2d(0.001 M, 0.004% Params, 0.0 GMac, 0.013% MACs, 480, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): SiLU(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, inplace=True)
          )
          (1): ConvNormActivation(
            0.013 M, 0.058% Params, 0.005 GMac, 0.180% MACs, 
            (0): Conv2d(0.012 M, 0.053% Params, 0.004 GMac, 0.166% MACs, 480, 480, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=480, bias=False)
            (1): BatchNorm2d(0.001 M, 0.004% Params, 0.0 GMac, 0.013% MACs, 480, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): SiLU(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, inplace=True)
          )
          (2): SqueezeExcitation(
            0.02 M, 0.088% Params, 0.0 GMac, 0.007% MACs, 
            (avgpool): AdaptiveAvgPool2d(0.0 M, 0.000% Params, 0.0 GMac, 0.007% MACs, output_size=1)
            (fc1): Conv2d(0.01 M, 0.043% Params, 0.0 GMac, 0.000% MACs, 480, 20, kernel_size=(1, 1), stride=(1, 1))
            (fc2): Conv2d(0.01 M, 0.045% Params, 0.0 GMac, 0.000% MACs, 20, 480, kernel_size=(1, 1), stride=(1, 1))
            (activation): SiLU(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, inplace=True)
            (scale_activation): Sigmoid(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, )
          )
          (3): ConvNormActivation(
            0.054 M, 0.240% Params, 0.019 GMac, 0.748% MACs, 
            (0): Conv2d(0.054 M, 0.239% Params, 0.019 GMac, 0.745% MACs, 480, 112, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (1): BatchNorm2d(0.0 M, 0.001% Params, 0.0 GMac, 0.003% MACs, 112, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (stochastic_depth): StochasticDepth(p=0.1, mode=row)
      )
      (1): MBConv(
        0.209 M, 0.928% Params, 0.062 GMac, 2.370% MACs, 
        (block): Sequential(
          0.209 M, 0.928% Params, 0.062 GMac, 2.370% MACs, 
          (0): ConvNormActivation(
            0.077 M, 0.341% Params, 0.028 GMac, 1.061% MACs, 
            (0): Conv2d(0.075 M, 0.335% Params, 0.027 GMac, 1.043% MACs, 112, 672, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (1): BatchNorm2d(0.001 M, 0.006% Params, 0.0 GMac, 0.019% MACs, 672, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): SiLU(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, inplace=True)
          )
          (1): ConvNormActivation(
            0.018 M, 0.081% Params, 0.007 GMac, 0.251% MACs, 
            (0): Conv2d(0.017 M, 0.075% Params, 0.006 GMac, 0.233% MACs, 672, 672, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=672, bias=False)
            (1): BatchNorm2d(0.001 M, 0.006% Params, 0.0 GMac, 0.019% MACs, 672, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): SiLU(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, inplace=True)
          )
          (2): SqueezeExcitation(
            0.038 M, 0.171% Params, 0.0 GMac, 0.011% MACs, 
            (avgpool): AdaptiveAvgPool2d(0.0 M, 0.000% Params, 0.0 GMac, 0.009% MACs, output_size=1)
            (fc1): Conv2d(0.019 M, 0.084% Params, 0.0 GMac, 0.001% MACs, 672, 28, kernel_size=(1, 1), stride=(1, 1))
            (fc2): Conv2d(0.019 M, 0.087% Params, 0.0 GMac, 0.001% MACs, 28, 672, kernel_size=(1, 1), stride=(1, 1))
            (activation): SiLU(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, inplace=True)
            (scale_activation): Sigmoid(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, )
          )
          (3): ConvNormActivation(
            0.075 M, 0.336% Params, 0.027 GMac, 1.046% MACs, 
            (0): Conv2d(0.075 M, 0.335% Params, 0.027 GMac, 1.043% MACs, 672, 112, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (1): BatchNorm2d(0.0 M, 0.001% Params, 0.0 GMac, 0.003% MACs, 112, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (stochastic_depth): StochasticDepth(p=0.1125, mode=row)
      )
      (2): MBConv(
        0.209 M, 0.928% Params, 0.062 GMac, 2.370% MACs, 
        (block): Sequential(
          0.209 M, 0.928% Params, 0.062 GMac, 2.370% MACs, 
          (0): ConvNormActivation(
            0.077 M, 0.341% Params, 0.028 GMac, 1.061% MACs, 
            (0): Conv2d(0.075 M, 0.335% Params, 0.027 GMac, 1.043% MACs, 112, 672, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (1): BatchNorm2d(0.001 M, 0.006% Params, 0.0 GMac, 0.019% MACs, 672, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): SiLU(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, inplace=True)
          )
          (1): ConvNormActivation(
            0.018 M, 0.081% Params, 0.007 GMac, 0.251% MACs, 
            (0): Conv2d(0.017 M, 0.075% Params, 0.006 GMac, 0.233% MACs, 672, 672, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=672, bias=False)
            (1): BatchNorm2d(0.001 M, 0.006% Params, 0.0 GMac, 0.019% MACs, 672, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): SiLU(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, inplace=True)
          )
          (2): SqueezeExcitation(
            0.038 M, 0.171% Params, 0.0 GMac, 0.011% MACs, 
            (avgpool): AdaptiveAvgPool2d(0.0 M, 0.000% Params, 0.0 GMac, 0.009% MACs, output_size=1)
            (fc1): Conv2d(0.019 M, 0.084% Params, 0.0 GMac, 0.001% MACs, 672, 28, kernel_size=(1, 1), stride=(1, 1))
            (fc2): Conv2d(0.019 M, 0.087% Params, 0.0 GMac, 0.001% MACs, 28, 672, kernel_size=(1, 1), stride=(1, 1))
            (activation): SiLU(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, inplace=True)
            (scale_activation): Sigmoid(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, )
          )
          (3): ConvNormActivation(
            0.075 M, 0.336% Params, 0.027 GMac, 1.046% MACs, 
            (0): Conv2d(0.075 M, 0.335% Params, 0.027 GMac, 1.043% MACs, 672, 112, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (1): BatchNorm2d(0.0 M, 0.001% Params, 0.0 GMac, 0.003% MACs, 112, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (stochastic_depth): StochasticDepth(p=0.125, mode=row)
      )
    )
    (6): Sequential(
      2.026 M, 9.014% Params, 0.186 GMac, 7.141% MACs, 
      (0): MBConv(
        0.262 M, 1.168% Params, 0.043 GMac, 1.632% MACs, 
        (block): Sequential(
          0.262 M, 1.168% Params, 0.043 GMac, 1.632% MACs, 
          (0): ConvNormActivation(
            0.077 M, 0.341% Params, 0.028 GMac, 1.061% MACs, 
            (0): Conv2d(0.075 M, 0.335% Params, 0.027 GMac, 1.043% MACs, 112, 672, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (1): BatchNorm2d(0.001 M, 0.006% Params, 0.0 GMac, 0.019% MACs, 672, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): SiLU(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, inplace=True)
          )
          (1): ConvNormActivation(
            0.018 M, 0.081% Params, 0.002 GMac, 0.070% MACs, 
            (0): Conv2d(0.017 M, 0.075% Params, 0.002 GMac, 0.064% MACs, 672, 672, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), groups=672, bias=False)
            (1): BatchNorm2d(0.001 M, 0.006% Params, 0.0 GMac, 0.005% MACs, 672, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): SiLU(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, inplace=True)
          )
          (2): SqueezeExcitation(
            0.038 M, 0.171% Params, 0.0 GMac, 0.004% MACs, 
            (avgpool): AdaptiveAvgPool2d(0.0 M, 0.000% Params, 0.0 GMac, 0.003% MACs, output_size=1)
            (fc1): Conv2d(0.019 M, 0.084% Params, 0.0 GMac, 0.001% MACs, 672, 28, kernel_size=(1, 1), stride=(1, 1))
            (fc2): Conv2d(0.019 M, 0.087% Params, 0.0 GMac, 0.001% MACs, 28, 672, kernel_size=(1, 1), stride=(1, 1))
            (activation): SiLU(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, inplace=True)
            (scale_activation): Sigmoid(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, )
          )
          (3): ConvNormActivation(
            0.129 M, 0.576% Params, 0.013 GMac, 0.497% MACs, 
            (0): Conv2d(0.129 M, 0.574% Params, 0.013 GMac, 0.495% MACs, 672, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (1): BatchNorm2d(0.0 M, 0.002% Params, 0.0 GMac, 0.001% MACs, 192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (stochastic_depth): StochasticDepth(p=0.1375, mode=row)
      )
      (1): MBConv(
        0.588 M, 2.616% Params, 0.048 GMac, 1.836% MACs, 
        (block): Sequential(
          0.588 M, 2.616% Params, 0.048 GMac, 1.836% MACs, 
          (0): ConvNormActivation(
            0.223 M, 0.994% Params, 0.022 GMac, 0.858% MACs, 
            (0): Conv2d(0.221 M, 0.984% Params, 0.022 GMac, 0.849% MACs, 192, 1152, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (1): BatchNorm2d(0.002 M, 0.010% Params, 0.0 GMac, 0.009% MACs, 1152, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): SiLU(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, inplace=True)
          )
          (1): ConvNormActivation(
            0.031 M, 0.138% Params, 0.003 GMac, 0.119% MACs, 
            (0): Conv2d(0.029 M, 0.128% Params, 0.003 GMac, 0.111% MACs, 1152, 1152, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=1152, bias=False)
            (1): BatchNorm2d(0.002 M, 0.010% Params, 0.0 GMac, 0.009% MACs, 1152, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): SiLU(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, inplace=True)
          )
          (2): SqueezeExcitation(
            0.112 M, 0.497% Params, 0.0 GMac, 0.009% MACs, 
            (avgpool): AdaptiveAvgPool2d(0.0 M, 0.000% Params, 0.0 GMac, 0.004% MACs, output_size=1)
            (fc1): Conv2d(0.055 M, 0.246% Params, 0.0 GMac, 0.002% MACs, 1152, 48, kernel_size=(1, 1), stride=(1, 1))
            (fc2): Conv2d(0.056 M, 0.251% Params, 0.0 GMac, 0.002% MACs, 48, 1152, kernel_size=(1, 1), stride=(1, 1))
            (activation): SiLU(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, inplace=True)
            (scale_activation): Sigmoid(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, )
          )
          (3): ConvNormActivation(
            0.222 M, 0.986% Params, 0.022 GMac, 0.850% MACs, 
            (0): Conv2d(0.221 M, 0.984% Params, 0.022 GMac, 0.849% MACs, 1152, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (1): BatchNorm2d(0.0 M, 0.002% Params, 0.0 GMac, 0.001% MACs, 192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (stochastic_depth): StochasticDepth(p=0.15000000000000002, mode=row)
      )
      (2): MBConv(
        0.588 M, 2.616% Params, 0.048 GMac, 1.836% MACs, 
        (block): Sequential(
          0.588 M, 2.616% Params, 0.048 GMac, 1.836% MACs, 
          (0): ConvNormActivation(
            0.223 M, 0.994% Params, 0.022 GMac, 0.858% MACs, 
            (0): Conv2d(0.221 M, 0.984% Params, 0.022 GMac, 0.849% MACs, 192, 1152, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (1): BatchNorm2d(0.002 M, 0.010% Params, 0.0 GMac, 0.009% MACs, 1152, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): SiLU(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, inplace=True)
          )
          (1): ConvNormActivation(
            0.031 M, 0.138% Params, 0.003 GMac, 0.119% MACs, 
            (0): Conv2d(0.029 M, 0.128% Params, 0.003 GMac, 0.111% MACs, 1152, 1152, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=1152, bias=False)
            (1): BatchNorm2d(0.002 M, 0.010% Params, 0.0 GMac, 0.009% MACs, 1152, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): SiLU(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, inplace=True)
          )
          (2): SqueezeExcitation(
            0.112 M, 0.497% Params, 0.0 GMac, 0.009% MACs, 
            (avgpool): AdaptiveAvgPool2d(0.0 M, 0.000% Params, 0.0 GMac, 0.004% MACs, output_size=1)
            (fc1): Conv2d(0.055 M, 0.246% Params, 0.0 GMac, 0.002% MACs, 1152, 48, kernel_size=(1, 1), stride=(1, 1))
            (fc2): Conv2d(0.056 M, 0.251% Params, 0.0 GMac, 0.002% MACs, 48, 1152, kernel_size=(1, 1), stride=(1, 1))
            (activation): SiLU(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, inplace=True)
            (scale_activation): Sigmoid(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, )
          )
          (3): ConvNormActivation(
            0.222 M, 0.986% Params, 0.022 GMac, 0.850% MACs, 
            (0): Conv2d(0.221 M, 0.984% Params, 0.022 GMac, 0.849% MACs, 1152, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (1): BatchNorm2d(0.0 M, 0.002% Params, 0.0 GMac, 0.001% MACs, 192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (stochastic_depth): StochasticDepth(p=0.1625, mode=row)
      )
      (3): MBConv(
        0.588 M, 2.616% Params, 0.048 GMac, 1.836% MACs, 
        (block): Sequential(
          0.588 M, 2.616% Params, 0.048 GMac, 1.836% MACs, 
          (0): ConvNormActivation(
            0.223 M, 0.994% Params, 0.022 GMac, 0.858% MACs, 
            (0): Conv2d(0.221 M, 0.984% Params, 0.022 GMac, 0.849% MACs, 192, 1152, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (1): BatchNorm2d(0.002 M, 0.010% Params, 0.0 GMac, 0.009% MACs, 1152, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): SiLU(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, inplace=True)
          )
          (1): ConvNormActivation(
            0.031 M, 0.138% Params, 0.003 GMac, 0.119% MACs, 
            (0): Conv2d(0.029 M, 0.128% Params, 0.003 GMac, 0.111% MACs, 1152, 1152, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=1152, bias=False)
            (1): BatchNorm2d(0.002 M, 0.010% Params, 0.0 GMac, 0.009% MACs, 1152, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): SiLU(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, inplace=True)
          )
          (2): SqueezeExcitation(
            0.112 M, 0.497% Params, 0.0 GMac, 0.009% MACs, 
            (avgpool): AdaptiveAvgPool2d(0.0 M, 0.000% Params, 0.0 GMac, 0.004% MACs, output_size=1)
            (fc1): Conv2d(0.055 M, 0.246% Params, 0.0 GMac, 0.002% MACs, 1152, 48, kernel_size=(1, 1), stride=(1, 1))
            (fc2): Conv2d(0.056 M, 0.251% Params, 0.0 GMac, 0.002% MACs, 48, 1152, kernel_size=(1, 1), stride=(1, 1))
            (activation): SiLU(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, inplace=True)
            (scale_activation): Sigmoid(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, )
          )
          (3): ConvNormActivation(
            0.222 M, 0.986% Params, 0.022 GMac, 0.850% MACs, 
            (0): Conv2d(0.221 M, 0.984% Params, 0.022 GMac, 0.849% MACs, 1152, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (1): BatchNorm2d(0.0 M, 0.002% Params, 0.0 GMac, 0.001% MACs, 192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (stochastic_depth): StochasticDepth(p=0.17500000000000002, mode=row)
      )
    )
    (7): Sequential(
      0.717 M, 3.191% Params, 0.061 GMac, 2.333% MACs, 
      (0): MBConv(
        0.717 M, 3.191% Params, 0.061 GMac, 2.333% MACs, 
        (block): Sequential(
          0.717 M, 3.191% Params, 0.061 GMac, 2.333% MACs, 
          (0): ConvNormActivation(
            0.223 M, 0.994% Params, 0.022 GMac, 0.858% MACs, 
            (0): Conv2d(0.221 M, 0.984% Params, 0.022 GMac, 0.849% MACs, 192, 1152, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (1): BatchNorm2d(0.002 M, 0.010% Params, 0.0 GMac, 0.009% MACs, 1152, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): SiLU(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, inplace=True)
          )
          (1): ConvNormActivation(
            0.013 M, 0.056% Params, 0.001 GMac, 0.049% MACs, 
            (0): Conv2d(0.01 M, 0.046% Params, 0.001 GMac, 0.040% MACs, 1152, 1152, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=1152, bias=False)
            (1): BatchNorm2d(0.002 M, 0.010% Params, 0.0 GMac, 0.009% MACs, 1152, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): SiLU(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, inplace=True)
          )
          (2): SqueezeExcitation(
            0.112 M, 0.497% Params, 0.0 GMac, 0.009% MACs, 
            (avgpool): AdaptiveAvgPool2d(0.0 M, 0.000% Params, 0.0 GMac, 0.004% MACs, output_size=1)
            (fc1): Conv2d(0.055 M, 0.246% Params, 0.0 GMac, 0.002% MACs, 1152, 48, kernel_size=(1, 1), stride=(1, 1))
            (fc2): Conv2d(0.056 M, 0.251% Params, 0.0 GMac, 0.002% MACs, 48, 1152, kernel_size=(1, 1), stride=(1, 1))
            (activation): SiLU(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, inplace=True)
            (scale_activation): Sigmoid(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, )
          )
          (3): ConvNormActivation(
            0.369 M, 1.643% Params, 0.037 GMac, 1.417% MACs, 
            (0): Conv2d(0.369 M, 1.640% Params, 0.037 GMac, 1.415% MACs, 1152, 320, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (1): BatchNorm2d(0.001 M, 0.003% Params, 0.0 GMac, 0.002% MACs, 320, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (stochastic_depth): StochasticDepth(p=0.1875, mode=row)
      )
    )
    (8): ConvNormActivation(
      0.412 M, 1.833% Params, 0.041 GMac, 1.582% MACs, 
      (0): Conv2d(0.41 M, 1.822% Params, 0.041 GMac, 1.572% MACs, 320, 1280, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (1): BatchNorm2d(0.003 M, 0.011% Params, 0.0 GMac, 0.010% MACs, 1280, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): SiLU(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, inplace=True)
    )
  )
  (encoder): DilatedEncoder(
    4.141 M, 18.421% Params, 0.414 GMac, 15.906% MACs, 
    (lateral_conv): Conv2d(0.656 M, 2.918% Params, 0.066 GMac, 2.517% MACs, 1280, 512, kernel_size=(1, 1), stride=(1, 1))
    (lateral_norm): BatchNorm2d(0.001 M, 0.005% Params, 0.0 GMac, 0.004% MACs, 512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (fpn_conv): Conv2d(2.36 M, 10.498% Params, 0.236 GMac, 9.058% MACs, 512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (fpn_norm): BatchNorm2d(0.001 M, 0.005% Params, 0.0 GMac, 0.004% MACs, 512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (dilated_encoder_blocks): Sequential(
      1.123 M, 4.997% Params, 0.113 GMac, 4.323% MACs, 
      (0): Bottleneck(
        0.281 M, 1.249% Params, 0.028 GMac, 1.081% MACs, 
        (conv1): Sequential(
          0.066 M, 0.293% Params, 0.007 GMac, 0.254% MACs, 
          (0): Conv2d(0.066 M, 0.292% Params, 0.007 GMac, 0.252% MACs, 512, 128, kernel_size=(1, 1), stride=(1, 1))
          (1): BatchNorm2d(0.0 M, 0.001% Params, 0.0 GMac, 0.001% MACs, 128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): LeakyReLU(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, negative_slope=0.01)
        )
        (conv2): Sequential(
          0.148 M, 0.658% Params, 0.015 GMac, 0.568% MACs, 
          (0): Conv2d(0.148 M, 0.657% Params, 0.015 GMac, 0.566% MACs, 128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(2, 2), dilation=(2, 2))
          (1): BatchNorm2d(0.0 M, 0.001% Params, 0.0 GMac, 0.001% MACs, 128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): LeakyReLU(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, negative_slope=0.01)
        )
        (conv3): Sequential(
          0.067 M, 0.298% Params, 0.007 GMac, 0.259% MACs, 
          (0): Conv2d(0.066 M, 0.294% Params, 0.007 GMac, 0.254% MACs, 128, 512, kernel_size=(1, 1), stride=(1, 1))
          (1): BatchNorm2d(0.001 M, 0.005% Params, 0.0 GMac, 0.004% MACs, 512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): LeakyReLU(0.0 M, 0.000% Params, 0.0 GMac, 0.002% MACs, negative_slope=0.01)
        )
      )
      (1): Bottleneck(
        0.281 M, 1.249% Params, 0.028 GMac, 1.081% MACs, 
        (conv1): Sequential(
          0.066 M, 0.293% Params, 0.007 GMac, 0.254% MACs, 
          (0): Conv2d(0.066 M, 0.292% Params, 0.007 GMac, 0.252% MACs, 512, 128, kernel_size=(1, 1), stride=(1, 1))
          (1): BatchNorm2d(0.0 M, 0.001% Params, 0.0 GMac, 0.001% MACs, 128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): LeakyReLU(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, negative_slope=0.01)
        )
        (conv2): Sequential(
          0.148 M, 0.658% Params, 0.015 GMac, 0.568% MACs, 
          (0): Conv2d(0.148 M, 0.657% Params, 0.015 GMac, 0.566% MACs, 128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(4, 4), dilation=(4, 4))
          (1): BatchNorm2d(0.0 M, 0.001% Params, 0.0 GMac, 0.001% MACs, 128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): LeakyReLU(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, negative_slope=0.01)
        )
        (conv3): Sequential(
          0.067 M, 0.298% Params, 0.007 GMac, 0.259% MACs, 
          (0): Conv2d(0.066 M, 0.294% Params, 0.007 GMac, 0.254% MACs, 128, 512, kernel_size=(1, 1), stride=(1, 1))
          (1): BatchNorm2d(0.001 M, 0.005% Params, 0.0 GMac, 0.004% MACs, 512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): LeakyReLU(0.0 M, 0.000% Params, 0.0 GMac, 0.002% MACs, negative_slope=0.01)
        )
      )
      (2): Bottleneck(
        0.281 M, 1.249% Params, 0.028 GMac, 1.081% MACs, 
        (conv1): Sequential(
          0.066 M, 0.293% Params, 0.007 GMac, 0.254% MACs, 
          (0): Conv2d(0.066 M, 0.292% Params, 0.007 GMac, 0.252% MACs, 512, 128, kernel_size=(1, 1), stride=(1, 1))
          (1): BatchNorm2d(0.0 M, 0.001% Params, 0.0 GMac, 0.001% MACs, 128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): LeakyReLU(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, negative_slope=0.01)
        )
        (conv2): Sequential(
          0.148 M, 0.658% Params, 0.015 GMac, 0.568% MACs, 
          (0): Conv2d(0.148 M, 0.657% Params, 0.015 GMac, 0.566% MACs, 128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(6, 6), dilation=(6, 6))
          (1): BatchNorm2d(0.0 M, 0.001% Params, 0.0 GMac, 0.001% MACs, 128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): LeakyReLU(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, negative_slope=0.01)
        )
        (conv3): Sequential(
          0.067 M, 0.298% Params, 0.007 GMac, 0.259% MACs, 
          (0): Conv2d(0.066 M, 0.294% Params, 0.007 GMac, 0.254% MACs, 128, 512, kernel_size=(1, 1), stride=(1, 1))
          (1): BatchNorm2d(0.001 M, 0.005% Params, 0.0 GMac, 0.004% MACs, 512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): LeakyReLU(0.0 M, 0.000% Params, 0.0 GMac, 0.002% MACs, negative_slope=0.01)
        )
      )
      (3): Bottleneck(
        0.281 M, 1.249% Params, 0.028 GMac, 1.081% MACs, 
        (conv1): Sequential(
          0.066 M, 0.293% Params, 0.007 GMac, 0.254% MACs, 
          (0): Conv2d(0.066 M, 0.292% Params, 0.007 GMac, 0.252% MACs, 512, 128, kernel_size=(1, 1), stride=(1, 1))
          (1): BatchNorm2d(0.0 M, 0.001% Params, 0.0 GMac, 0.001% MACs, 128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): LeakyReLU(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, negative_slope=0.01)
        )
        (conv2): Sequential(
          0.148 M, 0.658% Params, 0.015 GMac, 0.568% MACs, 
          (0): Conv2d(0.148 M, 0.657% Params, 0.015 GMac, 0.566% MACs, 128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(8, 8), dilation=(8, 8))
          (1): BatchNorm2d(0.0 M, 0.001% Params, 0.0 GMac, 0.001% MACs, 128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): LeakyReLU(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, negative_slope=0.01)
        )
        (conv3): Sequential(
          0.067 M, 0.298% Params, 0.007 GMac, 0.259% MACs, 
          (0): Conv2d(0.066 M, 0.294% Params, 0.007 GMac, 0.254% MACs, 128, 512, kernel_size=(1, 1), stride=(1, 1))
          (1): BatchNorm2d(0.001 M, 0.005% Params, 0.0 GMac, 0.004% MACs, 512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): LeakyReLU(0.0 M, 0.000% Params, 0.0 GMac, 0.002% MACs, negative_slope=0.01)
        )
      )
    )
  )
  (decoder): Decoder(
    14.331 M, 63.751% Params, 1.433 GMac, 55.018% MACs, 
    (cls_subnet): Sequential(
      4.722 M, 21.004% Params, 0.472 GMac, 18.127% MACs, 
      (0): Conv2d(2.36 M, 10.498% Params, 0.236 GMac, 9.058% MACs, 512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (1): BatchNorm2d(0.001 M, 0.005% Params, 0.0 GMac, 0.004% MACs, 512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): LeakyReLU(0.0 M, 0.000% Params, 0.0 GMac, 0.002% MACs, negative_slope=0.01)
      (3): Conv2d(2.36 M, 10.498% Params, 0.236 GMac, 9.058% MACs, 512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (4): BatchNorm2d(0.001 M, 0.005% Params, 0.0 GMac, 0.004% MACs, 512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (5): LeakyReLU(0.0 M, 0.000% Params, 0.0 GMac, 0.002% MACs, negative_slope=0.01)
    )
    (bbox_subnet): Sequential(
      9.443 M, 42.009% Params, 0.945 GMac, 36.254% MACs, 
      (0): Conv2d(2.36 M, 10.498% Params, 0.236 GMac, 9.058% MACs, 512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (1): BatchNorm2d(0.001 M, 0.005% Params, 0.0 GMac, 0.004% MACs, 512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): LeakyReLU(0.0 M, 0.000% Params, 0.0 GMac, 0.002% MACs, negative_slope=0.01)
      (3): Conv2d(2.36 M, 10.498% Params, 0.236 GMac, 9.058% MACs, 512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (4): BatchNorm2d(0.001 M, 0.005% Params, 0.0 GMac, 0.004% MACs, 512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (5): LeakyReLU(0.0 M, 0.000% Params, 0.0 GMac, 0.002% MACs, negative_slope=0.01)
      (6): Conv2d(2.36 M, 10.498% Params, 0.236 GMac, 9.058% MACs, 512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (7): BatchNorm2d(0.001 M, 0.005% Params, 0.0 GMac, 0.004% MACs, 512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (8): LeakyReLU(0.0 M, 0.000% Params, 0.0 GMac, 0.002% MACs, negative_slope=0.01)
      (9): Conv2d(2.36 M, 10.498% Params, 0.236 GMac, 9.058% MACs, 512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (10): BatchNorm2d(0.001 M, 0.005% Params, 0.0 GMac, 0.004% MACs, 512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (11): LeakyReLU(0.0 M, 0.000% Params, 0.0 GMac, 0.002% MACs, negative_slope=0.01)
    )
    (cls_score): Conv2d(0.074 M, 0.328% Params, 0.007 GMac, 0.283% MACs, 512, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (bbox_pred): Conv2d(0.074 M, 0.328% Params, 0.007 GMac, 0.283% MACs, 512, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (object_pred): Conv2d(0.018 M, 0.082% Params, 0.002 GMac, 0.071% MACs, 512, 4, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  )
)
Computational complexity:       2.61 GMac
Number of parameters:           22.48 M
```

