```
(base) oem@richard:~/Documents/rqp_proj/tx2dl$ python measure.py 
Warning: module InvertedResidual is treated as a zero-op.
Warning: module SSD is treated as a zero-op.
SSD(
  3.087 M, 100.000% Params, 0.663 GMac, 100.000% MACs, 
  (base_net): Sequential(
    2.224 M, 72.032% Params, 0.599 GMac, 90.236% MACs, 
    (0): Sequential(
      0.001 M, 0.030% Params, 0.022 GMac, 3.256% MACs, 
      (0): Conv2d(0.001 M, 0.028% Params, 0.019 GMac, 2.930% MACs, 3, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
      (1): BatchNorm2d(0.0 M, 0.002% Params, 0.001 GMac, 0.217% MACs, 32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU6(0.0 M, 0.000% Params, 0.001 GMac, 0.109% MACs, inplace=True)
    )
    (1): InvertedResidual(
      0.001 M, 0.029% Params, 0.021 GMac, 3.147% MACs, 
      (conv): Sequential(
        0.001 M, 0.029% Params, 0.021 GMac, 3.147% MACs, 
        (0): Conv2d(0.0 M, 0.009% Params, 0.006 GMac, 0.977% MACs, 32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=32, bias=False)
        (1): BatchNorm2d(0.0 M, 0.002% Params, 0.001 GMac, 0.217% MACs, 32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU6(0.0 M, 0.000% Params, 0.001 GMac, 0.109% MACs, inplace=True)
        (3): Conv2d(0.001 M, 0.017% Params, 0.012 GMac, 1.736% MACs, 32, 16, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (4): BatchNorm2d(0.0 M, 0.001% Params, 0.001 GMac, 0.109% MACs, 16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (2): InvertedResidual(
      0.005 M, 0.166% Params, 0.061 GMac, 9.157% MACs, 
      (conv): Sequential(
        0.005 M, 0.166% Params, 0.061 GMac, 9.157% MACs, 
        (0): Conv2d(0.002 M, 0.050% Params, 0.035 GMac, 5.209% MACs, 16, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (1): BatchNorm2d(0.0 M, 0.006% Params, 0.004 GMac, 0.651% MACs, 96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU6(0.0 M, 0.000% Params, 0.002 GMac, 0.326% MACs, inplace=True)
        (3): Conv2d(0.001 M, 0.028% Params, 0.005 GMac, 0.733% MACs, 96, 96, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=96, bias=False)
        (4): BatchNorm2d(0.0 M, 0.006% Params, 0.001 GMac, 0.163% MACs, 96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (5): ReLU6(0.0 M, 0.000% Params, 0.001 GMac, 0.081% MACs, inplace=True)
        (6): Conv2d(0.002 M, 0.075% Params, 0.013 GMac, 1.954% MACs, 96, 24, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (7): BatchNorm2d(0.0 M, 0.002% Params, 0.0 GMac, 0.041% MACs, 24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (3): InvertedResidual(
      0.009 M, 0.286% Params, 0.051 GMac, 7.733% MACs, 
      (conv): Sequential(
        0.009 M, 0.286% Params, 0.051 GMac, 7.733% MACs, 
        (0): Conv2d(0.003 M, 0.112% Params, 0.019 GMac, 2.930% MACs, 24, 144, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (1): BatchNorm2d(0.0 M, 0.009% Params, 0.002 GMac, 0.244% MACs, 144, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU6(0.0 M, 0.000% Params, 0.001 GMac, 0.122% MACs, inplace=True)
        (3): Conv2d(0.001 M, 0.042% Params, 0.007 GMac, 1.099% MACs, 144, 144, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=144, bias=False)
        (4): BatchNorm2d(0.0 M, 0.009% Params, 0.002 GMac, 0.244% MACs, 144, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (5): ReLU6(0.0 M, 0.000% Params, 0.001 GMac, 0.122% MACs, inplace=True)
        (6): Conv2d(0.003 M, 0.112% Params, 0.019 GMac, 2.930% MACs, 144, 24, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (7): BatchNorm2d(0.0 M, 0.002% Params, 0.0 GMac, 0.041% MACs, 24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (4): InvertedResidual(
      0.01 M, 0.324% Params, 0.031 GMac, 4.690% MACs, 
      (conv): Sequential(
        0.01 M, 0.324% Params, 0.031 GMac, 4.690% MACs, 
        (0): Conv2d(0.003 M, 0.112% Params, 0.019 GMac, 2.930% MACs, 24, 144, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (1): BatchNorm2d(0.0 M, 0.009% Params, 0.002 GMac, 0.244% MACs, 144, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU6(0.0 M, 0.000% Params, 0.001 GMac, 0.122% MACs, inplace=True)
        (3): Conv2d(0.001 M, 0.042% Params, 0.002 GMac, 0.282% MACs, 144, 144, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=144, bias=False)
        (4): BatchNorm2d(0.0 M, 0.009% Params, 0.0 GMac, 0.063% MACs, 144, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (5): ReLU6(0.0 M, 0.000% Params, 0.0 GMac, 0.031% MACs, inplace=True)
        (6): Conv2d(0.005 M, 0.149% Params, 0.007 GMac, 1.003% MACs, 144, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (7): BatchNorm2d(0.0 M, 0.002% Params, 0.0 GMac, 0.014% MACs, 32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (5): InvertedResidual(
      0.015 M, 0.481% Params, 0.022 GMac, 3.315% MACs, 
      (conv): Sequential(
        0.015 M, 0.481% Params, 0.022 GMac, 3.315% MACs, 
        (0): Conv2d(0.006 M, 0.199% Params, 0.009 GMac, 1.337% MACs, 32, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (1): BatchNorm2d(0.0 M, 0.012% Params, 0.001 GMac, 0.084% MACs, 192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU6(0.0 M, 0.000% Params, 0.0 GMac, 0.042% MACs, inplace=True)
        (3): Conv2d(0.002 M, 0.056% Params, 0.002 GMac, 0.376% MACs, 192, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=192, bias=False)
        (4): BatchNorm2d(0.0 M, 0.012% Params, 0.001 GMac, 0.084% MACs, 192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (5): ReLU6(0.0 M, 0.000% Params, 0.0 GMac, 0.042% MACs, inplace=True)
        (6): Conv2d(0.006 M, 0.199% Params, 0.009 GMac, 1.337% MACs, 192, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (7): BatchNorm2d(0.0 M, 0.002% Params, 0.0 GMac, 0.014% MACs, 32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (6): InvertedResidual(
      0.015 M, 0.481% Params, 0.022 GMac, 3.315% MACs, 
      (conv): Sequential(
        0.015 M, 0.481% Params, 0.022 GMac, 3.315% MACs, 
        (0): Conv2d(0.006 M, 0.199% Params, 0.009 GMac, 1.337% MACs, 32, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (1): BatchNorm2d(0.0 M, 0.012% Params, 0.001 GMac, 0.084% MACs, 192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU6(0.0 M, 0.000% Params, 0.0 GMac, 0.042% MACs, inplace=True)
        (3): Conv2d(0.002 M, 0.056% Params, 0.002 GMac, 0.376% MACs, 192, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=192, bias=False)
        (4): BatchNorm2d(0.0 M, 0.012% Params, 0.001 GMac, 0.084% MACs, 192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (5): ReLU6(0.0 M, 0.000% Params, 0.0 GMac, 0.042% MACs, inplace=True)
        (6): Conv2d(0.006 M, 0.199% Params, 0.009 GMac, 1.337% MACs, 192, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (7): BatchNorm2d(0.0 M, 0.002% Params, 0.0 GMac, 0.014% MACs, 32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (7): InvertedResidual(
      0.021 M, 0.682% Params, 0.015 GMac, 2.264% MACs, 
      (conv): Sequential(
        0.021 M, 0.682% Params, 0.015 GMac, 2.264% MACs, 
        (0): Conv2d(0.006 M, 0.199% Params, 0.009 GMac, 1.337% MACs, 32, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (1): BatchNorm2d(0.0 M, 0.012% Params, 0.001 GMac, 0.084% MACs, 192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU6(0.0 M, 0.000% Params, 0.0 GMac, 0.042% MACs, inplace=True)
        (3): Conv2d(0.002 M, 0.056% Params, 0.001 GMac, 0.094% MACs, 192, 192, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=192, bias=False)
        (4): BatchNorm2d(0.0 M, 0.012% Params, 0.0 GMac, 0.021% MACs, 192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (5): ReLU6(0.0 M, 0.000% Params, 0.0 GMac, 0.010% MACs, inplace=True)
        (6): Conv2d(0.012 M, 0.398% Params, 0.004 GMac, 0.669% MACs, 192, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (7): BatchNorm2d(0.0 M, 0.004% Params, 0.0 GMac, 0.007% MACs, 64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (8): InvertedResidual(
      0.054 M, 1.758% Params, 0.02 GMac, 2.995% MACs, 
      (conv): Sequential(
        0.054 M, 1.758% Params, 0.02 GMac, 2.995% MACs, 
        (0): Conv2d(0.025 M, 0.796% Params, 0.009 GMac, 1.337% MACs, 64, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (1): BatchNorm2d(0.001 M, 0.025% Params, 0.0 GMac, 0.042% MACs, 384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU6(0.0 M, 0.000% Params, 0.0 GMac, 0.021% MACs, inplace=True)
        (3): Conv2d(0.003 M, 0.112% Params, 0.001 GMac, 0.188% MACs, 384, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=384, bias=False)
        (4): BatchNorm2d(0.001 M, 0.025% Params, 0.0 GMac, 0.042% MACs, 384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (5): ReLU6(0.0 M, 0.000% Params, 0.0 GMac, 0.021% MACs, inplace=True)
        (6): Conv2d(0.025 M, 0.796% Params, 0.009 GMac, 1.337% MACs, 384, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (7): BatchNorm2d(0.0 M, 0.004% Params, 0.0 GMac, 0.007% MACs, 64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (9): InvertedResidual(
      0.054 M, 1.758% Params, 0.02 GMac, 2.995% MACs, 
      (conv): Sequential(
        0.054 M, 1.758% Params, 0.02 GMac, 2.995% MACs, 
        (0): Conv2d(0.025 M, 0.796% Params, 0.009 GMac, 1.337% MACs, 64, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (1): BatchNorm2d(0.001 M, 0.025% Params, 0.0 GMac, 0.042% MACs, 384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU6(0.0 M, 0.000% Params, 0.0 GMac, 0.021% MACs, inplace=True)
        (3): Conv2d(0.003 M, 0.112% Params, 0.001 GMac, 0.188% MACs, 384, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=384, bias=False)
        (4): BatchNorm2d(0.001 M, 0.025% Params, 0.0 GMac, 0.042% MACs, 384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (5): ReLU6(0.0 M, 0.000% Params, 0.0 GMac, 0.021% MACs, inplace=True)
        (6): Conv2d(0.025 M, 0.796% Params, 0.009 GMac, 1.337% MACs, 384, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (7): BatchNorm2d(0.0 M, 0.004% Params, 0.0 GMac, 0.007% MACs, 64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (10): InvertedResidual(
      0.054 M, 1.758% Params, 0.02 GMac, 2.995% MACs, 
      (conv): Sequential(
        0.054 M, 1.758% Params, 0.02 GMac, 2.995% MACs, 
        (0): Conv2d(0.025 M, 0.796% Params, 0.009 GMac, 1.337% MACs, 64, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (1): BatchNorm2d(0.001 M, 0.025% Params, 0.0 GMac, 0.042% MACs, 384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU6(0.0 M, 0.000% Params, 0.0 GMac, 0.021% MACs, inplace=True)
        (3): Conv2d(0.003 M, 0.112% Params, 0.001 GMac, 0.188% MACs, 384, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=384, bias=False)
        (4): BatchNorm2d(0.001 M, 0.025% Params, 0.0 GMac, 0.042% MACs, 384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (5): ReLU6(0.0 M, 0.000% Params, 0.0 GMac, 0.021% MACs, inplace=True)
        (6): Conv2d(0.025 M, 0.796% Params, 0.009 GMac, 1.337% MACs, 384, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (7): BatchNorm2d(0.0 M, 0.004% Params, 0.0 GMac, 0.007% MACs, 64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (11): InvertedResidual(
      0.067 M, 2.158% Params, 0.024 GMac, 3.667% MACs, 
      (conv): Sequential(
        0.067 M, 2.158% Params, 0.024 GMac, 3.667% MACs, 
        (0): Conv2d(0.025 M, 0.796% Params, 0.009 GMac, 1.337% MACs, 64, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (1): BatchNorm2d(0.001 M, 0.025% Params, 0.0 GMac, 0.042% MACs, 384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU6(0.0 M, 0.000% Params, 0.0 GMac, 0.021% MACs, inplace=True)
        (3): Conv2d(0.003 M, 0.112% Params, 0.001 GMac, 0.188% MACs, 384, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=384, bias=False)
        (4): BatchNorm2d(0.001 M, 0.025% Params, 0.0 GMac, 0.042% MACs, 384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (5): ReLU6(0.0 M, 0.000% Params, 0.0 GMac, 0.021% MACs, inplace=True)
        (6): Conv2d(0.037 M, 1.194% Params, 0.013 GMac, 2.006% MACs, 384, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (7): BatchNorm2d(0.0 M, 0.006% Params, 0.0 GMac, 0.010% MACs, 96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (12): InvertedResidual(
      0.118 M, 3.831% Params, 0.043 GMac, 6.498% MACs, 
      (conv): Sequential(
        0.118 M, 3.831% Params, 0.043 GMac, 6.498% MACs, 
        (0): Conv2d(0.055 M, 1.791% Params, 0.02 GMac, 3.009% MACs, 96, 576, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (1): BatchNorm2d(0.001 M, 0.037% Params, 0.0 GMac, 0.063% MACs, 576, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU6(0.0 M, 0.000% Params, 0.0 GMac, 0.031% MACs, inplace=True)
        (3): Conv2d(0.005 M, 0.168% Params, 0.002 GMac, 0.282% MACs, 576, 576, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=576, bias=False)
        (4): BatchNorm2d(0.001 M, 0.037% Params, 0.0 GMac, 0.063% MACs, 576, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (5): ReLU6(0.0 M, 0.000% Params, 0.0 GMac, 0.031% MACs, inplace=True)
        (6): Conv2d(0.055 M, 1.791% Params, 0.02 GMac, 3.009% MACs, 576, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (7): BatchNorm2d(0.0 M, 0.006% Params, 0.0 GMac, 0.010% MACs, 96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (13): InvertedResidual(
      0.118 M, 3.831% Params, 0.043 GMac, 6.498% MACs, 
      (conv): Sequential(
        0.118 M, 3.831% Params, 0.043 GMac, 6.498% MACs, 
        (0): Conv2d(0.055 M, 1.791% Params, 0.02 GMac, 3.009% MACs, 96, 576, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (1): BatchNorm2d(0.001 M, 0.037% Params, 0.0 GMac, 0.063% MACs, 576, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU6(0.0 M, 0.000% Params, 0.0 GMac, 0.031% MACs, inplace=True)
        (3): Conv2d(0.005 M, 0.168% Params, 0.002 GMac, 0.282% MACs, 576, 576, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=576, bias=False)
        (4): BatchNorm2d(0.001 M, 0.037% Params, 0.0 GMac, 0.063% MACs, 576, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (5): ReLU6(0.0 M, 0.000% Params, 0.0 GMac, 0.031% MACs, inplace=True)
        (6): Conv2d(0.055 M, 1.791% Params, 0.02 GMac, 3.009% MACs, 576, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (7): BatchNorm2d(0.0 M, 0.006% Params, 0.0 GMac, 0.010% MACs, 96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (14): InvertedResidual(
      0.155 M, 5.029% Params, 0.031 GMac, 4.601% MACs, 
      (conv): Sequential(
        0.155 M, 5.029% Params, 0.031 GMac, 4.601% MACs, 
        (0): Conv2d(0.055 M, 1.791% Params, 0.02 GMac, 3.009% MACs, 96, 576, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (1): BatchNorm2d(0.001 M, 0.037% Params, 0.0 GMac, 0.063% MACs, 576, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU6(0.0 M, 0.000% Params, 0.0 GMac, 0.031% MACs, inplace=True)
        (3): Conv2d(0.005 M, 0.168% Params, 0.001 GMac, 0.078% MACs, 576, 576, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=576, bias=False)
        (4): BatchNorm2d(0.001 M, 0.037% Params, 0.0 GMac, 0.017% MACs, 576, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (5): ReLU6(0.0 M, 0.000% Params, 0.0 GMac, 0.009% MACs, inplace=True)
        (6): Conv2d(0.092 M, 2.985% Params, 0.009 GMac, 1.389% MACs, 576, 160, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (7): BatchNorm2d(0.0 M, 0.010% Params, 0.0 GMac, 0.005% MACs, 160, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (15): InvertedResidual(
      0.32 M, 10.365% Params, 0.032 GMac, 4.852% MACs, 
      (conv): Sequential(
        0.32 M, 10.365% Params, 0.032 GMac, 4.852% MACs, 
        (0): Conv2d(0.154 M, 4.975% Params, 0.015 GMac, 2.315% MACs, 160, 960, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (1): BatchNorm2d(0.002 M, 0.062% Params, 0.0 GMac, 0.029% MACs, 960, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU6(0.0 M, 0.000% Params, 0.0 GMac, 0.014% MACs, inplace=True)
        (3): Conv2d(0.009 M, 0.280% Params, 0.001 GMac, 0.130% MACs, 960, 960, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=960, bias=False)
        (4): BatchNorm2d(0.002 M, 0.062% Params, 0.0 GMac, 0.029% MACs, 960, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (5): ReLU6(0.0 M, 0.000% Params, 0.0 GMac, 0.014% MACs, inplace=True)
        (6): Conv2d(0.154 M, 4.975% Params, 0.015 GMac, 2.315% MACs, 960, 160, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (7): BatchNorm2d(0.0 M, 0.010% Params, 0.0 GMac, 0.005% MACs, 160, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (16): InvertedResidual(
      0.32 M, 10.365% Params, 0.032 GMac, 4.852% MACs, 
      (conv): Sequential(
        0.32 M, 10.365% Params, 0.032 GMac, 4.852% MACs, 
        (0): Conv2d(0.154 M, 4.975% Params, 0.015 GMac, 2.315% MACs, 160, 960, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (1): BatchNorm2d(0.002 M, 0.062% Params, 0.0 GMac, 0.029% MACs, 960, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU6(0.0 M, 0.000% Params, 0.0 GMac, 0.014% MACs, inplace=True)
        (3): Conv2d(0.009 M, 0.280% Params, 0.001 GMac, 0.130% MACs, 960, 960, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=960, bias=False)
        (4): BatchNorm2d(0.002 M, 0.062% Params, 0.0 GMac, 0.029% MACs, 960, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (5): ReLU6(0.0 M, 0.000% Params, 0.0 GMac, 0.014% MACs, inplace=True)
        (6): Conv2d(0.154 M, 4.975% Params, 0.015 GMac, 2.315% MACs, 960, 160, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (7): BatchNorm2d(0.0 M, 0.010% Params, 0.0 GMac, 0.005% MACs, 160, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (17): InvertedResidual(
      0.474 M, 15.350% Params, 0.048 GMac, 7.173% MACs, 
      (conv): Sequential(
        0.474 M, 15.350% Params, 0.048 GMac, 7.173% MACs, 
        (0): Conv2d(0.154 M, 4.975% Params, 0.015 GMac, 2.315% MACs, 160, 960, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (1): BatchNorm2d(0.002 M, 0.062% Params, 0.0 GMac, 0.029% MACs, 960, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU6(0.0 M, 0.000% Params, 0.0 GMac, 0.014% MACs, inplace=True)
        (3): Conv2d(0.009 M, 0.280% Params, 0.001 GMac, 0.130% MACs, 960, 960, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=960, bias=False)
        (4): BatchNorm2d(0.002 M, 0.062% Params, 0.0 GMac, 0.029% MACs, 960, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (5): ReLU6(0.0 M, 0.000% Params, 0.0 GMac, 0.014% MACs, inplace=True)
        (6): Conv2d(0.307 M, 9.950% Params, 0.031 GMac, 4.631% MACs, 960, 320, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (7): BatchNorm2d(0.001 M, 0.021% Params, 0.0 GMac, 0.010% MACs, 320, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (18): Sequential(
      0.412 M, 13.350% Params, 0.041 GMac, 6.232% MACs, 
      (0): Conv2d(0.41 M, 13.267% Params, 0.041 GMac, 6.174% MACs, 320, 1280, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (1): BatchNorm2d(0.003 M, 0.083% Params, 0.0 GMac, 0.039% MACs, 1280, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU6(0.0 M, 0.000% Params, 0.0 GMac, 0.019% MACs, inplace=True)
    )
  )
  (extras): ModuleList(
    0.653 M, 21.142% Params, 0.039 GMac, 5.833% MACs, 
    (0): InvertedResidual(
      0.463 M, 15.000% Params, 0.036 GMac, 5.460% MACs, 
      (conv): Sequential(
        0.463 M, 15.000% Params, 0.036 GMac, 5.460% MACs, 
        (0): Conv2d(0.328 M, 10.614% Params, 0.033 GMac, 4.939% MACs, 1280, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (1): BatchNorm2d(0.001 M, 0.017% Params, 0.0 GMac, 0.008% MACs, 256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU6(0.0 M, 0.000% Params, 0.0 GMac, 0.004% MACs, inplace=True)
        (3): Conv2d(0.002 M, 0.075% Params, 0.0 GMac, 0.009% MACs, 256, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=256, bias=False)
        (4): BatchNorm2d(0.001 M, 0.017% Params, 0.0 GMac, 0.002% MACs, 256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (5): ReLU6(0.0 M, 0.000% Params, 0.0 GMac, 0.001% MACs, inplace=True)
        (6): Conv2d(0.131 M, 4.245% Params, 0.003 GMac, 0.494% MACs, 256, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (7): BatchNorm2d(0.001 M, 0.033% Params, 0.0 GMac, 0.004% MACs, 512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (1): InvertedResidual(
      0.1 M, 3.255% Params, 0.002 GMac, 0.296% MACs, 
      (conv): Sequential(
        0.1 M, 3.255% Params, 0.002 GMac, 0.296% MACs, 
        (0): Conv2d(0.066 M, 2.123% Params, 0.002 GMac, 0.247% MACs, 512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (1): BatchNorm2d(0.0 M, 0.008% Params, 0.0 GMac, 0.001% MACs, 128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU6(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, inplace=True)
        (3): Conv2d(0.001 M, 0.037% Params, 0.0 GMac, 0.002% MACs, 128, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=128, bias=False)
        (4): BatchNorm2d(0.0 M, 0.008% Params, 0.0 GMac, 0.000% MACs, 128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (5): ReLU6(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, inplace=True)
        (6): Conv2d(0.033 M, 1.061% Params, 0.0 GMac, 0.044% MACs, 128, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (7): BatchNorm2d(0.001 M, 0.017% Params, 0.0 GMac, 0.001% MACs, 256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (2): InvertedResidual(
      0.068 M, 2.193% Params, 0.0 GMac, 0.066% MACs, 
      (conv): Sequential(
        0.068 M, 2.193% Params, 0.0 GMac, 0.066% MACs, 
        (0): Conv2d(0.033 M, 1.061% Params, 0.0 GMac, 0.044% MACs, 256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (1): BatchNorm2d(0.0 M, 0.008% Params, 0.0 GMac, 0.000% MACs, 128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU6(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, inplace=True)
        (3): Conv2d(0.001 M, 0.037% Params, 0.0 GMac, 0.001% MACs, 128, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=128, bias=False)
        (4): BatchNorm2d(0.0 M, 0.008% Params, 0.0 GMac, 0.000% MACs, 128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (5): ReLU6(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, inplace=True)
        (6): Conv2d(0.033 M, 1.061% Params, 0.0 GMac, 0.020% MACs, 128, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (7): BatchNorm2d(0.001 M, 0.017% Params, 0.0 GMac, 0.000% MACs, 256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (3): InvertedResidual(
      0.021 M, 0.694% Params, 0.0 GMac, 0.011% MACs, 
      (conv): Sequential(
        0.021 M, 0.694% Params, 0.0 GMac, 0.011% MACs, 
        (0): Conv2d(0.016 M, 0.531% Params, 0.0 GMac, 0.010% MACs, 256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (1): BatchNorm2d(0.0 M, 0.004% Params, 0.0 GMac, 0.000% MACs, 64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU6(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, inplace=True)
        (3): Conv2d(0.001 M, 0.019% Params, 0.0 GMac, 0.000% MACs, 64, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=64, bias=False)
        (4): BatchNorm2d(0.0 M, 0.004% Params, 0.0 GMac, 0.000% MACs, 64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (5): ReLU6(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, inplace=True)
        (6): Conv2d(0.004 M, 0.133% Params, 0.0 GMac, 0.001% MACs, 64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (7): BatchNorm2d(0.0 M, 0.004% Params, 0.0 GMac, 0.000% MACs, 64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
  )
  (classification_headers): ModuleList(
    0.105 M, 3.413% Params, 0.013 GMac, 1.966% MACs, 
    (0): Sequential(
      0.021 M, 0.672% Params, 0.008 GMac, 1.161% MACs, 
      (0): Conv2d(0.006 M, 0.187% Params, 0.002 GMac, 0.313% MACs, 576, 576, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=576)
      (1): BatchNorm2d(0.001 M, 0.037% Params, 0.0 GMac, 0.063% MACs, 576, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU6(0.0 M, 0.000% Params, 0.0 GMac, 0.031% MACs, )
      (3): Conv2d(0.014 M, 0.449% Params, 0.005 GMac, 0.754% MACs, 576, 24, kernel_size=(1, 1), stride=(1, 1))
    )
    (1): Sequential(
      0.046 M, 1.493% Params, 0.005 GMac, 0.714% MACs, 
      (0): Conv2d(0.013 M, 0.415% Params, 0.001 GMac, 0.193% MACs, 1280, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=1280)
      (1): BatchNorm2d(0.003 M, 0.083% Params, 0.0 GMac, 0.039% MACs, 1280, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU6(0.0 M, 0.000% Params, 0.0 GMac, 0.019% MACs, )
      (3): Conv2d(0.031 M, 0.996% Params, 0.003 GMac, 0.463% MACs, 1280, 24, kernel_size=(1, 1), stride=(1, 1))
    )
    (2): Sequential(
      0.018 M, 0.598% Params, 0.0 GMac, 0.071% MACs, 
      (0): Conv2d(0.005 M, 0.166% Params, 0.0 GMac, 0.019% MACs, 512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=512)
      (1): BatchNorm2d(0.001 M, 0.033% Params, 0.0 GMac, 0.004% MACs, 512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU6(0.0 M, 0.000% Params, 0.0 GMac, 0.002% MACs, )
      (3): Conv2d(0.012 M, 0.399% Params, 0.0 GMac, 0.046% MACs, 512, 24, kernel_size=(1, 1), stride=(1, 1))
    )
    (3): Sequential(
      0.009 M, 0.299% Params, 0.0 GMac, 0.013% MACs, 
      (0): Conv2d(0.003 M, 0.083% Params, 0.0 GMac, 0.003% MACs, 256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=256)
      (1): BatchNorm2d(0.001 M, 0.017% Params, 0.0 GMac, 0.001% MACs, 256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU6(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, )
      (3): Conv2d(0.006 M, 0.200% Params, 0.0 GMac, 0.008% MACs, 256, 24, kernel_size=(1, 1), stride=(1, 1))
    )
    (4): Sequential(
      0.009 M, 0.299% Params, 0.0 GMac, 0.006% MACs, 
      (0): Conv2d(0.003 M, 0.083% Params, 0.0 GMac, 0.002% MACs, 256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=256)
      (1): BatchNorm2d(0.001 M, 0.017% Params, 0.0 GMac, 0.000% MACs, 256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU6(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, )
      (3): Conv2d(0.006 M, 0.200% Params, 0.0 GMac, 0.004% MACs, 256, 24, kernel_size=(1, 1), stride=(1, 1))
    )
    (5): Conv2d(0.002 M, 0.051% Params, 0.0 GMac, 0.000% MACs, 64, 24, kernel_size=(1, 1), stride=(1, 1))
  )
  (regression_headers): ModuleList(
    0.105 M, 3.413% Params, 0.013 GMac, 1.966% MACs, 
    (0): Sequential(
      0.021 M, 0.672% Params, 0.008 GMac, 1.161% MACs, 
      (0): Conv2d(0.006 M, 0.187% Params, 0.002 GMac, 0.313% MACs, 576, 576, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=576)
      (1): BatchNorm2d(0.001 M, 0.037% Params, 0.0 GMac, 0.063% MACs, 576, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU6(0.0 M, 0.000% Params, 0.0 GMac, 0.031% MACs, )
      (3): Conv2d(0.014 M, 0.449% Params, 0.005 GMac, 0.754% MACs, 576, 24, kernel_size=(1, 1), stride=(1, 1))
    )
    (1): Sequential(
      0.046 M, 1.493% Params, 0.005 GMac, 0.714% MACs, 
      (0): Conv2d(0.013 M, 0.415% Params, 0.001 GMac, 0.193% MACs, 1280, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=1280)
      (1): BatchNorm2d(0.003 M, 0.083% Params, 0.0 GMac, 0.039% MACs, 1280, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU6(0.0 M, 0.000% Params, 0.0 GMac, 0.019% MACs, )
      (3): Conv2d(0.031 M, 0.996% Params, 0.003 GMac, 0.463% MACs, 1280, 24, kernel_size=(1, 1), stride=(1, 1))
    )
    (2): Sequential(
      0.018 M, 0.598% Params, 0.0 GMac, 0.071% MACs, 
      (0): Conv2d(0.005 M, 0.166% Params, 0.0 GMac, 0.019% MACs, 512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=512)
      (1): BatchNorm2d(0.001 M, 0.033% Params, 0.0 GMac, 0.004% MACs, 512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU6(0.0 M, 0.000% Params, 0.0 GMac, 0.002% MACs, )
      (3): Conv2d(0.012 M, 0.399% Params, 0.0 GMac, 0.046% MACs, 512, 24, kernel_size=(1, 1), stride=(1, 1))
    )
    (3): Sequential(
      0.009 M, 0.299% Params, 0.0 GMac, 0.013% MACs, 
      (0): Conv2d(0.003 M, 0.083% Params, 0.0 GMac, 0.003% MACs, 256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=256)
      (1): BatchNorm2d(0.001 M, 0.017% Params, 0.0 GMac, 0.001% MACs, 256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU6(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, )
      (3): Conv2d(0.006 M, 0.200% Params, 0.0 GMac, 0.008% MACs, 256, 24, kernel_size=(1, 1), stride=(1, 1))
    )
    (4): Sequential(
      0.009 M, 0.299% Params, 0.0 GMac, 0.006% MACs, 
      (0): Conv2d(0.003 M, 0.083% Params, 0.0 GMac, 0.002% MACs, 256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=256)
      (1): BatchNorm2d(0.001 M, 0.017% Params, 0.0 GMac, 0.000% MACs, 256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU6(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, )
      (3): Conv2d(0.006 M, 0.200% Params, 0.0 GMac, 0.004% MACs, 256, 24, kernel_size=(1, 1), stride=(1, 1))
    )
    (5): Conv2d(0.002 M, 0.051% Params, 0.0 GMac, 0.000% MACs, 64, 24, kernel_size=(1, 1), stride=(1, 1))
  )
  (source_layer_add_ons): ModuleList(0.0 M, 0.000% Params, 0.0 GMac, 0.000% MACs, )
)
Computational complexity:       0.66 GMac
Number of parameters:           3.09 M
```

