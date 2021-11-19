import torch
import math

class Conv2dSamePadding(torch.nn.Conv2d):
    """
    A wrapper around :class:`torch.nn.Conv2d` to support "SAME" padding mode and more features.
    """

    def __init__(self, *args, **kwargs):
        """
        Extra keyword arguments supported in addition to those in `torch.nn.Conv2d`:

        Args:
            norm (nn.Module, optional): a normalization layer
            activation (callable(Tensor) -> Tensor): a callable activation function

        It assumes that norm layer is used before activation.
        """
        norm = kwargs.pop("norm", None)
        activation = kwargs.pop("activation", None)

        # parse padding mode
        self.padding_method = kwargs.pop("padding", None)
        if self.padding_method is None:
            if len(args) >= 5:
                self.padding_method = args[4]
            else:
                self.padding_method = 0  # default padding number

        if isinstance(self.padding_method, str):
            if self.padding_method.upper() == "SAME":
                # If the padding mode is `SAME`, it will be manually padded
                super().__init__(*args, **kwargs, padding=0)
                # stride
                if isinstance(self.stride, int):
                    self.stride = [self.stride] * 2
                elif len(self.stride) == 1:
                    self.stride = [self.stride[0]] * 2
                # kernel size
                if isinstance(self.kernel_size, int):
                    self.kernel_size = [self.kernel_size] * 2
                elif len(self.kernel_size) == 1:
                    self.kernel_size = [self.kernel_size[0]] * 2
                # dilation
                if isinstance(self.dilation, int):
                    self.dilation = [self.dilation] * 2
                elif len(self.dilation) == 1:
                    self.dilation = [self.dilation[0]] * 2
            else:
                raise ValueError("Unknown padding method: {}".format(self.padding_method))
        else:
            super().__init__(*args, **kwargs, padding=self.padding_method)
        self.norm = norm
        self.activation = activation

    def forward(self, x):
        if isinstance(self.padding_method, str):
            if self.padding_method.upper() == "SAME":
                input_h, input_w = x.shape[-2:]
                stride_h, stride_w = self.stride
                kernel_size_h, kernel_size_w = self.kernel_size
                dilation_h, dilation_w = self.dilation

                output_h = math.ceil(input_h / stride_h)
                output_w = math.ceil(input_w / stride_w)

                padding_needed_h = max(
                    0, (output_h - 1) * stride_h + (kernel_size_h - 1) * dilation_h + 1 - input_h
                )
                padding_needed_w = max(
                    0, (output_w - 1) * stride_w + (kernel_size_w - 1) * dilation_w + 1 - input_w
                )

                left = padding_needed_w // 2
                right = padding_needed_w - left
                top = padding_needed_h // 2
                bottom = padding_needed_h - top

                x = F.pad(x, [left, right, top, bottom])
            else:
                raise ValueError("Unknown padding method: {}".format(self.padding_method))

        x = super().forward(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x