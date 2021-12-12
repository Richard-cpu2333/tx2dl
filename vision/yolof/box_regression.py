import math
from typing import Tuple

import torch

_DEFAULT_SCALE_CLAMP = math.log(1000.0 / 16)


class YOLOFBox2BoxTransform(object):
    """
    The box-to-box transform defined in R-CNN. The transformation is
    parameterized by 4 deltas: (dx, dy, dw, dh). The transformation scales
    the box's width and height by exp(dw), exp(dh) and shifts a box's center
    by the offset (dx * width, dy * height).

    We add center clamp for the predict boxes.
    """

    def __init__(
            self,
            weights: Tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
            scale_clamp: float = _DEFAULT_SCALE_CLAMP,
            add_ctr_clamp: bool = False,
            ctr_clamp: int = 32
    ):
        """
        Args:
            weights (4-element tuple): Scaling factors that are applied to the
                (dx, dy, dw, dh) deltas. In Fast R-CNN, these were originally
                set such that the deltas have unit variance; now they are
                treated as hyperparameters of the system.
            scale_clamp (float): When predicting deltas, the predicted box
                scaling factors (dw and dh) are clamped such that they are
                <= scale_clamp.
            add_ctr_clamp (bool): Whether to add center clamp, when added, the
                predicted box is clamped is its center is too far away from
                the original anchor's center.
            ctr_clamp (int): the maximum pixel shift to clamp.

        """
        self.weights = weights
        self.scale_clamp = scale_clamp
        self.add_ctr_clamp = add_ctr_clamp
        self.ctr_clamp = ctr_clamp
        self.eps = 1e-4

    def get_deltas(self, src_boxes, target_boxes):
        """
        Get box regression transformation deltas (dx, dy, dw, dh) that can be
        used to transform the `src_boxes` into the `target_boxes`. That is,
        the relation ``target_boxes == self.apply_deltas(deltas,
        src_boxes)`` is true (unless any delta is too large and is clamped).

        Args:
            src_boxes (Tensor): source boxes, e.g., object proposals
            target_boxes (Tensor): target of the transformation, e.g.,
                ground-truth boxes.
        """
        assert isinstance(src_boxes, torch.Tensor), type(src_boxes)
        assert isinstance(target_boxes, torch.Tensor), type(target_boxes)

        src_widths = src_boxes[..., 2] - src_boxes[..., 0]
        src_heights = src_boxes[..., 3] - src_boxes[..., 1]
        src_ctr_x = src_boxes[..., 0] + 0.5 * src_widths
        src_ctr_y = src_boxes[..., 1] + 0.5 * src_heights

        target_widths = target_boxes[..., 2] - target_boxes[..., 0]
        target_heights = target_boxes[..., 3] - target_boxes[..., 1]
        target_ctr_x = target_boxes[..., 0] + 0.5 * target_widths
        target_ctr_y = target_boxes[..., 1] + 0.5 * target_heights

        wx, wy, ww, wh = self.weights
        
        dx = wx * (target_ctr_x - src_ctr_x) / src_widths
        dy = wy * (target_ctr_y - src_ctr_y) / src_heights
        dw = ww * torch.log(target_widths / src_widths)
        dh = wh * torch.log(target_heights / src_heights)

        deltas = torch.stack((dx, dy, dw, dh), dim=-1)
        assert (src_widths > 0).all().item(), \
            "Input boxes to Box2BoxTransform are not valid!"
        return deltas

    def apply_deltas(self, deltas, boxes):
        """
        Apply transformation `deltas` (dx, dy, dw, dh) to `boxes`.

        Args:
            deltas (Tensor): transformation deltas of shape (N, k*4),
                where k >= 1. deltas[i] represents k potentially different
                class-specific box transformations for the single box boxes[i].
            boxes (Tensor): boxes to transform, of shape (N, 4)
        """
        # print(f"deltas' device:{deltas.device}") ## cuda:0
        # print(f"boxes' device:{boxes.device}") 
        # print(f"boxes' shape:{boxes.shape}")     ## boxes' shape:torch.Size([batch_size*400, 4])
        # print(f"deltas' shape:{deltas.shape}")   ## deltas' shape:torch.Size([batch_size*400, 4])]
        deltas = deltas.float().to("cpu") # ensure fp32 for decoding precision
        N = deltas.shape[0]
        # boxes = torch.tile(boxes[None], [N,1,1])
        deltas = deltas.view(-1, 4)
        boxes = boxes.view(-1, 4)
        # boxes = boxes.to("cuda")

        widths = boxes[..., 2] - boxes[..., 0]
        heights = boxes[..., 3] - boxes[..., 1]
        ctr_x = boxes[..., 0] + 0.5 * widths
        ctr_y = boxes[..., 1] + 0.5 * heights

        wx, wy, ww, wh = self.weights
        dx = deltas[..., 0::4] / wx
        dy = deltas[..., 1::4] / wy
        dw = deltas[..., 2::4] / ww
        dh = deltas[..., 3::4] / wh
        
        # Prevent sending too large values into torch.exp()
        dx_width = dx * widths[..., None]
        dy_height = dy * heights[..., None]
        if self.add_ctr_clamp:
            dx_width = torch.clamp(dx_width,
                                   max=self.ctr_clamp,
                                   min=-self.ctr_clamp)
            dy_height = torch.clamp(dy_height,
                                    max=self.ctr_clamp,
                                    min=-self.ctr_clamp)
        dw = torch.clamp(dw, max=self.scale_clamp)
        dh = torch.clamp(dh, max=self.scale_clamp)

        pred_ctr_x = dx_width + ctr_x[..., None]
        pred_ctr_y = dy_height + ctr_y[..., None]
        pred_w = torch.exp(dw) * widths[..., None]
        pred_h = torch.exp(dh) * heights[..., None]

        x1 = pred_ctr_x - 0.5 * pred_w
        y1 = pred_ctr_y - 0.5 * pred_h
        x2 = pred_ctr_x + 0.5 * pred_w
        y2 = pred_ctr_y + 0.5 * pred_h
        pred_boxes = torch.stack((x1, y1, x2, y2), dim=-1)
        pred_boxes = pred_boxes.to("cuda")
        deltas = deltas.reshape(N, -1, 4)
        return pred_boxes.reshape(deltas.shape)


def clip_by_tensor(t,t_min,t_max):
    """
    clip_by_tensor
    :param t: tensor
    :param t_min: min
    :param t_max: max
    :return: cliped tensor
    """
    t=t.float()
    t_min=t_min.float()
    t_max=t_max.float()
 
    result = (t >= t_min).float() * t + (t < t_min).float() * t_min
    result = (result <= t_max).float() * result + (result > t_max).float() * t_max
    return result