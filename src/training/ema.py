"""Exponential moving average (Polyak averaging) of model weights.

The shadow weights are used for validation and saved to disk; the live model is
the one optimized by SGD. Acts as a cheap regularizer and smooths the
validation curve.
"""

import copy
import torch


class ModelEMA:
    """Maintains an EMA copy of a model's weights.

    Args:
        model: Model whose weights to track.
        decay: EMA decay; higher means slower tracking of the live weights.
    """

    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        """Blend the live model's weights into the shadow copy in place."""
        d = self.decay
        msd = model.state_dict()
        for k, v in self.shadow.state_dict().items():
            mv = msd[k]
            if v.dtype.is_floating_point:
                v.mul_(d).add_(mv.detach(), alpha=1.0 - d)
            else:
                v.copy_(mv)  # integer buffers (e.g. counters): copy as-is

    def state_dict(self):
        return self.shadow.state_dict()
