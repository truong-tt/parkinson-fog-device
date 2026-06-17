import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal cross-entropy for the last-step and dense heads.

    Accepts both head shapes: ``(B, C)`` logits with ``(B,)`` targets
    (last-step head), or ``(B, C, T)`` with ``(B, T)`` (dense per-timestep head).

    Args:
        alpha: Optional per-class weight tensor of shape ``(C,)``.
        gamma: Focusing exponent; larger values down-weight easy examples.
        reduction: One of ``'mean'``, ``'sum'``, ``'none'``.
    """

    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        if inputs.dim() == 3:
            B, C, T = inputs.shape
            inputs_flat = inputs.permute(0, 2, 1).reshape(B * T, C)
            targets_flat = targets.reshape(B * T)
        else:
            inputs_flat = inputs
            targets_flat = targets

        ce_loss = F.cross_entropy(inputs_flat, targets_flat, reduction='none')
        pt = torch.exp(-ce_loss)
        loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.alpha is not None:
            if self.alpha.device != inputs_flat.device:
                self.alpha = self.alpha.to(inputs_flat.device)
            loss = loss * self.alpha.gather(0, targets_flat.view(-1))

        if self.reduction == 'mean':
            return torch.mean(loss)
        if self.reduction == 'sum':
            return torch.sum(loss)
        return loss
