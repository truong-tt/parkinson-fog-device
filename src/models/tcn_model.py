"""Causal TCN building blocks and the HopeGaitTCN model.

All components are strictly causal (output at step ``t`` never depends on
``t+1``) and batch-size-1 friendly, so the offline training graph matches live
streaming inference on the MCU.
"""

import torch
import torch.nn as nn
from torch.ao.quantization import QuantStub, DeQuantStub


class TimeWiseLayerNorm(nn.Module):
    """LayerNorm over channels only, applied per-timestep.

    For input ``(B, C, T)`` each timestep is normalized using its own ``C``
    channels, so no statistics leak across time and the dense head stays causal
    (unlike BatchNorm/GroupNorm/InstanceNorm, which pool over ``T``). Also
    batch-size-1 friendly for MCU inference.
    """

    def __init__(self, channels):
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        return self.norm(x.transpose(1, 2)).transpose(1, 2)


def _gn(channels):  # name kept for call-site stability after the GroupNorm -> LayerNorm refactor
    return TimeWiseLayerNorm(channels)


class Chomp1d(nn.Module):
    # Trims right-side padding so the conv is strictly causal (no future leak).
    def __init__(self, chomp_size):
        super().__init__()
        # x[:, :, :-0] returns an empty time axis, not the full tensor — guard
        # against a misconfigured kernel/dilation that would silently break shape.
        if chomp_size <= 0:
            raise ValueError(
                f"Chomp1d requires chomp_size > 0, got {chomp_size}. "
                "Check that (kernel_size - 1) * dilation > 0."
            )
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class CausalSqueezeExcite1d(nn.Module):
    """Squeeze-and-excitation using a causal cumulative mean.

    Standard SE pools globally over time (``x.mean(dim=2)``), leaking the future
    when training the dense head. A running mean keeps each timestep's gating
    dependent only on the past; at the last step it equals standard SE (the
    cumulative mean over the full window is the global mean).

    Args:
        channels: Number of input/output channels.
        reduction: Bottleneck channel-reduction ratio.
    """

    def __init__(self, channels, reduction=8):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.fc1 = nn.Conv1d(channels, hidden, 1)
        self.fc2 = nn.Conv1d(hidden, channels, 1)

    def forward(self, x):
        # x: (B, C, T). cumsum along T, divide by 1..T to get a running mean.
        T = x.size(2)
        denom = torch.arange(1, T + 1, device=x.device, dtype=x.dtype).view(1, 1, T)
        s = torch.cumsum(x, dim=2) / denom
        s = torch.relu(self.fc1(s))
        s = torch.sigmoid(self.fc2(s))
        return x * s


class TemporalBlock(nn.Module):
    """Residual block: two causal dilated convs, norm, SE, and a skip connection."""

    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding,
                 dropout=0.2, drop_path=0.0, use_se=True):
        super().__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride,
                               padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.norm1 = _gn(n_outputs)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride,
                               padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.norm2 = _gn(n_outputs)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.se = CausalSqueezeExcite1d(n_outputs) if use_se else nn.Identity()

        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.norm1, self.relu1, self.dropout1,
            self.conv2, self.chomp2, self.norm2, self.relu2, self.dropout2,
            self.se,
        )
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.drop_path = drop_path

    def _drop_path(self, x):
        # Stochastic depth: zero the residual branch for whole batches at training time.
        if not self.training or self.drop_path <= 0.0:
            return x
        keep = 1.0 - self.drop_path
        mask = x.new_empty(x.size(0), 1, 1).bernoulli_(keep) / keep
        return x * mask

    def forward(self, x):
        out = self.net(x)
        out = self._drop_path(out)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class HopeGaitTCN(nn.Module):
    """Causal TCN with a last-step head and a dense per-timestep head.

    ``forward(x)`` returns last-step logits (what the MCU runs); ``forward_dense``
    returns both heads (the dense head gives denser training supervision). With 4
    blocks, exponential dilation ``(1, 2, 4, 8)`` and ``kernel_size=3`` the
    receptive field is ``1 + 2*(k-1)*sum(dilations) = 61`` samples (~0.95 s at
    64 Hz).

    Args:
        num_inputs: Input channels per timestep (9 for this pipeline).
        num_channels: Output channels per temporal block.
        kernel_size: Convolution kernel size.
        num_classes: Output classes (2: walk vs freeze).
        dropout: Dropout rate inside each block.
        drop_path: Max stochastic-depth rate, scaled linearly across blocks.
        use_se: Enable causal squeeze-and-excitation.
    """

    def __init__(self, num_inputs=9, num_channels=(32, 64, 96, 128), kernel_size=3,
                 num_classes=2, dropout=0.3, drop_path=0.1, use_se=True):
        super().__init__()
        # Quant/DeQuant stubs are no-ops in FP32; enabled by QAT in the edge phase.
        self.quant = QuantStub()
        self.dequant = DeQuantStub()

        layers = []
        n_blocks = len(num_channels)
        for i, ch in enumerate(num_channels):
            # Exponential dilation gives an exponential receptive field.
            dilation = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            # Linearly scale stochastic depth across blocks (deeper => more drop).
            dp = drop_path * i / max(n_blocks - 1, 1)
            layers.append(TemporalBlock(
                in_channels, ch, kernel_size, stride=1,
                dilation=dilation, padding=(kernel_size - 1) * dilation,
                dropout=dropout, drop_path=dp, use_se=use_se,
            ))
        self.network = nn.Sequential(*layers)
        # Single 1x1 conv shared by both heads — keeps causality, parameter-efficient.
        self.head = nn.Conv1d(num_channels[-1], num_classes, 1)

    def _features(self, x):
        x = self.quant(x)
        # (B, T, C) -> (B, C, T) for Conv1d
        x = x.transpose(1, 2)
        return self.network(x)

    def forward(self, x):
        feats = self._features(x)
        logits = self.head(feats)
        # Last-step logits for real-time inference.
        return self.dequant(logits[:, :, -1])

    def forward_dense(self, x):
        """Returns (last_logits: (B, C), dense_logits: (B, C, T))."""
        feats = self._features(x)
        logits = self.head(feats)
        last = self.dequant(logits[:, :, -1])
        dense = self.dequant(logits)
        return last, dense
