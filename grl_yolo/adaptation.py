"""Training-only spatial domain classifiers; detector inference stays standard."""
import torch
from torch import nn
from torch.nn import functional as F


class _Reverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, coefficient):
        ctx.coefficient = coefficient
        return x.view_as(x)

    @staticmethod
    def backward(ctx, gradient):
        return -ctx.coefficient * gradient, None


def reverse_gradient(x, coefficient=0.1):
    return _Reverse.apply(x, coefficient)


def capture_features(model, inputs, layers=(4, 6, 10)):
    """Capture backbone P3/P4/P5 without persistent hooks in saved models."""
    features = {}

    def hook(index):
        def capture(module, args, output):
            features[index] = output
        return capture

    handles = [model.model[i].register_forward_hook(hook(i)) for i in layers]
    try:
        result = model(inputs)
    finally:
        for handle in handles:
            handle.remove()
    return result, [features[i] for i in layers]


class MultiscaleDomainHeads(nn.Module):
    def __init__(self, channels, hidden_channels=None, norm="group", groups=8):
        super().__init__()
        hidden_channels = hidden_channels or [max(32, c // 2) for c in channels]
        if len(hidden_channels) != len(channels):
            raise ValueError("hidden_channels must contain one width per GRL feature layer")
        self.heads = nn.ModuleList()
        for c, hidden in zip(channels, hidden_channels):
            if norm == "group":
                normalization = nn.GroupNorm(groups, hidden)
            elif norm == "batch":
                normalization = nn.BatchNorm2d(hidden)
            else:
                raise ValueError("GRL norm must be 'group' or 'batch'")
            self.heads.append(nn.Sequential(nn.Conv2d(c, hidden, 1), normalization,
                                            nn.LeakyReLU(0.1), nn.Conv2d(hidden, 1, 1)))

    def forward(self, source, target, coefficient=0.1):
        if len(source) != len(self.heads) or len(target) != len(self.heads):
            raise ValueError("One source and target feature map is required per scale")
        losses = []
        for head, src, tgt in zip(self.heads, source, target):
            # Match original domain convention: source=1, target=0.
            s = head(reverse_gradient(src, coefficient))
            t = head(reverse_gradient(tgt, coefficient))
            losses.append(0.5 * (
                F.binary_cross_entropy_with_logits(s, torch.ones_like(s))
                + F.binary_cross_entropy_with_logits(t, torch.zeros_like(t))
            ))
        return torch.stack(losses).mean()
