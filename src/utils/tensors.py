import math

import torch


def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    """
    I fill `tensor` with values drawn from a truncated normal distribution.
    Values will lie between `a` and `b`, roughly following N(mean, std).

    Args:
        tensor (torch.Tensor): The tensor to fill in-place.
        mean (float): mean of the normal distribution
        std (float): std of the normal distribution
        a (float): lower bound of truncation
        b (float): upper bound of truncation
    """

    # I use the official PyTorch implementation if available (in newer versions),
    # but if not, I define my own method:
    def norm_cdf(x):
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
    return tensor
