"""Region-Enhanced Segmentation Loss (RESL), Section 3.4, Eq. (12)-(14).

RESL combines a pixel-wise cross-entropy term with a region-level Dice
term:

    L_RESL = alpha * L_CE + (1 - alpha) * L_Dice                 (Eq. 14)

The combination encourages both pixel-wise accuracy and region-level
overlap. alpha = 0.5 gives the best performance in our experiments
(Table 5 of the paper).

NOTE: the original experiment script fine-tuned the network with a
different, experiment-internal loss pair. This module implements the
RESL exactly as described in the paper and is what ``train_seg.py``
optimises.
"""

import torch
import torch.nn as nn


class CELoss(nn.Module):
    """Pixel-wise cross-entropy over one-hot targets (Eq. (12))."""

    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, y_pred, y_true):
        # y_pred: [B, C, H, W] class probabilities; y_true: one-hot [B, C, H, W]
        y_pred = torch.clamp(y_pred, self.eps, 1.0 - self.eps)
        n = y_true.shape[0] * y_true.shape[2] * y_true.shape[3]
        return -torch.sum(y_true * torch.log(y_pred + self.eps)) / n


class DiceLoss(nn.Module):
    """Soft Dice loss over all classes (Eq. (13))."""

    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps  # numerical stability constant in the paper formula

    def forward(self, y_pred, y_true):
        # flatten batch and spatial dimensions per class
        dims = (0, 2, 3)
        intersection = torch.sum(y_pred * y_true, dims)
        cardinality = torch.sum(y_pred, dims) + torch.sum(y_true, dims)
        dice = (2.0 * intersection + self.eps) / (cardinality + self.eps)
        return 1.0 - dice.mean()


class RESLoss(nn.Module):
    """Region-Enhanced Segmentation Loss (Eq. (14)).

    Args:
        alpha: balancing weight between cross-entropy and Dice
            (alpha = 0.5 in the paper).
    """

    def __init__(self, alpha=0.5):
        super().__init__()
        self.alpha = alpha
        self.ce = CELoss()
        self.dice = DiceLoss()

    def forward(self, y_pred, y_true):
        return self.alpha * self.ce(y_pred, y_true) \
            + (1.0 - self.alpha) * self.dice(y_pred, y_true)
