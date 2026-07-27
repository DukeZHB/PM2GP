"""Experimental segmentation losses from the original research code base.

These losses were explored during development but are NOT used by the
final PM2GP pipeline - segmentation training uses RESL
(``segmentation/resl.py``). They are preserved here for reference and
ablation reproduction.

All losses expect softmaxed probabilities: ``y_pred`` and ``y_true``
with shape ``N x C x H x W``.
"""

import math

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F


class FLoss(nn.Module):
    """Focal loss on softmaxed probabilities."""

    def __init__(self, gamma, weight=1.0, eps=1e-8):
        super(FLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight
        self.eps = eps  # numerical stability

    def forward(self, y_pred, y_true):
        # Clamp predictions into the valid range.
        y_pred = torch.clamp(y_pred, self.eps, 1.0 - self.eps)

        N = y_true.shape[0] * y_true.shape[2] * y_true.shape[3]
        # Focal loss; eps guards against log(0).
        t2 = -self.weight * torch.pow((1 - y_pred), self.gamma) * y_true * torch.log(y_pred + self.eps)
        loss = torch.sum(t2) / N

        return loss


class CELoss(nn.Module):
    """Cross-entropy on softmaxed probabilities."""

    def __init__(self, weight=None, eps=1e-8):
        super(CELoss, self).__init__()
        self.weights = weight
        self.eps = eps

    def forward(self, y_pred, y_true):
        y_pred = torch.clamp(y_pred, self.eps, 1.0 - self.eps)

        N = y_true.shape[0] * y_true.shape[2] * y_true.shape[3]
        loss = -torch.sum(y_true * torch.log(y_pred + self.eps)) / N

        return loss


class SSLoss(nn.Module):
    """Spatial-structure loss: penalizes normalized prediction errors at
    locations whose deviation exceeds a per-map adaptive threshold."""

    def __init__(self, beta=0.1, C=0.01, eps=1e-8):
        super(SSLoss, self).__init__()
        self.beta = beta
        self.C = C
        self.eps = eps
        self.LCE = CELoss(eps=eps)

    def forward(self, y_pred, y_true):
        y_pred = torch.clamp(y_pred, self.eps, 1.0 - self.eps)

        # Mean/std normalization; eps avoids division by zero.
        mean_true = torch.mean(y_true, (2, 3), keepdim=True)
        std_true = torch.std(y_true, (2, 3), keepdim=True) + self.eps
        mean_pred = torch.mean(y_pred, (2, 3), keepdim=True)
        std_pred = torch.std(y_pred, (2, 3), keepdim=True) + self.eps

        e1 = (y_true - mean_true + self.C) / (std_true + self.C)
        e2 = (y_pred - mean_pred + self.C) / (std_pred + self.C)
        e = torch.abs(e1 - e2)

        e_max, _ = torch.max(torch.flatten(e, start_dim=2), dim=2, keepdim=True)
        e_max = torch.unsqueeze(e_max, dim=2)
        f = (e > (self.beta * e_max)).float()

        lce = self.LCE(y_pred, y_true)

        loss = e * f * lce
        M = torch.sum(f) + self.eps  # eps avoids division by zero

        ssl_loss = torch.sum(loss) / M

        return ssl_loss


def tversky_coefficient(y_true, y_predict, smooth=1.0, beta=0.3, eps=1e-8):
    intersection = torch.sum(y_true * y_predict)
    i1 = beta * torch.sum((1 - y_true) * y_predict)
    i2 = (1 - beta) * torch.sum(y_true * (1 - y_predict))
    return (intersection + smooth) / (intersection + i1 + i2 + smooth + eps)


class Tversky_Loss(nn.Module):
    def __init__(self, beta, eps=1e-8):
        super(Tversky_Loss, self).__init__()
        self.beta = beta
        self.eps = eps

    def forward(self, y_predict, y_true):
        y_predict = torch.clamp(y_predict, self.eps, 1.0 - self.eps)

        tversky = 0.0
        N = y_true.shape[0]
        for i in range(y_true.shape[0]):
            tversky += (1 - tversky_coefficient(y_true[i], y_predict[i], beta=self.beta, eps=self.eps))
        loss = tversky / N

        return loss


class CosineLoss(nn.Module):
    def __init__(self, eps=1e-8):
        super(CosineLoss, self).__init__()
        self.eps = eps

    def forward(self, y_predict, y_true):
        y_predict = torch.clamp(y_predict, self.eps, 1.0 - self.eps)

        N = y_true.shape[0] * y_true.shape[2] * y_true.shape[3] + self.eps
        product_sum = torch.sum(y_true * y_predict, dim=1)
        loss = torch.sum(torch.cos((math.pi / 2) * product_sum)) / N

        return loss


class FocalLogLoss(nn.Module):
    def __init__(self, gamma, eps=1e-8):
        super(FocalLogLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps

    def forward(self, y_predict, y_true):
        y_predict = torch.clamp(y_predict, self.eps, 1.0 - self.eps)

        loss = torch.ones_like(y_true)
        N = y_true.shape[0] * y_true.shape[1] * y_true.shape[2] * y_true.shape[3] + self.eps
        wrong_predictions = y_predict[y_true == 0]
        loss[y_true == 0] = -15 * torch.pow(wrong_predictions, 2)
        right_predictions = y_predict[y_true == 1]
        loss[y_true != 0] = 15 * torch.pow((right_predictions - 1), self.gamma) * torch.log(
            right_predictions + self.eps)

        loss = -torch.sum(loss) / N

        return loss


class LogMaxLoss(nn.Module):
    def __init__(self, gamma, eps=1e-8):
        super(LogMaxLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps

    def forward(self, y_predict, y_true):
        y_predict = torch.clamp(y_predict, self.eps, 1.0 - self.eps)

        y_false = 1.0 * torch.logical_not(y_true)
        loss1 = 5 * torch.pow(torch.sum(y_false * y_predict, dim=1), 2)
        loss2 = -5 * torch.sum(torch.pow((y_predict - 1), self.gamma) * y_true * torch.log(y_predict + self.eps), dim=1)

        log_max_loss = torch.mean(torch.maximum(loss1, loss2))

        return log_max_loss


class PolyLogLoss(nn.Module):
    def __init__(self, gamma, weight=1.0, eps=1e-8):
        super(PolyLogLoss, self).__init__()
        self.gamma = gamma
        self.weights = weight
        self.eps = eps

    def forward(self, y_predict, y_true):
        y_predict = torch.clamp(y_predict, self.eps, 1.0 - self.eps)

        right_predictions = torch.sum(y_true * y_predict, 1)
        loss1 = -torch.pow((right_predictions - 1), self.gamma) * self.weights * torch.log(right_predictions + self.eps)

        y_false = 1.0 * torch.logical_not(y_true)
        wrong_predictions = torch.sum(y_false * y_predict, 1)
        loss2 = -torch.pow(wrong_predictions, self.gamma) * torch.log(wrong_predictions + self.eps)

        poly_log_loss = torch.mean(torch.abs(loss1 - loss2))

        return poly_log_loss
