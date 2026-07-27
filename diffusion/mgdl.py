"""Mask-Grouped Diffusion Loss (MGDL), Section 3.2.2, Eq. (4)-(8).

The standard denoising objective (Eq. (3)) treats every pixel uniformly
and ignores the semantic heterogeneity across tissue types. MGDL
decouples the noise prediction error according to the pseudo-mask
categories and adds two regularisation terms:

    L_MGDL = L_cls + lambda_reg * (L_intra + L_inter)            (Eq. 7)

    L_cls   : class-decoupled noise prediction error             (Eq. 4)
    L_intra : within-class variance penalty                      (Eq. 5)
    L_inter : inter-class separation penalty with margin m       (Eq. 6)

The final MDP training objective combines the uniform reconstruction
term with MGDL:

    L = L_simple + L_MGDL                                        (Eq. 8)

In our experiments lambda_reg = 0.5 gives the best downstream
segmentation performance (Table 3 of the paper).

NOTE: the original experiment script computed a plain (optionally
SNR-weighted) noise MSE through a helper with a dormant class-grouped
branch. This module implements the MGDL exactly as described in the
paper and is what the training entry point ``train_mdp.py`` calls.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskGroupedDiffusionLoss(nn.Module):
    """Class-decoupled diffusion objective (Eq. (4)-(7)).

    Args:
        num_classes: number of pseudo-mask categories, including
            background (5 for LUAD-HistoSeg / BCSS-WSSS).
        lambda_reg: weight of the two regularisation penalties
            (lambda = 0.5 in the paper).
        margin: margin m of the inter-class separation penalty (Eq. (6)).
        min_pixels: classes with fewer pixels in a sample are skipped so
            that their statistics stay meaningful.
    """

    def __init__(self, num_classes=5, lambda_reg=0.5, margin=1.0, min_pixels=4):
        super().__init__()
        self.num_classes = num_classes
        self.lambda_reg = lambda_reg
        self.margin = margin
        self.min_pixels = min_pixels

    def forward(self, noise, noise_pred, mask):
        """Compute L_MGDL.

        Args:
            noise: ground-truth Gaussian noise, [B, C, H, W].
            noise_pred: predicted noise, [B, C, H, W].
            mask: pseudo-mask class indices, [B, H, W] (long).

        Returns:
            A dict with the total loss and the individual terms.
        """
        batch_size = noise.shape[0]
        cls_errors = []       # per-class noise prediction errors (Eq. 4)
        intra_terms = []      # within-class variance penalties (Eq. 5)
        class_means = []      # mean predicted-noise vector per class

        for b in range(batch_size):
            means_b = {}
            for c in range(self.num_classes):
                sel = mask[b] == c
                if sel.sum() < self.min_pixels:
                    continue
                # pixels of class c: [C, N_c]
                true_c = noise[b][:, sel]
                pred_c = noise_pred[b][:, sel]

                # Eq. (4): class-specific noise prediction error
                cls_errors.append(F.mse_loss(pred_c, true_c))

                # mean noise vector of class c
                mu_c = pred_c.mean(dim=1)
                means_b[c] = mu_c

                # Eq. (5): within-class variance penalty
                intra_terms.append(((pred_c - mu_c.unsqueeze(1)) ** 2).mean())

            if means_b:
                class_means.append(means_b)

        l_cls = torch.stack(cls_errors).mean() if cls_errors else \
            torch.tensor(0.0, device=noise.device)
        l_intra = torch.stack(intra_terms).mean() if intra_terms else \
            torch.tensor(0.0, device=noise.device)

        # Eq. (6): inter-class separation penalty (hinge on the margin)
        inter_terms = []
        for means_b in class_means:
            keys = sorted(means_b.keys())
            for i in range(len(keys)):
                for j in range(i + 1, len(keys)):
                    dist = torch.norm(means_b[keys[i]] - means_b[keys[j]], p=2)
                    inter_terms.append(F.relu(self.margin - dist))
        l_inter = torch.stack(inter_terms).mean() if inter_terms else \
            torch.tensor(0.0, device=noise.device)

        # Eq. (7)
        loss = l_cls + self.lambda_reg * (l_intra + l_inter)
        return loss, {'l_cls': l_cls.item(),
                      'l_intra': l_intra.item(),
                      'l_inter': l_inter.item()}


class MDPLoss(nn.Module):
    """Full MDP objective (Eq. (8)): L = L_simple + L_MGDL.

    L_simple is the uniform noise-prediction MSE of Eq. (3). An optional
    Min-SNR weighting of L_simple is provided for ablation purposes but
    is disabled by default to match the paper.
    """

    def __init__(self, num_classes=5, lambda_reg=0.5, margin=1.0,
                 use_snr_weighting=False):
        super().__init__()
        self.mgdl = MaskGroupedDiffusionLoss(
            num_classes=num_classes, lambda_reg=lambda_reg, margin=margin)
        self.use_snr_weighting = use_snr_weighting

    def forward(self, noise, noise_pred, mask, t=None, betas_schedule=None):
        if self.use_snr_weighting:
            if t is None or betas_schedule is None:
                raise ValueError(
                    "t and betas_schedule are required when SNR weighting is on")
            t_cpu = t.cpu()
            snr = 1.0 / (1 - betas_schedule['alphas_cumprod'][t_cpu]) - 1
            # Min-SNR weighting with k = 1, gamma = 1
            lambda_t = 1.0 / (1.0 + snr)
            lambda_t = lambda_t.unsqueeze(1).unsqueeze(2).unsqueeze(3).to(noise.device)
            n = noise.shape[1] * noise.shape[2] * noise.shape[3]
            l_simple = torch.sum(
                lambda_t * F.mse_loss(noise, noise_pred, reduction='none')) / n
        else:
            l_simple = F.mse_loss(noise, noise_pred)

        l_mgdl, parts = self.mgdl(noise, noise_pred, mask)
        parts['l_simple'] = l_simple.item()
        return l_simple + l_mgdl, parts
