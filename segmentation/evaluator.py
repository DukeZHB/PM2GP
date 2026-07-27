"""Evaluation utilities for multi-tissue segmentation.

The paper reports mean Intersection-over-Union (mIoU) and pixel
accuracy (PA) computed over the foreground tissue classes only
(Section 4.2). The Evaluator below accumulates a confusion matrix over
the test set and derives per-class IoU / Dice / PA; AJI and HD95 are
provided as supplementary region-level measures.
"""

import numpy as np
import torch
from scipy.spatial.distance import cdist


class Evaluator(object):
    """Confusion-matrix based segmentation evaluator."""

    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((num_class, num_class), dtype=np.float64)

    def Pixel_Accuracy(self):
        acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return acc if not np.isnan(acc) else 0.0

    def Pixel_Accuracy_Class(self):
        acc = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        acc = np.nan_to_num(acc, nan=0.0)
        return np.mean(acc)

    def Intersection_over_Union(self):
        diag = np.diag(self.confusion_matrix)
        denominator = (self.confusion_matrix.sum(axis=1)
                       + self.confusion_matrix.sum(axis=0) - diag)
        return np.where(denominator != 0, diag / denominator, 0.0)

    def Mean_Intersection_over_Union(self):
        return np.mean(self.Intersection_over_Union())

    def Frequency_Weighted_Intersection_over_Union(self):
        freq = self.confusion_matrix.sum(axis=1) / self.confusion_matrix.sum()
        iu = self.Intersection_over_Union()
        freq = np.nan_to_num(freq, nan=0.0)
        return (freq[freq > 0] * iu[freq > 0]).sum()

    def Dice_Score(self):
        dice_scores = {}
        for i in range(self.num_class):
            tp = np.diag(self.confusion_matrix)[i]
            fp = self.confusion_matrix[:, i].sum() - tp
            fn = self.confusion_matrix[i, :].sum() - tp
            denominator = 2 * tp + fp + fn
            dice_scores[i] = 2 * tp / denominator if denominator != 0 else 0.0
        return np.mean(list(dice_scores.values())), dice_scores

    def _generate_matrix(self, gt_image, pre_image):
        gt_valid = gt_image.reshape(-1)
        pre_valid = pre_image.reshape(-1)
        mask = (gt_valid >= 0) & (gt_valid < self.num_class)
        label = self.num_class * gt_valid[mask].astype(int) + pre_valid[mask]
        count = np.bincount(label, minlength=self.num_class ** 2)
        return count.reshape(self.num_class, self.num_class)

    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class, self.num_class),
                                         dtype=np.float64)


def Aggregated_jaccard_index(gt_map, predicted_map, gpu, include_background=True):
    """Aggregated Jaccard Index (AJI) between one-hot gt and prediction."""
    _, gt_map = torch.max(gt_map, 1)
    _, predicted_map = torch.max(predicted_map, 1)

    gt_list = torch.unique(gt_map)
    pr_list = torch.unique(predicted_map)

    if not include_background and 0 in gt_list:
        gt_list = gt_list[gt_list != 0]
    if not include_background and 0 in pr_list:
        pr_list = pr_list[pr_list != 0]

    if len(gt_list) == 0 and len(pr_list) == 0:
        return 1.0

    pr_list = torch.cat((pr_list.view(-1, 1),
                         torch.zeros(pr_list.size(0), 1).to(gpu)), dim=1)

    overall_correct_count = 0.0
    union_pixel_count = 0.0

    while len(gt_list) > 0:
        gt = (gt_map == gt_list[-1]).float()
        predicted_match = gt * predicted_map.float()

        if predicted_match.sum() == 0:
            union_pixel_count += gt.sum()
            gt_list = gt_list[:-1]
        else:
            predicted_idx = torch.unique(predicted_match)
            if not include_background and 0 in predicted_idx:
                predicted_idx = predicted_idx[predicted_idx != 0]

            best_ji, best_match = 0.0, None
            for j in range(len(predicted_idx)):
                matched = (predicted_map == predicted_idx[j]).float()
                intersection = matched.logical_and(gt).sum()
                union = matched.logical_or(gt).sum()
                ji = intersection / union if union != 0 else 0.0
                if ji > best_ji:
                    best_ji, best_match = ji, predicted_idx[j]

            if best_match is not None:
                predicted_region = (predicted_map == best_match).float()
                overall_correct_count += (gt.logical_and(predicted_region)).sum()
                union_pixel_count += (gt.logical_or(predicted_region)).sum()
                best_idx = (pr_list[:, 0] == best_match).nonzero().item()
                pr_list[best_idx, 1] += 1

            gt_list = gt_list[:-1]

    unused = (pr_list[:, 1] == 0).nonzero().view(-1)
    for k in range(len(unused)):
        unused_region = (predicted_map == pr_list[unused[k], 0]).float()
        union_pixel_count += unused_region.sum()

    if union_pixel_count == 0:
        return 1.0
    return (overall_correct_count / union_pixel_count).cpu().numpy()


def Hausdorff_distance_95(y_true, y_pred, num_classes,
                          include_background=True, sample_ratio=1.0):
    """95th-percentile Hausdorff distance (HD95) per class.

    Args:
        y_true, y_pred: one-hot tensors of shape (1, C, H, W).
        sample_ratio: optional pixel subsampling ratio for speed.
    """
    class_hd95 = []
    start_idx = 0 if include_background else 1
    _, y_true_idx = torch.max(y_true, 1)
    _, y_pred_idx = torch.max(y_pred, 1)

    for i in range(start_idx, num_classes):
        true = (y_true_idx == i).squeeze(0).cpu().numpy()
        pred = (y_pred_idx == i).squeeze(0).cpu().numpy()

        if np.sum(true) == 0 or np.sum(pred) == 0:
            class_hd95.append(0.0)
            continue

        true_points = np.argwhere(true)
        pred_points = np.argwhere(pred)

        if sample_ratio < 1.0:
            n_true = max(1, int(len(true_points) * sample_ratio))
            n_pred = max(1, int(len(pred_points) * sample_ratio))
            true_points = true_points[
                np.random.choice(len(true_points), n_true, replace=False)]
            pred_points = pred_points[
                np.random.choice(len(pred_points), n_pred, replace=False)]

        dists = np.concatenate([
            cdist(true_points, pred_points).min(axis=1),
            cdist(pred_points, true_points).min(axis=1),
        ])
        class_hd95.append(np.percentile(dists, 95))
    return class_hd95
