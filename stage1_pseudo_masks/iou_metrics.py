"""IoU / Dice metrics for evaluating CAM-derived pseudo masks.

Note: the original experiment code forced every ground-truth background
pixel to be predicted as background before building the confusion
matrix (``lp[lt == n_class] = n_class``), which leaks ground-truth
information into the evaluation and inflates the foreground scores.
That line is removed here; the metrics below are computed from the raw
predictions, consistent with ``segmentation/evaluate.py``.
"""

import numpy as np


def _fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(
        n_class * label_true[mask].astype(int) + label_pred[mask],
        minlength=n_class ** 2,
    ).reshape(n_class, n_class)
    return hist


def scores(label_trues, label_preds, n_class):
    """Foreground-class metrics; the background class (index ``n_class``)
    is excluded from the mean IoU / accuracy / Dice averages."""
    n_classori = int(n_class)      # number of foreground classes
    n_class = n_class + 1          # + background

    hist = np.zeros((n_class, n_class))

    for lt, lp in zip(label_trues, label_preds):
        tmp = _fast_hist(lt.flatten(), lp.flatten(), n_class)
        hist += tmp
    hist[n_classori, n_classori] = 0
    acc = np.diag(hist).sum() / hist.sum()
    acc_cls = np.diag(hist)[0:n_classori] / hist.sum(axis=1)[0:n_classori]
    acc_cls = np.nanmean(acc_cls)
    iu = np.diag(hist)[0:n_classori] / ((hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))[0:n_classori])
    mean_iu = np.nanmean(iu)
    freq = hist.sum(axis=1)[0:n_classori] / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    cls_iu = dict(zip(range(n_class), iu))
    dice_scores = {}
    for i in range(n_classori):  # foreground classes only
        tp = np.diag(hist)[i]
        fp = hist[:, i].sum() - tp
        fn = hist[i, :].sum() - tp
        dice = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
        dice_scores[i] = dice

    mean_dice = np.mean(list(dice_scores.values()))

    return {
        "Pixel Accuracy": acc,
        "Mean Accuracy": acc_cls,
        "Frequency Weighted IoU": fwavacc,
        "Mean IoU": mean_iu,
        "Class IoU": cls_iu,
        "Dice Coefficients": dice_scores,
        "Mean Dice": mean_dice
    }
