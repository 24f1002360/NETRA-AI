"""
Shared metric functions for eval/run_all.py.

Owner: Anshika (Clinical XAI & Validation).
These are used across grading, segmentation, and XAI evaluation --
kept in one place so BENCHMARKS.md numbers always come from the same
tested code (see GUIDE_4_Anshika.md: "You are the single source of truth").
"""
import numpy as np


def quadratic_weighted_kappa(y_true, y_pred, n_classes=5):
    """QWK for ICDR grading (0-4). Standard formula, no external deps
    so this doesn't force a new requirements.txt entry.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    O = np.zeros((n_classes, n_classes))
    for t, p in zip(y_true, y_pred):
        O[t, p] += 1

    w = np.zeros((n_classes, n_classes))
    for i in range(n_classes):
        for j in range(n_classes):
            w[i, j] = ((i - j) ** 2) / ((n_classes - 1) ** 2)

    hist_true = O.sum(axis=1)
    hist_pred = O.sum(axis=0)
    E = np.outer(hist_true, hist_pred) / O.sum()

    num = (w * O).sum()
    den = (w * E).sum()
    return 1.0 - num / den if den > 0 else 0.0


def sensitivity_specificity(y_true_binary, y_pred_binary):
    """For referable DR (grade >=2) screening. Inputs are 0/1 arrays."""
    y_true_binary = np.asarray(y_true_binary)
    y_pred_binary = np.asarray(y_pred_binary)

    tp = np.sum((y_true_binary == 1) & (y_pred_binary == 1))
    fn = np.sum((y_true_binary == 1) & (y_pred_binary == 0))
    tn = np.sum((y_true_binary == 0) & (y_pred_binary == 0))
    fp = np.sum((y_true_binary == 0) & (y_pred_binary == 1))

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else None
    specificity = tn / (tn + fp) if (tn + fp) > 0 else None
    return sensitivity, specificity


def dice_score(pred_mask, gt_mask):
    """Dice coefficient for a single lesion class, binary masks."""
    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)
    intersection = np.logical_and(pred, gt).sum()
    denom = pred.sum() + gt.sum()
    return (2.0 * intersection / denom) if denom > 0 else 1.0  # both empty = perfect agreement


def iou_score(pred_mask, gt_mask):
    """Intersection-over-union for a single lesion class, binary masks."""
    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)
    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    return (intersection / union) if union > 0 else 1.0