"""
XAI evaluation harness (GUIDE_4_Anshika.md Part 4 - "XAI evaluation").

Runs against core.stubs.xai_stub for now (Day 1-6). Once Kanchan's
checkpoint lands, swap explain_fn to the real core.xai.explain.explain
and these functions don't need to change -- same interface.
"""
import numpy as np

from core.xai import guards


def guard_trigger_rates(cams, fov_masks, lesion_masks=None):
    """On a batch of (cam, fov_mask) pairs, what fraction trigger each
    guard status? GUIDE_4_Anshika.md: "a guard that fires on 40% of
    good images is broken; one that fires on 8%... is working."

    Returns dict: {"OK": frac, "CAM_OFF_RETINA": frac, "LOW_AGREEMENT": frac}
    """
    counts = {"OK": 0, "CAM_OFF_RETINA": 0, "LOW_AGREEMENT": 0}
    n = len(cams)
    if n == 0:
        return {k: 0.0 for k in counts}

    lesion_masks = lesion_masks or [None] * n

    for cam, fov_mask, lesion_mask in zip(cams, fov_masks, lesion_masks):
        outside_frac = guards.cam_outside_fov_fraction(cam, fov_mask)
        agreement = (guards.cam_lesion_agreement(cam, lesion_mask)
                     if lesion_mask is not None else None)
        anatomical = guards.anatomical_plausibility(cam, fov_mask)
        status = guards.guard_status(outside_frac, agreement, anatomical)
        counts[status] += 1

    return {k: v / n for k, v in counts.items()}


def lesion_localisation_hit_rate(cams, lesion_masks):
    """Fraction of images where the peak CAM pixel lands inside the
    annotated lesion mask. GUIDE_4_Anshika.md Part 4.
    """
    hits = 0
    n = 0
    for cam, lesion_mask in zip(cams, lesion_masks):
        if lesion_mask is None:
            continue
        n += 1
        peak_idx = np.unravel_index(np.argmax(cam), cam.shape)
        lesion_bin = lesion_mask.astype(bool)
        if lesion_bin.ndim == 3:
            lesion_bin = lesion_bin.any(axis=2)
        if lesion_bin[peak_idx]:
            hits += 1
    return (hits / n) if n > 0 else None


def deletion_insertion_auc(bgr, cam, predict_fn, class_idx, n_steps=20):
    """Deletion/insertion test (Petsiuk et al. 2018 RISE-style metric).

    predict_fn(bgr_masked) -> probability for class_idx. This works with
    ANY classifier callable, including a stub during development -- swap
    in Kanchan's real grading model later without changing this function.

    Deletion: progressively zero out the highest-CAM pixels, track how
    fast p(class) drops. Sharp drop = CAM points at pixels the model
    actually uses. Returns (deletion_auc, insertion_auc); lower deletion
    AUC and higher insertion AUC are both "good" (sharper CAM).
    """
    h, w = cam.shape[:2]
    order = np.argsort(cam.ravel())[::-1]  # highest CAM first
    n_pixels = h * w
    step = max(n_pixels // n_steps, 1)

    baseline_value = bgr.mean(axis=(0, 1)) if bgr.ndim == 3 else bgr.mean()

    deletion_probs = []
    insertion_probs = []

    img_del = bgr.copy()
    img_ins = np.full_like(bgr, baseline_value)

    for i in range(0, n_pixels, step):
        idx = order[i:i + step]
        ys, xs = np.unravel_index(idx, (h, w))

        img_del[ys, xs] = baseline_value
        deletion_probs.append(predict_fn(img_del)[class_idx])

        img_ins[ys, xs] = bgr[ys, xs]
        insertion_probs.append(predict_fn(img_ins)[class_idx])

    deletion_auc = float(np.trapz(deletion_probs) / len(deletion_probs))
    insertion_auc = float(np.trapz(insertion_probs) / len(insertion_probs))
    return deletion_auc, insertion_auc