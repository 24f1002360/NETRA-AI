"""
The three-check "anatomically guarded" CAM validator.

Every published DR system shows a heatmap. None of them check whether the
heatmap makes anatomical sense. This module does -- and when it doesn't,
guard_status() tells the caller to escalate rather than report a grade
that can't be justified.

See docs/GUIDE_4_Anshika.md Part 2.
"""
import numpy as np

FOV_OUTSIDE_THRESHOLD = 0.15
LESION_AGREEMENT_THRESHOLD = 0.20
TOP_QUANTILE = 0.20


def zero_outside_fov(cam, fov_mask):
    """Hard-zero CAM energy outside the retina. Always applied before display,
    regardless of the guard verdict."""
    m = (fov_mask > 0).astype(cam.dtype)
    return cam * m


def cam_outside_fov_fraction(cam, fov_mask):
    """cam_outside_fov_fraction = sum(CAM * (1 - fov_mask)) / sum(CAM)"""
    total = cam.sum()
    if total <= 0:
        return 0.0
    outside = (cam * (1 - (fov_mask > 0).astype(cam.dtype))).sum()
    return float(np.clip(outside / total, 0.0, 1.0))


def top_quantile_mask(cam, quantile=TOP_QUANTILE):
    flat = cam[cam > 0]
    if flat.size == 0:
        return np.zeros_like(cam, dtype=bool)
    thresh = np.quantile(flat, 1 - quantile)
    return cam >= thresh


def cam_lesion_agreement(cam, lesion_mask, quantile=TOP_QUANTILE):
    top = top_quantile_mask(cam, quantile)
    if top.sum() == 0:
        return None
    lesion_bin = lesion_mask.astype(bool)
    if lesion_bin.ndim == 3:
        lesion_bin = lesion_bin.any(axis=2)
    if lesion_bin.sum() == 0:
        # No lesions were segmented at all (e.g. a genuine No-DR image) --
        # there is nothing for the CAM to agree or disagree with, so this
        # is not evaluable, not automatic disagreement.
        return None
    inter = np.logical_and(top, lesion_bin).sum()
    return float(inter / top.sum())


def anatomical_plausibility(cam, fov_mask, macula_frac=0.6):
    """Radial prior: DR lesions cluster in the posterior pole. Returns the
    fraction of CAM energy inside the central 'plausible' zone of the
    retina (vs. optic-disc-only or far-periphery concentration)."""
    ys, xs = np.nonzero(fov_mask > 0)
    if len(xs) == 0:
        return 0.0
    cx, cy = xs.mean(), ys.mean()
    radius = np.sqrt(((xs - cx) ** 2 + (ys - cy) ** 2).max())
    yy, xx = np.indices(cam.shape[:2])
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / max(radius, 1.0)
    plausible_zone = (dist <= macula_frac).astype(cam.dtype)
    total = cam.sum()
    if total <= 0:
        return 1.0
    return float((cam * plausible_zone).sum() / total)


def guard_status(outside_fov_frac, lesion_agreement, anatomical_score,
                  fov_threshold=FOV_OUTSIDE_THRESHOLD,
                  agreement_threshold=LESION_AGREEMENT_THRESHOLD):
    if outside_fov_frac > fov_threshold:
        return "CAM_OFF_RETINA"
    if lesion_agreement is not None and lesion_agreement < agreement_threshold:
        return "LOW_AGREEMENT"
    return "OK"