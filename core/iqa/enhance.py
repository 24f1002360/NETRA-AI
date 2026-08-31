"""
Image enhancement for fundus images.

CRITICAL: this exact function is used in BOTH
  (a) Kanchan's model training preprocessing, and
  (b) live inference in the app.

If the two ever diverge, model accuracy silently collapses at inference time
and it takes days to diagnose. Freeze the behaviour of enhance() once Kanchan
starts training. If you must change it, tell him so he can retrain.
"""

import cv2
import numpy as np

from .fov import extract_fov

WORK_MAX_DIM = 1024   # everything downstream runs at this resolution or less


def resize_for_processing(bgr, max_dim=WORK_MAX_DIM):
    """Cap the long edge. Raw fundus images are ~4288x2848; scoring and
    enhancing at that size takes ~28 s, which blows the 1.5 s budget.
    At 1024 it is ~0.5 s and no quality signal is lost."""
    h, w = bgr.shape[:2]
    if max(h, w) <= max_dim:
        return bgr
    scale = max_dim / max(h, w)
    return cv2.resize(bgr, (int(w * scale), int(h * scale)),
                      interpolation=cv2.INTER_AREA)


def enhance(bgr, quality=None, clip_limit=2.5, tile_grid=(8, 8),
            correct_illumination=True, max_dim=WORK_MAX_DIM):
    """
    Enhance a fundus image for grading and lesion visibility.

    Args:
        bgr:     HxWx3 uint8 BGR image
        quality: optional quality block; if verdict is RETAKE this is a no-op
        clip_limit / tile_grid: CLAHE parameters
        correct_illumination: remove large-scale vignetting before CLAHE
        max_dim: working resolution cap (see resize_for_processing)

    Returns:
        HxWx3 uint8 BGR image, resized to at most max_dim on the long edge.

    Idempotent in spirit: running it twice degrades the image, so run it once.
    """
    if quality is not None and quality.get("verdict") == "RETAKE":
        return bgr

    out = resize_for_processing(bgr, max_dim)
    mask, _ = extract_fov(out)

    if correct_illumination:
        out = _remove_vignette(out, mask)

    # CLAHE on the green channel: retinal vessels, haemorrhages and
    # microaneurysms have the strongest contrast there.
    b, g, r = cv2.split(out)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    g = clahe.apply(g)
    out = cv2.merge([b, g, r])

    # Never let enhancement invent structure outside the retina.
    out = cv2.bitwise_and(out, out, mask=mask)
    return out


def _remove_vignette(bgr, mask, ksize_frac=0.1):
    """
    Estimate the slow illumination gradient with a large Gaussian and divide
    it out. Fixes the dark-corner effect typical of handheld fundus cameras.
    """
    h, w = bgr.shape[:2]
    k = int(max(h, w) * ksize_frac) | 1  # force odd
    background = cv2.GaussianBlur(bgr, (k, k), 0).astype(np.float32) + 1.0
    mean_bg = float(background[mask > 0].mean()) if (mask > 0).any() else 128.0
    corrected = bgr.astype(np.float32) / background * mean_bg
    return np.clip(corrected, 0, 255).astype(np.uint8)


def enhancement_names(clip_limit=2.5, correct_illumination=True):
    """Names of the operations applied, for the `enhancement_applied` field."""
    names = []
    if correct_illumination:
        names.append("VIGNETTE_CORRECTION")
    names.append(f"CLAHE_GREEN_{clip_limit}")
    return names