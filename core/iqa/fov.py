"""
FOV (field of view) mask extraction for fundus images.

The retinal disc sits inside a black surround. Almost every quality metric is
meaningless unless it is computed inside that disc only -- the black border
drags means and variances toward zero and makes every image look the same.

This module is used by:
  - core/iqa/quality.py   (all scores are masked)
  - core/xai/  (Anshika)  -- CAM off-retina guard
"""

import cv2
import numpy as np


def extract_fov(bgr, erode_frac=0.05):
    """
    Extract the circular retinal field of view.

    Args:
        bgr: HxWx3 uint8 BGR image
        erode_frac: shrink the mask by this fraction of its radius, so that
                    boundary pixels (which are dim and noisy) are excluded

    Returns:
        mask:   HxW uint8, 255 inside the retina, 0 outside
        info:   dict with coverage, centre offset and radius
    """
    h, w = bgr.shape[:2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # Otsu on a blurred copy is robust to noise and to varying exposure.
    blur = cv2.GaussianBlur(gray, (0, 0), 5)
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Otsu can pick a very low threshold on dark images and grab the surround.
    # Guard with a floor: retina pixels are brighter than near-black.
    binary[gray < 10] = 0

    # Close small holes (bright lesions / dark vessels near the edge).
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # Largest connected component = the retinal disc.
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, 8)
    if n_labels <= 1:
        # Nothing found -- fall back to the whole frame so the pipeline
        # degrades instead of crashing.
        mask = np.full((h, w), 255, np.uint8)
        return mask, _fov_info(mask, (w / 2, h / 2), min(h, w) / 2, w, h)

    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    mask = np.where(labels == largest, 255, 0).astype(np.uint8)

    # Fill interior holes.
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros((h, w), np.uint8)
    cv2.drawContours(mask, contours, -1, 255, thickness=cv2.FILLED)

    (cx, cy), radius = cv2.minEnclosingCircle(contours[0])

    # Erode so we never include the dim boundary ring.
    if erode_frac > 0:
        r_erode = max(1, int(radius * erode_frac))
        ek = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r_erode + 1,) * 2)
        mask = cv2.erode(mask, ek)

    return mask, _fov_info(mask, (cx, cy), radius, w, h)


def _fov_info(mask, centre, radius, w, h):
    area = float((mask > 0).sum())
    cx, cy = centre
    # Expected area if the disc were fully inside the frame.
    expected = np.pi * (min(w, h) / 2.0) ** 2
    offset = np.hypot(cx - w / 2.0, cy - h / 2.0) / (min(w, h) / 2.0)
    return {
        "area_px": area,
        "coverage": float(np.clip(area / max(expected, 1.0), 0.0, 1.0)),
        "centre": (float(cx), float(cy)),
        "radius": float(radius),
        "centre_offset": float(np.clip(offset, 0.0, 1.0)),
    }


def apply_fov(img, mask):
    """Zero everything outside the FOV. Works for 2D and 3D arrays."""
    if img.ndim == 2:
        return cv2.bitwise_and(img, img, mask=mask)
    return cv2.bitwise_and(img, img, mask=mask)
