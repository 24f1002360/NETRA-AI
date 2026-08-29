"""
The Quality Gate.

Scores a fundus image on four axes, all computed INSIDE the FOV mask, and
returns the `quality` block of the ScreeningResult contract
(see docs/01_INTERFACE_CONTRACTS.md).

Verdicts:
    PASS             -- good enough to grade
    AUTO_CORRECTED   -- borderline, enhancement applied, re-scored once
    RETAKE           -- health worker must take a new photo, with a reason

Reason codes are emitted, never English. Abhishek maps them to Hindi / Tamil /
Telugu text and audio in configs/i18n/.
"""

import time

import cv2
import numpy as np
import yaml

from .fov import extract_fov
from .enhance import enhance, enhancement_names

DEFAULT_THRESHOLDS = {
    "blur":         {"hard": 0.25, "soft": 0.45},
    "illumination": {"hard": 0.30, "soft": 0.50},
    "fov_coverage": {"hard": 0.45, "soft": 0.65},
    "contrast":     {"hard": 0.25, "soft": 0.45},
    "centre_offset_max": 0.35,
    "blur_ref_variance": 900.0,   # Laplacian variance considered "sharp"
    "clip_frac_max": 0.10,
}


def load_thresholds(path="configs/thresholds.yaml"):
    try:
        with open(path) as f:
            cfg = yaml.safe_load(f) or {}
        return {**DEFAULT_THRESHOLDS, **cfg.get("iqa", {})}
    except FileNotFoundError:
        return DEFAULT_THRESHOLDS


# --------------------------------------------------------------------------
# Individual scores. Each returns 0.0 (terrible) .. 1.0 (excellent).
# --------------------------------------------------------------------------

def blur_score(bgr, mask, ref_variance=900.0):
    """
    Variance of the Laplacian, computed INSIDE the FOV only.

    Computing this over the whole frame is the classic fundus IQA bug: the
    black surround has near-zero variance and swamps the real signal.
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)
    inside = lap[mask > 0]
    if inside.size == 0:
        return 0.0
    var = float(inside.var())
    return float(np.clip(var / ref_variance, 0.0, 1.0))


def illumination_score(bgr, mask, clip_frac_max=0.10):
    """
    Penalises both under- and over-exposure.
    Combines mean green intensity with the fraction of clipped pixels.
    """
    g = bgr[:, :, 1]
    inside = g[mask > 0]
    if inside.size == 0:
        return 0.0

    mean = float(inside.mean())
    # Ideal mean sits around 110-140 for a well-exposed fundus image.
    mean_term = 1.0 - min(abs(mean - 125.0) / 125.0, 1.0)

    clipped = float(((inside > 250) | (inside < 5)).mean())
    clip_term = 1.0 - min(clipped / clip_frac_max, 1.0)

    return float(np.clip(0.65 * mean_term + 0.35 * clip_term, 0.0, 1.0))


def contrast_score(bgr, mask):
    """
    RMS contrast of the green channel inside the FOV.

    (Muskan's original guide suggested a Frangi vessel filter here. That is a
    better signal but costs days of tuning -- RMS is the pragmatic choice.
    Swap it in later if you have spare time.)
    """
    g = bgr[:, :, 1].astype(np.float32)
    inside = g[mask > 0]
    if inside.size == 0:
        return 0.0
    rms = float(inside.std())
    return float(np.clip(rms / 55.0, 0.0, 1.0))


# --------------------------------------------------------------------------
# The gate
# --------------------------------------------------------------------------

def _score_all(bgr, thr):
    mask, info = extract_fov(bgr)
    return {
        "blur": blur_score(bgr, mask, thr["blur_ref_variance"]),
        "illumination": illumination_score(bgr, mask, thr["clip_frac_max"]),
        "fov_coverage": info["coverage"],
        "contrast": contrast_score(bgr, mask),
    }, mask, info


def _reasons(scores, info, thr):
    """Map failing scores to reason codes."""
    out = []
    if scores["blur"] < thr["blur"]["hard"]:
        out.append("BLUR_HIGH")
    if scores["illumination"] < thr["illumination"]["hard"]:
        # Distinguish the two directions so the voice prompt is useful.
        out.append("TOO_DARK" if info.get("mean_green", 128) < 125 else "TOO_BRIGHT")
    if scores["fov_coverage"] < thr["fov_coverage"]["hard"]:
        out.append("FOV_PARTIAL")
    if scores["contrast"] < thr["contrast"]["hard"]:
        out.append("LOW_CONTRAST")
    if info["centre_offset"] > thr["centre_offset_max"]:
        out.append("OFF_CENTRE")
    return out


def assess_quality(bgr, thresholds=None, fov_mask_path=None):
    """
    Contract function. Returns the `quality` block.

    Args:
        bgr: HxWx3 uint8 BGR image
        thresholds: dict, defaults to configs/thresholds.yaml
        fov_mask_path: if given, the FOV mask is written here (Anshika needs it)

    Returns:
        dict matching the `quality` block in 01_INTERFACE_CONTRACTS.md,
        plus a private "_mask" key holding the FOV mask array.
    """
    t0 = time.time()
    thr = thresholds or load_thresholds()

    scores, mask, info = _score_all(bgr, thr)
    g_inside = bgr[:, :, 1][mask > 0]
    info["mean_green"] = float(g_inside.mean()) if g_inside.size else 0.0

    applied = []
    hard_fail = _reasons(scores, info, thr)

    if hard_fail:
        verdict = "RETAKE"
        reasons = hard_fail
        msg_key = f"iqa.retake.{hard_fail[0].lower()}"
    else:
        soft_fail = [k for k in ("blur", "illumination", "contrast")
                     if scores[k] < thr[k]["soft"]]
        if soft_fail:
            # Enhance once, then re-score. Never loop.
            corrected = enhance(bgr)
            scores, mask, info = _score_all(corrected, thr)
            info["mean_green"] = float(corrected[:, :, 1][mask > 0].mean()) \
                if (mask > 0).any() else 0.0
            applied = enhancement_names()

            still_bad = _reasons(scores, info, thr)
            if still_bad:
                verdict, reasons = "RETAKE", still_bad
                msg_key = f"iqa.retake.{still_bad[0].lower()}"
            else:
                verdict, reasons, msg_key = "AUTO_CORRECTED", [], "iqa.corrected"
        else:
            verdict, reasons, msg_key = "PASS", [], "iqa.pass"

    if fov_mask_path:
        cv2.imwrite(fov_mask_path, mask)

    return {
        "verdict": verdict,
        "scores": {
            "blur": round(scores["blur"], 3),
            "illumination": round(scores["illumination"], 3),
            "fov_coverage": round(scores["fov_coverage"], 3),
            "contrast": round(scores["contrast"], 3),
            "centre_offset": round(info["centre_offset"], 3),
        },
        "reasons": reasons,
        "operator_message_key": msg_key,
        "enhancement_applied": applied,
        "processing_ms": int((time.time() - t0) * 1000),
        "_mask": mask,          # private, not serialised to JSON
    }
