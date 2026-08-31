"""
python eval/run_all.py

Single command to regenerate the numbers in docs/BENCHMARKS.md, per
GUIDE_4_Anshika.md: "Build eval/ early so results are reproducible
with one command."

CURRENT STATE (Day 1-6): runs the XAI-side metrics (guard trigger rate,
deletion/insertion scaffolding) against synthetic/stub data so the
harness is proven correct before Kanchan's checkpoint lands. Grading
and segmentation metrics (metrics.py) are ready but need Kanchan's
real predictions to run against -- see the TODOs below.

Run from the repo root: python eval/run_all.py
"""
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.iqa.quality import assess_quality
from eval.xai_eval import guard_trigger_rates, lesion_localisation_hit_rate

FIXTURES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "tests", "fixtures")


def _synthetic_cam(h, w):
    """Stand-in for a real CAM until Kanchan's checkpoint lands --
    same centred-blob shape as core/stubs/xai_stub.py so this harness
    exercises the exact code path the rest of the team builds against.
    """
    yy, xx = np.mgrid[0:h, 0:w]
    cy, cx = h / 2, w / 2
    cam = np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * (min(h, w) / 4) ** 2)))
    return (cam / cam.max()).astype(np.float32)


def run_xai_eval():
    print("=" * 60)
    print("XAI eval (guard trigger rates) -- stub CAMs, real FOV masks")
    print("=" * 60)

    cams, fov_masks = [], []
    for name in ("good.png", "blurry.png", "severe.png"):
        path = os.path.join(FIXTURES, name)
        img = cv2.imread(path)
        if img is None:
            print(f"  skip {name}: fixture not found")
            continue
        q = assess_quality(img)
        fov_mask = q["_mask"]
        h, w = fov_mask.shape[:2]
        cams.append(_synthetic_cam(h, w))
        fov_masks.append(fov_mask)

    rates = guard_trigger_rates(cams, fov_masks)
    for status, frac in rates.items():
        print(f"  {status:20s} {frac*100:5.1f}%")

    print()
    print("NOTE: these numbers use a synthetic stub CAM, not Kanchan's")
    print("real model output -- do NOT put these in BENCHMARKS.md yet.")
    print("Once explain.py is wired to the real checkpoint, swap")
    print("_synthetic_cam() for real explain() output and rerun.")
    return rates


def run_grading_eval():
    # TODO once Kanchan's checkpoint + APTOS held-out predictions exist:
    #   from eval.metrics import quadratic_weighted_kappa, sensitivity_specificity
    #   compute QWK, referable-DR sensitivity/specificity, write to BENCHMARKS.md
    print("Grading eval: waiting on Kanchan's held-out predictions (metrics.py ready).")


def run_segmentation_eval():
    # TODO once Kanchan's U-Net predictions on IDRiD test split exist:
    #   from eval.metrics import dice_score, iou_score
    print("Segmentation eval: waiting on Kanchan's U-Net predictions (metrics.py ready).")


if __name__ == "__main__":
    run_xai_eval()
    print()
    run_grading_eval()
    run_segmentation_eval()
