"""
python eval/run_all.py

Single command to regenerate the numbers in docs/BENCHMARKS.md.

Runs the real pipeline (DRGrader + DRSegmenter + core.xai.explain.explain)
against tests/fixtures/*.png when torch and both checkpoints are
available, and falls back to the old synthetic-CAM path with a loud
warning otherwise -- so the harness always runs, but never silently
mislabels stub numbers as real ones.

Run from the repo root: python eval/run_all.py
"""
import datetime
import os
import subprocess
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.iqa.quality import assess_quality
from eval.xai_eval import deletion_insertion_auc, lesion_localisation_hit_rate

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIXTURES = os.path.join(REPO_ROOT, "tests", "fixtures")
BENCHMARKS_PATH = os.path.join(REPO_ROOT, "docs", "BENCHMARKS.md")
FIXTURE_NAMES = ("good.png", "blurry.png", "severe.png")


def _load_fixtures():
    loaded = []
    for name in FIXTURE_NAMES:
        img = cv2.imread(os.path.join(FIXTURES, name))
        if img is None:
            print(f"  skip {name}: fixture not found")
            continue
        loaded.append((name, img))
    return loaded


def _load_real_pipeline():
    try:
        from core.models.grading import DRGrader
        from core.models.segmentation import DRSegmenter
        from core.xai.explain import explain
    except ImportError as e:
        print(f"  real pipeline unavailable ({e}) -- is torch installed?")
        return None

    try:
        grader = DRGrader()
        segmenter = DRSegmenter()
    except FileNotFoundError as e:
        print(f"  real pipeline unavailable -- checkpoint missing: {e}")
        return None

    model_handle = {
        "model": grader.model,
        "layer_name": grader.gradcam_layer,
        "device": str(grader.device),
        "preprocess": grader.preprocess,
    }
    return grader, segmenter, explain, model_handle


def _combined_lesion_mask(seg_result):
    channel_masks = list(seg_result["masks"].values())
    combined = np.zeros_like(channel_masks[0], dtype=np.uint8)
    for m in channel_masks:
        combined = np.logical_or(combined, m.astype(bool)).astype(np.uint8)
    return combined


def run_real_xai_eval(fixtures, pipeline):
    grader, segmenter, explain, model_handle = pipeline
    print(f"  REAL pipeline, n={len(fixtures)}")

    per_image = []
    for name, bgr in fixtures:
        fov_mask = assess_quality(bgr)["_mask"]
        grading = grader.grade(bgr)
        seg = segmenter.segment(bgr)
        lesion_mask = _combined_lesion_mask(seg)

        result = explain(bgr, model_handle, grading, lesion_mask, fov_mask,
                          screening_id=name.replace(".png", ""))

        cam = cv2.imread(result["gradcam_path"], cv2.IMREAD_GRAYSCALE)
        cam = cam.astype(np.float32) / 255.0
        lesion_resized = cv2.resize(lesion_mask, (cam.shape[1], cam.shape[0]),
                                     interpolation=cv2.INTER_NEAREST)
        hit = lesion_localisation_hit_rate([cam], [lesion_resized])

        bgr_at_cam_res = cv2.resize(bgr, (cam.shape[1], cam.shape[0]))

        def predict_fn(img, _grader=grader):
            return _grader.predict(img)["probabilities"]

        del_auc, ins_auc = deletion_insertion_auc(
            bgr_at_cam_res, cam, predict_fn, grading["icdr_grade"], n_steps=10)

        per_image.append({
            "name": name,
            "grade": grading["icdr_grade"],
            "confidence": grading["confidence"],
            "cam_outside_fov_fraction": result["cam_outside_fov_fraction"],
            "cam_lesion_agreement": result["cam_lesion_agreement"],
            "guard_status": result["guard_status"],
            "lesion_hit": hit,
            "deletion_auc": del_auc,
            "insertion_auc": ins_auc,
        })

        agree_str = ("n/a" if result["cam_lesion_agreement"] is None
                     else f"{result['cam_lesion_agreement']:.3f}")
        print(f"  {name:12s} grade={grading['icdr_grade']} "
              f"conf={grading['confidence']:.3f} "
              f"guard={result['guard_status']:16s} "
              f"agreement={agree_str} "
              f"del={del_auc:.4f} ins={ins_auc:.4f}")

    n = len(per_image)
    rates = {s: sum(1 for r in per_image if r["guard_status"] == s) / n
             for s in ("OK", "CAM_OFF_RETINA", "LOW_AGREEMENT")}

    print(f"\n  guard trigger rates (n={n}):")
    for status, frac in rates.items():
        print(f"    {status:20s} {frac * 100:5.1f}%")

    return {"real": True, "n": n, "per_image": per_image, "guard_trigger_rates": rates}


def run_stub_xai_eval(fixtures):
    from eval.xai_eval import guard_trigger_rates

    print("  torch/checkpoints unavailable -- using SYNTHETIC stub CAMs.")
    print("  These numbers must NOT go in BENCHMARKS.md as real results.\n")

    cams, fov_masks = [], []
    for name, bgr in fixtures:
        fov_mask = assess_quality(bgr)["_mask"]
        h, w = fov_mask.shape[:2]
        yy, xx = np.mgrid[0:h, 0:w]
        cy, cx = h / 2, w / 2
        cam = np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * (min(h, w) / 4) ** 2)))
        cams.append((cam / cam.max()).astype(np.float32))
        fov_masks.append(fov_mask)

    rates = guard_trigger_rates(cams, fov_masks)
    for status, frac in rates.items():
        print(f"    {status:20s} {frac * 100:5.1f}%")

    return {"real": False, "n": len(cams), "per_image": [], "guard_trigger_rates": rates}


def run_xai_eval():
    print("=" * 60)
    print("XAI eval")
    print("=" * 60)
    fixtures = _load_fixtures()
    if not fixtures:
        print("  no fixtures found -- nothing to run.")
        return None
    pipeline = _load_real_pipeline()
    if pipeline is None:
        return run_stub_xai_eval(fixtures)
    return run_real_xai_eval(fixtures, pipeline)


def run_grading_eval():
    print("Grading eval: needs held-out APTOS predictions (metrics.py ready).")


def run_segmentation_eval():
    print("Segmentation eval: needs IDRiD test-split predictions (metrics.py ready).")


def _git_commit_short():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT
        ).decode().strip()
    except Exception:
        return "unknown"


def patch_benchmarks_xai_section(xai_result):
    if xai_result is None:
        return

    with open(BENCHMARKS_PATH, "r", encoding="utf-8") as f:
        content = f.read()

    stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    commit = _git_commit_short()
    real = xai_result["real"]
    n = xai_result["n"]
    source_note = ("real pipeline, predicted lesion masks" if real
                   else "SYNTHETIC stub CAMs -- placeholder only, rerun with torch installed")

    content = content.replace(
        "Last updated: TBD · commit: TBD · **numbers frozen: NO**",
        f"Last updated: {stamp} · commit: {commit} · **numbers frozen: NO "
        f"(n={n}, {source_note})**",
    )

    if real:
        rates = xai_result["guard_trigger_rates"]
        content = content.replace(
            "| CAM_OFF_RETINA | TBD % | TBD |\n| LOW_AGREEMENT | TBD % | TBD |",
            f"| CAM_OFF_RETINA | {rates['CAM_OFF_RETINA'] * 100:.1f}% (n={n}) | "
            f"preliminary, see per-image log |\n"
            f"| LOW_AGREEMENT | {rates['LOW_AGREEMENT'] * 100:.1f}% (n={n}) | "
            f"preliminary, see per-image log |",
        )

        dels = [r["deletion_auc"] for r in xai_result["per_image"]]
        inss = [r["insertion_auc"] for r in xai_result["per_image"]]
        hits = [r["lesion_hit"] for r in xai_result["per_image"] if r["lesion_hit"] is not None]
        hit_rate = (sum(hits) / len(hits)) if hits else None
        content = content.replace(
            "| Deletion AUC | TBD |\n| Insertion AUC | TBD |\n"
            "| Lesion-localisation hit rate | TBD |",
            f"| Deletion AUC | {np.mean(dels):.4f} (n={n}, mean) |\n"
            f"| Insertion AUC | {np.mean(inss):.4f} (n={n}, mean) |\n"
            f"| Lesion-localisation hit rate | "
            f"{'TBD (no hits/misses)' if hit_rate is None else f'{hit_rate * 100:.1f}% (n={len(hits)}, predicted masks not ground truth)'} |",
        )

    with open(BENCHMARKS_PATH, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"\nPatched {os.path.relpath(BENCHMARKS_PATH, REPO_ROOT)}")


if __name__ == "__main__":
    xai_result = run_xai_eval()
    print()
    run_grading_eval()
    run_segmentation_eval()
    patch_benchmarks_xai_section(xai_result)