"""
python eval/compare_gradcam_variants.py [--dir PATH]

GUIDE_4_Anshika.md Part 1: "Consider Grad-CAM++ or Score-CAM if plain
Grad-CAM produces blobs too coarse to be useful... Compare the two on
~20 images and pick with evidence."

Picks a variant using deletion/insertion AUC: a good CAM should have
LOW deletion AUC (removing its top pixels tanks the prediction fast)
and HIGH insertion AUC (adding just its top pixels recovers the
prediction fast). We rank by (insertion - deletion), higher is better.

Run from the repo root: python eval/compare_gradcam_variants.py
"""
import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.xai.gradcam import GradCAM
from eval.xai_eval import deletion_insertion_auc

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DIR = os.path.join(REPO_ROOT, "tests", "fixtures")
BENCHMARKS_PATH = os.path.join(REPO_ROOT, "docs", "BENCHMARKS.md")


def _load_images(directory):
    exts = (".png", ".jpg", ".jpeg")
    images = []
    for name in sorted(os.listdir(directory)):
        if name.lower().endswith(exts):
            img = cv2.imread(os.path.join(directory, name))
            if img is not None:
                images.append((name, img))
    return images


def _run_variant(grader, bgr, variant):
    engine = GradCAM(grader.model, grader.gradcam_layer, variant=variant)
    tensor = grader.preprocess(bgr)
    cam, class_idx = engine(tensor, class_idx=None)

    bgr_at_cam_res = cv2.resize(bgr, (cam.shape[1], cam.shape[0]))

    def predict_fn(img, _grader=grader):
        return _grader.predict(img)["probabilities"]

    del_auc, ins_auc = deletion_insertion_auc(
        bgr_at_cam_res, cam, predict_fn, class_idx, n_steps=10)
    return del_auc, ins_auc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default=DEFAULT_DIR,
                         help="Folder of images to compare against (default: tests/fixtures, n=3)")
    args = parser.parse_args()

    try:
        from core.models.grading import DRGrader
    except ImportError as e:
        print(f"torch/model unavailable: {e}")
        return

    try:
        grader = DRGrader()
    except FileNotFoundError as e:
        print(f"checkpoint missing: {e}")
        return

    images = _load_images(args.dir)
    if not images:
        print(f"no images found in {args.dir}")
        return

    n = len(images)
    if n < 20:
        print(f"WARNING: n={n} images, guide asks for ~20 for this comparison.")
        print("Proceeding anyway -- report the real n in BENCHMARKS.md, don't round up.\n")

    print("=" * 70)
    print(f"Grad-CAM vs Grad-CAM++  (n={n} images from {args.dir})")
    print("=" * 70)

    results = {"gradcam": [], "gradcam++": []}
    for name, bgr in images:
        row = {"name": name}
        for variant in ("gradcam", "gradcam++"):
            del_auc, ins_auc = _run_variant(grader, bgr, variant)
            results[variant].append((del_auc, ins_auc))
            row[variant] = (del_auc, ins_auc)
        print(f"  {name:15s} "
              f"gradcam(del={row['gradcam'][0]:.4f} ins={row['gradcam'][1]:.4f})  "
              f"gradcam++(del={row['gradcam++'][0]:.4f} ins={row['gradcam++'][1]:.4f})")

    print()
    summary = {}
    for variant in ("gradcam", "gradcam++"):
        dels = [r[0] for r in results[variant]]
        inss = [r[1] for r in results[variant]]
        score = np.mean(inss) - np.mean(dels)
        summary[variant] = {
            "mean_del": float(np.mean(dels)),
            "mean_ins": float(np.mean(inss)),
            "score": float(score),
        }
        print(f"  {variant:10s} mean_del={summary[variant]['mean_del']:.4f}  "
              f"mean_ins={summary[variant]['mean_ins']:.4f}  "
              f"score(ins-del)={summary[variant]['score']:.4f}")

    winner = max(summary, key=lambda v: summary[v]["score"])
    print(f"\nWinner (higher insertion, lower deletion): {winner}")
    print(f"n={n} -- {'meets' if n >= 20 else 'does NOT meet'} the guide's ~20-image bar.")

    try:
        with open(BENCHMARKS_PATH, "r", encoding="utf-8") as f:
            content = f.read()
        old = "| Grad-CAM vs Grad-CAM++ chosen | TBD (~20-image comparison, not yet run) |"
        new = (f"| Grad-CAM vs Grad-CAM++ chosen | **{winner}** "
               f"(n={n}, score={summary[winner]['score']:.4f} vs "
               f"{summary['gradcam++' if winner == 'gradcam' else 'gradcam']['score']:.4f}"
               f"{'; n<20, directional only' if n < 20 else ''}) |")
        if old in content:
            content = content.replace(old, new)
            with open(BENCHMARKS_PATH, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"\nPatched {os.path.relpath(BENCHMARKS_PATH, REPO_ROOT)}.")
        else:
            print(f"\nCould not find the placeholder line in BENCHMARKS.md -- "
                  f"update it by hand with: {new}")
    except FileNotFoundError:
        print(f"\n{BENCHMARKS_PATH} not found -- skipping auto-patch.")


if __name__ == "__main__":
    main()