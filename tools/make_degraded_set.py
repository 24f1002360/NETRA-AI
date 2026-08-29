"""
Generate a labelled quality test set without hand-labelling anything.

Take N clean fundus images (known good), apply known degradations at known
severities, and you get a perfectly labelled set for free -- plus a severity
knob, which gives you a better result than a plain confusion matrix:
a curve of retake rate vs degradation severity.

Usage:
    python tools/make_degraded_set.py --src data/aptos/clean --out data/qa_set
    python tools/make_degraded_set.py --evaluate data/qa_set
"""

import argparse
import csv
import os
from pathlib import Path

import cv2
import numpy as np



def deg_blur(img, s):
    sigma = 0.5 + s * 6.0
    return cv2.GaussianBlur(img, (0, 0), sigma)


def deg_dark(img, s):
    gamma = 1.0 + s * 2.0
    lut = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)], np.uint8)
    return cv2.LUT(img, lut)


def deg_bright(img, s):
    gamma = 1.0 - s * 0.7
    lut = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)], np.uint8)
    return cv2.LUT(img, lut)


def deg_partial_fov(img, s):
    h, w = img.shape[:2]
    shift = int(s * 0.45 * w)
    M = np.float32([[1, 0, shift], [0, 1, 0]])
    return cv2.warpAffine(img, M, (w, h))


def deg_glare(img, s):
    h, w = img.shape[:2]
    overlay = np.zeros((h, w), np.float32)
    cx, cy = int(w * 0.55), int(h * 0.5)
    radius = int(min(h, w) * (0.05 + s * 0.15))
    cv2.circle(overlay, (cx, cy), radius, 1.0, -1)
    overlay = cv2.GaussianBlur(overlay, (0, 0), radius / 2.0)
    out = img.astype(np.float32)
    for c in range(3):
        out[:, :, c] += overlay * 255 * s
    return np.clip(out, 0, 255).astype(np.uint8)


DEGRADATIONS = {
    "blur": deg_blur,
    "dark": deg_dark,
    "bright": deg_bright,
    "partial_fov": deg_partial_fov,
    "glare": deg_glare,
}

SEVERITIES = [0.2, 0.4, 0.6, 0.8, 1.0]


def build(src, out):
    src, out = Path(src), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []

    images = sorted([p for p in src.iterdir()
                     if p.suffix.lower() in (".jpg", ".jpeg", ".png")])
    if not images:
        raise SystemExit(f"No images found in {src}")

    for p in images:
        img = cv2.imread(str(p))
        if img is None:
            continue

        clean_name = f"{p.stem}__clean.png"
        cv2.imwrite(str(out / clean_name), img)
        rows.append([clean_name, "none", 0.0, "good"])

        for name, fn in DEGRADATIONS.items():
            for s in SEVERITIES:
                dst = f"{p.stem}__{name}_{int(s*100)}.png"
                cv2.imwrite(str(out / dst), fn(img, s))
                label = "bad" if s >= 0.6 else "borderline"
                rows.append([dst, name, s, label])

    with open(out / "labels.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["filename", "degradation", "severity", "label"])
        w.writerows(rows)

    print(f"Wrote {len(rows)} images to {out}")
    print(f"Labels: {out / 'labels.csv'}")


def evaluate(qa_dir):
    """Run the Quality Gate over the set and report retake behaviour."""
    import sys
    sys.path.insert(0, os.getcwd())
    from core.iqa.quality import assess_quality

    qa_dir = Path(qa_dir)
    with open(qa_dir / "labels.csv") as f:
        rows = list(csv.DictReader(f))

    stats = {}
    tp = fp = tn = fn = 0
    for r in rows:
        img = cv2.imread(str(qa_dir / r["filename"]))
        q = assess_quality(img)
        retake = q["verdict"] == "RETAKE"
        key = (r["degradation"], r["severity"])
        stats.setdefault(key, [0, 0])
        stats[key][0] += int(retake)
        stats[key][1] += 1

        if r["label"] == "bad":
            tp += retake
            fn += (not retake)
        elif r["label"] == "good":
            fp += retake
            tn += (not retake)

    print("\nRetake rate by degradation and severity")
    print(f"{'degradation':<14}{'severity':>10}{'retake rate':>14}")
    for (d, s), (n_retake, n) in sorted(stats.items()):
        print(f"{d:<14}{s:>10}{n_retake/n:>13.0%}")

    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    print(f"\nOn clearly-bad vs clearly-good images:")
    print(f"  retake precision : {prec:.2f}   (target >= 0.85)")
    print(f"  retake recall    : {rec:.2f}   (target >= 0.80)")
    print("\nTune configs/thresholds.yaml until both targets are met.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", help="folder of clean fundus images")
    ap.add_argument("--out", default="data/qa_set")
    ap.add_argument("--evaluate", metavar="QA_DIR",
                    help="run the Quality Gate over an existing set")
    a = ap.parse_args()

    if a.evaluate:
        evaluate(a.evaluate)
    elif a.src:
        build(a.src, a.out)
    else:
        ap.print_help()
