"""
Generate the three shared test fixtures.

Everyone's tests run against these, so they must exist before anyone can
write a test. Two modes:

    # now, before APTOS is downloaded - synthetic placeholders
    python tools/make_fixtures.py --synthetic

    # once APTOS is available - real images, deterministic pick
    python tools/make_fixtures.py --aptos data/aptos/train_images \
                                 --csv data/aptos/train.csv

Fixtures produced:
    good.png     clean, gradable, expect PASS
    blurry.png   heavily blurred, expect RETAKE / BLUR_HIGH
    severe.png   gradable but high-grade disease, expect PASS
"""

import argparse
import os
import shutil

import cv2
import numpy as np

OUT = "tests/fixtures"


def _synthetic_fundus(seed=0, lesions=0, size=1400):
    rng = np.random.default_rng(seed)
    h = int(size * 0.67)
    w = size
    img = np.zeros((h, w, 3), np.uint8)
    cx, cy, r = w // 2, h // 2, int(min(h, w) * 0.47)
    cv2.circle(img, (cx, cy), r, (40, 110, 190), -1)

    for _ in range(140):                      # vessels
        p1 = (int(rng.integers(w * .25, w * .75)), int(rng.integers(h * .2, h * .8)))
        p2 = (p1[0] + int(rng.integers(-w * .08, w * .08)),
              p1[1] + int(rng.integers(-h * .12, h * .12)))
        cv2.line(img, p1, p2, (20, 50, 120), int(rng.integers(2, 5)))

    cv2.circle(img, (int(w * .66), cy), int(r * .13), (120, 210, 240), -1)   # optic disc

    for _ in range(lesions):                  # haemorrhage-like spots
        c = (int(rng.integers(w * .3, w * .7)), int(rng.integers(h * .25, h * .75)))
        cv2.circle(img, c, int(rng.integers(3, 9)), (55, 55, 200), -1)
    for _ in range(lesions // 2):             # exudate-like spots
        c = (int(rng.integers(w * .3, w * .7)), int(rng.integers(h * .25, h * .75)))
        cv2.circle(img, c, int(rng.integers(4, 11)), (90, 235, 245), -1)

    img = np.clip(img.astype(np.float32) + rng.normal(0, 6, img.shape), 0, 255).astype(np.uint8)
    mask = np.zeros((h, w), np.uint8)
    cv2.circle(mask, (cx, cy), r, 255, -1)
    return cv2.bitwise_and(img, img, mask=mask)


def build_synthetic():
    os.makedirs(OUT, exist_ok=True)
    good = _synthetic_fundus(seed=1, lesions=0)
    cv2.imwrite(f"{OUT}/good.png", good)
    cv2.imwrite(f"{OUT}/blurry.png", cv2.GaussianBlur(good, (0, 0), 7))
    cv2.imwrite(f"{OUT}/severe.png", _synthetic_fundus(seed=2, lesions=60))
    print(f"wrote 3 synthetic fixtures to {OUT}/")
    print("PLACEHOLDERS. Rerun with --aptos once the dataset is available.")


def build_from_aptos(img_dir, csv_path):
    import pandas as pd
    os.makedirs(OUT, exist_ok=True)
    df = pd.read_csv(csv_path)

    # deterministic pick so everyone gets the same three images
    g0 = df[df.diagnosis == 0].sort_values("id_code").iloc[0]
    g4 = df[df.diagnosis >= 3].sort_values("id_code").iloc[0]

    def find(stem):
        for ext in (".png", ".jpg", ".jpeg"):
            p = os.path.join(img_dir, stem + ext)
            if os.path.exists(p):
                return p
        raise FileNotFoundError(stem)

    shutil.copy(find(g0.id_code), f"{OUT}/good.png")
    good = cv2.imread(f"{OUT}/good.png")
    cv2.imwrite(f"{OUT}/blurry.png", cv2.GaussianBlur(good, (0, 0), 9))
    shutil.copy(find(g4.id_code), f"{OUT}/severe.png")

    with open(f"{OUT}/SOURCES.txt", "w") as f:
        f.write(f"good.png   APTOS {g0.id_code} (grade 0)\n")
        f.write(f"blurry.png APTOS {g0.id_code} + Gaussian blur sigma=9\n")
        f.write(f"severe.png APTOS {g4.id_code} (grade {g4.diagnosis})\n")
    print(f"wrote 3 real fixtures to {OUT}/ (see SOURCES.txt)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--aptos", help="APTOS train_images directory")
    ap.add_argument("--csv", help="APTOS train.csv")
    a = ap.parse_args()
    if a.aptos and a.csv:
        build_from_aptos(a.aptos, a.csv)
    else:
        build_synthetic()