"""
Bandwidth-adaptive compression for the sync queue.

Village connectivity is intermittent and slow. A raw fundus capture is a
few MB; pushing it over a 256 kbps link takes a couple of minutes per
patient. This picks a JPEG quality rung that fits the available bandwidth.

The interesting question, and the one worth measuring, is how far you can
compress before the grade changes. See `sweep_quality_ladder()` - run it
against Kanchan's model to find the operating point.
"""

import os

import cv2

# quality, max long edge
LADDER = [
    ("full",   95, None),
    ("high",   85, 2048),
    ("medium", 75, 1536),
    ("low",    60, 1024),
    ("minimal", 45, 768),
]


def _encode(bgr, quality, max_dim, path):
    img = bgr
    if max_dim:
        h, w = img.shape[:2]
        if max(h, w) > max_dim:
            s = max_dim / max(h, w)
            img = cv2.resize(img, (int(w * s), int(h * s)),
                             interpolation=cv2.INTER_AREA)
    cv2.imwrite(path, img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return os.path.getsize(path)


def compress_for_sync(image_path, bandwidth_kbps, target_seconds=20.0,
                      out_dir=None):
    """Pick the highest rung that uploads within target_seconds.

    Returns (path, meta) where meta carries the rung, byte count and the
    estimated upload time - Ishank's sync model consumes these.
    """
    bgr = cv2.imread(str(image_path))
    if bgr is None:
        raise FileNotFoundError(image_path)

    out_dir = out_dir or os.path.dirname(image_path) or "."
    stem = os.path.splitext(os.path.basename(image_path))[0]
    budget_bytes = bandwidth_kbps * 1000 / 8 * target_seconds

    chosen = None
    for rung, q, max_dim in LADDER:
        path = os.path.join(out_dir, f"{stem}_{rung}.jpg")
        size = _encode(bgr, q, max_dim, path)
        est = size * 8 / (bandwidth_kbps * 1000)
        if size <= budget_bytes:
            chosen = (path, rung, q, size, est)
            break
        os.remove(path)

    if chosen is None:                       # even the smallest rung is too big
        rung, q, max_dim = LADDER[-1]
        path = os.path.join(out_dir, f"{stem}_{rung}.jpg")
        size = _encode(bgr, q, max_dim, path)
        chosen = (path, rung, q, size, size * 8 / (bandwidth_kbps * 1000))

    path, rung, q, size, est = chosen
    return path, {
        "rung": rung, "quality": q, "bytes": size,
        "mb": round(size / 1e6, 3),
        "est_seconds": round(est, 1),
        "bandwidth_kbps": bandwidth_kbps,
        "fits_budget": est <= target_seconds,
    }


def ladder_sizes(image_path, out_dir=None):
    """Byte count at every rung. Feeds the sync model and the payload chart."""
    bgr = cv2.imread(str(image_path))
    out_dir = out_dir or "/tmp"
    stem = os.path.splitext(os.path.basename(image_path))[0]
    rows = []
    for rung, q, max_dim in LADDER:
        p = os.path.join(out_dir, f"{stem}_{rung}.jpg")
        size = _encode(bgr, q, max_dim, p)
        rows.append({"rung": rung, "quality": q, "max_dim": max_dim,
                     "bytes": size, "mb": round(size / 1e6, 3)})
        os.remove(p)
    return rows


def sweep_quality_ladder(image_paths, grade_fn, out_dir="/tmp"):
    """How much compression before the grade changes?

    grade_fn takes a path and returns an ICDR grade. Pass Kanchan's model.
    The output is the payload-vs-accuracy curve - a slide nobody else will
    have, and the number that justifies the chosen operating point.
    """
    results = {rung: {"agree": 0, "n": 0, "bytes": 0} for rung, _, _ in LADDER}
    for path in image_paths:
        bgr = cv2.imread(str(path))
        if bgr is None:
            continue
        base = grade_fn(path)
        stem = os.path.splitext(os.path.basename(path))[0]
        for rung, q, max_dim in LADDER:
            p = os.path.join(out_dir, f"{stem}_{rung}.jpg")
            size = _encode(bgr, q, max_dim, p)
            results[rung]["agree"] += int(grade_fn(p) == base)
            results[rung]["n"] += 1
            results[rung]["bytes"] += size
            os.remove(p)
    return [{"rung": r,
             "grade_agreement": v["agree"] / max(v["n"], 1),
             "mean_mb": round(v["bytes"] / max(v["n"], 1) / 1e6, 3)}
            for r, v in results.items()]