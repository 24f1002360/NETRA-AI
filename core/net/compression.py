"""
Bandwidth-adaptive compression for the NETRA sync queue.

Selects the highest-quality JPEG rung that fits the estimated
upload-time budget for the available bandwidth.
"""

from __future__ import annotations

import os
from pathlib import Path

import cv2


# quality, max long edge
LADDER = [
    ("full", 95, None),
    ("high", 85, 2048),
    ("medium", 75, 1536),
    ("low", 60, 1024),
    ("minimal", 45, 768),
]


def _encode(
    bgr,
    quality: int,
    max_dim: int | None,
    path: str | Path,
) -> int:
    """Encode an image as JPEG and return its byte size."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    img = bgr

    if max_dim:
        height, width = img.shape[:2]

        if max(height, width) > max_dim:
            scale = max_dim / max(height, width)

            new_width = max(
                1,
                int(width * scale),
            )
            new_height = max(
                1,
                int(height * scale),
            )

            img = cv2.resize(
                img,
                (new_width, new_height),
                interpolation=cv2.INTER_AREA,
            )

    success = cv2.imwrite(
        str(path),
        img,
        [cv2.IMWRITE_JPEG_QUALITY, quality],
    )

    if not success:
        raise IOError(
            f"Failed to write compressed image: {path}"
        )

    return path.stat().st_size


def compress_for_sync(
    image_path,
    bandwidth_kbps: float,
    target_seconds: float = 20.0,
    out_dir=None,
):
    """
    Pick the highest-quality rung that uploads within target_seconds.

    Returns:
        (
            compressed_path,
            {
                "rung": str,
                "quality": int,
                "bytes": int,
                "mb": float,
                "est_seconds": float,
                "bandwidth_kbps": float,
                "fits_budget": bool,
            }
        )
    """

    if bandwidth_kbps <= 0:
        raise ValueError(
            "bandwidth_kbps must be greater than zero"
        )

    if target_seconds <= 0:
        raise ValueError(
            "target_seconds must be greater than zero"
        )

    image_path = Path(image_path)

    bgr = cv2.imread(str(image_path))

    if bgr is None:
        raise FileNotFoundError(
            str(image_path)
        )

    if out_dir is None:
        out_dir = image_path.parent

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = image_path.stem

    # bandwidth_kbps -> bytes/second
    bytes_per_second = (
        bandwidth_kbps * 1000.0 / 8.0
    )

    budget_bytes = (
        bytes_per_second * target_seconds
    )

    chosen = None

    for rung, quality, max_dim in LADDER:

        output_path = (
            out_dir
            / f"{stem}_{rung}.jpg"
        )

        size = _encode(
            bgr,
            quality,
            max_dim,
            output_path,
        )

        estimated_seconds = (
            size / bytes_per_second
        )

        if size <= budget_bytes:

            chosen = (
                output_path,
                rung,
                quality,
                size,
                estimated_seconds,
            )

            break

        try:
            output_path.unlink()
        except FileNotFoundError:
            pass

    # Even minimal quality can exceed the budget.
    if chosen is None:

        rung, quality, max_dim = LADDER[-1]

        output_path = (
            out_dir
            / f"{stem}_{rung}.jpg"
        )

        size = _encode(
            bgr,
            quality,
            max_dim,
            output_path,
        )

        estimated_seconds = (
            size / bytes_per_second
        )

        chosen = (
            output_path,
            rung,
            quality,
            size,
            estimated_seconds,
        )

    (
        output_path,
        rung,
        quality,
        size,
        estimated_seconds,
    ) = chosen

    return str(output_path), {
        "rung": rung,
        "quality": quality,
        "bytes": size,
        "mb": round(size / 1e6, 3),
        "est_seconds": round(
            estimated_seconds,
            1,
        ),
        "bandwidth_kbps": bandwidth_kbps,
        "fits_budget": (
            estimated_seconds <= target_seconds
        ),
    }


def ladder_sizes(
    image_path,
    out_dir=None,
):
    """
    Return byte counts for every compression rung.
    """

    image_path = Path(image_path)

    bgr = cv2.imread(str(image_path))

    if bgr is None:
        raise FileNotFoundError(
            str(image_path)
        )

    if out_dir is None:
        out_dir = image_path.parent

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = image_path.stem

    rows = []

    for rung, quality, max_dim in LADDER:

        output_path = (
            out_dir
            / f"{stem}_{rung}.jpg"
        )

        size = _encode(
            bgr,
            quality,
            max_dim,
            output_path,
        )

        rows.append(
            {
                "rung": rung,
                "quality": quality,
                "max_dim": max_dim,
                "bytes": size,
                "mb": round(
                    size / 1e6,
                    3,
                ),
            }
        )

        try:
            output_path.unlink()
        except FileNotFoundError:
            pass

    return rows


def sweep_quality_ladder(
    image_paths,
    grade_fn,
    out_dir=None,
):
    """
    Measure grade agreement across compression levels.

    grade_fn receives an image path and returns an ICDR grade.
    """

    if out_dir is None:
        out_dir = Path("/tmp")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {
        rung: {
            "agree": 0,
            "n": 0,
            "bytes": 0,
        }
        for rung, _, _ in LADDER
    }

    for image_path in image_paths:

        image_path = Path(image_path)

        bgr = cv2.imread(str(image_path))

        if bgr is None:
            continue

        base_grade = grade_fn(
            str(image_path)
        )

        stem = image_path.stem

        for rung, quality, max_dim in LADDER:

            output_path = (
                out_dir
                / f"{stem}_{rung}.jpg"
            )

            size = _encode(
                bgr,
                quality,
                max_dim,
                output_path,
            )

            compressed_grade = grade_fn(
                str(output_path)
            )

            results[rung]["agree"] += int(
                compressed_grade == base_grade
            )

            results[rung]["n"] += 1
            results[rung]["bytes"] += size

            try:
                output_path.unlink()
            except FileNotFoundError:
                pass

    return [
        {
            "rung": rung,
            "grade_agreement": (
                values["agree"]
                / max(values["n"], 1)
            ),
            "mean_mb": round(
                values["bytes"]
                / max(values["n"], 1)
                / 1e6,
                3,
            ),
        }
        for rung, values in results.items()
    ]