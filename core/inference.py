
from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import yaml
import cv2

from core.contracts import new_result


PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP_CONFIG_PATH = PROJECT_ROOT / "configs" / "app.yaml"


def load_config(path: str | Path = APP_CONFIG_PATH) -> dict[str, Any]:
    """Load NETRA AI application configuration from YAML."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}
    


MODULE_PATHS = {
    "iqa": {
        "stub": "core.stubs.iqa_stub",
        "real": "core.iqa.quality",
    },
    "grading": {
        "stub": "core.stubs.grading_stub",
        "real": "core.models.grading",
    },
    "segmentation": {
        "stub": "core.stubs.segmentation_stub",
        "real": "core.models.segmentation",
    },
    "xai": {
        "stub": "core.stubs.xai_stub",
        "real": "core.xai.explain",
    },
}


def load_module(name: str, mode: str):
    """Load a configured NETRA module implementation."""
    if name not in MODULE_PATHS:
        raise ValueError(f"Unknown module: {name}")

    if mode not in MODULE_PATHS[name]:
        raise ValueError(
            f"Unsupported mode '{mode}' for module '{name}'"
        )

    module_path = MODULE_PATHS[name][mode]
    return importlib.import_module(module_path)

def load_image(image_path: str | Path):
    """Load a fundus image as a BGR uint8 OpenCV image."""
    image_path = Path(image_path)

    image = cv2.imread(str(image_path))

    if image is None:
        raise FileNotFoundError(
            f"Unable to read fundus image: {image_path}"
        )

    return image

def run_screening(
    image_path: str | Path,
    patient_id: str,
    eye: str,
    operator_id: str = "",
    phc_id: str = "",
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run the NETRA screening pipeline.

    Current stage:
        - create ScreeningResult
        - load image
        - populate image metadata

    Additional pipeline stages will be added incrementally.
    """

    result = new_result(
        patient_id=patient_id,
        eye=eye,
        operator_id=operator_id,
        phc_id=phc_id,
    )

    bgr = load_image(image_path)

    height, width = bgr.shape[:2]

    result["image"]["raw_path"] = str(image_path)
    result["image"]["width"] = width
    result["image"]["height"] = height

    config = config or load_config()

    iqa_module = load_module(
        "iqa",
        config["modules"]["iqa"],
    )

    quality = iqa_module.assess_quality(bgr)

    quality.pop("_mask", None)

    result["quality"] = quality

    if quality["verdict"] == "RETAKE":
        return result

    grading_module = load_module(
        "grading",
        config["modules"]["grading"],
    )

    grading = grading_module.grade(bgr)

    result["grading"] = grading

    segmentation_module = load_module(
        "segmentation",
        config["modules"]["segmentation"],
    )

    lesions, lesion_mask = segmentation_module.segment(bgr)

    result["lesions"] = lesions

    return result
    