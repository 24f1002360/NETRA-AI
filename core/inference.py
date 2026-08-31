from __future__ import annotations

"""
NETRA-AI end-to-end inference orchestrator.

Pipeline:

    Image
      ↓
    Muskan IQA / Quality Gate
      ↓
    Kanchan DR grading
      ↓
    Kanchan lesion segmentation
      ↓
    Anshika XAI + guards
      ↓
    Divyanshu routing
      ↓
    ScreeningResult

Important integration notes:

1. Kanchan's current grading.py already calls Muskan's enhance()
   internally. Therefore this file passes the RAW image to grading
   to avoid double enhancement.

2. Kanchan's current segmentation.py returns a dictionary, while
   the original interface document described (lesions, mask).
   This file supports both formats.

3. Anshika's real core/xai/explain.py is not yet present in the
   supplied repository. Until it is added, the existing XAI stub
   is used so the complete pipeline can still run.

4. Database persistence/history belongs to Divyanshu's db/ layer.
   This file computes routing and supports an optional prior_result,
   but does not invent a database implementation.
"""

import importlib
import logging
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

from core.contracts import new_result
from core.iqa.enhance import enhance, resize_for_processing


PROJECT_ROOT = Path(__file__).resolve().parents[1]

APP_CONFIG_PATH = PROJECT_ROOT / "configs" / "app.yaml"
THRESHOLDS_PATH = PROJECT_ROOT / "configs" / "thresholds.yaml"

logger = logging.getLogger(__name__)


# ============================================================
# MODULE REGISTRY
# ============================================================

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


# ============================================================
# DEFAULT CONFIG
# ============================================================

DEFAULT_CONFIG = {
    "runtime": "pytorch",

    "modules": {
        "iqa": "real",
        "grading": "real",
        "segmentation": "real",
        "xai": "real",
    },

    "alerts": {
        "enabled": False,
    },

    "database": {
        "path": "data/netra.db",
    },

    "sync": {
        "enabled": True,
    },
}


DEFAULT_ROUTING = {
    "urgent_grade": 4,
    "urgent_grade_3_confidence": 0.70,
    "referable_grade": 2,
    "low_confidence": 0.55,
}


# ============================================================
# CONFIGURATION
# ============================================================

def load_config(
    path: str | Path = APP_CONFIG_PATH,
) -> dict[str, Any]:
    """
    Load NETRA application configuration.

    If configs/app.yaml does not exist, use a safe default
    configuration so the project can still run.
    """

    path = Path(path)

    if not path.exists():

        cfg = _deep_copy(DEFAULT_CONFIG)

        # Anshika's real explain.py is currently absent.
        real_xai = (
            PROJECT_ROOT
            / "core"
            / "xai"
            / "explain.py"
        )

        if real_xai.exists():
            cfg["modules"]["xai"] = "real"
        else:
            cfg["modules"]["xai"] = "stub"

        return cfg

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:

        loaded = yaml.safe_load(f) or {}

    cfg = _deep_merge(
        _deep_copy(DEFAULT_CONFIG),
        loaded,
    )

    # Do not select a real XAI implementation that
    # does not exist yet.
    if cfg["modules"].get("xai") == "real":

        real_xai = (
            PROJECT_ROOT
            / "core"
            / "xai"
            / "explain.py"
        )

        if not real_xai.exists():
            cfg["modules"]["xai"] = "stub"

    return cfg


def load_thresholds(
    path: str | Path = THRESHOLDS_PATH,
) -> dict[str, Any]:

    defaults = {
        "routing": dict(DEFAULT_ROUTING)
    }

    path = Path(path)

    if not path.exists():
        return defaults

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:

        loaded = yaml.safe_load(f) or {}

    routing = dict(DEFAULT_ROUTING)

    routing.update(
        loaded.get("routing", {}) or {}
    )

    return {
        "routing": routing
    }


def _deep_copy(value: Any) -> Any:

    if isinstance(value, dict):
        return {
            k: _deep_copy(v)
            for k, v in value.items()
        }

    if isinstance(value, list):
        return [
            _deep_copy(v)
            for v in value
        ]

    return value


def _deep_merge(
    base: dict[str, Any],
    extra: dict[str, Any],
) -> dict[str, Any]:

    for key, value in extra.items():

        if (
            isinstance(value, dict)
            and isinstance(base.get(key), dict)
        ):
            _deep_merge(
                base[key],
                value,
            )

        else:
            base[key] = value

    return base


# ============================================================
# MODULE LOADING
# ============================================================

def load_module(
    name: str,
    mode: str,
):

    if name not in MODULE_PATHS:
        raise ValueError(
            f"Unknown module: {name}"
        )

    if mode not in MODULE_PATHS[name]:
        raise ValueError(
            f"Unsupported mode '{mode}' "
            f"for module '{name}'"
        )

    module_path = MODULE_PATHS[name][mode]

    return importlib.import_module(
        module_path
    )


@lru_cache(maxsize=None)
def _load_model_object(
    name: str,
    mode: str,
):

    module = load_module(
        name,
        mode,
    )

    # Current Kanchan grading implementation
    if (
        name == "grading"
        and hasattr(module, "DRGrader")
    ):
        return module.DRGrader()

    # Current Kanchan segmentation implementation
    if (
        name == "segmentation"
        and hasattr(module, "DRSegmenter")
    ):
        return module.DRSegmenter()

    return module


# ============================================================
# IMAGE LOADING
# ============================================================

def load_image(
    image_path: str | Path,
) -> np.ndarray:

    path = Path(image_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Fundus image not found: {path}"
        )

    image = cv2.imread(
        str(path),
        cv2.IMREAD_COLOR,
    )

    if image is None:
        raise FileNotFoundError(
            f"Unable to read fundus image: {path}"
        )

    if image.dtype != np.uint8:
        image = image.astype(
            np.uint8
        )

    return image


# ============================================================
# KANCHAN — GRADING ADAPTER
# ============================================================

def _call_grade(
    module_or_model: Any,
    bgr: np.ndarray,
) -> dict[str, Any]:

    # Module/function style
    if callable(
        getattr(
            module_or_model,
            "grade",
            None,
        )
    ):
        return module_or_model.grade(
            bgr
        )

    raise AttributeError(
        "Grading module must expose "
        "grade(bgr) or DRGrader.grade()."
    )


# ============================================================
# KANCHAN — SEGMENTATION ADAPTER
# ============================================================

def _call_segment(
    module_or_model: Any,
    bgr: np.ndarray,
) -> tuple[
    dict[str, Any],
    np.ndarray | None,
]:

    if not hasattr(
        module_or_model,
        "segment",
    ):
        raise AttributeError(
            "Segmentation module must expose "
            "segment(bgr)."
        )

    raw = module_or_model.segment(
        bgr
    )

    # Original contract:
    #
    #     segment(bgr) -> (lesions, mask)
    #
    if (
        isinstance(raw, tuple)
        and len(raw) == 2
    ):

        lesions, mask = raw

        return (
            _normalise_lesions(
                lesions
            ),
            _normalise_mask_stack(
                mask
            ),
        )

    # Current Kanchan implementation:
    #
    #     segment(bgr) -> dict
    #
    if isinstance(raw, dict):

        return _normalise_segmentation_dict(
            raw
        )

    raise TypeError(
        "Unsupported segmentation output: "
        f"{type(raw).__name__}"
    )


def _normalise_segmentation_dict(
    raw: dict[str, Any],
) -> tuple[
    dict[str, Any],
    np.ndarray | None,
]:

    masks = raw.get(
        "masks"
    ) or {}

    statistics = raw.get(
        "statistics"
    ) or {}

    lesion_names = {
        "MA": "microaneurysms",
        "HE": "haemorrhages",
        "EX": "hard_exudates",
        "SE": "soft_exudates",
    }

    counts = {}

    for (
        short_name,
        contract_name,
    ) in lesion_names.items():

        mask = masks.get(
            short_name
        )

        if mask is None:
            counts[contract_name] = 0
            continue

        mask_bin = (
            np.asarray(mask) > 0
        )

        counts[
            contract_name
        ] = _connected_component_count(
            mask_bin
        )

    # Current Kanchan implementation reports
    # percentage coverage.
    #
    # Contract expects fraction 0..1.

    area_fraction = {
        "hard_exudates": 0.0,
        "haemorrhages": 0.0,
    }

    for (
        short_name,
        contract_name,
    ) in (
        ("EX", "hard_exudates"),
        ("HE", "haemorrhages"),
    ):

        stat = statistics.get(
            short_name
        ) or {}

        percentage = stat.get(
            "percentage",
            0.0,
        )

        area_fraction[
            contract_name
        ] = float(
            np.clip(
                float(percentage)
                / 100.0,
                0.0,
                1.0,
            )
        )

    lesions = {
        "mask_path": "",
        "counts": counts,
        "area_fraction": area_fraction,
        "model_version": raw.get(
            "model_version"
        ),
    }

    mask_stack = _masks_to_stack(
        masks
    )

    return (
        lesions,
        mask_stack,
    )


def _normalise_lesions(
    lesions: Any,
) -> dict[str, Any]:

    if not isinstance(
        lesions,
        dict,
    ):
        raise TypeError(
            "Lesions block must be a dictionary."
        )

    counts = (
        lesions.get(
            "counts",
            {},
        )
        or {}
    )

    area_fraction = (
        lesions.get(
            "area_fraction",
            {},
        )
        or {}
    )

    return {
        "mask_path": lesions.get(
            "mask_path",
            "",
        ),

        "counts": {
            "microaneurysms": int(
                counts.get(
                    "microaneurysms",
                    0,
                )
            ),

            "haemorrhages": int(
                counts.get(
                    "haemorrhages",
                    0,
                )
            ),

            "hard_exudates": int(
                counts.get(
                    "hard_exudates",
                    0,
                )
            ),

            "soft_exudates": int(
                counts.get(
                    "soft_exudates",
                    0,
                )
            ),
        },

        "area_fraction": {
            "hard_exudates": float(
                area_fraction.get(
                    "hard_exudates",
                    0.0,
                )
            ),

            "haemorrhages": float(
                area_fraction.get(
                    "haemorrhages",
                    0.0,
                )
            ),
        },

        "model_version": lesions.get(
            "model_version"
        ),
    }


def _normalise_mask_stack(
    mask: Any,
) -> np.ndarray | None:

    if mask is None:
        return None

    arr = np.asarray(mask)

    if arr.ndim == 2:
        return arr.astype(
            np.uint8
        )

    if arr.ndim == 3:

        if arr.shape[2] == 4:
            return arr.astype(
                np.uint8
            )

        if arr.shape[0] == 4:
            return np.transpose(
                arr,
                (1, 2, 0),
            ).astype(
                np.uint8
            )

    raise ValueError(
        f"Unsupported lesion mask shape: "
        f"{arr.shape}"
    )


def _masks_to_stack(
    masks: dict[str, Any],
) -> np.ndarray | None:

    ordered = []

    for key in (
        "MA",
        "HE",
        "EX",
        "SE",
    ):

        if key not in masks:
            continue

        arr = np.asarray(
            masks[key]
        )

        if arr.ndim != 2:
            continue

        ordered.append(
            (arr > 0).astype(
                np.uint8
            )
        )

    if not ordered:
        return None

    return np.stack(
        ordered,
        axis=-1,
    )


def _connected_component_count(
    mask: np.ndarray,
) -> int:

    mask_u8 = (
        mask > 0
    ).astype(
        np.uint8
    )

    if mask_u8.size == 0:
        return 0

    n_labels, _, _, _ = (
        cv2.connectedComponentsWithStats(
            mask_u8,
            connectivity=8,
        )
    )

    return max(
        0,
        int(n_labels) - 1,
    )


def _combined_lesion_mask(
    mask: np.ndarray | None,
) -> np.ndarray | None:

    if mask is None:
        return None

    arr = np.asarray(mask)

    if arr.ndim == 2:
        return (
            arr > 0
        ).astype(
            np.uint8
        )

    if arr.ndim == 3:

        if arr.shape[2] == 4:
            return (
                arr.any(axis=2)
            ).astype(
                np.uint8
            )

        if arr.shape[0] == 4:
            return (
                arr.any(axis=0)
            ).astype(
                np.uint8
            )

    return None


def _resize_mask(
    mask: np.ndarray | None,
    shape_hw: tuple[int, int],
):

    if mask is None:
        return None

    h, w = shape_hw

    return cv2.resize(
        (mask > 0).astype(
            np.uint8
        ),
        (w, h),
        interpolation=cv2.INTER_NEAREST,
    )


# ============================================================
# RUNTIME ARTIFACTS
# ============================================================

def _save_runtime_artifacts(
    screening_id: str,
    raw_bgr: np.ndarray,
    quality: dict[str, Any],
    fov_mask: np.ndarray | None,
    lesion_mask: np.ndarray | None,
    output_dir: str | Path | None,
) -> tuple[str, str]:

    if output_dir is None:
        return "", ""

    root = (
        Path(output_dir)
        / screening_id
    )

    root.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Processed image is for UI/report display.
    processed = enhance(
        raw_bgr,
        quality=quality,
    )

    processed_path = (
        root
        / "processed.png"
    )

    cv2.imwrite(
        str(processed_path),
        processed,
    )

    fov_path = ""

    if fov_mask is not None:

        fov_path_obj = (
            root
            / "fov_mask.png"
        )

        cv2.imwrite(
            str(fov_path_obj),
            fov_mask,
        )

        fov_path = str(
            fov_path_obj
        )

    lesion_path = ""

    combined = (
        _combined_lesion_mask(
            lesion_mask
        )
    )

    if combined is not None:

        lesion_path_obj = (
            root
            / "lesion_mask.png"
        )

        cv2.imwrite(
            str(lesion_path_obj),
            combined * 255,
        )

        lesion_path = str(
            lesion_path_obj
        )

    return (
        str(processed_path),
        lesion_path,
    )


# ============================================================
# FAIL-SAFE BLOCKS
# ============================================================

def _empty_grading():

    return {
        "icdr_grade": None,
        "grade_label": None,
        "probabilities": [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        "referable_dr": False,
        "confidence": 0.0,
        "uncertain": True,
        "model_id": None,
        "model_version": None,
    }


def _empty_xai():

    return {
        "gradcam_path": None,
        "overlay_path": None,
        "cam_lesion_agreement": None,
        "cam_outside_fov_fraction": None,
        "guard_status": "LOW_AGREEMENT",
    }


# ============================================================
# ANSHIKA — XAI
# ============================================================

def _safe_xai(
    bgr: np.ndarray,
    grading: dict[str, Any],
    lesion_mask: np.ndarray | None,
    fov_mask: np.ndarray | None,
    grading_model: Any,
    xai_mode: str,
) -> dict[str, Any]:

    try:

        xai = _load_model_object(
            "xai",
            xai_mode,
        )

        fn = getattr(
            xai,
            "explain",
            None,
        )

        if fn is None:
            raise AttributeError(
                "XAI module must expose explain()."
            )

        # Real Anshika contract:
        #
        # explain(
        #     bgr,
        #     model_handle,
        #     grading,
        #     lesion_mask,
        #     fov_mask
        # )

        if xai_mode == "real":

            model_handle = getattr(
                grading_model,
                "model",
                grading_model,
            )

            try:

                raw = fn(
                    bgr,
                    model_handle,
                    grading,
                    lesion_mask,
                    fov_mask,
                )

            except TypeError:

                # Temporary compatibility with
                # a simplified implementation.
                raw = fn(
                    bgr,
                    grading=grading,
                    lesion_mask=lesion_mask,
                    fov_mask=fov_mask,
                )

        else:

            raw = fn(
                bgr,
                grading=grading,
                lesion_mask=lesion_mask,
                fov_mask=fov_mask,
            )

        return _normalise_xai(
            raw
        )

    except Exception:

        logger.exception(
            "XAI stage failed. "
            "Continuing without XAI."
        )

        return _empty_xai()


def _normalise_xai(
    raw: Any,
) -> dict[str, Any]:

    if not isinstance(
        raw,
        dict,
    ):
        raise TypeError(
            "XAI output must be a dictionary."
        )

    status = raw.get(
        "guard_status",
        "LOW_AGREEMENT",
    )

    if status not in {
        "OK",
        "CAM_OFF_RETINA",
        "LOW_AGREEMENT",
    }:

        status = "LOW_AGREEMENT"

    return {
        "gradcam_path": raw.get(
            "gradcam_path"
        ),

        "overlay_path": raw.get(
            "overlay_path"
        ),

        "cam_lesion_agreement":
            _optional_float(
                raw.get(
                    "cam_lesion_agreement"
                )
            ),

        "cam_outside_fov_fraction":
            _optional_float(
                raw.get(
                    "cam_outside_fov_fraction"
                )
            ),

        "guard_status": status,
    }


def _optional_float(
    value: Any,
) -> float | None:

    if value is None:
        return None

    try:
        return float(value)

    except (
        TypeError,
        ValueError,
    ):
        return None


# ======================