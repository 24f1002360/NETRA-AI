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
# ============================================================
# SCREENING ORCHESTRATOR
# ============================================================

def _empty_quality():
    return {
        "verdict": "RETAKE",
        "scores": {
            "blur": 0.0,
            "illumination": 0.0,
            "fov_coverage": 0.0,
            "contrast": 0.0,
            "artefact": 0.0,
        },
        "reasons": [],
        "operator_message_key": "",
        "enhancement_applied": [],
    }


def _safe_quality(
    module: Any,
    bgr: np.ndarray,
) -> dict[str, Any]:

    try:
        fn = getattr(module, "assess_quality", None)

        if fn is None:
            raise AttributeError(
                "IQA module must expose assess_quality()."
            )

        raw = fn(bgr)

        if not isinstance(raw, dict):
            raise TypeError(
                "IQA output must be a dictionary."
            )

        return raw

    except Exception:
        logger.exception(
            "IQA stage failed. Treating image as RETAKE."
        )

        return _empty_quality()


def _safe_enhance(
    bgr: np.ndarray,
    quality: dict[str, Any],
) -> np.ndarray:

    try:
        # Use the shared preprocessing implementation.
        return enhance(
            bgr,
            quality=quality,
        )

    except TypeError:
        # Compatibility with current implementations
        # exposing enhance(bgr).
        return enhance(bgr)

    except Exception:
        logger.exception(
            "Enhancement failed. Using original image."
        )
        return bgr


def _safe_grading(
    model: Any,
    bgr: np.ndarray,
) -> dict[str, Any]:

    try:
        result = _call_grade(
            model,
            bgr,
        )

        if not isinstance(result, dict):
            raise TypeError(
                "Grading output must be a dictionary."
            )

        return result

    except Exception:
        logger.exception(
            "Grading stage failed."
        )
        return _empty_grading()


def _safe_segmentation(
    model: Any,
    bgr: np.ndarray,
) -> tuple[
    dict[str, Any],
    np.ndarray | None,
]:

    try:
        return _call_segment(
            model,
            bgr,
        )

    except Exception:
        logger.exception(
            "Segmentation stage failed. "
            "Continuing without lesions."
        )

        return (
            {
                "mask_path": "",
                "counts": {
                    "microaneurysms": 0,
                    "haemorrhages": 0,
                    "hard_exudates": 0,
                    "soft_exudates": 0,
                },
                "area_fraction": {
                    "hard_exudates": 0.0,
                    "haemorrhages": 0.0,
                },
                "model_version": None,
            },
            None,
        )


def _normalise_quality(
    quality: Any,
) -> dict[str, Any]:

    if not isinstance(quality, dict):
        return _empty_quality()

    scores = quality.get("scores") or {}

    verdict = quality.get(
        "verdict",
        "RETAKE",
    )

    if verdict not in {
        "PASS",
        "AUTO_CORRECTED",
        "RETAKE",
    }:
        verdict = "RETAKE"

    return {
        "verdict": verdict,
        "scores": {
            "blur": float(scores.get("blur", 0.0)),
            "illumination": float(
                scores.get("illumination", 0.0)
            ),
            "fov_coverage": float(
                scores.get("fov_coverage", 0.0)
            ),
            "contrast": float(
                scores.get("contrast", 0.0)
            ),
            "artefact": float(
                scores.get("artefact", 0.0)
            ),
        },
        "reasons": list(
            quality.get("reasons") or []
        ),
        "operator_message_key": quality.get(
            "operator_message_key",
            "",
        ),
        "enhancement_applied": list(
            quality.get(
                "enhancement_applied",
                [],
            )
            or []
        ),
    }


def _get_fov_mask(
    quality: dict[str, Any],
    bgr: np.ndarray,
) -> np.ndarray | None:

    # Prefer a mask path supplied by IQA.
    path = quality.get(
        "fov_mask_path"
    )

    if path:
        try:
            mask = cv2.imread(
                str(path),
                cv2.IMREAD_GRAYSCALE,
            )

            if mask is not None:
                return mask
        except Exception:
            logger.exception(
                "Unable to load FOV mask."
            )

    # Current IQA implementation may expose FOV
    # functionality through core.iqa.fov.
    try:
        fov_module = importlib.import_module(
            "core.iqa.fov"
        )

        for fn_name in (
            "get_fov_mask",
            "fov_mask",
            "create_fov_mask",
            "detect_fov",
        ):

            fn = getattr(
                fov_module,
                fn_name,
                None,
            )

            if callable(fn):

                try:
                    mask = fn(bgr)

                    if mask is not None:
                        return np.asarray(
                            mask
                        ).astype(
                            np.uint8
                        )
                except Exception:
                    continue

    except Exception:
        pass

    return None


def _update_image_block(
    result: dict[str, Any],
    image_path: str | Path,
    bgr: np.ndarray,
    processed_path: str = "",
    fov_mask_path: str = "",
):

    image = result["image"]

    image["raw_path"] = str(
        image_path
    )

    image["processed_path"] = (
        processed_path
    )

    image["width"] = int(
        bgr.shape[1]
    )

    image["height"] = int(
        bgr.shape[0]
    )

    image["fov_mask_path"] = (
        fov_mask_path
    )


def run_screening(
    image_path: str | Path,
    patient_id: str,
    eye: str,
    cfg: dict[str, Any] | None = None,
    operator_id: str = "",
    phc_id: str = "",
    output_dir: str | Path | None = "data/captures",
) -> dict[str, Any]:
    """
    Run the complete NETRA screening pipeline.

    The orchestrator is deliberately fail-soft:
    individual model failures do not crash the complete screening.

    RETAKE exits immediately after IQA.
    """

    total_start = time.perf_counter()

    cfg = cfg or load_config()

    thresholds = load_thresholds()

    result = new_result(
        patient_id,
        eye,
        operator_id=operator_id,
        phc_id=phc_id,
    )

    # --------------------------------------------------------
    # IMAGE LOAD
    # --------------------------------------------------------

    bgr = load_image(
        image_path
    )

    result["image"]["raw_path"] = str(
        image_path
    )
    result["image"]["width"] = int(
        bgr.shape[1]
    )
    result["image"]["height"] = int(
        bgr.shape[0]
    )

    # --------------------------------------------------------
    # IQA
    # --------------------------------------------------------

    iqa_start = time.perf_counter()

    iqa_mode = (
        cfg.get("modules", {})
        .get("iqa", "stub")
    )

    iqa_module = load_module(
        "iqa",
        iqa_mode,
    )

    quality = _safe_quality(
        iqa_module,
        bgr,
    )

    quality = _normalise_quality(
        quality
    )

    result["quality"] = quality

    result["timings_ms"]["iqa"] = int(
        (time.perf_counter() - iqa_start)
        * 1000
    )

    # --------------------------------------------------------
    # RETAKE EARLY EXIT
    # --------------------------------------------------------

    if quality["verdict"] == "RETAKE":

        result["routing"] = {
            "action": "ROUTINE",
            "reason": "RETAKE_REQUESTED",
            "alert_sent": False,
            "sync_status": "PENDING",
        }

        result["timings_ms"]["total"] = int(
            (time.perf_counter() - total_start)
            * 1000
        )

        return result

    # --------------------------------------------------------
    # FOV
    # --------------------------------------------------------

    fov_mask = _get_fov_mask(
        quality,
        bgr,
    )

    # --------------------------------------------------------
    # ENHANCEMENT
    # --------------------------------------------------------

    processed = _safe_enhance(
        bgr,
        quality,
    )

    # --------------------------------------------------------
    # RUNTIME ARTIFACTS
    # --------------------------------------------------------

    processed_path = ""
    lesion_path = ""

    try:
        processed_path, lesion_path = (
            _save_runtime_artifacts(
                result["screening_id"],
                bgr,
                quality,
                fov_mask,
                None,
                output_dir,
            )
        )

    except Exception:
        logger.exception(
            "Unable to save runtime artifacts."
        )

    fov_path = ""

    if fov_mask is not None and output_dir is not None:

        try:
            root = (
                Path(output_dir)
                / result["screening_id"]
            )

            root.mkdir(
                parents=True,
                exist_ok=True,
            )

            fov_path_obj = (
                root / "fov_mask.png"
            )

            cv2.imwrite(
                str(fov_path_obj),
                fov_mask,
            )

            fov_path = str(
                fov_path_obj
            )

        except Exception:
            logger.exception(
                "Unable to save FOV mask."
            )

    _update_image_block(
        result,
        image_path,
        bgr,
        processed_path,
        fov_path,
    )

    # --------------------------------------------------------
    # GRADING
    # --------------------------------------------------------

    grading_start = time.perf_counter()

    grading_mode = (
        cfg.get("modules", {})
        .get("grading", "stub")
    )

    try:
        grading_model = _load_model_object(
            "grading",
            grading_mode,
        )

        grading = _safe_grading(
            grading_model,
            bgr,
        )

    except Exception:
        logger.exception(
            "Unable to initialise grading."
        )

        grading_model = None
        grading = _empty_grading()

    result["grading"] = grading

    result["timings_ms"]["grading"] = int(
        (time.perf_counter() - grading_start)
        * 1000
    )

    # --------------------------------------------------------
    # SEGMENTATION
    # --------------------------------------------------------

    segmentation_start = time.perf_counter()

    segmentation_mode = (
        cfg.get("modules", {})
        .get("segmentation", "stub")
    )

    segmentation_model = None

    try:
        segmentation_model = _load_model_object(
            "segmentation",
            segmentation_mode,
        )

        lesions, lesion_mask = (
            _safe_segmentation(
                segmentation_model,
                bgr,
            )
        )

    except Exception:
        logger.exception(
            "Unable to initialise segmentation."
        )

        lesions, lesion_mask = (
            _safe_segmentation(
                object(),
                bgr,
            )
        )

    result["lesions"] = lesions

    result["timings_ms"]["segmentation"] = int(
        (time.perf_counter() - segmentation_start)
        * 1000
    )

    # Save combined lesion mask.
    if lesion_mask is not None and output_dir is not None:

        try:
            root = (
                Path(output_dir)
                / result["screening_id"]
            )

            root.mkdir(
                parents=True,
                exist_ok=True,
            )

            combined = _combined_lesion_mask(
                lesion_mask
            )

            if combined is not None:

                lesion_file = (
                    root
                    / "lesion_mask.png"
                )

                cv2.imwrite(
                    str(lesion_file),
                    combined * 255,
                )

                result["lesions"][
                    "mask_path"
                ] = str(lesion_file)

        except Exception:
            logger.exception(
                "Unable to save lesion mask."
            )

    # --------------------------------------------------------
    # XAI
    # --------------------------------------------------------

    xai_start = time.perf_counter()

    xai_mode = (
        cfg.get("modules", {})
        .get("xai", "stub")
    )

    xai = _safe_xai(
        processed,
        grading,
        lesion_mask,
        fov_mask,
        grading_model,
        xai_mode,
    )

    result["xai"] = xai

    result["timings_ms"]["xai"] = int(
        (time.perf_counter() - xai_start)
        * 1000
    )

    # --------------------------------------------------------
    # LONGITUDINAL
    # --------------------------------------------------------

    try:
        from db.dao import compare_with_prior

        result["longitudinal"] = (
            compare_with_prior(
                patient_id,
                result,
            )
        )

    except Exception:
        logger.exception(
            "Longitudinal comparison failed."
        )

        result["longitudinal"] = {
            "prior_screening_id": None,
            "prior_grade": None,
            "delta": None,
            "trend": "FIRST_VISIT",
        }

    # --------------------------------------------------------
    # ROUTING
    # --------------------------------------------------------

    try:
        from db.dao import compute_routing

        result["routing"] = (
            compute_routing(
                result,
                thresholds,
            )
        )

    except Exception:
        logger.exception(
            "Routing computation failed."
        )

        result["routing"] = {
            "action": "REVIEW",
            "reason": "ROUTING_ERROR",
            "alert_sent": False,
            "sync_status": "PENDING",
        }

    # --------------------------------------------------------
    # DATABASE
    # --------------------------------------------------------

    db_start = time.perf_counter()

    try:
        from db.dao import save_screening

        save_screening(
            result
        )

    except Exception:
        logger.exception(
            "Database persistence failed."
        )

    result["timings_ms"]["db"] = int(
        (time.perf_counter() - db_start)
        * 1000
    )

    # --------------------------------------------------------
    # TOTAL
    # --------------------------------------------------------

    result["timings_ms"]["total"] = int(
        (time.perf_counter() - total_start)
        * 1000
    )

    return result