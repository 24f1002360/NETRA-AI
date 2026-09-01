from __future__ import annotations

"""NETRA-AI end-to-end screening orchestrator.

Pipeline:

    image
      -> IQA
      -> RETAKE or enhancement
      -> grading
      -> segmentation
      -> XAI / guards
      -> longitudinal comparison
      -> routing
      -> database

The orchestrator keeps the module contracts stable and allows each
specialised module to own its own model-specific preprocessing.
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
from core.iqa.enhance import enhance

PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP_CONFIG_PATH = PROJECT_ROOT / "configs" / "app.yaml"
THRESHOLDS_PATH = PROJECT_ROOT / "configs" / "thresholds.yaml"

logger = logging.getLogger(__name__)


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
        "interval_seconds": 30,
        "max_attempts": 5,
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

def _deep_copy(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _deep_copy(item)
            for key, item in value.items()
        }

    if isinstance(value, list):
        return [
            _deep_copy(item)
            for item in value
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


def load_config(
    path: str | Path = APP_CONFIG_PATH,
) -> dict[str, Any]:

    path = Path(path)

    if not path.is_absolute():
        path = PROJECT_ROOT / path

    if not path.exists():
        return _deep_copy(DEFAULT_CONFIG)

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        loaded = yaml.safe_load(f) or {}

    return _deep_merge(
        _deep_copy(DEFAULT_CONFIG),
        loaded,
    )


def load_thresholds(
    path: str | Path = THRESHOLDS_PATH,
) -> dict[str, Any]:

    path = Path(path)

    if not path.is_absolute():
        path = PROJECT_ROOT / path

    result = {
        "routing": dict(DEFAULT_ROUTING)
    }

    if not path.exists():
        return result

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        loaded = yaml.safe_load(f) or {}

    result["routing"].update(
        loaded.get("routing") or {}
    )

    return result


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

    return importlib.import_module(
        MODULE_PATHS[name][mode]
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

    if (
        name == "grading"
        and hasattr(module, "DRGrader")
    ):
        return module.DRGrader()

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
        image = image.astype(np.uint8)

    return image


# ============================================================
# EMPTY / FALLBACK RESULTS
# ============================================================

def _empty_quality() -> dict[str, Any]:

    return {
        "verdict": "RETAKE",
        "scores": {
            "blur": 0.0,
            "illumination": 0.0,
            "fov_coverage": 0.0,
            "contrast": 0.0,
            "artefact": 0.0,
            "centre_offset": 1.0,
        },
        "reasons": [
            "IQA_ERROR"
        ],
        "operator_message_key":
            "iqa.retake.iqa_error",
        "enhancement_applied": [],
    }


def _empty_grading() -> dict[str, Any]:

    return {
        "icdr_grade": None,
        "grade_label": None,
        "probabilities": [0.0] * 5,
        "referable_dr": False,
        "confidence": 0.0,
        "uncertain": True,
        "model_id": None,
        "model_version": None,
    }


def _empty_lesions() -> dict[str, Any]:

    return {
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
    }


def _empty_xai() -> dict[str, Any]:

    return {
        "gradcam_path": None,
        "overlay_path": None,
        "cam_lesion_agreement": None,
        "cam_outside_fov_fraction": None,
        "guard_status": "LOW_AGREEMENT",
    }


# ============================================================
# NORMALISATION
# ============================================================

def _normalise_quality(
    raw: Any,
) -> dict[str, Any]:

    if not isinstance(raw, dict):
        return _empty_quality()

    scores = raw.get("scores") or {}

    verdict = raw.get(
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
            "blur": float(
                scores.get("blur", 0.0)
            ),
            "illumination": float(
                scores.get("illumination", 0.0)
            ),
            "fov_coverage": float(
                scores.get("fov_coverage", 0.0)
            ),
            "contrast": float(
                scores.get("contrast", 0.0)
            ),
            "centre_offset": float(
                scores.get("centre_offset", 0.0)
            ),
        },
        "reasons": list(
            raw.get("reasons") or []
        ),
        "operator_message_key": str(
            raw.get(
                "operator_message_key",
                "",
            )
        ),
        "enhancement_applied": list(
            raw.get("enhancement_applied") or []
        ),
        **(
            {
                "processing_ms":
                    int(raw["processing_ms"])
            }
            if "processing_ms" in raw
            else {}
        ),
        **(
            {
                "work_size":
                    list(raw["work_size"])
            }
            if "work_size" in raw
            else {}
        ),
    }


def _normalise_lesions(
    raw: Any,
) -> dict[str, Any]:

    if not isinstance(raw, dict):
        return _empty_lesions()

    counts = raw.get("counts") or {}
    area = raw.get("area_fraction") or {}

    return {
        "mask_path":
            raw.get("mask_path", "") or "",

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
                area.get(
                    "hard_exudates",
                    0.0,
                )
            ),
            "haemorrhages": float(
                area.get(
                    "haemorrhages",
                    0.0,
                )
            ),
        },

        "model_version":
            raw.get("model_version"),
    }


def _normalise_mask(
    mask: Any,
) -> np.ndarray | None:

    if mask is None:
        return None

    arr = np.asarray(mask)

    if arr.ndim == 2:
        return (
            arr > 0
        ).astype(np.uint8)

    if arr.ndim == 3:

        if arr.shape[2] == 4:
            return (
                arr > 0
            ).astype(np.uint8)

        if arr.shape[0] == 4:
            return np.transpose(
                arr > 0,
                (1, 2, 0),
            ).astype(np.uint8)

    raise ValueError(
        f"Unsupported lesion mask shape: {arr.shape}"
    )


def _connected_components(
    mask: np.ndarray,
) -> int:

    m = (
        mask > 0
    ).astype(np.uint8)

    if not m.size:
        return 0

    n, _, _, _ = cv2.connectedComponentsWithStats(
        m,
        connectivity=8,
    )

    return max(
        0,
        int(n) - 1,
    )


def _segmentation_output(
    raw: Any,
) -> tuple[
    dict[str, Any],
    np.ndarray | None,
]:

    if (
        isinstance(raw, tuple)
        and len(raw) == 2
    ):
        return (
            _normalise_lesions(raw[0]),
            _normalise_mask(raw[1]),
        )

    if not isinstance(raw, dict):
        raise TypeError(
            "Unsupported segmentation output: "
            f"{type(raw).__name__}"
        )

    masks = raw.get("masks") or {}
    stats = raw.get("statistics") or {}

    names = {
        "MA": "microaneurysms",
        "HE": "haemorrhages",
        "EX": "hard_exudates",
        "SE": "soft_exudates",
    }

    counts = {}

    for short, name in names.items():

        mask = masks.get(short)

        counts[name] = (
            _connected_components(
                np.asarray(mask)
            )
            if mask is not None
            else 0
        )

    area = {
        "hard_exudates": 0.0,
        "haemorrhages": 0.0,
    }

    for short, name in (
        ("EX", "hard_exudates"),
        ("HE", "haemorrhages"),
    ):

        pct = float(
            (stats.get(short) or {}).get(
                "percentage",
                0.0,
            )
        )

        area[name] = float(
            np.clip(
                pct / 100.0,
                0.0,
                1.0,
            )
        )

    ordered = []

    for short in (
        "MA",
        "HE",
        "EX",
        "SE",
    ):

        if short in masks:

            arr = np.asarray(
                masks[short]
            )

            if arr.ndim == 2:
                ordered.append(
                    (arr > 0).astype(
                        np.uint8
                    )
                )

    stack = (
        np.stack(
            ordered,
            axis=-1,
        )
        if ordered
        else None
    )

    lesions = {
        "mask_path": "",
        "counts": counts,
        "area_fraction": area,
        "model_version":
            raw.get("model_version"),
    }

    return lesions, stack


def _combined_mask(
    mask: np.ndarray | None,
) -> np.ndarray | None:

    if mask is None:
        return None

    arr = np.asarray(mask)

    if arr.ndim == 2:
        return (
            arr > 0
        ).astype(np.uint8)

    if arr.ndim == 3:

        if arr.shape[2] == 4:
            return arr.any(
                axis=2
            ).astype(np.uint8)

        if arr.shape[0] == 4:
            return arr.any(
                axis=0
            ).astype(np.uint8)

    return None


def _resize_binary(
    mask: np.ndarray | None,
    shape_hw: tuple[int, int],
) -> np.ndarray | None:

    if mask is None:
        return None

    h, w = shape_hw

    arr = _combined_mask(mask)

    if arr is None:
        return None

    return cv2.resize(
        arr,
        (w, h),
        interpolation=cv2.INTER_NEAREST,
    ).astype(np.uint8)


# ============================================================
# SAFE MODULE CALLS
# ============================================================

def _safe_quality(
    module: Any,
    bgr: np.ndarray,
    fov_path: str | None,
) -> tuple[
    dict[str, Any],
    np.ndarray | None,
]:

    try:

        fn = getattr(
            module,
            "assess_quality",
        )

        kwargs = {}

        if fov_path is not None:
            kwargs["fov_mask_path"] = fov_path

        raw = fn(
            bgr,
            **kwargs,
        )

        if not isinstance(raw, dict):
            raise TypeError(
                "IQA output must be a dictionary"
            )

        mask = raw.get("_mask")

        return (
            _normalise_quality(raw),
            (
                np.asarray(mask)
                .astype(np.uint8)
                if mask is not None
                else None
            ),
        )

    except TypeError:

        try:

            raw = module.assess_quality(
                bgr
            )

            mask = (
                raw.get("_mask")
                if isinstance(raw, dict)
                else None
            )

            return (
                _normalise_quality(raw),
                (
                    np.asarray(mask)
                    .astype(np.uint8)
                    if mask is not None
                    else None
                ),
            )

        except Exception:

            logger.exception(
                "IQA stage failed"
            )

            return (
                _empty_quality(),
                None,
            )

    except Exception:

        logger.exception(
            "IQA stage failed"
        )

        return (
            _empty_quality(),
            None,
        )


def _safe_enhance(
    bgr: np.ndarray,
    quality: dict[str, Any],
) -> np.ndarray:

    try:
        return enhance(
            bgr,
            quality=quality,
        )

    except TypeError:
        return enhance(bgr)

    except Exception:

        logger.exception(
            "Enhancement failed; "
            "using original image"
        )

        return bgr


def _safe_grade(
    model: Any,
    bgr: np.ndarray,
) -> dict[str, Any]:

    try:

        result = model.grade(
            bgr
        )

        if not isinstance(result, dict):
            raise TypeError(
                "grading output must be a dictionary"
            )

        return result

    except Exception:

        logger.exception(
            "Grading stage failed"
        )

        return _empty_grading()


def _safe_segment(
    model: Any,
    bgr: np.ndarray,
) -> tuple[
    dict[str, Any],
    np.ndarray | None,
]:

    try:

        return _segmentation_output(
            model.segment(bgr)
        )

    except Exception:

        logger.exception(
            "Segmentation stage failed; "
            "continuing without lesions"
        )

        return (
            _empty_lesions(),
            None,
        )


# ============================================================
# XAI
# ============================================================

def _normalise_xai(
    raw: Any,
) -> dict[str, Any]:

    if not isinstance(raw, dict):
        raise TypeError(
            "XAI output must be a dictionary"
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

    def optional_float(
        value: Any,
    ) -> float | None:

        try:
            return (
                None
                if value is None
                else float(value)
            )

        except (
            TypeError,
            ValueError,
        ):
            return None

    return {
        "gradcam_path":
            raw.get("gradcam_path"),

        "overlay_path":
            raw.get("overlay_path"),

        "cam_lesion_agreement":
            optional_float(
                raw.get(
                    "cam_lesion_agreement"
                )
            ),

        "cam_outside_fov_fraction":
            optional_float(
                raw.get(
                    "cam_outside_fov_fraction"
                )
            ),

        "guard_status":
            status,
    }


def _safe_xai(
    raw_bgr: np.ndarray,
    display_bgr: np.ndarray,
    grading: dict[str, Any],
    lesion_mask: np.ndarray | None,
    fov_mask: np.ndarray | None,
    grading_model: Any,
    xai_mode: str,
    screening_id: str,
    output_dir: str | Path | None,
) -> dict[str, Any]:

    try:

        xai_module = _load_model_object(
            "xai",
            xai_mode,
        )

        fn = getattr(
            xai_module,
            "explain",
            None,
        )

        if fn is None:
            raise AttributeError(
                "XAI module must expose explain()"
            )

        # ----------------------------------------------------
        # Stub XAI
        # ----------------------------------------------------

        if xai_mode != "real":

            raw = fn(
                display_bgr,
                grading=grading,
                lesion_mask=lesion_mask,
                fov_mask=fov_mask,
            )

            return _normalise_xai(raw)

        # ----------------------------------------------------
        # Real XAI
        # ----------------------------------------------------

        if (
            grading_model is None
            or not hasattr(
                grading_model,
                "model",
            )
        ):
            raise RuntimeError(
                "Real XAI requires a loaded "
                "grading model"
            )

        size = int(
            getattr(
                grading_model,
                "checkpoint_image_size",
                384,
            )
        )

        # XAI visualization is resized to the
        # model's CAM working resolution.
        xai_image = cv2.resize(
            display_bgr,
            (size, size),
            interpolation=cv2.INTER_AREA,
        )

        xai_fov = _resize_binary(
            fov_mask,
            (size, size),
        )

        xai_lesions = _resize_binary(
            lesion_mask,
            (size, size),
        )

        # IMPORTANT:
        #
        # Do NOT independently perform enhancement here.
        #
        # Kanchan's DRGrader.preprocess() is the
        # source of truth for grading-time preprocessing.
        #
        # XAI receives a callable that invokes that
        # exact preprocessing on the original raw image.
        def exact_grading_preprocess(
            _unused_image,
        ):

            return grading_model.preprocess(
                raw_bgr
            )

        handle = {
            "model":
                grading_model.model,

            "layer_name":
                getattr(
                    grading_model,
                    "gradcam_layer",
                    "features.8",
                ),

            "device":
                grading_model.device,

            "preprocess":
                exact_grading_preprocess,

            "variant":
                "gradcam",
        }

        kwargs = {
            "out_dir": str(
                output_dir
                or (
                    PROJECT_ROOT
                    / "artifacts"
                    / "xai"
                )
            ),
            "screening_id":
                screening_id,
        }

        try:

            raw = fn(
                xai_image,
                handle,
                grading,
                xai_lesions,
                xai_fov,
                **kwargs,
            )

        except TypeError:

            raw = fn(
                xai_image,
                handle,
                grading,
                xai_lesions,
                xai_fov,
            )

        return _normalise_xai(
            raw
        )

    except Exception:

        logger.exception(
            "XAI stage failed; "
            "continuing without XAI"
        )

        return _empty_xai()


# ============================================================
# ARTIFACTS / DATABASE
# ============================================================

def _set_image_paths(
    result: dict[str, Any],
    image_path: str | Path,
    bgr: np.ndarray,
    processed_path: str,
    fov_path: str,
) -> None:

    result["image"].update(
        {
            "raw_path":
                str(image_path),

            "processed_path":
                processed_path,

            "width":
                int(bgr.shape[1]),

            "height":
                int(bgr.shape[0]),

            "fov_mask_path":
                fov_path,
        }
    )


def _save_artifacts(
    screening_id: str,
    raw_bgr: np.ndarray,
    processed: np.ndarray,
    fov_mask: np.ndarray | None,
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

    processed_path = (
        root / "processed.png"
    )

    cv2.imwrite(
        str(processed_path),
        processed,
    )

    fov_path = ""

    if fov_mask is not None:

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

    return (
        str(processed_path),
        fov_path,
    )


def _persist(
    result: dict[str, Any],
    cfg: dict[str, Any],
) -> None:

    try:

        from db.dao import save_screening

        db_path = (
            cfg.get("database") or {}
        ).get("path")

        if db_path:
            save_screening(
                result,
                db_path=db_path,
            )
        else:
            save_screening(result)

    except Exception:

        logger.exception(
            "Database persistence failed"
        )


# ============================================================
# LONGITUDINAL / ROUTING
# ============================================================

def _longitudinal(
    result: dict[str, Any],
    cfg: dict[str, Any],
) -> dict[str, Any]:

    try:

        from db.dao import compare_with_prior

        db_path = (
            cfg.get("database") or {}
        ).get("path")

        if db_path:
            return compare_with_prior(
                result["patient_id"],
                result,
                db_path=db_path,
            )

        return compare_with_prior(
            result["patient_id"],
            result,
        )

    except Exception:

        logger.exception(
            "Longitudinal comparison failed"
        )

        return {
            "prior_screening_id": None,
            "prior_grade": None,
            "delta": None,
            "trend": "FIRST_VISIT",
        }


def _routing(
    result: dict[str, Any],
    cfg: dict[str, Any],
    thresholds: dict[str, Any],
) -> dict[str, Any]:

    try:

        from db.dao import compute_routing

        return compute_routing(
            result,
            thresholds,
        )

    except Exception:

        logger.exception(
            "DB routing computation failed; "
            "using fallback"
        )

        try:

            grading = (
                result.get("grading")
                or {}
            )

            xai = (
                result.get("xai")
                or {}
            )

            conditions = (
                result.get(
                    "other_conditions"
                )
                or {}
            )

            rcfg = (
                thresholds.get("routing")
                or {}
            )

            grade = grading.get(
                "icdr_grade"
            )

            confidence = float(
                grading.get(
                    "confidence",
                    0.0,
                )
                or 0.0
            )

            urgent_grade = int(
                rcfg.get(
                    "urgent_grade",
                    4,
                )
            )

            urgent_g3 = float(
                rcfg.get(
                    "urgent_grade_3_confidence",
                    0.70,
                )
            )

            referable_grade = int(
                rcfg.get(
                    "referable_grade",
                    2,
                )
            )

            low_conf = float(
                rcfg.get(
                    "low_confidence",
                    0.55,
                )
            )

            if (
                grade is not None
                and int(grade)
                >= urgent_grade
            ):

                action = "URGENT_REFERRAL"
                reason = "URGENT_GRADE"

            elif (
                grade is not None
                and int(grade) == 3
                and confidence >= urgent_g3
            ):

                action = "URGENT_REFERRAL"
                reason = (
                    "GRADE_3_HIGH_CONFIDENCE"
                )

            elif (
                grade is not None
                and int(grade)
                >= referable_grade
            ):

                action = "REVIEW"
                reason = "REFERABLE_DR"

            elif confidence < low_conf:

                action = "REVIEW"
                reason = "LOW_CONFIDENCE"

            elif (
                xai.get(
                    "guard_status"
                )
                or "OK"
            ) != "OK":

                action = "REVIEW"
                reason = "XAI_GUARD"

            elif bool(
                (
                    conditions.get(
                        "glaucoma_suspect"
                    )
                    or {}
                ).get(
                    "flag",
                    False,
                )
            ):

                action = "REVIEW"
                reason = (
                    "GLAUCOMA_SUSPECT"
                )

            else:

                action = "ROUTINE"
                reason = (
                    "NO_REFERRAL_CRITERIA"
                )

            return {
                "action": action,
                "reason": reason,
                "alert_sent": False,
                "sync_status": "PENDING",
            }

        except Exception:

            logger.exception(
                "Fallback routing computation failed"
            )

            return {
                "action": "REVIEW",
                "reason": "ROUTING_ERROR",
                "alert_sent": False,
                "sync_status": "PENDING",
            }


# ============================================================
# MAIN PIPELINE
# ============================================================

def run_screening(
    image_path: str | Path,
    patient_id: str,
    eye: str,
    cfg: dict[str, Any] | None = None,
    operator_id: str = "",
    phc_id: str = "",
    output_dir: str | Path | None = "data/captures",
) -> dict[str, Any]:

    total_start = time.perf_counter()

    cfg = (
        cfg
        if cfg is not None
        else load_config()
    )

    thresholds = load_thresholds()

    result = new_result(
        patient_id,
        eye,
        operator_id=operator_id,
        phc_id=phc_id,
    )

    bgr = load_image(
        image_path
    )

    result["image"]["raw_path"] = (
        str(image_path)
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
        cfg.get("modules") or {}
    ).get(
        "iqa",
        "stub",
    )

    iqa_module = load_module(
        "iqa",
        iqa_mode,
    )

    fov_path_hint = None

    if output_dir is not None:

        fov_path_hint = str(
            Path(output_dir)
            / result["screening_id"]
            / "fov_mask.png"
        )

    quality, fov_mask = _safe_quality(
        iqa_module,
        bgr,
        fov_path_hint,
    )

    result["quality"] = quality

    result["timings_ms"]["iqa"] = int(
        (
            time.perf_counter()
            - iqa_start
        )
        * 1000
    )

    # --------------------------------------------------------
    # RETAKE
    # --------------------------------------------------------

    if quality["verdict"] == "RETAKE":

        result["routing"] = {
            "action": "ROUTINE",
            "reason":
                "RETAKE_REQUESTED",
            "alert_sent": False,
            "sync_status": "PENDING",
        }

        result["timings_ms"]["total"] = int(
            (
                time.perf_counter()
                - total_start
            )
            * 1000
        )

        return result

    # --------------------------------------------------------
    # DISPLAY ENHANCEMENT
    #
    # This is for artifacts/UI.
    # It is NOT used as a second grading preprocessing path.
    # --------------------------------------------------------

    processed = _safe_enhance(
        bgr,
        quality,
    )

    processed_path, saved_fov_path = (
        _save_artifacts(
            result["screening_id"],
            bgr,
            processed,
            fov_mask,
            output_dir,
        )
    )

    _set_image_paths(
        result,
        image_path,
        bgr,
        processed_path,
        saved_fov_path,
    )

    # --------------------------------------------------------
    # GRADING
    # --------------------------------------------------------

    grading_start = time.perf_counter()

    grading_mode = (
        cfg.get("modules") or {}
    ).get(
        "grading",
        "stub",
    )

    grading_model = None

    try:

        grading_model = _load_model_object(
            "grading",
            grading_mode,
        )

        result["grading"] = _safe_grade(
            grading_model,
            bgr,
        )

    except Exception:

        logger.exception(
            "Unable to initialise grading model"
        )

        result["grading"] = (
            _empty_grading()
        )

    result["timings_ms"]["grading"] = int(
        (
            time.perf_counter()
            - grading_start
        )
        * 1000
    )

    # --------------------------------------------------------
    # SEGMENTATION
    # --------------------------------------------------------

    segmentation_start = (
        time.perf_counter()
    )

    segmentation_mode = (
        cfg.get("modules") or {}
    ).get(
        "segmentation",
        "stub",
    )

    segmentation_model = None

    try:

        segmentation_model = (
            _load_model_object(
                "segmentation",
                segmentation_mode,
            )
        )

        lesions, lesion_mask = (
            _safe_segment(
                segmentation_model,
                bgr,
            )
        )

    except Exception:

        logger.exception(
            "Unable to initialise segmentation model"
        )

        lesions = _empty_lesions()
        lesion_mask = None

    result["lesions"] = lesions

    result["timings_ms"][
        "segmentation"
    ] = int(
        (
            time.perf_counter()
            - segmentation_start
        )
        * 1000
    )

    # Save combined lesion mask.

    if (
        lesion_mask is not None
        and output_dir is not None
    ):

        root = (
            Path(output_dir)
            / result["screening_id"]
        )

        root.mkdir(
            parents=True,
            exist_ok=True,
        )

        combined = _combined_mask(
            lesion_mask
        )

        if combined is not None:

            lesion_path = (
                root
                / "lesion_mask.png"
            )

            cv2.imwrite(
                str(lesion_path),
                combined * 255,
            )

            result["lesions"][
                "mask_path"
            ] = str(lesion_path)

    # --------------------------------------------------------
    # XAI
    # --------------------------------------------------------

    xai_start = time.perf_counter()

    xai_mode = (
        cfg.get("modules") or {}
    ).get(
        "xai",
        "stub",
    )

    result["xai"] = _safe_xai(
        raw_bgr=bgr,
        display_bgr=processed,
        grading=result["grading"],
        lesion_mask=lesion_mask,
        fov_mask=fov_mask,
        grading_model=(
            grading_model
            if grading_mode == "real"
            else None
        ),
        xai_mode=xai_mode,
        screening_id=result[
            "screening_id"
        ],
        output_dir=output_dir,
    )

    result["timings_ms"]["xai"] = int(
        (
            time.perf_counter()
            - xai_start
        )
        * 1000
    )

    # --------------------------------------------------------
    # OPTIONAL CDR
    # --------------------------------------------------------

    try:

        iqa_real = (
            iqa_module
            if iqa_mode == "real"
            else None
        )

        cdr_fn = (
            getattr(
                iqa_real,
                "cup_disc_ratio",
                None,
            )
            if iqa_real
            else None
        )

        if (
            callable(cdr_fn)
            and fov_mask is not None
        ):

            cdr = float(
                cdr_fn(
                    processed,
                    fov_mask,
                )
            )

            result[
                "other_conditions"
            ][
                "glaucoma_suspect"
            ] = {
                "cup_disc_ratio": cdr,
                "flag": cdr >= 0.60,
            }

    except Exception:

        logger.exception(
            "CDR calculation failed"
        )

    # --------------------------------------------------------
    # LONGITUDINAL
    # --------------------------------------------------------

    result["longitudinal"] = (
        _longitudinal(
            result,
            cfg,
        )
    )

    # --------------------------------------------------------
    # ROUTING
    # --------------------------------------------------------

    result["routing"] = _routing(
        result,
        cfg,
        thresholds,
    )

    # --------------------------------------------------------
    # DATABASE
    # --------------------------------------------------------

    db_start = time.perf_counter()

    _persist(
        result,
        cfg,
    )

    result["timings_ms"]["db"] = int(
        (
            time.perf_counter()
            - db_start
        )
        * 1000
    )

    # --------------------------------------------------------
    # TOTAL
    # --------------------------------------------------------

    result["timings_ms"]["total"] = int(
        (
            time.perf_counter()
            - total_start
        )
        * 1000
    )

    return result