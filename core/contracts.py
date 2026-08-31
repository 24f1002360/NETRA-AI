from __future__ import annotations
from datetime import datetime, timezone
from uuid import uuid4
from typing import Any


def _now_iso() -> str:
    """Return the current timestamp in ISO-8601 format."""
    return datetime.now(timezone.utc).astimezone().isoformat()


def new_result(
    patient_id: str,
    eye: str,
    operator_id: str = "",
    phc_id: str = "",
) -> dict[str, Any]:
    """
    Create a new, contract-valid ScreeningResult skeleton.
    Every block is present from the beginning so downstream modules
    do not need to handle missing keys.
    """

    screening_id = str(uuid4())
    captured_at = _now_iso()

    return {
        "schema_version": "1.0",
        "screening_id": screening_id,
        "patient_id": patient_id,
        "captured_at": captured_at,
        "eye": eye,
        "operator_id": operator_id,
        "phc_id": phc_id,

        "image": {
            "raw_path": "",
            "processed_path": "",
            "width": 0,
            "height": 0,
            "fov_mask_path": "",
        },

        "quality": {
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
        },

        "grading": {
            "icdr_grade": 0,
            "grade_label": "",
            "probabilities": [0.0, 0.0, 0.0, 0.0, 0.0],
            "referable_dr": False,
            "confidence": 0.0,
            "uncertain": False,
            "model_id": "",
            "model_version": "",
        },

        "lesions": {
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
            "model_version": "",
        },

        "xai": {
            "gradcam_path": "",
            "overlay_path": "",
            "cam_lesion_agreement": 0.0,
            "cam_outside_fov_fraction": 0.0,
            "guard_status": "OK",
        },

        "other_conditions": {
            "glaucoma_suspect": {
                "cup_disc_ratio": 0.0,
                "flag": False,
            },
            "hypertensive_retinopathy": {
                "flag": False,
                "evidence": [],
            },
        },

        "longitudinal": {
            "prior_screening_id": None,
            "prior_grade": None,
            "delta": "",
            "trend": "FIRST_VISIT",
        },

        "routing": {
            "action": "ROUTINE",
            "reason": "",
            "alert_sent": False,
            "sync_status": "PENDING",
        },

        "timings_ms": {
            "iqa": 0,
            "grading": 0,
            "segmentation": 0,
            "xai": 0,
            "report": 0,
            "db": 0,
            "total": 0,
        },
    }