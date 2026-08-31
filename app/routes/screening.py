from __future__ import annotations

from pathlib import Path

from flask import Blueprint, jsonify, request

from core.inference import run_screening
from db.dao import get_history


screening_bp = Blueprint(
    "screening",
    __name__,
)


@screening_bp.post("/screenings")
def create_screening():

    data = request.get_json(
        silent=True
    ) or {}

    image_path = data.get(
        "image_path"
    )

    patient_id = data.get(
        "patient_id"
    )

    eye = data.get(
        "eye",
        "OD",
    )

    if not image_path or not patient_id:
        return jsonify(
            {
                "error": (
                    "image_path and "
                    "patient_id are required"
                )
            }
        ), 400

    if not Path(image_path).exists():
        return jsonify(
            {
                "error": "image not found"
            }
        ), 404

    result = run_screening(
        image_path=image_path,
        patient_id=patient_id,
        eye=eye,
        operator_id=data.get(
            "operator_id",
            "",
        ),
        phc_id=data.get(
            "phc_id",
            "",
        ),
    )

    return jsonify(result), 200


@screening_bp.get(
    "/patients/<patient_id>/history"
)
def patient_history(patient_id):

    try:
        limit = int(
            request.args.get(
                "limit",
                10,
            )
        )
    except ValueError:
        limit = 10

    return jsonify(
        get_history(
            patient_id,
            limit=limit,
        )
    )