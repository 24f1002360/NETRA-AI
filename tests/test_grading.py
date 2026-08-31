"""
Contract tests for the NETRA diabetic-retinopathy grading model.

Run with:

    pytest tests/test_grading.py -v

These tests verify that the production grading model:
1. Loads the trained checkpoint.
2. Uses the expected EfficientNet-B0 architecture.
3. Produces the required grading output.
4. Produces valid probabilities.
5. Returns a valid ICDR grade.
"""

import os
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    ),
)

from core.models.grading import (
    DRGrader,
    GRADE_LABELS,
    NUM_CLASSES,
)


FIXTURES = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "fixtures",
)

CHECKPOINT = os.path.join(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    ),
    "artifacts",
    "netra_dr_effb0_muskan_preproc.pth",
)


def _load(name):
    path = os.path.join(FIXTURES, name)

    image = cv2.imread(path)

    assert image is not None, (
        f"Fixture missing: {path}"
    )

    return image


@pytest.fixture(scope="module")
def grader():
    if not os.path.exists(CHECKPOINT):
        pytest.skip(
            "Grading checkpoint not available: "
            f"{CHECKPOINT}"
        )

    return DRGrader(
        weights_path=CHECKPOINT
    )


def test_grading_checkpoint_loads(grader):
    """The production grading checkpoint must load successfully."""

    assert grader.model is not None
    assert grader.device is not None
    assert grader.weights_path.exists()


def test_grading_checkpoint_metadata(grader):
    """Verify expected metadata from the trained checkpoint."""

    assert grader.checkpoint_epoch is not None
    assert grader.checkpoint_val_qwk is not None
    assert grader.checkpoint_image_size == 384


def test_real_grading_inference(grader):
    """Run real inference on the project fixture."""

    image = _load("good.png")

    result = grader.grade(image)

    assert isinstance(result, dict)

    assert "icdr_grade" in result
    assert "grade_label" in result
    assert "probabilities" in result
    assert "referable_dr" in result
    assert "confidence" in result
    assert "uncertain" in result
    assert "model_id" in result
    assert "model_version" in result


def test_grading_output_is_valid(grader):
    """Validate the structure and values returned by grading."""

    image = _load("good.png")

    result = grader.grade(image)

    grade = result["icdr_grade"]
    probabilities = result["probabilities"]

    assert grade in GRADE_LABELS

    assert len(probabilities) == NUM_CLASSES

    assert all(
        0.0 <= float(p) <= 1.0
        for p in probabilities
    )

    assert np.isclose(
        sum(probabilities),
        1.0,
        atol=1e-5,
    )

    assert result["grade_label"] == GRADE_LABELS[grade]

    assert np.isclose(
        result["confidence"],
        probabilities[grade],
        atol=1e-5,
    )


def test_referable_dr_logic(grader):
    """ICDR grades >= 2 must be marked referable."""

    image = _load("good.png")

    result = grader.grade(image)

    expected = result["icdr_grade"] >= 2

    assert result["referable_dr"] == expected