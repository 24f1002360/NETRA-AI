"""
Contract tests for core/xai. Run with:  pytest tests/test_xai_contracts.py -v

Validates the `xai` block against core/schema/screening_result.json and
checks the three guard functions in isolation (GUIDE_4_Anshika.md Part 2).
"""
import json
import os
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.xai import guards
from core.stubs import xai_stub

FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")
SCHEMA = json.load(open(os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "core", "schema", "screening_result.json")))
XAI_SCHEMA = SCHEMA["properties"]["xai"]


def _load(name):
    img = cv2.imread(os.path.join(FIXTURES, name))
    assert img is not None, f"fixture missing: {name}"
    return img


def test_stub_matches_schema():
    import jsonschema
    result = xai_stub.explain(_load("good.png"))
    jsonschema.validate(result, XAI_SCHEMA)


def test_stub_guard_status_is_known_enum():
    result = xai_stub.explain(_load("good.png"))
    assert result["guard_status"] in ("OK", "CAM_OFF_RETINA", "LOW_AGREEMENT")


def _disc_mask(h, w, radius_frac=0.4):
    yy, xx = np.mgrid[0:h, 0:w]
    cy, cx = h / 2, w / 2
    r = radius_frac * min(h, w)
    return (((xx - cx) ** 2 + (yy - cy) ** 2) <= r ** 2).astype(np.uint8) * 255


def test_fov_guard_fires_when_cam_is_outside_retina():
    h, w = 100, 100
    fov = _disc_mask(h, w, radius_frac=0.3)
    cam = np.zeros((h, w), dtype=np.float32)
    cam[:10, :10] = 1.0  # corner, guaranteed outside the disc
    frac = guards.cam_outside_fov_fraction(cam, fov)
    assert frac > guards.FOV_OUTSIDE_THRESHOLD
    assert guards.guard_status(frac, None, 1.0) == "CAM_OFF_RETINA"


def test_fov_guard_passes_when_cam_is_inside_retina():
    h, w = 100, 100
    fov = _disc_mask(h, w, radius_frac=0.45)
    cam = np.zeros((h, w), dtype=np.float32)
    cam[45:55, 45:55] = 1.0
    frac = guards.cam_outside_fov_fraction(cam, fov)
    assert frac < guards.FOV_OUTSIDE_THRESHOLD


def test_lesion_agreement_perfect_overlap():
    h, w = 100, 100
    cam = np.zeros((h, w), dtype=np.float32)
    cam[40:60, 40:60] = 1.0
    lesion = np.zeros((h, w), dtype=np.uint8)
    lesion[40:60, 40:60] = 1
    assert guards.cam_lesion_agreement(cam, lesion) == pytest.approx(1.0, abs=0.01)


def test_lesion_agreement_no_overlap_routes_low_agreement():
    h, w = 100, 100
    cam = np.zeros((h, w), dtype=np.float32)
    cam[10:20, 10:20] = 1.0
    lesion = np.zeros((h, w), dtype=np.uint8)
    lesion[80:90, 80:90] = 1
    agreement = guards.cam_lesion_agreement(cam, lesion)
    assert agreement == pytest.approx(0.0, abs=0.01)
    assert guards.guard_status(0.0, agreement, 1.0) == "LOW_AGREEMENT"


def test_zero_outside_fov_is_a_hard_zero():
    h, w = 20, 20
    fov = np.zeros((h, w), dtype=np.uint8)
    fov[5:15, 5:15] = 255
    cam = np.ones((h, w), dtype=np.float32)
    masked = guards.zero_outside_fov(cam, fov)
    assert masked[0, 0] == 0.0
    assert masked[10, 10] == 1.0