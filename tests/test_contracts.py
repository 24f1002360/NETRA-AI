"""
Shared contract tests. Run on every PR.

    pytest tests/ -v

These are the tests everyone's module has to keep passing. They validate
against core/schema/screening_result.json, which is the single source of
truth for the ScreeningResult shape.
"""

import json
import os
import sys
import time

import cv2
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.iqa.quality import assess_quality
from core.iqa.enhance import enhance
from core.iqa.fov import extract_fov
from core.stubs import iqa_stub

FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")
SCHEMA = json.load(open(os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "core", "schema", "screening_result.json")))
QUALITY_SCHEMA = SCHEMA["properties"]["quality"]

IQA_BUDGET_MS = 1500


def _load(name):
    p = os.path.join(FIXTURES, name)
    img = cv2.imread(p)
    assert img is not None, f"fixture missing: {p} (run tools/make_fixtures.py)"
    return img


@pytest.mark.parametrize("name", ["good.png", "blurry.png", "severe.png"])
def test_fixtures_exist(name):
    _load(name)


@pytest.mark.parametrize("name", ["good.png", "blurry.png", "severe.png"])
def test_quality_block_matches_schema(name):
    import jsonschema
    q = assess_quality(_load(name))
    q.pop("_mask")                      # private, never serialised
    jsonschema.validate(q, QUALITY_SCHEMA)


def test_stub_matches_schema():
    import jsonschema
    jsonschema.validate(iqa_stub.assess_quality(None), QUALITY_SCHEMA)


def test_stub_and_real_have_same_keys():
    """The stub is what unblocks everyone else, so it must not drift."""
    real = assess_quality(_load("good.png")); real.pop("_mask")
    stub = iqa_stub.assess_quality(None)
    assert set(stub) <= set(real), f"stub has keys real does not: {set(stub) - set(real)}"
    assert set(stub["scores"]) == set(real["scores"])


def test_blurry_is_rejected():
    q = assess_quality(_load("blurry.png"))
    assert q["verdict"] == "RETAKE"
    assert "BLUR_HIGH" in q["reasons"]


def test_good_is_not_rejected():
    q = assess_quality(_load("good.png"))
    assert q["verdict"] in ("PASS", "AUTO_CORRECTED")


@pytest.mark.parametrize("name", ["good.png", "blurry.png", "severe.png"])
def test_iqa_within_latency_budget(name):
    img = _load(name)
    t = time.perf_counter()
    assess_quality(img)
    ms = (time.perf_counter() - t) * 1000
    assert ms < IQA_BUDGET_MS, f"{name}: {ms:.0f} ms exceeds {IQA_BUDGET_MS} ms"


def test_large_image_is_downscaled():
    """Raw fundus captures are ~4288x2848. Without the working-resolution
    cap this takes ~28 s and blows the budget."""
    big = cv2.resize(_load("good.png"), (4288, 2848))
    t = time.perf_counter()
    q = assess_quality(big)
    ms = (time.perf_counter() - t) * 1000
    assert ms < IQA_BUDGET_MS, f"{ms:.0f} ms on a full-size image"
    assert max(q["work_size"]) <= 1024


def test_enhance_is_shape_stable():
    img = _load("good.png")
    out = enhance(img)
    assert out.dtype == img.dtype
    assert out.ndim == 3 and out.shape[2] == 3


def test_fov_mask_matches_enhanced_size():
    """Anshika's CAM guard needs the mask and the image to line up."""
    img = _load("good.png")
    q = assess_quality(img)
    out = enhance(img)
    assert q["_mask"].shape[:2] == out.shape[:2]


def test_reason_codes_are_known():
    allowed = set(QUALITY_SCHEMA["properties"]["reasons"]["items"]["enum"])
    for name in ("good.png", "blurry.png", "severe.png"):
        q = assess_quality(_load(name))
        assert set(q["reasons"]) <= allowed, f"{name}: unknown code {set(q['reasons']) - allowed}"


def test_compression_ladder():
    from core.net.compression import compress_for_sync
    _, meta = compress_for_sync(os.path.join(FIXTURES, "severe.png"), 256,
                                out_dir="/tmp")
    assert meta["bytes"] > 0
    assert meta["est_seconds"] > 0
    os.remove(_)