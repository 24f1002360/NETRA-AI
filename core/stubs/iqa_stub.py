"""Fake IQA implementation. Used until the real module lands.

Selected via configs/app.yaml -> modules.iqa: stub | real
"""


def assess_quality(bgr, **kwargs):
    return {
        "verdict": "PASS",
        "scores": {"blur": 0.85, "illumination": 0.90,
                   "fov_coverage": 0.95, "contrast": 0.80,
                   "centre_offset": 0.05},
        "reasons": [],
        "operator_message_key": "iqa.pass",
        "enhancement_applied": [],
        "processing_ms": 120,
    }


def enhance(bgr, quality=None, **kwargs):
    return bgr
