"""
Contract function owned by Anshika (Clinical XAI & Validation Lead):

    explain(bgr, model_handle, grading, lesion_mask, fov_mask) -> dict

Returns the `xai` block of the ScreeningResult contract
(core/schema/screening_result.json). Keys prefixed with `_` are private
diagnostics for eval/, pop them before schema validation -- same pattern
as `_mask` in core/iqa/quality.py.

model_handle is None until Kanchan's Day-7 handoff, so this falls back to
core.stubs.xai_stub and the rest of the pipeline is never blocked.
"""
import os
import time
import uuid

import cv2
import numpy as np

from core.xai import guards
from core.stubs import xai_stub

try:
    from core.xai.gradcam import GradCAM
    _GRADCAM_READY = True
except ImportError:
    _GRADCAM_READY = False

ARTIFACT_DIR = "artifacts/xai"


def explain(bgr, model_handle, grading, lesion_mask, fov_mask,
            out_dir=ARTIFACT_DIR, screening_id=None):
    t0 = time.perf_counter()

    if model_handle is None or not _GRADCAM_READY:
        result = xai_stub.explain(bgr, grading, lesion_mask, fov_mask, out_dir)
        result["_timing_ms"] = (time.perf_counter() - t0) * 1000
        return result

    cam, class_idx = _run_gradcam(bgr, model_handle, grading)
    cam = guards.zero_outside_fov(cam, fov_mask)

    outside_frac = guards.cam_outside_fov_fraction(cam, fov_mask)
    agreement = (guards.cam_lesion_agreement(cam, lesion_mask)
                 if lesion_mask is not None else None)
    anatomical = guards.anatomical_plausibility(cam, fov_mask)
    status = guards.guard_status(outside_frac, agreement, anatomical)

    os.makedirs(out_dir, exist_ok=True)
    sid = screening_id or uuid.uuid4().hex[:8]
    gradcam_path = os.path.join(out_dir, f"{sid}_cam.png")
    overlay_path = os.path.join(out_dir, f"{sid}_overlay.png")
    cv2.imwrite(gradcam_path, (cam * 255).astype(np.uint8))
    cv2.imwrite(overlay_path, _overlay(bgr, cam))

    return {
        "gradcam_path": gradcam_path,
        "overlay_path": overlay_path,
        "cam_lesion_agreement": agreement,
        "cam_outside_fov_fraction": outside_frac,
        "guard_status": status,
        "_timing_ms": (time.perf_counter() - t0) * 1000,
        "_anatomical_plausibility": anatomical,
    }


def _run_gradcam(bgr, model_handle, grading):
    model = model_handle["model"]
    layer_name = model_handle["layer_name"]
    device = model_handle.get("device", "cpu")
    preprocess = model_handle["preprocess"]

    engine = GradCAM(model, layer_name,
                      variant=model_handle.get("variant", "gradcam"))
    input_tensor = preprocess(bgr).to(device)
    class_idx = grading.get("icdr_grade") if grading else None
    return engine(input_tensor, class_idx=class_idx)


def _overlay(bgr, cam, alpha=0.45):
    heat = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
    return cv2.addWeighted(heat, alpha, bgr, 1 - alpha, 0)