"""
Contract function owned by Anshika (Clinical XAI & Validation Lead):

    explain(bgr, model_handle, grading, lesion_mask, fov_mask) -> dict

Returns the `xai` block of the ScreeningResult contract
(core/schema/screening_result.json). Keys prefixed with `_` are private
diagnostics for eval/, pop them before schema validation -- same pattern
as `_mask` in core/iqa/quality.py.

model_handle is None whenever the real model/checkpoint isn't available
to the caller (e.g. checkpoint missing locally, or explain() called
directly without one), so this falls back to core.stubs.xai_stub and
the rest of the pipeline is never blocked.
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

    # fov_mask and lesion_mask are almost never the same shape as the CAM
    # (the model's fixed input size, e.g. 384x384). Resize both to the
    # CAM's shape before any guard math -- otherwise cam * fov_mask
    # raises a shape-mismatch error, or silently broadcasts wrong.
    # Nearest-neighbour to keep masks binary.
    fov_mask = _resize_mask_to(fov_mask, cam.shape)
    if lesion_mask is not None:
        lesion_mask = _resize_mask_to(lesion_mask, cam.shape)

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


def _resize_mask_to(mask, target_hw):
    """Resize a binary/float mask to (H, W) = target_hw[:2] with
    nearest-neighbour interpolation so it stays binary."""
    h, w = target_hw[0], target_hw[1]
    if mask.shape[:2] == (h, w):
        return mask
    return cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)


def _overlay(bgr, cam, alpha=0.45):
    heat = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heat = cv2.resize(heat, (bgr.shape[1], bgr.shape[0]))
    return cv2.addWeighted(heat, alpha, bgr, 1 - alpha, 0)