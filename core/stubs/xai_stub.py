"""Fake XAI implementation. Used until Kanchan's model + layer name land (Day 7).

Selected automatically by core/xai/explain.py when model_handle is None.
Everyone else (Divyanshu's routing, Abhishek's report) can build against
this from Day 1 -- keys and shapes match the real output exactly.
"""
import os

import cv2
import numpy as np


def explain(bgr, grading=None, lesion_mask=None, fov_mask=None, out_dir="artifacts/xai"):
    os.makedirs(out_dir, exist_ok=True)
    h, w = bgr.shape[:2] if bgr is not None else (512, 512)

    # Fake CAM: soft blob centred on the frame -- non-trivial enough that
    # downstream code (routing, PDF report) gets real-shaped data to build
    # against, not just zeros.
    yy, xx = np.mgrid[0:h, 0:w]
    cy, cx = h / 2, w / 2
    cam = np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * (min(h, w) / 4) ** 2)))
    cam = (cam / cam.max()).astype(np.float32)

    gradcam_path = os.path.join(out_dir, "stub_cam.png")
    overlay_path = os.path.join(out_dir, "stub_overlay.png")
    cv2.imwrite(gradcam_path, (cam * 255).astype(np.uint8))

    if bgr is not None:
        heat = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(heat, 0.45, bgr, 0.55, 0)
        cv2.imwrite(overlay_path, overlay)
    else:
        cv2.imwrite(overlay_path, (cam * 255).astype(np.uint8))

    return {
        "gradcam_path": gradcam_path,
        "overlay_path": overlay_path,
        "cam_lesion_agreement": 0.5,
        "cam_outside_fov_fraction": 0.05,
        "guard_status": "OK",
    }
    