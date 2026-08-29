import numpy as np


def segment(bgr):
    height, width = bgr.shape[:2]

    mask = np.zeros((height, width, 4), dtype=np.uint8)

    lesions = {
        "mask_path": "",
        "counts": {
            "microaneurysms": 0,
            "haemorrhages": 0,
            "hard_exudates": 0,
            "soft_exudates": 0
        },
        "area_fraction": {
            "hard_exudates": 0.0,
            "haemorrhages": 0.0
        },
        "model_version": "stub-v0.1"
    }

    return lesions, mask
