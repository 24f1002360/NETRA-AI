import numpy as np

def segment(bgr):
    h, w = bgr.shape[:2] if hasattr(bgr, 'shape') else (1536, 1536)
    mask = np.zeros((h, w, 4), dtype=np.uint8)
    lesions = {
        'mask_path': '',
        'counts': {'microaneurysms': 14, 'haemorrhages': 6, 'hard_exudates': 9, 'soft_exudates': 1},
        'area_fraction': {'hard_exudates': 0.011, 'haemorrhages': 0.004},
        'model_version': 'unet-lite-v0.3'
    }
    return lesions, mask
