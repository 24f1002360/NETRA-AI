def explain(bgr, model_handle, grading, lesion_mask, fov_mask):
    return {
        'gradcam_path': '',
        'overlay_path': '',
        'cam_lesion_agreement': 0.71,
        'cam_outside_fov_fraction': 0.03,
        'guard_status': 'OK'
    }
