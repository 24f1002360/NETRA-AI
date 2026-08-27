import uuid
from datetime import datetime, timezone

def make_stub_result(patient_id='PHC12-0043', eye='OD', screening_id=None):
    if not screening_id:
        screening_id = str(uuid.uuid4())
    return {
        'schema_version': '1.0',
        'screening_id': screening_id,
        'patient_id': patient_id,
        'captured_at': datetime.now(timezone.utc).isoformat(),
        'eye': eye,
        'operator_id': 'OP-01',
        'phc_id': 'PHC-99',
        'image': {'raw_path': '', 'processed_path': '', 'width': 1536, 'height': 1536, 'fov_mask_path': ''},
        'quality': {'verdict': 'PASS', 'scores': {'blur': 0.85, 'illumination': 0.9, 'fov_coverage': 0.95, 'contrast': 0.8, 'artefact': 0.02}, 'reasons': [], 'operator_message_key': 'iqa.pass', 'enhancement_applied': []},
        'grading': {'icdr_grade': 2, 'grade_label': 'Moderate NPDR', 'probabilities': [0.03, 0.11, 0.68, 0.15, 0.03], 'referable_dr': True, 'confidence': 0.68, 'uncertain': False, 'model_id': 'effnetb0-dr', 'model_version': 'v0.4-int8'},
        'lesions': {'mask_path': '', 'counts': {'microaneurysms': 14, 'haemorrhages': 6, 'hard_exudates': 9, 'soft_exudates': 1}, 'area_fraction': {'hard_exudates': 0.011, 'haemorrhages': 0.004}, 'model_version': 'unet-lite-v0.3'},
        'xai': {'gradcam_path': '', 'overlay_path': '', 'cam_lesion_agreement': 0.71, 'cam_outside_fov_fraction': 0.03, 'guard_status': 'OK'},
        'other_conditions': {'glaucoma_suspect': {'cup_disc_ratio': 0.3, 'flag': False}, 'hypertensive_retinopathy': {'flag': False, 'evidence': []}},
        'longitudinal': {'prior_screening_id': None, 'prior_grade': None, 'delta': None, 'trend': 'FIRST_VISIT'},
        'routing': {'action': 'review', 'reason': 'referable_dr', 'alert_sent': False, 'sync_status': 'PENDING'},
        'timings_ms': {'iqa': 120, 'grading': 250, 'segmentation': 300, 'xai': 45, 'report': 10, 'db': 5, 'total': 730}
    }

def make_stub_retake_result(patient_id='PHC12-0043', eye='OD', screening_id=None):
    if not screening_id:
        screening_id = str(uuid.uuid4())
    res = make_stub_result(patient_id, eye, screening_id)
    res['quality']['verdict'] = 'RETAKE'
    res['quality']['reasons'] = ['BLUR_HIGH']
    res['quality']['operator_message_key'] = 'iqa.retake.blur'
    res['grading'] = {'icdr_grade': None, 'grade_label': '', 'probabilities': [0.0]*5, 'referable_dr': False, 'confidence': 0.0, 'uncertain': False, 'model_id': '', 'model_version': ''}
    res['lesions'] = {'mask_path': '', 'counts': {'microaneurysms': 0, 'haemorrhages': 0, 'hard_exudates': 0, 'soft_exudates': 0}, 'area_fraction': {'hard_exudates': 0.0, 'haemorrhages': 0.0}, 'model_version': ''}
    res['xai'] = {'gradcam_path': '', 'overlay_path': '', 'cam_lesion_agreement': 0.0, 'cam_outside_fov_fraction': 0.0, 'guard_status': ''}
    return res

def make_stub_history(patient_id='PHC12-0043'):
    return [
        make_stub_result(patient_id, 'OD'),
        make_stub_result(patient_id, 'OD'),
        make_stub_result(patient_id, 'OD')
    ]
