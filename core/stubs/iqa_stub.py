def assess_quality(bgr):
    return {
        'verdict': 'PASS',
        'scores': {'blur': 0.85, 'illumination': 0.9, 'fov_coverage': 0.95, 'contrast': 0.8, 'artefact': 0.02},
        'reasons': [],
        'operator_message_key': 'iqa.pass',
        'enhancement_applied': []
    }

def enhance(bgr, quality):
    return bgr
