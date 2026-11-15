"""Evidence-gated six-domain rubric. No invented scores for unmeasured domains."""
from typing import Mapping
import math

WEIGHTS = {
    'notation_accuracy': 25,
    'intonation': 20,
    'rhythm': 20,
    'tempo': 15,
    'articulation': 10,
    'expression': 10,
}

def grade(metrics: Mapping, *, validated: bool = False):
    """Metrics must be normalized [0,100] and independently measured.

    `validated` means upstream has established score transcription, instrument
    compatibility, audio transcription confidence, and verified all six metrics.
    """
    details = {}
    for key, weight in WEIGHTS.items():
        item = metrics.get(key)
        value = item.get('value') if isinstance(item, dict) else None
        source = item.get('source') if isinstance(item, dict) else None
        reliable = bool(item.get('validated')) if isinstance(item, dict) else False
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 100:
            value = None
        details[key] = {'score_percent': value, 'weight': weight,
                        'weighted_points': round(value*weight/100,2) if value is not None else None,
                        'source': source, 'validated': reliable}
    missing = [k for k,v in details.items() if v['score_percent'] is None or not v['validated']]
    if not validated or missing:
        return {'status': 'incomplete', 'professional_score_100': None,
                'rubric': details, 'unverified_criteria': missing,
                'reason': 'Full professional grade requires independently validated measurements for every criterion'}
    return {'status':'complete', 'professional_score_100': round(sum(v['weighted_points'] for v in details.values()),1),
            'rubric': details, 'unverified_criteria': [],
            'warning':'Rubric score is not a substitute for an independent teacher assessment'}
