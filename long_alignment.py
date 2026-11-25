"""Memory-bounded exact alignment for long monophonic note sequences.

Uses Hirschberg's algorithm (linear memory). Never report a full score for an
alignment exceeding the configured work budget. Do not conflate an estimated
pitch similarity with a teacher-certified performance grade.
"""
from __future__ import annotations

import os
from fastapi import HTTPException

_MIDI_MIN, _MIDI_MAX = 21, 108


def _row(a, b):
    # Scores for transforming a into all prefixes of b; O(len(b)) memory.
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            substitution = previous[j - 1] + (ca != cb)
            current[j] = min(previous[j] + 1, current[j - 1] + 1, substitution)
        previous = current
    return previous


def _small_align(a, b, offset_a, offset_b):
    n, m = len(a), len(b)
    rows = [list(range(m + 1))]
    for i in range(1, n + 1):
        prev = rows[-1]
        cur = [i] + [0] * m
        for j in range(1, m + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1,
                         prev[j - 1] + (a[i - 1] != b[j - 1]))
        rows.append(cur)
    i, j = n, m
    result = []
    while i or j:
        if i and j and rows[i][j] == rows[i - 1][j - 1] + (a[i - 1] != b[j - 1]):
            result.append(('match' if a[i - 1] == b[j - 1] else 'replace',
                           offset_a + i - 1, offset_b + j - 1))
            i -= 1; j -= 1
        elif i and rows[i][j] == rows[i - 1][j] + 1:
            result.append(('delete', offset_a + i - 1, None))
            i -= 1
        else:
            result.append(('insert', None, offset_b + j - 1))
            j -= 1
    result.reverse()
    return result


def _align(a, b, ia=0, ib=0):
    # Trimming equal prefixes/suffixes reduces complexity enormously for
    # common, well-performed passages.
    left = 0
    while left < len(a) and left < len(b) and a[left] == b[left]:
        left += 1
    prefix = [('match', ia + k, ib + k) for k in range(left)]
    a = a[left:]; b = b[left:]; ia += left; ib += left
    right = 0
    while right < len(a) and right < len(b) and a[-1 - right] == b[-1 - right]:
        right += 1
    suffix = [('match', ia + len(a) - right + k, ib + len(b) - right + k)
              for k in range(right)]
    if right:
        a = a[:-right]; b = b[:-right]
    if not a:
        middle = [('insert', None, ib + j) for j in range(len(b))]
    elif not b:
        middle = [('delete', ia + i, None) for i in range(len(a))]
    elif min(len(a), len(b)) <= 1 or len(a) * len(b) <= 2500:
        middle = _small_align(a, b, ia, ib)
    else:
        middle_index = len(a) // 2
        forward = _row(a[:middle_index], b)
        backward = _row(a[middle_index:][::-1], b[::-1])
        split = min(range(len(b) + 1), key=lambda j: forward[j] + backward[len(b) - j])
        middle = (_align(a[:middle_index], b[:split], ia, ib) +
                  _align(a[middle_index:], b[split:], ia + middle_index, ib + split))
    return prefix + middle + suffix


def compare_complete(expected, actual):
    if not isinstance(expected, list) or not isinstance(actual, list):
        raise HTTPException(422, 'MIDI note sequences must be lists')
    if not expected or not actual:
        raise HTTPException(422, 'Neither score nor performance can be empty')
    if not all(type(x) is int and _MIDI_MIN <= x <= _MIDI_MAX
               for x in expected + actual):
        raise HTTPException(422, 'Invalid MIDI note')
    max_notes = int(os.getenv('NOTA_MAX_ALIGNMENT_NOTES', '12000'))
    if len(expected) > max_notes or len(actual) > max_notes:
        raise HTTPException(413, 'Note count exceeds safe analysis limit')
    # Long exact alignments may be too expensive for an HTTP request. Reject
    # rather than inventing a similarity estimate from a partial passage.
    leading = 0
    while leading < min(len(expected), len(actual)) and expected[leading] == actual[leading]:
        leading += 1
    trailing = 0
    while (trailing < min(len(expected), len(actual)) - leading and
           expected[-trailing-1] == actual[-trailing-1]):
        trailing += 1
    work_cells = (len(expected) - leading - trailing) * (len(actual) - leading - trailing)
    max_work = int(os.getenv('NOTA_MAX_ALIGNMENT_CELLS', '24000000'))
    if work_cells > max_work:
        raise HTTPException(422, 'Long alignment needs background batch processing')
    ops = _align(expected, actual)
    errors = []
    counts = {'wrong_pitch': 0, 'missing_note': 0, 'extra_note': 0}
    correct = 0
    for tag, i, j in ops:
        if tag == 'match':
            correct += 1
        elif tag == 'replace':
            counts['wrong_pitch'] += 1
            if len(errors) < 100:
                errors.append({'type': 'wrong_pitch', 'reference_position': i + 1,
                               'expected_midi': expected[i], 'performed_midi': actual[j],
                               'semitone_difference': actual[j] - expected[i]})
        elif tag == 'delete':
            counts['missing_note'] += 1
            if len(errors) < 100:
                errors.append({'type': 'missing_note', 'reference_position': i + 1,
                               'expected_midi': expected[i]})
        else:
            counts['extra_note'] += 1
            if len(errors) < 100:
                errors.append({'type': 'extra_note', 'performed_position': j + 1,
                               'performed_midi': actual[j]})
    edits = sum(counts.values())
    n, m = len(expected), len(actual)
    return {
        'reference_count': n,
        'performed_count': m,
        'performed_midi': actual,
        'note_similarity_percent': max(0, round(100 * (1 - edits / max(n, m)))),
        'edit_distance': edits,
        'detailed_feedback': {
            'correct_notes': correct, 'error_counts': counts, 'errors': errors,
            'reported_error_sample_count': len(errors),
            'total_error_count': edits, 'errors_sampled': edits > len(errors),
            'assessment_scope': 'Entire detected pitch-event sequence; monophonic estimate',
            'rhythm': {'status': 'not_measured'},
            'tempo': {'status': 'not_measured'},
            'intonation_cents': {'status': 'not_measured'},
            'feedback': 'Review pitch differences; full professional grading needs rhythm/tempo evidence.',
        },
        'rhythm_scored': False,
        'all_detected_events_considered': True,
        'warning': 'Detected pitch comparison only; not a certified performance assessment.',
    }
