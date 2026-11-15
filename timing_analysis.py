"""Evidence-based note/time alignment and partial performance scoring.

This module intentionally refuses to grade criteria without measured data.
"""
from dataclasses import dataclass
from typing import Optional
import math
import statistics

@dataclass(frozen=True)
class Event:
    midi: int
    start: float
    duration: float

def validate(events):
    if not 1 <= len(events) <= 500:
        raise ValueError("Expected 1..500 note events")
    for e in events:
        if not (21 <= e.midi <= 108 and math.isfinite(e.start)
                and math.isfinite(e.duration) and e.start >= 0 and e.duration > 0):
            raise ValueError("Invalid pitch or note timing")
    if any(b.start < a.start for a, b in zip(events, events[1:])):
        raise ValueError("Events must be ordered by onset")
    return events

def align(reference, performed):
    """Needleman-Wunsch: note matches favored; timestamps remain evidence only."""
    validate(reference)
    validate(performed)
    n, m = len(reference), len(performed)
    dp = [[0]*(m+1) for _ in range(n+1)]
    for i in range(n+1): dp[i][0] = i
    for j in range(m+1): dp[0][j] = j
    for i in range(1, n+1):
        for j in range(1, m+1):
            cost = min(1, abs(reference[i-1].midi-performed[j-1].midi))
            dp[i][j] = min(dp[i-1][j]+1, dp[i][j-1]+1, dp[i-1][j-1]+cost)
    i,j=n,m
    pairs=[]
    errors=[]
    while i or j:
        if i and j and dp[i][j]==dp[i-1][j-1]+min(1,abs(reference[i-1].midi-performed[j-1].midi)):
            a,b=reference[i-1],performed[j-1]
            pairs.append((i-1,j-1))
            if a.midi!=b.midi:
                errors.append({"kind":"wrong_pitch","reference_note":i,
                               "audio_time_seconds":round(b.start,3),
                               "expected_midi":a.midi,"actual_midi":b.midi,
                               "semitones":b.midi-a.midi})
            i-=1;j-=1
        elif i and dp[i][j]==dp[i-1][j]+1:
            errors.append({"kind":"missing_note","reference_note":i,
                           "expected_midi":reference[i-1].midi})
            i-=1
        else:
            b=performed[j-1]
            errors.append({"kind":"extra_note","audio_time_seconds":round(b.start,3),
                           "actual_midi":b.midi})
            j-=1
    pairs.reverse()
    errors.reverse()
    return pairs,errors,dp[n][m]

def analyze(reference,performed,beat_seconds=None,time_signature_numerator=None):
    """Reference and performance times must be actual measured/verified seconds."""
    pairs,errors,edit=align(reference,performed)
    pitch_percent=round(100*(1-edit/max(len(reference),len(performed))),1)
    result={"status":"partial","pitch_similarity_percent":max(0,pitch_percent),
            "note_errors":errors,"note_matches":len(pairs),
            "rhythm": {"status":"not_measured"},
            "tempo": {"status":"not_measured"},
            "overall_score_100":None,
            "intonation":{"status":"not_measured"},
            "articulation":{"status":"not_measured"},
            "dynamics":{"status":"not_measured"},
            "expression":{"status":"not_measured"}}
    # Compare relative intervals, robust to the recording starting later than score.
    if len(pairs)>=3:
        ref_intervals=[]
        play_intervals=[]
        for (ia,ja),(ib,jb) in zip(pairs,pairs[1:]):
            if ib==ia+1 and jb==ja+1:
                ref_delta=reference[ib].start-reference[ia].start
                play_delta=performed[jb].start-performed[ja].start
                if ref_delta>0.02 and play_delta>0.02:
                    ref_intervals.append(ref_delta)
                    play_intervals.append(play_delta)
        if len(ref_intervals)>=2:
            ratios=[b/a for a,b in zip(ref_intervals,play_intervals)]
            scale=statistics.median(ratios)
            deviations=[abs(b/(a*scale)-1) for a,b in zip(ref_intervals,play_intervals)]
            # Rhythm score measures normalized onset interval consistency, not expression.
            accuracy=max(0,round(100*(1-statistics.mean(min(1,x) for x in deviations)),1))
            result["rhythm"]={"status":"estimated","relative_onset_accuracy_percent":accuracy,
                              "compared_intervals":len(deviations),
                              "tempo_normalization_factor":round(scale,3)}
            if beat_seconds and math.isfinite(beat_seconds) and beat_seconds>0:
                # Tempo estimate only when score tempo and time units are known.
                target_bpm=60/beat_seconds
                result["tempo"]={"status":"estimated","target_bpm":round(target_bpm,1),
                                 "estimated_bpm":round(target_bpm/scale,1),
                                 "relative_deviation_percent":round((1/scale-1)*100,1)}
    if time_signature_numerator and beat_seconds and beat_seconds>0:
        if isinstance(time_signature_numerator,int) and 1<=time_signature_numerator<=12:
            for e in result["note_errors"]:
                idx=e.get("reference_note")
                if idx is not None:
                    e["estimated_measure"]=int(reference[idx-1].start/(beat_seconds*time_signature_numerator))+1
    result["score_details"]={"pitch_sequence":{"value":result["pitch_similarity_percent"],
                                               "status":"estimated"},
                             "rhythm":result["rhythm"],
                             "tempo":result["tempo"]}
    result["limitations"]=[
        "Overall 100-point score withheld until all rubric criteria have reliable measurements",
        "Note event timing must come from validated notation and audio transcription",
        "No polyphonic instrument-specific articulation or expression grading"]
    return result
