"""Monophonic event extraction and transparent technical grading.
Not a substitute for teacher-validated professional performance assessment.
"""
import math
import numpy as np
from scipy.signal import find_peaks


def extract_events(samples, rate, hop_seconds=.025):
    """Track fundamental by autocorrelation, segment stable pitch regions.
    Returns events with MIDI pitch, onset and duration (seconds), confidence.
    Intended for clean monophonic recordings; no polyphony support.
    """
    from timing_analysis import Event
    hop = max(1, round(rate * hop_seconds))
    win = max(2048, round(rate * .08))
    frames=[]
    for start in range(0,len(samples)-win,hop):
        y=np.asarray(samples[start:start+win],dtype=float)
        rms=float(np.sqrt(np.mean(y*y)))
        if rms < .015:
            frames.append((start/rate,None,0.));continue
        y=y-y.mean()
        corr=np.fft.irfft(abs(np.fft.rfft(y,n=2*win))**2,n=2*win)[:win]
        lo=max(1,int(rate/1100));hi=min(win-1,int(rate/80))
        region=corr[lo:hi+1]
        peaks,_=find_peaks(region)
        if len(peaks)==0:
            frames.append((start/rate,None,0.));continue
        lag=lo+int(peaks[np.argmax(region[peaks])])
        confidence=float(corr[lag]/max(corr[0],1e-8))
        hz=rate/lag
        midi=int(round(69+12*math.log2(hz/440)))
        frames.append((start/rate,midi if confidence>=.48 and 21<=midi<=108 else None,confidence))
    # Suppress isolated pitch glitches with a three-frame median.
    notes=[f[1] for f in frames]
    smooth=[]
    for i,x in enumerate(notes):
        local=[v for v in notes[max(0,i-1):min(len(notes),i+2)] if v is not None]
        smooth.append(int(np.median(local)) if x is not None and len(local)>=2 else x)
    events=[];start_idx=None;last=None
    def emit(end_idx):
        if start_idx is None or last is None:return
        duration=(end_idx-start_idx)*hop/rate
        if duration>=.085:
            conf=float(np.median([f[2] for f in frames[start_idx:end_idx]]))
            events.append({'midi':last,'start':round(frames[start_idx][0],3),
                           'duration':round(duration,3),'confidence':round(conf,3)})
    for i,n in enumerate(smooth):
        if n!=last:
            emit(i)
            start_idx=i if n is not None else None
            last=n
    emit(len(smooth))
    return events[:500]


def technical_score(alignment, *, score_confidence, audio_events):
    """A 100-point technical score only if pitch, rhythm and tempo measured.
    This is NOT a full professional 100-point grade. Missing evidence => None.
    """
    pitch=alignment.get('pitch_similarity_percent')
    rhythm=alignment.get('rhythm',{})
    tempo=alignment.get('tempo',{})
    if (score_confidence!='high' or len(audio_events)<5 or pitch is None
        or rhythm.get('status')!='estimated' or tempo.get('status')!='estimated'):
        return {'status':'insufficient_evidence','technical_score_100':None,
                'professional_score_100':None,
                'reason':'Validated notation, stable audio, rhythm and tempo are all required'}
    r=rhythm['relative_onset_accuracy_percent']
    deviation=abs(tempo['relative_deviation_percent'])
    t=max(0,100-2*deviation)
    return {'status':'estimated','technical_score_100':round(.5*pitch+.3*r+.2*t,1),
            'professional_score_100':None,
            'rubric':{'pitch_weight':50,'rhythm_weight':30,'tempo_weight':20},
            'warning':'Technical estimate only; not a teacher-validated professional grade'}
