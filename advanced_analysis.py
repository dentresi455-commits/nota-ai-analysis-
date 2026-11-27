"""Monophonic event extraction and transparent technical grading."""
import math
import numpy as np
from scipy.signal import find_peaks


def _strong_attacks(samples, rate):
    # Detect independent pluck/envelope resets, including repeated pitches.
    # A conservative 2.2x rise over a preceding quiet window prevents ordinary vibrato splits.
    step=max(1,round(rate*.005)); x=np.asarray(samples,dtype=float)
    if len(x)<step*20:return []
    rms=np.sqrt(np.mean(x[:len(x)//step*step].reshape(-1,step)**2,axis=1))
    rise=np.diff(np.log(rms+.001))
    peaks,_=find_peaks(rise,height=math.log(2.2),distance=max(1,round(.085*rate/step)))
    return [float((i+1)*step/rate) for i in peaks if rms[i+1]>.025 and rms[i+1]-rms[i]>.018]


def extract_events(samples, rate, hop_seconds=.025):
    """Extract stable pitch and retain high-confidence fractional MIDI measurements."""
    hop=max(1,round(rate*hop_seconds));win=max(2048,round(rate*.08));frames=[]
    for start in range(0,len(samples)-win,hop):
        y=np.asarray(samples[start:start+win],dtype=float)
        rms=float(np.sqrt(np.mean(y*y)))
        if rms<.015:frames.append((start/rate,None,0.,None));continue
        y=y-y.mean();corr=np.fft.irfft(abs(np.fft.rfft(y,n=2*win))**2,n=2*win)[:win]
        lo=max(1,int(rate/1100));hi=min(win-1,int(rate/80));region=corr[lo:hi+1];peaks,_=find_peaks(region)
        if len(peaks)==0:frames.append((start/rate,None,0.,None));continue
        lag=lo+int(peaks[np.argmax(region[peaks])]);confidence=float(corr[lag]/max(corr[0],1e-8))
        # Sub-sample lag interpolation reduces quantization of high pitches.
        left,mid,right=map(float,corr[lag-1:lag+2]);den=left-2*mid+right
        offset=max(-.5,min(.5,.5*(left-right)/den)) if den<0 else 0.
        pitch=69+12*math.log2(rate/(lag+offset)/440);midi=int(round(pitch))
        frames.append((start/rate,midi if confidence>=.48 and 21<=midi<=108 else None,confidence,pitch))
    notes=[f[1] for f in frames];smooth=[]
    for i,x in enumerate(notes):
        local=[v for v in notes[max(0,i-1):min(len(notes),i+2)] if v is not None]
        smooth.append(int(np.median(local)) if x is not None and len(local)>=2 else x)
    events=[];start_idx=None;last=None
    attacks=_strong_attacks(samples,rate)
    # The pitch frame is forward-looking. Map the envelope attack to the frame center.
    attack_frames={max(1,round((t-win/(2*rate))/(hop/rate))) for t in attacks}
    def emit(end_idx):
        if start_idx is None or last is None:return
        duration=(end_idx-start_idx)*hop/rate
        if duration>=.085:
            conf=float(np.median([f[2] for f in frames[start_idx:end_idx]]))
            e={'midi':last,'start':round(frames[start_idx][0],3),'duration':round(duration,3),'confidence':round(conf,3)}
            # Only steady, confident interior frames; avoid onset/offset transients.
            values=[(f[3]-last)*100 for f in frames[start_idx+1:end_idx-1] if f[1]==last and f[2]>=.65 and f[3] is not None]
            if len(values)>=3:
                e['pitch_deviation_cents']=round(float(np.median(values)),2)
                e['pitch_spread_cents']=round(float(np.percentile(values,75)-np.percentile(values,25)),2)
                e['pitch_measurement_frames']=len(values)
            events.append(e)
    for i,n in enumerate(smooth):
        if n!=last:
            emit(i);start_idx=i if n is not None else None;last=n
        elif n is not None and i in attack_frames and start_idx is not None and (i-start_idx)*hop/rate>=.085:
            # Only split an existing stable pitch, not an already detected pitch transition.
            if i+4<=len(smooth) and all(v==n for v in smooth[i:i+4]):
                emit(i);start_idx=i
    emit(len(smooth));return events


def technical_score(alignment, *, score_confidence, audio_events):
    pitch=alignment.get('pitch_similarity_percent');rhythm=alignment.get('rhythm',{});tempo=alignment.get('tempo',{})
    if (score_confidence!='high' or len(audio_events)<5 or pitch is None or rhythm.get('status')!='estimated' or tempo.get('status')!='estimated'):
        return {'status':'insufficient_evidence','technical_score_100':None,'professional_score_100':None,'reason':'Validated notation, stable audio, rhythm and tempo are all required'}
    r=rhythm['relative_onset_accuracy_percent'];deviation=abs(tempo['relative_deviation_percent']);t=max(0,100-2*deviation)
    return {'status':'estimated','technical_score_100':round(.5*pitch+.3*r+.2*t,1),'professional_score_100':None,'rubric':{'pitch_weight':50,'rhythm_weight':30,'tempo_weight':20},'warning':'Technical estimate only; not a teacher-validated professional grade'}
