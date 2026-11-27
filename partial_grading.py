"""Conservative excerpt estimates; never a full professional grade."""
import math
from difflib import SequenceMatcher
from statistics import median, mean

WEIGHTS = {'notation_accuracy':25,'intonation':20,'rhythm':20,'tempo':15,'articulation':10,'expression':10}

def estimate(score, audio, bpm=None):
    s=score or {}; a=audio or {}
    ref=list(s.get('notes') or [])[:256]; played=list(a.get('notes') or [])[:256]
    if min(len(ref),len(played))<5:return None
    if any(type(n) is not int or not 21<=n<=108 for n in ref+played):return None
    blocks=[b for b in SequenceMatcher(None,ref,played,autojunk=False).get_matching_blocks() if b.size>=3 and len(set(ref[b.a:b.a+b.size]))>=2]
    if not blocks:return None
    first=last=blocks[0]; chosen=[first]
    if first.a>4 or first.b>4:return None
    for b in blocks[1:]:
        if b.a-last.a-last.size>8 or b.b-last.b-last.size>8:break
        chosen.append(b);last=b
    matches=sum(b.size for b in chosen)
    x=ref[first.a:last.a+last.size]; y=played[first.b:last.b+last.size]
    if matches<5 or matches/max(len(x),len(y))<.6:return None
    # Edit distance over the bounded excerpt, not the whole unread score.
    prev=list(range(len(y)+1))
    for i,n in enumerate(x,1):
        curr=[i]
        for j,p in enumerate(y,1):curr.append(min(curr[-1]+1,prev[j]+1,prev[j-1]+(n!=p)))
        prev=curr
    pitch=round(100*(1-prev[-1]/max(len(x),len(y))),1)
    metrics={'notation_accuracy':pitch}
    intervals=[]
    se=s.get('events') or []; ae=a.get('events') or []
    def valid_events(events, notes, start, duration, audio=False):
        if len(events)<len(notes):return False
        for e,n in zip(events,notes):
            if not isinstance(e,dict) or e.get('midi')!=n:return False
            for k in (start,duration):
                v=e.get(k)
                if isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or v<0 or (k==duration and v==0):return False
        return all(events[i][start]<=events[i+1][start] for i in range(len(notes)-1))
    # Timing only on consecutive exact anchors, with sufficiently confident audio.
    if s.get('confidence')=='high' and valid_events(se,ref,'onset_beats','duration_beats') and valid_events(ae,played,'start','duration'):
        for b in chosen:
            for k in range(b.size-1):
                i,j=b.a+k,b.b+k
                if any(not isinstance(e.get('confidence'),(int,float)) or e['confidence']<.65 for e in ae[j:j+2]):continue
                rb=se[i+1]['onset_beats']-se[i]['onset_beats']; ps=ae[j+1]['start']-ae[j]['start']
                if rb>0 and ps>.02:intervals.append((rb,ps))
        if len(intervals)>=4:
            scale=median(p/r for r,p in intervals)
            metrics['rhythm']=round(100*(1-mean(min(1,abs(p/(r*scale)-1)) for r,p in intervals)),1)
            if isinstance(bpm,(int,float)) and not isinstance(bpm,bool) and math.isfinite(bpm) and 20<=bpm<=300:
                metrics['tempo']=round(max(0,100-200*abs((60/scale)/bpm-1)),1)
    weight=sum(WEIGHTS[k] for k in metrics)
    grade=round(sum(WEIGHTS[k]*v for k,v in metrics.items())/weight,1)
    return {'status':'estimated','scope':'observed_excerpt_only','partial_score_100':grade,'measured_weight_percent':weight,
            'criteria':{k:{'score_percent':metrics.get(k),'weight':w,'status':'estimated' if k in metrics else 'not_measured'} for k,w in WEIGHTS.items()},
            'reference_start':first.a+1,'performed_start':first.b+1,'reference_count':len(x),'performed_count':len(y),
            'compared_note_count':min(len(x),len(y)),'matched_anchor_notes':matches,'compared_intervals':len(intervals),
            'professional_score_100':None,'technical_score_100':None,'warning':'Taxminiy parcha bahosi, yakuniy professional baho emas.'}
