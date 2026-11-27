"""Bounded audio decoding with genuine per-chunk checkpoints and a cooperative deadline."""
from __future__ import annotations
import math, os, subprocess, tempfile, wave, time
from pathlib import Path
import numpy as np
from fastapi import HTTPException
from advanced_analysis import extract_events
_SUPPORTED = {'.wav','.mp3','.m4a','.ogg','.flac','.webm','.aac','.mp4','.mov','.mkv','.avi','.m4v'}
_MAX_SECONDS=900
_MAX_INPUT_BYTES=96*1024*1024
_RATE=16000
_CHUNK_SECONDS=30
_OVERLAP_SECONDS=.35

def _allowed_seconds():
    try: seconds=int(os.getenv('NOTA_MAX_AUDIO_SECONDS','900'))
    except ValueError: seconds=_MAX_SECONDS
    return max(30,min(seconds,_MAX_SECONDS))

def _add_clipped_event(result, item, offset, core_start, core_end):
    try:
        pitch=int(item['midi']); start=offset+float(item['start']); end=start+float(item['duration'])
    except (KeyError,ValueError,TypeError,OverflowError): return
    if not (21<=pitch<=108) or not (math.isfinite(start) and math.isfinite(end)): return
    start=max(start,core_start); end=min(end,core_end)
    if end-start<.035:return
    confidence=float(item.get('confidence',0)); confidence=confidence if math.isfinite(confidence) else 0.
    confidence=max(0.,min(1.,confidence))
    if result and result[-1]['midi']==pitch:
        previous=result[-1]; previous_end=previous['start']+previous['duration']
        if core_start>0 and previous_end>=core_start-.13 and start<=core_start+.12 and start<=previous_end+.095:
            previous['duration']=round(max(previous_end,end)-previous['start'],3)
            previous['confidence']=round(min(previous['confidence'],confidence),3); return
    result.append({'midi':pitch,'start':round(start,3),'duration':round(end-start,3),'confidence':round(confidence,3)})

def audio_to_notes(raw,filename,*,on_progress=None,deadline=None):
    ext=Path(filename or '').suffix.lower()
    if ext not in _SUPPORTED:raise HTTPException(415,'Unsupported audio/video format')
    if not raw or len(raw)>_MAX_INPUT_BYTES:raise HTTPException(413,'Audio file is empty or exceeds 96 MB')
    limit=_allowed_seconds()
    events=[]; analyzed_chunks=0; analyzed_seconds=0.; duration=0.
    def snapshot(incomplete,reason=None):
        return {'notes':[e['midi'] for e in events], 'events':[dict(e) for e in events],
            'timing_status':'estimated_from_monophonic_audio','duration_seconds':duration,
            'analyzed_seconds':round(analyzed_seconds,3),'incomplete':incomplete,
            'method':'overlapping 30-second autocorrelation chunks (monophonic estimate)',
            'chunks_analyzed':analyzed_chunks,'max_audio_seconds':limit,'total_detected_events':len(events),
            'note_sequence_capped':False,'warnings':[reason] if reason else [],'stop_reason':reason}
    with tempfile.TemporaryDirectory(prefix='nota_long_audio_') as tmp:
        source=Path(tmp)/('input'+ext); target=Path(tmp)/'normalized.wav'; source.write_bytes(raw)
        command=['ffmpeg','-hide_banner','-loglevel','error','-nostdin','-y','-i',str(source),'-map','0:a:0','-vn','-sn','-dn','-t',str(limit+1),'-ac','1','-ar',str(_RATE),'-c:a','pcm_s16le',str(target)]
        try:
            timeout=min(100,max(.1,deadline-time.monotonic())) if deadline is not None else 100
            proc=subprocess.run(command,capture_output=True,timeout=timeout,check=False)
        except FileNotFoundError as ex:raise HTTPException(503,'FFmpeg audio decoder unavailable') from ex
        except subprocess.TimeoutExpired as ex:raise HTTPException(504,'Audio decoding exceeded time budget') from ex
        if proc.returncode!=0 or not target.exists():raise HTTPException(422,'Audio could not be decoded')
        with wave.open(str(target),'rb') as decoded:
            if decoded.getnchannels()!=1 or decoded.getsampwidth()!=2 or decoded.getframerate()!=_RATE:raise HTTPException(422,'Unexpected decoded audio format')
            total=decoded.getnframes(); duration=round(total/_RATE,3)
            if total<_RATE:raise HTTPException(422,'Audio is too short')
            if total>limit*_RATE+20:raise HTTPException(413,f'Audio exceeds {limit} seconds; use a queued long-analysis job')
            try:
                for core_left in range(0,total,_CHUNK_SECONDS*_RATE):
                    if deadline is not None and time.monotonic()>=deadline:raise TimeoutError('audio_time_budget_exceeded')
                    core_right=min(total,core_left+_CHUNK_SECONDS*_RATE); pad=round(_OVERLAP_SECONDS*_RATE)
                    read_left=max(0,core_left-pad); read_right=min(total,core_right+pad); decoded.setpos(read_left)
                    data=decoded.readframes(read_right-read_left)
                    if not data:raise ValueError('audio_chunk_empty')
                    chunk=np.frombuffer(data,dtype='<i2').astype(np.float32)/32768.0
                    segment=extract_events(chunk,_RATE)
                    # Commit an entire chunk atomically, so a failed chunk never corrupts the last checkpoint.
                    next_events=events.copy()
                    if next_events:next_events[-1]=dict(next_events[-1])
                    for item in segment:_add_clipped_event(next_events,item,read_left/_RATE,core_left/_RATE,core_right/_RATE)
                    events=next_events; analyzed_chunks+=1; analyzed_seconds=core_right/_RATE
                    if on_progress and events:on_progress(snapshot(True))
            except Exception as exc:
                if events:return snapshot(True,'audio_time_budget_exceeded' if isinstance(exc,TimeoutError) else 'audio_chunk_failed_'+type(exc).__name__)
                raise
    if not events:raise HTTPException(422,'No stable pitched sound detected')
    return snapshot(False)
