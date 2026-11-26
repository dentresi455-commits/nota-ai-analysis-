import io
import math
import os
import struct
import sys
import types
import wave

import numpy as np
import pytest
from fastapi import HTTPException

# Only the detector is substituted: tests exercise real FFmpeg decoding,
# chunk boundaries, duration guards, and the API contract.
if 'advanced_analysis' not in sys.modules:
    mod = types.ModuleType('advanced_analysis')
    mod.extract_events = lambda chunk, rate: []
    sys.modules['advanced_analysis'] = mod
import long_audio


def audio_file(seconds: int, *, note: int = 69) -> bytes:
    rate = 16000
    w = io.BytesIO()
    with wave.open(w, 'wb') as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(rate)
        hz = 440 * 2 ** ((note-69)/12)
        # Build in 1-second bounded chunks.
        sample = np.arange(rate, dtype=np.float32)
        tone = (np.sin(2*np.pi*hz*sample/rate)*6000).astype('<i2').tobytes()
        for _ in range(seconds):
            f.writeframes(tone)
    return w.getvalue()


def detected_tone(chunk, rate):
    # A consistent held note across the chunk's overlap, no need for real OMR.
    return [{'midi': 69, 'start': 0, 'duration': len(chunk) / rate,
             'confidence': .89}]


def test_95_seconds_not_cut_at_90(monkeypatch):
    monkeypatch.setattr(long_audio, 'extract_events', detected_tone)
    out = long_audio.audio_to_notes(audio_file(95), 'long.wav')
    assert out['duration_seconds'] == 95
    assert out['chunks_analyzed'] == 4
    assert len(out['events']) == 1
    assert abs(out['events'][0]['duration']-95) < .003
    assert out['notes'] == [69]


def test_155_seconds_auto_next_chunk(monkeypatch):
    monkeypatch.setattr(long_audio, 'extract_events', detected_tone)
    out=long_audio.audio_to_notes(audio_file(155), 'long.wav')
    assert out['duration_seconds']==155
    assert out['chunks_analyzed']==6
    assert out['events'][-1]['start']+out['events'][-1]['duration'] >= 154.9


def test_exceed_configured_duration_is_rejected_not_silently_cut(monkeypatch):
    monkeypatch.setattr(long_audio, 'extract_events', detected_tone)
    monkeypatch.setenv('NOTA_MAX_AUDIO_SECONDS', '90')
    with pytest.raises(HTTPException) as e:
        long_audio.audio_to_notes(audio_file(93), 'long.wav')
    assert e.value.status_code==413
    assert 'exceeds 90' in str(e.value.detail)


def test_midi_boundary_and_timing_integrity():
    result=[]
    long_audio._add_clipped_event(result,{'midi':69,'start':0,'duration':30.2,'confidence':.9},0,0,30)
    long_audio._add_clipped_event(result,{'midi':69,'start':0,'duration':30.3,'confidence':.88},29.7,30,60)
    long_audio._add_clipped_event(result,{'midi':72,'start':0,'duration':1,'confidence':.92},60,60,90)
    assert result == [
        {'midi':69,'start':0,'duration':60,'confidence':.88},
        {'midi':72,'start':60,'duration':1,'confidence':.92}
    ]


def test_nearby_repeated_note_not_accidentally_merged():
    result=[]
    long_audio._add_clipped_event(result,{'midi':69,'start':5,'duration':.25,'confidence':.9},0,0,30)
    long_audio._add_clipped_event(result,{'midi':69,'start':5.31,'duration':.25,'confidence':.9},0,0,30)
    assert len(result)==2


def test_validates_format_and_silence(monkeypatch):
    with pytest.raises(HTTPException) as e:
        long_audio.audio_to_notes(b'audio','score.pdf')
    assert e.value.status_code==415
    monkeypatch.setattr(long_audio, 'extract_events', lambda *a: [])
    with pytest.raises(HTTPException) as e:
        long_audio.audio_to_notes(audio_file(2),'silence.wav')
    assert e.value.status_code==422



def test_15_min_500_plus_events_all_preserved(monkeypatch):
    # A real 900-second PCM WAV is decoded with FFmpeg. Only the pitch
    # detector is mocked to make event counts deterministic and inexpensive.
    def many_events(chunk, rate):
        return [{'midi':60+(i%7),'start':i*1.42,'duration':.54,'confidence':.83}
                for i in range(20)]
    monkeypatch.setattr(long_audio, 'extract_events', many_events)
    result=long_audio.audio_to_notes(audio_file(900),'fifteen_minutes.wav')
    assert result['duration_seconds']==900
    assert result['chunks_analyzed']==30
    assert len(result['notes'])>500
    assert result['total_detected_events']==len(result['notes'])
    assert result['events'][-1]['start']>870
    assert result['note_sequence_capped'] is False
