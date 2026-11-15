import base64
import io
import json
import math
import os
import subprocess
import tempfile
from typing import Optional

import fitz
import httpx
import numpy as np
from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from pydantic import BaseModel, Field
from scipy.io import wavfile
from scipy.signal import find_peaks, resample_poly

app = FastAPI(title='NOTA AI Analysis', version='0.2.0')
MAX_PDF = 8 * 1024 * 1024
MAX_AUDIO = 12 * 1024 * 1024


def authorize(value):
    secret = os.getenv('NOTA_API_TOKEN')
    if not secret or value != 'Bearer ' + secret:
        raise HTTPException(401, 'API token is missing or invalid')


class CompareRequest(BaseModel):
    reference_midi: list[int] = Field(min_length=1, max_length=1000)
    performed_hz: list[float] = Field(min_length=1, max_length=1000)


def compare_notes(expected, actual):
    n, m = len(expected), len(actual)
    prev = list(range(m + 1))
    for i, note in enumerate(expected, 1):
        curr = [i] + [0] * m
        for j, played in enumerate(actual, 1):
            curr[j] = min(curr[j-1] + 1, prev[j] + 1, prev[j-1] + (note != played))
        prev = curr
    edits = prev[m]
    return {'reference_count': n, 'performed_count': m, 'performed_midi': actual,
            'note_similarity_percent': max(0, round(100 * (1 - edits / max(n, m)))),
            'edit_distance': edits, 'rhythm_scored': False,
            'warning': 'Approximate melody pitch matching; not a certified performance assessment.'}


@app.get('/health')
def health():
    return {'status': 'ok', 'version': '0.2.0', 'pdf_ai_configured': bool(os.getenv('GEMINI_API_KEY'))}


@app.post('/v1/compare')
def compare(data: CompareRequest, authorization: Optional[str] = Header(None)):
    authorize(authorization)
    if any(not 21 <= x <= 108 for x in data.reference_midi):
        raise HTTPException(422, 'MIDI notes must be between 21 and 108')
    if any(not math.isfinite(x) or not 50 <= x <= 2000 for x in data.performed_hz):
        raise HTTPException(422, 'Invalid audio frequency')
    actual = [round(69 + 12 * math.log2(x / 440)) for x in data.performed_hz]
    return compare_notes(data.reference_midi, actual)


async def read_limited(upload, limit):
    contents = await upload.read(limit + 1)
    if len(contents) > limit:
        raise HTTPException(413, 'File exceeds size limit')
    if not contents:
        raise HTTPException(422, 'Empty file')
    return contents


async def pdf_to_notes(data):
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        raise HTTPException(503, 'GEMINI_API_KEY is not configured')
    if not data.startswith(b'%PDF'):
        raise HTTPException(415, 'Expected PDF file')
    try:
        doc = fitz.open(stream=data, filetype='pdf')
        if doc.needs_pass:
            raise HTTPException(422, 'Encrypted PDFs are not supported')
        if not 1 <= len(doc) <= 4:
            raise HTTPException(422, 'PDF must contain 1 to 4 pages')
        parts = [{'text': ('Read the melody on this music score in page order. Return ONLY JSON with '
                           '"notes" as MIDI pitch integers and "confidence" as high, medium, or low. '
                           'Follow clefs and accidentals; do not invent unreadable notes. '
                           'For polyphony choose the top melody line only. Maximum 300 notes.')}]
        for page in doc:
            pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5), alpha=False)
            parts.append({'inline_data': {'mime_type': 'image/png',
                                          'data': base64.b64encode(pix.tobytes('png')).decode()}})
        doc.close()
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(422, f'Could not read PDF: {type(exc).__name__}')
    model = os.getenv('GEMINI_MODEL', 'gemini-2.5-flash')
    url = f'https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent'
    try:
        async with httpx.AsyncClient(timeout=75) as client:
            result = await client.post(url, params={'key': api_key}, json={
                'contents': [{'parts': parts}],
                'generationConfig': {'responseMimeType': 'application/json', 'temperature': 0}})
        if result.status_code != 200:
            raise HTTPException(502, f'AI notation recognition unavailable (HTTP {result.status_code}); verify GEMINI_MODEL and key')
        output = result.json()['candidates'][0]['content']['parts'][0]['text']
        parsed = json.loads(output)
        notes = [x for x in parsed.get('notes', []) if type(x) is int and 21 <= x <= 108][:300]
        if not notes:
            raise HTTPException(422, 'No reliable music notes recognized in PDF')
        return {'notes': notes, 'confidence': parsed.get('confidence', 'unknown')}
    except HTTPException:
        raise
    except (ValueError, KeyError, IndexError, httpx.HTTPError):
        raise HTTPException(502, 'AI did not return valid notation data')


def audio_to_notes(raw, filename):
    ext = os.path.splitext(filename or '')[1].lower()
    if ext not in ('.wav', '.mp3', '.m4a', '.ogg', '.flac', '.webm'):
        raise HTTPException(415, 'Supported audio: wav, mp3, m4a, ogg, flac, webm')
    with tempfile.TemporaryDirectory() as folder:
        path = os.path.join(folder, 'input' + ext)
        with open(path, 'wb') as f:
            f.write(raw)
        command = ['ffmpeg', '-v', 'error', '-nostdin', '-i', path, '-t', '90',
                   '-ac', '1', '-ar', '16000', '-f', 'wav', os.path.join(folder, 'audio.wav')]
        try:
            p = subprocess.run(command, capture_output=True, timeout=25)
        except (FileNotFoundError, subprocess.TimeoutExpired):
            raise HTTPException(503, 'Audio converter unavailable')
        if p.returncode:
            raise HTTPException(422, 'Audio could not be decoded')
        rate, samples = wavfile.read(os.path.join(folder, 'audio.wav'))
    if samples.size < rate:
        raise HTTPException(422, 'Audio is too short')
    samples = samples.astype(np.float32)
    samples /= max(float(np.max(np.abs(samples))), 1)
    hop, window = 1600, 4096
    notes = []
    for start in range(0, len(samples)-window, hop):
        frame = samples[start:start+window]
        rms = np.sqrt(np.mean(frame*frame))
        if rms < 0.018:
            continue
        frame = frame - frame.mean()
        spectrum = np.abs(np.fft.rfft(frame * np.hanning(window)))
        freqs = np.fft.rfftfreq(window, 1/rate)
        mask = (freqs >= 80) & (freqs <= 1100)
        candidates = np.flatnonzero(mask)
        peaks, _ = find_peaks(spectrum)
        peaks = np.intersect1d(peaks, candidates)
        if not len(peaks):
            continue
        top = peaks[np.argmax(spectrum[peaks])]
        hz = float(freqs[top])
        midi = round(69 + 12 * math.log2(hz/440))
        if 21 <= midi <= 108:
            notes.append(midi)
    if not notes:
        raise HTTPException(422, 'No pitched sound detected')
    # Remove brief transitions and collapse sustained pitches into events.
    smoothed = []
    for i in range(1, len(notes)-1):
        smoothed.append(int(np.median(notes[i-1:i+2])))
    collapsed = []
    for note in smoothed:
        if not collapsed or collapsed[-1] != note:
            collapsed.append(note)
    return {'notes': collapsed[:300], 'duration_seconds': round(len(samples)/rate, 2),
            'method': 'spectral dominant pitch (monophonic approximation)'}


@app.post('/v1/analyze-files')
async def analyze_files(score: UploadFile = File(...), audio: UploadFile = File(...),
                        authorization: Optional[str] = Header(None)):
    authorize(authorization)
    pdf_bytes = await read_limited(score, MAX_PDF)
    audio_bytes = await read_limited(audio, MAX_AUDIO)
    score_data = await pdf_to_notes(pdf_bytes)
    audio_data = audio_to_notes(audio_bytes, audio.filename)
    result = compare_notes(score_data['notes'], audio_data['notes'])
    return {'score': score_data, 'audio': audio_data, 'comparison': result,
            'limitations': ['Monophonic melody only', 'No verified rhythm assessment',
                            'AI sheet-music extraction may be inaccurate',
                            'Instrument harmonics may affect pitch detection']}
