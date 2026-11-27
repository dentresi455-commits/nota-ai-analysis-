import base64
import asyncio
import time
import logging
from fastapi.responses import StreamingResponse
import json
import math
import os
import subprocess
import tempfile
import io
from typing import Optional
from starlette.concurrency import run_in_threadpool
from timing_analysis import Event, analyze
from advanced_analysis import extract_events, technical_score
from professional_grading import grade as professional_grade
from instrument_profiles import INSTRUMENTS, get_instrument, instrument_guidance

import fitz
import httpx
import numpy as np
from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from pydantic import BaseModel, Field
from scipy.io import wavfile
from scipy.signal import find_peaks
from PIL import Image, ImageOps, UnidentifiedImageError
try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    pass

app = FastAPI(title="NOTA AI Analysis", version="0.9.8")
# Reject concurrent heavy jobs rather than accumulating uploads in RAM.
_ANALYSIS_LOCK = asyncio.Lock()
MAX_PDF = (32 if os.getenv("NOTA_OMR_V3_ENABLED", "false").lower() == "true" else 8) * 1024 * 1024
MAX_AUDIO = 96 * 1024 * 1024
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif", ".tif", ".tiff", ".bmp"}
VIDEO_EXT = {".mp4", ".mov", ".webm", ".mkv", ".avi", ".m4v"}
AUDIO_EXT = {".wav", ".mp3", ".m4a", ".ogg", ".flac", ".webm", ".aac"}


def authorize(value):
    secret = os.getenv("NOTA_API_TOKEN")
    if not secret or value != "Bearer " + secret:
        raise HTTPException(401, "API token is missing or invalid")


# Keep the small legacy limit unless linear-memory alignment is explicitly on.
COMPARE_LIMIT = 12000 if os.getenv("NOTA_FULL_ALIGNMENT_ENABLED", "false").lower() == "true" else 1000


class CompareRequest(BaseModel):
    reference_midi: list[int] = Field(min_length=1, max_length=COMPARE_LIMIT)
    performed_hz: list[float] = Field(min_length=1, max_length=COMPARE_LIMIT)


def detailed_note_feedback(expected, actual):
    n, m = len(expected), len(actual)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            dp[i][j] = min(dp[i-1][j] + 1, dp[i][j-1] + 1,
                           dp[i-1][j-1] + (expected[i-1] != actual[j-1]))
    i, j = n, m
    issues, correct = [], 0
    while i or j:
        if i and j and dp[i][j] == dp[i-1][j-1] + (expected[i-1] != actual[j-1]):
            if expected[i-1] == actual[j-1]:
                correct += 1
            else:
                issues.append({"type": "wrong_pitch", "reference_position": i,
                               "expected_midi": expected[i-1],
                               "performed_midi": actual[j-1],
                               "semitone_difference": actual[j-1] - expected[i-1]})
            i -= 1
            j -= 1
        elif i and dp[i][j] == dp[i-1][j] + 1:
            issues.append({"type": "missing_note", "reference_position": i,
                           "expected_midi": expected[i-1]})
            i -= 1
        else:
            issues.append({"type": "extra_note", "performed_position": j,
                           "performed_midi": actual[j-1]})
            j -= 1
    issues.reverse()
    counts = {kind: sum(x["type"] == kind for x in issues)
              for kind in ("wrong_pitch", "missing_note", "extra_note")}
    return {"correct_notes": correct, "error_counts": counts,
            "errors": issues[:100], "assessment_scope": "Pitch-event sequence only",
            "rhythm": {"status": "not_measured"},
            "tempo": {"status": "not_measured"},
            "intonation_cents": {"status": "not_measured"},
            "feedback": "Review wrong-pitch positions and repeat the melody slowly."}


def professional_report(comparison, score_info, audio_info):
    """Evidence-bound Uzbek pedagogical report; no invented rhythm/tempo grades."""
    details = comparison["detailed_feedback"]
    issues = details["errors"]
    advice = []
    labels = {
        "wrong_pitch": "Noto‘g‘ri nota",
        "missing_note": "Tushib qolgan nota",
        "extra_note": "Ortiqcha nota",
    }
    for issue in issues:
        kind = issue["type"]
        if kind == "wrong_pitch":
            delta = issue["semitone_difference"]
            advice.append({
                "category": labels[kind],
                "reference_note": issue["reference_position"],
                "expected_midi": issue["expected_midi"],
                "performed_midi": issue["performed_midi"],
                "semitones": delta,
                "explanation": f"Ijro etilgan tovush notadan {abs(delta)} yarim ton {'baland' if delta > 0 else 'past'}.",
                "practice": "Shu notani alohida, so‘ng yonidagi notalar bilan sekin tempda mashq qiling.",
            })
        elif kind == "missing_note":
            advice.append({
                "category": labels[kind],
                "reference_note": issue["reference_position"],
                "expected_midi": issue["expected_midi"],
                "explanation": "Notada mavjud tovush ijro ketma-ketligida topilmadi.",
                "practice": "Ushbu nota oldi va keyingi tovushlar bilan bog‘lab qayta mashq qilinsin.",
            })
        else:
            advice.append({
                "category": labels[kind],
                "performed_note": issue["performed_position"],
                "performed_midi": issue["performed_midi"],
                "explanation": "Notaga mos kelmaydigan qo‘shimcha tovush aniqlangan.",
                "practice": "Ortiqcha tovush paydo bo‘lgan o‘tishni sekin va nazorat bilan qaytaring.",
            })
    measured = comparison["note_similarity_percent"]
    confidence = score_info.get("confidence", "unknown")
    return {
        "title": "NOTA AI — ijroning pedagogik tahlili",
        "status": "partial",
        "overall_grade_100": None,
        "pitch_sequence_similarity_percent": measured,
        "pitch_score_is_not_overall_grade": True,
        "notation_recognition_confidence": confidence,
        "sections": {
            "umumiy_xulosa": (
                f"Nota va ijro tovushlari ketma-ketligi bo‘yicha taxminiy moslik {measured}%. "
                "Bu yakuniy ijro bahosi emas: ritm, temp, artikulyatsiya va ifodalilik hali o‘lchanmagan."
            ),
            "baholash_mezonlari": {
                "nota_ketma_ketligi": {"value_percent": measured, "status": "estimated"},
                "intonatsiya": {"value": None, "status": "not_measured"},
                "ritm": {"value": None, "status": "not_measured"},
                "temp": {"value": None, "status": "not_measured"},
                "artikulyatsiya": {"value": None, "status": "not_measured"},
                "ifodalilik": {"value": None, "status": "not_measured"},
            },
            "aniqlangan_xatolar": advice,
            "pedagog_xulosasi": (
                "Aniqlangan noto‘g‘ri, tushib qolgan va ortiqcha tovushlar ustida "
                "alohida ishlang. Takt va soniya bo‘yicha aniq xulosa chiqarish uchun "
                "ishonchli nota davomiyligi va audio vaqt moslashuvi zarur."
            ),
            "individual_mashgulot_rejasi": [
                {"minutes": 5, "task": "Aniqlangan xato notalarni alohida takrorlash"},
                {"minutes": 7, "task": "Xato bo‘lgan melodik o‘tishlarni sekin mashq qilish"},
                {"minutes": 5, "task": "Asarni to‘liq qayta ijro etish"},
                {"minutes": 3, "task": "Yangi yozuvni oldingi ijro bilan solishtirish"},
            ],
        },
        "measurement_limitations": [
            "Takt raqami va xato soniyasi hisoblanmagan",
            "Ritm, temp, dinamika, artikulyatsiya va ifodalilik baholanmagan",
            "Ko‘p ovozli asarlarda natija ishonchsiz bo‘lishi mumkin",
            "Tovush ajratish usuli taxminiy, nota tanish AI xato qilishi mumkin",
        ],
    }


def compare_notes(expected, actual):
    n, m = len(expected), len(actual)
    prev = list(range(m + 1))
    for i, note in enumerate(expected, 1):
        curr = [i] + [0] * m
        for j, played in enumerate(actual, 1):
            curr[j] = min(curr[j-1] + 1, prev[j] + 1,
                          prev[j-1] + (note != played))
        prev = curr
    edits = prev[m]
    return {"reference_count": n, "performed_count": m,
            "performed_midi": actual,
            "note_similarity_percent": max(0, round(100 * (1 - edits / max(n, m)))),
            "edit_distance": edits,
            "detailed_feedback": detailed_note_feedback(expected, actual),
            "rhythm_scored": False,
            "warning": "Approximate melody pitch matching; not a certified performance assessment."}


# Exact, memory-bounded whole-sequence alignment, staging opt-in.
if os.getenv("NOTA_FULL_ALIGNMENT_ENABLED", "false").lower() == "true":
    from long_alignment import compare_complete as compare_notes


@app.get("/health")
def health():
    return {"status": "ok", "version": "0.9.3",
            "partial_grading_enabled": True,
            "pdf_ai_configured": bool(os.getenv("GEMINI_API_KEY")),
            "omr_v3_enabled": os.getenv("NOTA_OMR_V3_ENABLED", "false").lower() == "true",
            "long_audio_enabled": os.getenv("NOTA_LONG_AUDIO_ENABLED", "false").lower() == "true",
            "full_alignment_enabled": os.getenv("NOTA_FULL_ALIGNMENT_ENABLED", "false").lower() == "true"}



class TimedNote(BaseModel):
    midi: int = Field(ge=21, le=108)
    start: float = Field(ge=0)
    duration: float = Field(gt=0)

class TimedAnalysisRequest(BaseModel):
    reference: list[TimedNote] = Field(min_length=1, max_length=12000)
    performed: list[TimedNote] = Field(min_length=1, max_length=12000)
    beat_seconds: Optional[float] = Field(default=None, gt=0)
    time_signature_numerator: Optional[int] = Field(default=None, ge=1, le=12)

@app.post("/v1/analyze-timed")
def analyze_timed(data: TimedAnalysisRequest, authorization: Optional[str] = Header(None)):
    authorize(authorization)
    reference = [Event(**e.model_dump()) for e in data.reference]
    performed = [Event(**e.model_dump()) for e in data.performed]
    try:
        return analyze(reference, performed, data.beat_seconds, data.time_signature_numerator)
    except ValueError as error:
        raise HTTPException(422, str(error))


@app.post("/v1/compare")
def compare(data: CompareRequest, authorization: Optional[str] = Header(None)):
    authorize(authorization)
    if any(not 21 <= x <= 108 for x in data.reference_midi):
        raise HTTPException(422, "MIDI notes must be between 21 and 108")
    if any(not math.isfinite(x) or not 50 <= x <= 2000 for x in data.performed_hz):
        raise HTTPException(422, "Invalid audio frequency")
    actual = [round(69 + 12 * math.log2(x / 440)) for x in data.performed_hz]
    return compare_notes(data.reference_midi, actual)


async def read_limited(upload, limit):
    contents = await upload.read(limit + 1)
    if len(contents) > limit:
        raise HTTPException(413, "File exceeds size limit")
    if not contents:
        raise HTTPException(422, "Empty file")
    return contents


def normalize_score(raw, filename):
    """Convert supported image files into a single-page PDF for the notation model."""
    ext = os.path.splitext(filename or "")[1].lower()
    if raw.startswith(b"%PDF"):
        return raw, "pdf"
    if ext not in IMAGE_EXT:
        raise HTTPException(415, "Score must be PDF or a supported image")
    try:
        with Image.open(io.BytesIO(raw)) as img:
            if img.width * img.height > 24_000_000:
                raise HTTPException(413, "Image dimensions too large")
            from photo_preprocessing import bounded_size
            target = bounded_size(*img.size)
            if target != img.size:
                if img.format == "JPEG":
                    img.draft("RGB", target)
                target = bounded_size(*img.size)
                if target != img.size:
                    img = img.resize(target, Image.Resampling.LANCZOS)
            rgb = ImageOps.exif_transpose(img).convert("RGB")
            output = io.BytesIO()
            rgb.save(output, format="PDF", resolution=150)
            return output.getvalue(), "image"
    except HTTPException:
        raise
    except (UnidentifiedImageError, OSError, ValueError):
        raise HTTPException(422, "Invalid or unsupported score image")


async def pdf_to_notes(data, exercise=None):
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(503, "GEMINI_API_KEY is not configured")
    if not data.startswith(b"%PDF"):
        raise HTTPException(415, "Expected PDF file")
    try:
        doc = fitz.open(stream=data, filetype="pdf")
        if doc.needs_pass:
            raise HTTPException(422, "Encrypted PDFs are not supported")
        if not 1 <= len(doc) <= 4:
            raise HTTPException(422, "PDF must contain 1 to 4 pages")
        prompt = (f"Find ONLY exercise number {exercise:02d}. Ignore other exercises. "
                  if exercise is not None else "")
        prompt += ('Read the melody on this music score in page order. Return ONLY JSON '
                   'with "notes" as MIDI pitch integers, "events" as an array of objects '
                   'each with midi (integer), onset_beats (number), duration_beats (positive number), '
                   'and "confidence" as high, medium or low. Follow clefs, accidentals, rests, '
                   'and time signatures; never invent unreadable notes. Return one melody, max 300 notes.')
        parts = [{"text": prompt}]
        for page in doc:
            pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5), alpha=False)
            parts.append({"inline_data": {"mime_type": "image/png",
                           "data": base64.b64encode(pix.tobytes("png")).decode()}})
        doc.close()
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(422, f"Could not read PDF: {type(exc).__name__}")
    model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    try:
        async with httpx.AsyncClient(timeout=75) as client:
            result = await client.post(url, params={"key": api_key}, json={
                "contents": [{"parts": parts}],
                "generationConfig": {"responseMimeType": "application/json", "temperature": 0}})
        if result.status_code != 200:
            raise HTTPException(502, f"AI notation recognition unavailable (HTTP {result.status_code}); verify GEMINI_MODEL and key")
        output = result.json()["candidates"][0]["content"]["parts"][0]["text"]
        parsed = json.loads(output)
        notes = [x for x in parsed.get("notes", []) if type(x) is int and 21 <= x <= 108][:300]
        if not notes:
            raise HTTPException(422, "No reliable music notes recognized in PDF")
        events=[]
        for e in parsed.get("events", [])[:300]:
            if not isinstance(e,dict):continue
            midi=e.get("midi");onset=e.get("onset_beats");dur=e.get("duration_beats")
            if (type(midi) is int and 21<=midi<=108 and isinstance(onset,(int,float))
                and isinstance(dur,(int,float)) and math.isfinite(onset) and math.isfinite(dur)
                and onset>=0 and 0<dur<=32):
                events.append({"midi":midi,"onset_beats":round(onset,4),"duration_beats":round(dur,4)})
        if len(events)!=len(notes) or [e["midi"] for e in events]!=notes:
            events=[]
        return {"notes": notes, "events": events,
                "confidence": parsed.get("confidence", "unknown"),
                "timing_status":"estimated_from_omr" if events else "not_available"}
    except HTTPException:
        raise
    except (ValueError, KeyError, IndexError, httpx.HTTPError):
        raise HTTPException(502, "AI did not return valid notation data")


def audio_to_notes(raw, filename):
    ext = os.path.splitext(filename or "")[1].lower()
    if ext not in AUDIO_EXT | VIDEO_EXT:
        raise HTTPException(415, "Unsupported audio/video format")
    with tempfile.TemporaryDirectory() as folder:
        path = os.path.join(folder, "input" + ext)
        with open(path, "wb") as f:
            f.write(raw)
        command = ["ffmpeg", "-v", "error", "-nostdin", "-i", path, "-t", "900",
                   "-ac", "1", "-ar", "16000", "-f", "wav", os.path.join(folder, "audio.wav")]
        try:
            proc = subprocess.run(command, capture_output=True, timeout=180)
        except (FileNotFoundError, subprocess.TimeoutExpired):
            raise HTTPException(503, "Audio converter unavailable")
        if proc.returncode:
            raise HTTPException(422, "Audio could not be decoded")
        rate, samples = wavfile.read(os.path.join(folder, "audio.wav"))
    if samples.size < rate:
        raise HTTPException(422, "Audio is too short")
    samples = samples.astype(np.float32)
    samples /= max(float(np.max(np.abs(samples))), 1)
    hop, window = 1600, 4096
    notes = []
    for start in range(0, len(samples) - window, hop):
        frame = samples[start:start + window]
        rms = np.sqrt(np.mean(frame * frame))
        if rms < 0.018:
            continue
        frame = frame - frame.mean()
        spectrum = np.abs(np.fft.rfft(frame * np.hanning(window)))
        freqs = np.fft.rfftfreq(window, 1 / rate)
        mask = (freqs >= 80) & (freqs <= 1100)
        candidates = np.flatnonzero(mask)
        peaks, _ = find_peaks(spectrum)
        peaks = np.intersect1d(peaks, candidates)
        if not len(peaks):
            continue
        top = peaks[np.argmax(spectrum[peaks])]
        hz = float(freqs[top])
        midi = round(69 + 12 * math.log2(hz / 440))
        if 21 <= midi <= 108:
            notes.append(midi)
    if not notes:
        raise HTTPException(422, "No pitched sound detected")
    smoothed = [int(np.median(notes[i-1:i+2])) for i in range(1, len(notes)-1)]
    collapsed = []
    for note in smoothed:
        if not collapsed or collapsed[-1] != note:
            collapsed.append(note)
    events=extract_events(samples,rate)
    if events:
        collapsed=[e["midi"] for e in events]
    return {"notes": collapsed[:300], "events":events,
            "timing_status":"estimated_from_monophonic_audio" if events else "not_available",
            "duration_seconds": round(len(samples) / rate, 2),
            "method": "spectral dominant pitch (monophonic approximation)"}


# NOTA long audio: explicit staging opt-in; fallback remains untouched.
if os.getenv("NOTA_LONG_AUDIO_ENABLED", "false").lower() == "true":
    from long_audio import audio_to_notes


@app.get("/v1/instruments")
def list_instruments():
    return {"count": len(INSTRUMENTS), "instruments": list(INSTRUMENTS.values())}


# NOTA v0.9.3: independent measured stages, bounded soft deadline, streaming checkpoints.
# Full JSON compatibility is retained for existing clients. An explicit progress header
# opts into NDJSON so a caller can recover the last measured result on disconnect.
_ANALYSIS_SECONDS = 225
_LOG = logging.getLogger("uvicorn.error")

def _safe_reason(exc):
    if isinstance(exc, HTTPException):
        detail=exc.detail
        return str(detail.get("code", "stage_failed") if isinstance(detail, dict) else detail)[:160]
    return "time_budget_exceeded" if isinstance(exc, TimeoutError) else type(exc).__name__

from difflib import SequenceMatcher
from statistics import median, mean

_PARTIAL_WEIGHTS = {'notation_accuracy':25,'intonation':20,'rhythm':20,'tempo':15,'articulation':10,'expression':10}

def _align_excerpt(ref, played):
    """Semi-global edit alignment: all of the shorter observed sequence, free outer tails of the longer.
    Exact MIDI anchors only; no automatic key or octave shift. Bounded work for checkpoints.
    """
    swapped = len(played) > len(ref)
    short, long = (ref, played) if swapped else (played, ref)
    n, m = len(short), len(long)
    if n < 5 or m < n:return None
    # Edit cost is primary; exact matches break ties. Store traceback in compact rows.
    unit = n + 1
    prev = [0] * (m+1)
    trace = [bytearray(m+1)]
    for i, note in enumerate(short, 1):
        row = [i*unit] + [0]*m
        moves = bytearray(m+1); moves[0] = 1
        for j, other in enumerate(long, 1):
            diag = prev[j-1] + ( -1 if note == other else unit )
            up = prev[j] + unit
            left = row[j-1] + unit
            if diag <= up and diag <= left:row[j]=diag;moves[j]=0
            elif up <= left:row[j]=up;moves[j]=1
            else:row[j]=left;moves[j]=2
        trace.append(moves);prev=row
    best = min(prev[1:]); endpoints = [j for j in range(1,m+1) if prev[j]==best]
    def backtrack(j):
        finish=j; i=n; pairs=[]; edits=0; operations=[]
        while i:
            move=trace[i][j]
            if j and move==0:
                if short[i-1]==long[j-1]:pairs.append((i-1,j-1))
                else:edits+=1;operations.append(("wrong_pitch",i-1,j-1))
                i-=1;j-=1
            elif move==1 or not j:operations.append(("extra_note",i-1,None));i-=1;edits+=1
            else:operations.append(("missing_note",None,j-1));j-=1;edits+=1
        return j,finish,list(reversed(pairs)),edits,list(reversed(operations))
    begin,finish,pairs,edits,operations=backtrack(endpoints[0])
    span=finish-begin; matches=len(pairs)
    # Require enough distinctive exact evidence; a coincidental or all-wrong sequence is not a grade.
    if span < 5 or matches < 5 or matches/max(n,span)<.40:return None
    if len({short[i] for i,j in pairs})<3:return None
    if span > max(n+8, int(n*1.6)) or span < max(5,int(n*.60)):return None
    # Repeated phrases may share the same grade, but their location is ambiguous.
    ambiguous=any(j>=finish+max(3,n//2) for j in endpoints[1:])
    if swapped:
        return (0,n,begin,finish,[(i,j) for i,j in pairs],edits,ambiguous,[("missing_note" if k=="extra_note" else "extra_note" if k=="missing_note" else k,i,j) for k,i,j in operations])
    return (begin,finish,0,n,[(j,i) for i,j in pairs],edits,ambiguous,[(k,j,i) for k,i,j in operations])


def estimate_partial(score, audio, bpm=None):
    s=score or {}; a=audio or {}
    # Search farther into the longer stream while limiting the scored shorter excerpt.
    ref=list(s.get('notes') or [])[:1024]; played=list(a.get('notes') or [])[:1024]
    if len(ref)<=len(played):ref=ref[:256]
    else:played=played[:256]
    if any(type(n) is not int or not 21<=n<=108 for n in ref+played):return None
    aligned=_align_excerpt(ref,played)
    if aligned is None:return None
    rs,re,ps,pe,pairs,edits,ambiguous,operations=aligned
    x=ref[rs:re]; y=played[ps:pe]; matches=len(pairs)
    pitch=round(max(0,100*(1-edits/max(len(x),len(y)))),1)
    metrics={'notation_accuracy':pitch}
    intervals=[]; interval_positions=[]
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
    if not ambiguous and s.get('confidence')=='high' and valid_events(se,ref,'onset_beats','duration_beats') and valid_events(ae,played,'start','duration'):
        for (i,j),(i2,j2) in zip(pairs,pairs[1:]):
            if i2!=i+1 or j2!=j+1:continue
            if any(isinstance(e.get('confidence'),bool) or not isinstance(e.get('confidence'),(int,float)) or not math.isfinite(e['confidence']) or not .65<=e['confidence']<=1 for e in ae[j:j+2]):continue
            rb=se[i2]['onset_beats']-se[i]['onset_beats']; ps_delta=ae[j2]['start']-ae[j]['start']
            if rb>0 and ps_delta>.02:intervals.append((rb,ps_delta));interval_positions.append((i2,j2))
        if len(intervals)>=4:
            scale=median(p/r for r,p in intervals)
            metrics['rhythm']=round(100*(1-mean(min(1,abs(p/(r*scale)-1)) for r,p in intervals)),1)
            if isinstance(bpm,(int,float)) and not isinstance(bpm,bool) and math.isfinite(bpm) and 20<=bpm<=300:
                metrics['tempo']=round(max(0,100-200*abs((60/scale)/bpm-1)),1)
    errors=[]
    labels={'wrong_pitch':'Noto‘g‘ri tovush','missing_note':'Tushib qolgan nota','extra_note':'Ortiqcha tovush'}
    def note_name(m):return ('Do','Do♯','Re','Re♯','Mi','Fa','Fa♯','Sol','Sol♯','La','La♯','Si')[m%12]+str(m//12-1)
    for kind,i,j in operations:
        e={'type':kind,'category':labels[kind]}
        if i is not None:
            e.update(expected_midi=ref[i],reference_position=i+1,reference_note=i+1)
        if j is not None:
            e.update(performed_midi=played[j],performed_position=j+1,performed_note=j+1)
            if len(ae)>j and isinstance(ae[j],dict) and ae[j].get('midi')==played[j]:
                t=ae[j].get('start')
                if type(t) in (int,float) and math.isfinite(t) and t>=0:e['audio_time_seconds']=round(t,2)
        if kind=='wrong_pitch':
            delta=played[j]-ref[i];e['semitone_difference']=delta
            e['explanation']=f"{note_name(ref[i])} o‘rniga {note_name(played[j])}: {abs(delta)} yarim ton {'baland' if delta>0 else 'past'}."
            e['practice']=f"{note_name(ref[i])} tovushini qo‘shni notalar bilan sekin mashq qiling."
        elif kind=='missing_note':
            e['explanation']=f"{note_name(ref[i])} ijro ketma-ketligida topilmadi."
            e['practice']=f"{note_name(ref[i])} va yonidagi notalarning o‘tishini takrorlang."
        else:
            e['explanation']=f"{note_name(played[j])} moslashtirilgan notalar orasida qo‘shimcha tovush sifatida ajraldi."
            e['practice']='Shu o‘tishni sekin chalib, tovushlar ketma-ketligini tekshiring.'
        if ambiguous:
            e.pop('reference_note',None);e.pop('reference_position',None)
        errors.append(e)
    if 'rhythm' in metrics:
        for (rb,p),(i,j) in zip(intervals,interval_positions):
            deviation=p/(rb*scale)-1
            if abs(deviation)>.25:
                errors.append({'type':'rhythm_deviation','category':'Ritmik og‘ish','reference_note':i+1,'performed_note':j+1,'audio_time_seconds':round(ae[j]['start'],2),'explanation':f"Nota oralig‘i parchaning asosiy tempiga nisbatan {abs(deviation)*100:.0f}% {'uzunroq' if deviation>0 else 'qisqaroq'}.",'practice':'Nota oralig‘ini metronom bilan mashq qiling.'})
    intonation=[]
    for i,j in pairs:
        if j>=len(ae) or not isinstance(ae[j],dict):continue
        ev=ae[j]; cents=ev.get('pitch_deviation_cents'); frames=ev.get('pitch_measurement_frames'); conf=ev.get('confidence')
        if ev.get('midi')!=played[j] or type(cents) not in (int,float) or not math.isfinite(cents) or abs(cents)>50 or type(frames) is not int or frames<3 or type(conf) not in (int,float) or not .65<=conf<=1:continue
        intonation.append((i,j,cents))
    # A4=440 equal-tempered reference, not a claim about expressive tuning or instrument setup.
    if len(intonation)>=5 and len(intonation)>=math.ceil(matches*.5):
        metrics['intonation']=round(100*mean(max(0,1-abs(c)/50) for _,_,c in intonation),1)
        for i,j,c in intonation:
            if abs(c)>25:
                e={'type':'intonation_deviation','category':'Intonatsiya og‘ishi','performed_note':j+1,'explanation':f"{note_name(played[j])} tovushi A4=440 Hz etaloniga nisbatan {abs(c):.1f} sent {'baland' if c>0 else 'past'}.",'practice':'Cholg‘u sozini tekshirib, shu tovushni tyuner bilan mashq qiling.'}
                if not ambiguous:e['reference_note']=i+1
                if type(ae[j].get('start')) in (int,float) and math.isfinite(ae[j]['start']):e['audio_time_seconds']=round(ae[j]['start'],2)
                errors.append(e)
    counts={k:sum(e['type']==k for e in errors) for k in labels}
    weight=sum(_PARTIAL_WEIGHTS[k] for k in metrics)
    grade=round(sum(_PARTIAL_WEIGHTS[k]*v for k,v in metrics.items())/weight,1)
    return {'status':'estimated','scope':'observed_excerpt_only','partial_score_100':grade,'measured_weight_percent':weight,
            'criteria':{k:{'score_percent':metrics.get(k),'weight':w,'status':'estimated' if k in metrics else 'not_measured'} for k,w in _PARTIAL_WEIGHTS.items()},
            'reference_start':rs+1,'performed_start':ps+1,'reference_count':len(x),'performed_count':len(y),
            'compared_note_count':min(len(x),len(y)),'matched_anchor_notes':matches,'compared_intervals':len(intervals),
            'edit_count':edits,'error_counts':counts,'errors':errors,'intonation_note_count':len(intonation) if 'intonation' in metrics else 0,'observed_tempo_bpm':round(60/scale,1) if 'rhythm' in metrics else None,
            'professional_score_100':None,'technical_score_100':None,'alignment_ambiguous':ambiguous,'warning':'Taxminiy parcha bahosi, yakuniy professional baho emas.'+(' Bir nechta mos parcha bor; joylashuv noaniq.' if ambiguous else '')}


def _partial_report(score_data, audio_data, comparison, report, profile, score_type, stage, reason, bpm=None, observed=None):
    s=score_data or {}; a=audio_data or {}; sn=len(s.get("notes",[])); an=len(a.get("notes",[]))
    if not (sn or an):return None
    covered=[]
    if sn:covered.append(f"Nota faylidan {sn} ta tovush o‘qildi ({s.get('staff_system_count', 0)} ta nota satri).")
    if an:
        secs=a.get("analyzed_seconds",a.get("duration_seconds"))
        covered.append(f"Ijrodan {an} ta tovush ajratildi"+(f"; {secs:.1f} soniya o‘lchandi." if isinstance(secs,(int,float)) else "."))
    summary="Tahlil qilingan parcha. "+" ".join(covered)
    metrics={k:None for k in ("nota_ketma_ketligi","ritm","temp","intonatsiya","artikulyatsiya","ifodalilik")}
    issues=[]; plans=[]; teacher="Nota va ijroni solishtirish uchun ravshanroq parcha kerak."
    # Do not reuse whole-score errors for partial results: unheard tails are not mistakes.
    comparison=None
    if sn:plans.append({"minutes":5,"task":"O‘qilgan nota qismini asl partitura bilan tekshirish"})
    if an:plans.append({"minutes":5,"task":"Yozilgan ijro parchasini sekin tempda qayta tinglash"})
    if observed is not None:
        metrics_map={"notation_accuracy":"nota_ketma_ketligi","intonation":"intonatsiya","rhythm":"ritm","tempo":"temp","articulation":"artikulyatsiya","expression":"ifodalilik"}
        for k,item in observed["criteria"].items():
            metrics[metrics_map[k]]={"value_percent":item["score_percent"],"status":item["status"],"scope":"observed_excerpt_only"}
        issues=observed['errors']; counts=observed['error_counts']; exact=observed['matched_anchor_notes']; edit=observed['edit_count']
        summary=f"Tahlil qilingan parcha: {observed['partial_score_100']}/100 ball. {observed['reference_count']} ta nota va {observed['performed_count']} ta ijro tovushi solishtirildi; {exact} ta tovush mos keldi."
        teacher=f"Nota ketma-ketligida {observed['criteria']['notation_accuracy']['score_percent']}% moslik qayd etildi. "
        if edit:
            teacher+=f"Moslashtirishda {counts['wrong_pitch']} ta balandlik farqi, {counts['missing_note']} ta topilmagan nota va {counts['extra_note']} ta qo‘shimcha tovush qayd etildi. "
        else:teacher+="Moslashtirilgan tovushlar ketma-ketligida farq topilmadi. "
        if observed.get('alignment_ambiguous'):teacher+="Asarda o‘xshash takroriy parchalar bor; nota joylashuvini asl partituradan tekshiring. "
        if s.get('confidence')!='high':teacher+="Nota tanish natijasini asl partitura bilan solishtirish tavsiya etiladi. "
        if observed.get('observed_tempo_bpm') is not None:teacher+=f"Mos nota oraliqlaridagi ijro sur’ati {observed['observed_tempo_bpm']} BPM. "
        if observed['criteria']['rhythm']['score_percent'] is not None:teacher+=f"Ritmik moslik {observed['criteria']['rhythm']['score_percent']}%. "
        if observed['criteria']['intonation']['score_percent'] is not None:teacher+=f"A4=440 Hz etaloni bo‘yicha {observed['intonation_note_count']} ta tovushda intonatsiya mosligi {observed['criteria']['intonation']['score_percent']}%. "
        plans=[]
        if any(e['type']=='intonation_deviation' for e in issues):plans.append({'minutes':5,'task':'Cholg‘u sozini tekshirish va belgilangan tovushlarni tyuner bilan mashq qilish'})
        for category,task in [('wrong_pitch','Balandligi farqlangan notalarni alohida chalish'),('missing_note','Topilmagan notalar va qo‘shni o‘tishlarni takrorlash'),('extra_note','Qo‘shimcha tovushlar ajralgan o‘tishlarni nazorat qilish')]:
            if counts[category]:plans.append({'minutes':5,'task':task})
        if any(e['type']=='rhythm_deviation' for e in issues):plans.append({'minutes':5,'task':'Ritmik og‘ish qayd etilgan oraliqlarni metronom bilan takrorlash'})
        plans.append({'minutes':5,'task':'Moslashtirilgan parchani sekin tempda qayta ijro etish'})
        plans.append({'minutes':3,'task':'Shu parchani qayta yozib natijalarni solishtirish'})
        comparison={"reference_count":observed["reference_count"],"performed_count":observed["performed_count"],"compared_note_count":observed["compared_note_count"],"note_similarity_percent":observed["criteria"]["notation_accuracy"]["score_percent"],"comparison_status":"tentative","assessment_scope":"observed_excerpt_only","reference_start":observed["reference_start"],"performed_start":observed["performed_start"],"rhythm_scored":observed["criteria"]["rhythm"]["score_percent"] is not None,"detailed_feedback":{"correct_notes":exact,"errors":issues,"error_counts":counts}}
    guidance=instrument_guidance(profile)
    return {"analysis_status":"partial","completed_stages":[k for k,v in (("score",sn),("audio",an),("comparison",comparison is not None)) if v],
        "stopped_stage":stage,"stop_reason":reason,"score":s or None,"score_input_type":score_type,"audio":a or None,
        "comparison":comparison,"partial_grading":observed,"professional_report":{"title":"NOTA AI — ijroning pedagogik tahlili","status":"partial","overall_grade_100":None,"partial_grading":observed,
        "sections":{"umumiy_xulosa":summary,"baholash_mezonlari":metrics,"aniqlangan_xatolar":issues,"pedagog_xulosasi":teacher,"individual_mashgulot_rejasi":plans},
        "instrument_guidance":guidance,"measurement_limitations":["O‘qilmagan yoki solishtirilmagan qism bo‘yicha xulosa berilmadi."]},
        "selected_instrument":profile["id"] if profile else None,"instrument_guidance":guidance,"timed_analysis":None,
        "technical_grading":{"status":"insufficient_evidence","technical_score_100":None,"professional_score_100":None},
        "professional_grading":professional_grade({k:{"value":v["score_percent"],"source":"observed_excerpt_only","validated":False} for k,v in observed["criteria"].items()} if observed else {}),"limitations":["Qisman natija; yakuniy baho emas.","O‘lchovlar taxminiy va pedagog tekshiruvini talab qiladi."]}

async def _run_files(pdf_bytes,audio_bytes,score_name,audio_name,exercise,bpm,beats_per_measure,profile,emit):
    loop=asyncio.get_running_loop(); deadline=loop.time()+_ANALYSIS_SECONDS
    score_data=audio_data=comparison=report=score_type=None
    failures=[]
    cached_key=None; cached_grade=None
    def checkpoint_grade():
        nonlocal cached_key,cached_grade
        if not score_data or not audio_data:return None
        # Cache identical snapshots, while allowing a longer measured prefix to replace them.
        key=(score_data,audio_data)
        if cached_key is None or key[0] is not cached_key[0] or key[1] is not cached_key[1]:
            try:cached_grade=estimate_partial(score_data,audio_data,bpm)
            except Exception as exc:
                cached_grade=None;_LOG.warning("NOTA_PARTIAL grading %s",_safe_reason(exc))
            cached_key=key
        return cached_grade
    def progress(stage,reason="in_progress"):
        if emit:
            partial=_partial_report(score_data,audio_data,None,None,profile,score_type,stage,reason,bpm=bpm,observed=checkpoint_grade())
            if partial:emit(partial)
    def audio_checkpoint(data):
        nonlocal audio_data
        audio_data=data;progress("audio")
    def score_checkpoint(data):
        nonlocal score_data
        score_data=data;progress("score")
    # Audio is measured independently even if Gemini cannot read the first staff.
    try:
        if os.getenv("NOTA_LONG_AUDIO_ENABLED","false").lower()=="true":
            from functools import partial
            audio_data=await run_in_threadpool(partial(audio_to_notes,audio_bytes,audio_name,on_progress=lambda d:loop.call_soon_threadsafe(audio_checkpoint,d),deadline=min(deadline-10,loop.time()+105)))
        else:
            audio_data=await run_in_threadpool(audio_to_notes,audio_bytes,audio_name)
        if audio_data.get("incomplete"):failures.append(("audio",audio_data.get("stop_reason") or "audio_incomplete"))
        progress("score")
    except Exception as exc:
        failures.append(("audio",_safe_reason(exc)));_LOG.warning("NOTA_PARTIAL audio %s",failures[-1][1])
    try:
        if loop.time()>=deadline-5:raise TimeoutError()
        normalized_score,score_type=await run_in_threadpool(normalize_score,pdf_bytes,score_name)
        if os.getenv("NOTA_OMR_V3_ENABLED","false").lower()=="true":
            from score_omr_v3 import pdf_to_notes as pdf_to_notes_v3
            score_data=await pdf_to_notes_v3(normalized_score,exercise,on_progress=score_checkpoint,deadline=deadline-5)
        else:
            async with asyncio.timeout(max(.1,deadline-loop.time()-5)):
                score_data=await pdf_to_notes(normalized_score,exercise)
        if score_data.get("incomplete"):failures.append(("score",score_data.get("stop_reason") or "score_incomplete"))
        progress("comparison")
    except Exception as exc:
        failures.append(("score",_safe_reason(exc)));_LOG.warning("NOTA_PARTIAL score %s",failures[-1][1])
    # Incomplete transcriptions must never be compared as though they are full pieces.
    if score_data and audio_data:
        rn=len(score_data.get("notes") or []); pn=len(audio_data.get("notes") or [])
        if min(rn,pn)>0 and min(rn,pn)/max(rn,pn)<.8:
            observed=checkpoint_grade()
            _LOG.info("NOTA_EXCERPT_ALIGNMENT ref=%d audio=%d compared=%d estimated=%s",rn,pn,observed["compared_note_count"] if observed else 0,bool(observed))
            partial=_partial_report(score_data,audio_data,None,None,profile,score_type,"comparison","unequal_observed_coverage",bpm=bpm,observed=observed)
            if partial:
                partial["stage_failures"]=[{"stage":k,"reason":v} for k,v in failures]
                return partial
    if not failures and score_data and audio_data:
        try:
            if loop.time()>=deadline-5:raise TimeoutError()
            comparison=await run_in_threadpool(compare_notes,score_data["notes"],audio_data["notes"])
            progress("report")
            report=professional_report(comparison,score_data,audio_data)
            guidance=instrument_guidance(profile);report["instrument_guidance"]=guidance
            timed=None;grade={"status":"insufficient_evidence","technical_score_100":None,"professional_score_100":None}
            if bpm and score_data.get("events") and audio_data.get("events"):
                if loop.time()>=deadline-3:raise TimeoutError()
                reference=[Event(midi=e["midi"],start=e["onset_beats"]*60/bpm,duration=e["duration_beats"]*60/bpm) for e in score_data["events"]]
                performed=[Event(midi=e["midi"],start=e["start"],duration=e["duration"]) for e in audio_data["events"]]
                timed=await run_in_threadpool(analyze,reference,performed,60/bpm,beats_per_measure)
                grade=technical_score(timed,score_confidence=score_data["confidence"],audio_events=audio_data["events"])
                report["sections"]["baholash_mezonlari"]["ritm"]=timed["rhythm"]
                report["sections"]["baholash_mezonlari"]["temp"]=timed["tempo"]
                report["timed_errors"]=timed["note_errors"]
            # Transcription completion is not the same as a validated final grade.
            # Keep the observed estimate separately; never put it in a final-grade field.
            final_grade=grade.get("professional_score_100")
            if final_grade is None:final_grade=grade.get("technical_score_100")
            if not isinstance(final_grade,(int,float)) or isinstance(final_grade,bool) or not math.isfinite(final_grade):
                observed=checkpoint_grade()
                partial=_partial_report(score_data,audio_data,None,None,profile,score_type,"grading","final_grade_not_validated",bpm=bpm,observed=observed)
                if partial:
                    partial["stage_failures"]=[]
                    return partial
            return {"analysis_status":"complete","score":score_data,"score_input_type":score_type,"audio":audio_data,"comparison":comparison,"professional_report":report,"selected_instrument":profile["id"] if profile else None,"instrument_guidance":guidance,"timed_analysis":timed,"technical_grading":grade,"professional_grading":professional_grade({}),"limitations":["Monophonic melody only","No verified rhythm assessment","AI sheet-music extraction may be inaccurate","Instrument harmonics may affect pitch detection","Video is assessed from its extracted audio only","File accepted does not guarantee accurate transcription"]}
        except Exception as exc:
            failures.append(("report" if comparison is not None else "comparison",_safe_reason(exc)))
    observed=checkpoint_grade()
    stage,reason=failures[-1] if failures else ("analysis","no_evidence")
    partial=_partial_report(score_data,audio_data,None,None,profile,score_type,stage,reason,bpm=bpm,observed=observed)
    if partial:
        partial["stage_failures"]=[{"stage":k,"reason":v} for k,v in failures]
        return partial
    raise HTTPException(502,detail={"code":"analysis_no_evidence","stage":stage,"reason":reason})

@app.post("/v1/analyze-files")
async def analyze_files(score: UploadFile=File(...),audio: UploadFile=File(...),
                        authorization: Optional[str]=Header(None),exercise: Optional[int]=Form(None),
                        bpm: Optional[float]=Form(None),beats_per_measure: Optional[int]=Form(None),
                        instrument: Optional[str]=Form(None),x_nota_progress: Optional[str]=Header(None)):
    authorize(authorization)
    profile=get_instrument(instrument)
    if instrument is not None and profile is None:raise HTTPException(422,"Unknown instrument")
    if exercise is not None and not 1<=exercise<=99:raise HTTPException(422,"Exercise must be between 1 and 99")
    if bpm is not None and not 20<=bpm<=300:raise HTTPException(422,"BPM must be 20..300")
    if beats_per_measure is not None and not 1<=beats_per_measure<=12:raise HTTPException(422,"beats_per_measure must be 1..12")
    if _ANALYSIS_LOCK.locked():raise HTTPException(503,"analysis_busy_retry_later",headers={"Retry-After":"10"})
    # UploadFile lifetime is independent of the stream; copy bounded inputs before returning.
    pdf_bytes=await read_limited(score,MAX_PDF);audio_bytes=await read_limited(audio,MAX_AUDIO)
    args=(pdf_bytes,audio_bytes,score.filename,audio.filename,exercise,bpm,beats_per_measure,profile)
    if x_nota_progress!="ndjson-v1":
        if _ANALYSIS_LOCK.locked():raise HTTPException(503,"analysis_busy_retry_later")
        async with _ANALYSIS_LOCK:return await _run_files(*args,emit=None)
    async def stream():
        queue=asyncio.Queue(maxsize=1);last=None;finished=False
        def emit(data):
            nonlocal last
            if finished:return
            last=data
            if queue.full():queue.get_nowait()
            queue.put_nowait({"type":"checkpoint","data":data})
        async def produce():
            nonlocal finished
            try:
                if _ANALYSIS_LOCK.locked():raise HTTPException(503,"analysis_busy_retry_later")
                async with _ANALYSIS_LOCK:
                    result=await _run_files(*args,emit=emit)
                frame={"type":"final","data":result}
            except Exception as exc:
                frame={"type":"final","data":last} if last else {"type":"error","status":exc.status_code if isinstance(exc,HTTPException) else 502,"code":_safe_reason(exc)}
            finished=True
            if queue.full():queue.get_nowait()
            queue.put_nowait(frame)
        task=asyncio.create_task(produce())
        try:
            while True:
                try:frame=await asyncio.wait_for(queue.get(),timeout=12)
                except TimeoutError:frame={"type":"heartbeat"}
                yield json.dumps(frame,ensure_ascii=False,separators=(",",":"))+"\n"
                if frame["type"] in ("final","error"):break
        finally:
            if not task.done():task.cancel()
            await asyncio.gather(task,return_exceptions=True)
    return StreamingResponse(stream(),media_type="application/x-ndjson",headers={"Cache-Control":"no-store","X-Accel-Buffering":"no"})
