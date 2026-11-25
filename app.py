import base64
import json
import math
import os
import subprocess
import tempfile
import io
from typing import Optional
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

app = FastAPI(title="NOTA AI Analysis", version="0.9.0")
MAX_PDF = (32 if os.getenv("NOTA_OMR_V3_ENABLED", "false").lower() == "true" else 8) * 1024 * 1024
MAX_AUDIO = 96 * 1024 * 1024
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif", ".tif", ".tiff", ".bmp"}
VIDEO_EXT = {".mp4", ".mov", ".webm", ".mkv", ".avi", ".m4v"}
AUDIO_EXT = {".wav", ".mp3", ".m4a", ".ogg", ".flac", ".webm", ".aac"}


def authorize(value):
    secret = os.getenv("NOTA_API_TOKEN")
    if not secret or value != "Bearer " + secret:
        raise HTTPException(401, "API token is missing or invalid")


class CompareRequest(BaseModel):
    reference_midi: list[int] = Field(min_length=1, max_length=1000)
    performed_hz: list[float] = Field(min_length=1, max_length=1000)


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
    return {"status": "ok", "version": "0.9.0",
            "pdf_ai_configured": bool(os.getenv("GEMINI_API_KEY"))}



class TimedNote(BaseModel):
    midi: int = Field(ge=21, le=108)
    start: float = Field(ge=0)
    duration: float = Field(gt=0)

class TimedAnalysisRequest(BaseModel):
    reference: list[TimedNote] = Field(min_length=1, max_length=500)
    performed: list[TimedNote] = Field(min_length=1, max_length=500)
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


@app.post("/v1/analyze-files")
async def analyze_files(score: UploadFile = File(...), audio: UploadFile = File(...),
                        authorization: Optional[str] = Header(None),
                        exercise: Optional[int] = Form(None),
                        bpm: Optional[float] = Form(None),
                        beats_per_measure: Optional[int] = Form(None),
                        instrument: Optional[str] = Form(None)):
    authorize(authorization)
    profile = get_instrument(instrument)
    if instrument is not None and profile is None:
        raise HTTPException(422, "Unknown instrument. See GET /v1/instruments")
    if exercise is not None and not 1 <= exercise <= 99:
        raise HTTPException(422, "Exercise must be between 1 and 99")
    pdf_bytes = await read_limited(score, MAX_PDF)
    audio_bytes = await read_limited(audio, MAX_AUDIO)
    normalized_score, score_type = normalize_score(pdf_bytes, score.filename)
    if os.getenv("NOTA_OMR_V3_ENABLED", "false").lower() == "true":
        from score_omr_v3 import pdf_to_notes as pdf_to_notes_v3
        score_data = await pdf_to_notes_v3(normalized_score, exercise)
    else:
        score_data = await pdf_to_notes(normalized_score, exercise)
    audio_data = audio_to_notes(audio_bytes, audio.filename)
    result = compare_notes(score_data["notes"], audio_data["notes"])
    report = professional_report(result, score_data, audio_data)
    guidance = instrument_guidance(profile)
    report["instrument_guidance"] = guidance
    timed=None
    grade={"status":"insufficient_evidence","technical_score_100":None,"professional_score_100":None}
    if bpm is not None and not 20 <= bpm <= 300:
        raise HTTPException(422,"BPM must be 20..300")
    if beats_per_measure is not None and not 1 <= beats_per_measure <= 12:
        raise HTTPException(422,"beats_per_measure must be 1..12")
    if bpm and score_data.get("events") and audio_data.get("events"):
        try:
            reference=[Event(midi=e["midi"],start=e["onset_beats"]*60/bpm,
                             duration=e["duration_beats"]*60/bpm) for e in score_data["events"]]
            performed=[Event(midi=e["midi"],start=e["start"],duration=e["duration"])
                       for e in audio_data["events"]]
            timed=analyze(reference,performed,60/bpm,beats_per_measure)
            grade=technical_score(timed,score_confidence=score_data["confidence"],
                                  audio_events=audio_data["events"])
        except ValueError:
            pass
    if timed is not None:
        report["sections"]["baholash_mezonlari"]["ritm"] = timed["rhythm"]
        report["sections"]["baholash_mezonlari"]["temp"] = timed["tempo"]
        report["timed_errors"] = timed["note_errors"]
        report["sections"]["umumiy_xulosa"] += (
            " Ritm va temp natijalari dastlabki hisoblangan baholar bo‘lib, "
            "professional tekshiruvdan o‘tmagan."
        )
    return {"score": score_data, "score_input_type": score_type, "audio": audio_data,
            "comparison": result, "professional_report": report,
            "selected_instrument": profile["id"] if profile else None,
            "instrument_guidance": guidance,
            "timed_analysis":timed,"technical_grading":grade,
            "professional_grading":professional_grade({}),
            "limitations": ["Monophonic melody only", "No verified rhythm assessment",
                            "AI sheet-music extraction may be inaccurate",
                            "Instrument harmonics may affect pitch detection",
                            "Video is assessed from its extracted audio only",
                            "File accepted does not guarantee accurate transcription"]}
