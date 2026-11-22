"""NOTA AI photo-aware score OMR v3 (staging candidate).

A backwards-compatible, page-wise optical music recognition reader.
No scores are invented, and uncertain scans are flagged for manual review.
It is NOT a complete background-job queue, nor a verified music transcription model.
"""
from __future__ import annotations

import base64
import io
import json
import math
import os
from typing import Any

import fitz
from photo_preprocessing import prepare_pdf_page, crop_systems
import httpx
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from fastapi import HTTPException

MAX_PAGES = 48
MAX_NOTES = 12000
MAX_PAGE_PIXELS = 7_000_000
MAX_PDF_BYTES = 32 * 1024 * 1024
MAX_PAGE_AI_ATTEMPTS = 2
MAX_SYSTEMS_PER_PAGE = 12

PROMPT = """You are a careful optical music transcription specialist.
Transcribe ONE page of a conventional printed music score from left to right,
system by system, in written order. Only transcribe the principal monophonic
melody; do not invent illegible notes. Pay attention to clefs, key signatures,
accidentals that persist through a bar, octave positions, rests, note durations,
dots, ties, pickup bars, barlines and time signatures. Tied notes must be
combined, not counted as an extra attack. Return JSON only:
{"events":[{"midi":60,"onset_beats":0,"duration_beats":1,"bar":1}],
 "confidence":"high", "needs_review":false, "warnings":[]}
MIDI 21-108. Onset and duration are measured in quarter-note beats.
The first notated attack on THIS page is onset 0; use positions relative to
THIS page; bar numbers start at 1 on each page. If rhythmic values are not
readable but note pitches are, return {"notes":[60,62],"events":[],
"confidence":"low","needs_review":true,"warnings":["rhythm_unreadable"]}.
For unreadable pages return empty notes/events and confidence low.
Do NOT guess the notes. If this is tablature, alternative tuning or multi-voice
notation rather than a standard monophonic five-line staff, set needs_review true.
If a tie continues across the page break, include "page_boundary_tie" in warnings.
For DUTOR notation with a conventional five-line staff: notes are placed on the
staff; the printed digits 1, 2, 3 and technique symbols ABOVE the notes are
FINGERINGS/ARTICULATION, NOT pitches or notes. Do not transcribe finger numbers
as pitch values or extra notes. Ignore pen circles and handwritten annotations.
If two notes form a chord, do not silently invent a single melodic note;
set needs_review true and warn "polyphony" or "chord_not_supported".
If the score has repeats, da capo, dal segno or volta markers, do NOT expand
them without structural verification; record a warning "repeat_structure".
Do not infer actual concert tuning of a transposing dutor from the staff alone.

"""


def _image_from_page(page: fitz.Page) -> Image.Image:
    # Render above former 1.5x raster quality while bounding memory.
    scale = min(2.4, math.sqrt(MAX_PAGE_PIXELS / max(1.0, page.rect.width * page.rect.height)))
    scale = max(1.0, scale)
    pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False, colorspace=fitz.csRGB)
    img = Image.open(io.BytesIO(pix.tobytes("png")))
    return img.convert("RGB")


def _diagnose_image(img: Image.Image) -> list[str]:
    """Heuristic warnings; these do not imply a fabricated confidence score."""
    gray = ImageOps.grayscale(img)
    small = gray.copy()
    small.thumbnail((1000, 1200))
    sample = np.asarray(small, dtype=np.float32)
    warnings = []
    if min(img.size) < 900:
        warnings.append("low_resolution")
    if sample.shape[0] > 3 and sample.shape[1] > 3:
        gradient = np.abs(sample[1:] - sample[:-1]).mean() + np.abs(sample[:, 1:] - sample[:, :-1]).mean()
        if gradient < 4.0:
            warnings.append("low_detail_or_blurry_scan")
    if (sample < 80).mean() < .001:
        warnings.append("few_visible_symbols")
    return warnings


def _deskew_if_justified(img: Image.Image) -> Image.Image:
    """Conservative horizontal-staff deskew for slightly tilted camera images.

    Only changes orientation if the darkness projection of the aligned page
    improves markedly. Does not fix perspective distortion or curled paper.
    """
    test = ImageOps.grayscale(img)
    test.thumbnail((850, 1100))
    def concentration(view: Image.Image) -> float:
        array = np.asarray(view, dtype=np.float32)
        # Avoid border artifacts, especially white wedges after rotation.
        h, w = array.shape
        middle = array[h//12:h*11//12, w//12:w*11//12]
        rows = (middle < 155).mean(axis=1)
        # Concentrated dark horizontal staff lines yield peaked row sums.
        return float(np.square(rows).sum())
    baseline = concentration(test)
    best_angle, best_score = 0.0, baseline
    if baseline < 0.1:
        return img
    for angle in (-2.5,-2.0,-1.5,-1.0,-0.5,0.5,1.0,1.5,2.0,2.5):
        rotated = test.rotate(angle, resample=Image.Resampling.BICUBIC, fillcolor=255)
        candidate = concentration(rotated)
        if candidate > best_score:
            best_score, best_angle = candidate, angle
    if best_angle and best_score > baseline * 1.13:
        return img.rotate(best_angle, resample=Image.Resampling.BICUBIC, expand=True, fillcolor=(255,255,255))
    return img


def _improve_image(img: Image.Image) -> Image.Image:
    """Non-destructive enhancement used on retry, original is first pass."""
    # autocontrast often improves gray/yellowish smartphone photographs
    img = _deskew_if_justified(img)
    gray = ImageOps.autocontrast(ImageOps.grayscale(img), cutoff=1)
    gray = ImageEnhance.Contrast(gray).enhance(1.25)
    gray = gray.filter(ImageFilter.UnsharpMask(radius=1.25, percent=145, threshold=3))
    return gray.convert("RGB")


def _to_png_base64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _valid_midi(value: Any) -> bool:
    return type(value) is int and 21 <= value <= 108


def _validated_page(data: Any, page_number: int) -> dict:
    if not isinstance(data, dict):
        raise ValueError("not_json_object")
    events_raw = data.get("events", [])
    if not isinstance(events_raw, list):
        raise ValueError("events_not_array")
    events = []
    for e in events_raw:
        if not isinstance(e, dict):
            raise ValueError("event_not_object")
        midi, onset, duration, bar = e.get("midi"), e.get("onset_beats"), e.get("duration_beats"), e.get("bar", 1)
        if (not _valid_midi(midi) or type(onset) not in (float, int)
            or type(duration) not in (float, int) or type(bar) is not int
            or not math.isfinite(onset) or not math.isfinite(duration)
            or onset < 0 or duration <= 0 or duration > 32 or not 1 <= bar <= 500):
            raise ValueError("invalid_event")
        events.append({"midi":midi,"onset_beats":float(onset),"duration_beats":float(duration),"bar":bar})
    if any(events[i]["onset_beats"] < events[i - 1]["onset_beats"] for i in range(1, len(events))):
        raise ValueError("unsorted_events")
    raw_notes = data.get("notes", None)
    if raw_notes is None:
        notes = [e["midi"] for e in events]
    elif isinstance(raw_notes, list) and all(_valid_midi(n) for n in raw_notes):
        notes = raw_notes
    else:
        raise ValueError("invalid_notes")
    if events and notes != [e["midi"] for e in events]:
        raise ValueError("pitches_do_not_match_events")
    if not notes:
        raise ValueError("no_reliable_notes")
    confidence = str(data.get("confidence", "low")).lower()
    if confidence not in {"high", "medium", "low"}:
        confidence = "low"
    warnings = data.get("warnings", [])
    if not isinstance(warnings, list):
        warnings = []
    warnings = [str(x)[:100] for x in warnings[:12]]
    needs_review = bool(data.get("needs_review", False)) or confidence != "high" or not events
    return {"page":page_number,"notes":notes,"events":events,"confidence":confidence,
            "needs_review":needs_review,"warnings":warnings}


async def _gemini_page(img: Image.Image, *, page_number: int, exercise: int | None,
                       client: httpx.AsyncClient, alternate: Image.Image | None = None,
                       require_second_pass: bool = False, system: int | None = None) -> dict:
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        raise HTTPException(status_code=503, detail="GEMINI_API_KEY is not configured")
    model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    instruction = PROMPT
    if system is not None:
        instruction += f"\nTranscribe ONLY printed staff system {system} in this cropped image."
    if exercise is not None:
        instruction += f"\nOnly exercise number {exercise}; ignore other exercises."
    last_reason = "invalid_response"
    first_valid = None
    for attempt in range(MAX_PAGE_AI_ATTEMPTS):
        render = img if attempt == 0 else (alternate if alternate is not None else _improve_image(img))
        payload = {
            "contents":[{"parts":[
                {"text": instruction},
                {"inline_data":{"mime_type":"image/png","data":_to_png_base64(render)}}
            ]}],
            "generationConfig":{"temperature":0,"responseMimeType":"application/json", "maxOutputTokens":8192}
        }
        try:
            response = await client.post(url, headers={"x-goog-api-key": key}, json=payload)
            if response.status_code in (408,429,500,502,503,504):
                last_reason = f"provider_{response.status_code}"
                continue
            if response.status_code != 200:
                raise HTTPException(status_code=502, detail=f"OMR provider error: HTTP {response.status_code} on page {page_number}")
            body = response.json()
            response_text = "".join(p.get("text", "") for p in body["candidates"][0]["content"]["parts"])
            try:
                obj = json.loads(response_text)
            except json.JSONDecodeError:
                clean = response_text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
                obj = json.loads(clean)
            result = _validated_page(obj, page_number)
            if first_valid is None:
                first_valid = result
                if (result["confidence"] == "high" and not result["needs_review"]
                    and not require_second_pass):
                    return result
                # A medium/low-confidence first reading gets a second enhanced scan.
                continue
            if (result["notes"] != first_valid["notes"] or
                result.get("events") != first_valid.get("events")):
                first_valid["warnings"].append("omr_pass_disagreement")
                first_valid["needs_review"] = True
                first_valid["confidence"] = "low"
                return first_valid
            # Two passes agree on pitches, but their confidence remains an estimate.
            chosen = result if result["confidence"] == "high" and result["events"] else first_valid
            chosen["warnings"].append("confirmed_pitch_sequence_by_second_pass")
            chosen["needs_review"] = bool(first_valid["needs_review"] or result["needs_review"])
            return chosen
        except HTTPException:
            if first_valid is None:
                raise
            last_reason = "provider_error_on_recheck"
        except (httpx.TimeoutException, httpx.TransportError):
            last_reason = "provider_timeout_or_network"
        except (json.JSONDecodeError, ValueError, KeyError, IndexError, TypeError):
            last_reason = "invalid_response"
    if first_valid is not None:
        first_valid["warnings"].append("recheck_unavailable_" + last_reason)
        first_valid["needs_review"] = True
        return first_valid
    raise HTTPException(status_code=502, detail=f"OMR failed on page {page_number}: {last_reason}")



def _join_pages(pages: list[dict]) -> dict:
    """Join page-relative timing; mark ambiguous boundaries for review."""
    if not pages:
        raise ValueError("no_pages")
    notes: list[int] = []
    events: list[dict] = []
    metadata: list[dict] = []
    beat_offset = 0.0
    bar_offset = 0
    timing_complete = True
    for p in pages:
        page_notes = p["notes"]
        if len(notes) + len(page_notes) > MAX_NOTES:
            raise HTTPException(status_code=413, detail="Too many notes; use smaller batches")
        notes.extend(page_notes)
        warnings = list(p.get("warnings", []))
        if not p["events"]:
            timing_complete = False
        else:
            for e in p["events"]:
                events.append({"midi": e["midi"], "onset_beats": round(beat_offset + e["onset_beats"],4),
                               "duration_beats": round(e["duration_beats"],4), "bar":bar_offset + e["bar"]})
            beat_offset += max(e["onset_beats"]+e["duration_beats"] for e in p["events"])
            bar_offset += max(e["bar"] for e in p["events"])
        if p["page"] > 1:
            warnings.append("review_cross_page_bar_or_tie")
        metadata.append({"page":p["page"],"note_count":len(page_notes),"confidence":p["confidence"],
                         "needs_review":p["needs_review"] or bool(warnings),"warnings":warnings})
    confidence = "high" if all(p["confidence"]=="high" and not p["needs_review"] and not p["warnings"] for p in pages) else "low"
    # Notation OMR is estimated, never independent teacher-validated evidence.
    return {"notes":notes,"events":events if timing_complete else [],
            "confidence":confidence,
            "timing_status":"estimated_from_omr" if timing_complete else "not_available",
            "pages":metadata,"page_count":len({p["page"] for p in pages}),
            "needs_review":any(p["needs_review"] for p in metadata),
            "transcription_status":"estimated_unverified"}


async def pdf_to_notes(data: bytes, exercise: int | None = None) -> dict:
    """Photo-aware, system-aware score OMR.

    Intended for preview with bounded short requests; long scores must use
    a durable job worker. Each system carries an explicit review flag.
    """
    if not isinstance(data, bytes) or len(data) > MAX_PDF_BYTES:
        raise HTTPException(status_code=413, detail="PDF too large")
    if not data.startswith(b"%PDF"):
        raise HTTPException(status_code=415, detail="PDF expected")
    try:
        doc = fitz.open(stream=data, filetype="pdf")
    except Exception as ex:
        raise HTTPException(status_code=422, detail="Invalid PDF") from ex
    try:
        if doc.needs_pass:
            raise HTTPException(status_code=422, detail="Encrypted PDF unsupported")
        if not 1 <= len(doc) <= MAX_PAGES:
            raise HTTPException(status_code=413, detail=f"PDF must have 1..{MAX_PAGES} pages")
        chunks = []
        summaries = []
        # These sequential calls are bounded locally for staging.  In a durable
        # queue, each chunk can be retried independently and checkpointed.
        async with httpx.AsyncClient(timeout=httpx.Timeout(45.0, connect=10.0)) as client:
            for page_idx in range(len(doc)):
                prepared = prepare_pdf_page(doc[page_idx])
                original_crops = crop_systems(prepared, "original")
                cleaned_crops = crop_systems(prepared, "ink_suppressed")
                if len(original_crops)>MAX_SYSTEMS_PER_PAGE:
                    raise HTTPException(status_code=413, detail="Too many staff systems on a page")
                page_warnings=list(prepared.warnings)
                page_chunks = []
                for system_idx, (original_crop, cleaned_crop) in enumerate(zip(original_crops,cleaned_crops), 1):
                    p = await _gemini_page(
                        original_crop, page_number=page_idx+1, exercise=exercise,
                        client=client,
                        alternate=cleaned_crop if "colored_handwritten_annotations_detected" in page_warnings else None,
                        require_second_pass="colored_handwritten_annotations_detected" in page_warnings,
                        system=system_idx if len(original_crops)>1 else None,
                    )
                    p["system"] = system_idx
                    p["warnings"] = list(dict.fromkeys([*p.get("warnings",[]),*page_warnings]))
                    p["needs_review"] = p["needs_review"] or bool(page_warnings)
                    if system_idx>1:
                        p["warnings"].append("review_cross_system_bar_or_tie")
                        p["needs_review"] = True
                    page_chunks.append(p)
                    chunks.append(p)
                summaries.append({
                    "page":page_idx+1,
                    "systems":len(original_crops),
                    "orientation_degrees_ccw":prepared.orientation_degrees,
                    "rotation_confidence":prepared.rotation_confidence,
                    "warnings":page_warnings,
                    "annotation_fraction":round(prepared.masks.get("annotation_fraction",0),6)
                })
        combined=_join_pages(chunks)
        combined["pages"]=summaries
        combined["staff_system_count"]=len(chunks)
        combined["omr_engine"]="photo-aware-system-v3-staging"
        combined["needs_review"] = True  # Never treat AI transcription as ground truth.
        combined["transcription_status"]="estimated_unverified"
        return combined
    finally:
        doc.close()
