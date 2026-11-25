"""NOTA AI photo-aware score OMR v3 (staging candidate).

A backwards-compatible, page-wise optical music recognition reader.
No scores are invented, and uncertain scans are flagged for manual review.
It is NOT a complete background-job queue, nor a verified music transcription model.
"""
from __future__ import annotations

import asyncio
import base64
import io
import json
import math
import os
import re
import random
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

# Bounded synchronous preview. Large scores need a durable background job queue.
# Do not increase the timeout beyond the calling frontend/proxy's limit.
OMR_TOTAL_TIMEOUT_SECONDS = 52
OMR_REQUEST_TIMEOUT_SECONDS = 20
OMR_CONCURRENT_SYSTEMS = max(1, min(3, int(os.getenv("NOTA_OMR_CONCURRENT_SYSTEMS", "2"))))

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


# NOTA_OMR_FALLBACK_V1: opt-in third-party vision fallback for repeated Gemini 5xx.
# Third-party requests MUST be explicitly enabled in Render staging. Provider
# keys and model IDs must be configured there; no credentials are embedded here.
async def _optional_vision_fallback(img: Image.Image, instruction: str,
                                    page_number: int, client: httpx.AsyncClient,
                                    failed_reason: str) -> dict | None:
    if os.getenv("NOTA_ENABLE_VISION_FALLBACK", "false").lower() != "true":
        return None
    data_url = "data:image/png;base64," + _to_png_base64(img)
    content = [
        {"type": "text", "text": instruction},
        {"type": "image_url", "image_url": {"url": data_url}},
    ]
    providers = [
        ("openrouter", "OPENROUTER_API_KEY", "NOTA_OPENROUTER_VISION_MODEL",
         "https://openrouter.ai/api/v1/chat/completions"),
        ("groq", "GROQ_API_KEY", "NOTA_GROQ_VISION_MODEL",
         "https://api.groq.com/openai/v1/chat/completions"),
    ]
    for name, key_env, model_env, url in providers:
        api_key, model = os.getenv(key_env, "").strip(), os.getenv(model_env, "").strip()
        # Never guess a provider/model.  Skip unavailable or improperly configured.
        if not api_key or not model or len(model) > 150 or not re.fullmatch(r"[a-zA-Z0-9_.:/-]+", model):
            continue
        try:
            response = await client.post(
                url,
                headers={"Authorization": "Bearer " + api_key,
                         "Content-Type": "application/json"},
                json={"model": model,
                      "messages": [{"role": "user", "content": content}],
                      "temperature": 0, "max_tokens": 4096},
                timeout=15,
            )
            # Fail closed for auth, unsupported model, quota or provider outage.
            if response.status_code != 200:
                continue
            payload = response.json()
            message = payload["choices"][0]["message"]["content"]
            if isinstance(message, list):
                message = "".join(p.get("text", "") for p in message if isinstance(p, dict))
            if not isinstance(message, str):
                continue
            clean = message.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            result = _validated_page(json.loads(clean), page_number)
            result["confidence"] = "low"
            result["needs_review"] = True
            result["warnings"] = list(dict.fromkeys([
                *result["warnings"], "vision_fallback_used_" + name,
                "gemini_retryable_failure", "fallback_requires_teacher_review",
            ]))
            return result
        except (httpx.HTTPError, ValueError, KeyError, TypeError, IndexError):
            continue
    return None


async def _gemini_page(img: Image.Image, *, page_number: int, exercise: int | None,
                       client: httpx.AsyncClient, alternate: Image.Image | None = None,
                       require_second_pass: bool = False, system: int | None = None) -> dict:
    """Read one cropped staff; provider errors are explicit and never become scores."""
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        raise HTTPException(status_code=503, detail="gemini_key_missing")
    model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip()
    if not re.fullmatch(r"gemini-[a-zA-Z0-9_.-]+", model):
        raise HTTPException(status_code=503, detail="gemini_model_invalid")
    # Use an explicit, opt-in backup model when an upstream overload occurs.
    # Do not rotate models on bad credentials or exhausted quota.
    backup_model = os.getenv("GEMINI_BACKUP_MODEL", "").strip()
    if backup_model and not re.fullmatch(r"gemini-[a-zA-Z0-9_.-]+", backup_model):
        raise HTTPException(status_code=503, detail="gemini_backup_model_invalid")
    overloaded = False
    instruction = PROMPT
    if system is not None:
        instruction += f"\nTranscribe ONLY printed staff system {system} in this cropped image."
    if exercise is not None:
        instruction += f"\nOnly exercise number {exercise}; ignore other exercises."
    last_reason = "invalid_response"
    retryable_upstream_failure = False
    first_valid = None
    retry_happened = False

    for attempt in range(MAX_PAGE_AI_ATTEMPTS):
        active_model = backup_model if overloaded and backup_model else model
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{active_model}:generateContent"
        render = img if attempt == 0 else (alternate if alternate is not None else _improve_image(img))
        payload = {
            "contents": [{"parts": [
                {"text": instruction},
                {"inline_data": {"mime_type": "image/png", "data": _to_png_base64(render)}}
            ]}],
            "generationConfig": {"temperature": 0, "responseMimeType": "application/json", "maxOutputTokens": 4096}
        }
        try:
            response = await client.post(url, headers={"x-goog-api-key": key}, json=payload)
            if response.status_code in (401, 403):
                raise HTTPException(status_code=502, detail="gemini_key_not_authorized")
            if response.status_code == 404:
                raise HTTPException(status_code=502, detail="gemini_model_not_found")
            if response.status_code == 429:
                # Avoid burning quota with an immediate storm of retries.
                raise HTTPException(status_code=503, detail="gemini_rate_limited")
            if response.status_code in (408, 500, 502, 503, 504):
                last_reason = f"gemini_upstream_{response.status_code}"
                retryable_upstream_failure = True
                retry_happened = True
                overloaded = response.status_code in (500, 502, 503, 504)
                if attempt + 1 < MAX_PAGE_AI_ATTEMPTS:
                    # Backoff with jitter; stay inside the caller's hard deadline.
                    await asyncio.sleep(min(4.0, 1.2 * (2 ** attempt)) + random.random() * 0.25)
                continue
            if response.status_code != 200:
                raise HTTPException(status_code=502, detail=f"gemini_unexpected_http_{response.status_code}")
            body = response.json()
            response_text = "".join(p.get("text", "") for p in body["candidates"][0]["content"]["parts"])
            try:
                obj = json.loads(response_text)
            except json.JSONDecodeError:
                clean = response_text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
                obj = json.loads(clean)
            result = _validated_page(obj, page_number)
            if retry_happened:
                result["warnings"].append("gemini_retry_recovered")
                result["needs_review"] = True
                result["confidence"] = "low"
            if first_valid is None:
                first_valid = result
                if result["confidence"] == "high" and not result["needs_review"]:
                    if require_second_pass:
                        # Manual annotations need review, but never force a
                        # duplicate model call when the first reading is clear.
                        result["warnings"].append("annotation_recheck_deferred")
                        result["needs_review"] = True
                        result["confidence"] = "medium"
                    return result
                continue
            if (result["notes"] != first_valid["notes"] or
                result.get("events") != first_valid.get("events")):
                first_valid["warnings"].append("omr_pass_disagreement")
                first_valid["needs_review"] = True
                first_valid["confidence"] = "low"
                return first_valid
            chosen = result if result["confidence"] == "high" and result["events"] else first_valid
            chosen["warnings"].append("confirmed_pitch_sequence_by_second_pass")
            chosen["needs_review"] = bool(first_valid["needs_review"] or result["needs_review"])
            return chosen
        except HTTPException:
            # Invalid credentials/model/quota are fatal: do not suppress them.
            raise
        except (httpx.TimeoutException, httpx.TransportError):
            last_reason = "gemini_timeout_or_network"
            retryable_upstream_failure = True
            retry_happened = True
        except (json.JSONDecodeError, ValueError, KeyError, IndexError, TypeError):
            last_reason = "gemini_invalid_response"
            retry_happened = True

    if first_valid is not None:
        first_valid["warnings"].append("recheck_unavailable_" + last_reason)
        first_valid["needs_review"] = True
        first_valid["confidence"] = "low"
        return first_valid
    if retryable_upstream_failure:
        backup_result = await _optional_vision_fallback(
            img, instruction, page_number, client, last_reason)
        if backup_result is not None:
            return backup_result
    raise HTTPException(status_code=502, detail=f"gemini_omr_failed_{last_reason}")


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
    """Bounded, concurrent per-system OMR for synchronous staging requests.

    Per-page order is preserved, even when Gemini calls finish out of order.
    Any failed system rejects the complete comparison (never fabricate notes
    or award a score from an incomplete transcription). Long music scores
    require a durable queue/worker, not a longer HTTP timeout.
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

        chunks: list[dict] = []
        summaries: list[dict] = []
        limit = asyncio.Semaphore(OMR_CONCURRENT_SYSTEMS)
        max_conn = httpx.Limits(max_connections=OMR_CONCURRENT_SYSTEMS,
                                max_keepalive_connections=OMR_CONCURRENT_SYSTEMS)
        request_timeout = httpx.Timeout(OMR_REQUEST_TIMEOUT_SECONDS, connect=8.0)

        async def read_one(*, original_crop, cleaned_crop, page_idx, system_idx,
                           page_warnings, total_systems, client):
            async with limit:
                p = await _gemini_page(
                    original_crop, page_number=page_idx + 1, exercise=exercise,
                    client=client,
                    alternate=cleaned_crop if "colored_handwritten_annotations_detected" in page_warnings else None,
                    require_second_pass="colored_handwritten_annotations_detected" in page_warnings,
                    system=system_idx if total_systems > 1 else None,
                )
            p["system"] = system_idx
            p["warnings"] = list(dict.fromkeys([*p.get("warnings", []), *page_warnings]))
            p["needs_review"] = p["needs_review"] or bool(page_warnings)
            if system_idx > 1:
                p["warnings"].append("review_cross_system_bar_or_tie")
                p["needs_review"] = True
            return p

        try:
            async with asyncio.timeout(OMR_TOTAL_TIMEOUT_SECONDS):
                async with httpx.AsyncClient(timeout=request_timeout, limits=max_conn) as client:
                    for page_idx in range(len(doc)):
                        prepared = prepare_pdf_page(doc[page_idx])
                        original_crops = crop_systems(prepared, "original")
                        cleaned_crops = crop_systems(prepared, "ink_suppressed")
                        if len(original_crops) != len(cleaned_crops):
                            raise HTTPException(status_code=422, detail="omr_staff_crop_mismatch")
                        if not original_crops:
                            raise HTTPException(status_code=422, detail="omr_no_staff_detected")
                        if len(original_crops) > MAX_SYSTEMS_PER_PAGE:
                            raise HTTPException(status_code=413, detail="Too many staff systems on a page")
                        page_warnings = list(prepared.warnings)
                        jobs = [
                            read_one(
                                original_crop=img, cleaned_crop=cleaned_crops[idx],
                                page_idx=page_idx, system_idx=idx + 1,
                                page_warnings=page_warnings, total_systems=len(original_crops),
                                client=client,
                            )
                            for idx, img in enumerate(original_crops)
                        ]
                        # Stop remaining provider requests on the first failed staff.
                        # This avoids exhausting a free Gemini quota after a 429.
                        tasks = [asyncio.create_task(job) for job in jobs]
                        try:
                            pending = set(tasks)
                            while pending:
                                done, pending = await asyncio.wait(
                                    pending, return_when=asyncio.FIRST_COMPLETED,
                                )
                                for task in done:
                                    error = task.exception()
                                    if error is not None:
                                        idx = tasks.index(task)
                                        if isinstance(error, HTTPException):
                                            code, status = str(error.detail), error.status_code
                                        else:
                                            code, status = "omr_system_failed", 502
                                        raise HTTPException(
                                            status_code=status,
                                            detail={"code": code, "page": page_idx + 1,
                                                    "system": idx + 1,
                                                    "retryable": status in (502, 503, 504)},
                                        ) from error
                            # Await all tasks in *input* order, never in completion order.
                            chunks.extend([task.result() for task in tasks])
                        finally:
                            for task in tasks:
                                if not task.done():
                                    task.cancel()
                            await asyncio.gather(*tasks, return_exceptions=True)
                        summaries.append({
                            "page": page_idx + 1,
                            "systems": len(original_crops),
                            "orientation_degrees_ccw": prepared.orientation_degrees,
                            "rotation_confidence": prepared.rotation_confidence,
                            "warnings": page_warnings,
                            "annotation_fraction": round(prepared.masks.get("annotation_fraction", 0), 6),
                        })
        except TimeoutError:
            raise HTTPException(status_code=504, detail={
                "code": "omr_time_budget_exceeded", "retryable": True,
                "message": "Try a shorter section; long scores require queued analysis",
            }) from None

        combined = _join_pages(chunks)
        combined["pages"] = summaries
        combined["staff_system_count"] = len(chunks)
        combined["omr_engine"] = "photo-aware-system-v3-staging"
        combined["omr_version"] = "3.1"
        combined["needs_review"] = True
        combined["transcription_status"] = "estimated_unverified"
        return combined
    finally:
        doc.close()
