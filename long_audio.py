"""NOTA AI long audio analyzer -- staging-safe drop-in module.

Processes up to 15 minutes (configurable downwards), in 30-second chunks
with overlap and absolute timestamps. Analysis remains monophonic / estimated.
Designed to be enabled by NOTA_LONG_AUDIO_ENABLED in app.py.
"""
from __future__ import annotations

import math
import os
import subprocess
import tempfile
import wave
from pathlib import Path

import numpy as np
from fastapi import HTTPException

# This module is installed alongside the existing NOTA AI backend.
from advanced_analysis import extract_events

_SUPPORTED = {".wav", ".mp3", ".m4a", ".ogg", ".flac", ".webm", ".aac",
              ".mp4", ".mov", ".mkv", ".avi", ".m4v"}
_MAX_SECONDS = 900  # 15 minutes, direct HTTP request mode only
_MAX_INPUT_BYTES = 96 * 1024 * 1024
_RATE = 16_000
_CHUNK_SECONDS = 30
_OVERLAP_SECONDS = .35


def _allowed_seconds() -> int:
    try:
        seconds = int(os.getenv("NOTA_MAX_AUDIO_SECONDS", "900"))
    except ValueError:
        seconds = _MAX_SECONDS
    return max(30, min(seconds, _MAX_SECONDS))


def _add_clipped_event(result: list[dict], item: dict, offset: float,
                       core_start: float, core_end: float) -> None:
    """Clip overlapping detections to the nonoverlapping core, then stitch them."""
    try:
        pitch = int(item["midi"])
        start = offset + float(item["start"])
        end = start + float(item["duration"])
    except (KeyError, ValueError, TypeError, OverflowError):
        return
    if not (21 <= pitch <= 108) or not (math.isfinite(start) and math.isfinite(end)):
        return
    start = max(start, core_start)
    end = min(end, core_end)
    if end - start < .035:
        return
    confidence = float(item.get("confidence", 0))
    confidence = confidence if math.isfinite(confidence) else 0.0
    confidence = max(0.0, min(1.0, confidence))
    if result and result[-1]["midi"] == pitch:
        previous = result[-1]
        previous_end = previous["start"] + previous["duration"]
        # Consecutive identical pitches at a segment boundary represent one note.
        if (core_start > 0 and previous_end >= core_start - .13
                and start <= core_start + .12 and start <= previous_end + .095):
            previous["duration"] = round(max(previous_end, end) - previous["start"], 3)
            previous["confidence"] = round(min(previous["confidence"], confidence), 3)
            return
    result.append({"midi": pitch, "start": round(start, 3),
                   "duration": round(end - start, 3), "confidence": round(confidence, 3)})


def audio_to_notes(raw: bytes, filename: str | None) -> dict:
    """Decode bounded audio to mono 16k WAV, stream 30s overlapping chunks.

    Long files are never silently truncated; requests exceeding 15 minutes
    receive HTTP 413. Automatic chunk advancing is internal and needs no user
    intervention. This does not implement a durable background queue.
    """
    ext = Path(filename or "").suffix.lower()
    if ext not in _SUPPORTED:
        raise HTTPException(status_code=415, detail="Unsupported audio/video format")
    if not raw or len(raw) > _MAX_INPUT_BYTES:
        raise HTTPException(status_code=413, detail="Audio file is empty or exceeds 96 MB")
    limit = _allowed_seconds()
    with tempfile.TemporaryDirectory(prefix="nota_long_audio_") as tmp:
        source = Path(tmp) / ("input" + ext)
        target = Path(tmp) / "normalized.wav"
        source.write_bytes(raw)
        command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
                   "-i", str(source), "-map", "0:a:0", "-vn", "-sn", "-dn",
                   "-t", str(limit + 1), "-ac", "1", "-ar", str(_RATE),
                   "-c:a", "pcm_s16le", str(target)]
        try:
            proc = subprocess.run(command, capture_output=True, timeout=100, check=False)
        except FileNotFoundError as ex:
            raise HTTPException(status_code=503, detail="FFmpeg audio decoder unavailable") from ex
        except subprocess.TimeoutExpired as ex:
            raise HTTPException(status_code=504, detail="Audio decoding exceeded time budget") from ex
        if proc.returncode != 0 or not target.exists():
            raise HTTPException(status_code=422, detail="Audio could not be decoded")

        events: list[dict] = []
        analyzed_chunks = 0
        with wave.open(str(target), "rb") as decoded:
            if decoded.getnchannels() != 1 or decoded.getsampwidth() != 2 or decoded.getframerate() != _RATE:
                raise HTTPException(status_code=422, detail="Unexpected decoded audio format")
            total = decoded.getnframes()
            if total < _RATE:
                raise HTTPException(status_code=422, detail="Audio is too short")
            if total > limit * _RATE + 20:
                raise HTTPException(status_code=413, detail=f"Audio exceeds {limit} seconds; use a queued long-analysis job")
            for core_left in range(0, total, _CHUNK_SECONDS * _RATE):
                core_right = min(total, core_left + _CHUNK_SECONDS * _RATE)
                pad = round(_OVERLAP_SECONDS * _RATE)
                read_left = max(0, core_left - pad)
                read_right = min(total, core_right + pad)
                decoded.setpos(read_left)
                data = decoded.readframes(read_right - read_left)
                if not data:
                    continue
                chunk = np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0
                # Per-chunk normalization is deliberately omitted to retain
                # consistent silence thresholds and performance dynamics.
                segment = extract_events(chunk, _RATE)
                for item in segment:
                    _add_clipped_event(events, item, read_left / _RATE,
                                       core_left / _RATE, core_right / _RATE)
                analyzed_chunks += 1
            duration = round(total / _RATE, 3)

    if not events:
        raise HTTPException(status_code=422, detail="No stable pitched sound detected")
    # Every detected event remains available to the alignment stage. Never
    # silently discard the second part of a long performance.
    pitches = [e["midi"] for e in events]
    return {
        "notes": pitches,
        "events": events,
        "timing_status": "estimated_from_monophonic_audio",
        "duration_seconds": duration,
        "method": "overlapping 30-second autocorrelation chunks (monophonic estimate)",
        "chunks_analyzed": analyzed_chunks,
        "max_audio_seconds": limit,
        "total_detected_events": len(events),
        "note_sequence_capped": False,
        "warnings": [],
    }
