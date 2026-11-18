#!/usr/bin/env python3
"""Safely patch NOTA AI Render app.py with Gemini -> OpenRouter -> Groq OMR fallback.

Run from the Render project root: python nota_ai_render_failover_patch.py
Makes app.py.backup-before-ai-failover, verifies the exact function boundaries,
and compiles the resulting module without executing provider requests.
"""
from pathlib import Path
import ast
import shutil
import sys

NEW_FUNCTION = '''async def pdf_to_notes(data, exercise=None):
    """Transcribe a rendered music score with bounded sequential AI fallback.

    Only the first successful schema-validated reading is returned. AI never
    assigns a performance grade. Provider errors never leak tokens or images.
    """
    import logging
    import re

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
        images = []
        for page in doc:
            pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5), alpha=False)
            images.append(base64.b64encode(pix.tobytes("png")).decode("ascii"))
        doc.close()
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(422, f"Could not read PDF: {type(exc).__name__}")

    providers = []
    if os.getenv("GEMINI_API_KEY"):
        providers.append(("gemini", os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
                          os.getenv("GEMINI_API_KEY")))
    if os.getenv("OPENROUTER_API_KEY"):
        providers.append(("openrouter", os.getenv("OPENROUTER_VISION_MODEL", "qwen/qwen3-vl-30b-a3b-instruct"),
                          os.getenv("OPENROUTER_API_KEY")))
    if os.getenv("GROQ_API_KEY"):
        providers.append(("groq", os.getenv("GROQ_VISION_MODEL", "qwen/qwen3.8-27b"),
                          os.getenv("GROQ_API_KEY")))
    if not providers:
        raise HTTPException(503, "No AI notation provider is configured")

    def validate(text):
        if not isinstance(text, str):
            raise ValueError("Not a text response")
        stripped = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            found = re.search(r"\\{[\\s\\S]*\\}", stripped)
            if not found:
                raise ValueError("No JSON object")
            parsed = json.loads(found.group(0))
        if not isinstance(parsed, dict):
            raise ValueError("Invalid JSON shape")
        raw = parsed.get("notes", [])
        if not isinstance(raw, list) or len(raw) > 300:
            raise ValueError("Invalid notes list")
        notes = [x for x in raw if type(x) is int and 21 <= x <= 108]
        if len(notes) != len(raw) or len(notes) < 3:
            raise ValueError("Insufficient reliable notes")
        confidence = parsed.get("confidence", "unknown")
        if str(confidence).strip().lower() == "low":
            raise ValueError("Low reading confidence")
        events = []
        raw_events = parsed.get("events", [])
        if isinstance(raw_events, list):
            for e in raw_events[:300]:
                if not isinstance(e, dict):
                    continue
                midi, onset, dur = e.get("midi"), e.get("onset_beats"), e.get("duration_beats")
                if (type(midi) is int and 21 <= midi <= 108
                    and type(onset) in (int, float) and math.isfinite(onset)
                    and type(dur) in (int, float) and math.isfinite(dur)
                    and onset >= 0 and 0 < dur <= 32):
                    events.append({"midi": midi, "onset_beats": round(onset, 4),
                                   "duration_beats": round(dur, 4)})
        if len(events) != len(notes) or [e["midi"] for e in events] != notes:
            events = []
        return {"notes": notes, "events": events, "confidence": confidence,
                "timing_status": "estimated_from_omr" if events else "not_available"}

    logger = logging.getLogger("nota.ai")
    for provider, model, key in providers:
        # Groq vision supports at most three images per call. Never skip pages.
        if provider == "groq" and len(images) > 3:
            logger.warning("notation_fallback_skipped provider=groq reason=page_limit")
            continue
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                if provider == "gemini":
                    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
                    parts = [{"text": prompt}] + [{"inline_data": {"mime_type": "image/png", "data": im}} for im in images]
                    response = await client.post(url, params={"key": key}, json={
                        "contents": [{"parts": parts}],
                        "generationConfig": {"responseMimeType": "application/json", "temperature": 0,
                                             "maxOutputTokens": 5000}})
                    if response.status_code != 200:
                        raise ValueError(f"HTTP_{response.status_code}")
                    data_json = response.json()
                    content = data_json["candidates"][0]["content"]["parts"]
                    output = "".join(part.get("text", "") for part in content)
                else:
                    url = ("https://openrouter.ai/api/v1/chat/completions" if provider == "openrouter"
                           else "https://api.groq.com/openai/v1/chat/completions")
                    parts = [{"type": "text", "text": prompt}] + [
                        {"type": "image_url", "image_url": {"url": "data:image/png;base64," + im}}
                        for im in images]
                    response = await client.post(url, headers={"Authorization": "Bearer " + key}, json={
                        "model": model, "messages": [{"role": "user", "content": parts}],
                        "temperature": 0, "max_tokens": 5000})
                    if response.status_code != 200:
                        raise ValueError(f"HTTP_{response.status_code}")
                    output = response.json()["choices"][0]["message"]["content"]
                result = validate(output)
                result.update({"ai_provider": provider, "ai_model": model})
                return result
        except (httpx.HTTPError, ValueError, KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
            # Never log the upstream response body, URL query parameters or credentials.
            logger.warning("notation_provider_failed provider=%s error_type=%s",
                           provider, type(exc).__name__)
            continue
    raise HTTPException(502, "All configured AI notation providers failed; no result was fabricated")
'''


def patch(path: Path) -> None:
    original = path.read_text(encoding="utf-8")
    start = original.find("async def pdf_to_notes(")
    end = original.find("\ndef audio_to_notes(", start)
    if start < 0 or end < 0 or original.count("async def pdf_to_notes(") != 1:
        raise ValueError("Unexpected app.py version; no changes made")
    if '"ai_provider": provider' in original[start:end] or "All configured AI notation providers failed" in original:
        raise ValueError("Failover patch appears to be installed already; no changes made")
    revised = original[:start] + NEW_FUNCTION + "\n" + original[end:]
    ast.parse(revised, filename=str(path))
    backup = path.with_name(path.name + ".backup-before-ai-failover")
    if backup.exists():
        raise ValueError(f"Backup already exists: {backup}; no changes made")
    shutil.copy2(path, backup)
    path.write_text(revised, encoding="utf-8")
    print("PATCH_OK: Python syntax verified. Backup:", backup)
    print("NEXT: run tests and deploy only after checking API credentials and integration.")


if __name__ == "__main__":
    filename = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("app.py")
    patch(filename)
