# NOTA AI – PDF + Audio Analysis API (prototype v0.2)

## Render configuration
Runtime: Python, build: `pip install -r requirements.txt`, start: `uvicorn app:app --host 0.0.0.0 --port $PORT`.
Set **NOTA_API_TOKEN** to a long random secret and **GEMINI_API_KEY** to your Google Gemini API key using Render Environment. Optional **GEMINI_MODEL** defaults to `gemini-2.5-flash`; choose a model available to your API key. Never put keys in GitHub. `.python-version` pins Python 3.11 for binary-wheel compatibility.

Audio conversion requires `ffmpeg` executable on the host. If Render Python runtime does not provide it, use a Docker image with ffmpeg installed or add a supported build installation step. `/health` reports server and Gemini-key configuration only, not ffmpeg availability.

## Endpoints
- `GET /health` public health check
- `POST /v1/compare` JSON `{ "reference_midi": [60,62,64], "performed_hz": [261.63,293.66,329.63] }`
- `POST /v1/analyze-files` multipart fields `score` (PDF, max 8 MB, max 4 pages), `audio` (WAV/MP3/M4A/OGG/FLAC/WEBM, max 12 MB, first 90 seconds)

Both POST endpoints require `Authorization: Bearer YOUR_NOTA_API_TOKEN`.

## Important limitations
This is an experimental monophonic melody comparator, not a complete/validated music-performance grader. Gemini reads sheet-music images; AI may misread notes. Audio uses a basic FFT dominant-frequency tracker and may confuse overtones. Timing/rhythm, polyphony, expressive phrasing and instrument-specific evaluation are **not** scored. Score is an approximate note-sequence match, not a validated musical grade.

For a reliable production release, validate against annotated test scores and recordings, improve onset/pitch extraction, add rhythm alignment, and test runtime memory, latency, ffmpeg, and Gemini availability.
