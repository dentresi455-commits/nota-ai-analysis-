# NOTA AI v0.8.0 — evidence-bound analysis

## Implemented
- Image/PDF score normalization and Gemini-based single-melody recognition (not independently validated).
- Audio/video conversion via ffmpeg and basic monophonic pitch/onset segmentation.
- Alignment of pitch, relative rhythm and tempo when note timings and BPM are available.
- Pedagogical report includes timed errors and preliminary rhythm/tempo values when available.
- Professional 100-point grade remains `null` without independently validated metrics for all six categories. A technical score is not a professional score.

## Testing
Run `python -m unittest discover -p 'test_validation.py' -v`. Six deterministic tests cover perfect/wrong notes, timing, and evidence gating.

## Deployment
This package is NOT deployed. Test in Render staging, set NOTA_API_TOKEN and GEMINI_API_KEY, install ffmpeg, verify /health and /v1/analyze-files, then integrate professional_report into the existing Vercel AI analysis menu. Do not overwrite production until end-to-end tests pass.

## Remaining prerequisites for credible professional assessment
1. Benchmark real score PDFs against manually annotated MusicXML or note-event references.
2. Benchmark real instrument recordings with labeled pitch/onset/duration ground truth; add polyphonic transcription for piano and other multi-voice music.
3. Implement instrument-specific articulation, dynamics and expression measurements; compare with multiple teachers' ratings and calibrate score weights.
4. Integrate Vercel frontend with the Render API and test file types, authorization, failures, and result rendering.

This release is an incremental tested foundation, NOT a validated professional assessor.


## v0.9.0: 15 ta cholg‘u profili
- GET /v1/instruments: 15 ta cholg‘u nomi va tahlil cheklovlari.
- POST /v1/analyze-files: optional `instrument` form field (e.g. `dutor`, `piano`, `doira`).
- Response: `selected_instrument`, `instrument_guidance` and report guidance.
- Cholg‘uga mos pedagogik yo‘nalishlar qo‘shildi; **15 ta maxsus AI model hali o‘qitilmagan va validatsiya qilinmagan**.
- Bir ovozli algoritm akkordli fortepiano, gitara, chang va doira uchun professional ball bermaydi.
- Frontend cholg‘u tanlash menyusi uchun `/v1/instruments` dan foydalanishi kerak; Vercel hali o‘zgartirilmagan.
- Renderga chiqarishdan oldin staging sinovi va ffmpeg/Gemini tekshiruvi talab etiladi.
