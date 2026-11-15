# NOTA AI — Render AI analysis service (prototype)

This is a standalone prototype for comparing **already-extracted** MIDI note sequences with detected audio frequencies. It does NOT yet read PDF scores or audio files, and it does NOT assess rhythm.

Deploy to Render from a new GitHub repository. Render blueprint: `render.yaml`.

- `GET /health` — service health
- `POST /v1/compare` — authenticated comparison (`Authorization: Bearer <NOTA_API_TOKEN>`)

Example body:
```json
{"reference_midi":[60,62,64],"performed_hz":[261.63,293.66,329.63]}
```

Do not put the API token in browser JavaScript. Call from the Next.js server. Future work: real PDF optical music recognition, robust audio pitch/onset extraction, timing alignment, validation against labelled test fixtures.
