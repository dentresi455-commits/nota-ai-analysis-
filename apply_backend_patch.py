"""Safe staging integration for the current NOTA AI analysis backend.

Run in the existing nota-ai-analysis- working tree alongside app.py.
Creates backups, validates exact source markers, and never edits production
unless the operator deliberately runs it there. Defaults opt-in flags to OFF.
"""
from __future__ import annotations
import ast
from pathlib import Path
import shutil
import sys


def patch(root: Path):
    root = Path(root)
    path = root / "app.py"
    advanced = root / "advanced_analysis.py"
    if not path.is_file() or not advanced.is_file():
        raise RuntimeError("Missing app.py or advanced_analysis.py")
    src = path.read_text(encoding="utf-8")
    adv = advanced.read_text(encoding="utf-8")
    # On the main branch both limits are still 24/8 MiB. Increase them only
    # while the corresponding feature is explicitly enabled. Staging already
    # has a 96 MiB audio limit, so leave that setting untouched.
    if 'MAX_AUDIO = 24 * 1024 * 1024' in src:
        src = src.replace(
            'MAX_AUDIO = 24 * 1024 * 1024',
            'MAX_AUDIO = (96 if os.getenv("NOTA_LONG_AUDIO_ENABLED", "false").lower() == "true" else 24) * 1024 * 1024', 1)
    if 'MAX_PDF = 8 * 1024 * 1024' in src:
        src = src.replace(
            'MAX_PDF = 8 * 1024 * 1024',
            'MAX_PDF = (32 if os.getenv("NOTA_OMR_V3_ENABLED", "false").lower() == "true" else 8) * 1024 * 1024', 1)
    # The scripts are intended for precisely the inspected integration points.
    # Allow re-running, but never blindly manipulate a different codebase.
    aud_marker = '\n\n@app.get("/v1/instruments")'
    compare_marker = '\n\n@app.get("/health")'
    addition = '''\n\n# NOTA long audio: explicit staging opt-in; fallback remains untouched.
if os.getenv("NOTA_LONG_AUDIO_ENABLED", "false").lower() == "true":
    from long_audio import audio_to_notes
'''
    alignment = '''\n\n# Exact, memory-bounded whole-sequence alignment, staging opt-in.
if os.getenv("NOTA_FULL_ALIGNMENT_ENABLED", "false").lower() == "true":
    from long_alignment import compare_complete as compare_notes
'''
    if 'from long_audio import audio_to_notes' not in src:
        if src.count(aud_marker) != 1 or 'def audio_to_notes(raw, filename):' not in src:
            raise RuntimeError('Unknown app.py audio structure; no changes made')
        src = src.replace(aud_marker, addition + aud_marker, 1)
    if 'from long_alignment import compare_complete as compare_notes' not in src:
        if src.count(compare_marker) != 1 or 'def compare_notes(expected, actual):' not in src:
            raise RuntimeError('Unknown app.py comparison structure; no changes made')
        src = src.replace(compare_marker, alignment + compare_marker, 1)
    old_advanced = 'return events[:500]'
    if old_advanced in adv:
        if adv.count(old_advanced) != 1:
            raise RuntimeError('Ambiguous 500-event limit; no changes made')
        adv = adv.replace(old_advanced, 'return events', 1)
    elif 'return events\n' not in adv:
        raise RuntimeError('Unknown advanced_analysis.py; no changes made')
    # Validate before touching disk.
    ast.parse(src, filename='app.py')
    ast.parse(adv, filename='advanced_analysis.py')
    needed = ['long_audio.py', 'long_alignment.py', 'score_omr_v3.py', 'photo_preprocessing.py']
    if not all((root / p).is_file() for p in needed):
        raise RuntimeError('Copy the four backend files to the project root first')
    modified = []
    for p, updated in [(path, src), (advanced, adv)]:
        original = p.read_text(encoding='utf-8')
        if original != updated:
            backup = p.with_name(p.name + '.nota-backup')
            if backup.exists():
                raise RuntimeError(f'Refusing to overwrite existing backup: {backup}')
            modified.append((p, backup, original, updated))
    for p, backup, original, new in modified:
        backup.write_text(original, encoding='utf-8')
        p.write_text(new, encoding='utf-8')
    return [str(x[0].name) for x in modified]


if __name__ == '__main__':
    project = Path(sys.argv[1] if len(sys.argv) > 1 else '.')
    changed = patch(project)
    print('Staging patch:', ', '.join(changed) if changed else 'already applied')
    print('Enable staging only: NOTA_LONG_AUDIO_ENABLED=true, NOTA_FULL_ALIGNMENT_ENABLED=true')
