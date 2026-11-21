"""Opt-in integration helper for NOTA AI FastAPI Render app.

Runs only against a checked-out app.py on a staging branch. Creates backup.
Changes one import and one call site; leaves legacy OMR in place for rollback.
Does not deploy and does not modify DB, tokens, or provider configuration.
"""
from __future__ import annotations
from pathlib import Path
import argparse

IMPORT = 'from score_omr_v3 import pdf_to_notes as pdf_to_notes_v3'
OLD_CALL = 'score_data = await pdf_to_notes(normalized_score, exercise)'
NEW_CALL = '''if os.getenv("NOTA_OMR_V3_ENABLED", "false").lower() == "true":
        from score_omr_v3 import pdf_to_notes as pdf_to_notes_v3
        score_data = await pdf_to_notes_v3(normalized_score, exercise)
    else:
        score_data = await pdf_to_notes(normalized_score, exercise)'''

def apply_patch(path:Path)->str:
    source=path.read_text(encoding='utf-8')
    if NEW_CALL in source:
        return 'already_applied'
    if source.count(OLD_CALL)!=1:
        raise RuntimeError('Unexpected app.py: integration call site not found exactly once')
    if not ('import os' in source and 'async def pdf_to_notes(' in source):
        raise RuntimeError('Unexpected app.py: expected imports or legacy reader missing')
    backup=path.with_suffix('.py.omr-v3-backup')
    if backup.exists():
        raise RuntimeError('Backup already exists: do not overwrite')
    # Feature-flag guarded lazy import: old code path remains loadable even if
    # the new OpenCV optional dependency is not yet installed.
    updated=source.replace(OLD_CALL,NEW_CALL,1)
    compile(updated,str(path),'exec')
    backup.write_text(source,encoding='utf-8')
    path.write_text(updated,encoding='utf-8')
    return 'applied'

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('app',type=Path,help='Path to Render app.py in staging checkout')
    args=parser.parse_args()
    print(apply_patch(args.app))
