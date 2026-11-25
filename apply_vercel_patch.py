"""Guarded patch for restored Next.js source tree. Staging branch ONLY.

Does not change authentication, the credit reserve/commit logic, or production
configuration. Raises on missing/unexpected route source rather than creating
an incomplete replacement route.
"""
from __future__ import annotations
from pathlib import Path
import re
import sys

ROUTES = [
    ('app/api/nota-render/route.js', 60),
    ('app/api/credit-analysis/route.js', 120),
    ('app/api/score-compare/route.js', 60),
]


def locate(root: Path, relative: str) -> Path:
    for prefix in (root, root / 'src'):
        candidate = prefix / relative
        if candidate.is_file():
            return candidate
    raise RuntimeError(f'Missing existing Next.js source: {relative}. Do not overwrite original route.')


def patch(root: Path):
    root = Path(root)
    originals = {}
    updates = {}
    for relative, old in ROUTES:
        file = locate(root, relative)
        original = file.read_text(encoding='utf-8')
        expr = re.compile(r'export\s+const\s+maxDuration\s*=\s*(\d+)\s*;')
        values = expr.findall(original)
        if len(values) != 1 or int(values[0]) not in (old, 300):
            raise RuntimeError(f'Unexpected maxDuration for {file}: {values}')
        changed = expr.sub('export const maxDuration=300;', original, count=1)
        if relative.startswith('app/api/nota-render'):
            # These are verified exact patterns from the last deployed source.
            if 'list.slice(0,220)' in changed:
                changed = changed.replace('list.slice(0,220)', 'list.slice(0,12000)', 1)
            elif 'list.slice(0,12000)' not in changed:
                raise RuntimeError('Unexpected expectedNotes parser; no changes made')
            if 'list.length>300' in changed:
                changed = changed.replace('list.length>300', 'list.length>12000', 1)
            elif 'list.length>12000' not in changed:
                raise RuntimeError('Unexpected measuredNotes parser; no changes made')
        if relative.startswith('app/api/score-compare'):
            if '.slice(0,300)' in changed:
                changed = changed.replace('.slice(0,300)', '.slice(0,12000)', 1)
            elif '.slice(0,12000)' not in changed:
                raise RuntimeError('Unexpected score parser; no changes made')
        originals[file] = original
        updates[file] = changed
    changes = [(p, originals[p], updates[p]) for p in updates if originals[p] != updates[p]]
    if any(p.with_suffix(p.suffix + '.nota-backup').exists() for p,_,__ in changes):
        raise RuntimeError('Backup already exists; refusing to overwrite it')
    for p, old, new in changes:
        p.with_suffix(p.suffix + '.nota-backup').write_text(old, encoding='utf-8')
        p.write_text(new, encoding='utf-8')
    return [str(p) for p,_,__ in changes]


if __name__ == '__main__':
    root = Path(sys.argv[1] if len(sys.argv) > 1 else '.')
    changed = patch(root)
    print('Changed files:', ', '.join(changed) if changed else 'already patched')
    print('WARNING: 300s is a synchronous route limit, NOT a 15-minute queued analysis.')
    print('Run preview build, smoke tests, credit/idempotency tests before promotion.')
