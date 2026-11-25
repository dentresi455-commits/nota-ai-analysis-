import sys
from pathlib import Path
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'backend'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'vercel'))
from apply_backend_patch import patch as backend_patch
from apply_vercel_patch import patch as vercel_patch


def test_backend_patcher(tmp_path):
    s='''import os\nMAX_AUDIO=96 * 1024 * 1024\n\ndef compare_notes(expected, actual):\n    return None\n\n@app.get("/health")\ndef health():pass\n\ndef audio_to_notes(raw, filename):\n    return None\n\n@app.get("/v1/instruments")\ndef list_instruments():pass\n'''
    (tmp_path/'app.py').write_text(s)
    (tmp_path/'advanced_analysis.py').write_text('def extract_events(samples,rate):\n    events=[]\n    return events[:500]\n')
    for f in ['long_audio.py','long_alignment.py','score_omr_v3.py','photo_preprocessing.py']:
        (tmp_path/f).write_text('x=1')
    assert set(backend_patch(tmp_path))=={'app.py','advanced_analysis.py'}
    assert backend_patch(tmp_path)==[]
    assert 'return events\n' in (tmp_path/'advanced_analysis.py').read_text()
    assert 'NOTA_FULL_ALIGNMENT_ENABLED' in (tmp_path/'app.py').read_text()


def test_vercel_patcher(tmp_path):
    routes=['nota-render','credit-analysis','score-compare']
    bodies=["export const runtime='nodejs';export const maxDuration=60;\nconst a=list.slice(0,220);const b=list.length>300;",
            "export const runtime='nodejs';\nexport const maxDuration=120;",
            "export const runtime='nodejs';export const maxDuration=60;\nconst notes=a.slice(0,300);"]
    for n,s in zip(routes,bodies):
        p=tmp_path/'src'/'app'/'api'/n/'route.js';p.parent.mkdir(parents=True,exist_ok=True);p.write_text(s)
    assert len(vercel_patch(tmp_path))==3
    assert vercel_patch(tmp_path)==[]
    assert 'maxDuration=300' in (tmp_path/'src/app/api/credit-analysis/route.js').read_text()
    assert 'list.length>12000' in (tmp_path/'src/app/api/nota-render/route.js').read_text()


def test_vercel_refuses_missing_routes_without_mutation(tmp_path):
    p=tmp_path/'src/app/api/nota-render/route.js';p.parent.mkdir(parents=True);p.write_text('export const maxDuration=60;')
    with pytest.raises(RuntimeError):vercel_patch(tmp_path)
    assert p.read_text()=='export const maxDuration=60;'
