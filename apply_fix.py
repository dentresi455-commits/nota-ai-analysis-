"""Apply to v0.9.3 app.py in a test checkout, alongside partial_grading.py."""
import ast
from pathlib import Path
import sys

path=Path(sys.argv[1] if len(sys.argv)>1 else 'app.py')
s=path.read_text(encoding='utf-8')
if 'from partial_grading import estimate as estimate_partial' in s:raise SystemExit('Already patched')
changes=[
('from professional_grading import grade as professional_grade','from professional_grading import grade as professional_grade\nfrom partial_grading import estimate as estimate_partial'),
('def _partial_report(score_data, audio_data, comparison, report, profile, score_type, stage, reason):','def _partial_report(score_data, audio_data, comparison, report, profile, score_type, stage, reason, bpm=None, observed=None):'),
('    guidance=instrument_guidance(profile)\n    return {"analysis_status":"partial"', '''    if observed is not None:
        metrics_map={"notation_accuracy":"nota_ketma_ketligi","intonation":"intonatsiya","rhythm":"ritm","tempo":"temp","articulation":"artikulyatsiya","expression":"ifodalilik"}
        for k, item in observed["criteria"].items():
            metrics[metrics_map[k]]={"value_percent":item["score_percent"],"status":item["status"],"scope":"observed_excerpt_only"}
        summary=f"Qisman tahlil. {' '.join(covered)} Moslashtirilgan parchada taxminiy baho {observed['partial_score_100']}/100. Mezonlar vaznining {observed['measured_weight_percent']} foizi o‘lchandi; bu asar qamrovi yoki yakuniy baho emas."
        issues=[]
        teacher="Faqat moslashtirilgan parcha bo‘yicha taxminiy baho. O‘qilmagan qismga va o‘lchanmagan mezonlarga baho berilmadi."
        plans=[{"minutes":5,"task":"Moslashtirilgan parchani asl nota bilan tekshirish"},{"minutes":5,"task":"Parchani sekin tempda takrorlash"},{"minutes":5,"task":"Qolgan qismni alohida tekshirish"}]
        if comparison is None:
            comparison={"reference_count":observed["reference_count"],"performed_count":observed["performed_count"],"compared_note_count":observed["compared_note_count"],"note_similarity_percent":observed["criteria"]["notation_accuracy"]["score_percent"],"comparison_status":"tentative","assessment_scope":"observed_excerpt_only","reference_start":observed["reference_start"],"performed_start":observed["performed_start"],"rhythm_scored":observed["criteria"]["rhythm"]["score_percent"] is not None,"detailed_feedback":{"correct_notes":observed["matched_anchor_notes"],"errors":[],"error_counts":{}}}
    guidance=instrument_guidance(profile)
    return {"analysis_status":"partial"'''),
('"comparison":comparison,"professional_report":{"title":"NOTA AI — ijroning pedagogik tahlili","status":"partial","overall_grade_100":None,','"comparison":comparison,"partial_grading":observed,"professional_report":{"title":"NOTA AI — ijroning pedagogik tahlili","status":"partial","overall_grade_100":None,"partial_grading":observed,'),
('    stage,reason=failures[-1] if failures else ("analysis","no_evidence")\n    partial=_partial_report(score_data,audio_data,comparison,report,profile,score_type,stage,reason)', '''    observed=None
    if score_data and audio_data:
        try:
            observed=await run_in_threadpool(estimate_partial,score_data,audio_data,bpm)
        except Exception as exc:
            _LOG.warning("NOTA_PARTIAL grading %s",_safe_reason(exc))
    stage,reason=failures[-1] if failures else ("analysis","no_evidence")
    partial=_partial_report(score_data,audio_data,comparison,report,profile,score_type,stage,reason,bpm=bpm,observed=observed)''')]
for old,new in changes:
    if s.count(old)!=1:raise SystemExit('Unexpected source; stopped without writing: '+old[:80])
    s=s.replace(old,new,1)
ast.parse(s)
backup=path.with_suffix('.py.v093.bak')
if backup.exists():raise SystemExit('Backup already exists; stopped')
backup.write_bytes(path.read_bytes())
path.write_text(s,encoding='utf-8')
print('Patched, syntax checked; run integration tests before deploy.')
