import unittest
from timing_analysis import Event, analyze
from professional_grading import grade, WEIGHTS

class Validation(unittest.TestCase):
    def test_perfect_sequence(self):
        e=[Event(60+i,i*.5,.4) for i in range(8)]
        r=analyze(e,e,.5,4)
        self.assertEqual(r['pitch_similarity_percent'],100)
        self.assertEqual(r['rhythm']['relative_onset_accuracy_percent'],100)
        self.assertEqual(r['tempo']['estimated_bpm'],120)
        self.assertEqual(r['note_errors'],[])
    def test_wrong_pitch_at_time(self):
        ref=[Event(60+i,i*.5,.4) for i in range(8)]
        played=[Event(60+i+(1 if i==3 else 0),i*.5,.4) for i in range(8)]
        r=analyze(ref,played,.5,4)
        self.assertTrue(any(e['kind']=='wrong_pitch' and e['audio_time_seconds']==1.5 for e in r['note_errors']))
    def test_tempo_slower(self):
        ref=[Event(60+i,i*.5,.4) for i in range(8)]
        played=[Event(60+i,i*.6,.48) for i in range(8)]
        self.assertAlmostEqual(analyze(ref,played,.5,4)['tempo']['estimated_bpm'],100,delta=.1)
    def test_grade_refuses_missing_evidence(self):
        self.assertIsNone(grade({})['professional_score_100'])
    def test_grade_with_verified_metrics(self):
        metrics={k:{'value':80,'source':'teacher-verified measurement','validated':True} for k in WEIGHTS}
        self.assertEqual(grade(metrics,validated=True)['professional_score_100'],80)
    def test_grade_rejects_unverified(self):
        metrics={k:{'value':80,'source':'estimated','validated':False} for k in WEIGHTS}
        self.assertIsNone(grade(metrics,validated=True)['professional_score_100'])

if __name__=='__main__':unittest.main()
