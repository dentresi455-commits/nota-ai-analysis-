import unittest
from partial_grading import estimate

class Tests(unittest.TestCase):
    def setUp(self):
        self.notes=[60,62,64,65,67,69,71,72,74,76,77,79,81,83,84,86,88,89,91,93,95,96,98,100,101,103,105,107,108,106,104,102,99,97,94,92,90,87,85,82,80,78,75,73]
    def test_44_33(self):
        r=estimate({'notes':self.notes},{'notes':self.notes[:33]});self.assertEqual(r['compared_note_count'],33);self.assertEqual(r['partial_score_100'],100);self.assertEqual(r['measured_weight_percent'],25);self.assertIsNone(r['professional_score_100'])
    def test_wrong_inside(self):
        p=self.notes[:33];p[12]=60;r=estimate({'notes':self.notes},{'notes':p});self.assertLess(r['partial_score_100'],100)
    def test_empty(self): self.assertIsNone(estimate({},{}))
    def test_short(self): self.assertIsNone(estimate({'notes':[60,62]},{'notes':[60,62]}))
    def test_unrelated(self): self.assertIsNone(estimate({'notes':self.notes},{'notes':[21,22,23,24,25,26]}))
    def test_repeated(self): self.assertIsNone(estimate({'notes':[60]*20},{'notes':[60]*10}))
    def test_bad_note(self): self.assertIsNone(estimate({'notes':[True]*10},{'notes':[True]*10}))
    def timed(self):
        n=self.notes[:10];s={'notes':n,'confidence':'high','events':[{'midi':p,'onset_beats':i,'duration_beats':1} for i,p in enumerate(n)]};a={'notes':n,'events':[{'midi':p,'start':i*.5,'duration':.45,'confidence':.9} for i,p in enumerate(n)]};return s,a
    def test_timing(self):
        s,a=self.timed();r=estimate(s,a,120);self.assertEqual(r['measured_weight_percent'],60);self.assertEqual(r['partial_score_100'],100)
    def test_no_bpm(self):
        s,a=self.timed();r=estimate(s,a);self.assertEqual(r['measured_weight_percent'],45);self.assertIsNone(r['criteria']['tempo']['score_percent'])
    def test_low_confidence(self):
        s,a=self.timed();s['confidence']='low';self.assertEqual(estimate(s,a,120)['measured_weight_percent'],25)
    def test_slow_tempo(self):
        s,a=self.timed();r=estimate(s,a,240);self.assertEqual(r['criteria']['tempo']['score_percent'],0)

if __name__=='__main__':unittest.main()
