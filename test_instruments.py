import unittest
from instrument_profiles import INSTRUMENTS, instrument_guidance

class InstrumentTests(unittest.TestCase):
    def test_exactly_fifteen(self):
        self.assertEqual(len(INSTRUMENTS),15)
    def test_unique_names(self):
        self.assertEqual(len({x["name_uz"] for x in INSTRUMENTS.values()}),15)
    def test_percussion_not_pitch_graded(self):
        self.assertFalse(instrument_guidance(INSTRUMENTS["doira"])["pitch_assessment_supported"])
    def test_polyphonic_limit(self):
        self.assertEqual(instrument_guidance(INSTRUMENTS["piano"])["analysis_compatibility"],"limited")
    def test_no_false_professional_claim(self):
        self.assertTrue(all(not x["professional_grading_available"] for x in INSTRUMENTS.values()))

if __name__ == "__main__":
    unittest.main()
