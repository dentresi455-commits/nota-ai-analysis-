import tempfile
import unittest
from pathlib import Path
from apply_to_render_app import apply_patch

SAMPLE='''import os\nimport json\n\nasync def pdf_to_notes(data, exercise=None):\n    return None\n\nasync def analyze_files(normalized_score,exercise):\n    score_data = await pdf_to_notes(normalized_score, exercise)\n    return score_data\n'''
class TestIntegration(unittest.TestCase):
    def test_flag_off_and_backup(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'app.py';p.write_text(SAMPLE)
            self.assertEqual(apply_patch(p),'applied')
            self.assertEqual(apply_patch(p),'already_applied')
            self.assertTrue(p.with_suffix('.py.omr-v3-backup').exists())
            self.assertIn('NOTA_OMR_V3_ENABLED',p.read_text())
    def test_unknown_source_refused(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'app.py';p.write_text('print(123)')
            with self.assertRaises(RuntimeError):apply_patch(p)
if __name__=='__main__':unittest.main()
