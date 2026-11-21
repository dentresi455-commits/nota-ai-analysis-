import asyncio
import io
import json
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image, ImageDraw
from fastapi import HTTPException

from photo_preprocessing import prepare_photo, crop_systems
from score_omr_v3 import _gemini_page, _validated_page, _join_pages, pdf_to_notes

EXAMPLE = Path(__file__).parent / 'qa_private' / 'dilnoz_photo.jpeg'

class PreprocessingTests(unittest.TestCase):
    @unittest.skipUnless(EXAMPLE.exists(), "Private Dilnoz photo fixture unavailable")
    def test_dilnoz_photo_orientation_and_systems(self):
        p=prepare_photo(Image.open(EXAMPLE))
        self.assertEqual(p.orientation_degrees,90)
        self.assertEqual(len(p.staff_systems),6)
        self.assertIn('rotation_180_ambiguity_review',p.warnings)
        self.assertEqual(len(crop_systems(p)),6)
        self.assertEqual(p.original.size,p.ink_suppressed.size)

    @unittest.skipUnless(EXAMPLE.exists(), "Private Dilnoz photo fixture unavailable")
    def test_rotation_override(self):
        p=prepare_photo(Image.open(EXAMPLE),rotation_override=90)
        self.assertEqual(p.rotation_confidence,'override')
        self.assertEqual(p.orientation_degrees,90)

    def test_invalid_rotation_rejected(self):
        with self.assertRaises(ValueError):
            prepare_photo(Image.new('RGB',(200,200),'white'),rotation_override=93)

    def test_blue_annotation_suppressed_alternative(self):
        im=Image.new('RGB',(1000,700),'white')
        d=ImageDraw.Draw(im)
        for y in range(140,190,10):d.line((50,y,950,y), fill='black',width=2)
        d.ellipse((140,100,470,250),outline=(25,25,190),width=9)
        p=prepare_photo(im, rotation_override=0)
        self.assertGreater(p.masks['annotation_fraction'],0.001)
        self.assertIn('colored_handwritten_annotations_detected',p.warnings)
        self.assertNotEqual(p.original.tobytes(),p.ink_suppressed.tobytes())

    def test_no_annotation_does_not_invent_colored_ink(self):
        im=Image.new('RGB',(1000,700),'white')
        d=ImageDraw.Draw(im)
        for y in range(140,190,10):d.line((50,y,950,y), fill='black',width=2)
        p=prepare_photo(im, rotation_override=0)
        self.assertEqual(p.masks['annotation_fraction'],0)

    def test_exif_and_large_limit(self):
        with self.assertRaises(ValueError):
            prepare_photo(Image.new('RGB',(6000,6000),'white'))

class ContractTests(unittest.TestCase):
    def test_excludes_fingering_as_midi(self):
        with self.assertRaises(ValueError):
            _validated_page({'notes':[1,3], 'events':[], 'confidence':'high'},1)

    def test_json_contract_needs_review(self):
        x=_validated_page({'notes':[60,62,64], 'events':[], 'confidence':'low'},1)
        self.assertTrue(x['needs_review'])

    def test_rejects_unmatched_notes_and_events(self):
        with self.assertRaises(ValueError):
            _validated_page({'notes':[60], 'events':[{'midi':61,'onset_beats':0,'duration_beats':1,'bar':1}]},1)

    def test_join_keeps_system_sequence(self):
        p1={'page':1,'notes':[60,62],'events':[{'midi':60,'onset_beats':0,'duration_beats':1,'bar':1}, {'midi':62,'onset_beats':1,'duration_beats':1,'bar':1}], 'warnings':[], 'needs_review':False, 'confidence':'high'}
        p2={'page':1,'notes':[64],'events':[{'midi':64,'onset_beats':0,'duration_beats':1,'bar':1}], 'warnings':[], 'needs_review':False, 'confidence':'high'}
        r=_join_pages([p1,p2])
        self.assertEqual(r['notes'],[60,62,64])
        self.assertEqual([x['onset_beats'] for x in r['events']],[0,1,2])
        self.assertEqual([x['bar'] for x in r['events']],[1,1,2])

class MockedAITests(unittest.IsolatedAsyncioTestCase):
    async def test_second_pass_disagreement_becomes_review(self):
        class Response:
            status_code=200
            def __init__(self,notes): self.notes=notes
            def json(self):return {'candidates':[{'content':{'parts':[{'text':json.dumps({'notes':self.notes,'events':[], 'confidence':'high','needs_review':False})}]}}]}
        class Client:
            def __init__(self):self.calls=0
            async def post(self,*args,**kw):
                self.calls+=1
                return Response([60,62,64] if self.calls==1 else [60,61,64])
        client=Client()
        with patch.dict('os.environ',{'GEMINI_API_KEY':'dummy-key'}):
            result=await _gemini_page(Image.new('RGB',(300,200),'white'),page_number=1,exercise=None,
                                      client=client,require_second_pass=True)
        self.assertEqual(client.calls,2)
        self.assertTrue(result['needs_review'])
        self.assertIn('omr_pass_disagreement',result['warnings'])

    @unittest.skipUnless(EXAMPLE.exists(), "Private Dilnoz photo fixture unavailable")
    async def test_omr_pdf_flow_with_real_sample_mocked_provider(self):
        im=Image.open(EXAMPLE).convert('RGB')
        buf=io.BytesIO()
        im.save(buf,format='PDF',resolution=150)
        calls=[]
        async def fake_reader(img,*,page_number,exercise,client,alternate=None,require_second_pass=False,system=None):
            calls.append(system)
            return {'page':page_number,'notes':[60,62,64],'events':[], 'confidence':'low','needs_review':True,'warnings':[]}
        with patch('score_omr_v3._gemini_page', side_effect=fake_reader):
            result=await pdf_to_notes(buf.getvalue())
        self.assertEqual(result['omr_engine'],'photo-aware-system-v3-staging')
        self.assertEqual(result['staff_system_count'],6)
        self.assertEqual(len(calls),6)
        self.assertEqual(len(result['notes']),18)
        self.assertTrue(result['needs_review'])
        self.assertIsNone(result.get('professional_score_100'))
        self.assertEqual(len(result['pages']),1)

if __name__=='__main__':unittest.main()
