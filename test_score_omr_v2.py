import io
import unittest
from unittest.mock import AsyncMock, patch
from fastapi import HTTPException
from PIL import Image
import fitz

from score_omr_v3 import _validated_page, _join_pages, _diagnose_image, pdf_to_notes, MAX_PAGES


def page(page_number,notes, *, confidence='high', events=True, warnings=None):
    return {"page":page_number,"notes":notes,
            "events":[{"midi":n,"onset_beats":float(i),"duration_beats":1.,"bar":1} for i,n in enumerate(notes)] if events else [],
            "confidence":confidence,"needs_review":confidence!='high',"warnings":warnings or []}


def dummy_pdf(n):
    d=fitz.open()
    for i in range(n):
        p=d.new_page(width=595,height=842)
        p.insert_text((50,70),f'Page {i+1}',fontsize=14)
    data=d.tobytes()
    d.close()
    return data


class TestOMRV2(unittest.TestCase):
    def test_event_schema(self):
        v=_validated_page({'events':[{'midi':60,'onset_beats':0,'duration_beats':1,'bar':1}], 'confidence':'high'},1)
        self.assertEqual(v['notes'],[60]); self.assertFalse(v['needs_review'])

    def test_wrong_note_rejected(self):
        with self.assertRaises(ValueError):
            _validated_page({'notes':[500],'confidence':'high'},1)

    def test_mismatch_rejected(self):
        with self.assertRaises(ValueError):
            _validated_page({'notes':[60], 'events':[{'midi':61,'onset_beats':0,'duration_beats':1}]},1)

    def test_unsorted_rejected(self):
        with self.assertRaises(ValueError):
            _validated_page({'events':[{'midi':60,'onset_beats':2,'duration_beats':1},
                 {'midi':61,'onset_beats':1,'duration_beats':1}]},1)

    def test_onset_offset_across_pages(self):
        result=_join_pages([page(1,[60,62]),page(2,[64,65])])
        self.assertEqual([e['onset_beats'] for e in result['events']],[0.,1.,2.,3.])
        self.assertEqual(result['notes'],[60,62,64,65])
        self.assertTrue(result['needs_review']) # page boundary needs review

    def test_partial_timing_cannot_be_presented_as_complete(self):
        result=_join_pages([page(1,[60]),page(2,[64],events=False)])
        self.assertEqual(result['timing_status'],'not_available')
        self.assertEqual(result['events'],[])

    def test_many_pages_without_truncation(self):
        result=_join_pages([page(i+1,[60,62,64]) for i in range(20)])
        self.assertEqual(result['page_count'],20)
        self.assertEqual(len(result['notes']),60)

    def test_low_quality_hint(self):
        bad=Image.new('RGB',(400,600),'white')
        hints=_diagnose_image(bad)
        self.assertIn('low_resolution',hints)
        self.assertIn('few_visible_symbols',hints)

    def test_invalid_pdf(self):
        async def run():
            with self.assertRaises(HTTPException):
                await pdf_to_notes(b'not a PDF')
        import asyncio
        asyncio.run(run())

    def test_page_limit(self):
        async def run():
            with self.assertRaises(HTTPException) as ctx:
                await pdf_to_notes(dummy_pdf(MAX_PAGES + 1))
            self.assertEqual(ctx.exception.status_code,413)
        import asyncio
        asyncio.run(run())

    def test_pdf_workflow_with_fake_ai(self):
        async def fake(img,*,page_number,exercise,client,**kwargs):
            return page(page_number,[60+page_number,62+page_number])
        async def run():
            with patch('score_omr_v3._gemini_page',side_effect=fake):
                result=await pdf_to_notes(dummy_pdf(3))
            self.assertEqual(result['page_count'],3)
            self.assertEqual(result['notes'],[61,63,62,64,63,65])
        import asyncio
        asyncio.run(run())

if __name__=='__main__':
    unittest.main()

# Extra HTTP-shape checks without external AI requests.
class TestGeminiRecheck(unittest.TestCase):
    def test_uncertain_result_rechecked_and_disagreement_flagged(self):
        import asyncio, json, os
        from unittest.mock import AsyncMock
        from score_omr_v3 import _gemini_page
        class Reply:
            status_code=200
            def __init__(self, pitch): self.pitch=pitch
            def json(self):
                x={"events":[{"midi":self.pitch,"onset_beats":0,"duration_beats":1,"bar":1}],
                   "confidence":"medium","needs_review":True}
                return {"candidates":[{"content":{"parts":[{"text":json.dumps(x)}]}}]}
        async def run():
            client=AsyncMock()
            client.post.side_effect=[Reply(60),Reply(61)]
            with patch.dict(os.environ, {"GEMINI_API_KEY":"fake_for_unit_test"}):
                r=await _gemini_page(Image.new('RGB',(1200,1300),'white'),page_number=1,exercise=None,client=client)
            self.assertEqual(r['notes'],[60]); self.assertIn('omr_pass_disagreement',r['warnings'])
            self.assertTrue(r['needs_review']);self.assertEqual(client.post.call_count,2)
        asyncio.run(run())

    def test_high_confidence_result_one_call(self):
        import asyncio, json, os
        from unittest.mock import AsyncMock
        from score_omr_v3 import _gemini_page
        class Reply:
            status_code=200
            def json(self):
                x={"events":[{"midi":60,"onset_beats":0,"duration_beats":1,"bar":1}],
                    "confidence":"high","needs_review":False}
                return {"candidates":[{"content":{"parts":[{"text":json.dumps(x)}]}}]}
        async def run():
            client=AsyncMock()
            client.post.return_value=Reply()
            with patch.dict(os.environ,{"GEMINI_API_KEY":"fake_for_unit_test"}):
                r=await _gemini_page(Image.new('RGB',(1200,1300),'white'),page_number=1,exercise=None,client=client)
            self.assertEqual(r['notes'],[60]);self.assertEqual(client.post.call_count,1)
        asyncio.run(run())
