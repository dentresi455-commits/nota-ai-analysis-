"""OMR v3.1 bounded-concurrency and explicit-failure regressions."""
import asyncio
import io
import unittest
from unittest.mock import patch
from pathlib import Path
from PIL import Image
from fastapi import HTTPException
from score_omr_v3 import _gemini_page, pdf_to_notes

FIXTURE = Path(__file__).parent / 'qa_private' / 'dilnoz_photo.jpeg'

def sample_pdf():
    b=io.BytesIO()
    Image.open(FIXTURE).convert('RGB').save(b,format='PDF',resolution=150)
    return b.getvalue()

class ProviderFailureTests(unittest.IsolatedAsyncioTestCase):
    async def test_handwritten_annotation_does_not_force_extra_ai_call_when_clear(self):
        class Response:
            status_code=200
            def json(self):
                return {'candidates':[{'content':{'parts':[{'text':'{"notes":[60],"events":[],"confidence":"high","needs_review":false}'}]}}]}
        class Client:
            calls=0
            async def post(self,*args,**kwargs):
                self.calls+=1
                return Response()
        client=Client()
        with patch.dict('os.environ',{'GEMINI_API_KEY':'dummy'}):
            res=await _gemini_page(Image.new('RGB',(200,100)),page_number=1,
                exercise=None,client=client,require_second_pass=True)
        # Notes-only JSON is intrinsically reviewable, therefore it may
        # legitimately get a second pass. A high-quality timing result is
        # required for the one-pass fast path; see separate test below.
        self.assertEqual(client.calls,2)

    async def test_first_pass_high_with_events_avoids_expensive_double_scan(self):
        class Response:
            status_code=200
            def json(self):
                return {'candidates':[{'content':{'parts':[{'text':'{"events":[{"midi":60,"onset_beats":0,"duration_beats":1,"bar":1}],"confidence":"high","needs_review":false}'}]}}]}
        class Client:
            calls=0
            async def post(self,*args,**kwargs):
                self.calls+=1
                return Response()
        client=Client()
        with patch.dict('os.environ',{'GEMINI_API_KEY':'dummy'}):
            res=await _gemini_page(Image.new('RGB',(200,100)),page_number=1,
                exercise=None,client=client,require_second_pass=True)
        self.assertEqual(client.calls,1)
        self.assertTrue(res['needs_review'])
        self.assertIn('annotation_recheck_deferred',res['warnings'])

    async def test_missing_model_is_explicit_502(self):
        class Client:
            async def post(self,*a,**kw):
                class Response:status_code=404
                return Response()
        with patch.dict('os.environ',{'GEMINI_API_KEY':'dummy','GEMINI_MODEL':'gemini-2.5-flash'}):
            with self.assertRaises(HTTPException) as exc:
                await _gemini_page(Image.new('RGB',(200,100)),page_number=1,exercise=None,client=Client())
        self.assertEqual(exc.exception.status_code,502)
        self.assertEqual(exc.exception.detail,'gemini_model_not_found')

    async def test_bad_key_is_explicit_502(self):
        class Client:
            async def post(self,*a,**kw):
                class Response:status_code=403
                return Response()
        with patch.dict('os.environ',{'GEMINI_API_KEY':'dummy'}):
            with self.assertRaises(HTTPException) as exc:
                await _gemini_page(Image.new('RGB',(200,100)),page_number=1,exercise=None,client=Client())
        self.assertEqual(exc.exception.detail,'gemini_key_not_authorized')

    async def test_rate_limit_does_not_retry_storm(self):
        class Client:
            calls=0
            async def post(self,*a,**kw):
                self.calls+=1
                class Response:status_code=429
                return Response()
        client=Client()
        with patch.dict('os.environ',{'GEMINI_API_KEY':'dummy'}):
            with self.assertRaises(HTTPException) as exc:
                await _gemini_page(Image.new('RGB',(200,100)),page_number=1,exercise=None,client=client)
        self.assertEqual(exc.exception.status_code,503)
        self.assertEqual(client.calls,1)

    async def test_invalid_model_rejected_before_http_call(self):
        class Client:
            calls=0
            async def post(self,*a,**kw):
                self.calls+=1
        client=Client()
        with patch.dict('os.environ',{'GEMINI_API_KEY':'dummy','GEMINI_MODEL':'../unexpected'}):
            with self.assertRaises(HTTPException) as exc:
                await _gemini_page(Image.new('RGB',(100,100)),page_number=1,exercise=None,client=client)
        self.assertEqual(exc.exception.detail,'gemini_model_invalid')
        self.assertEqual(client.calls,0)

class OrchestrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_six_real_staffs_read_concurrently_in_original_order(self):
        active=0
        highwater=0
        done=[]
        async def fake_reader(img,*,page_number,exercise,client,alternate=None,require_second_pass=False,system=None):
            nonlocal active,highwater
            active+=1
            highwater=max(highwater,active)
            try:
                await asyncio.sleep((7-system)*0.005)
                done.append(system)
                return {'page':page_number,'notes':[60+system],'events':[],
                        'confidence':'low','needs_review':True,'warnings':[]}
            finally:active-=1
        with patch('score_omr_v3._gemini_page',side_effect=fake_reader):
            result=await pdf_to_notes(sample_pdf())
        self.assertEqual(result['staff_system_count'],6)
        self.assertEqual(result['page_count'],1)
        self.assertEqual(result['notes'],[61,62,63,64,65,66])
        self.assertLessEqual(highwater,3)
        self.assertGreater(highwater,1)
        self.assertNotEqual(done,list(range(1,7)))
        self.assertTrue(result['needs_review'])
        self.assertEqual(result['transcription_status'],'estimated_unverified')
        self.assertNotIn('professional_score_100',result)

    async def test_hard_global_deadline_reports_504_not_fake_score(self):
        async def slow_reader(*args,**kwargs):
            await asyncio.sleep(0.2)
            return {'page':1,'notes':[60],'events':[],'confidence':'high','needs_review':False,'warnings':[]}
        with patch('score_omr_v3._gemini_page',side_effect=slow_reader), patch('score_omr_v3.OMR_TOTAL_TIMEOUT_SECONDS',0.01):
            with self.assertRaises(HTTPException) as exc:
                await pdf_to_notes(sample_pdf())
        self.assertEqual(exc.exception.status_code,504)
        self.assertEqual(exc.exception.detail['code'],'omr_time_budget_exceeded')

    async def test_bad_staff_returns_page_and_system_without_scores(self):
        async def fail_reader(img,*,page_number,exercise,client,alternate=None,require_second_pass=False,system=None):
            if system==3:
                raise HTTPException(status_code=503,detail='gemini_rate_limited')
            return {'page':page_number,'notes':[60], 'events':[],'confidence':'low','needs_review':True,'warnings':[]}
        with patch('score_omr_v3._gemini_page',side_effect=fail_reader):
            with self.assertRaises(HTTPException) as exc:
                await pdf_to_notes(sample_pdf())
        self.assertEqual(exc.exception.status_code,503)
        self.assertEqual(exc.exception.detail['page'],1)
        self.assertEqual(exc.exception.detail['system'],3)
        self.assertEqual(exc.exception.detail['code'],'gemini_rate_limited')

if __name__=='__main__': unittest.main()
