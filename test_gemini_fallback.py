import asyncio
import unittest
from unittest.mock import patch, AsyncMock
from PIL import Image
from fastapi import HTTPException
from score_omr_v3 import _gemini_page


class TestGeminiFailover(unittest.IsolatedAsyncioTestCase):
    async def test_transient_503_retries_on_configured_backup(self):
        class Result:
            def __init__(self,status): self.status_code=status
            def json(self):
                return {'candidates':[{'content':{'parts':[{'text':'{"notes":[60],"events":[],"confidence":"medium","needs_review":true}'}]}}]}
        class Client:
            def __init__(self): self.urls=[]
            async def post(self,url,**kwargs):
                self.urls.append(url)
                return Result(503 if len(self.urls)==1 else 200)
        client=Client()
        with patch.dict('os.environ',{'GEMINI_API_KEY':'mock','GEMINI_MODEL':'gemini-2.5-flash',
                                    'GEMINI_BACKUP_MODEL':'gemini-2.5-flash-lite'}), patch('asyncio.sleep',new_callable=AsyncMock):
            value=await _gemini_page(Image.new('RGB',(120,80)),page_number=1,exercise=None,client=client)
        self.assertEqual(len(client.urls),2)
        self.assertIn('gemini-2.5-flash:',client.urls[0])
        self.assertIn('gemini-2.5-flash-lite:',client.urls[1])
        self.assertEqual(value['confidence'],'low')
        self.assertTrue(value['needs_review'])

    async def test_rate_limit_429_does_not_generate_retry_storm(self):
        class Client:
            calls=0
            async def post(self,*args,**kwargs):
                self.calls+=1
                class R:status_code=429
                return R()
        c=Client()
        with patch.dict('os.environ',{'GEMINI_API_KEY':'mock','GEMINI_BACKUP_MODEL':'gemini-2.5-flash-lite'}):
            with self.assertRaises(HTTPException) as x:
                await _gemini_page(Image.new('RGB',(120,80)),page_number=1,exercise=None,client=c)
        self.assertEqual(x.exception.status_code,503)
        self.assertEqual(c.calls,1)
