"""Offline regressions. No real provider keys or network calls are used."""
import json
import os
import unittest
from unittest.mock import patch, AsyncMock
from PIL import Image
from fastapi import HTTPException
import score_omr_v3 as omr
from timing_analysis import Event, analyze, align
from long_alignment import compare_complete


def seq(n):
    return [Event(60 + i % 7, i * .25, .20) for i in range(n)]


class LongTimingTests(unittest.TestCase):
    def test_tail_error_after_500(self):
        a = seq(1001)
        b = a[:]
        b[-1] = Event(75, b[-1].start, .20)
        r = analyze(a, b, .25, 4)
        self.assertEqual(r['note_errors'][-1]['reference_note'], 1001)
        self.assertEqual(r['note_errors'][-1]['actual_midi'], 75)
        self.assertIsNone(r['overall_score_100'])

    def test_12000_identical_notes(self):
        a = seq(12000)
        r = analyze(a, a, .25, 4)
        self.assertEqual(r['pitch_similarity_percent'], 100)
        self.assertEqual(r['note_matches'], 12000)
        self.assertEqual(r['rhythm']['compared_intervals'], 11999)

    def test_short_alignment_compatible(self):
        a = [Event(60, 0, .4), Event(62, .5, .4), Event(64, 1, .4)]
        b = [Event(60, 0, .4), Event(63, .5, .4), Event(64, 1, .4)]
        pairs, errors, edits = align(a, b)
        self.assertEqual(pairs, [(0, 0), (1, 1), (2, 2)])
        self.assertEqual(edits, 1)
        self.assertEqual(errors[0]['audio_time_seconds'], .5)

    def test_long_insert_and_delete_count(self):
        a = seq(805)
        b = a[:400] + [Event(80, 99.9, .1)] + a[400:700] + a[701:]
        _, errors, edits = align(a, b)
        self.assertEqual(edits, 2)
        self.assertEqual({x['kind'] for x in errors}, {'extra_note', 'missing_note'})

    def test_over_limit_explicit(self):
        with self.assertRaises(ValueError):
            analyze(seq(12001), seq(12001))

    def test_expensive_alignment_rejected(self):
        with patch.dict(os.environ, {'NOTA_MAX_ALIGNMENT_CELLS': '40000'}):
            with self.assertRaisesRegex(ValueError, 'queued'):
                align(seq(600), [Event(80, i*.25, .20) for i in range(600)])

    def test_existing_complete_pitch_alignment_tail(self):
        a = [60, 62, 64, 65] * 500
        b = a[:]
        b[-1] = 73
        result = compare_complete(a, b)
        self.assertEqual(result['edit_distance'], 1)
        self.assertEqual(result['detailed_feedback']['errors'][0]['reference_position'], 2000)


class Reply:
    def __init__(self, status_code, data=None):
        self.status_code = status_code
        self.data = data or {}
    def json(self):
        return self.data


def valid_data():
    return {'events': [{'midi': 60, 'onset_beats': 0, 'duration_beats': 1, 'bar': 1}],
            'confidence': 'high', 'needs_review': False, 'warnings': []}


def gemini(data=None, finish='STOP'):
    return Reply(200, {'candidates': [{'finishReason': finish, 'content': {'parts': [{'text': json.dumps(data if data is not None else valid_data())}]}}]})


def vision(data=None, finish='stop'):
    return Reply(200, {'choices': [{'finish_reason': finish, 'message': {'content': json.dumps(data if data is not None else valid_data())}}]})


class Client:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []
    async def post(self, url, **kwargs):
        self.calls.append(url)
        if not self.replies:
            raise AssertionError('Unexpected provider request')
        response = self.replies.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class ProviderTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {
            'GEMINI_API_KEY': 'test-only', 'GEMINI_MODEL': 'gemini-2.5-flash',
            'GEMINI_BACKUP_MODEL': '', 'NOTA_ENABLE_VISION_FALLBACK': 'true',
            'OPENROUTER_API_KEY': 'test-only', 'NOTA_OPENROUTER_VISION_MODEL': 'test/vision',
            'GROQ_API_KEY': 'test-only', 'NOTA_GROQ_VISION_MODEL': 'test-vision',
        }, clear=False)
        self.env.start()
        self.addCleanup(self.env.stop)
        self.sleep = patch('score_omr_v3.asyncio.sleep', new_callable=AsyncMock)
        self.sleep.start()
        self.addCleanup(self.sleep.stop)
        self.img = Image.new('RGB', (120, 80), 'white')

    async def run_page(self, client):
        return await omr._gemini_page(self.img, page_number=1, exercise=None, client=client)

    async def test_gemini_success_no_fallback(self):
        c = Client([gemini()])
        r = await self.run_page(c)
        self.assertEqual(r['notes'], [60])
        self.assertEqual(len(c.calls), 1)

    async def test_503_retry_then_openrouter(self):
        c = Client([Reply(503), Reply(503), vision()])
        r = await self.run_page(c)
        self.assertIn('openrouter.ai', c.calls[-1])
        self.assertIn('vision_fallback_used_openrouter', r['warnings'])
        self.assertEqual(r['confidence'], 'low')
        self.assertTrue(r['needs_review'])

    async def test_429_no_gemini_retry_then_groq(self):
        c = Client([Reply(429), Reply(503), vision()])
        r = await self.run_page(c)
        self.assertEqual(sum('googleapis' in u for u in c.calls), 1)
        self.assertIn('api.groq.com', c.calls[-1])
        self.assertIn('vision_fallback_used_groq', r['warnings'])

    async def test_missing_key_can_fallback(self):
        c = Client([vision()])
        with patch.dict(os.environ, {'GEMINI_API_KEY': ''}):
            r = await self.run_page(c)
        self.assertIn('openrouter.ai', c.calls[0])
        self.assertIn('gemini_key_missing', r['warnings'])

    async def test_bad_key_fallback(self):
        c = Client([Reply(403), vision()])
        r = await self.run_page(c)
        self.assertIn('gemini_key_not_authorized', r['warnings'])

    async def test_invalid_response_fallback(self):
        c = Client([gemini({}), gemini({}), vision()])
        r = await self.run_page(c)
        self.assertIn('gemini_invalid_response', r['warnings'])

    async def test_incomplete_provider_responses_rejected(self):
        c = Client([gemini(finish='MAX_TOKENS'), gemini(finish='MAX_TOKENS'), vision(finish='length'), vision()])
        r = await self.run_page(c)
        self.assertIn('vision_fallback_used_groq', r['warnings'])
        self.assertEqual(len(c.calls), 4)

    async def test_fallback_disabled_no_provider_call(self):
        c = Client([Reply(429)])
        with patch.dict(os.environ, {'NOTA_ENABLE_VISION_FALLBACK': 'false'}):
            with self.assertRaises(HTTPException) as error:
                await self.run_page(c)
        self.assertEqual(error.exception.status_code, 503)
        self.assertEqual(len(c.calls), 1)

    async def test_all_providers_fail_no_fake_score(self):
        c = Client([Reply(503), Reply(503), Reply(401), Reply(429)])
        with self.assertRaises(HTTPException) as error:
            await self.run_page(c)
        self.assertEqual(error.exception.status_code, 502)

    async def test_first_valid_read_preserved_after_recheck_failure(self):
        partial = valid_data()
        partial['confidence'] = 'medium'
        c = Client([gemini(partial), Reply(429)])
        r = await self.run_page(c)
        self.assertEqual(r['notes'], [60])
        self.assertEqual(r['confidence'], 'low')
        self.assertTrue(r['needs_review'])
        self.assertEqual(len(c.calls), 2)

    def test_timeout_is_bounded(self):
        with patch.dict(os.environ, {'TEST_NOTA_TIMEOUT': '99999'}):
            self.assertEqual(omr._bounded_int('TEST_NOTA_TIMEOUT', 52, 30, 240), 240)
        with patch.dict(os.environ, {'TEST_NOTA_TIMEOUT': 'wrong'}):
            self.assertEqual(omr._bounded_int('TEST_NOTA_TIMEOUT', 52, 30, 240), 52)


if __name__ == '__main__':
    unittest.main()
