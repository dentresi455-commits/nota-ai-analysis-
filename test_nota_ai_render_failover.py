"""Offline deterministic tests of NOTA AI fallback patch (no API keys, no charges)."""
import asyncio
import importlib.util
import json
import math
import os
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch
import base64

here = Path(__file__).parent
spec = importlib.util.spec_from_file_location("nota_patch", here / "nota_ai_render_failover_patch.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class HTTPException(Exception):
    def __init__(self, status_code, detail):
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


class FakePix:
    def tobytes(self, fmt):
        assert fmt == "png"
        return b"PNG-TEST"


class FakePage:
    def get_pixmap(self, matrix, alpha):
        return FakePix()


class FakeDoc:
    needs_pass = False
    def __init__(self, pages=1):
        self.pages = pages
    def __len__(self):
        return self.pages
    def __iter__(self):
        return iter([FakePage() for _ in range(self.pages)])
    def close(self):
        pass


class FakeFitz:
    Matrix = lambda self, a, b: (a, b)
    def __init__(self, pages=1):
        self.pages = pages
    def open(self, stream, filetype):
        return FakeDoc(self.pages)


class StubResponse:
    def __init__(self, code, data):
        self.status_code = code
        self.data = data
    def json(self):
        return self.data


GOOD = {"notes": [60, 62, 64], "events": [
    {"midi": 60, "onset_beats": 0, "duration_beats": 1},
    {"midi": 62, "onset_beats": 1, "duration_beats": 1},
    {"midi": 64, "onset_beats": 2, "duration_beats": 1}],
        "confidence": "high"}


class TestFallback(unittest.TestCase):
    def run_with(self, statuses, env, pages=1):
        seen = []
        class FakeAsyncClient:
            def __init__(self, timeout):
                assert timeout <= 15
            async def __aenter__(self):
                return self
            async def __aexit__(self, *args):
                return None
            async def post(self, url, **kwargs):
                seen.append(url)
                if "generativelanguage" in url:
                    provider = "gemini"
                elif "openrouter" in url:
                    provider = "openrouter"
                else:
                    provider = "groq"
                status = statuses[provider]
                if status != 200:
                    return StubResponse(status, {"error": "quota"})
                text = json.dumps(GOOD)
                if provider == "gemini":
                    return StubResponse(200, {"candidates": [{"content": {"parts": [{"text": text}]}}]})
                return StubResponse(200, {"choices": [{"message": {"content": text}}]})
        httpx = types.SimpleNamespace(AsyncClient=FakeAsyncClient, HTTPError=OSError)
        namespace = {"fitz": FakeFitz(pages), "base64": base64,
                     "json": json, "math": math, "os": os,
                     "httpx": httpx, "HTTPException": HTTPException}
        exec(module.NEW_FUNCTION, namespace)
        with patch.dict(os.environ, env, clear=True):
            try:
                return asyncio.run(namespace["pdf_to_notes"](b"%PDF-1.6 dummy")), seen
            except HTTPException as exc:
                return exc, seen

    def test_gemini_works_without_fallback(self):
        got, seen = self.run_with({"gemini": 200, "openrouter": 200, "groq": 200},
                                  {"GEMINI_API_KEY": "fake", "OPENROUTER_API_KEY": "fake", "GROQ_API_KEY": "fake"})
        self.assertEqual(got["ai_provider"], "gemini")
        self.assertEqual(len(seen), 1)

    def test_gemini_429_falls_to_openrouter(self):
        got, seen = self.run_with({"gemini": 429, "openrouter": 200, "groq": 200},
                                  {"GEMINI_API_KEY": "fake", "OPENROUTER_API_KEY": "fake", "GROQ_API_KEY": "fake"})
        self.assertEqual(got["ai_provider"], "openrouter")
        self.assertEqual(len(seen), 2)
        self.assertEqual(len(got["events"]), 3)

    def test_two_fail_then_groq(self):
        got, seen = self.run_with({"gemini": 500, "openrouter": 503, "groq": 200},
                                  {"GEMINI_API_KEY": "fake", "OPENROUTER_API_KEY": "fake", "GROQ_API_KEY": "fake"})
        self.assertEqual(got["ai_provider"], "groq")
        self.assertEqual(len(seen), 3)

    def test_all_fail_no_fabricated_result(self):
        got, seen = self.run_with({"gemini": 429, "openrouter": 500, "groq": 503},
                                  {"GEMINI_API_KEY": "fake", "OPENROUTER_API_KEY": "fake", "GROQ_API_KEY": "fake"})
        self.assertIsInstance(got, HTTPException)
        self.assertEqual(got.status_code, 502)
        self.assertEqual(len(seen), 3)

    def test_groq_four_page_limit(self):
        got, seen = self.run_with({"gemini": 500, "openrouter": 500, "groq": 200},
                                  {"GEMINI_API_KEY": "fake", "OPENROUTER_API_KEY": "fake", "GROQ_API_KEY": "fake"}, pages=4)
        self.assertIsInstance(got, HTTPException)
        self.assertEqual(got.status_code, 502)
        self.assertEqual(len(seen), 2)

    def test_missing_keys_returns_503(self):
        got, seen = self.run_with({"gemini": 200, "openrouter": 200, "groq": 200}, {})
        self.assertIsInstance(got, HTTPException)
        self.assertEqual(got.status_code, 503)
        self.assertEqual(len(seen), 0)

    def test_patch_makes_backup_and_preserves_rest(self):
        src = 'import os\n\nasync def pdf_to_notes(data, exercise=None):\n    return {"old": True}\n\ndef audio_to_notes(raw, filename):\n    return raw\n'
        with tempfile.TemporaryDirectory() as d:
            target = Path(d, "app.py")
            target.write_text(src)
            module.patch(target)
            self.assertEqual(Path(d, "app.py.backup-before-ai-failover").read_text(), src)
            self.assertIn('def audio_to_notes(raw, filename):', target.read_text())
            with self.assertRaises(ValueError):
                module.patch(target)


if __name__ == "__main__":
    unittest.main(verbosity=2)
