"""
Unit tests cho Checkpoint 4 — Guardrails AI Validators (PIIDetector & JSONFormatter).
"""

import json
import os
import sys
import unittest
from pathlib import Path

# Thêm src vào sys.path
SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

# Tắt telemetry OTLP
os.environ["OTEL_SDK_DISABLED"] = "true"
os.environ["OTEL_TRACES_EXPORTER"] = "none"

from guardrails import Guard
from guardrails.validators import FailResult, PassResult
from guardrails.validator_base import OnFailAction

import importlib
cp4 = importlib.import_module("04_guardrails_validator")
JSONFormatter = cp4.JSONFormatter
PIIDetector = cp4.PIIDetector


class TestGuardrailsValidators(unittest.TestCase):
    """Kiểm tra tính đúng đắn của PIIDetector và JSONFormatter."""

    def setUp(self):
        self.pii_guard = Guard().use(PIIDetector(on_fail=OnFailAction.FIX))
        self.json_guard = Guard().use(JSONFormatter(on_fail=OnFailAction.FIX))

    # ── Test PIIDetector ────────────────────────────────────────────────────

    def test_pii_email_redacted(self):
        text = "Send your CV to recruiter@company.org before Monday."
        res = self.pii_guard.validate(text)
        self.assertIn("[EMAIL_REDACTED]", res.validated_output)
        self.assertNotIn("recruiter@company.org", res.validated_output)

    def test_pii_phone_redacted(self):
        text = "Hotline is (555) 867-5309 or 555-123-4567."
        res = self.pii_guard.validate(text)
        self.assertIn("[PHONE_REDACTED]", res.validated_output)
        self.assertNotIn("867-5309", res.validated_output)
        self.assertNotIn("123-4567", res.validated_output)

    def test_pii_ssn_redacted(self):
        text = "Government ID / SSN: 123-45-6789."
        res = self.pii_guard.validate(text)
        self.assertIn("[SSN_REDACTED]", res.validated_output)
        self.assertNotIn("123-45-6789", res.validated_output)

    def test_pii_credit_card_redacted(self):
        text = "Card number: 4532 1234 5678 9010."
        res = self.pii_guard.validate(text)
        self.assertIn("[CREDIT_CARD_REDACTED]", res.validated_output)
        self.assertNotIn("4532 1234", res.validated_output)

    def test_pii_clean_text_unchanged(self):
        clean_text = "Machine learning models require good data and compute resources."
        res = self.pii_guard.validate(clean_text)
        self.assertEqual(res.validated_output, clean_text)

    # ── Test JSONFormatter ──────────────────────────────────────────────────

    def test_valid_json_passes(self):
        valid = '{"model": "gemini-3.8-flash", "temperature": 0.0}'
        res = self.json_guard.validate(valid)
        parsed = json.loads(res.validated_output)
        self.assertEqual(parsed["model"], "gemini-3.8-flash")

    def test_markdown_fences_repaired(self):
        raw = '```json\n{"status": "ok", "count": 42}\n```'
        res = self.json_guard.validate(raw)
        parsed = json.loads(res.validated_output)
        self.assertEqual(parsed["status"], "ok")
        self.assertEqual(parsed["count"], 42)

    def test_single_quotes_repaired(self):
        raw = "{'user': 'alex', 'role': 'admin'}"
        res = self.json_guard.validate(raw)
        parsed = json.loads(res.validated_output)
        self.assertEqual(parsed["user"], "alex")
        self.assertEqual(parsed["role"], "admin")

    def test_trailing_commas_repaired(self):
        raw = '{"items": ["apple", "banana",], "total": 2,}'
        res = self.json_guard.validate(raw)
        parsed = json.loads(res.validated_output)
        self.assertEqual(len(parsed["items"]), 2)
        self.assertEqual(parsed["total"], 2)

    def test_truly_invalid_json_fallback(self):
        raw = "Error 404: Not Found at https://api.endpoint/v1"
        res = self.json_guard.validate(raw)
        parsed = json.loads(res.validated_output)
        self.assertIn("error", parsed)
        self.assertIn("raw", parsed)


if __name__ == "__main__":
    unittest.main()
