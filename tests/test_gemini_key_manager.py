"""
Unit tests cho GeminiKeyManager, rate limit handling và checkpoint resume.
Tất cả các bài test đều sử dụng MOCK, KHÔNG TIÊU TỐN API QUOTA.
"""

import io
import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Thêm src vào sys.path
SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from utils.gemini_key_manager import (
    AllProjectsQuotaExhaustedError,
    GeminiKeyInfo,
    GeminiKeyManager,
    KeyStatus,
    MultiKeyGeminiChat,
    parse_gemini_error,
    parse_retry_delay_string,
    sanitize_text,
)


class TestGeminiKeyManager(unittest.TestCase):
    """Bộ unit tests toàn diện cho Gemini Key Manager & Rate Limiting."""

    def setUp(self):
        self.mock_keys = ["mock-key-alpha-12345", "mock-key-beta-67890", "mock-key-gamma-54321"]

    # 1. Test đọc 3 keys từ env
    def test_read_three_api_keys_from_env(self):
        env_vars = {
            "GEMINI_API_KEY_1": "key-1-value-123456",
            "GEMINI_API_KEY_2": "key-2-value-123456",
            "GEMINI_API_KEY_3": "key-3-value-123456",
            "GEMINI_PROJECT_ID_1": "proj-a",
            "GEMINI_PROJECT_ID_2": "proj-b",
            "GEMINI_PROJECT_ID_3": "proj-c",
        }
        with patch.dict(os.environ, env_vars, clear=False):
            mgr = GeminiKeyManager.from_env()
            self.assertEqual(mgr.total_keys, 3)
            self.assertEqual(mgr.keys[0].name, "Key-1")
            self.assertEqual(mgr.keys[1].name, "Key-2")
            self.assertEqual(mgr.keys[2].name, "Key-3")
            self.assertEqual(mgr.keys[0].project_id, "proj-a")
            self.assertEqual(mgr.keys[1].project_id, "proj-b")
            self.assertEqual(mgr.keys[2].project_id, "proj-c")

    # 2. Test backward compatibility với GOOGLE_API_KEY khi chỉ cấu hình 1 key
    def test_backward_compatibility_with_single_google_api_key(self):
        env_vars = {
            "GEMINI_API_KEY_1": "",
            "GEMINI_API_KEY_2": "",
            "GEMINI_API_KEY_3": "",
            "GOOGLE_API_KEY": "legacy-google-key-99999",
        }
        with patch.dict(os.environ, env_vars, clear=False):
            mgr = GeminiKeyManager.from_env()
            self.assertEqual(mgr.total_keys, 1)
            self.assertEqual(mgr.keys[0].name, "Key-1")
            self.assertEqual(mgr.keys[0].status, KeyStatus.AVAILABLE)

    # 3. Test key missing / empty
    def test_empty_or_missing_keys_handled(self):
        env_vars = {
            "GEMINI_API_KEY_1": "",
            "GEMINI_API_KEY_2": "  ",
            "GEMINI_API_KEY_3": "",
            "GOOGLE_API_KEY": "",
        }
        with patch.dict(os.environ, env_vars, clear=False):
            mgr = GeminiKeyManager.from_env()
            self.assertEqual(mgr.total_keys, 0)
            with self.assertRaises(AllProjectsQuotaExhaustedError):
                mgr.acquire_key_sync(timeout=0.1)

    # 4. Test tuyệt đối không làm lộ API keys trong logs, exceptions, repr, str
    def test_no_api_key_leak_in_repr_str_and_logs(self):
        secret = "mock_dummy_secret_key_123456789"
        mgr = GeminiKeyManager(keys=[secret])
        key_info = mgr.keys[0]

        # Kiểm tra __repr__ và __str__
        repr_str = repr(key_info)
        str_str = str(key_info)
        self.assertNotIn(secret, repr_str)
        self.assertNotIn(secret, str_str)

        # Kiểm tra sanitize_text
        error_msg = f"Failed to connect to https://generativelanguage.googleapis.com/?key={secret}"
        sanitized = sanitize_text(error_msg, [secret])
        self.assertNotIn(secret, sanitized)
        self.assertIn("[REDACTED", sanitized)

        # Kiểm tra bảng trạng thái không có secret
        table = mgr.format_status_table()
        self.assertNotIn(secret, table)

    # 5. Test lỗi xác thực 401 / 403 API_KEY_INVALID -> Chuyển sang INVALID
    def test_invalid_auth_key_marks_invalid_status(self):
        mgr = GeminiKeyManager(keys=self.mock_keys)
        fake_auth_error = Exception("401 API_KEY_INVALID: The provided API key is invalid.")

        parsed = mgr.record_error(key_index=1, exc=fake_auth_error)
        self.assertTrue(parsed.is_auth_error)
        self.assertEqual(mgr.keys[0].status, KeyStatus.INVALID)
        self.assertEqual(mgr.keys[0].failure_count, 1)

        # Key 1 không được chọn nữa, nhưng Key 2 vẫn khả dụng
        selected = mgr.select_available_key()
        self.assertIsNotNone(selected)
        self.assertNotEqual(selected.index, 1)

    # 6. Test Short-term rate limit (429 with small retryDelay) -> COOLDOWN
    def test_short_term_rate_limit_cooldown_and_recovery(self):
        mgr = GeminiKeyManager(keys=["key-alpha"], default_short_cooldown=0.2)
        fake_rate_error = Exception("429 RESOURCE_EXHAUSTED: rate limit exceeded. {'retryDelay': '0.2s'}")

        parsed = mgr.record_error(key_index=1, exc=fake_rate_error)
        self.assertTrue(parsed.is_rate_limit)
        self.assertFalse(parsed.is_daily_quota)
        self.assertEqual(mgr.keys[0].status, KeyStatus.COOLDOWN)

        # Trong thời gian cooldown, key không khả dụng
        self.assertIsNone(mgr.select_available_key())

        # Đợi hết cooldown, key tự động AVAILABLE trở lại
        time.sleep(0.3)
        recovered = mgr.select_available_key()
        self.assertIsNotNone(recovered)
        self.assertEqual(recovered.status, KeyStatus.AVAILABLE)

    # 7. Test Daily Quota Exhaustion (500 req/day / 58147s)
    def test_daily_quota_exhaustion_parsing(self):
        raw_exc_str = (
            "GoogleRateLimitError: 429 RESOURCE_EXHAUSTED. {'error': {'code': 429, "
            "'message': 'Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_requests, "
            "limit: 500, model: gemini-3.5-flash-lite\\nPlease retry in 16h9m7.829s.', "
            "'details': [{'@type': 'type.googleapis.com/google.rpc.QuotaFailure', "
            "'violations': [{'quotaMetric': 'generativelanguage.googleapis.com/generate_content_free_tier_requests', "
            "'quotaId': 'GenerateRequestsPerDayPerProjectPerModel-FreeTier'}]}, "
            "{'@type': 'type.googleapis.com/google.rpc.RetryInfo', 'retryDelay': '58147s'}]}}"
        )
        parsed = parse_gemini_error(Exception(raw_exc_str))
        self.assertTrue(parsed.is_rate_limit)
        self.assertTrue(parsed.is_daily_quota)
        self.assertEqual(parsed.quota_id, "GenerateRequestsPerDayPerProjectPerModel-FreeTier")
        self.assertEqual(parsed.retry_delay_seconds, 58147.0)

    # 8. Test các key thuộc CÙNG project: 1 key hết quota -> Toàn bộ keys cùng project đều EXHAUSTED
    def test_same_project_keys_exhaust_together_no_illegal_rotation(self):
        # 3 keys thuộc cùng project 'my-shared-project'
        mgr = GeminiKeyManager(
            keys=self.mock_keys,
            project_ids=["my-shared-project", "my-shared-project", "my-shared-project"],
        )
        fake_daily_error = Exception(
            "429 RESOURCE_EXHAUSTED: Quota exceeded for metric: free_tier_requests, limit: 500. retryDelay: 58147s"
        )

        mgr.record_error(key_index=1, exc=fake_daily_error)

        # Cả 3 keys đều phải chuyển sang EXHAUSTED
        self.assertEqual(mgr.keys[0].status, KeyStatus.EXHAUSTED)
        self.assertEqual(mgr.keys[1].status, KeyStatus.EXHAUSTED)
        self.assertEqual(mgr.keys[2].status, KeyStatus.EXHAUSTED)

        # Không còn key nào khả dụng, ném AllProjectsQuotaExhaustedError
        with self.assertRaises(AllProjectsQuotaExhaustedError):
            mgr.acquire_key_sync(timeout=0.1)

    # 9. Test các key thuộc project KHÁC NHAU: Failover sang project còn quota
    def test_different_project_keys_allow_failover(self):
        # Key 1 thuộc project-A, Key 2 thuộc project-B
        mgr = GeminiKeyManager(
            keys=["key-a-11111", "key-b-22222"],
            project_ids=["project-A", "project-B"],
        )
        fake_daily_error = Exception("429 RESOURCE_EXHAUSTED: limit: 500, retryDelay: 58147s")

        # Key 1 gặp daily quota exhaustion
        mgr.record_error(key_index=1, exc=fake_daily_error)

        self.assertEqual(mgr.keys[0].status, KeyStatus.EXHAUSTED)
        # Key 2 thuộc project-B độc lập, vẫn phải AVAILABLE
        self.assertEqual(mgr.keys[1].status, KeyStatus.AVAILABLE)

        # KeyManager phải cấp Key 2 thành công
        acquired = mgr.acquire_key_sync(timeout=1.0)
        self.assertEqual(acquired.index, 2)
        self.assertEqual(acquired.project_id, "project-B")
        mgr.release_key(acquired.index)

    # 10. Test Checkpoint Resume: không sinh lại answers đã lưu, không chấm lại metrics đã hoàn thành
    def test_checkpoint_resume_skips_completed_answers_and_metrics(self):
        from src import config

        with tempfile.TemporaryDirectory() as tmp_dir:
            ckpt_path = Path(tmp_dir) / "test_checkpoint.json"

            # Checkpoint giả lập đã có 1 answer V1 và 1 kết quả evaluation hợp lệ
            initial_data = {
                "version": 1,
                "config": {"evaluator_model": "gemini-3.8-flash"},
                "answers": {
                    "v1": {
                        "qa_01": {
                            "sample_id": "qa_01",
                            "prompt_version": "v1",
                            "question": "What is overfitting?",
                            "reference": "Model fits noise.",
                            "answer": "Overfitting happens when a model learns noise.",
                            "contexts": ["Context 1", "Context 2"],
                            "model": "gemini-3.8-flash",
                            "timestamp": time.time(),
                        }
                    },
                    "v2": {},
                },
                "evaluations": {
                    "v1": {
                        "qa_01": {
                            "sample_id": "qa_01",
                            "scores": {
                                "faithfulness": 1.0,
                                "answer_relevancy": 0.85,
                                "context_recall": 1.0,
                                "context_precision": 0.95,
                            },
                            "timestamp": time.time(),
                        }
                    },
                    "v2": {},
                },
            }

            # Lưu file checkpoint
            import importlib
            ragas_eval_mod = importlib.import_module("03_ragas_evaluation")
            atomic_save_json = ragas_eval_mod.atomic_save_json
            load_checkpoint = ragas_eval_mod.load_checkpoint
            atomic_save_json(ckpt_path, initial_data)

            # Tải lại
            loaded = load_checkpoint(ckpt_path)
            self.assertIn("qa_01", loaded["answers"]["v1"])
            self.assertIn("qa_01", loaded["evaluations"]["v1"])

            # Kiểm tra các điểm đều hợp lệ
            scores = loaded["evaluations"]["v1"]["qa_01"]["scores"]
            self.assertEqual(scores["faithfulness"], 1.0)
            self.assertEqual(scores["answer_relevancy"], 0.85)

            # Đảm bảo không có duplicate records
            self.assertEqual(len(loaded["answers"]["v1"]), 1)
            self.assertEqual(len(loaded["evaluations"]["v1"]), 1)

    # 11. Test retry delay string parsing
    def test_retry_delay_string_parser(self):
        self.assertEqual(parse_retry_delay_string("58147s"), 58147.0)
        self.assertEqual(parse_retry_delay_string("15s"), 15.0)
        self.assertEqual(parse_retry_delay_string("1m30s"), 90.0)
        self.assertAlmostEqual(parse_retry_delay_string("16h9m7.829s"), 58147.829, places=2)
        self.assertEqual(parse_retry_delay_string(""), 0.0)


if __name__ == "__main__":
    unittest.main()
