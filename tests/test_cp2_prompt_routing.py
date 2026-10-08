"""Unit checks for CP2 deterministic prompt routing."""
import hashlib
import importlib
import sys
import unittest
from pathlib import Path


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))
cp2 = importlib.import_module("02_prompt_hub_ab_routing")


class PromptRoutingTests(unittest.TestCase):
    def test_same_request_id_always_selects_same_prompt(self):
        for request_id in ("req-0000", "req-0001", "customer-42", "stable-id"):
            routes = {cp2.get_prompt_version(request_id) for _ in range(20)}
            self.assertEqual(len(routes), 1)

    def test_route_matches_md5_parity(self):
        for request_id in ("req-0000", "req-0001", "req-0049"):
            value = int(hashlib.md5(request_id.encode("utf-8")).hexdigest(), 16)
            expected = cp2.PROMPT_V1_NAME if value % 2 == 0 else cp2.PROMPT_V2_NAME
            self.assertEqual(cp2.get_prompt_version(request_id), expected)

    def test_fifty_request_ids_cover_both_versions(self):
        routes = {cp2.get_prompt_version(f"req-{i:04d}") for i in range(50)}
        self.assertEqual(routes, {cp2.PROMPT_V1_NAME, cp2.PROMPT_V2_NAME})

    def test_prompts_keep_required_variables(self):
        for prompt in (cp2.PROMPT_V1, cp2.PROMPT_V2):
            self.assertEqual(set(prompt.input_variables), {"context", "question"})


if __name__ == "__main__":
    unittest.main()
