"""
Bước 4 — Guardrails AI Validators
====================================
NHIỆM VỤ:
  1. Xây dựng PIIDetector: phát hiện & redact email, số điện thoại, SSN, số thẻ tín dụng
  2. Xây dựng JSONFormatter: tự động sửa JSON lỗi (fences, nháy đơn, dấu phẩy thừa)
  3. Bọc mỗi validator trong Guard và test với các mẫu đầu vào
  4. Chạy demo với 6 trường hợp PII và 5 trường hợp JSON

DELIVERABLE: Tất cả test cases pass (PII bị redact, JSON được sửa thành công)

CÁC KHÁI NIỆM CHÍNH:
  - @register_validator     — khai báo custom validator class
  - Validator.validate()    — implement logic kiểm tra + sửa
  - OnFailAction.FIX        — thay thế output thay vì raise error
  - Guard().use(validator)  — gắn validator instance vào guard
  - guard.validate(text)    → ValidationOutcome
      .validation_passed    — bool
      .validated_output     — output đã được xử lý

⚠️  QUAN TRỌNG: on_fail phải truyền vào CONSTRUCTOR của VALIDATOR, KHÔNG phải Guard.use()
    SAI  : Guard().use(PIIDetector, on_fail=OnFailAction.FIX)   ← TypeError
    ĐÚNG : Guard().use(PIIDetector(on_fail=OnFailAction.FIX))   ← correct
"""

import json
import os
import re
import sys
from pathlib import Path

# Đảm bảo UTF-8 cho console Windows
if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Tắt telemetry OTLP của Guardrails để loại bỏ cảnh báo mạng và tăng tốc độ xử lý
os.environ["OTEL_SDK_DISABLED"] = "true"
os.environ["OTEL_TRACES_EXPORTER"] = "none"

from guardrails import Guard
from guardrails.validators import FailResult, PassResult, Validator, register_validator

try:
    from guardrails.hub import OnFailAction
except ImportError:
    from guardrails.validator_base import OnFailAction


# ── 1. PII Detector Validator ──────────────────────────────────────────────
@register_validator(name="custom/pii-detector", data_type="string")
class PIIDetector(Validator):
    """
    Phát hiện và redact Personally Identifiable Information (PII).

    Các pattern được phát hiện:
      EMAIL       : xxx@xxx.xxx
      PHONE       : (123) 456-7890 hoặc 123-456-7890
      SSN         : 123-45-6789
      CREDIT_CARD : 1234 5678 9012 3456 (hoặc dấu gạch nối)
    """

    PII_PATTERNS = {
        "EMAIL":       r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
        "PHONE":       r"(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}\b",
        "SSN":         r"\b\d{3}-\d{2}-\d{4}\b",
        "CREDIT_CARD": r"\b(?:\d{4}[-\s]?){3}\d{4}\b",
    }

    def validate(self, value: str, metadata: dict = {}) -> PassResult | FailResult:
        """
        Tìm PII trong value; nếu phát hiện, redact và trả về FailResult với fix_value là text đã xử lý.

        ⚠️ Với OnFailAction.FIX, Guardrails CHỈ thay output bằng FailResult.fix_value.
           PassResult(value_override=...) KHÔNG có tác dụng → output giống hệt input.
        """
        redacted_text = value
        found_pii = []

        for pii_type, pattern in self.PII_PATTERNS.items():
            matches = re.findall(pattern, value)
            for match in matches:
                redacted_text = redacted_text.replace(match, f"[{pii_type}_REDACTED]")
                found_pii.append((pii_type, match))

        if found_pii:
            print(f"  ⚠️  Đã redact {len(found_pii)} PII: {[p[0] for p in found_pii]}")
            return FailResult(error_message="Phát hiện PII", fix_value=redacted_text)

        return PassResult()


# ── 2. JSON Formatter Validator ────────────────────────────────────────────
@register_validator(name="custom/json-formatter", data_type="string")
class JSONFormatter(Validator):
    """
    Validate và tự động sửa JSON lỗi.

    Các lỗi có thể sửa tự động:
      - Strip markdown code fences (``` hoặc ```json)
      - Thay single quotes → double quotes
      - Xóa trailing commas trước } hoặc ]
      - Re-serialize với json.dumps để định dạng chuẩn
    """

    @staticmethod
    def _repair(text: str) -> str:
        """
        Cố gắng sửa chuỗi JSON lỗi:
          1. Strip whitespace đầu/cuối
          2. Xóa markdown fences
          3. Thay single quotes → double quotes
          4. Xóa trailing commas trước } hoặc ]
        """
        text = text.strip()

        # Xóa markdown fences
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        text = text.strip()

        # Thay single quotes → double quotes
        text = text.replace("'", '"')

        # Xóa trailing commas trước } hoặc ]
        text = re.sub(r",\s*([}\]])", r"\1", text)

        return text

    def validate(self, value: str, metadata: dict = {}) -> PassResult | FailResult:
        """
        Thử parse value thành JSON:
          - JSON hợp lệ sẵn          → PassResult()
          - Sửa được                 → FailResult(error_message=..., fix_value=json.dumps(parsed, indent=2))
          - Không sửa được           → FailResult(error_message=..., fix_value=<JSON dự phòng>)
        """
        # 1. Thử parse JSON trực tiếp
        try:
            json.loads(value)
            return PassResult()
        except (json.JSONDecodeError, TypeError):
            pass

        # 2. Thử sửa JSON rồi parse lại
        try:
            repaired_text = self._repair(value)
            parsed = json.loads(repaired_text)
            print("  🔧 JSON đã được sửa thành công")
            return FailResult(
                error_message="JSON lỗi, đã tự sửa",
                fix_value=json.dumps(parsed, indent=2, ensure_ascii=False),
            )
        except (json.JSONDecodeError, TypeError):
            # 3. Không sửa được → trả về JSON dự phòng để output vẫn là JSON hợp lệ
            fallback = json.dumps(
                {"error": "Không thể phân tích JSON", "raw": value[:200]},
                ensure_ascii=False,
                indent=2,
            )
            return FailResult(error_message="Không thể sửa JSON", fix_value=fallback)


# ── 3. Demo: PII Guard ─────────────────────────────────────────────────────
def demo_pii_guard() -> bool:
    print("\n" + "=" * 55)
    print("  Demo: PII Detection & Redaction")
    print("=" * 55)

    guard = Guard().use(PIIDetector(on_fail=OnFailAction.FIX))

    test_cases = [
        ("Email",        "Contact John at john.doe@example.com for details."),
        ("Phone",        "Call our support line at (555) 867-5309."),
        ("SSN",          "Patient SSN is 123-45-6789 on file."),
        ("Credit Card",  "Payment made with card 4532 1234 5678 9010."),
        ("Multi-PII",    "Email: alice@example.com, Phone: 555-123-4567"),
        ("Clean",        "No sensitive information in this text."),
    ]

    log_lines = [
        "=" * 55,
        "  Demo: PII Detection & Redaction",
        "=" * 55,
    ]

    all_passed = True
    for label, text in test_cases:
        result = guard.validate(text)
        out = result.validated_output

        print(f"\n[{label}]")
        print(f"  Input:  {text}")
        print(f"  Output: {out}")

        log_lines.append(f"\n[{label}]")
        log_lines.append(f"  Input:  {text}")
        log_lines.append(f"  Output: {out}")

        # Xác minh output thực tế đã bị redact
        if label == "Email":
            assert "[EMAIL_REDACTED]" in out and "john.doe@example.com" not in out
        elif label == "Phone":
            assert "[PHONE_REDACTED]" in out and "867-5309" not in out
        elif label == "SSN":
            assert "[SSN_REDACTED]" in out and "123-45-6789" not in out
        elif label == "Credit Card":
            assert "[CREDIT_CARD_REDACTED]" in out and "4532" not in out
        elif label == "Multi-PII":
            assert "[EMAIL_REDACTED]" in out and "[PHONE_REDACTED]" in out
        elif label == "Clean":
            assert out == text

    # Lưu log evidence theo đúng chuẩn Checkpoint 4
    evidence_dir = Path(__file__).resolve().parent.parent / "evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    pii_log_path = evidence_dir / "04_pii_demo_log.txt"
    pii_log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    print(f"\n💾 Đã lưu evidence: {pii_log_path}")

    return all_passed


# ── 4. Demo: JSON Guard ────────────────────────────────────────────────────
def demo_json_guard() -> bool:
    print("\n" + "=" * 55)
    print("  Demo: JSON Formatting & Repair")
    print("=" * 55)

    guard = Guard().use(JSONFormatter(on_fail=OnFailAction.FIX))

    test_cases = [
        ("Valid JSON",       '{"name": "Alice", "age": 30}'),
        ("Markdown fences",  '```json\n{"name": "Bob"}\n```'),
        ("Single quotes",    "{'name': 'Charlie', 'score': 95}"),
        ("Trailing comma",   '{"key": "value",}'),
        ("Truly invalid",    "This is not JSON at all: ??? {]"),
    ]

    log_lines = [
        "=" * 55,
        "  Demo: JSON Formatting & Repair",
        "=" * 55,
    ]

    all_passed = True
    for label, text in test_cases:
        result = guard.validate(text)
        out = result.validated_output
        status = "✅ Pass" if result.validation_passed else "❌ Fail"

        print(f"\n[{label}] {status}")
        print(f"  Input:  {text}")
        print(f"  Output: {out}")

        log_lines.append(f"\n[{label}] {status}")
        log_lines.append(f"  Input:  {text}")
        log_lines.append(f"  Output: {out}")

        # Xác minh output luôn là JSON hợp lệ
        try:
            parsed = json.loads(out)
            assert isinstance(parsed, (dict, list))
            if label == "Valid JSON":
                assert parsed.get("name") == "Alice"
            elif label == "Markdown fences":
                assert parsed.get("name") == "Bob"
            elif label == "Single quotes":
                assert parsed.get("name") == "Charlie"
            elif label == "Trailing comma":
                assert parsed.get("key") == "value"
            elif label == "Truly invalid":
                assert "error" in parsed
        except Exception as exc:
            print(f"  ❌ Assertion error for [{label}]: {exc}")
            all_passed = False
            raise

    # Lưu log evidence theo đúng chuẩn Checkpoint 4
    evidence_dir = Path(__file__).resolve().parent.parent / "evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    json_log_path = evidence_dir / "04_json_demo_log.txt"
    json_log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    print(f"\n💾 Đã lưu evidence: {json_log_path}")

    return all_passed


# ── 5. Main ────────────────────────────────────────────────────────────────
def main():
    print("=" * 55)
    print("  Bước 4: Guardrails AI Validators")
    print("=" * 55)

    pii_ok = demo_pii_guard()
    json_ok = demo_json_guard()

    if pii_ok and json_ok:
        print("\n✅ Bước 4 hoàn thành! Tất cả test cases đều PASS.")
    else:
        print("\n⚠️ Bước 4 có test case thất bại.")


if __name__ == "__main__":
    main()
