"""
gemini_key_manager.py — Multi-Key Manager & Intelligent Rate Limit Handler for Google Gemini.

Tính năng chính:
  1. Hỗ trợ 3 API keys (GEMINI_API_KEY_1, GEMINI_API_KEY_2, GEMINI_API_KEY_3).
  2. Tương thích ngược với GOOGLE_API_KEY.
  3. Tuyệt đối không log / in / làm rò rỉ API key ra logs, exceptions, reports, traces.
  4. Quản lý quota group theo Google Cloud Project ID (GEMINI_PROJECT_ID_1/2/3).
  5. Xử lý bảo thủ: nếu project ID chưa được cấu hình, coi các key thuộc cùng một project group.
  6. Health status từng key: AVAILABLE, COOLDOWN, EXHAUSTED, INVALID.
  7. Phân tích chi tiết lỗi 429: phân biệt short-term rate limit (15 RPM) và daily quota (500 req/day).
  8. Tôn trọng retryDelay, không retry vô vọng khi phải chờ nhiều giờ.
  9. Hỗ trợ failover giữa các project độc lập khi một project hết quota.
  10. Thread-safe & Async-safe.
"""

from __future__ import annotations

import asyncio
import os
import re
import sys
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.callbacks.manager import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatResult


# ── 1. Enums & Data Structures ──────────────────────────────────────────────

class KeyStatus(str, Enum):
    AVAILABLE = "AVAILABLE"    # Sẵn sàng nhận request
    COOLDOWN  = "COOLDOWN"     # Rate limit ngắn hạn (ví dụ: 15 RPM), đợi vài giây
    EXHAUSTED = "EXHAUSTED"    # Daily quota cạn kiệt (500 req/day), đợi qua ngày / retryDelay
    INVALID   = "INVALID"      # Lỗi xác thực (401 / 403 API key không hợp lệ)


@dataclass
class GeminiKeyInfo:
    """Thông tin và trạng thái sức khỏe của một Gemini API key."""
    index: int                          # 1, 2, 3
    name: str                           # "Key-1", "Key-2", "Key-3"
    api_key: str = field(repr=False)    # Secret — KHÔNG BAO GIỜ hiển thị trong repr
    project_id: str = "unverified-shared"
    status: KeyStatus = KeyStatus.AVAILABLE
    cooldown_until: float = 0.0         # Unix timestamp khi kết thúc cooldown ngắn hạn
    exhausted_until: float = 0.0        # Unix timestamp khi reset daily quota
    active_requests: int = 0            # Số request đồng thời đang xử lý
    success_count: int = 0              # Số request thành công
    failure_count: int = 0              # Số request thất bại
    last_error_message: str = ""        # Lỗi gần nhất (đã che giấu secret)
    last_quota_metric: str = ""         # quotaMetric từ Google API
    last_quota_id: str = ""             # quotaId từ Google API
    last_retry_delay: float = 0.0       # retryDelay theo giây

    @property
    def masked_name(self) -> str:
        """Định danh an toàn để log (không để lộ nội dung key)."""
        suffix = f"...{self.api_key[-4:]}" if len(self.api_key) >= 8 else "***"
        return f"{self.name} (proj: {self.project_id}, {suffix})"

    def __repr__(self) -> str:
        # Đảm bảo không bao giờ in api_key kể cả khi gọi repr()
        return (
            f"<GeminiKeyInfo {self.name} index={self.index} "
            f"project={self.project_id} status={self.status.value} "
            f"active={self.active_requests} success={self.success_count} "
            f"fail={self.failure_count}>"
        )

    def __str__(self) -> str:
        return self.__repr__()


@dataclass
class ParsedErrorInfo:
    """Kết quả phân tích lỗi từ Gemini API."""
    is_auth_error: bool = False
    is_rate_limit: bool = False
    is_daily_quota: bool = False
    quota_metric: str = ""
    quota_id: str = ""
    retry_delay_seconds: float = 0.0
    sanitized_message: str = ""


class AllProjectsQuotaExhaustedError(RuntimeError):
    """Ném ra khi toàn bộ project / keys đã cạn kiệt daily quota."""
    def __init__(self, message: str, retry_after_seconds: float = 0.0):
        super().__init__(message)
        self.retry_after_seconds = retry_after_seconds


# ── 2. Tiện ích che giấu & phân tích lỗi ────────────────────────────────────

def sanitize_text(text: str, secret_keys: Optional[List[str]] = None) -> str:
    """Loại bỏ triệt để các API keys hoặc mẫu chuỗi nhạy cảm khỏi text."""
    if not text:
        return ""
    sanitized = str(text)

    # Thay thế các keys cụ thể đã biết
    if secret_keys:
        for k in secret_keys:
            if k and len(k) >= 6:
                sanitized = sanitized.replace(k, "[REDACTED_GEMINI_KEY]")

    # Thay thế các chuỗi khớp mẫu Google API Key phổ biến (AIzaSy...)
    sanitized = re.sub(r"AIza[0-9A-Za-z_-]{35}", "[REDACTED_GEMINI_KEY]", sanitized)
    sanitized = re.sub(r"(api[_-]?key=)[^&\s]+", r"\1[REDACTED]", sanitized, flags=re.IGNORECASE)
    sanitized = re.sub(r"('api_key':\s*')[^']+'", r"\1[REDACTED]'", sanitized)
    return sanitized


def parse_retry_delay_string(delay_str: str) -> float:
    """
    Chuyển đổi các định dạng thời gian retry từ Gemini sang giây:
      '58147s' -> 58147.0
      '16h9m7.829s' -> 58147.829
      '15s' -> 15.0
    """
    if not delay_str:
        return 0.0
    s = str(delay_str).strip()

    # Định dạng thuần số + 's' (ví dụ '58147s')
    m_simple = re.match(r"^(\d+(?:\.\d+)?)\s*s$", s, flags=re.IGNORECASE)
    if m_simple:
        return float(m_simple.group(1))

    # Định dạng phức hợp: NhXmYs
    total = 0.0
    found_any = False
    m_h = re.search(r"(\d+(?:\.\d+)?)\s*h", s, flags=re.IGNORECASE)
    if m_h:
        total += float(m_h.group(1)) * 3600.0
        found_any = True
    m_m = re.search(r"(\d+(?:\.\d+)?)\s*m", s, flags=re.IGNORECASE)
    if m_m:
        total += float(m_m.group(1)) * 60.0
        found_any = True
    m_s = re.search(r"(\d+(?:\.\d+)?)\s*s", s, flags=re.IGNORECASE)
    if m_s:
        total += float(m_s.group(1))
        found_any = True

    if found_any:
        return total

    # Thử ép kiểu số nguyên / thực
    try:
        return float(s)
    except ValueError:
        return 0.0


def parse_gemini_error(exc: Exception, secret_keys: Optional[List[str]] = None) -> ParsedErrorInfo:
    """
    Phân tích exception từ LangChain / Google GenAI để trích xuất:
    - QuotaFailure (quotaMetric, quotaId)
    - RetryInfo (retryDelay)
    - Phân biệt short-term RPM limit vs daily quota exhaustion.
    """
    raw_str = str(exc)
    sanitized = sanitize_text(raw_str, secret_keys)
    err_lower = raw_str.lower()

    info = ParsedErrorInfo(sanitized_message=sanitized)

    # 1. Kiểm tra Authentication / Invalid Key
    if (
        "401" in raw_str
        or "403" in raw_str
        or "api_key_invalid" in err_lower
        or "api key not valid" in err_lower
        or "permission_denied" in err_lower
    ):
        info.is_auth_error = True
        return info

    # 2. Kiểm tra Rate Limit / Quota Exceeded (429 / RESOURCE_EXHAUSTED)
    if (
        "429" in raw_str
        or "resource_exhausted" in err_lower
        or "quota" in err_lower
        or "rate limit" in err_lower
    ):
        info.is_rate_limit = True

        # Trích xuất quotaMetric
        m_metric = re.search(r"['\"]quotaMetric['\"]\s*:\s*['\"]([^'\"]+)['\"]", raw_str)
        if m_metric:
            info.quota_metric = m_metric.group(1)
        else:
            m_metric_text = re.search(r"Quota exceeded for metric:\s*([^\s,]+)", raw_str)
            if m_metric_text:
                info.quota_metric = m_metric_text.group(1)

        # Trích xuất quotaId
        m_id = re.search(r"['\"]quotaId['\"]\s*:\s*['\"]([^'\"]+)['\"]", raw_str)
        if m_id:
            info.quota_id = m_id.group(1)

        # Trích xuất retryDelay
        m_delay_rpc = re.search(r"['\"]retryDelay['\"]\s*:\s*['\"]([^'\"]+)['\"]", raw_str)
        if m_delay_rpc:
            info.retry_delay_seconds = parse_retry_delay_string(m_delay_rpc.group(1))
        else:
            m_retry_text = re.search(r"retry in\s+([0-9hms\.]+)", raw_str, flags=re.IGNORECASE)
            if m_retry_text:
                info.retry_delay_seconds = parse_retry_delay_string(m_retry_text.group(1))

        # 3. Phân biệt Daily Quota vs Short-Term Rate Limit
        # Các dấu hiệu của daily quota:
        #  - quotaId chứa 'perday' hoặc 'daily'
        #  - retry_delay_seconds > 300 (ví dụ 58147s = 16h)
        #  - quotaMetric chứa 'free_tier_requests' và limit: 500
        #  - message có 'limit: 500' hoặc 'per day'
        is_daily = False
        if "perday" in info.quota_id.lower() or "daily" in info.quota_id.lower():
            is_daily = True
        elif info.retry_delay_seconds > 300.0:
            is_daily = True
        elif "limit: 500" in raw_str or "limit: 500" in err_lower:
            is_daily = True
        elif "generate_content_free_tier_requests" in info.quota_metric and info.retry_delay_seconds > 60.0:
            is_daily = True

        info.is_daily_quota = is_daily

        # Nếu không có retryDelay mà là short-term limit, đặt mặc định 15 giây (an toàn cho 15 RPM)
        if not is_daily and info.retry_delay_seconds <= 0.0:
            info.retry_delay_seconds = 15.0

    return info


# ── 3. GeminiKeyManager Core ────────────────────────────────────────────────

class GeminiKeyManager:
    """
    Quản lý danh sách Gemini API keys, theo dõi quota và xử lý rate limit thông minh.
    Thread-safe và async-safe.
    """

    def __init__(
        self,
        keys: Optional[List[str]] = None,
        project_ids: Optional[List[str]] = None,
        max_concurrency_per_key: int = 1,
        default_short_cooldown: float = 15.0,
    ):
        self._lock = threading.RLock()
        self.max_concurrency_per_key = max(1, max_concurrency_per_key)
        self.default_short_cooldown = default_short_cooldown
        self._round_robin_counter = 0

        # Khởi tạo danh sách keys
        self.keys: List[GeminiKeyInfo] = []
        if keys:
            self._init_keys(keys, project_ids)

    @classmethod
    def from_env(cls, max_concurrency_per_key: int = 1) -> GeminiKeyManager:
        """Đọc 3 keys và project IDs từ biến môi trường .env."""
        raw_keys: List[str] = []
        raw_projects: List[str] = []

        # 1. Đọc GEMINI_API_KEY_1, 2, 3
        for i in range(1, 4):
            val = (os.getenv(f"GEMINI_API_KEY_{i}") or "").strip()
            if val:
                raw_keys.append(val)
                proj = (os.getenv(f"GEMINI_PROJECT_ID_{i}") or "").strip()
                raw_projects.append(proj if proj else "")

        # 2. Tương thích ngược: nếu không có GEMINI_API_KEY_*, đọc GOOGLE_API_KEY
        if not raw_keys:
            fallback = (os.getenv("GOOGLE_API_KEY") or "").strip()
            if fallback:
                raw_keys.append(fallback)
                proj = (os.getenv("GEMINI_PROJECT_ID") or os.getenv("GOOGLE_CLOUD_PROJECT") or "").strip()
                raw_projects.append(proj if proj else "")

        manager = cls(
            keys=raw_keys,
            project_ids=raw_projects,
            max_concurrency_per_key=max_concurrency_per_key,
        )
        return manager

    def _init_keys(self, keys: List[str], project_ids: Optional[List[str]] = None) -> None:
        """Thiết lập thông tin ban đầu cho các keys."""
        cleaned_keys = [k.strip() for k in keys if k and k.strip()]
        if not cleaned_keys:
            return

        p_ids = list(project_ids or [])
        while len(p_ids) < len(cleaned_keys):
            p_ids.append("")

        has_any_explicit_project = any(bool(p) for p in p_ids)

        self.keys = []
        for idx, key in enumerate(cleaned_keys, start=1):
            explicit_proj = p_ids[idx - 1].strip()
            # Yêu cầu 5: Không giả định 3 keys có quota độc lập.
            # Nếu project ID chưa xác minh, gom vào chung một project group: 'unverified-shared'
            proj_id = explicit_proj if explicit_proj else "unverified-shared"

            key_info = GeminiKeyInfo(
                index=idx,
                name=f"Key-{idx}",
                api_key=key,
                project_id=proj_id,
                status=KeyStatus.AVAILABLE,
            )
            self.keys.append(key_info)

        if not has_any_explicit_project and len(self.keys) > 1:
            print(
                "[KeyManager] ⚠️ CHÚ Ý: Chưa cấu hình GEMINI_PROJECT_ID_1/2/3. "
                "Hệ thống sẽ coi các keys thuộc chung project group ('unverified-shared') "
                "để tránh vượt giới hạn 500 req/ngày của Google Cloud project.",
                file=sys.stderr,
                flush=True,
            )

    @property
    def total_keys(self) -> int:
        return len(self.keys)

    @property
    def secret_keys_list(self) -> List[str]:
        """Danh sách các secret keys cần được scrub khỏi logs."""
        return [k.api_key for k in self.keys]

    # ── 4. Key Selection & Acquisition ──────────────────────────────────────

    def _refresh_cooldowns(self, now: float) -> None:
        """Đưa các key đã hết thời gian cooldown ngắn hạn trở lại trạng thái AVAILABLE."""
        for k in self.keys:
            if k.status == KeyStatus.COOLDOWN and now >= k.cooldown_until:
                k.status = KeyStatus.AVAILABLE
                k.cooldown_until = 0.0

    def select_available_key(self) -> Optional[GeminiKeyInfo]:
        """
        Chọn một key khả dụng theo thuật toán Round-Robin và Least-Loaded.
        Phải được gọi bên trong self._lock.
        """
        now = time.time()
        self._refresh_cooldowns(now)

        # 1. Tìm các key có trạng thái AVAILABLE và chưa vượt quá concurrency limit
        candidates = [
            k for k in self.keys
            if k.status == KeyStatus.AVAILABLE
            and k.active_requests < self.max_concurrency_per_key
        ]

        if not candidates:
            return None

        # 2. Sắp xếp ưu tiên: ít active request nhất, sau đó dùng round-robin
        self._round_robin_counter += 1
        num_candidates = len(candidates)
        candidates.sort(key=lambda k: (k.active_requests, (k.index + self._round_robin_counter) % num_candidates))
        return candidates[0]

    def acquire_key_sync(self, timeout: float = 60.0) -> GeminiKeyInfo:
        """
        Lấy một key khả dụng (chờ nếu cần cho đến khi hết timeout).
        Raise AllProjectsQuotaExhaustedError nếu mọi project đều cạn quota.
        """
        start_time = time.time()
        while True:
            with self._lock:
                now = time.time()
                self._refresh_cooldowns(now)

                # Kiểm tra xem có còn key nào sống không
                available_or_cooling = [
                    k for k in self.keys
                    if k.status in (KeyStatus.AVAILABLE, KeyStatus.COOLDOWN)
                ]

                if not available_or_cooling:
                    # Tất cả các key đều EXHAUSTED hoặc INVALID
                    exhausted_keys = [k for k in self.keys if k.status == KeyStatus.EXHAUSTED]
                    max_exhaust_wait = max((k.exhausted_until - now for k in exhausted_keys), default=0.0)
                    raise AllProjectsQuotaExhaustedError(
                        f"Tất cả Gemini API keys ({len(self.keys)} keys) đều đã cạn kiệt daily quota "
                        f"(500 req/day). Vui lòng đợi reset hoặc thêm key từ Google Cloud project khác.",
                        retry_after_seconds=max(0.0, max_exhaust_wait),
                    )

                key = self.select_available_key()
                if key is not None:
                    key.active_requests += 1
                    return key

                # Nếu tất cả các key đang trong COOLDOWN, tính thời gian chờ nhỏ nhất
                cooling_keys = [k for k in self.keys if k.status == KeyStatus.COOLDOWN]
                if cooling_keys:
                    min_wait = min(max(0.1, k.cooldown_until - now) for k in cooling_keys)
                else:
                    min_wait = 0.5  # Chờ slot concurrency trống

            if (time.time() - start_time) + min_wait > timeout:
                raise TimeoutError(f"Hết thời gian chờ ({timeout}s) để lấy Gemini API key khả dụng.")

            sleep_dur = min(min_wait, 2.0)
            time.sleep(sleep_dur)

    async def acquire_key_async(self, timeout: float = 60.0) -> GeminiKeyInfo:
        """Phiên bản async của acquire_key."""
        start_time = time.time()
        while True:
            with self._lock:
                now = time.time()
                self._refresh_cooldowns(now)

                available_or_cooling = [
                    k for k in self.keys
                    if k.status in (KeyStatus.AVAILABLE, KeyStatus.COOLDOWN)
                ]

                if not available_or_cooling:
                    exhausted_keys = [k for k in self.keys if k.status == KeyStatus.EXHAUSTED]
                    max_exhaust_wait = max((k.exhausted_until - now for k in exhausted_keys), default=0.0)
                    raise AllProjectsQuotaExhaustedError(
                        f"Tất cả Gemini API keys ({len(self.keys)} keys) đều đã cạn kiệt daily quota. "
                        f"Vui lòng đợi reset quota.",
                        retry_after_seconds=max(0.0, max_exhaust_wait),
                    )

                key = self.select_available_key()
                if key is not None:
                    key.active_requests += 1
                    return key

                cooling_keys = [k for k in self.keys if k.status == KeyStatus.COOLDOWN]
                if cooling_keys:
                    min_wait = min(max(0.1, k.cooldown_until - now) for k in cooling_keys)
                else:
                    min_wait = 0.5

            if (time.time() - start_time) + min_wait > timeout:
                raise TimeoutError(f"Hết thời gian chờ async ({timeout}s) để lấy Gemini API key khả dụng.")

            sleep_dur = min(min_wait, 2.0)
            await asyncio.sleep(sleep_dur)

    def release_key(self, key_index: int) -> None:
        """Giải phóng slot concurrency cho key."""
        with self._lock:
            for k in self.keys:
                if k.index == key_index:
                    k.active_requests = max(0, k.active_requests - 1)
                    break

    # ── 5. Feedback & Recording ─────────────────────────────────────────────

    def record_success(self, key_index: int) -> None:
        """Ghi nhận một request thành công cho key."""
        with self._lock:
            for k in self.keys:
                if k.index == key_index:
                    k.success_count += 1
                    k.active_requests = max(0, k.active_requests - 1)
                    break

    def record_error(self, key_index: int, exc: Exception) -> ParsedErrorInfo:
        """
        Xử lý khi key gặp lỗi:
          - Parse lỗi (auth, rate limit, daily quota).
          - Cập nhật trạng thái key (INVALID, COOLDOWN, EXHAUSTED).
          - Nếu là daily quota, đánh dấu TOÀN BỘ các key chung project_id là EXHAUSTED.
        """
        info = parse_gemini_error(exc, self.secret_keys_list)
        now = time.time()

        with self._lock:
            target_key: Optional[GeminiKeyInfo] = None
            for k in self.keys:
                if k.index == key_index:
                    target_key = k
                    break

            if not target_key:
                return info

            target_key.failure_count += 1
            target_key.active_requests = max(0, target_key.active_requests - 1)
            target_key.last_error_message = info.sanitized_message
            target_key.last_quota_metric = info.quota_metric
            target_key.last_quota_id = info.quota_id
            target_key.last_retry_delay = info.retry_delay_seconds

            # 1. Xử lý lỗi xác thực (401 / 403 API_KEY_INVALID)
            if info.is_auth_error:
                target_key.status = KeyStatus.INVALID
                print(
                    f"[KeyManager] ❌ {target_key.name} gặp lỗi xác thực API key. "
                    f"Đã vô hiệu hóa {target_key.name}.",
                    file=sys.stderr,
                    flush=True,
                )
                return info

            # 2. Xử lý lỗi cạn kiệt daily quota (500 req/day/project/model)
            if info.is_daily_quota:
                target_key.status = KeyStatus.EXHAUSTED
                target_key.exhausted_until = now + (info.retry_delay_seconds if info.retry_delay_seconds > 0 else 86400.0)
                delay_hours = target_key.last_retry_delay / 3600.0 if target_key.last_retry_delay else 24.0

                print(
                    f"\n[KeyManager] 🛑 {target_key.name} ĐÃ HẾT DAILY QUOTA "
                    f"(metric: {info.quota_metric or 'free_tier_requests'}, "
                    f"retryDelay: {target_key.last_retry_delay:.0f}s ~ {delay_hours:.1f}h).",
                    file=sys.stderr,
                    flush=True,
                )

                # Yêu cầu 5 & 8 & 9: Các key cùng project chia sẻ chung quota.
                # Đánh dấu toàn bộ key thuộc cùng project group là EXHAUSTED!
                affected_group = target_key.project_id
                for k in self.keys:
                    if k.project_id == affected_group and k.index != target_key.index:
                        k.status = KeyStatus.EXHAUSTED
                        k.exhausted_until = target_key.exhausted_until
                        print(
                            f"[KeyManager] 🛑 {k.name} thuộc cùng project group ('{affected_group}') "
                            f"với {target_key.name} -> Đánh dấu EXHAUSTED để tránh vi phạm quota chung.",
                            file=sys.stderr,
                            flush=True,
                        )
                return info

            # 3. Xử lý short-term rate limit (ví dụ: 15 RPM)
            if info.is_rate_limit:
                target_key.status = KeyStatus.COOLDOWN
                delay = max(info.retry_delay_seconds, self.default_short_cooldown)
                target_key.cooldown_until = now + delay
                print(
                    f"[KeyManager] ⏳ {target_key.name} gặp short-term rate limit (RPM). "
                    f"Tạm dừng (cooldown) {delay:.1f} giây...",
                    file=sys.stderr,
                    flush=True,
                )
                return info

        return info

    # ── 6. Trạng thái & Báo cáo ─────────────────────────────────────────────

    def get_status_summary(self) -> Dict[str, Any]:
        """Tóm tắt trạng thái của toàn bộ keys dưới dạng dict an toàn (không chứa secret)."""
        now = time.time()
        with self._lock:
            self._refresh_cooldowns(now)
            key_summaries = []
            for k in self.keys:
                cooldown_left = max(0.0, k.cooldown_until - now) if k.status == KeyStatus.COOLDOWN else 0.0
                exhausted_left = max(0.0, k.exhausted_until - now) if k.status == KeyStatus.EXHAUSTED else 0.0
                key_summaries.append({
                    "index": k.index,
                    "name": k.name,
                    "project_id": k.project_id,
                    "status": k.status.value,
                    "active_requests": k.active_requests,
                    "success_count": k.success_count,
                    "failure_count": k.failure_count,
                    "cooldown_remaining_sec": round(cooldown_left, 1),
                    "exhausted_remaining_sec": round(exhausted_left, 1),
                })
            return {
                "total_keys": len(self.keys),
                "available_keys": sum(1 for k in self.keys if k.status == KeyStatus.AVAILABLE),
                "keys": key_summaries,
            }

    def format_status_table(self) -> str:
        """In bảng trạng thái định dạng đẹp cho terminal."""
        summary = self.get_status_summary()
        lines = [
            "┌──────┬────────┬─────────────────────────┬──────────────┬────────┬────────┬────────────────────────┐",
            "│ Index│ Name   │ Project / Quota Group   │ Status       │ Active │ OK/Fail│ Wait / Cooldown        │",
            "├──────┼────────┼─────────────────────────┼──────────────┼────────┼────────┼────────────────────────┤",
        ]
        for k in summary["keys"]:
            wait_info = "-"
            if k["status"] == "COOLDOWN":
                wait_info = f"Cooling: {k['cooldown_remaining_sec']}s"
            elif k["status"] == "EXHAUSTED":
                h = k['exhausted_remaining_sec'] / 3600.0
                wait_info = f"Exhausted: {h:.1f}h"
            elif k["status"] == "INVALID":
                wait_info = "AUTH ERROR"

            lines.append(
                f"│ {k['index']:<4} │ {k['name']:<6} │ {k['project_id'][:23]:<23} │ "
                f"{k['status']:<12} │ {k['active_requests']:<6} │ "
                f"{k['success_count']}/{k['failure_count']:<4} │ {wait_info:<22} │"
            )
        lines.append(
            "└──────┴────────┴─────────────────────────┴──────────────┴────────┴────────┴────────────────────────┘"
        )
        return "\n".join(lines)


# ── 7. LangChain Multi-Key Gemini Adapters ──────────────────────────────────

class MultiKeyGeminiChat(BaseChatModel):
    """
    LangChain BaseChatModel adapter tích hợp GeminiKeyManager.
    Mỗi request lấy key khả dụng từ KeyManager, không sửa đổi os.environ toàn cục,
    tự động retry hoặc failover giữa các project keys độc lập khi gặp lỗi quota.
    """
    model_name: str
    temperature: float = 0.0
    key_manager: Any = None
    max_retries_per_call: int = 3
    _clients: Dict[str, Any] = {}

    def __init__(
        self,
        model_name: str,
        temperature: float = 0.0,
        key_manager: Optional[GeminiKeyManager] = None,
        max_retries_per_call: int = 3,
        **kwargs: Any,
    ):
        km = key_manager or get_key_manager()
        super().__init__(
            model_name=model_name,
            temperature=temperature,
            key_manager=km,
            max_retries_per_call=max_retries_per_call,
            **kwargs,
        )
        object.__setattr__(self, "_clients", {})

    @property
    def _llm_type(self) -> str:
        return "chat-google-generative-ai-multi-key"

    @property
    def model(self) -> str:
        """Tương thích với Ragas usage tracking."""
        return self.model_name

    def _get_client_for_key(self, api_key: str):
        """Lấy hoặc tạo client ChatGoogleGenerativeAI riêng cho từng key."""
        if api_key not in self._clients:
            from langchain_google_genai import ChatGoogleGenerativeAI
            self._clients[api_key] = ChatGoogleGenerativeAI(
                model=self.model_name,
                google_api_key=api_key,
                temperature=self.temperature,
            )
        return self._clients[api_key]

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        attempt = 0
        last_exc: Optional[Exception] = None

        while attempt < self.max_retries_per_call:
            attempt += 1
            key_info = self.key_manager.acquire_key_sync()
            client = self._get_client_for_key(key_info.api_key)

            try:
                result = client._generate(messages, stop=stop, **kwargs)
                self.key_manager.record_success(key_info.index)
                return result
            except Exception as exc:
                last_exc = exc
                err_info = self.key_manager.record_error(key_info.index, exc)

                # Nếu là short-term limit và còn lượt thử, chờ thời gian cooldown rồi tiếp tục
                if err_info.is_rate_limit and not err_info.is_daily_quota and attempt < self.max_retries_per_call:
                    sleep_time = min(max(err_info.retry_delay_seconds, 5.0), 30.0)
                    time.sleep(sleep_time)
                    continue

                # Nếu là daily quota nhưng còn project khác khả dụng, loop sẽ tự acquire key từ project khác
                if err_info.is_daily_quota and attempt < self.max_retries_per_call:
                    continue

                # Nếu không retry được nữa, ném lỗi ra
                raise

        if last_exc:
            raise last_exc
        raise RuntimeError("Không thể thực hiện request tới Gemini.")

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        attempt = 0
        last_exc: Optional[Exception] = None

        while attempt < self.max_retries_per_call:
            attempt += 1
            key_info = await self.key_manager.acquire_key_async()
            client = self._get_client_for_key(key_info.api_key)

            try:
                result = await client._agenerate(messages, stop=stop, **kwargs)
                self.key_manager.record_success(key_info.index)
                return result
            except Exception as exc:
                last_exc = exc
                err_info = self.key_manager.record_error(key_info.index, exc)

                if err_info.is_rate_limit and not err_info.is_daily_quota and attempt < self.max_retries_per_call:
                    sleep_time = min(max(err_info.retry_delay_seconds, 5.0), 30.0)
                    await asyncio.sleep(sleep_time)
                    continue

                if err_info.is_daily_quota and attempt < self.max_retries_per_call:
                    continue

                raise

        if last_exc:
            raise last_exc
        raise RuntimeError("Không thể thực hiện request async tới Gemini.")


class MultiKeyGeminiEmbeddings(Embeddings):
    """
    LangChain Embeddings adapter tích hợp GeminiKeyManager.
    """
    def __init__(
        self,
        model_name: str,
        key_manager: Optional[GeminiKeyManager] = None,
    ):
        self.model_name = model_name
        self.key_manager = key_manager or get_key_manager()
        self._clients: Dict[str, Any] = {}

    def _get_client_for_key(self, api_key: str):
        if api_key not in self._clients:
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
            self._clients[api_key] = GoogleGenerativeAIEmbeddings(
                model=self.model_name,
                google_api_key=api_key,
            )
        return self._clients[api_key]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        key_info = self.key_manager.acquire_key_sync()
        client = self._get_client_for_key(key_info.api_key)
        try:
            vectors = client.embed_documents(texts)
            self.key_manager.record_success(key_info.index)
            return vectors
        except Exception as exc:
            self.key_manager.record_error(key_info.index, exc)
            raise

    def embed_query(self, text: str) -> List[float]:
        key_info = self.key_manager.acquire_key_sync()
        client = self._get_client_for_key(key_info.api_key)
        try:
            vector = client.embed_query(text)
            self.key_manager.record_success(key_info.index)
            return vector
        except Exception as exc:
            self.key_manager.record_error(key_info.index, exc)
            raise


# ── 8. Global Singleton Accessors ───────────────────────────────────────────

_GLOBAL_KEY_MANAGER: Optional[GeminiKeyManager] = None
_GLOBAL_LOCK = threading.Lock()


def get_key_manager() -> GeminiKeyManager:
    """Trả về global singleton GeminiKeyManager instance."""
    global _GLOBAL_KEY_MANAGER
    with _GLOBAL_LOCK:
        if _GLOBAL_KEY_MANAGER is None:
            _GLOBAL_KEY_MANAGER = GeminiKeyManager.from_env()
        return _GLOBAL_KEY_MANAGER


def reset_key_manager() -> None:
    """Reset singleton (dùng khi chạy unit test)."""
    global _GLOBAL_KEY_MANAGER
    with _GLOBAL_LOCK:
        _GLOBAL_KEY_MANAGER = None


def get_gemini_chat_model(
    model: Optional[str] = None,
    temperature: float = 0.0,
    key_manager: Optional[GeminiKeyManager] = None,
) -> MultiKeyGeminiChat:
    """Helper factory tạo MultiKeyGeminiChat."""
    import config
    target_model = model or config.GEMINI_MODEL
    return MultiKeyGeminiChat(
        model_name=target_model,
        temperature=temperature,
        key_manager=key_manager,
    )


def get_gemini_embeddings(
    model: Optional[str] = None,
    key_manager: Optional[GeminiKeyManager] = None,
) -> MultiKeyGeminiEmbeddings:
    """Helper factory tạo MultiKeyGeminiEmbeddings."""
    import config
    target_model = model or config.GEMINI_EMBEDDING_MODEL
    return MultiKeyGeminiEmbeddings(
        model_name=target_model,
        key_manager=key_manager,
    )
