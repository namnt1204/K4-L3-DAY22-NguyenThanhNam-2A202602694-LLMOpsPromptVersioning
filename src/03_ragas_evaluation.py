"""
Bước 3 — RAGAS Evaluation với Checkpoint & Resume thông minh.
=============================================================
Tính năng:
  1. Tách biệt 2 phase:
     - Phase 1: Sinh câu trả lời RAG cho V1 và V2 (lưu vào cache/checkpoint).
     - Phase 2: Chấm 4 RAGAS metrics (lưu kết quả từng sample vào checkpoint).
  2. Cơ chế Resume:
     - Bỏ qua các answer đã sinh thành công.
     - Bỏ qua các metric đã có điểm hợp lệ (không phải None / NaN).
     - Ghi checkpoint atomic (tránh hỏng file JSON khi bị ngắt giữa chừng).
  3. Xử lý Quota & Rate Limit:
     - Tích hợp GeminiKeyManager (3 keys, quota group theo project).
     - Concurrency an toàn (mặc định 1 worker) để tránh chạm trần 15 RPM.
     - Dừng an toàn khi cạn kiệt daily quota (500 req/day).
     - Xuất partial report nếu chưa hoàn thành toàn bộ 50 QA pairs.
  4. Hỗ trợ CLI:
     - python src/03_ragas_evaluation.py --status
     - python src/03_ragas_evaluation.py --smoke-test
     - python src/03_ragas_evaluation.py --resume
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

# Đảm bảo UTF-8 cho console Windows
if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

warnings.filterwarnings("ignore")

# Thêm thư mục gốc src vào sys.path
SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_DIR))

import config  # ⚠️ PHẢI import trước LangChain

# Tắt LangSmith tracing khi chạy RAGAS evaluation để tránh lỗi ConnectTimeout và multipart flood
os.environ["LANGCHAIN_TRACING_V2"] = "false"

import numpy as np
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from ragas import EvaluationDataset, SingleTurnSample, evaluate
from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness
from ragas.run_config import RunConfig

from qa_pairs import QA_PAIRS
from utils.data_loader import build_vectorstore, load_knowledge_base, split_text
from utils.gemini_key_manager import (
    AllProjectsQuotaExhaustedError,
    KeyStatus,
    get_key_manager,
)
from utils.llm_factory import get_embeddings, get_llm


# ── DualStreamLogger ────────────────────────────────────────────────────────
class DualStreamLogger:
    """Nhân đôi output ra cả console và file log UTF-8."""
    def __init__(self, filepath: Path):
        self.terminal = sys.stdout
        filepath.parent.mkdir(parents=True, exist_ok=True)
        self.log_file = open(filepath, "w", encoding="utf-8", buffering=1)

    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()

    def close(self):
        if hasattr(self, "log_file") and not self.log_file.closed:
            self.log_file.close()


# ── Prompt Templates ────────────────────────────────────────────────────────
SYSTEM_V1 = (
    "You are a concise RAG assistant. Answer in 2-4 sentences using only facts "
    "explicitly supported by the context. Do not add outside knowledge or make "
    "unsupported inferences. If the context does not contain enough evidence, "
    "state clearly that the information is not available in the provided context.\n\n"
    "Context:\n{context}"
)
PROMPT_V1 = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_V1),
    ("human",  "{question}"),
])

SYSTEM_V2 = (
    "You are an evidence-focused RAG analyst. Use only information supported by "
    "the context and prioritize a complete account of all relevant evidence. "
    "Write a clear, structured answer in 3-5 sentences: give the direct answer "
    "first, then the supporting details. Never speculate or invent facts; when "
    "the evidence is insufficient, explicitly say the information is not available "
    "in the provided context.\n\n"
    "Context:\n{context}"
)
PROMPT_V2 = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_V2),
    ("human",  "{question}"),
])

PROMPTS = {"v1": PROMPT_V1, "v2": PROMPT_V2}
ALL_METRICS = [faithfulness, answer_relevancy, context_recall, context_precision]
METRIC_NAMES = ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]


# ── Atomic Checkpoint Manager ───────────────────────────────────────────────
DEFAULT_CHECKPOINT_PATH = Path(__file__).resolve().parent.parent / "data" / "checkpoints" / "ragas_checkpoint.json"


def atomic_save_json(filepath: Path, data: dict) -> None:
    """Ghi dữ liệu JSON an toàn (atomic write) thông qua file tạm và os.replace."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    temp_path = filepath.with_suffix(".tmp")
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp_path, filepath)


def load_checkpoint(filepath: Path = DEFAULT_CHECKPOINT_PATH) -> dict:
    """Tải checkpoint nếu tồn tại, hoặc tạo cấu trúc rỗng ban đầu."""
    if filepath.exists():
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict) and "answers" in data and "evaluations" in data:
                    return data
        except Exception as exc:
            print(f"[Checkpoint] ⚠️ Lỗi khi đọc {filepath}: {exc}. Khởi tạo checkpoint mới.")

    return {
        "version": 1,
        "config": {
            "evaluator_model": getattr(config, "GEMINI_MODEL", "gemini-3.8-flash"),
            "embedding_model": getattr(config, "GEMINI_EMBEDDING_MODEL", "models/gemini-embedding-001"),
            "provider": getattr(config, "PROVIDER", "gemini"),
        },
        "answers": {
            "v1": {},
            "v2": {},
        },
        "evaluations": {
            "v1": {},
            "v2": {},
        },
    }


def is_valid_score(val: Any) -> bool:
    """Kiểm tra một điểm đánh giá có phải là số thực hợp lệ (không None, không NaN)."""
    if val is None:
        return False
    if isinstance(val, (int, float)):
        return not np.isnan(float(val))
    return False


# ── Vectorstore Setup ───────────────────────────────────────────────────────
def setup_vectorstore():
    """Tải knowledge base và tạo vectorstore FAISS."""
    print("📂 Đang nạp knowledge base và vectorstore FAISS...", flush=True)
    embeddings = get_embeddings()
    text = load_knowledge_base()
    chunks = split_text(text)
    return build_vectorstore(chunks, embeddings)


# ── Phase 1: Sinh RAG Answers có Checkpoint ─────────────────────────────────
def run_rag_single(retriever, llm, prompt, question: str, max_retries: int = 3) -> dict:
    """Chạy 1 câu hỏi RAG: trả về answer (str) và contexts (list of strings)."""
    docs = retriever.invoke(question)
    contexts = [doc.page_content for doc in docs]
    ctx_str = "\n\n".join(contexts)

    chain = prompt | llm | StrOutputParser()
    raw_answer = chain.invoke({
        "context": ctx_str,
        "question": question,
    })
    return {"answer": str(raw_answer).strip(), "contexts": contexts}


def collect_rag_answers_with_checkpoint(
    vectorstore,
    prompt_version: str,
    checkpoint: dict,
    checkpoint_path: Path,
    limit: Optional[int] = None,
    delay_between_requests: float = 1.0,
) -> List[dict]:
    """
    Phase 1: Thu thập câu trả lời cho prompt version.
    Nếu đã có trong checkpoint thì bỏ qua, nếu chưa thì gọi LLM và lưu ngay lập tức.
    """
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm = get_llm()
    prompt = PROMPTS[prompt_version]
    pairs = QA_PAIRS[:limit] if limit else QA_PAIRS

    answers_cache = checkpoint["answers"].setdefault(prompt_version, {})
    results: List[dict] = []

    print(f"\n🚀 [Phase 1] Thu thập câu trả lời RAG cho prompt {prompt_version.upper()} ({len(pairs)} câu hỏi)...", flush=True)
    cached_count = 0
    generated_count = 0

    for i, qa in enumerate(pairs, 1):
        sample_id = f"qa_{i:02d}"

        # 1. Kiểm tra cache/checkpoint
        existing = answers_cache.get(sample_id)
        if existing and existing.get("answer") and existing.get("contexts"):
            cached_count += 1
            print(f"  [{i:02d}/{len(pairs)}] [CACHE] {qa['question'][:60]}", flush=True)
            results.append(existing)
            continue

        # 2. Sinh câu trả lời mới
        try:
            out = run_rag_single(retriever, llm, prompt, qa["question"])
            rec = {
                "sample_id": sample_id,
                "prompt_version": prompt_version,
                "question": qa["question"],
                "reference": qa["reference"],
                "answer": out["answer"],
                "contexts": out["contexts"],
                "model": getattr(config, "GEMINI_MODEL", "gemini-3.8-flash"),
                "timestamp": time.time(),
            }
            answers_cache[sample_id] = rec
            atomic_save_json(checkpoint_path, checkpoint)
            results.append(rec)
            generated_count += 1
            print(f"  [{i:02d}/{len(pairs)}] [NEW]   {qa['question'][:60]}", flush=True)

            if delay_between_requests > 0:
                time.sleep(delay_between_requests)

        except AllProjectsQuotaExhaustedError as quota_err:
            print(f"\n🛑 Hết quota trong lúc sinh câu trả lời tại câu [{i:02d}]: {quota_err}", flush=True)
            atomic_save_json(checkpoint_path, checkpoint)
            raise
        except Exception as exc:
            print(f"\n❌ Lỗi khi sinh câu trả lời câu [{i:02d}]: {exc}", flush=True)
            atomic_save_json(checkpoint_path, checkpoint)
            raise

    print(
        f"  -> Hoàn thành Phase 1 ({prompt_version.upper()}): {len(results)}/{len(pairs)} sẵn sàng "
        f"({cached_count} từ cache, {generated_count} mới sinh).",
        flush=True,
    )
    return results


# ── Phase 2: Chấm điểm RAGAS Metrics có Checkpoint ──────────────────────────
def evaluate_single_sample_metrics(
    sample_record: dict,
    metrics_to_run: List[Any],
    llm_eval,
    emb_eval,
    run_config: RunConfig,
) -> Dict[str, Optional[float]]:
    """Đánh giá 1 sample record với danh sách metrics được chỉ định."""
    sample = SingleTurnSample(
        user_input=sample_record["question"],
        response=sample_record["answer"],
        retrieved_contexts=sample_record["contexts"],
        reference=sample_record["reference"],
    )
    dataset = EvaluationDataset(samples=[sample])

    old_tracing = os.environ.get("LANGCHAIN_TRACING_V2", "true")
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
    try:
        result = evaluate(
            dataset,
            metrics=metrics_to_run,
            llm=llm_eval,
            embeddings=emb_eval,
            run_config=run_config,
            raise_exceptions=False,
        )
        scores: Dict[str, Optional[float]] = {}
        for m in metrics_to_run:
            raw = result[m.name]
            if isinstance(raw, (int, float)):
                val = float(raw)
            elif isinstance(raw, list) and len(raw) > 0:
                val = raw[0]
            else:
                val = None

            if is_valid_score(val):
                scores[m.name] = float(val)
            else:
                scores[m.name] = None
        return scores
    finally:
        os.environ["LANGCHAIN_TRACING_V2"] = old_tracing


def evaluate_ragas_with_checkpoint(
    rag_results: List[dict],
    prompt_version: str,
    checkpoint: dict,
    checkpoint_path: Path,
    concurrency: int = 1,
    pacing_delay: float = 1.5,
) -> Dict[str, float]:
    """
    Phase 2: Chấm điểm 4 RAGAS metrics cho từng sample.
    Lưu điểm sau mỗi sample vào checkpoint. Bỏ qua các metric đã có điểm hợp lệ.
    """
    print(f"\n📐 [Phase 2] Đánh giá RAGAS cho prompt {prompt_version.upper()} ({len(rag_results)} mẫu)...", flush=True)

    llm_eval = get_llm(temperature=0.0)
    emb_eval = get_embeddings()
    run_config = RunConfig(max_workers=max(1, concurrency), max_retries=3, timeout=120)
    key_mgr = get_key_manager()

    eval_cache = checkpoint["evaluations"].setdefault(prompt_version, {})
    total_samples = len(rag_results)

    for i, rec in enumerate(rag_results, 1):
        sample_id = rec["sample_id"]
        sample_eval = eval_cache.setdefault(sample_id, {
            "sample_id": sample_id,
            "scores": {},
            "timestamp": time.time(),
        })

        scores_dict = sample_eval.setdefault("scores", {})

        # Xác định metric nào còn thiếu hoặc chưa có điểm hợp lệ
        needed_metrics = []
        for m in ALL_METRICS:
            val = scores_dict.get(m.name)
            if not is_valid_score(val):
                needed_metrics.append(m)

        if not needed_metrics:
            # Tất cả 4 metrics của sample này đã có điểm hợp lệ!
            score_summary = ", ".join(f"{k}: {v:.4f}" for k, v in scores_dict.items() if is_valid_score(v))
            print(f"  [{i:02d}/{total_samples}] [CACHE] {sample_id} ({score_summary})", flush=True)
            continue

        needed_names = [m.name for m in needed_metrics]
        print(f"  [{i:02d}/{total_samples}] [EVAL]  {sample_id} -> Chấm {len(needed_metrics)} metrics: {needed_names} ...", flush=True)

        try:
            new_scores = evaluate_single_sample_metrics(
                sample_record=rec,
                metrics_to_run=needed_metrics,
                llm_eval=llm_eval,
                emb_eval=emb_eval,
                run_config=run_config,
            )

            # Cập nhật điểm hợp lệ
            updated_any = False
            for m_name, val in new_scores.items():
                if is_valid_score(val):
                    scores_dict[m_name] = float(val)
                    updated_any = True

            sample_eval["timestamp"] = time.time()
            if updated_any:
                atomic_save_json(checkpoint_path, checkpoint)

            # Kiểm tra xem có gặp lỗi cạn kiệt daily quota không
            summary = key_mgr.get_status_summary()
            if summary["available_keys"] == 0:
                all_exhausted = all(k["status"] == KeyStatus.EXHAUSTED.value for k in summary["keys"])
                if all_exhausted:
                    print("\n🛑 Tất cả Gemini API keys đều đã cạn kiệt daily quota. Lưu tiến độ và dừng an toàn.", flush=True)
                    atomic_save_json(checkpoint_path, checkpoint)
                    break

            if pacing_delay > 0:
                time.sleep(pacing_delay)

        except AllProjectsQuotaExhaustedError as quota_err:
            print(f"\n🛑 Hết daily quota tại sample {sample_id}: {quota_err}", flush=True)
            atomic_save_json(checkpoint_path, checkpoint)
            break
        except Exception as exc:
            print(f"  ⚠️ Lỗi khi đánh giá {sample_id}: {exc}", flush=True)
            atomic_save_json(checkpoint_path, checkpoint)

    # Tính điểm trung bình của các sample đã hoàn thành hợp lệ
    final_scores: Dict[str, float] = {}
    for m_name in METRIC_NAMES:
        valid_vals = []
        for s_id, s_data in eval_cache.items():
            sc = s_data.get("scores", {}).get(m_name)
            if is_valid_score(sc):
                valid_vals.append(float(sc))

        if valid_vals:
            final_scores[m_name] = float(np.mean(valid_vals))
        else:
            final_scores[m_name] = 0.0

    print(f"\n📊 Kết quả tính toán — Prompt {prompt_version.upper()}:", flush=True)
    for k, v in final_scores.items():
        star = " ⭐" if k == "faithfulness" and v >= 0.8 else ""
        print(f"  {k:30s}: {v:.4f}{star}", flush=True)

    return final_scores


# ── Status Reporter ─────────────────────────────────────────────────────────
def print_status(checkpoint_path: Path = DEFAULT_CHECKPOINT_PATH) -> None:
    """In trạng thái checkpoint và key manager mà không tiêu thụ API call nào."""
    print("=" * 70)
    print("  BÁO CÁO TIẾN ĐỘ RAGAS EVALUATION & GEMINI KEY STATUS")
    print("=" * 70)

    key_mgr = get_key_manager()
    print("\n🔑 Trạng thái Gemini API Keys:")
    print(key_mgr.format_status_table())

    if not checkpoint_path.exists():
        print(f"\n📂 Checkpoint file: CHƯA CÓ ({checkpoint_path})")
        print("   Tiến độ: 0/50 câu trả lời V1, 0/50 câu trả lời V2, 0 evaluations.")
        return

    checkpoint = load_checkpoint(checkpoint_path)
    print(f"\n📂 Checkpoint file: ĐÃ CÓ ({checkpoint_path})")
    eval_model = checkpoint.get("config", {}).get("evaluator_model", "unknown")
    print(f"   Evaluator Model: {eval_model}")

    ans_v1 = len(checkpoint.get("answers", {}).get("v1", {}))
    ans_v2 = len(checkpoint.get("answers", {}).get("v2", {}))
    print(f"   Phase 1 (Answers): V1 = {ans_v1}/50, V2 = {ans_v2}/50")

    print("\n   Phase 2 (Evaluations theo từng metric):")
    for ver in ["v1", "v2"]:
        evals = checkpoint.get("evaluations", {}).get(ver, {})
        print(f"     Prompt {ver.upper()}:")
        for m_name in METRIC_NAMES:
            valid_count = sum(1 for s in evals.values() if is_valid_score(s.get("scores", {}).get(m_name)))
            print(f"       - {m_name:<20}: {valid_count}/50 hoàn thành")

    # Đếm số evaluations còn thiếu
    total_target = 50 * 2 * 4  # 400 jobs
    completed_evals = 0
    for ver in ["v1", "v2"]:
        evals = checkpoint.get("evaluations", {}).get(ver, {})
        for s in evals.values():
            for m_name in METRIC_NAMES:
                if is_valid_score(s.get("scores", {}).get(m_name)):
                    completed_evals += 1

    remaining = max(0, total_target - completed_evals)
    pct = (completed_evals / total_target) * 100.0 if total_target > 0 else 0.0
    print(f"\n   Tổng tiến độ Phase 2: {completed_evals}/{total_target} jobs ({pct:.1f}%). Còn thiếu: {remaining} evaluations.")
    print("=" * 70)


# ── Main Controller ─────────────────────────────────────────────────────────
def main(
    limit: Optional[int] = None,
    resume: bool = True,
    smoke_test: bool = False,
    concurrency: int = 1,
    pacing_delay: float = 1.5,
) -> Optional[dict]:
    # Nếu là smoke-test, chạy nhanh 2 samples
    if smoke_test:
        limit = 2

    evidence_log_path = Path(__file__).resolve().parent.parent / "evidence" / "03_ragas_run_log.txt"
    logger = DualStreamLogger(evidence_log_path)
    original_stdout = sys.stdout
    sys.stdout = logger

    checkpoint_path = DEFAULT_CHECKPOINT_PATH

    try:
        print("=" * 65, flush=True)
        print("  Bước 3: RAGAS Evaluation (Multi-Key & Checkpoint Resume)", flush=True)
        print("=" * 65, flush=True)

        if not config.validate():
            sys.exit(1)

        key_mgr = get_key_manager()
        print(f"\n🔑 Gemini Key Manager ({key_mgr.total_keys} keys):", flush=True)
        print(key_mgr.format_status_table(), flush=True)
        print(f"📂 Checkpoint file: {checkpoint_path}", flush=True)

        # Tải checkpoint
        checkpoint = load_checkpoint(checkpoint_path)

        # Kiểm tra tính nhất quán của model trong checkpoint
        curr_model = getattr(config, "GEMINI_MODEL", "gemini-3.8-flash")
        ckpt_model = checkpoint.get("config", {}).get("evaluator_model", curr_model)
        if ckpt_model != curr_model:
            print(
                f"[Cảnh báo] Model trong checkpoint ({ckpt_model}) khác với cấu hình hiện tại ({curr_model}). "
                f"Sử dụng {ckpt_model} để đảm bảo tính nhất quán.",
                flush=True,
            )

        # 1. Thiết lập vectorstore
        vectorstore = setup_vectorstore()

        # 2. Phase 1: Thu thập câu trả lời RAG cho V1 và V2
        v1_results = collect_rag_answers_with_checkpoint(
            vectorstore, "v1", checkpoint, checkpoint_path, limit=limit
        )
        v2_results = collect_rag_answers_with_checkpoint(
            vectorstore, "v2", checkpoint, checkpoint_path, limit=limit
        )

        # 3. Phase 2: Đánh giá RAGAS cho V1 và V2
        v1_scores = evaluate_ragas_with_checkpoint(
            v1_results, "v1", checkpoint, checkpoint_path,
            concurrency=concurrency, pacing_delay=pacing_delay
        )
        v2_scores = evaluate_ragas_with_checkpoint(
            v2_results, "v2", checkpoint, checkpoint_path,
            concurrency=concurrency, pacing_delay=pacing_delay
        )

        # 4. In bảng so sánh V1 vs V2
        print("\n" + "=" * 65, flush=True)
        print(f"  {'Metric':30s}  {'V1':>8}  {'V2':>8}  Winner", flush=True)
        print("=" * 65, flush=True)
        for metric in METRIC_NAMES:
            s1, s2 = v1_scores[metric], v2_scores[metric]
            if abs(s1 - s2) < 1e-4:
                winner = "TIE"
            else:
                winner = "← V1" if s1 > s2 else "V2 →"
            print(f"  {metric:30s}  {s1:>8.4f}  {s2:>8.4f}  {winner}", flush=True)
        print("=" * 65, flush=True)

        best_faith = max(v1_scores["faithfulness"], v2_scores["faithfulness"])
        if best_faith >= 0.8:
            print(f"\n✅ Đạt mục tiêu: faithfulness = {best_faith:.4f} ≥ 0.8", flush=True)
        else:
            print(f"\n⚠️  Chưa đạt mục tiêu ({best_faith:.4f} < 0.8).", flush=True)

        # 5. Kiểm tra tính đầy đủ trước khi xuất báo cáo chính thức
        target_count = limit if limit is not None else 50
        v1_evals = checkpoint.get("evaluations", {}).get("v1", {})
        v2_evals = checkpoint.get("evaluations", {}).get("v2", {})

        is_complete = True
        missing_details = []
        for ver_name, ev_dict in [("V1", v1_evals), ("V2", v2_evals)]:
            for i in range(1, target_count + 1):
                sid = f"qa_{i:02d}"
                s_scores = ev_dict.get(sid, {}).get("scores", {})
                for m in METRIC_NAMES:
                    if not is_valid_score(s_scores.get(m)):
                        is_complete = False
                        missing_details.append(f"{ver_name}-{sid}-{m}")

        data_dir = Path(__file__).resolve().parent.parent / "data"
        evidence_dir = Path(__file__).resolve().parent.parent / "evidence"
        data_dir.mkdir(parents=True, exist_ok=True)
        evidence_dir.mkdir(parents=True, exist_ok=True)

        if is_complete:
            # Toàn bộ evaluations hợp lệ -> Xuất data/ragas_report.json chính thức
            report = {
                "prompt_v1_scores": v1_scores,
                "prompt_v2_scores": v2_scores,
                "target_met": bool(best_faith >= 0.8),
                "sample_count": target_count,
            }
            report_path = data_dir / "ragas_report.json"
            report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"\n💾 Đã lưu báo cáo hoàn chỉnh vào {report_path}", flush=True)

            evidence_report_path = evidence_dir / "03_ragas_report.json"
            evidence_report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"💾 Đã sao lưu báo cáo hoàn chỉnh vào {evidence_report_path}", flush=True)
            return report
        else:
            # Chưa hoàn thành đủ -> Lưu partial report, không ghi đè fake scores vào ragas_report.json
            partial_report = {
                "status": "PARTIAL",
                "completed_samples_target": target_count,
                "missing_count": len(missing_details),
                "missing_evaluations": missing_details[:20],
                "partial_v1_scores": v1_scores,
                "partial_v2_scores": v2_scores,
            }
            partial_path = data_dir / "ragas_partial_report.json"
            partial_path.write_text(json.dumps(partial_report, indent=2, ensure_ascii=False), encoding="utf-8")
            print(
                f"\n⚠️  Chưa hoàn thành toàn bộ evaluations ({len(missing_details)} evaluations còn thiếu). "
                f"Đã lưu partial report vào {partial_path}. File ragas_report.json chính thức chưa được ghi đè.",
                flush=True,
            )
            return None

    finally:
        sys.stdout = original_stdout
        logger.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bước 3: RAGAS Evaluation với Resume & Multi-Key")
    parser.add_argument(
        "--status",
        action="store_true",
        help="Kiểm tra trạng thái tiến độ checkpoint và API keys, không gọi API",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Chạy thử nghiệm nhanh tối đa 2 samples để kiểm tra pipeline",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Tiếp tục chạy đánh giá từ checkpoint đã lưu",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Số lượng QA pairs cần chạy (mặc định 50)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Số luồng đánh giá đồng thời (mặc định 1 để tránh rate limit 15 RPM)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.5,
        help="Khoảng thời gian chờ (giây) giữa các requests",
    )

    args = parser.parse_args()

    if args.status:
        print_status()
    else:
        main(
            limit=args.limit,
            resume=args.resume,
            smoke_test=args.smoke_test,
            concurrency=args.concurrency,
            pacing_delay=args.delay,
        )
