"""
Chạy tất cả các bước lab theo thứ tự hoặc chỉ một bước cụ thể.

Cách dùng:
    python run_all.py            # chạy tất cả 4 bước
    python run_all.py --step 1   # chỉ chạy Bước 1
    python run_all.py --step 3   # chỉ chạy Bước 3 (RAGAS ~15-30 phút)
"""
import sys
import argparse
import importlib
from pathlib import Path

# Đảm bảo UTF-8 cho console Windows
if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).parent))


STEPS = {
    1: ("Bước 1: LangSmith RAG Pipeline",   "01_langsmith_rag_pipeline"),
    2: ("Bước 2: Prompt Hub & A/B Routing",  "02_prompt_hub_ab_routing"),
    3: ("Bước 3: RAGAS Evaluation",          "03_ragas_evaluation"),
    4: ("Bước 4: Guardrails AI Validators",  "04_guardrails_validator"),
}


def run_step(step_num: int):
    title, module_name = STEPS[step_num]
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")
    try:
        module = importlib.import_module(module_name)
        module.main()
        print(f"\n✅ {title} — HOÀN THÀNH")
        return True
    except SystemExit as e:
        if e.code != 0:
            print(f"\n❌ {title} — DỪNG (config thiếu hoặc lỗi)")
        return e.code == 0
    except Exception as e:
        print(f"\n❌ {title} — LỖI: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Chạy Day22 Lab: LangSmith + Prompt Versioning + RAGAS + Guardrails"
    )
    parser.add_argument(
        "--step", type=int, choices=[1, 2, 3, 4],
        help="Chỉ chạy bước được chỉ định (1-4)"
    )
    args = parser.parse_args()

    steps_to_run = [args.step] if args.step else list(STEPS.keys())

    results = {}
    for step_num in steps_to_run:
        success = run_step(step_num)
        results[step_num] = success
        if not success and not args.step:
            print(f"\n⛔ Dừng lại do Bước {step_num} thất bại.")
            break

    # Tổng kết
    print(f"\n{'=' * 60}")
    print("  Tổng kết")
    print(f"{'=' * 60}")
    for step_num, success in results.items():
        title = STEPS[step_num][0]
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {status}  {title}")


if __name__ == "__main__":
    main()
