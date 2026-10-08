"""Checkpoint 2 - LangSmith Prompt Hub and deterministic A/B routing."""
import argparse
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# config must be imported before LangChain so tracing environment variables exist.
import config

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langsmith import Client, traceable
from langsmith.utils import LangSmithConflictError

from qa_pairs import SAMPLE_QUESTIONS
from utils.data_loader import build_vectorstore, load_knowledge_base, split_text
from utils.llm_factory import get_embeddings, get_llm


PROMPT_V1_NAME = "namnt1204-day22-rag-v1"
PROMPT_V2_NAME = "namnt1204-day22-rag-v2"


SYSTEM_V1 = (
    "You are a concise RAG assistant. Answer in 2-4 sentences using only facts "
    "explicitly supported by the context. Do not add outside knowledge or make "
    "unsupported inferences. If the context does not contain enough evidence, "
    "state clearly that the information is not available in the provided context.\n\n"
    "Context:\n{context}"
)
PROMPT_V1 = ChatPromptTemplate.from_messages(
    [("system", SYSTEM_V1), ("human", "{question}")]
)

SYSTEM_V2 = (
    "You are an evidence-focused RAG analyst. Use only information supported by "
    "the context and prioritize a complete account of all relevant evidence. "
    "Write a clear, structured answer in 3-5 sentences: give the direct answer "
    "first, then the supporting details. Never speculate or invent facts; when "
    "the evidence is insufficient, explicitly say the information is not available "
    "in the provided context.\n\n"
    "Context:\n{context}"
)
PROMPT_V2 = ChatPromptTemplate.from_messages(
    [("system", SYSTEM_V2), ("human", "{question}")]
)


def push_prompts_to_hub(client: Client) -> dict:
    """Push both prompts independently and surface every real Hub error."""
    prompt_specs = (
        (
            "V1",
            PROMPT_V1_NAME,
            PROMPT_V1,
            "Day 22 RAG V1 - concise 2-4 sentence answers grounded strictly in context.",
        ),
        (
            "V2",
            PROMPT_V2_NAME,
            PROMPT_V2,
            "Day 22 RAG V2 - structured 3-5 sentence answers with complete supported evidence.",
        ),
    )
    results, errors = {}, []

    for label, name, prompt, description in prompt_specs:
        try:
            url = client.push_prompt(name, object=prompt, description=description)
            results[name] = url
            print(f"✅ Pushed {label} '{name}' → {url}")
        except LangSmithConflictError as exc:
            # A repeated run may return 409 when its manifest matches the latest
            # commit. Ignore only that documented no-op, never other conflicts.
            if "nothing to commit" not in str(exc).lower():
                errors.append((name, exc))
                print(f"❌ Failed to push {label} '{name}': {exc}")
                continue
            results[name] = "unchanged (already on Hub)"
            print(f"ℹ️  {label} '{name}' unchanged; latest Hub commit retained.")
        except Exception as exc:
            errors.append((name, exc))
            print(f"❌ Failed to push {label} '{name}': {exc}")

    if errors:
        details = "; ".join(f"{name}: {error}" for name, error in errors)
        raise RuntimeError(
            f"Could not push all CP2 prompts to LangSmith Hub: {details}"
        ) from errors[0][1]
    return results


def pull_prompts_from_hub(
    client: Client, allow_local_fallback: bool = False
) -> dict:
    """Pull both Hub prompts; evidence mode (default) forbids local fallback."""
    prompts, errors = {}, []
    local_prompts = {PROMPT_V1_NAME: PROMPT_V1, PROMPT_V2_NAME: PROMPT_V2}

    for name in (PROMPT_V1_NAME, PROMPT_V2_NAME):
        try:
            prompts[name] = client.pull_prompt(name, skip_cache=True)
            print(f"↓ Pulled '{name}' successfully from LangSmith Hub")
        except Exception as exc:
            if allow_local_fallback:
                prompts[name] = local_prompts[name]
                print(
                    f"⚠️  LOCAL FALLBACK for '{name}': {exc}\n"
                    "   CP2 HUB CRITERIA NOT MET - do not use this run as evidence."
                )
            else:
                errors.append((name, exc))
                print(f"❌ Failed to pull '{name}' from Hub: {exc}")

    if errors:
        details = "; ".join(f"{name}: {error}" for name, error in errors)
        raise RuntimeError(
            "CP2 requires both prompts to be pulled from LangSmith Hub; " + details
        ) from errors[0][1]
    return prompts


def get_prompt_version(request_id: str) -> str:
    """Route an ID by MD5 parity: even to V1, odd to V2."""
    hash_int = int(hashlib.md5(request_id.encode("utf-8")).hexdigest(), 16)
    return PROMPT_V1_NAME if hash_int % 2 == 0 else PROMPT_V2_NAME


def verify_deterministic_routing(repetitions: int = 5) -> dict:
    """Assert repeated IDs are stable and the 50-question set covers both arms."""
    assignments = {}
    for i in range(len(SAMPLE_QUESTIONS)):
        request_id = f"req-{i:04d}"
        routed = {get_prompt_version(request_id) for _ in range(repetitions)}
        if len(routed) != 1:
            raise AssertionError(f"Non-deterministic routing for {request_id}: {routed}")
        assignments[request_id] = routed.pop()

    counts = {
        PROMPT_V1_NAME: sum(v == PROMPT_V1_NAME for v in assignments.values()),
        PROMPT_V2_NAME: sum(v == PROMPT_V2_NAME for v in assignments.values()),
    }
    if not all(counts.values()):
        raise AssertionError(f"Routing sample does not cover both versions: {counts}")
    print(
        "✅ Deterministic routing test passed "
        f"({repetitions} repeats/ID): V1={counts[PROMPT_V1_NAME]}, "
        f"V2={counts[PROMPT_V2_NAME]}"
    )
    return counts


@traceable(name="ab-rag-query", tags=["ab-test", "step2"])
def ask_ab(retriever, llm, prompt, question: str, version: str) -> dict:
    """Retrieve top-3 context, invoke the selected Hub prompt, and label output."""
    docs = retriever.invoke(question)
    context = "\n\n".join(doc.page_content for doc in docs)
    answer = (prompt | llm | StrOutputParser()).invoke(
        {"context": context, "question": question}
    )
    return {"question": question, "answer": answer, "version": version}


def setup_vectorstore():
    embeddings = get_embeddings()
    text = load_knowledge_base()
    chunks = split_text(text)
    return build_vectorstore(chunks, embeddings)


def main(limit: int = None):
    print("=" * 60)
    print("  Step 2: Prompt Hub & A/B Routing")
    print("=" * 60)

    if not config.validate():
        sys.exit(1)

    verify_deterministic_routing()
    client = Client(api_key=config.LANGSMITH_API_KEY)
    hub_results = push_prompts_to_hub(client)
    prompts = pull_prompts_from_hub(client, allow_local_fallback=False)

    vectorstore = setup_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm = get_llm()

    questions = SAMPLE_QUESTIONS[:limit] if limit else SAMPLE_QUESTIONS
    v1_count, v2_count, success_count = 0, 0, 0
    for i, question in enumerate(questions):
        request_id = f"req-{i:04d}"
        version_key = get_prompt_version(request_id)
        version_tag = "v1" if version_key == PROMPT_V1_NAME else "v2"
        prompt = prompts[version_key]

        result = ask_ab(retriever, llm, prompt, question, version_tag)
        success_count += 1
        if version_tag == "v1":
            v1_count += 1
        else:
            v2_count += 1
        print(
            f"[{i + 1:02d}/{len(questions)}] [prompt-{result['version']}] "
            f"{question[:55]}..."
        )

    print(f"\n📊 Routing: V1={v1_count} | V2={v2_count} | Total={len(questions)}")
    print(f"✅ Successful traces: {success_count}/{len(questions)}")
    print("Hub prompts:")
    for name, url in hub_results.items():
        print(f"   - {name}: {url}")
    print("✅ Step 2 complete. Check Prompt Hub and traces in LangSmith.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Step 2: Prompt Hub & deterministic A/B routing"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Question count for a smoke test; default runs all 50 questions.",
    )
    args = parser.parse_args()
    main(limit=args.limit)
