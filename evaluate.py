"""Stage 5: compare the four setups on GENERATED answers.

    A = fixed    + no rerank      B = fixed    + rerank
    C = semantic + no rerank      D = semantic + rerank
Only chunking and re-ranking vary; embeddings, TOP_K, prompt and Gemini models are identical.

Per answer:
  fact_recall    share of key_facts found in the answer          (deterministic)
  context_recall share of key_facts found in the retrieved text  (did retrieval even find it?)
  correctness    1-5 from a Gemini judge, against your reference answer
  faithfulness   1-5 from a Gemini judge, is the answer supported by the retrieved text
Everything is cached in results/cache.jsonl, so an interrupted run resumes where it stopped.

Usage: python evaluate.py [--limit N] [--setups A,B,C,D]
"""
import argparse
import hashlib
import json
import re

import pandas as pd

import config
from chunking import get_embeddings
from rag import answer, call_llm, format_context, is_daily_quota, make_llm, parse_scores
from retrieve import Retriever

SETUPS = {"A": ("fixed", False), "B": ("fixed", True), "C": ("semantic", False), "D": ("semantic", True)}
QUESTIONS = config.ROOT / "questions.json"
CACHE = config.RESULTS_DIR / "cache.jsonl"

JUDGE_PROMPT = """You are a strict grader of answers about Tesla's 10-K filings.

Question: {question}
Reference answer (ground truth): {reference}

Retrieved excerpts shown to the assistant:
{context}

Assistant's answer: {answer}

Score two things from 1 (worst) to 5 (best):
- correctness: does the answer agree with the reference on every fact and number? If the reference says the
  information is not in the filings, a correct answer states that it is not available. A wrong or missing figure
  must score 2 or lower.
- faithfulness: is every claim in the answer supported by the retrieved excerpts? Any figure not present in the
  excerpts must score 2 or lower.
Reply with JSON only: {{"correctness": <1-5>, "faithfulness": <1-5>}}"""


def norm(s: str) -> str:
    return re.sub(r"[\s,$*]", "", str(s).lower())


def recall(text: str, facts: list) -> float | None:
    """Each fact is a string, or a list of acceptable alternatives (e.g. ['96,773', '96.8 billion'])."""
    if not facts:
        return None
    t = norm(text)
    hit = sum(any(norm(alt) in t for alt in ([f] if isinstance(f, str) else f)) for f in facts)
    return hit / len(facts)


def load_cache() -> dict:
    if not CACHE.exists():
        return {}
    return {r["key"]: r for r in map(json.loads, CACHE.read_text(encoding="utf-8").splitlines())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, help="only the first N questions (smoke test)")
    ap.add_argument("--setups", default="A,B,C,D")
    args = ap.parse_args()

    questions = json.loads(QUESTIONS.read_text(encoding="utf-8"))
    questions = [q for q in questions if not q["question"].startswith("EXAMPLE")]
    if args.limit:
        questions = questions[:args.limit]
    if not questions:
        raise SystemExit("questions.json has no questions yet.")
    config.RESULTS_DIR.mkdir(exist_ok=True)
    cache = load_cache()

    embeddings = get_embeddings()
    gen_llm, judge_llm = make_llm(config.GEN_MODEL), make_llm(config.JUDGE_MODEL)
    retrievers = {}
    rows = []
    stopped = False
    for setup in args.setups.split(","):
        if stopped:
            break
        strategy, rerank = SETUPS[setup]
        if setup not in retrievers:
            retrievers[setup] = Retriever(strategy, rerank, embeddings)
        for i, q in enumerate(questions):
            qid = q.get("id", i + 1)
            # Multi-year questions keep TOP_K chunks per year (retrieve.py), so their cached results from the
            # older shared-TOP_K retrieval must not be reused; single-year keys are unchanged.
            per_year = "|per-year" if len(q.get("fiscal_years") or []) > 1 else ""
            sig = hashlib.md5((q["question"] + str(q.get("reference")) + config.GEN_MODEL + config.JUDGE_MODEL
                               + str(config.TOP_K) + per_year).encode()).hexdigest()[:10]
            key = f"{setup}|{qid}|{sig}"
            if key not in cache:
                try:
                    hits = retrievers[setup].retrieve(q["question"], q.get("fiscal_years"))
                    ans = answer(gen_llm, q["question"], hits)
                    scores = parse_scores(call_llm(judge_llm, JUDGE_PROMPT.format(
                        question=q["question"], reference=q["reference"], context=format_context(hits), answer=ans)))
                except Exception as e:
                    if not is_daily_quota(e):
                        raise
                    print(f"\nDaily request quota reached at [{setup}] q{qid}. Re-run tomorrow to resume; "
                          "saving results for everything finished so far.")
                    stopped = True
                    break
                rec = {"key": key, "setup": setup, "q_id": qid, "answer": ans, **scores,
                       "context": format_context(hits),
                       "sources": [f"FY{int(h['metadata']['fiscal_year'])} Item {h['metadata']['section']}" for h in hits]}
                with open(CACHE, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                cache[key] = rec
            r = cache[key]
            facts = q.get("key_facts", [])
            rows.append({
                "setup": setup, "strategy": strategy, "rerank": rerank, "q_id": qid,
                "type": q.get("type", ""), "wording": q.get("wording", "filing"),
                "found_in_testing": q.get("source") == "found_in_testing",
                "question": q["question"], "reference": q["reference"],
                "answer": r["answer"], "fact_recall": recall(r["answer"], facts),
                "context_recall": recall(r["context"], facts),
                "correctness": r["correctness"], "faithfulness": r["faithfulness"], "sources": "; ".join(r["sources"]),
            })
            print(f"[{setup}] q{qid}: correctness={r['correctness']} faithfulness={r['faithfulness']}", flush=True)

    if not rows:
        raise SystemExit("No results yet, nothing to save.")
    df = pd.DataFrame(rows)
    df.to_csv(config.RESULTS_DIR / "detail.csv", index=False)
    metrics = ["fact_recall", "context_recall", "correctness", "faithfulness"]
    summary = df.groupby("setup")[metrics].mean().round(3)
    summary.to_csv(config.RESULTS_DIR / "summary.csv")
    print("\n=== Average scores per setup (A fixed, B fixed+rerank, C semantic, D semantic+rerank) ===")
    print(summary.to_string())
    if df["type"].ne("").any():
        print("\n=== Correctness by question type ===")
        print(df.pivot_table(index="type", columns="setup", values="correctness").round(2).to_string())

    if df["wording"].nunique() > 1:
        by_wording = df.groupby(["wording", "setup"])[metrics].mean().round(3)
        by_wording.to_csv(config.RESULTS_DIR / "summary_by_wording.csv")
        print("\n=== Average scores by question wording ('filing' = the filing's own terms, 'casual' = everyday) ===")
        print(by_wording.to_string())
        if df["found_in_testing"].any():
            print("\n=== Correctness, casual questions excluding the ones found while testing the app ===")
            print(df[(df["wording"] == "casual") & ~df["found_in_testing"]].groupby("setup")["correctness"]
                  .mean().round(3).to_string())

    def wins(piv, a, b):
        if a in piv and b in piv:
            return f"{a} wins {(piv[a] > piv[b]).sum()}, {b} wins {(piv[b] > piv[a]).sum()}, ties {(piv[a] == piv[b]).sum()}"

    groups = [("all questions", df)] + ([(f"{w} wording", df[df["wording"] == w]) for w in sorted(df["wording"].unique())]
                                        if df["wording"].nunique() > 1 else [])
    for name, part in groups:
        piv = part.pivot(index="q_id", columns="setup", values="correctness")
        print(f"\n=== Head-to-head on correctness (per question), {name} ===")
        for label, a, b in [("chunking, no rerank (A vs C)", "A", "C"), ("chunking, with rerank (B vs D)", "B", "D"),
                            ("rerank effect, fixed (A vs B)", "A", "B"), ("rerank effect, semantic (C vs D)", "C", "D")]:
            out = wins(piv, a, b)
            if out:
                print(f"{label}: {out}")
    print(f"\nSaved results/detail.csv and results/summary.csv")


if __name__ == "__main__":
    main()
