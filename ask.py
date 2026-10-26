"""Ask a free-form question.

Usage:
    python ask.py "What was Tesla's total revenue in 2023?" --strategy semantic --rerank --years 2023
"""
import argparse

import config
from rag import answer, make_llm
from retrieve import Retriever


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--strategy", choices=config.STRATEGIES, default="semantic")
    ap.add_argument("--rerank", action="store_true")
    ap.add_argument("--years", type=int, nargs="*", help="restrict to these fiscal years")
    args = ap.parse_args()

    hits = Retriever(args.strategy, args.rerank).retrieve(args.question, args.years)
    print(answer(make_llm(config.GEN_MODEL), args.question, hits))
    print("\nSources:")
    for h in hits:
        m = h["metadata"]
        print(f"  FY{int(m['fiscal_year'])} {m['form_type']} Item {m['section']} (sim {h['score']:.3f})")


if __name__ == "__main__":
    main()
