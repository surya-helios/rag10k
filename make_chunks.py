"""Stage 2: chunk every cleaned filing with each strategy and save to data/chunks/<strategy>.jsonl.

Usage:
    python make_chunks.py                       # both strategies
    python make_chunks.py --strategy fixed      # one strategy
Semantic chunking embeds every sentence, so it is slow on CPU; results are cached on disk so
indexing and re-runs never repeat it.
"""
import argparse
import json
import statistics
import time

import config
from chunking import build_chunks, get_embeddings

CHUNK_DIR = config.ROOT / "data" / "chunks"


def report(strategy: str, chunks: list[dict], embeddings) -> None:
    lens = sorted(len(c["text"]) for c in chunks)
    over = "n/a (embedding model not loaded)"
    if embeddings is not None:
        tok = embeddings._client.tokenizer
        over = sum(len(tok.encode(c["text"], add_special_tokens=True)) > 512 for c in chunks)
    print(f"[{strategy}] {len(chunks):,} chunks | chars min {lens[0]} / median {statistics.median(lens):.0f} / "
          f"p95 {lens[int(len(lens) * .95)]} / max {lens[-1]} | over 512 tokens: {over}")
    by_year = {}
    for c in chunks:
        by_year[c["metadata"]["fiscal_year"]] = by_year.get(c["metadata"]["fiscal_year"], 0) + 1
    print("   per fiscal year:", dict(sorted(by_year.items())))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", choices=config.STRATEGIES)
    args = ap.parse_args()

    docs = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(config.CLEAN_DIR.glob("*.json"))]
    if not docs:
        raise SystemExit("No cleaned filings found. Run prepare.py first.")
    strategies = [args.strategy] if args.strategy else config.STRATEGIES
    # Only semantic chunking (and the token-length report) needs the embedding model.
    embeddings = get_embeddings() if "semantic" in strategies else None
    CHUNK_DIR.mkdir(parents=True, exist_ok=True)

    for strategy in strategies:
        t0, chunks = time.time(), []
        for doc in docs:
            part = build_chunks(doc, strategy, embeddings)
            chunks.extend(part)
            print(f"  [{strategy}] FY{doc['meta']['fiscal_year']} {doc['meta']['form_type']}: {len(part)} chunks "
                  f"({time.time() - t0:.0f}s elapsed)", flush=True)
        with open(CHUNK_DIR / f"{strategy}.jsonl", "w", encoding="utf-8") as fh:
            for c in chunks:
                fh.write(json.dumps(c, ensure_ascii=False) + "\n")
        report(strategy, chunks, embeddings)


if __name__ == "__main__":
    main()
