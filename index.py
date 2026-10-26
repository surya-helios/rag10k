"""Stage 3: embed the saved chunks with bge and upsert them to Pinecone, one namespace per strategy.

Usage:
    python index.py                        # both strategies
    python index.py --strategy semantic
    python index.py --reset                # wipe the namespace(s) first
Chunk IDs are deterministic, so re-running overwrites instead of duplicating.
"""
import argparse
import json
import time

from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

import config
from make_chunks import CHUNK_DIR
from chunking import get_embeddings

BATCH = 128


def ensure_index(pc: Pinecone):
    existing = {i["name"]: i for i in pc.list_indexes()}
    if config.PINECONE_INDEX in existing:
        dim = pc.describe_index(config.PINECONE_INDEX).dimension
        if dim != config.EMBED_DIM:
            raise SystemExit(f"Index '{config.PINECONE_INDEX}' has dimension {dim}, but {config.EMBED_MODEL} "
                             f"produces {config.EMBED_DIM}. Use another PINECONE_INDEX name or delete the index.")
    else:
        pc.create_index(name=config.PINECONE_INDEX, dimension=config.EMBED_DIM, metric="cosine",
                        spec=ServerlessSpec(cloud=config.PINECONE_CLOUD, region=config.PINECONE_REGION))
        while not pc.describe_index(config.PINECONE_INDEX).status["ready"]:
            time.sleep(2)
        print(f"Created index '{config.PINECONE_INDEX}' ({config.EMBED_DIM} dims, cosine)")
    return pc.Index(config.PINECONE_INDEX)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", choices=config.STRATEGIES)
    ap.add_argument("--reset", action="store_true", help="delete the namespace(s) before indexing")
    args = ap.parse_args()

    pc = Pinecone(api_key=config.require("PINECONE_API_KEY"))
    index = ensure_index(pc)
    embeddings = get_embeddings()

    for strategy in ([args.strategy] if args.strategy else config.STRATEGIES):
        path = CHUNK_DIR / f"{strategy}.jsonl"
        if not path.exists():
            raise SystemExit(f"{path} not found. Run make_chunks.py first.")
        chunks = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()]
        if args.reset:
            try:
                index.delete(delete_all=True, namespace=strategy)
            except Exception as e:  # namespace may not exist yet
                print(f"  (reset skipped: {e})")
        store = PineconeVectorStore(index=index, embedding=embeddings, namespace=strategy)
        t0 = time.time()
        for i in range(0, len(chunks), BATCH):
            b = chunks[i:i + BATCH]
            store.add_texts([c["text"] for c in b], metadatas=[c["metadata"] for c in b],
                            ids=[c["id"] for c in b], batch_size=BATCH)
            print(f"  [{strategy}] {min(i + BATCH, len(chunks)):,}/{len(chunks):,} ({time.time() - t0:.0f}s)", flush=True)
        print(f"[{strategy}] done. Namespace stats: {index.describe_index_stats().namespaces.get(strategy)}")


if __name__ == "__main__":
    main()
