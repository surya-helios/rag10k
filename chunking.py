"""The two chunking strategies being compared. Everything else in the pipeline is held constant."""
import re

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings

import config

MIN_CHUNK_CHARS = 40  # drop fragments that carry no meaning on their own
_NL = " "        # stands in for newlines so the semantic splitter can break on them and we can restore them


def get_embeddings() -> HuggingFaceEmbeddings:
    import torch
    return HuggingFaceEmbeddings(
        model_name=config.EMBED_MODEL,
        model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
        encode_kwargs={"normalize_embeddings": True, "batch_size": 64},
    )


def fixed_chunks(text: str) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.FIXED_CHUNK_SIZE, chunk_overlap=config.FIXED_CHUNK_OVERLAP)
    return splitter.split_text(text)


def semantic_chunks(text: str, embeddings) -> list[str]:
    """Split where the meaning shifts between neighbouring sentences / table rows.

    Table rows have no sentence punctuation, so newlines also count as split points.
    Chunks over SEMANTIC_MAX_CHARS are cut further so they fit bge's 512-token window.
    """
    chunker = SemanticChunker(
        embeddings,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=config.SEMANTIC_BREAKPOINT_PERCENTILE,
        sentence_split_regex=rf"(?<=[.?!])\s+|(?<={_NL})",
        min_chunk_size=MIN_CHUNK_CHARS,
    )
    capper = RecursiveCharacterTextSplitter(chunk_size=config.SEMANTIC_MAX_CHARS, chunk_overlap=0)
    out = []
    for c in chunker.split_text(text.replace("\n", _NL)):
        c = re.sub(rf"{_NL}\s?", "\n", c).strip()
        out.extend(capper.split_text(c) if len(c) > config.SEMANTIC_MAX_CHARS else [c])
    return out


def build_chunks(doc: dict, strategy: str, embeddings=None) -> list[dict]:
    """Chunk every Item of one cleaned filing; each chunk carries the filing's metadata."""
    meta = doc["meta"]
    tag = f"FY{meta['fiscal_year']}_{meta['form_type'].replace('/', '')}"
    chunks = []
    for sec in doc["sections"]:
        if strategy == "fixed":
            pieces = fixed_chunks(sec["text"])
        elif strategy == "semantic":
            pieces = semantic_chunks(sec["text"], embeddings)
        else:
            raise ValueError(strategy)
        for n, piece in enumerate(p for p in pieces if len(p) >= MIN_CHUNK_CHARS):
            chunks.append({
                "id": f"{strategy}-{tag}-{sec['item']}-{n}",
                "text": piece,
                "metadata": {
                    "company": meta["company"],
                    "doc_type": meta["doc_type"],
                    "form_type": meta["form_type"],
                    "fiscal_year": meta["fiscal_year"],
                    "source_file": meta["source_file"],
                    "section": sec["item"],
                    "section_title": sec["title"][:120],
                    "chunk_index": n,
                    "strategy": strategy,
                },
            })
    return chunks
