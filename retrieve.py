"""Retrieval from Pinecone, with optional bge re-ranking.

Without re-ranking: top TOP_K chunks by vector similarity.
With re-ranking:    FETCH_K candidates by similarity, re-scored by a cross-encoder, best TOP_K kept.
If fiscal years are given, each year is searched and ranked separately and keeps its own TOP_K, so a
multi-year question gets TOP_K chunks from every year instead of one year crowding out the others.
"""
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

import config
from chunking import get_embeddings

_reranker = None


def get_reranker() -> HuggingFaceCrossEncoder:
    global _reranker
    if _reranker is None:
        import torch
        _reranker = HuggingFaceCrossEncoder(
            model_name=config.RERANK_MODEL,
            model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu", "max_length": 512},
        )
    return _reranker


class Retriever:
    def __init__(self, strategy: str, rerank: bool, embeddings=None):
        self.strategy, self.rerank = strategy, rerank
        self.embeddings = embeddings or get_embeddings()
        index = Pinecone(api_key=config.require("PINECONE_API_KEY")).Index(config.PINECONE_INDEX)
        self.store = PineconeVectorStore(index=index, embedding=self.embeddings, namespace=strategy)

    def retrieve(self, question: str, fiscal_years: list[int] | None = None) -> list[dict]:
        vec = self.embeddings.embed_query(config.QUERY_PREFIX + question)
        n = config.FETCH_K if self.rerank else config.TOP_K
        filters = [{"fiscal_year": y} for y in fiscal_years] if fiscal_years else [None]
        out = []
        for flt in filters:
            hits = [{"text": doc.page_content, "metadata": doc.metadata, "score": float(score)}
                    for doc, score in self.store.similarity_search_by_vector_with_score(vec, k=n, filter=flt)]
            out.extend(self._rank(question, hits)[:config.TOP_K])
        return out

    def _rank(self, question: str, hits: list[dict]) -> list[dict]:
        if self.rerank and hits:
            scores = get_reranker().score([(question, h["text"]) for h in hits])
            for h, s in zip(hits, scores):
                h["rerank_score"] = float(s)
            return sorted(hits, key=lambda h: h["rerank_score"], reverse=True)
        return sorted(hits, key=lambda h: h["score"], reverse=True)
