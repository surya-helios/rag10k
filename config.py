"""Central settings. Values come from .env (see .env.example); nothing secret lives in code."""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

# --- paths ---
FILINGS_DIR = Path(os.getenv("FILINGS_DIR", ROOT / "filings"))  # put the 10-K files here
RAW_DIR = ROOT / "data" / "raw"      # cached text extracted from PDF/HTML
CLEAN_DIR = ROOT / "data" / "clean"  # cleaned text split into 10-K Items, one JSON per filing
RESULTS_DIR = ROOT / "results"

# --- constants describing the corpus ---
COMPANY = "Tesla"
DOC_TYPE = "10-K"

# --- chunking (bge-base reads at most 512 tokens ~ 2000 chars; keep a margin for number-heavy tables) ---
FIXED_CHUNK_SIZE = int(os.getenv("FIXED_CHUNK_SIZE", "1000"))
FIXED_CHUNK_OVERLAP = int(os.getenv("FIXED_CHUNK_OVERLAP", "200"))
SEMANTIC_MAX_CHARS = int(os.getenv("SEMANTIC_MAX_CHARS", "1500"))
SEMANTIC_BREAKPOINT_PERCENTILE = float(os.getenv("SEMANTIC_BREAKPOINT_PERCENTILE", "90"))

# --- models ---
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-base-en-v1.5")
EMBED_DIM = 768  # fixed by the model; Pinecone index must match
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "  # bge: questions only
RERANK_MODEL = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-base")
# LLM provider for generation and judging: "google" (Gemini) or "nebius" (Nebius Token Factory,
# OpenAI-compatible). Check model names with `python list_models.py` and override in .env.
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "google").strip().lower()
NEBIUS_BASE_URL = os.getenv("NEBIUS_BASE_URL", "https://api.tokenfactory.nebius.com/v1/")
# gemini-2.5-* still appear in list_models.py but return 404 for new API keys (checked 2026-10-24).
_DEFAULT_MODELS = {
    "google": ("gemini-3.5-flash", "gemini-3.5-flash-lite"),
    "nebius": ("", ""),  # set GEN_MODEL / JUDGE_MODEL in .env after running list_models.py
}
if LLM_PROVIDER not in _DEFAULT_MODELS:
    raise SystemExit(f"LLM_PROVIDER must be one of {sorted(_DEFAULT_MODELS)}, got '{LLM_PROVIDER}'.")
GEN_MODEL = os.getenv("GEN_MODEL", _DEFAULT_MODELS[LLM_PROVIDER][0])
JUDGE_MODEL = os.getenv("JUDGE_MODEL", _DEFAULT_MODELS[LLM_PROVIDER][1])

# --- retrieval ---
TOP_K = int(os.getenv("TOP_K", "5"))          # chunks sent to the LLM
FETCH_K = int(os.getenv("FETCH_K", "25"))     # candidates fetched before re-ranking

# --- Pinecone ---
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "tenk-rag")
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "aws")
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")
STRATEGIES = ["fixed", "semantic"]  # also the Pinecone namespaces

# --- evaluation / Gemini free tier ---
SECONDS_BETWEEN_CALLS = float(os.getenv("SECONDS_BETWEEN_CALLS", "6"))


def require(name: str) -> str:
    """Return a required secret from the environment or stop with a clear message."""
    val = os.getenv(name, "").strip()
    if not val:
        raise SystemExit(f"Missing {name}. Add it to {ROOT / '.env'} (see .env.example).")
    return val
