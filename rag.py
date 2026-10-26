"""Answer generation with Gemini, grounded in the retrieved chunks."""
import re
import time

from langchain_google_genai import ChatGoogleGenerativeAI
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

import config

ANSWER_PROMPT = """You are a financial analyst assistant answering questions about Tesla's SEC Form 10-K filings.
Use ONLY the excerpts below. Each excerpt is labelled with its fiscal year and 10-K section.
- Give exact figures with units and the fiscal year they refer to.
- If the excerpts do not contain the answer, reply exactly: "The provided filings do not contain this information."
- Do not use outside knowledge and do not guess.

Excerpts:
{context}

Question: {question}
Answer:"""


def format_context(hits: list[dict]) -> str:
    parts = []
    for h in hits:
        m = h["metadata"]
        parts.append(f"[FY{int(m['fiscal_year'])} {m['form_type']} | Item {m['section']}]\n{h['text']}")
    return "\n\n---\n\n".join(parts)


def is_daily_quota(exc: Exception) -> bool:
    """A per-day quota will not clear by retrying for a few minutes."""
    s = str(exc).lower()
    return "perday" in s or "per day" in s


def _is_transient(exc: Exception) -> bool:
    s = str(exc).lower()
    if is_daily_quota(exc):
        return False
    return any(t in s for t in ("429", "503", "500", "quota", "rate", "resource_exhausted", "unavailable", "timeout"))


@retry(retry=retry_if_exception(_is_transient), wait=wait_exponential(multiplier=5, max=90), stop=stop_after_attempt(6),
       reraise=True)
def call_llm(llm, prompt: str) -> str:
    time.sleep(config.SECONDS_BETWEEN_CALLS)  # stay under free-tier requests-per-minute
    content = llm.invoke(prompt).content
    if isinstance(content, list):  # some Gemini responses come back as parts
        content = "".join(p if isinstance(p, str) else p.get("text", "") for p in content)
    return content.strip()


def make_llm(model: str):
    if not model:
        raise SystemExit(f"No model set for LLM_PROVIDER={config.LLM_PROVIDER}. Run list_models.py and set "
                         "GEN_MODEL and JUDGE_MODEL in .env.")
    if config.LLM_PROVIDER == "nebius":
        from langchain_openai import ChatOpenAI  # OpenAI-compatible client pointed at Nebius, not OpenAI
        return ChatOpenAI(model=model, base_url=config.NEBIUS_BASE_URL, api_key=config.require("NEBIUS_API_KEY"),
                          temperature=0, max_retries=0, timeout=120)
    config.require("GOOGLE_API_KEY")
    return ChatGoogleGenerativeAI(model=model, temperature=0, max_retries=0)


def answer(llm, question: str, hits: list[dict]) -> str:
    return call_llm(llm, ANSWER_PROMPT.format(context=format_context(hits), question=question))


def parse_scores(text: str) -> dict:
    """Pull 'correctness' and 'faithfulness' (1-5) out of the judge's reply."""
    out = {}
    for key in ("correctness", "faithfulness"):
        m = re.search(rf'"?{key}"?\s*[:=]\s*([1-5])', text, re.I)
        out[key] = int(m.group(1)) if m else None
    return out
