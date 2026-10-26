"""Web interface for asking free-form questions about the Tesla 10-K filings.

Usage: streamlit run app.py
Uses the same retrieval, prompt and models as ask.py and evaluate.py (LLM_PROVIDER, GEN_MODEL in .env).
"""
import re

import streamlit as st

import config
from chunking import get_embeddings
from rag import answer, make_llm
from retrieve import Retriever

YEARS = list(range(2016, 2026))
SETUPS = {"A: fixed": ("fixed", False), "B: fixed + rerank": ("fixed", True),
          "C: semantic": ("semantic", False), "D: semantic + rerank": ("semantic", True)}
EXAMPLES = ["What were Tesla's total revenues in 2023?",
            "How did Tesla's worldwide employee count change between the end of 2020 and the end of 2024?",
            "What are the stages of Tesla's cybersecurity incident response process in 2023?",
            "What did the 2018 SEC settlement require of Tesla and Elon Musk?"]

if config.LLM_PROVIDER != "google":
    config.SECONDS_BETWEEN_CALLS = 0  # the pause only exists for Gemini's free-tier rate limit


@st.cache_resource(show_spinner="Loading the embedding model...")
def get_shared_embeddings():
    return get_embeddings()


@st.cache_resource(show_spinner="Connecting to Pinecone...")
def get_retriever(strategy: str, rerank: bool) -> Retriever:
    return Retriever(strategy, rerank, get_shared_embeddings())


@st.cache_resource
def get_llm():
    return make_llm(config.GEN_MODEL)


def years_in(question: str) -> list[int]:
    return sorted({int(y) for y in re.findall(r"\b(20(?:1[6-9]|2[0-5]))\b", question)})


def run(setup: str, question: str, years: list[int]) -> tuple[str, list[dict]]:
    strategy, rerank = SETUPS[setup]
    with st.spinner(f"{setup}: retrieving and answering..."):
        hits = get_retriever(strategy, rerank).retrieve(question, years or None)
        return answer(get_llm(), question, hits), hits


def show_sources(hits: list[dict]) -> None:
    st.caption(f"{len(hits)} retrieved chunks")
    for i, h in enumerate(hits, 1):
        m = h["metadata"]
        score = f"similarity {h['score']:.3f}"
        if "rerank_score" in h:
            score += f", rerank {h['rerank_score']:.2f}"
        title = str(m.get("section_title", ""))[:60]
        with st.expander(f"{i}. FY{int(m['fiscal_year'])} {m['form_type']}, Item {m['section']} {title} ({score})"):
            st.text(h["text"])


st.set_page_config(page_title="Tesla 10-K Q&A", layout="wide")
st.title("Tesla 10-K Q&A")
st.caption(f"Answers come only from Tesla's 10-K filings, FY2016-FY2025. Model: {config.GEN_MODEL} "
           f"({config.LLM_PROVIDER}).")

with st.sidebar:
    setup = st.radio("Setup", list(SETUPS), help="A scored best in the evaluation (results/summary.csv).")
    compare = st.checkbox("Compare all four setups")
    picked_years = st.multiselect("Fiscal years", YEARS,
                                  help="Leave empty to use the years mentioned in the question, "
                                       "or all filings if it mentions none.")
    st.markdown("**Example questions**")
    for ex in EXAMPLES:
        st.caption(ex)

question = st.text_input("Question", placeholder=EXAMPLES[0])
if st.button("Ask", type="primary") and question.strip():
    years = picked_years or years_in(question)
    st.caption("Searching " + (", ".join(f"FY{y}" for y in years) if years else "all filings")
               + (" (one top-5 per year)" if len(years) > 1 else ""))
    if compare:
        for col, name in zip(st.columns(len(SETUPS)), SETUPS):
            with col:
                st.subheader(name)
                text, hits = run(name, question, years)
                st.markdown(text)
                show_sources(hits)
    else:
        text, hits = run(setup, question, years)
        st.subheader("Answer")
        st.markdown(text)
        st.subheader("Sources")
        show_sources(hits)
