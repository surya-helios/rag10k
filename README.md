# Tesla 10-K RAG: fixed vs semantic chunking, with and without re-ranking

Answers questions about Tesla's 10-K filings (FY2016-FY2025) and compares chunking strategies on the
quality of the **generated answers**.

## Setup on a new machine (needs internet access to huggingface.co, Pinecone and Google AI)
1. Install **Python 3.10 or newer**.
2. Open a terminal in this folder and create the environment:
   - Windows: `python -m venv .venv` then `.venv\Scripts\activate`
   - Mac/Linux: `python3 -m venv .venv` then `source .venv/bin/activate`
3. `pip install -r requirements.txt`
4. Optional, NVIDIA GPU only (makes semantic chunking and re-ranking much faster): install the CUDA build of
   PyTorch from https://pytorch.org/get-started/locally/ before step 3, or re-install it afterwards.
6. Run `python list_models.py` to see which Gemini models your key can use; put two different ones in `.env`
   as `GEN_MODEL` and `JUDGE_MODEL` if the defaults in `config.py` are not available.
7. The `bge` models download automatically from Hugging Face the first time they are used (~1.5 GB total).

The 10 filings are already in `filings/`. `data/clean/` and `data/chunks/fixed.jsonl` are already built,
so you can start at step 2 below (or re-run step 1 to rebuild them).

## Run order
1. `python prepare.py`      extract + clean + split into 10-K Items -> `data/clean/*.json` (`--force` re-extracts)
2. `python make_chunks.py`  chunk with both strategies -> `data/chunks/*.jsonl` (semantic is slow without a GPU)
3. `python index.py`        embed + upsert to Pinecone, namespaces `fixed` and `semantic` (`--reset` wipes first)
4. `python ask.py "What was Tesla's total revenue in 2023?" --strategy semantic --rerank --years 2023`
5. Fill `questions.json` (20 questions), then `python evaluate.py`
   -> `results/detail.csv`, `results/summary.csv`, head-to-head wins

Status (2026-10-24): steps 1-3 are done and verified. `data/chunks/semantic.jsonl` holds 7,936 chunks (4 are
8-21 tokens over 512 and get truncated slightly when embedded), and Pinecone holds 5,225 vectors in `fixed` and
7,936 in `semantic`. `questions.json` has 20 questions. First full evaluation done with
`LLM_PROVIDER=nebius` (answers: Qwen3-235B-A22B-Instruct-2507, judge: DeepSeek-V4-Pro-0813); with 45
filing-worded questions and TOP_K chunks per fiscal year (`retrieve.py`), correctness A 4.67, B 4.38, C 4.36, D 4.36.
22 casually worded questions added (`"wording": "casual"`): correctness A 3.41, B 2.86, C 3.05, D 3.18
(`results/summary_by_wording.csv`); casual wording mainly hurts retrieval (context recall 0.58-0.66 vs 0.80-0.86);
per-year retrieval raised multi-year correctness by 0.5-1.0 for every setup. Remaining multi-year losses:
the answer prompt makes the model refuse the whole question when one year is missing, and the judge marks
correct computed differences as unfaithful. Known issue: `prepare.py` keeps the PDF's hard line wraps (76% of lines end
mid-sentence), and semantic chunking splits on them, so semantic chunks often end mid-sentence. Fix this and
re-run steps 2-5 before drawing conclusions about semantic chunking.

## Choices made
| Piece | Choice |
|---|---|
| Embeddings (also drives semantic chunking) | `BAAI/bge-base-en-v1.5` - 768 dims, 512-token max input, cosine, free, local |
| Re-ranker | `BAAI/bge-reranker-base` - free, local |
| Vector DB | Pinecone serverless, index `tenk-rag`, namespaces `fixed` and `semantic` |
| Generator / judge | Gemini, two different models (`GEN_MODEL`, `JUDGE_MODEL`) |
| Fixed chunking | 1000 chars, 200 overlap (`RecursiveCharacterTextSplitter`) |
| Semantic chunking | LangChain `SemanticChunker`, 90th-percentile breakpoint, chunks capped at 1500 chars |
| Metadata on every chunk | company, doc_type=`10-K`, form_type, fiscal_year, section (Item), source_file |

Corpus notes: FY2021 comes from the HTML file; FY2025 is a **10-K/A** (Part III only: Items 10-14, no
financials). Items 15-16 (exhibit lists, signatures) are dropped. All PDFs are text-based, no OCR.
FY2016-FY2018 report dollar figures in **thousands**; FY2019 onward in **millions**.

## Chunk file format (`data/chunks/<strategy>.jsonl`)
JSON Lines: one JSON object per line, one line per chunk.
`{"id": "fixed-FY2023_10-K-8-16", "text": "...", "metadata": {"fiscal_year": 2023, "section": "8", ...}}`
In Pinecone each chunk is stored as: id + 768-number vector + metadata (the same metadata plus the text).

## Evaluation design
Four setups: A fixed, B fixed+rerank, C semantic, D semantic+rerank. Only chunking and re-ranking vary.
Metrics: `fact_recall` (key figures present in the answer), `context_recall` (key figures present in the
retrieved text), `correctness` and `faithfulness` (1-5, Gemini judge). `evaluate.py` caches every answer in
`results/cache.jsonl` and resumes after rate-limit stops.

`questions.json` fields: `question`, `reference`, `key_facts` (strings, or lists of acceptable alternatives),
optional `fiscal_years` (restrict retrieval to those filings), optional `type`, optional `wording`
(`filing` (default) or `casual`, reported separately), optional `source: "found_in_testing"` (questions first
seen while trying the app; reported with and without them).
