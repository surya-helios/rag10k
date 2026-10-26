"""Stage 1: extract text from each filing, clean it, split it into 10-K Items, attach metadata.

Usage:
    python prepare.py            # reuse cached raw extraction if present
    python prepare.py --force    # re-extract every file (slow: PDFs take minutes)

Outputs:
    data/raw/<stem>.txt        cached extraction (pages separated by form feed)
    data/clean/FY<year>.json   {"meta": {...}, "sections": [{"item", "title", "text"}, ...]}
"""
import argparse
import json
import re
from collections import Counter

import config

PAGE_SEP = "\f"
ITEM_ORDER = ["1", "1A", "1B", "1C", "2", "3", "4", "5", "6", "7", "7A", "8",
              "9", "9A", "9B", "9C", "10", "11", "12", "13", "14", "15", "16"]
ITEM_RANK = {k: i for i, k in enumerate(ITEM_ORDER)}
# Exhibit lists, signatures and the "summary" page hold no answerable content.
DROP_ITEMS = {"15", "16"}

ITEM_RE = re.compile(r"^\s*item\s+(\d{1,2}[A-C]?)\s*[\.\:\-–—]?\s*(.*)$", re.I)
FYE_RE = re.compile(r"for\s+the\s+fiscal\s+year\s+ended\s+([A-Za-z]+\s+\d{1,2},?\s*\d{4})", re.I)
BOILERPLATE_RES = [
    re.compile(r"^table of contents$", re.I),
    re.compile(r"^\d{1,3}$"),                       # bare page numbers
    re.compile(r"^[A-Z]-\d{1,3}$"),                 # F-12 style page numbers
    re.compile(r"^page \d+( of \d+)?$", re.I),
    re.compile(r"^[\-\|\s:=_\.]+$"),                # separator / empty table rows
]


# ---------------------------------------------------------------- extraction
def extract_pdf(path) -> list[str]:
    import pymupdf4llm
    pages = pymupdf4llm.to_markdown(str(path), page_chunks=True, show_progress=False)
    return [p["text"] for p in pages]


def extract_html(path) -> list[str]:
    """Inline-XBRL HTML: drop the hidden XBRL header, keep table rows on one line."""
    from bs4 import BeautifulSoup, NavigableString
    raw = path.read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(raw, "lxml")
    for tag in soup.find_all(["ix:header", "script", "style"]):
        tag.decompose()
    for tr in soup.find_all("tr"):
        cells = [c.get_text(" ", strip=True) for c in tr.find_all(["td", "th"])]
        cells = [c for c in cells if c and c not in {"$", ")", "%"}]
        tr.replace_with(NavigableString(" | ".join(cells) + "\n"))
    for br in soup.find_all("br"):
        br.replace_with(NavigableString("\n"))
    for blk in soup.find_all(["p", "div", "li", "h1", "h2", "h3", "h4", "h5", "h6"]):
        blk.append(NavigableString("\n"))
    return [soup.get_text("")]


def load_raw(path, force: bool) -> list[str]:
    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    cache = config.RAW_DIR / (path.stem + ".txt")
    if cache.exists() and not force:
        return cache.read_text(encoding="utf-8").split(PAGE_SEP)
    pages = extract_html(path) if path.suffix.lower() in {".html", ".htm"} else extract_pdf(path)
    cache.write_text(PAGE_SEP.join(pages), encoding="utf-8")
    return pages


# ------------------------------------------------------------------ cleaning
def _clean_line(line: str) -> str:
    line = line.replace(" ", " ").replace("<br>", " ")
    line = re.sub(r"\*\*|__", "", line)
    line = re.sub(r"(?<!\w)_(.+?)_(?!\w)", r"\1", line)          # markdown italics
    if line.lstrip().startswith("|"):                              # markdown table row -> "a | b | c"
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        line = " | ".join(c for c in cells if c and not re.fullmatch(r":?-{2,}:?", c))
    return re.sub(r"[ \t]+", " ", line).strip()


def find_repeated_lines(pages: list[str]) -> set[str]:
    """Running headers/footers: short lines that recur at the top/bottom of many pages."""
    if len(pages) < 8:
        return set()
    counts = Counter()
    for p in pages:
        lines = [_clean_line(l) for l in p.splitlines()]
        lines = [l for l in lines if l]
        for l in set(lines[:3] + lines[-3:]):
            if len(l) <= 100:
                counts[l] += 1
    return {l for l, c in counts.items() if c >= max(5, 0.25 * len(pages)) and not l[:1].isdigit()}


def clean_pages(pages: list[str]) -> str:
    repeated = find_repeated_lines(pages)
    out = []
    for p in pages:
        for raw in p.splitlines():
            line = _clean_line(raw)
            if not line or line in repeated or any(r.match(line) for r in BOILERPLATE_RES):
                continue
            out.append(line)
    text = "\n\n".join(out)  # one paragraph (or table row) per block
    return re.sub(r"\n{3,}", "\n\n", text).strip()


# ------------------------------------------------------- Item segmentation
def split_items(text: str) -> list[dict]:
    """Cut the text at the real Item headings.

    The table of contents and in-text cross-references also look like headings, so among
    all candidates we keep the in-order sequence (1, 1A, 1B, ... 16) that covers the most text.
    The TOC forms a short in-order run, the body a long one, so the body wins.
    """
    cands = []  # (position, item, title)
    pos = 0
    for line in text.split("\n"):
        m = ITEM_RE.match(line)
        if m and m.group(1).upper() in ITEM_RANK and len(line) < 250:
            cands.append((pos, m.group(1).upper(), m.group(2).strip(" .")))
        pos += len(line) + 1
    if not cands:
        return []
    ends = [c[0] for c in cands[1:]] + [len(text)]
    spans = [e - c[0] for c, e in zip(cands, ends)]
    best = [spans[i] for i in range(len(cands))]
    prev = [-1] * len(cands)
    for i in range(len(cands)):
        for j in range(i):
            if ITEM_RANK[cands[j][1]] < ITEM_RANK[cands[i][1]] and best[j] + spans[i] > best[i]:
                best[i], prev[i] = best[j] + spans[i], j
    i = max(range(len(cands)), key=lambda k: best[k])
    chain = []
    while i != -1:
        chain.append(i)
        i = prev[i]
    chain.reverse()
    sections = []
    for n, ci in enumerate(chain):
        start = cands[ci][0]
        end = cands[chain[n + 1]][0] if n + 1 < len(chain) else len(text)
        body = text[start:end].split("\n", 1)
        sections.append({"item": cands[ci][1], "title": cands[ci][2], "text": (body[1] if len(body) > 1 else "").strip()})
    return sections


# ------------------------------------------------------------------ metadata
def detect_metadata(first_page: str, path) -> dict:
    head = " ".join(first_page.replace("**", "").split())[:6000]
    m = FYE_RE.search(head)
    period_end = m.group(1) if m else None
    year = None
    if period_end:
        year = int(re.search(r"\d{4}", period_end).group())
    else:
        fm = re.search(r"(20\d{2})1231", path.name)
        year = int(fm.group(1)) if fm else None
    return {
        "company": config.COMPANY,
        "doc_type": config.DOC_TYPE,
        "form_type": "10-K/A" if re.search(r"form\s+10-K/A", head, re.I) else "10-K",
        "fiscal_year": year,
        "period_end": period_end,
        "source_file": path.name,
    }


def process(path, force: bool) -> dict:
    pages = load_raw(path, force)
    meta = detect_metadata(pages[0], path)
    text = clean_pages(pages)
    sections = split_items(text)
    if len(sections) < 3:  # amendment or unusual layout: keep everything as one block
        print(f"  ! {path.name}: only {len(sections)} Item headings found, keeping as a single section")
        sections = [{"item": "ALL", "title": "Full document", "text": text}]
    sections = [s for s in sections if s["item"] not in DROP_ITEMS and len(s["text"]) > 0]
    meta.update(pages=len(pages), clean_chars=sum(len(s["text"]) for s in sections))
    return {"meta": meta, "sections": sections}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="re-extract raw text")
    args = ap.parse_args()

    files = sorted(p for p in config.FILINGS_DIR.iterdir() if p.suffix.lower() in {".pdf", ".html", ".htm"})
    if not files:
        raise SystemExit(f"No filings found in {config.FILINGS_DIR}")
    config.CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    for old in config.CLEAN_DIR.glob("*.json"):
        old.unlink()  # avoid stale outputs from earlier runs

    for path in files:
        print(f"{path.name}")
        doc = process(path, args.force)
        m = doc["meta"]
        out = config.CLEAN_DIR / f"FY{m['fiscal_year']}_{m['form_type'].replace('/', '')}.json"
        out.write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
        sizes = ", ".join(f"{s['item']}:{len(s['text']) // 1000}k" for s in doc["sections"])
        print(f"  FY{m['fiscal_year']} {m['form_type']} | {m['pages']} pages | {m['clean_chars']:,} chars\n  sections -> {sizes}")


if __name__ == "__main__":
    main()
