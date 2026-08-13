"""Download arXiv PDFs and extract their full text.

Abstracts give roughly two chunks per paper. That is too few for a retrieval benchmark:
returning the top 5 of ~140 chunks is trivially easy, and BM25, dense and hybrid all
score the same — not because they are equivalent, but because the task is not
discriminating. Full text gives ~40 chunks per paper, which makes retrieval a real
problem and makes questions answerable from a *passage* rather than from recognising a
whole abstract.

Downloads are cached on disk and the run is resumable, because fetching several hundred
PDFs at arXiv's requested rate takes a while and should never have to start over.

    python -m src.ingest.pdf                 # all papers in papers.jsonl
    python -m src.ingest.pdf --limit 10      # smoke test first
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path

from src.config import load_config
from src.ingest.clean import clean_pdf_text

HEADERS = {"User-Agent": "rag-degradation/0.1 (research; contact via GitHub)"}


@dataclass
class ExtractedPaper:
    doc_id: str
    title: str
    text: str
    n_pages: int
    original_chars: int
    cleaned_chars: int
    retained_ratio: float
    references_stripped: bool
    source: str            # "pdf" or "abstract"


def pdf_path(cache_dir: Path, doc_id: str) -> Path:
    """One file per paper. `/` appears in old-style ids (cs/0701001)."""
    return cache_dir / f"{doc_id.replace('/', '_')}.pdf"


def download_pdf(url: str, dest: Path, timeout: int = 60) -> bool:
    """Fetch one PDF unless it is already cached. Returns True if newly downloaded."""
    if dest.exists() and dest.stat().st_size > 1024:
        return False

    dest.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        data = response.read()

    # A truncated or error-page download is worse than none: it would extract as a few
    # lines of HTML and enter the corpus as a legitimate-looking document.
    if len(data) < 1024 or not data[:5].startswith(b"%PDF"):
        raise ValueError(f"not a PDF ({len(data)} bytes)")

    dest.write_bytes(data)
    return True


def require_pypdf() -> None:
    """Fail fast if the extractor is unavailable.

    A missing module is not a per-paper problem — it affects every paper — so it must
    not be caught by the per-paper handler below. Doing so once turned a one-line fix
    into 77 identical warnings and a corpus that had silently fallen back to abstracts
    while reporting success.
    """
    try:
        import pypdf  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "pypdf is not installed, so no PDF can be read.\n"
            "    pip install -r requirements.txt\n"
            f"  ({exc})"
        ) from exc


def extract_text(path: Path) -> tuple[str, int]:
    """Pull raw text out of a PDF. Returns (text, page_count).

    pypdf rather than pymupdf: pymupdf is AGPL, which would force the whole repository
    to the same licence. Extraction quality is slightly lower, which is exactly what
    `clean.py` exists to repair.
    """
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    pages = []
    for page in reader.pages:
        try:
            pages.append(page.extract_text() or "")
        except Exception:                              # noqa: BLE001
            # One unparseable page should not lose the other thirty.
            pages.append("")
    return "\n".join(pages), len(reader.pages)


def process_paper(paper: dict, cache_dir: Path, delay: float,
                  strip_references: bool = True) -> tuple[ExtractedPaper, bool]:
    """Download, extract and clean one paper. Returns (result, was_downloaded).

    Falls back to the abstract if anything goes wrong, so a failed PDF costs detail
    rather than losing the document entirely — and `source` records which happened.
    """
    doc_id = paper["doc_id"]
    dest = pdf_path(cache_dir, doc_id)
    downloaded = False

    try:
        url = paper.get("pdf_url") or f"https://arxiv.org/pdf/{doc_id}"
        downloaded = download_pdf(url, dest)
        if downloaded and delay:
            time.sleep(delay)

        raw, n_pages = extract_text(dest)
        if len(raw.strip()) < 500:
            raise ValueError(f"extracted only {len(raw.strip())} chars")

        text, report = clean_pdf_text(raw, n_pages, strip_references)
        if len(text) < 500:
            raise ValueError(f"cleaning left only {len(text)} chars")

        return ExtractedPaper(
            doc_id=doc_id, title=paper["title"], text=text, n_pages=n_pages,
            original_chars=report["original_chars"],
            cleaned_chars=report["cleaned_chars"],
            retained_ratio=report["retained_ratio"],
            references_stripped=report["references_stripped"],
            source="pdf",
        ), downloaded

    except (ImportError, SystemExit):
        # Environment problems are not this paper's fault and will not fix themselves
        # on the next iteration. Let them terminate the run.
        raise
    except Exception as exc:                           # noqa: BLE001
        fallback = f"{paper['title']}\n\n{paper['abstract']}"
        print(f"      {doc_id}: {type(exc).__name__}: {str(exc)[:70]} — using abstract")
        return ExtractedPaper(
            doc_id=doc_id, title=paper["title"], text=fallback, n_pages=0,
            original_chars=len(fallback), cleaned_chars=len(fallback),
            retained_ratio=1.0, references_stripped=False, source="abstract",
        ), downloaded


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download and extract paper full text.")
    parser.add_argument("--limit", type=int, default=None,
                        help="process only the first N papers — use to smoke-test")
    parser.add_argument("--keep-references", action="store_true",
                        help="do not strip the bibliography (default: strip it)")
    parser.add_argument("--config", default=None)
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    raw_dir = cfg.path_for("paths.raw_dir")
    papers_path = raw_dir / "papers.jsonl"
    if not papers_path.exists():
        print(f"No papers at {papers_path}. Run `python -m src.ingest.arxiv` first.")
        return 1

    require_pypdf()

    papers = [json.loads(l) for l in papers_path.open(encoding="utf-8") if l.strip()]
    if args.limit:
        papers = papers[: args.limit]

    cache_dir = raw_dir / "pdfs"
    delay = float(cfg.get("corpus.request_delay_seconds", 3.0))

    print("=" * 74)
    print("DOWNLOADING AND EXTRACTING FULL TEXT")
    print("=" * 74)
    print(f"  papers      : {len(papers)}")
    print(f"  cache       : {cache_dir}")
    print(f"  delay       : {delay}s between new downloads")
    print()

    results: list[ExtractedPaper] = []
    n_downloaded = 0
    for i, paper in enumerate(papers, 1):
        result, downloaded = process_paper(paper, cache_dir, delay,
                                           strip_references=not args.keep_references)
        results.append(result)
        n_downloaded += downloaded
        if i % 10 == 0 or i == len(papers):
            ok = sum(r.source == "pdf" for r in results)
            print(f"  [{i}/{len(papers)}] {ok} from PDF, "
                  f"{len(results) - ok} fell back to abstract")

    out_path = raw_dir / "fulltext.jsonl"
    with out_path.open("w", encoding="utf-8") as fh:
        for r in sorted(results, key=lambda x: x.doc_id):
            fh.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")

    from_pdf = [r for r in results if r.source == "pdf"]
    lengths = sorted(r.cleaned_chars for r in from_pdf) or [0]
    ratios = sorted(r.retained_ratio for r in from_pdf) or [0]
    stripped = sum(r.references_stripped for r in from_pdf)

    print()
    print("=" * 74)
    print("RESULT")
    print("=" * 74)
    print(f"  extracted from PDF : {len(from_pdf)}/{len(results)}")
    print(f"  newly downloaded   : {n_downloaded} (rest were cached)")
    if from_pdf:
        print(f"  chars per paper    : min {lengths[0]:,}, "
              f"median {lengths[len(lengths)//2]:,}, max {lengths[-1]:,}")
        print(f"  retained after     : median {ratios[len(ratios)//2]:.0%} of raw text")
        print(f"  references stripped: {stripped}/{len(from_pdf)}")
        if stripped < len(from_pdf) * 0.7:
            print("    ^ low. Those papers still carry their bibliography, which is")
            print("      strong bait for lexical retrieval. Worth inspecting.")
    print(f"\n  written            : {out_path}")

    if not from_pdf:
        print("\n  *** No paper was extracted from its PDF. The corpus would be")
        print("      abstracts only, which is what this stage exists to avoid.")
        print("      Fix the errors above rather than continuing.")
        return 1

    print("\nNext: python -m src.ingest.build_corpus --source fulltext")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
