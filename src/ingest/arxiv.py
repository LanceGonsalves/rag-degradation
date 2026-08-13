"""Fetch paper metadata and abstracts from the arXiv API.

The corpus is *not* redistributed — this fetches it, exactly as the HAM10000 project
does. Anyone cloning the repo runs this and gets the same papers.

arXiv returns Atom XML. The parser below is defensive about missing fields because a
single malformed entry should not lose the other ninety-nine in the same response.

    python -m src.ingest.arxiv --max-per-query 50
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from pathlib import Path

API = "https://export.arxiv.org/api/query"
NS = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}

# arXiv asks for a descriptive User-Agent and at least 3 seconds between requests.
# Being impolite here gets the IP blocked, which would break the project for everyone
# who clones it, not just this machine.
HEADERS = {"User-Agent": "rag-degradation/0.1 (research; contact via GitHub)"}


@dataclass
class Paper:
    doc_id: str          # arXiv id without version, e.g. 2104.08663
    title: str
    abstract: str
    authors: list[str]
    published: str
    updated: str
    primary_category: str
    categories: list[str]
    pdf_url: str
    abs_url: str

    @property
    def text(self) -> str:
        """What gets chunked. Title included so a chunk carries its own context."""
        return f"{self.title}\n\n{self.abstract}"


def _clean(text: str | None) -> str:
    """arXiv wraps abstracts at ~80 chars; unwrap into flowing text."""
    if not text:
        return ""
    return " ".join(text.split())


def _strip_version(entry_id: str) -> str:
    """`http://arxiv.org/abs/2104.08663v2` -> `2104.08663`.

    Versions are stripped so re-fetching after a paper is revised does not create a
    second document that duplicates the first in every retrieval result.
    """
    tail = entry_id.rstrip("/").split("/")[-1]
    if "v" in tail:
        head, _, ver = tail.rpartition("v")
        if ver.isdigit() and head:
            return head
    return tail


def parse_entry(entry: ET.Element) -> Paper | None:
    """Parse one <entry>. Returns None if it lacks the fields we need."""
    raw_id = entry.findtext("atom:id", default="", namespaces=NS)
    title = _clean(entry.findtext("atom:title", default="", namespaces=NS))
    abstract = _clean(entry.findtext("atom:summary", default="", namespaces=NS))

    if not raw_id or not title or not abstract:
        return None

    authors = [
        _clean(a.findtext("atom:name", default="", namespaces=NS))
        for a in entry.findall("atom:author", NS)
    ]

    primary = entry.find("arxiv:primary_category", NS)
    primary_cat = primary.get("term", "") if primary is not None else ""
    categories = [c.get("term", "") for c in entry.findall("atom:category", NS)]

    pdf_url = abs_url = ""
    for link in entry.findall("atom:link", NS):
        if link.get("title") == "pdf":
            pdf_url = link.get("href", "")
        elif link.get("rel") == "alternate":
            abs_url = link.get("href", "")

    return Paper(
        doc_id=_strip_version(raw_id),
        title=title,
        abstract=abstract,
        authors=[a for a in authors if a],
        published=entry.findtext("atom:published", default="", namespaces=NS),
        updated=entry.findtext("atom:updated", default="", namespaces=NS),
        primary_category=primary_cat,
        categories=[c for c in categories if c],
        pdf_url=pdf_url,
        abs_url=abs_url or raw_id,
    )


def parse_response(xml_text: str) -> list[Paper]:
    """Parse a full Atom feed into papers, skipping entries that fail."""
    root = ET.fromstring(xml_text)
    papers = []
    for entry in root.findall("atom:entry", NS):
        paper = parse_entry(entry)
        if paper is not None:
            papers.append(paper)
    return papers


def fetch_query(query: str, max_results: int, timeout: int = 60) -> list[Paper]:
    """Run one search query against the arXiv API."""
    params = urllib.parse.urlencode({
        "search_query": query,
        "start": 0,
        "max_results": max_results,
        "sortBy": "relevance",
        "sortOrder": "descending",
    })
    url = f"{API}?{params}"
    request = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return parse_response(response.read().decode("utf-8"))


def fetch_corpus(cfg, verbose: bool = True) -> list[Paper]:
    """Run every configured query and de-duplicate across them.

    Queries overlap by design — a paper about benchmark contamination may match three
    of them — so de-duplication by arXiv id is not optional. Without it the corpus
    would contain repeated documents, and any retrieval metric computed on it would be
    quietly wrong.
    """
    queries = cfg.require("corpus.queries")
    per_query = int(cfg.get("corpus.max_per_query", 100))
    delay = float(cfg.get("corpus.request_delay_seconds", 3.0))
    min_abstract = int(cfg.get("corpus.min_abstract_chars", 0))

    seen: dict[str, Paper] = {}
    for i, query in enumerate(queries, 1):
        if verbose:
            print(f"  [{i}/{len(queries)}] {query}")
        try:
            papers = fetch_query(query, per_query)
        except Exception as exc:                      # noqa: BLE001
            print(f"      failed: {type(exc).__name__}: {exc}")
            continue

        new = short = 0
        for paper in papers:
            if len(paper.abstract) < min_abstract:
                short += 1
                continue
            if paper.doc_id not in seen:
                seen[paper.doc_id] = paper
                new += 1

        if verbose:
            note = f", {short} too short" if short else ""
            print(f"      {len(papers)} returned, {new} new{note}")

        if i < len(queries):
            time.sleep(delay)

    return list(seen.values())


def main(argv: list[str] | None = None) -> int:
    from src.config import load_config

    parser = argparse.ArgumentParser(description="Fetch the arXiv corpus.")
    parser.add_argument("--max-per-query", type=int, default=None,
                        help="override config; use a small value to smoke-test")
    parser.add_argument("--config", default=None)
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    if args.max_per_query is not None:
        cfg._data.setdefault("corpus", {})["max_per_query"] = args.max_per_query

    print("=" * 74)
    print("FETCHING CORPUS FROM ARXIV")
    print("=" * 74)

    papers = fetch_corpus(cfg)

    if not papers:
        print("\nNo papers fetched. Check network access and the queries in config.yaml.")
        return 1

    out_dir = cfg.path_for("paths.raw_dir")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "papers.jsonl"
    with out_path.open("w", encoding="utf-8") as fh:
        for paper in sorted(papers, key=lambda p: p.doc_id):
            fh.write(json.dumps(asdict(paper), ensure_ascii=False) + "\n")

    lengths = sorted(len(p.abstract) for p in papers)
    cats: dict[str, int] = {}
    for p in papers:
        cats[p.primary_category] = cats.get(p.primary_category, 0) + 1

    print()
    print("=" * 74)
    print("RESULT")
    print("=" * 74)
    print(f"  unique papers      : {len(papers):,}")
    print(f"  abstract chars     : min {lengths[0]}, median {lengths[len(lengths)//2]}, "
          f"max {lengths[-1]}")
    print(f"  primary categories : "
          + ", ".join(f"{k} ({v})" for k, v in sorted(cats.items(), key=lambda x: -x[1])[:6]))
    print(f"  written            : {out_path}")
    print()
    print("Next: python -m src.ingest.build_corpus")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
