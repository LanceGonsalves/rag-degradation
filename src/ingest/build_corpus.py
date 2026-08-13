"""Turn fetched papers into a chunked corpus.

Separated from fetching on purpose: chunk size and overlap are experimental variables,
so this stage gets re-run many times while `papers.jsonl` is fetched once. Coupling
them would mean hammering the arXiv API every time a parameter changed.

    python -m src.ingest.build_corpus
    python -m src.ingest.build_corpus --chunk-size 500 --chunk-overlap 100
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from src.config import load_config
from src.ingest.chunk import chunk_corpus


def load_jsonl(path: Path, hint: str) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"No file at {path}. Run `{hint}` first.")
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def to_documents(papers: list[dict], fulltext: list[dict] | None = None) -> list[dict]:
    """Shape papers into what the chunker expects, carrying metadata forward.

    When full text is available it replaces the abstract, but the *metadata* still
    comes from `papers.jsonl` — the extractor only carries title and text, and losing
    the category and URL would make retrieved chunks unattributable.

    `source` is recorded per document so a corpus that silently fell back to abstracts
    for half its papers is visible in the statistics rather than discovered later.
    """
    text_by_id = {f["doc_id"]: f for f in (fulltext or [])}

    docs = []
    for p in papers:
        extracted = text_by_id.get(p["doc_id"])
        if extracted:
            text = extracted["text"]
            source = extracted.get("source", "pdf")
        else:
            text = f"{p['title']}\n\n{p['abstract']}"
            source = "abstract"

        docs.append({
            "doc_id": p["doc_id"],
            "text": text,
            "title": p["title"],
            "primary_category": p.get("primary_category", ""),
            "published": p.get("published", ""),
            "abs_url": p.get("abs_url", ""),
            "source": source,
        })
    return docs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Chunk the fetched corpus.")
    parser.add_argument("--chunk-size", type=int, default=None)
    parser.add_argument("--chunk-overlap", type=int, default=None)
    parser.add_argument("--strategy", default=None, choices=["fixed", "sentence_window"])
    parser.add_argument("--source", default="abstract", choices=["abstract", "fulltext"],
                        help="abstract: ~2 chunks/paper. fulltext: ~40, and a corpus "
                             "large enough for retrieval strategy to matter.")
    parser.add_argument("--out", default=None, help="output filename inside corpus_dir")
    parser.add_argument("--config", default=None)
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    chunking = cfg._data.setdefault("chunking", {})
    if args.chunk_size is not None:
        chunking["chunk_size"] = args.chunk_size
    if args.chunk_overlap is not None:
        chunking["chunk_overlap"] = args.chunk_overlap
    if args.strategy is not None:
        chunking["strategy"] = args.strategy

    raw_dir = cfg.path_for("paths.raw_dir")
    papers = load_jsonl(raw_dir / "papers.jsonl", "python -m src.ingest.arxiv")

    fulltext = None
    if args.source == "fulltext":
        fulltext = load_jsonl(raw_dir / "fulltext.jsonl", "python -m src.ingest.pdf")

    documents = to_documents(papers, fulltext)
    chunks = list(chunk_corpus(documents, cfg))

    if not chunks:
        print("No chunks produced — check min_chunk_chars against your abstracts.")
        return 1

    out_dir = cfg.path_for("paths.corpus_dir")
    out_dir.mkdir(parents=True, exist_ok=True)
    name = args.out or (
        f"chunks_{args.source}_{chunking.get('strategy','sentence_window')}"
        f"_{chunking.get('chunk_size',1000)}_{chunking.get('chunk_overlap',200)}.jsonl"
    )
    out_path = out_dir / name
    with out_path.open("w", encoding="utf-8") as fh:
        for c in chunks:
            fh.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")

    sizes = sorted(c.n_chars for c in chunks)
    per_doc: dict[str, int] = {}
    for c in chunks:
        per_doc[c.doc_id] = per_doc.get(c.doc_id, 0) + 1
    counts = sorted(per_doc.values())

    print("=" * 74)
    print("CORPUS BUILT")
    print("=" * 74)
    print(f"  strategy         : {chunking.get('strategy', 'sentence_window')}")
    print(f"  chunk_size       : {chunking.get('chunk_size', 1000)}  "
          f"overlap: {chunking.get('chunk_overlap', 200)}")
    print()
    n_pdf = sum(d["source"] == "pdf" for d in documents)
    print(f"  documents        : {len(documents):,} "
          f"({n_pdf} full text, {len(documents) - n_pdf} abstract only)")
    print(f"  chunks           : {len(chunks):,}")
    print(f"  chunks per doc   : min {counts[0]}, median {counts[len(counts)//2]}, "
          f"max {counts[-1]}")
    print(f"  chunk chars      : min {sizes[0]}, median {sizes[len(sizes)//2]}, "
          f"max {sizes[-1]}")
    print(f"  unique chunk ids : {len({c.chunk_id for c in chunks}):,} "
          f"({'no collisions' if len({c.chunk_id for c in chunks}) == len(chunks) else '*** COLLISIONS ***'})")
    print()
    top_k = int(cfg.get("retrieval.top_k", 5))
    share = top_k / len(chunks) * 100
    print(f"  top_{top_k} returns     : {share:.2f}% of the corpus", end="")
    print("  (want <1% for retrieval strategy to matter)"
          if share >= 1 else "  ✓")
    print()
    print(f"  written          : {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
