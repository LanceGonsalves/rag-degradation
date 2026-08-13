"""Deterministic text chunking.

Chunking is the quietest source of error in a RAG system. Get it wrong and nothing
crashes — retrieval simply gets worse, every downstream metric shifts, and there is no
stack trace pointing at the cause. Chunk size and overlap also become experimental
variables later, so the same input must always produce byte-identical output.

Two strategies:

- `fixed` — character windows with overlap. Simple, predictable, splits mid-sentence.
- `sentence_window` — packs whole sentences up to the target size. Chunks end where a
  thought ends, which makes retrieved context readable and makes a "was this claim
  supported?" judgement possible at all.

`sentence_window` is the default. `fixed` is kept because it is the honest baseline: if
the fancier strategy does not beat it on retrieval metrics, that is worth knowing and
reporting rather than assuming.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Iterator

# Sentence boundaries: terminal punctuation followed by whitespace and a capital or
# digit. Deliberately not a full NLP sentence splitter — this runs over academic
# abstracts, and the failure mode of a heavier dependency (different versions splitting
# differently) is worse here than the occasional missed boundary.
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")

# Common abbreviations that would otherwise split a sentence in the middle.
_PROTECTED = ["e.g.", "i.e.", "et al.", "cf.", "vs.", "Fig.", "Eq.", "Sec.", "Ref.",
              "approx.", "resp.", "Dr.", "Prof."]


@dataclass(frozen=True)
class Chunk:
    """One retrievable unit of text.

    `chunk_id` is derived from the document id and the chunk's position, not from a
    counter, so re-running ingestion on the same corpus produces the same ids. Anything
    that stores retrieval results keyed by chunk id therefore stays valid.
    """

    chunk_id: str
    doc_id: str
    index: int
    text: str
    start_char: int
    end_char: int
    metadata: dict = field(default_factory=dict)

    @property
    def n_chars(self) -> int:
        return len(self.text)


def _protect_abbreviations(text: str) -> tuple[str, dict[str, str]]:
    """Temporarily replace abbreviations so the sentence splitter ignores them."""
    replacements: dict[str, str] = {}
    for i, abbr in enumerate(_PROTECTED):
        if abbr in text:
            token = f"\x00ABBR{i}\x00"
            replacements[token] = abbr
            text = text.replace(abbr, token)
    return text, replacements


def _restore(text: str, replacements: dict[str, str]) -> str:
    for token, abbr in replacements.items():
        text = text.replace(token, abbr)
    return text


def split_sentences(text: str) -> list[str]:
    """Split into sentences, keeping abbreviations intact."""
    protected, replacements = _protect_abbreviations(text)
    parts = _SENTENCE_END.split(protected)
    return [_restore(p, replacements).strip() for p in parts if p.strip()]


def _make_chunk_id(doc_id: str, index: int, text: str) -> str:
    """Stable id: document, position, and a hash of the content.

    The content hash means a chunk whose text changed (because chunk_size changed, or
    the source was re-fetched and differs) gets a different id rather than silently
    inheriting the old one. That turns a subtle "my cached results are stale" bug into
    an obvious mismatch.
    """
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]
    return f"{doc_id}::{index:04d}::{digest}"


def chunk_fixed(text: str, doc_id: str, chunk_size: int, overlap: int,
                min_chars: int = 0, metadata: dict | None = None) -> list[Chunk]:
    """Fixed-size character windows with overlap."""
    if overlap >= chunk_size:
        raise ValueError(
            f"chunk_overlap ({overlap}) must be smaller than chunk_size ({chunk_size}); "
            "otherwise the window never advances and chunking does not terminate"
        )

    text = text.strip()
    if not text:
        return []

    chunks: list[Chunk] = []
    stride = chunk_size - overlap
    start = 0
    index = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        piece = text[start:end].strip()

        # A trailing fragment shorter than min_chars is dropped rather than kept as a
        # near-empty chunk that can never be usefully retrieved.
        if piece and (len(piece) >= min_chars or not chunks):
            chunks.append(Chunk(
                chunk_id=_make_chunk_id(doc_id, index, piece),
                doc_id=doc_id, index=index, text=piece,
                start_char=start, end_char=end,
                metadata=dict(metadata or {}),
            ))
            index += 1

        if end >= len(text):
            break
        start += stride

    return chunks


def chunk_sentence_window(text: str, doc_id: str, chunk_size: int, overlap: int,
                          min_chars: int = 0,
                          metadata: dict | None = None) -> list[Chunk]:
    """Pack whole sentences up to `chunk_size`, overlapping by whole sentences.

    Overlap is expressed in characters but applied at sentence granularity: trailing
    sentences are carried into the next chunk until the overlap budget is met. A chunk
    therefore never begins or ends mid-clause, which is what makes a retrieved passage
    judgeable by a human or a model.
    """
    if overlap >= chunk_size:
        raise ValueError(
            f"chunk_overlap ({overlap}) must be smaller than chunk_size ({chunk_size})"
        )

    text = text.strip()
    if not text:
        return []

    sentences = split_sentences(text)
    if not sentences:
        return []

    chunks: list[Chunk] = []
    current: list[str] = []
    current_len = 0
    index = 0
    cursor = 0          # character offset of the current chunk's start

    def flush() -> None:
        nonlocal current, current_len, index, cursor
        if not current:
            return
        piece = " ".join(current).strip()
        if piece and (len(piece) >= min_chars or not chunks):
            start = text.find(piece[:40], cursor) if len(piece) >= 40 else cursor
            start = start if start >= 0 else cursor
            chunks.append(Chunk(
                chunk_id=_make_chunk_id(doc_id, index, piece),
                doc_id=doc_id, index=index, text=piece,
                start_char=start, end_char=start + len(piece),
                metadata=dict(metadata or {}),
            ))
            index += 1
            cursor = start

    for sentence in sentences:
        # A single sentence longer than the target cannot be packed; emit what we have,
        # then hard-split the outlier rather than producing one enormous chunk.
        if len(sentence) > chunk_size:
            flush()
            current, current_len = [], 0
            for sub in chunk_fixed(sentence, doc_id, chunk_size, overlap,
                                   min_chars, metadata):
                chunks.append(Chunk(
                    chunk_id=_make_chunk_id(doc_id, index, sub.text),
                    doc_id=doc_id, index=index, text=sub.text,
                    start_char=sub.start_char, end_char=sub.end_char,
                    metadata=dict(metadata or {}),
                ))
                index += 1
            continue

        if current_len + len(sentence) + 1 > chunk_size and current:
            flush()
            # Carry trailing sentences forward until the overlap budget is met.
            carried: list[str] = []
            carried_len = 0
            for s in reversed(current):
                if carried_len + len(s) + 1 > overlap:
                    break
                carried.insert(0, s)
                carried_len += len(s) + 1
            current = carried
            current_len = carried_len

        current.append(sentence)
        current_len += len(sentence) + 1

    flush()
    return chunks


def chunk_document(text: str, doc_id: str, cfg, metadata: dict | None = None) -> list[Chunk]:
    """Chunk one document according to the configured strategy."""
    strategy = cfg.get("chunking.strategy", "sentence_window")
    size = int(cfg.get("chunking.chunk_size", 1000))
    overlap = int(cfg.get("chunking.chunk_overlap", 200))
    min_chars = int(cfg.get("chunking.min_chunk_chars", 0))

    if strategy == "fixed":
        return chunk_fixed(text, doc_id, size, overlap, min_chars, metadata)
    if strategy == "sentence_window":
        return chunk_sentence_window(text, doc_id, size, overlap, min_chars, metadata)
    raise ValueError(
        f"Unknown chunking strategy {strategy!r} — expected 'fixed' or 'sentence_window'"
    )


def chunk_corpus(documents: list[dict], cfg) -> Iterator[Chunk]:
    """Chunk many documents, carrying their metadata onto every chunk."""
    for doc in documents:
        meta = {k: v for k, v in doc.items() if k not in ("text", "doc_id")}
        yield from chunk_document(doc["text"], doc["doc_id"], cfg, meta)
