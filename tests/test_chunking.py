"""Chunking correctness.

Chunking fails silently. A bug here does not raise — retrieval just gets worse, every
downstream metric shifts, and nothing points at the cause. Since chunk size and overlap
become experimental variables later, these tests also pin determinism: the same input
must always produce byte-identical output, or a sweep is measuring noise.
"""

from __future__ import annotations

import pytest

from src.ingest.chunk import (
    Chunk,
    chunk_fixed,
    chunk_sentence_window,
    split_sentences,
)

ABSTRACT = (
    "Machine learning benchmarks are often contaminated. We show that standard "
    "random splits leak information between training and test sets. This inflates "
    "reported accuracy by a substantial margin. We propose a grouped splitting "
    "strategy that respects the underlying data structure. Our experiments across "
    "four architectures confirm the effect is universal in direction."
)


# --------------------------------------------------------------------------------------
# Sentence splitting
# --------------------------------------------------------------------------------------

def test_splits_on_sentence_boundaries():
    assert len(split_sentences(ABSTRACT)) == 5


def test_abbreviations_do_not_split_a_sentence():
    """'et al.' and 'e.g.' end in a full stop but do not end a sentence."""
    text = "Prior work by Smith et al. found no effect. We disagree."
    assert len(split_sentences(text)) == 2

    text2 = "Several methods exist, e.g. bagging and boosting. We use both."
    assert len(split_sentences(text2)) == 2


def test_empty_text_yields_no_sentences():
    assert split_sentences("") == []
    assert split_sentences("   \n  ") == []


# --------------------------------------------------------------------------------------
# Fixed-window chunking
# --------------------------------------------------------------------------------------

def test_fixed_covers_the_whole_document():
    """No text may be silently dropped — a lost passage is a question that can
    never be answered, and nothing would report it."""
    chunks = chunk_fixed(ABSTRACT, "doc1", chunk_size=100, overlap=20)
    joined = "".join(c.text for c in chunks)
    # every non-space character survives somewhere
    assert set(ABSTRACT.replace(" ", "")) <= set(joined.replace(" ", ""))
    assert chunks[0].start_char == 0
    assert chunks[-1].end_char >= len(ABSTRACT.strip()) - 1


def test_fixed_respects_the_size_limit():
    chunks = chunk_fixed(ABSTRACT, "doc1", chunk_size=120, overlap=20)
    assert all(c.n_chars <= 120 for c in chunks)


def test_overlap_at_least_as_large_as_chunk_size_is_rejected():
    """Otherwise the window never advances and chunking does not terminate."""
    with pytest.raises(ValueError, match="smaller than chunk_size"):
        chunk_fixed(ABSTRACT, "doc1", chunk_size=100, overlap=100)
    with pytest.raises(ValueError):
        chunk_fixed(ABSTRACT, "doc1", chunk_size=100, overlap=150)


def test_fixed_is_deterministic():
    a = chunk_fixed(ABSTRACT, "doc1", 100, 20)
    b = chunk_fixed(ABSTRACT, "doc1", 100, 20)
    assert [c.chunk_id for c in a] == [c.chunk_id for c in b]
    assert [c.text for c in a] == [c.text for c in b]


# --------------------------------------------------------------------------------------
# Sentence-window chunking
# --------------------------------------------------------------------------------------

def test_sentence_window_does_not_split_mid_sentence():
    """The point of this strategy: a retrieved passage must be readable, or a
    'was this claim supported?' judgement is impossible."""
    chunks = chunk_sentence_window(ABSTRACT, "doc1", chunk_size=200, overlap=50)
    for c in chunks:
        assert c.text.rstrip().endswith((".", "!", "?")), c.text


def test_sentence_window_respects_the_size_limit():
    chunks = chunk_sentence_window(ABSTRACT, "doc1", chunk_size=200, overlap=50)
    # a chunk may exceed the target only when one sentence alone is longer
    for c in chunks:
        assert c.n_chars <= 200 or len(split_sentences(c.text)) == 1


def test_sentence_window_overlaps():
    chunks = chunk_sentence_window(ABSTRACT, "doc1", chunk_size=200, overlap=80)
    if len(chunks) > 1:
        first = set(split_sentences(chunks[0].text))
        second = set(split_sentences(chunks[1].text))
        assert first & second, "consecutive chunks should share at least one sentence"


def test_a_sentence_longer_than_the_chunk_is_split_not_dropped():
    """The outlier case that would otherwise produce one enormous chunk or lose text."""
    long_sentence = "word " * 400 + "end."
    chunks = chunk_sentence_window(long_sentence, "doc1", chunk_size=200, overlap=40)
    assert len(chunks) > 1
    assert all(c.n_chars <= 200 for c in chunks)


def test_sentence_window_is_deterministic():
    a = chunk_sentence_window(ABSTRACT, "doc1", 200, 50)
    b = chunk_sentence_window(ABSTRACT, "doc1", 200, 50)
    assert [c.chunk_id for c in a] == [c.chunk_id for c in b]


# --------------------------------------------------------------------------------------
# Chunk identity
# --------------------------------------------------------------------------------------

def test_chunk_ids_are_unique_within_a_document():
    chunks = chunk_sentence_window(ABSTRACT, "doc1", 150, 40)
    assert len({c.chunk_id for c in chunks}) == len(chunks)


def test_chunk_ids_differ_across_documents():
    a = chunk_sentence_window(ABSTRACT, "doc1", 200, 50)
    b = chunk_sentence_window(ABSTRACT, "doc2", 200, 50)
    assert not ({c.chunk_id for c in a} & {c.chunk_id for c in b})


def test_changing_the_text_changes_the_id():
    """Content is hashed into the id so a stale cached result cannot silently be
    matched against a chunk whose text has since changed."""
    a = chunk_fixed("Some original text here.", "doc1", 100, 10)[0]
    b = chunk_fixed("Some different text here.", "doc1", 100, 10)[0]
    assert a.index == b.index
    assert a.chunk_id != b.chunk_id


# --------------------------------------------------------------------------------------
# Edge cases
# --------------------------------------------------------------------------------------

@pytest.mark.parametrize("fn", [chunk_fixed, chunk_sentence_window])
def test_empty_document_yields_no_chunks(fn):
    assert fn("", "doc1", 100, 20) == []
    assert fn("     ", "doc1", 100, 20) == []


@pytest.mark.parametrize("fn", [chunk_fixed, chunk_sentence_window])
def test_document_shorter_than_one_chunk(fn):
    chunks = fn("Short text.", "doc1", 1000, 200)
    assert len(chunks) == 1
    assert chunks[0].text == "Short text."


@pytest.mark.parametrize("fn", [chunk_fixed, chunk_sentence_window])
def test_metadata_is_carried_onto_every_chunk(fn):
    meta = {"title": "A Paper", "primary_category": "cs.LG"}
    chunks = fn(ABSTRACT, "doc1", 150, 40, metadata=meta)
    assert chunks
    for c in chunks:
        assert c.metadata["title"] == "A Paper"
        assert c.metadata["primary_category"] == "cs.LG"


def test_metadata_is_not_shared_between_chunks():
    """Mutating one chunk's metadata must not reach into another's."""
    chunks = chunk_fixed(ABSTRACT, "doc1", 100, 20, metadata={"k": "v"})
    assert len(chunks) > 1
    chunks[0].metadata["k"] = "changed"
    assert chunks[1].metadata["k"] == "v"


def test_tiny_trailing_fragments_are_dropped():
    """A 3-character chunk can never be usefully retrieved; it is noise in the index."""
    text = "a" * 205
    chunks = chunk_fixed(text, "doc1", chunk_size=100, overlap=0, min_chars=50)
    assert all(c.n_chars >= 50 for c in chunks)


def test_a_single_short_document_survives_min_chars():
    """min_chars must not delete a document entirely — better one small chunk than
    a paper silently absent from the corpus."""
    chunks = chunk_fixed("Tiny.", "doc1", chunk_size=100, overlap=0, min_chars=50)
    assert len(chunks) == 1
