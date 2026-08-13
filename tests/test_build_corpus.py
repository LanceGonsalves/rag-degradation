"""Assembling documents from papers and extracted full text.

The failure guarded here is silent substitution: a corpus that fell back to abstracts
for most of its papers looks perfectly healthy — every document is present, every chunk
is well-formed — and produces a benchmark that measures nothing, because retrieving the
top 5 of a few hundred abstract-sized chunks is trivially easy.
"""

from __future__ import annotations

from src.ingest.build_corpus import to_documents

PAPERS = [
    {"doc_id": "2104.08663", "title": "BEIR", "abstract": "A benchmark for retrieval.",
     "primary_category": "cs.IR", "published": "2021-04-17", "abs_url": "http://a/1"},
    {"doc_id": "2007.01282", "title": "Leakage", "abstract": "A survey of leakage.",
     "primary_category": "cs.LG", "published": "2020-07-02", "abs_url": "http://a/2"},
]

FULLTEXT = [
    {"doc_id": "2104.08663", "title": "BEIR",
     "text": "Full body text of the BEIR paper, considerably longer.", "source": "pdf"},
]


def test_without_fulltext_everything_falls_back_to_abstracts():
    docs = to_documents(PAPERS)
    assert len(docs) == 2
    assert all(d["source"] == "abstract" for d in docs)
    assert "A benchmark for retrieval." in docs[0]["text"]


def test_fulltext_replaces_the_abstract_where_available():
    docs = to_documents(PAPERS, FULLTEXT)
    beir = next(d for d in docs if d["doc_id"] == "2104.08663")
    assert beir["source"] == "pdf"
    assert "Full body text" in beir["text"]
    assert "A benchmark for retrieval." not in beir["text"]


def test_papers_without_fulltext_still_appear():
    """A PDF that failed to download must cost detail, not the whole document."""
    docs = to_documents(PAPERS, FULLTEXT)
    leakage = next(d for d in docs if d["doc_id"] == "2007.01282")
    assert leakage["source"] == "abstract"
    assert "A survey of leakage." in leakage["text"]


def test_metadata_always_comes_from_papers_not_the_extractor():
    """The extractor carries only title and text. Taking metadata from it would lose
    the category and URL, and retrieved chunks would be unattributable."""
    docs = to_documents(PAPERS, FULLTEXT)
    beir = next(d for d in docs if d["doc_id"] == "2104.08663")
    assert beir["primary_category"] == "cs.IR"
    assert beir["abs_url"] == "http://a/1"
    assert beir["published"] == "2021-04-17"


def test_source_is_recorded_per_document():
    """This is what makes a mostly-fallback corpus visible in the statistics rather
    than discovered three stages later when the metrics look wrong."""
    docs = to_documents(PAPERS, FULLTEXT)
    assert sorted(d["source"] for d in docs) == ["abstract", "pdf"]


def test_extractor_entries_for_unknown_papers_are_ignored():
    """Stale fulltext.jsonl from a previous, larger fetch must not inject documents
    that are no longer in the corpus."""
    stale = FULLTEXT + [{"doc_id": "9999.99999", "title": "Gone",
                         "text": "From an older fetch.", "source": "pdf"}]
    docs = to_documents(PAPERS, stale)
    assert len(docs) == 2
    assert all(d["doc_id"] != "9999.99999" for d in docs)


def test_an_extractor_entry_that_itself_fell_back_is_labelled_honestly():
    fallback = [{"doc_id": "2104.08663", "title": "BEIR",
                 "text": "BEIR\n\nA benchmark for retrieval.", "source": "abstract"}]
    docs = to_documents(PAPERS, fallback)
    beir = next(d for d in docs if d["doc_id"] == "2104.08663")
    assert beir["source"] == "abstract"
