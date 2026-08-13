"""arXiv Atom parsing.

The parser cannot be developed against the live API from every environment (CI has no
outbound access to export.arxiv.org), so it is pinned against a fixture that mirrors a
real response — including the parts that are easy to get wrong: namespaced elements,
versioned ids, wrapped abstracts, and entries missing fields.
"""

from __future__ import annotations

import pytest

from src.ingest.arxiv import _strip_version, parse_entry, parse_response

# Trimmed from a real export.arxiv.org response. Whitespace inside <summary> is
# deliberately preserved: arXiv hard-wraps abstracts and the parser must unwrap them.
FEED = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom"
      xmlns:arxiv="http://arxiv.org/schemas/atom">
  <entry>
    <id>http://arxiv.org/abs/2104.08663v2</id>
    <updated>2021-08-17T14:22:01Z</updated>
    <published>2021-04-17T09:11:43Z</published>
    <title>BEIR: A Heterogeneous Benchmark for Zero-shot
      Evaluation of Information Retrieval Models</title>
    <summary>  Existing neural information retrieval models are often
studied in homogeneous settings. We introduce BEIR, a robust
benchmark spanning nine tasks.
</summary>
    <author><name>Nandan Thakur</name></author>
    <author><name>Nils Reimers</name></author>
    <arxiv:primary_category term="cs.IR" scheme="http://arxiv.org/schemas/atom"/>
    <category term="cs.IR" scheme="http://arxiv.org/schemas/atom"/>
    <category term="cs.CL" scheme="http://arxiv.org/schemas/atom"/>
    <link href="http://arxiv.org/abs/2104.08663v2" rel="alternate" type="text/html"/>
    <link title="pdf" href="http://arxiv.org/pdf/2104.08663v2" rel="related"
          type="application/pdf"/>
  </entry>
  <entry>
    <id>http://arxiv.org/abs/2007.01282v1</id>
    <updated>2020-07-02T18:00:00Z</updated>
    <published>2020-07-02T18:00:00Z</published>
    <title>Leakage in Machine Learning Pipelines</title>
    <summary>We survey how data leakage arises in practice.</summary>
    <author><name>A Researcher</name></author>
    <arxiv:primary_category term="cs.LG" scheme="http://arxiv.org/schemas/atom"/>
    <category term="cs.LG" scheme="http://arxiv.org/schemas/atom"/>
    <link href="http://arxiv.org/abs/2007.01282v1" rel="alternate" type="text/html"/>
    <link title="pdf" href="http://arxiv.org/pdf/2007.01282v1" rel="related"
          type="application/pdf"/>
  </entry>
</feed>
"""

MALFORMED = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom"
      xmlns:arxiv="http://arxiv.org/schemas/atom">
  <entry>
    <id>http://arxiv.org/abs/1111.1111v1</id>
    <title>Has a title but no abstract</title>
  </entry>
  <entry>
    <id>http://arxiv.org/abs/2222.2222v1</id>
    <title>A Complete Entry</title>
    <summary>This one has everything it needs.</summary>
    <arxiv:primary_category term="cs.LG" scheme="http://arxiv.org/schemas/atom"/>
    <link href="http://arxiv.org/abs/2222.2222v1" rel="alternate" type="text/html"/>
  </entry>
</feed>
"""


def test_parses_every_well_formed_entry():
    papers = parse_response(FEED)
    assert len(papers) == 2


def test_version_is_stripped_from_the_id():
    """A revised paper must not enter the corpus twice under v1 and v2 — it would
    duplicate in every retrieval result and quietly distort precision."""
    papers = parse_response(FEED)
    assert papers[0].doc_id == "2104.08663"
    assert papers[1].doc_id == "2007.01282"


@pytest.mark.parametrize("raw,expected", [
    ("http://arxiv.org/abs/2104.08663v2", "2104.08663"),
    ("http://arxiv.org/abs/2104.08663v11", "2104.08663"),
    ("http://arxiv.org/abs/2104.08663", "2104.08663"),
    ("http://arxiv.org/abs/cs/0701001v1", "cs/0701001".split("/")[-1]),
])
def test_strip_version(raw, expected):
    assert _strip_version(raw) == expected


def test_wrapped_abstract_is_unwrapped():
    """arXiv hard-wraps at ~80 chars. Left as-is, the newlines would land inside
    chunks and change how text is split."""
    paper = parse_response(FEED)[0]
    assert "\n" not in paper.abstract
    assert "  " not in paper.abstract
    assert paper.abstract.startswith("Existing neural information retrieval")


def test_multiline_title_is_flattened():
    paper = parse_response(FEED)[0]
    assert "\n" not in paper.title
    assert paper.title.startswith("BEIR: A Heterogeneous Benchmark")


def test_authors_and_categories_are_captured():
    paper = parse_response(FEED)[0]
    assert paper.authors == ["Nandan Thakur", "Nils Reimers"]
    assert paper.primary_category == "cs.IR"
    assert set(paper.categories) == {"cs.IR", "cs.CL"}


def test_pdf_and_abs_links_are_distinguished():
    paper = parse_response(FEED)[0]
    assert paper.pdf_url.endswith("pdf/2104.08663v2")
    assert "abs/2104.08663" in paper.abs_url


def test_an_incomplete_entry_is_skipped_not_fatal():
    """One malformed entry must not lose the other ninety-nine in the same response."""
    papers = parse_response(MALFORMED)
    assert len(papers) == 1
    assert papers[0].doc_id == "2222.2222"


def test_text_property_prepends_the_title():
    """Chunks carry their own context, so a retrieved passage is identifiable
    without a separate lookup."""
    paper = parse_response(FEED)[1]
    assert paper.text.startswith("Leakage in Machine Learning Pipelines")
    assert paper.abstract in paper.text


def test_empty_feed_yields_nothing():
    empty = '<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom"></feed>'
    assert parse_response(empty) == []
