"""PDF text cleaning.

These guard failures that produce no error and no obviously wrong output — they just
make retrieval quietly worse. A hyphenated word that never matches its query, a
bibliography full of other papers' titles ranking above the real answer, a running
header repeated in forty chunks. Each is invisible until the metrics are unexplainably
bad, and by then the cause is three stages upstream.
"""

from __future__ import annotations

from src.ingest.clean import (
    clean_pdf_text,
    collapse_whitespace,
    dehyphenate,
    drop_noise_lines,
    drop_repeated_lines,
    normalise_unicode,
    strip_back_matter,
)


# --------------------------------------------------------------------------------------
# Unicode
# --------------------------------------------------------------------------------------

def test_ligatures_are_expanded():
    """`classiﬁcation` is one codepoint for `fi` — it never matches a query typed
    with two, under lexical search or dense embedding."""
    assert normalise_unicode("classiﬁcation") == "classification"
    assert normalise_unicode("eﬀective") == "effective"
    assert normalise_unicode("ﬂow") == "flow"


def test_typographic_quotes_and_dashes_are_straightened():
    assert normalise_unicode("“quoted”") == '"quoted"'
    assert normalise_unicode("model’s") == "model's"
    assert normalise_unicode("2020–2021") == "2020-2021"


def test_soft_hyphens_are_removed():
    assert normalise_unicode("eval­uation") == "evaluation"


# --------------------------------------------------------------------------------------
# Hyphenation
# --------------------------------------------------------------------------------------

def test_words_split_across_lines_are_rejoined():
    assert dehyphenate("evalua-\ntion") == "evaluation"
    assert dehyphenate("contami-\n  nation") == "contamination"


def test_genuine_compounds_survive():
    """`train-test` and `state-of-the-art` must not be silently welded together —
    they carry meaning and appear in queries exactly as written."""
    assert dehyphenate("train-test split") == "train-test split"
    assert dehyphenate("state-of-the-art") == "state-of-the-art"


def test_hyphen_before_a_capital_is_left_alone():
    assert dehyphenate("Cross-\nValidation") == "Cross-\nValidation"


# --------------------------------------------------------------------------------------
# Back matter
# --------------------------------------------------------------------------------------

BODY = "\n".join([f"Body line {i} with real content about evaluation." for i in range(20)])


def test_references_section_is_removed():
    """Thirty citations of other papers' titles are perfect bait for lexical
    retrieval and contain no answers."""
    text = BODY + "\nReferences\n[1] Someone. A paper title. 2020.\n[2] Another. 2021."
    cleaned, truncated = strip_back_matter(text)
    assert truncated
    assert "A paper title" not in cleaned
    assert "Body line 19" in cleaned


def test_numbered_references_heading_is_matched():
    text = BODY + "\n7. References\n[1] Someone. 2020."
    cleaned, truncated = strip_back_matter(text)
    assert truncated
    assert "Someone" not in cleaned


def test_acknowledgements_also_count_as_back_matter():
    text = BODY + "\nAcknowledgements\nWe thank the reviewers."
    cleaned, truncated = strip_back_matter(text)
    assert truncated
    assert "thank the reviewers" not in cleaned


def test_the_word_references_mid_sentence_does_not_truncate():
    text = "We compare against references from prior work.\n" + BODY
    cleaned, truncated = strip_back_matter(text)
    assert not truncated
    assert "Body line 19" in cleaned


def test_a_heading_too_early_is_refused():
    """Cutting at 5% would destroy the paper. Better to keep the bibliography than
    to lose the content and never notice."""
    text = "Intro line.\nReferences\n" + BODY
    cleaned, truncated = strip_back_matter(text)
    assert not truncated
    assert "Body line 19" in cleaned


def test_no_heading_reports_not_truncated():
    """The flag is the signal that this paper still carries its bibliography."""
    cleaned, truncated = strip_back_matter(BODY)
    assert not truncated
    assert cleaned == BODY


# --------------------------------------------------------------------------------------
# Running headers and footers
# --------------------------------------------------------------------------------------

def test_repeated_headers_are_dropped():
    header = "Data Leakage in Machine Learning Pipelines"
    lines = []
    for page in range(10):
        lines += [header, f"Content unique to page {page}.", str(page + 1)]
    cleaned = drop_repeated_lines("\n".join(lines), page_count=10)
    assert header not in cleaned
    assert "Content unique to page 5." in cleaned


def test_short_documents_are_left_alone():
    """With two pages there is no evidence a repeated line is furniture."""
    text = "Title\nContent.\nTitle\nMore."
    assert drop_repeated_lines(text, page_count=2) == text


def test_a_long_repeated_line_is_not_treated_as_furniture():
    long_line = "This is a genuinely long sentence of body text that happens to recur." * 2
    lines = [long_line] * 8
    cleaned = drop_repeated_lines("\n".join(lines), page_count=8)
    assert long_line in cleaned


# --------------------------------------------------------------------------------------
# Noise
# --------------------------------------------------------------------------------------

def test_equation_and_number_lines_are_dropped():
    text = "Real prose about models.\n= 0.95 + 3.2 ( )\n42\nMore real prose here."
    cleaned = drop_noise_lines(text)
    assert "Real prose" in cleaned
    assert "More real prose" in cleaned
    assert "= 0.95" not in cleaned
    assert "\n42\n" not in cleaned


def test_prose_containing_numbers_survives():
    text = "We report 6.46 points of inflation across 14 paired runs."
    assert "6.46" in drop_noise_lines(text)


# --------------------------------------------------------------------------------------
# Whitespace
# --------------------------------------------------------------------------------------

def test_lines_within_a_paragraph_are_joined():
    text = "This sentence was\nwrapped across lines\nby the extractor."
    out = collapse_whitespace(text)
    assert "\n" not in out
    assert "was wrapped across lines by" in out


def test_paragraph_breaks_survive():
    text = "First paragraph.\n\nSecond paragraph."
    out = collapse_whitespace(text)
    assert "\n\n" in out


# --------------------------------------------------------------------------------------
# The pipeline
# --------------------------------------------------------------------------------------

def test_full_pipeline_reports_what_it_did():
    raw = "\n".join([
        "Paper Title Header",
        "We study classiﬁcation under contami-",
        "nation of the evalua-",
        "tion set.",
        "Paper Title Header",
        "Results show a large eﬀect.",
    ] + [f"Line {i} of substantive body text about evaluation methods." for i in range(30)]
      + ["References", "[1] Someone. A cited title. 2020."])

    cleaned, report = clean_pdf_text(raw, page_count=6)

    assert "classification" in cleaned
    assert "contamination" in cleaned
    assert "evaluation set" in cleaned
    assert "effect" in cleaned
    assert "A cited title" not in cleaned
    assert report["references_stripped"] is True
    assert 0 < report["retained_ratio"] < 1


def test_pipeline_can_keep_references_when_asked():
    raw = BODY + "\nReferences\n[1] Someone. A cited title. 2020."
    cleaned, report = clean_pdf_text(raw, page_count=5, strip_references=False)
    assert "A cited title" in cleaned
    assert report["references_stripped"] is False


def test_empty_input_does_not_crash():
    cleaned, report = clean_pdf_text("", page_count=0)
    assert cleaned == ""
    assert report["retained_ratio"] == 0.0
