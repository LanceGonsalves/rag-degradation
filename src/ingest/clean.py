"""Cleaning text extracted from PDFs.

PDF extraction produces text that *looks* fine and retrieves badly. The damage is
systematic rather than random, so it is worth fixing explicitly:

- **Hyphenation.** A word broken across a line becomes `evalua- tion`, which no
  tokeniser will match against `evaluation`. Every occurrence of a hyphenated term is
  a retrieval miss waiting to happen.
- **Ligatures.** `ﬁ` and `ﬂ` are single codepoints in most PDF fonts, so `classiﬁcation`
  never matches `classification` under exact search and embeds differently under dense
  retrieval.
- **Running headers and footers.** The paper title and page number repeat on every
  page. Left in, they appear in dozens of chunks and become the most "common" content
  in the document — actively misleading a lexical retriever.
- **References.** Thirty citations of other papers' titles are excellent bait for a
  keyword search and contain no answers. They are the single largest source of
  plausible-looking wrong retrievals in an academic corpus.

Every function here is pure text-in, text-out, so the behaviour can be pinned by tests
without a PDF anywhere near them.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter

# Ligatures and typographic characters that PDF fonts emit as single codepoints.
_REPLACEMENTS = {
    "ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl", "ﬃ": "ffi", "ﬄ": "ffl",
    "ﬅ": "st", "ﬆ": "st",
    "‘": "'", "’": "'", "“": '"', "”": '"',
    "–": "-", "—": "-", "−": "-", "­": "",
    " ": " ", " ": " ", " ": " ", " ": " ",
}

# Headings that begin the back matter. Matched on a line of their own, optionally
# numbered, because "references" appears mid-sentence in normal prose all the time.
_BACK_MATTER = re.compile(
    r"^\s*(?:\d+\.?\s*)?(references|bibliography|acknowledg(?:e)?ments)\s*$",
    re.IGNORECASE,
)

# A line that is mostly symbols or digits: equations, page furniture, table rules.
_MOSTLY_NON_ALPHA = re.compile(r"^[^A-Za-z]*$")


def normalise_unicode(text: str) -> str:
    """Expand ligatures, straighten quotes, and drop soft hyphens."""
    for src, dst in _REPLACEMENTS.items():
        text = text.replace(src, dst)
    # NFKC folds remaining compatibility forms (e.g. fullwidth characters).
    return unicodedata.normalize("NFKC", text)


def dehyphenate(text: str) -> str:
    """Rejoin words split across a line break.

    Only joins when a lowercase letter precedes the hyphen and follows the break,
    which leaves genuine compounds ("state-of-the-art", "train-test") intact — those
    have no line break inside them.
    """
    # "evalua-\ntion" -> "evaluation"
    text = re.sub(r"([a-z])-\s*\n\s*([a-z])", r"\1\2", text)
    # the same after whitespace has already been flattened to spaces
    text = re.sub(r"([a-z])-\s+([a-z])", lambda m: m.group(1) + m.group(2), text)
    return text


def strip_back_matter(text: str) -> tuple[str, bool]:
    """Cut everything from the references heading onward.

    Returns `(text, was_truncated)`. The flag matters: a paper where no heading was
    found still carries its bibliography, and that is worth knowing when the retrieval
    numbers look odd.

    Uses the *last* matching heading, because papers cite the word "References" in
    figure captions and appendix pointers before the real section.
    """
    lines = text.split("\n")
    cut = None
    for i, line in enumerate(lines):
        if _BACK_MATTER.match(line):
            cut = i
    if cut is None or cut < len(lines) * 0.3:
        # Refuse to cut in the first 30% — that is not a bibliography, it is a
        # mis-detected heading, and truncating there would destroy the paper.
        return text, False
    return "\n".join(lines[:cut]), True


def drop_repeated_lines(text: str, page_count: int, threshold: float = 0.5) -> str:
    """Remove running headers and footers.

    A short line appearing on more than `threshold` of the pages is furniture, not
    content. Length-capped so a genuinely repeated sentence in the body is not removed.
    """
    if page_count < 3:
        return text

    lines = text.split("\n")
    counts = Counter(l.strip() for l in lines if 0 < len(l.strip()) <= 120)
    furniture = {
        line for line, n in counts.items()
        if n >= max(3, int(page_count * threshold)) and line
    }
    if not furniture:
        return text
    return "\n".join(l for l in lines if l.strip() not in furniture)


def drop_noise_lines(text: str, min_alpha_ratio: float = 0.5) -> str:
    """Drop lines that are mostly not prose — equations, page numbers, table rules."""
    kept = []
    for line in text.split("\n"):
        stripped = line.strip()
        if not stripped:
            kept.append("")
            continue
        if _MOSTLY_NON_ALPHA.match(stripped):
            continue
        alpha = sum(c.isalpha() or c.isspace() for c in stripped)
        if alpha / len(stripped) >= min_alpha_ratio:
            kept.append(line)
    return "\n".join(kept)


def collapse_whitespace(text: str) -> str:
    """Flatten to paragraphs: single newlines join, blank lines separate."""
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    # join lines inside a paragraph
    text = re.sub(r"(?<![\n.])\n(?![\n])", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def clean_pdf_text(text: str, page_count: int = 1,
                   strip_references: bool = True) -> tuple[str, dict]:
    """Run the full cleaning pipeline.

    Returns the cleaned text and a report of what happened, so a paper that came out
    badly can be identified rather than silently polluting the corpus.
    """
    original_len = len(text)

    text = normalise_unicode(text)
    text = drop_repeated_lines(text, page_count)

    truncated = False
    if strip_references:
        text, truncated = strip_back_matter(text)

    text = dehyphenate(text)
    text = drop_noise_lines(text)
    text = collapse_whitespace(text)

    report = {
        "original_chars": original_len,
        "cleaned_chars": len(text),
        "retained_ratio": round(len(text) / original_len, 3) if original_len else 0.0,
        "references_stripped": truncated,
    }
    return text, report
