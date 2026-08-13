<h1 align="center">📄 RAG Degradation</h1>

<p align="center">
  <i>Retrieval-augmented systems are shipped constantly and measured rarely.<br>
  This one measures what happens when retrieval fails.</i>
</p>

<p align="center">
  <b>When retrieval degrades, how fast does the model start making things up —<br>
  and does it say "I don't know", or answer anyway with the same confidence?</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white">
  <img src="https://img.shields.io/badge/status-in%20progress-F39C12?style=flat-square">
</p>

---

## The question

Most RAG projects demonstrate that the happy path works: a good question, a good
retrieval, a good answer. Very few measure the failure mode that actually matters in
production — what the system does when retrieval returns the wrong passage.

Two numbers almost nobody reports:

1. **The dose–response curve.** How much retrieval quality can you lose before
   groundedness collapses? Is it linear, or is there a cliff?
2. **The abstention rate.** When the correct passage is absent, does the model decline
   to answer, or does it produce a confident, fluent, wrong answer? A system that says
   "I don't know" is safe. One that doesn't is dangerous.

## Method

The same experimental move as
[skin-lesion-leakage-benchmark](https://github.com/LanceGonsalves/skin-lesion-leakage-benchmark):
hold everything fixed, vary one thing on a controlled scale, publish the curve.

Generator, prompt and question set stay constant. **Only the quality of the retrieved
context changes**, across five conditions: oracle passage supplied, normal top-*k*
retrieval, correct passage removed with probability *p*, correct passage padded with
plausible distractors, and no context at all.

## Corpus

arXiv papers on machine-learning evaluation, benchmarking and data leakage.

Chosen for one reason: **ground-truth questions are only worth writing if you can verify
the answers.** This is a literature I have worked in — the leakage benchmark above is a
contribution to it — so I can tell when a generated answer is subtly wrong, which is the
entire basis of the eval set.

The corpus is **not redistributed**. `data/` is gitignored; fetch it yourself:

```bash
pip install -r requirements.txt
python -m src.ingest.arxiv                          # metadata + abstracts
python -m src.ingest.pdf                            # download PDFs, extract full text
python -m src.ingest.build_corpus --source fulltext # chunk into a retrievable corpus
```

**Why full text rather than abstracts.** Abstracts give ~2 chunks per paper. At that
size, returning the top 5 of a few hundred chunks is trivially easy and BM25, dense and
hybrid all score the same — not because they are equivalent, but because the task does
not discriminate. Full text gives ~40 chunks per paper, which makes retrieval a real
problem and lets a question require finding a *passage* rather than recognising a whole
abstract.

## Progress

- [x] **Corpus ingestion** — arXiv API client, deterministic chunking
- [x] **PDF extraction** — download, extract, and clean full text (66 tests)
- [ ] Retrieval: BM25, dense, hybrid
- [ ] Ground-truth question set (60–100, LLM-drafted and hand-verified)
- [ ] Retrieval evaluation: precision@k, recall@k, MRR, nDCG with bootstrap CIs
- [ ] Chunk-size sweep
- [ ] Generation + groundedness and abstention measurement
- [ ] The degradation experiment
- [ ] Deploy as a live service
- [ ] Write-up

## Design notes

**Chunking is the quietest source of error in a RAG system.** A bug there does not
raise — retrieval simply gets worse, every downstream metric shifts, and no stack trace
points at the cause. Chunk size and overlap are also experimental variables later, so
chunking is deterministic by construction: the same input always produces byte-identical
output, and chunk ids embed a content hash so a stale cached result cannot silently be
matched against text that has since changed.

**Two chunking strategies, on purpose.** `sentence_window` packs whole sentences so a
retrieved passage is readable and judgeable. `fixed` is the honest baseline — if the
fancier strategy does not beat it on retrieval metrics, that is worth reporting rather
than assuming.

**Versions are stripped from arXiv ids.** A paper revised from v1 to v2 would otherwise
enter the corpus twice, duplicate in every retrieval result, and quietly distort
precision.

**PDF text is cleaned, not just extracted.** Raw extraction produces text that looks
fine and retrieves badly, and the damage is systematic rather than random:

| problem | why it matters |
|---|---|
| `evalua-\ntion` split across a line | never matches `evaluation`; every hyphenated term is a retrieval miss |
| `classiﬁcation` with a single-codepoint ligature | never matches text typed with two characters |
| Running headers repeated on every page | become the most "common" content in the document and mislead a lexical retriever |
| The references section | thirty of *other* papers' titles — perfect bait for keyword search, containing no answers |

Each is fixed explicitly and each fix is pinned by a test. A paper where cleaning went
wrong is reported rather than silently entering the corpus, and `source` records per
document whether full text was used or the abstract was a fallback — so a corpus that
quietly degraded is visible in the statistics.

## Tests

```bash
python -m pytest -q     # 66 tests
```

The arXiv parser is pinned against a fixture mirroring a real Atom response, because
`export.arxiv.org` is not reachable from every CI runner. The fixture deliberately
includes the parts that are easy to get wrong: namespaced elements, versioned ids,
hard-wrapped abstracts, and an entry missing required fields.

---

<sub>Part of a portfolio at <a href="https://github.com/LanceGonsalves">github.com/LanceGonsalves</a>.</sub>
