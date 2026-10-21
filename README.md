# Robust Search: BM25 vs. Neural Retrieval under Noisy and Ambiguous Queries

Dual-engine retrieval system: Engine A is BM25 (implemented from scratch). Engine B is dense retrieval (BAAI/bge-small-en-v1.5 embeddings, cosine similarity). Baselines are TF-IDF cosine and a seeded random ranking. All engines run on the same Wikipedia subset and the adversarial query set, and are scored with NDCG@5 and MAP
against manually labelled relevance judgments.

## Setup

    pip install -r requirements.txt

Run all scripts from the project root, e.g. `python3 src/run_engines.py`.

## Folder structure

    data/          corpus, queries, labels (embeddings/ is cached and not committed)
    results/       ranked outputs and timing
    src/           pipeline code
    test_scripts/  sanity checks and diagnostics
    notebooks/     exploration and final plots
    docs/          architecture diagram, report drafts

## Source files (src/)

| File                  | Purpose                                                                                                                                                                                                                                                        |
| --------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `config.py`           | Single place for settings: paths, Wikipedia user agent, the 26 category labels and their caps, length limits (`MIN_WORDS`, `MAX_WORDS`), subcategory depth, random seed, and `INDEX_WORDS` (words per document seen by every engine).                          |
| `wiki_utils.py`       | Wikipedia API client and `collect_titles()`, which walks categories and returns candidate article titles. Used by the checker and the builder.                                                                                                                 |
| `check_categories.py` | Dry run before building. Prints, for each label, how many candidate articles are not yet in the corpus. Flags labels that are missing or low.                                                                                                                  |
| `build_corpus.py`     | Builds or extends `data/corpus.jsonl`. Resumable. Fills only the gap for each label, never changes existing doc IDs, and skips stubs and disambiguation pages.                                                                                                 |
| `data_utils.py`       | `load_corpus()`: the only place the corpus is loaded. Cuts text to `INDEX_WORDS` so every engine indexes identical content.                                                                                                                                    |
| `bm25_engine.py`      | `Tokenizer` (lowercase, regex tokens, optional stopword removal and Porter stemming) and `BM25` (inverted index, Manning Eq. 11.32, `explain()` for per-term score contributions).                                                                             |
| `tfidf_engine.py`     | Baseline: sublinear TF-IDF vectors ranked by cosine similarity. Shares BM25's tokenizer so only the weighting differs.                                                                                                                                         |
| `dense_engine.py`     | Engine B: encodes the corpus once with a sentence-embedding model, caches vectors in `data/embeddings/`, ranks by cosine similarity.                                                                                                                           |
| `run_engines.py`      | Runs random, TF-IDF, BM25, BM25 (stemmed, stopwords removed) and dense on every query in `queries.json`. Saves ranked lists with scores and records build time, per-query latency and memory. `--retime` re-encodes the corpus to re-measure dense build time. |
| `make_pool.py`        | Builds the labelling sheet from the union of each engine's top-10. Engine names and ranks are hidden and document order is shuffled, so labelling is unbiased.                                                                                                 |
| `evaluate.py`         | _(to be added)_ NDCG@5 and MAP, overall and per query category, plus results tables.                                                                                                                                                                           |
| `corpus_stats.py`     | Corpus statistics and a homonym check: which categories mention each ambiguous word.                                                                                                                                                                           |
| `coverage_check.py`   | Regex check that the corpus contains documents for each planned query type (negation, temporal, identifiers, and so on).                                                                                                                                       |
| `token_check.py`      | Counts wordpiece tokens per document at several word cutoffs against the embedding model's limit. Takes an optional model name. Used to choose `INDEX_WORDS`.                                                                                                  |
| `find_docs.py`        | Searches the indexed text with a regex and prints matches with their category. Used to check that a query has enough relevant documents and to find ones the engines missed.                                                                                   |

## Test and diagnostic scripts (test_scripts/)

| File                | Purpose                                                                                                                                    |
| ------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| `bm25_test.py`      | Checks BM25: stopword-list behaviour, build time and memory, demo queries, and top-10 overlap with the `rank_bm25` library.                |
| `dense_test.py`     | Checks the dense engine on demo queries and runs the negation probe, reporting top-10 overlap and cosine between a query and its negation. |
| `negation.py`       | BM25 negation probe: compares top-10 for a query and its negated form.                                                                     |
| `diagnostics.py`    | Warmed-up dense latency, cosine controls (negated vs. paraphrase vs. related vs. unrelated queries), and a TF-IDF smoke test.              |
| `corpus_details.py` | Exact corpus size, token counts and vocabulary, per label.                                                                                 |

If you move `bm25_test.py` from `src/` to `test_scripts/`, add these two lines at the top:

    import sys; from pathlib import Path
    sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

## Data and results files

| File                        | What it is                                                                                                      |
| --------------------------- | --------------------------------------------------------------------------------------------------------------- |
| `data/corpus.jsonl`         | The frozen corpus: 3,030 Wikipedia articles, up to 300 words each, with `doc_id`, `title`, `category`, `text`.  |
| `data/corpus_v1.jsonl`      | Backup of the first 1,100 documents (git-ignored).                                                              |
| `data/queries.json`         | The query set. Each entry has `qid`, `category`, `text`, and a `rule` that defines labels 0/1/2 for that query. |
| `data/label_sheet.csv`      | Generated by `make_pool.py`. Fill the `label` column (0/1/2), then save as `data/qrels.csv`.                    |
| `data/pool_meta.csv`        | Which engine retrieved each pooled document and at what rank. Do not open it until labelling is finished.       |
| `data/qrels.csv`            | Final relevance judgments (`qid`, `doc_id`, `label`).                                                           |
| `data/embeddings/`          | Cached document vectors, named by model and `INDEX_WORDS`. Delete if the corpus changes.                        |
| `results/run_<engine>.json` | Top-200 ranking per query as `[doc_id, score]`.                                                                 |
| `results/timing.json`       | Build time, latency and memory per engine.                                                                      |

## Pipeline order

1. `python3 src/check_categories.py`
2. `python3 src/build_corpus.py` (done; corpus is frozen)
3. `python3 src/corpus_stats.py`, `src/coverage_check.py`, `src/token_check.py`
4. Write `data/queries.json` using `src/find_docs.py` to verify each query
5. `python3 src/run_engines.py --retime`
6. `python3 src/make_pool.py`, label `data/label_sheet.csv`, save `data/qrels.csv`
7. `python3 src/evaluate.py` _(to be added)_

## Design decisions

- Both engines index the same first 250 words of each article. The embedding model reads at most 512 tokens, and only 7 of 3,030 documents exceed that at 250 words.
- BM25 keeps stopwords by default so that words like "not" and "after" are at least present in the index. A stemmed, stopword-removed variant runs as an ablation.
- The corpus is Wikipedia, chosen so that homonym senses and negation cases occur naturally. Relevance is judged only on the indexed text.
- Pooling is blind: engine identity and rank are hidden while labelling.
- The corpus was frozen at 3,030 documents (26 labels) before any labelling, because changing it invalidates the labels and the cached embeddings.

## Status

- [x] Repository and folder structure
- [x] Corpus (3,030 documents, frozen)
- [x] BM25 engine, validated against `rank_bm25`
- [x] Dense engine and TF-IDF baseline
- [x] Runner and pooling script
- [ ] Final query set
- [ ] Relevance labels (qrels)
- [ ] Evaluation (NDCG@5, MAP, per category)
- [ ] Failure-mode analysis
- [ ] Stretch: hybrid fusion, negation-aware preprocessing, multiple embedding models, label agreement check

## Data attribution

Article text comes from English Wikipedia (CC BY-SA 4.0).
