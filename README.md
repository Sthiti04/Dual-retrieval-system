# Robust Search: BM25 vs. Neural Retrieval under Noisy and Ambiguous Queries

Information Retrieval course project.Members: Sthiti Swain, Ritu Laharwani, Sarthak Shashi.

## What this project does

We compare two ways of ranking documents. BM25 scores a document by the words it shares with the query. Dense retrieval turns text into embeddings and ranks by cosine similarity. We run both on 3,030 Wikipedia articles with 20 hard queries, then score them against manually labelled relevance judgments (NDCG@5, P@5, MAP).

The queries fall into five categories: negation (5), homonym (5), temporal (4), identifier (3) and typo (3).

## Engines

| Engine       | Description                                                      |
| ------------ | ---------------------------------------------------------------- |
| Random       | Seeded shuffle, a floor baseline                                 |
| TF-IDF       | Lexical baseline, cosine similarity                              |
| BM25         | Engine A, implemented from scratch (Manning et al., Eq. 11.32)   |
| BM25 stemmed | Same BM25 with stopwords removed and Porter stemming             |
| Dense        | Engine B, `BAAI/bge-small-en-v1.5` embeddings, cosine similarity |

## Setup

    pip install -r requirements.txt

Run every script from the project root, for example `python3 src/run_engines.py`. The first dense run downloads a model of about 130 MB. `memory_check.py` works on macOS and Linux only.

## Folder layout

    data/          corpus, queries, relevance judgments (embeddings/ is cached, not committed)
    results/       rankings, metric tables, timing
    src/           main pipeline
    test_scripts/  sanity checks and diagnostics

## Scripts in src/

**Corpus building**

| File                  | Purpose                                                                                                                                  |
| --------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| `config.py`           | All settings: paths, 26 Wikipedia category labels, length limits, random seed, `INDEX_WORDS` (words per article that every engine reads) |
| `wiki_utils.py`       | Wikipedia API client that collects article titles from categories                                                                        |
| `check_categories.py` | Dry run that counts new candidate articles per label before building                                                                     |
| `build_corpus.py`     | Builds `data/corpus.jsonl`. Skips stubs, disambiguation and "List of" pages                                                              |
| `data_utils.py`       | `load_corpus()`. The only place the corpus is loaded, so every engine sees the same text                                                 |

**Engines and runner**

| File              | Purpose                                                                              |
| ----------------- | ------------------------------------------------------------------------------------ |
| `bm25_engine.py`  | Tokenizer and BM25 with an inverted index. `explain()` shows each query term's score |
| `tfidf_engine.py` | TF-IDF baseline, shares BM25's tokenizer                                             |
| `dense_engine.py` | Encodes the corpus once, caches embeddings, ranks by cosine similarity               |
| `run_engines.py`  | Runs all five engines on every query. Saves top-200 rankings and build times         |

**Relevance judgments**

| File               | Purpose                                                                                                                   |
| ------------------ | ------------------------------------------------------------------------------------------------------------------------- |
| `make_pool.py`     | Round 1: labelling sheet from the top 5 of BM25, TF-IDF and dense (historical, do not rerun)                              |
| `make_pool_new.py` | Round 2: pools the top 5 of four engines for all queries, skips documents already judged, hides engine names and shuffles |
| `merge_qrels.py`   | Checks new labels and merges them into `data/qrels.csv`                                                                   |
| `copy_qrels.py`    | Copies a clean query's labels to its typo copy                                                                            |

**Evaluation and analysis**

| File                | Purpose                                                                                                                                  |
| ------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| `evaluate.py`       | NDCG@5, P@5 and approximate MAP, overall, per category and per query. `--strict` counts only label 2 as relevant. Saves CSVs and a chart |
| `robustness.py`     | Typo stability, negation sensitivity and engine agreement. Needs no labels                                                               |
| `memory_check.py`   | Peak memory of one engine in its own process, for example `python3 src/memory_check.py dense`                                            |
| `latency_check.py`  | Mean query time for each engine                                                                                                          |
| `show_top5.py`      | Prints each engine's top 5 with labels for chosen queries                                                                                |
| `probe_bm25.py`     | Shows how much each query word contributes to a BM25 score                                                                               |
| `probe_variants.py` | Compares BM25 with stemming only, stopword removal only, and both                                                                        |
| `demo_app.py`       | Streamlit page showing BM25 and dense results side by side                                                                               |

**Corpus and query checks**

| File                | Purpose                                                                               |
| ------------------- | ------------------------------------------------------------------------------------- |
| `corpus_stats.py`   | Corpus size per label and which categories mention each ambiguous word                |
| `coverage_check.py` | Regex check that the corpus supports each query type                                  |
| `token_check.py`    | Counts model tokens per article at several word cutoffs. Used to choose `INDEX_WORDS` |
| `find_docs.py`      | Regex search over the indexed text. Used to check each query has relevant documents   |
| `verify_queries.py` | Counts regex matches for candidate queries                                            |

## Scripts in test_scripts/

| File                | Purpose                                                                                         |
| ------------------- | ----------------------------------------------------------------------------------------------- |
| `bm25_test.py`      | Compares our BM25 with the `rank_bm25` library                                                  |
| `diagnostics.py`    | Dense query-encoding time and cosine controls (negated, paraphrase, related, unrelated queries) |
| `corpus_details.py` | Exact corpus size and token counts per label                                                    |

## Data and results

| File                                                       | What it is                                                              |
| ---------------------------------------------------------- | ----------------------------------------------------------------------- |
| `data/corpus.jsonl`                                        | The frozen corpus. Fields: `doc_id`, `title`, `category`, `text`        |
| `data/queries.json`                                        | The 20 queries, each with a written rule that defines labels 0, 1 and 2 |
| `data/qrels.csv`                                           | Final relevance judgments (`qid`, `doc_id`, `label`)                    |
| `data/pool_meta_v2.csv`                                    | Which engine retrieved each pooled document, and at what rank           |
| `results/run_<engine>.json`                                | Top-200 ranking per query as `[doc_id, score]`                          |
| `results/timing.json`                                      | Index build time per engine                                             |
| `results/metrics_*.csv`, `results/ndcg5_by_category_*.png` | Evaluation tables and chart, in lenient and strict settings             |

## Reproducing the results

    python3 -m pip install -r requirements.txt
    python3 src/run_engines.py
    python3 src/evaluate.py
    python3 src/evaluate.py --strict
    python3 src/robustness.py
    python3 src/latency_check.py
    python3 src/show_top5.py q03 q06

The first command downloads the model, and `run_engines.py` encodes the corpus once (about a minute on an Apple GPU, longer on a CPU).

Do not run these, because they would change the corpus or the labels: `build_corpus.py` (Wikipedia changes over time, so the committed corpus is the corpus), `merge_qrels.py`, `copy_qrels.py`.
