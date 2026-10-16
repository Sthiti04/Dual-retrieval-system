# Robust Search: BM25 vs. Neural Retrieval under Noisy and Ambiguous Queries

## Overview
Dual-engine retrieval system comparing BM25 (Engine A) and dense retrieval with
sentence embeddings + cosine similarity (Engine B) on a curated Wikipedia subset
using negation, homonym and temporal queries.

## Setup
pip install -r requirements.txt

## Pipeline (run from project root)
1. python src/check_categories.py   # verify Wikipedia categories
2. python src/build_corpus.py       # builds data/corpus.jsonl
3. python src/corpus_stats.py       # corpus statistics
(more steps will be added as modules are completed)

## Status
- [x] Repository and folder structure
- [ ] Corpus
- [ ] BM25 engine
- [ ] Dense engine
- [ ] Query suite and qrels
- [ ] Evaluation