import time, tracemalloc
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from bm25_engine import BM25
from data_utils import load_corpus

ids, titles, texts = load_corpus()
title = dict(zip(ids, titles))

# 1. Which important words are in sklearn's stopword list?
print("In sklearn stopword list?")
for w in ["not", "no", "never", "without", "after", "before", "first", "during", "until", "only"]:
    print(f"  {w}: {w in ENGLISH_STOP_WORDS}")

# 2. Index build time and peak memory (Python allocations only)
tracemalloc.start()
t0 = time.perf_counter()
bm25 = BM25().fit(ids, texts)
build_s = time.perf_counter() - t0
_, peak = tracemalloc.get_traced_memory()
tracemalloc.stop()
print(f"\nIndex build: {build_s:.3f}s | peak memory: {peak/1e6:.1f} MB | "
      f"vocab: {len(bm25.postings)} terms | avg doc length: {bm25.avgdl:.1f} tokens")

# 3. Demo queries (NOT the final suite)
demo = ["uncrewed lunar probe", "apollo", "virus",
        "languages with garbage collection", "Mariner 4"]
t0 = time.perf_counter()
for q in demo:
    bm25.search(q, 10)
print(f"Avg query latency: {(time.perf_counter() - t0) / len(demo) * 1000:.2f} ms")

for q in demo:
    print(f"\nQuery: {q}")
    for d, s in bm25.search(q, 5):
        print(f"   {s:7.3f}  {d}  {title[d]}")

# 4. Sanity check against the rank_bm25 library
ref = BM25Okapi([bm25.tokenize(t) for t in texts], k1=bm25.k1, b=bm25.b)
print("\nTop-10 overlap with rank_bm25 (expect high, not always 10/10):")
for q in demo:
    mine = {d for d, _ in bm25.search(q, 10)}
    sc = ref.get_scores(list(set(bm25.tokenize(q))))
    theirs = {ids[i] for i in sorted(range(len(ids)), key=lambda i: -sc[i])[:10]}
    print(f"   {q!r}: {len(mine & theirs)}/10")

# 5. Per-term explanation for one top hit
top = bm25.search("uncrewed lunar probe", 1)[0][0]
print("\nExplain top hit for 'uncrewed lunar probe':", title[top])
print("  ", {k: round(v, 3) for k, v in bm25.explain("uncrewed lunar probe", top).items()})