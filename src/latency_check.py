import json, time
import numpy as np
from config import DATA
from data_utils import load_corpus
from bm25_engine import BM25
from tfidf_engine import TfidfEngine
from dense_engine import DenseEngine

ids, titles, texts = load_corpus()
queries = [q["text"] for q in json.load((DATA / "queries.json").open(encoding="utf-8"))]
engines = {"tfidf": TfidfEngine().fit(ids, texts),
           "bm25": BM25().fit(ids, texts),
           "bm25_stem_stop": BM25(remove_stopwords=True, stem=True).fit(ids, texts),
           "dense": DenseEngine().index(ids, texts)}
REPEATS = 5
for name, eng in engines.items():
    for q in queries[:3]:
        eng.search(q, 10)                      # warm-up, not timed
    times = []
    for _ in range(REPEATS):
        for q in queries:
            t0 = time.perf_counter()
            eng.search(q, 10)
            times.append((time.perf_counter() - t0) * 1000)
    print(f"{name:15s} mean {np.mean(times):7.2f} ms | median {np.median(times):7.2f} ms | "
          f"p95 {np.percentile(times, 95):7.2f} ms")