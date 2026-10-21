import sys, time
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
from dense_engine import DenseEngine
from tfidf_engine import TfidfEngine
from data_utils import load_corpus

ids, titles, texts = load_corpus()
title = dict(zip(ids, titles))
eng = DenseEngine().index(ids, texts)
queries = ["uncrewed lunar probe", "apollo", "virus",
           "languages with garbage collection", "Mariner 4"]

# 1. Fair latency: warm up first, then average over repeated runs
for _ in range(3):
    eng.search(queries[0], 10)
n, t_enc, t_sim = 20, 0.0, 0.0
for _ in range(n):
    for q in queries:
        t0 = time.perf_counter(); qv = eng.encode_query(q); t1 = time.perf_counter()
        np.argsort(-(eng.emb @ qv))[:10]; t2 = time.perf_counter()
        t_enc += t1 - t0; t_sim += t2 - t1
m = n * len(queries)
print(f"device={eng.model.device} | query encoding: {t_enc/m*1000:.1f} ms | "
      f"similarity + ranking: {t_sim/m*1000:.3f} ms")

# 2. Cosine between query vectors: negation pair vs. reference pairs
def cos(a, b):
    return float(eng.encode_query(a) @ eng.encode_query(b))

sets = {
  "garbage collection": [
    ("negated",    "languages with garbage collection", "languages without garbage collection"),
    ("paraphrase", "languages with garbage collection", "programming languages that have automatic memory management"),
    ("related",    "languages with garbage collection", "languages with static typing"),
    ("unrelated",  "languages with garbage collection", "chemical elements that are metals")],
  "metals": [
    ("negated",    "chemical elements that are metals", "chemical elements that are not metals"),
    ("paraphrase", "chemical elements that are metals", "metallic elements of the periodic table"),
    ("related",    "chemical elements that are metals", "chemical elements that are gases"),
    ("unrelated",  "chemical elements that are metals", "languages with garbage collection")],
}
for name, rows in sets.items():
    print(f"\nCosine to the base query ({name}):")
    for label, a, b in rows:
        print(f"   {label:10s} {cos(a, b):.3f}   <- {b!r}")

# 3. TF-IDF baseline smoke test
t0 = time.perf_counter()
tf = TfidfEngine().fit(ids, texts)
print(f"\nTF-IDF build: {time.perf_counter() - t0:.3f}s | vocab: {len(tf.vec.vocabulary_)}")
for q in queries:
    print(f"\nTF-IDF | {q}")
    for d, s in tf.search(q, 5):
        print(f"   {s:6.3f}  {title[d]}")