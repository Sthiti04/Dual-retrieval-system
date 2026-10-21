import sys, time, resource
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
from dense_engine import DenseEngine
from data_utils import load_corpus

ids, titles, texts = load_corpus()
title = dict(zip(ids, titles))

eng = DenseEngine().index(ids, texts)
rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
rss_mb = rss / 1e6 if sys.platform == "darwin" else rss / 1e3   # macOS reports bytes, Linux KB

print(f"Model: {eng.model_name} | device: {eng.model.device}")
print(f"Model load: {eng.load_s:.2f}s | encode corpus: {eng.encode_s:.1f}s "
      f"(0 means loaded from cache)")
print(f"Embeddings: {eng.emb.shape} | size: {eng.emb.nbytes/1e6:.2f} MB | "
      f"peak process memory (RSS): {rss_mb:.0f} MB")

demo = ["uncrewed lunar probe", "apollo", "virus",
        "languages with garbage collection", "Mariner 4"]
t0 = time.perf_counter()
for q in demo:
    eng.search(q, 10)
print(f"Avg query latency: {(time.perf_counter() - t0) / len(demo) * 1000:.1f} ms")

for q in demo:
    print(f"\nQuery: {q}")
    for d, s in eng.search(q, 5):
        print(f"   {s:6.3f}  {d}  {title[d]}")

# Negation probe: same pairs as the BM25 probe
pairs = [("languages with garbage collection", "languages without garbage collection"),
         ("chemical elements that are metals", "chemical elements that are not metals")]
for a, b in pairs:
    ra = [d for d, _ in eng.search(a, 10)]
    rb = eng.search(b, 10)
    cos_q = float(eng.encode_query(a) @ eng.encode_query(b))
    print(f"\n{a!r} vs {b!r}")
    print(f"   top-10 overlap: {len(set(ra) & {d for d, _ in rb})}/10 | "
          f"cosine between the two query vectors: {cos_q:.3f}")
    for d, s in rb[:5]:
        print(f"   {s:6.3f}  {title[d]}")