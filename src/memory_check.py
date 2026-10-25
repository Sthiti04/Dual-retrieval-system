import resource, sys, time
from data_utils import load_corpus

name = sys.argv[1]
ids, titles, texts = load_corpus()
t0 = time.perf_counter()
if name == "bm25":
    from bm25_engine import BM25
    BM25().fit(ids, texts)
elif name == "tfidf":
    from tfidf_engine import TfidfEngine
    TfidfEngine().fit(ids, texts)
elif name == "dense":
    from dense_engine import DenseEngine
    DenseEngine().index(ids, texts, use_cache=False)    # re-encodes, so encoding memory is included
elif name != "baseline":
    sys.exit("usage: memory_check.py baseline|bm25|tfidf|dense")
rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
mb = rss / 1e6 if sys.platform == "darwin" else rss / 1e3
print(f"{name}: peak RSS {mb:.0f} MB | build {time.perf_counter() - t0:.1f}s")