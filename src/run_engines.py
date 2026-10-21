import json, random, sys, time
import psutil
from config import DATA, RESULTS, SEED
from data_utils import load_corpus
from bm25_engine import BM25
from tfidf_engine import TfidfEngine
from dense_engine import DenseEngine

DEPTH = 200                                  # ranked results stored per query
proc = psutil.Process()
rss_mb = lambda: proc.memory_info().rss / 1e6


class RandomEngine:
    """Floor baseline: a seeded random ranking for each query."""
    def fit(self, doc_ids, texts):
        self.doc_ids = list(doc_ids)
        return self

    def search(self, query, k=None):
        order = self.doc_ids[:]
        random.Random(f"{SEED}-{query}").shuffle(order)
        return [(d, 0.0) for d in (order[:k] if k else order)]


# dense is last so earlier memory deltas are not inflated by PyTorch allocations
BUILDERS = {
    "random":         lambda: RandomEngine(),
    "tfidf":          lambda: TfidfEngine(),
    "bm25":           lambda: BM25(),
    "bm25_stem_stop": lambda: BM25(remove_stopwords=True, stem=True),
    "dense":          lambda: DenseEngine(),
}


def main():
    retime = "--retime" in sys.argv
    ids, titles, texts = load_corpus()
    queries = json.load((DATA / "queries.json").open(encoding="utf-8"))
    RESULTS.mkdir(exist_ok=True)
    tpath = RESULTS / "timing.json"
    prev = json.load(tpath.open()) if tpath.exists() else {}
    timing = {}

    for name, make in BUILDERS.items():
        m0, t0 = rss_mb(), time.perf_counter()
        eng = make()
        if name == "dense":
            eng.index(ids, texts, use_cache=not retime)
            build_s = (eng.load_s + eng.encode_s) if eng.encode_s else prev.get(name, {}).get("build_s")
        else:
            eng.fit(ids, texts)
            build_s = time.perf_counter() - t0
        mem_mb = rss_mb() - m0

        for q in queries[:3]:                              # warm-up
            eng.search(q["text"], DEPTH)
        t0 = time.perf_counter()
        run = {q["qid"]: eng.search(q["text"], DEPTH) for q in queries}
        lat_ms = (time.perf_counter() - t0) / len(queries) * 1000

        with (RESULTS / f"run_{name}.json").open("w") as f:
            json.dump({k: [[d, round(s, 6)] for d, s in v] for k, v in run.items()}, f)
        timing[name] = {"build_s": build_s, "latency_ms": lat_ms, "rss_delta_mb": mem_mb}
        b = "n/a (cached; use --retime)" if build_s is None else f"{build_s:.2f}s"
        print(f"{name:15s} build {b} | {lat_ms:7.2f} ms/query | +{mem_mb:6.1f} MB RSS")

    with tpath.open("w") as f:
        json.dump(timing, f, indent=2)


if __name__ == "__main__":
    main()