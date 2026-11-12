import json
from config import DATA
from data_utils import load_corpus
from bm25_engine import BM25

ids, titles, texts = load_corpus()
title = dict(zip(ids, titles))
qs = {q["qid"]: q["text"] for q in json.load((DATA / "queries.json").open(encoding="utf-8"))}
engines = {"plain": BM25().fit(ids, texts),
           "stem+stop": BM25(remove_stopwords=True, stem=True).fit(ids, texts)}

def probe(qid, doc):
    q = qs[qid]
    print(f"\n{qid} '{q}' | {doc} {title[doc]}")
    for name, eng in engines.items():
        rank = [d for d, _ in eng.search(q)].index(doc) + 1
        parts = {k: round(v, 2) for k, v in eng.explain(q, doc).items() if v > 0}
        print(f"  {name:10s} rank {rank:4d} | {parts}")

probe("q02", "d01256")   # Pontiac fever, states no person-to-person spread
probe("q02", "d01164")   # Legionella
probe("q03", "d00004")   # Mercury-Atlas 1, says "uncrewed"
probe("q06", "d01369")   # 1875-1876 scarlet fever epidemic