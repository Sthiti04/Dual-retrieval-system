import json, sys
import pandas as pd
from config import DATA
from data_utils import load_corpus
from bm25_engine import BM25

ids, titles, texts = load_corpus()
qs = {q["qid"]: q["text"] for q in json.load((DATA / "queries.json").open(encoding="utf-8"))}
qr = pd.read_csv(DATA / "qrels.csv", encoding="utf-8-sig").drop_duplicates(["qid", "doc_id"], keep="last")
variants = {"plain": BM25(), "stem only": BM25(stem=True),
            "stopwords only": BM25(remove_stopwords=True),
            "stem + stopwords": BM25(remove_stopwords=True, stem=True)}
for v in variants.values():
    v.fit(ids, texts)

for qid in sys.argv[1:]:
    rel = set(qr[(qr.qid == qid) & (qr.label >= 1)].doc_id)
    judged = set(qr[qr.qid == qid].doc_id)
    print(f"\n{qid} {qs[qid]} | judged relevant docs: {len(rel)}")
    for name, eng in variants.items():
        top5 = [d for d, _ in eng.search(qs[qid], 5)]
        print(f"  {name:17s} relevant in top 5: {sum(d in rel for d in top5)} | unjudged in top 5: {sum(d not in judged for d in top5)}")