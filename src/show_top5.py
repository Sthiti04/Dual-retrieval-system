import sys
import pandas as pd
from config import DATA
from data_utils import load_corpus

ids, titles, _ = load_corpus()
title = dict(zip(ids, titles))
m = pd.read_csv(DATA / "pool_meta_v2.csv")
q = pd.read_csv(DATA / "qrels.csv", encoding="utf-8-sig")[["qid", "doc_id", "label"]]
q = q.drop_duplicates(["qid", "doc_id"], keep="last")
d = m.merge(q, on=["qid", "doc_id"], how="left")
for qid in sys.argv[1:]:
    print("\n==", qid)
    g = d[d.qid == qid]
    for e in ["bm25", "tfidf", "dense", "bm25_stem_stop"]:
        t = g[g[e].notna()].sort_values(e)
        print(f"  {e:15s}|", " ; ".join(
            f"{title[r.doc_id][:26]}({'?' if pd.isna(r.label) else int(r.label)})" for r in t.itertuples()))