import csv, json, random
from collections import defaultdict
import pandas as pd
from config import DATA, RESULTS, SEED
from data_utils import load_corpus

POOL_ENGINES = ["bm25", "tfidf", "dense", "bm25_stem_stop"]
K = 5

ids, titles, texts = load_corpus()
info = {i: (t, x) for i, t, x in zip(ids, titles, texts)}
queries = json.load((DATA / "queries.json").open(encoding="utf-8"))
runs = {n: json.load((RESULTS / f"run_{n}.json").open()) for n in POOL_ENGINES}
qr = pd.read_csv(DATA / "qrels.csv", encoding="utf-8-sig")
judged = set(zip(qr["qid"], qr["doc_id"]))


def pool_for(qid):
    pool = defaultdict(dict)
    for n in POOL_ENGINES:
        for r, (d, _) in enumerate(runs[n][qid][:K], 1):
            pool[d][n] = r
    return pool


pools = {q["qid"]: pool_for(q["qid"]) for q in queries}
sheet, meta = [], []
for q in queries:
    qid = q["qid"]
    skip = {d for (x, d) in judged if x == qid}
    src = q.get("variant_of")
    if src:    # documents judged, or about to be judged, under the clean source query
        skip |= {d for (x, d) in judged if x == src} | set(pools[src])
    todo = sorted(d for d in pools[qid] if d not in skip)
    random.Random(f"{SEED}-{qid}-new").shuffle(todo)
    print(f"{qid}: pool {len(pools[qid])} docs, {len(todo)} still to judge")
    for d in todo:
        sheet.append([qid, q["text"], q.get("rule", ""), d, info[d][0], info[d][1], ""])
    for d in sorted(pools[qid]):
        meta.append([qid, d] + [pools[qid][d].get(n, "") for n in POOL_ENGINES])

with (DATA / "label_sheet_new.csv").open("w", newline="", encoding="utf-8-sig") as f:
    w = csv.writer(f)
    w.writerow(["qid", "query", "rule", "doc_id", "title", "text", "label"])
    w.writerows(sheet)
with (DATA / "pool_meta_v2.csv").open("w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["qid", "doc_id"] + POOL_ENGINES)
    w.writerows(meta)
print("total new judgments needed:", len(sheet))