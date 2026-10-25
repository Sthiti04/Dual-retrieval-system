import csv, json, random, sys
from collections import defaultdict
from config import DATA, RESULTS, SEED
from data_utils import load_corpus

POOL_ENGINES = ["bm25", "tfidf", "dense"]
K = 5                                   # reduced plan: top-5 per engine
KEEP = set(sys.argv[1:])                # optional: only pool these qids

ids, titles, texts = load_corpus()
info = {i: (t, x) for i, t, x in zip(ids, titles, texts)}
queries = json.load((DATA / "queries.json").open(encoding="utf-8"))
runs = {n: json.load((RESULTS / f"run_{n}.json").open()) for n in POOL_ENGINES}

sheet, meta = [], []
for q in queries:
    if KEEP and q["qid"] not in KEEP:
        continue
    pool = defaultdict(dict)
    for n in POOL_ENGINES:
        for r, (d, _) in enumerate(runs[n][q["qid"]][:K], 1):
            pool[d][n] = r
    docs = sorted(pool)
    random.Random(f"{SEED}-{q['qid']}").shuffle(docs)         # hide engine ordering
    print(f"{q['qid']}: {len(docs)} docs to judge")
    for d in docs:
        sheet.append([q["qid"], q["text"], q.get("rule", ""), d, info[d][0], info[d][1], ""])
        meta.append([q["qid"], d] + [pool[d].get(n, "") for n in POOL_ENGINES])

with (DATA / "label_sheet.csv").open("w", newline="", encoding="utf-8-sig") as f:
    w = csv.writer(f)
    w.writerow(["qid", "query", "rule", "doc_id", "title", "text", "label"])
    w.writerows(sheet)
with (DATA / "pool_meta.csv").open("w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["qid", "doc_id"] + POOL_ENGINES)
    w.writerows(meta)
print("total judgments needed:", len(sheet))