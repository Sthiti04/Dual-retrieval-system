import json
import pandas as pd
from config import DATA

queries = json.load((DATA / "queries.json").open(encoding="utf-8"))
qr = pd.read_csv(DATA / "qrels.csv", encoding="utf-8-sig")
parts = []
for q in queries:
    src = q.get("variant_of")
    if not src:
        continue
    have = set(qr.loc[qr["qid"] == q["qid"], "doc_id"])
    rows = qr[(qr["qid"] == src) & (~qr["doc_id"].isin(have))].copy()
    rows["qid"], rows["query"], rows["rule"] = q["qid"], q["text"], q["rule"]
    print(q["qid"], "copied", len(rows), "judgments from", src)
    parts.append(rows)
out = pd.concat([qr] + parts, ignore_index=True)
out.to_csv(DATA / "qrels.csv", index=False, encoding="utf-8-sig")
print("qrels rows:", len(qr), "->", len(out))