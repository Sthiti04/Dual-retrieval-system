import json, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from config import DATA, RESULTS

ENGINES = ["random", "tfidf", "bm25", "dense"]
POOLED = ["tfidf", "bm25", "dense"]       # random was not pooled: its unjudged hits count as 0
K = 5
STRICT = "--strict" in sys.argv
REL_THR = 2 if STRICT else 1              # label needed to count as relevant for P@5 and AP
TAG = "strict" if STRICT else "lenient"

queries = json.load((DATA / "queries.json").open(encoding="utf-8"))
qr = pd.read_csv(DATA / "qrels.csv", encoding="utf-8-sig")
qr = qr[qr["label"].notna()].copy()
qr["label"] = qr["label"].astype(int)
assert qr["label"].isin([0, 1, 2]).all(), "labels must be 0, 1 or 2"
qr = qr.drop_duplicates(["qid", "doc_id"], keep="last")
qrels = {}
for r in qr.itertuples():
    qrels.setdefault(r.qid, {})[r.doc_id] = r.label

runs = {e: json.load((RESULTS / f"run_{e}.json").open()) for e in ENGINES}
judged = [q for q in queries if q["qid"] in qrels]
print(f"judged queries: {len(judged)} of {len(queries)} | relevant means label >= {REL_THR}")
for q in judged:
    qrel = qrels[q["qid"]]
    print(f'  {q["qid"]} [{q["category"]}] judged docs: {len(qrel)} | '
          f'relevant: {sum(v >= REL_THR for v in qrel.values())}')
for e in POOLED:
    miss = sum(d not in qrels[q["qid"]] for q in judged for d, _ in runs[e][q["qid"]][:K])
    if miss:
        print(f"WARNING: {e} has {miss} unjudged documents in its top-{K} (counted as non-relevant)")


def dcg(rels):
    return sum((2 ** r - 1) / np.log2(i + 2) for i, r in enumerate(rels))

def ndcg(ranked, qrel):
    ideal = dcg(sorted(qrel.values(), reverse=True)[:K])
    return dcg([qrel.get(d, 0) for d in ranked[:K]]) / ideal if ideal > 0 else np.nan

def p_at_k(ranked, qrel):
    return sum(qrel.get(d, 0) >= REL_THR for d in ranked[:K]) / K

def avg_prec(ranked, qrel):
    R = sum(v >= REL_THR for v in qrel.values())
    if R == 0:
        return np.nan
    hits, total = 0, 0.0
    for rank, d in enumerate(ranked, 1):
        if qrel.get(d, 0) >= REL_THR:
            hits += 1
            total += hits / rank
    return total / R


rows = []
for e in ENGINES:
    for q in judged:
        ranked = [d for d, _ in runs[e][q["qid"]]]
        qrel = qrels[q["qid"]]
        rows.append({"engine": e, "qid": q["qid"], "category": q["category"],
                     "ndcg@5": ndcg(ranked, qrel), "p@5": p_at_k(ranked, qrel),
                     "ap_approx": avg_prec(ranked, qrel)})
df = pd.DataFrame(rows)
df.to_csv(RESULTS / f"metrics_per_query_{TAG}.csv", index=False)

metrics = ["ndcg@5", "p@5", "ap_approx"]
rename = {"ap_approx": "MAP(approx)"}
overall = df.groupby("engine")[metrics].mean().loc[ENGINES].round(3)
bycat = df.groupby(["category", "engine"])[metrics].mean().round(3)
overall.rename(columns=rename).to_csv(RESULTS / f"metrics_overall_{TAG}.csv")
bycat.rename(columns=rename).to_csv(RESULTS / f"metrics_by_category_{TAG}.csv")

print(f"\nOVERALL (mean over {len(judged)} judged queries)")
print(overall.rename(columns=rename))
print("\nBY CATEGORY")
print(bycat.rename(columns=rename))
print("\nNDCG@5 PER QUERY")
print(df.pivot(index="qid", columns="engine", values="ndcg@5")[ENGINES].round(3))

piv = df.pivot_table(index="category", columns="engine", values="ndcg@5", aggfunc="mean")[ENGINES]
n = df[df["engine"] == "bm25"].groupby("category").size()
piv.index = [f"{c} (n={n[c]})" for c in piv.index]
ax = piv.plot(kind="bar", figsize=(8, 4.5), rot=0)
ax.set_ylabel("NDCG@5")
ax.set_ylim(0, 1)
ax.set_title(f"NDCG@5 by query category ({TAG} relevance)")
plt.tight_layout()
plt.savefig(RESULTS / f"ndcg5_by_category_{TAG}.png", dpi=200)

print("\nNOTE: P@5 and NDCG@5 are fully judged for tfidf, bm25 and dense (top-5 pooled). "
      "AP is computed over the full ranking with unjudged documents counted as non-relevant, "
      "so MAP is approximate. 'random' was not pooled.")