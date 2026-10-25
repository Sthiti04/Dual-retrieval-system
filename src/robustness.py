import json, random
import pandas as pd
from config import DATA, RESULTS, SEED
from data_utils import load_corpus
from bm25_engine import BM25
from tfidf_engine import TfidfEngine
from dense_engine import DenseEngine

K = 10
RATES = [0.05, 0.10, 0.20]     # chance that each letter of the query gets an error
TRIALS = 5

ids, titles, texts = load_corpus()
queries = json.load((DATA / "queries.json").open(encoding="utf-8"))
engines = {"tfidf": TfidfEngine().fit(ids, texts),
           "bm25": BM25().fit(ids, texts),
           "dense": DenseEngine().index(ids, texts)}

def top(eng, q):
    return [d for d, _ in eng.search(q, K)]

def overlap(a, b):
    return len(set(a) & set(b)) / K

def corrupt(text, rate, rng):
    out = []
    for ch in text:
        if ch.isalpha() and rng.random() < rate:
            op = rng.choice(["delete", "duplicate", "substitute"])
            if op == "duplicate":
                out.append(ch * 2)
            elif op == "substitute":
                out.append(rng.choice("abcdefghijklmnopqrstuvwxyz"))
        else:
            out.append(ch)
    return "".join(out) or text

# 1. Typo stability: overlap between top-10 for the clean query and a corrupted copy
rows = []
for rate in RATES:
    for q in queries:
        for t in range(TRIALS):
            noisy = corrupt(q["text"], rate, random.Random(f"{SEED}-{q['qid']}-{rate}-{t}"))
            for name, eng in engines.items():
                rows.append({"rate": rate, "qid": q["qid"], "category": q["category"],
                             "engine": name,
                             "overlap@10": overlap(top(eng, q["text"]), top(eng, noisy))})
typo = pd.DataFrame(rows)
typo.to_csv(RESULTS / "robustness_typos_raw.csv", index=False)
tab = typo.groupby(["rate", "engine"])["overlap@10"].mean().unstack().round(3)
tab.to_csv(RESULTS / "robustness_typos.csv")
print("TYPO STABILITY: mean top-10 overlap with the clean query's results")
print(tab)
print("\nExample corrupted queries at rate 0.20:")
for q in queries[:4]:
    print("  ", q["text"], "->", corrupt(q["text"], 0.20, random.Random(f"{SEED}-demo-{q['qid']}")))

# 2. Negation sensitivity: affirmative vs. negated version of the same query
pairs = [("chemical elements that are metals", "chemical elements that are not metals"),
         ("programming languages with garbage collection", "programming languages without garbage collection"),
         ("diseases that spread from person to person", "diseases that do not spread from person to person"),
         ("spacecraft that carried a crew", "spacecraft that carried no crew")]
dense = engines["dense"]
neg = []
for a, b in pairs:
    row = {"negated_query": b}
    for name, eng in engines.items():
        row[f"{name}_overlap@10"] = overlap(top(eng, a), top(eng, b))
    row["dense_query_cosine"] = round(float(dense.encode_query(a) @ dense.encode_query(b)), 3)
    neg.append(row)
neg = pd.DataFrame(neg)
neg.to_csv(RESULTS / "robustness_negation.csv", index=False)
print("\nNEGATION: overlap of top-10 between a query and its negation")
print(neg.to_string(index=False))

# 3. How much do the engines agree?
agree = pd.DataFrame([{"qid": q["qid"], "category": q["category"],
                       "bm25_vs_dense@10": overlap(top(engines["bm25"], q["text"]), top(engines["dense"], q["text"])),
                       "bm25_vs_tfidf@10": overlap(top(engines["bm25"], q["text"]), top(engines["tfidf"], q["text"]))}
                      for q in queries])
agree.to_csv(RESULTS / "engine_agreement.csv", index=False)
print("\nENGINE AGREEMENT (top-10 overlap)")
print(agree.to_string(index=False))