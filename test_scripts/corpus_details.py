import sys, json
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np, pandas as pd
from config import DATA
from data_utils import load_corpus
from bm25_engine import Tokenizer

ids, titles, texts = load_corpus()
cat = {json.loads(l)["doc_id"]: json.loads(l)["category"]
       for l in (DATA / "corpus.jsonl").open(encoding="utf-8")}
tok = Tokenizer()
toks = [tok(t) for t in texts]
lens = np.array([len(t) for t in toks])
vocab = set().union(*map(set, toks))

print(f"documents: {len(ids)} | total tokens: {lens.sum():,} | vocabulary: {len(vocab):,}")
print(f"tokens per doc: mean {lens.mean():.1f} | median {np.median(lens):.0f} | "
      f"min {lens.min()} | max {lens.max()}")
df = pd.DataFrame({"category": [cat[i] for i in ids], "tokens": lens})
print(df.groupby("category")["tokens"].agg(["count", "mean", "min", "max"]).round(1))