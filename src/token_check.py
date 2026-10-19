import json, sys
import numpy as np
from sentence_transformers import SentenceTransformer
from config import DATA

name = sys.argv[1] if len(sys.argv) > 1 else "BAAI/bge-small-en-v1.5"
model = SentenceTransformer(name)
limit = model.max_seq_length - 2          # 2 slots are used by [CLS] and [SEP]
print(name, "| usable tokens:", limit)

docs = [json.loads(l) for l in (DATA / "corpus.jsonl").open(encoding="utf-8")]
for w in (150, 200, 250, 300):
    toks = [len(model.tokenizer.tokenize(" ".join(d["text"].split()[:w]))) for d in docs]
    over = sum(t > limit for t in toks)
    print(f"first {w} words: mean {np.mean(toks):.0f} tokens | "
          f"{over}/{len(docs)} docs over the limit")