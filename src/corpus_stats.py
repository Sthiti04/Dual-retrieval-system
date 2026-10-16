import json
import pandas as pd
from config import DATA

df = pd.DataFrame([json.loads(l) for l in (DATA / "corpus.jsonl").open(encoding="utf-8")])
df["n_words"] = df["text"].str.split().str.len()

print("Total documents:", len(df))
print("File size (MB):", round((DATA / "corpus.jsonl").stat().st_size / 1e6, 2))
print("Duplicate titles:", df["title"].duplicated().sum())
print("\nPer category:")
print(df.groupby("category")["n_words"].agg(["count", "mean", "min", "max"]).round(1))

print("\n--- Homonym check: how many docs mention each word, split by category ---")
for word in ["mercury", "titan", "apollo", "atlas", "virus", "worm", "trojan", "java"]:
    hits = df[df["text"].str.lower().str.contains(rf"\b{word}\b")]
    print(f"\n{word}: {len(hits)} docs")
    if len(hits):
        print(hits["category"].value_counts().to_string())