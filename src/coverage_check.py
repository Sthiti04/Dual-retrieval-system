import json
import pandas as pd
from config import DATA

df = pd.DataFrame([json.loads(l) for l in (DATA / "corpus.jsonl").open(encoding="utf-8")])
low = df["text"].str.lower()

checks = {
    "negation": [r"\b(?:uncrewed|unmanned)\b", r"\bno (?:known )?(?:cure|vaccine)\b",
                 r"garbage collect", r"\b(?:failed|failure|lost contact)\b", r"\bnon-?metal"],
    "temporal": [r"\bfirst (?:crewed|manned)\b", r"\b19[5-7]\d\b", r"\bshortly after\b"],
    "identifiers": [r"\bmariner\b", r"\bvoyager\b", r"\bh1n1\b", r"\batomic number\b"],
    "descriptive": [r"\bransomware\b", r"\bencrypt", r"\bpandemic\b"],
    "multi-constraint": [r"\bfunctional\b", r"\bstatically typed\b"],
}

for group, patterns in checks.items():
    print(f"\n== {group} ==")
    for p in patterns:
        hits = df[low.str.contains(p, regex=True)]
        print(f"{p}: {len(hits)} docs -> {hits['category'].value_counts().to_dict()}")