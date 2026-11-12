import sys
import pandas as pd
from config import DATA

old = pd.read_csv(DATA / "qrels.csv", encoding="utf-8-sig")
new = pd.read_csv(sys.argv[1], encoding="utf-8-sig")
blank = int(new["label"].isna().sum())
if blank:
    sys.exit(f"{blank} rows have no label. Fill them before merging.")
assert new["label"].isin([0, 1, 2]).all(), "labels must be 0, 1 or 2"
both = pd.concat([old, new], ignore_index=True).drop_duplicates(["qid", "doc_id"], keep="last")
both.to_csv(DATA / "qrels.csv", index=False, encoding="utf-8-sig")
print(len(old), "+", len(new), "->", len(both))