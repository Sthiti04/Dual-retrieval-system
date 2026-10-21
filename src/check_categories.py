import json
from config import DATA, CATEGORIES
from wiki_utils import collect_titles

OUT = DATA / "corpus.jsonl"
existing = [json.loads(l) for l in OUT.open(encoding="utf-8")] if OUT.exists() else []
have = {d["title"] for d in existing}
per_label = {}
for d in existing:
    per_label[d["category"]] = per_label.get(d["category"], 0) + 1

for label, (cats, cap) in CATEGORIES.items():
    need = cap - per_label.get(label, 0)
    if need <= 0:
        print(f"\n{label}: complete ({per_label.get(label, 0)}/{cap}), skipped")
        continue
    print(f"\n{label}: need {need} more")
    pool = []
    for c in cats:
        titles = collect_titles([c], cap * 3)
        new = [t for t in titles if t not in have]
        print(f"   {c}: {len(titles)} candidates, {len(new)} not yet in corpus")
        pool += new
    total = len(set(pool))
    flag = "OK" if total >= need * 1.2 else "LOW - fix this one"
    print(f"   -> net-new candidates: {total} [{flag}]")