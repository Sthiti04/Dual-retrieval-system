import json, re, sys
from collections import Counter
from config import DATA, INDEX_WORDS

pattern = re.compile(sys.argv[1], re.I)
limit = int(sys.argv[2]) if len(sys.argv) > 2 else 15
docs = [json.loads(l) for l in (DATA / "corpus.jsonl").open(encoding="utf-8")]
hits = []
for d in docs:
    text = " ".join(d["text"].split()[:INDEX_WORDS])
    m = pattern.search(text)
    if m:
        hits.append((d, text, m))
print(f"{len(hits)} docs match /{sys.argv[1]}/")
print(dict(Counter(d["category"] for d, _, _ in hits)))
for d, text, m in hits[:limit]:
    s = max(0, m.start() - 80)
    print(f'{d["doc_id"]} | {d["category"]} | {d["title"]} | ...{text[s:m.end() + 80]}...')