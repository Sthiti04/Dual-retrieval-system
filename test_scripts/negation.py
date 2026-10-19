import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from bm25_engine import BM25
from data_utils import load_corpus

ids, titles, texts = load_corpus()
title = dict(zip(ids, titles))
bm25 = BM25().fit(ids, texts)

pairs = [("languages with garbage collection", "languages without garbage collection"),
         ("chemical elements that are metals", "chemical elements that are not metals")]
for a, b in pairs:
    ra = [d for d, _ in bm25.search(a, 10)]
    rb = [d for d, _ in bm25.search(b, 10)]
    print(f"\n{a!r} vs {b!r}: top-10 overlap {len(set(ra) & set(rb))}/10")
    for d in rb[:5]:
        print("   ", title[d])