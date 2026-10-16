from config import CATEGORIES
from wiki_utils import collect_titles

for label, (cats, cap) in CATEGORIES.items():
    print(f"\n{label} (target {cap})")
    total = 0
    for c in cats:
        n = len(collect_titles([c], cap * 3))
        total += n
        print(f"   {c}: {n} candidate articles")
    flag = "OK" if total >= cap else "LOW - fix this one"
    print(f"   -> total candidates: {total} [{flag}]")