import json, random, re, time
from collections import Counter
from tqdm import tqdm
from config import DATA, CATEGORIES, MIN_WORDS, MAX_WORDS, SEED
from wiki_utils import wiki, collect_titles

OUT = DATA / "corpus.jsonl"
DATA.mkdir(exist_ok=True)


def clean(text):
    text = re.sub(r"={2,}[^=\n]+={2,}", " ", text)   # leftover == headings ==
    return " ".join(text.split())


def main():
    t0 = time.time()
    existing = [json.loads(l) for l in OUT.open(encoding="utf-8")] if OUT.exists() else []
    done_titles = {d["title"] for d in existing}
    per_label = Counter(d["category"] for d in existing)
    next_id = max((int(d["doc_id"][1:]) for d in existing), default=0)
    added = 0

    with OUT.open("a", encoding="utf-8") as f:
        for label, (cats, cap) in CATEGORIES.items():
            need = cap - per_label[label]          # only fill the gap for this label
            if need <= 0:
                print(f"{label}: already {per_label[label]}/{cap}, skipping")
                continue
            titles = collect_titles(cats, cap * 3)
            random.Random(SEED).shuffle(titles)
            kept = 0
            for title in tqdm(titles, desc=label):
                if kept >= need:
                    break
                if title in done_titles:
                    continue
                page = wiki.page(title)
                if not page.exists():
                    continue
                words = clean(page.text).split()
                if len(words) < MIN_WORDS:
                    continue
                if "may refer to" in " ".join(words[:40]).lower():
                    continue                        # skip disambiguation pages
                next_id += 1
                kept += 1
                added += 1
                done_titles.add(title)
                f.write(json.dumps({
                    "doc_id": f"d{next_id:05d}",
                    "title": title,
                    "category": label,
                    "text": " ".join(words[:MAX_WORDS]),
                }, ensure_ascii=False) + "\n")
                f.flush()                           # an interrupted run keeps what it wrote
                time.sleep(0.05)
            if kept < need:
                print(f"   [WARN] {label}: only added {kept} of {need}")

    print(f"Added {added} docs | total now {len(done_titles)} | {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()