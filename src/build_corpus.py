import json, random, re, time
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
    done = set()
    if OUT.exists():                  # resume support if the script is interrupted
        done = {json.loads(l)["title"] for l in OUT.open(encoding="utf-8")}
    n = len(done)

    with OUT.open("a", encoding="utf-8") as f:
        for label, (cats, cap) in CATEGORIES.items():
            titles = collect_titles(cats, cap * 3)
            random.Random(SEED).shuffle(titles)       # reproducible sample
            kept = 0
            for title in tqdm(titles, desc=label):
                if kept >= cap:
                    break
                if title in done:
                    continue
                page = wiki.page(title)
                if not page.exists():
                    continue
                words = clean(page.text).split()
                if len(words) < MIN_WORDS:
                    continue
                if "may refer to" in " ".join(words[:40]).lower():
                    continue                           # skip disambiguation pages
                n += 1
                kept += 1
                done.add(title)
                f.write(json.dumps({
                    "doc_id": f"d{n:05d}",
                    "title": title,
                    "category": label,
                    "text": " ".join(words[:MAX_WORDS]),
                }, ensure_ascii=False) + "\n")
                time.sleep(0.05)                       # be polite to the API

    print(f"Documents: {n} | build time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()