import json
from config import DATA, INDEX_WORDS


def load_corpus():
    """Returns (doc_ids, titles, texts). Texts are cut to INDEX_WORDS words,
    so BM25 and the dense engine index exactly the same content."""
    docs = [json.loads(l) for l in (DATA / "corpus.jsonl").open(encoding="utf-8")]
    ids = [d["doc_id"] for d in docs]
    titles = [d["title"] for d in docs]
    texts = [" ".join(d["text"].split()[:INDEX_WORDS]) for d in docs]
    return ids, titles, texts