import time
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer
from config import DATA, INDEX_WORDS

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"


class DenseEngine:
    """Bi-encoder retrieval: embed every document once, rank by cosine similarity.
    Embeddings are L2-normalized, so the dot product equals cosine similarity exactly."""

    def __init__(self, model_name=DEFAULT_MODEL, query_prefix="", doc_prefix=""):
        self.model_name = model_name
        self.query_prefix = query_prefix      # e.g. "query: " for E5-style models
        self.doc_prefix = doc_prefix          # e.g. "passage: " for E5-style models
        t0 = time.perf_counter()
        self.model = SentenceTransformer(model_name)
        self.load_s = time.perf_counter() - t0
        self.encode_s = 0.0                   # stays 0 if embeddings come from the cache

    def _cache_file(self):
        d = DATA / "embeddings"
        d.mkdir(parents=True, exist_ok=True)
        return d / f"{self.model_name.replace('/', '_')}_{INDEX_WORDS}w.npy"

    def index(self, doc_ids, texts, use_cache=True):
        self.doc_ids = list(doc_ids)
        f = self._cache_file()
        if use_cache and f.exists():
            self.emb = np.load(f)
        else:
            t0 = time.perf_counter()
            self.emb = self.model.encode(
                [self.doc_prefix + t for t in texts],
                batch_size=32, normalize_embeddings=True, show_progress_bar=True)
            self.encode_s = time.perf_counter() - t0
            np.save(f, self.emb)
        assert len(self.emb) == len(self.doc_ids), "stale cache: delete data/embeddings/"
        return self

    def encode_query(self, query):
        return self.model.encode([self.query_prefix + query], normalize_embeddings=True)[0]

    def scores(self, query):
        return self.emb @ self.encode_query(query)

    def search(self, query, k=None):
        sims = self.scores(query)
        order = np.argsort(-sims, kind="stable")     # stable sort: ties keep corpus order
        order = order[:k] if k else order
        return [(self.doc_ids[i], float(sims[i])) for i in order]