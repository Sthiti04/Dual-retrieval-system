import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from bm25_engine import Tokenizer


class TfidfEngine:
    """Baseline: sublinear tf-idf vectors ranked by cosine similarity (Manning Ch. 6 style).
    Uses BM25's tokenizer so only the weighting differs. Note: sklearn's idf is the smoothed
    ln((1+N)/(1+df)) + 1, which differs slightly from Manning's log(N/df)."""

    def __init__(self, remove_stopwords=False, stem=False):
        self.vec = TfidfVectorizer(analyzer=Tokenizer(remove_stopwords, stem), sublinear_tf=True)

    def fit(self, doc_ids, texts):
        self.doc_ids = list(doc_ids)
        self.X = self.vec.fit_transform(texts)     # rows are L2-normalized
        return self

    def scores(self, query):
        q = self.vec.transform([query])
        return (self.X @ q.T).toarray().ravel()

    def search(self, query, k=None):
        s = self.scores(query)
        order = np.argsort(-s, kind="stable")
        order = order[:k] if k else order
        return [(self.doc_ids[i], float(s[i])) for i in order]