import math, re
from collections import Counter, defaultdict
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

TOKEN_RE = re.compile(r"[a-z0-9]+")


class Tokenizer:
    """lowercase -> regex tokens -> (optional) stopword removal -> (optional) Porter stemming"""
    def __init__(self, remove_stopwords=False, stem=False):
        self.remove_stopwords = remove_stopwords
        self.stem = stem
        self._stemmer = PorterStemmer()
        self._cache = {}

    def __call__(self, text):
        toks = TOKEN_RE.findall(text.lower())
        if self.remove_stopwords:
            toks = [t for t in toks if t not in ENGLISH_STOP_WORDS]
        if self.stem:
            out = []
            for t in toks:
                s = self._cache.get(t)
                if s is None:
                    s = self._cache[t] = self._stemmer.stem(t)
                out.append(s)
            toks = out
        return toks


class BM25:
    """BM25 following Manning et al., Eq. 11.32:
       RSV(d,q) = sum over query terms t of
           log(N / df_t) * ((k1 + 1) * tf_td) / (k1 * ((1 - b) + b * L_d / L_ave) + tf_td)
       Each distinct query term is counted once (no query-term-frequency weighting)."""

    def __init__(self, k1=1.5, b=0.75, remove_stopwords=False, stem=False):
        self.k1, self.b = k1, b
        self.tokenize = Tokenizer(remove_stopwords, stem)

    def fit(self, doc_ids, texts):
        self.doc_ids = list(doc_ids)
        self.N = len(texts)
        self.doc_len = []
        self.postings = defaultdict(list)      # inverted index: term -> [(doc_index, tf)]
        for i, text in enumerate(texts):
            toks = self.tokenize(text)
            self.doc_len.append(len(toks))
            for term, f in Counter(toks).items():
                self.postings[term].append((i, f))
        self.avgdl = sum(self.doc_len) / self.N
        return self

    def _idf(self, df):
        return math.log(self.N / df)

    def _term_score(self, i, f, idf):
        norm = self.k1 * ((1 - self.b) + self.b * self.doc_len[i] / self.avgdl)
        return idf * ((self.k1 + 1) * f) / (f + norm)

    def scores(self, query):
        s = [0.0] * self.N
        for term in set(self.tokenize(query)):
            plist = self.postings.get(term)
            if not plist:
                continue
            idf = self._idf(len(plist))
            for i, f in plist:
                s[i] += self._term_score(i, f, idf)
        return s

    def search(self, query, k=None):
        s = self.scores(query)
        order = sorted(range(self.N), key=lambda i: (-s[i], i))   # ties broken by index
        order = order[:k] if k else order
        return [(self.doc_ids[i], s[i]) for i in order]

    def explain(self, query, doc_id):
        """Per-term contribution to one document's score (useful for failure analysis)."""
        i = self.doc_ids.index(doc_id)
        out = {}
        for term in set(self.tokenize(query)):
            plist = dict(self.postings.get(term, []))
            out[term] = (self._term_score(i, plist[i], self._idf(len(plist)))
                         if i in plist else 0.0)
        return out