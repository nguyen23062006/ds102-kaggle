"""Đặc trưng v2: văn bản review và mức trùng với thông tin sản phẩm."""
import re
from collections import Counter
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from train_submit import normalize_text, ReviewStats


def prepare(frame):
    frame = frame.copy()
    for col in ['comment', 'product_name', 'description', 'category', 'categories']:
        frame[col] = frame[col].fillna('').astype(str).map(normalize_text)
    return frame


def tokens(text):
    return re.findall(r'(?u)\b\w+\b', text)


def ngrams(words, n):
    return set(zip(*(words[i:] for i in range(n))))


def dense_features(df):
    basic = ReviewStats().transform(df.comment).toarray()
    extra = []
    for row in df.itertuples(index=False):
        text = row.comment
        words = tokens(text)
        wl = [len(w) for w in words] or [0]
        wc = Counter(words)
        n = max(len(words), 1)
        feats = [np.log1p(max(wl)), np.mean(wl), np.std(wl),
                 sum(len(w) > 12 for w in words) / n,
                 max(wc.values(), default=0) / n,
                 sum(v * (v - 1) for v in wc.values()) / max(n * (n - 1), 1),
                 text.count('.') / max(len(text), 1),
                 len(re.findall(r'(.)\1{3,}', text)) / n,
                 np.log1p(len(text.split())),
                 float(row.rating) / 5,
                 np.log1p(max(float(row.num_sold), 0)),
                 np.log1p(max(float(row.num_review), 0))]
        for context in [row.product_name, row.description]:
            context_words = tokens(context)
            for size in [1, 2, 3, 4]:
                a, b = ngrams(words, size), ngrams(context_words, size)
                overlap = len(a & b)
                ratio = overlap / max(len(a), 1)
                feats += [ratio, ratio ** 2, ratio * np.log1p(len(words)), np.log1p(overlap)]
            feats += [float(len(text) > 10 and text in context), np.log1p(len(context))]
        extra.append(feats)
    return np.hstack([basic, np.asarray(extra, dtype=np.float32)])


class FeaturesV2:
    def __init__(self):
        self.word = TfidfVectorizer(ngram_range=(1, 3), min_df=2, max_features=140000,
                                   sublinear_tf=True, lowercase=False, dtype=np.float32,
                                   token_pattern=r'(?u)\b\w+\b')
        self.char = TfidfVectorizer(analyzer='char', ngram_range=(2, 5), min_df=3,
                                   max_features=180000, sublinear_tf=True,
                                   lowercase=False, dtype=np.float32)
        self.product = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=20000,
                                      sublinear_tf=True, lowercase=False, dtype=np.float32)
        self.category = OneHotEncoder(handle_unknown='ignore', dtype=np.float32)
        self.scale = StandardScaler()

    def fit_transform(self, df):
        text = sparse.hstack([self.word.fit_transform(df.comment), self.char.fit_transform(df.comment)], format='csr')
        meta = sparse.hstack([self.product.fit_transform(df.product_name) * .2,
                             self.category.fit_transform(df[['category', 'rating']].astype(str)) * .2,
                             sparse.csr_matrix(self.scale.fit_transform(dense_features(df))) * .15], format='csr')
        return text, meta

    def transform(self, df):
        text = sparse.hstack([self.word.transform(df.comment), self.char.transform(df.comment)], format='csr')
        meta = sparse.hstack([self.product.transform(df.product_name) * .2,
                             self.category.transform(df[['category', 'rating']].astype(str)) * .2,
                             sparse.csr_matrix(self.scale.transform(dense_features(df))) * .15], format='csr')
        return text, meta


def nb_ratios(X, y):
    """Log-count ratio one-vs-rest, fit chỉ từ phần train của fold."""
    ratios = []
    for c in range(4):
        pos = np.asarray(X[y == c].sum(axis=0)).ravel() + 1
        neg = np.asarray(X[y != c].sum(axis=0)).ravel() + 1
        ratios.append(np.log((pos / pos.sum()) / (neg / neg.sum())))
    return ratios
