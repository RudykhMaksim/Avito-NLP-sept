"""BM25 по полям, поиск топ-K по строкам разреженной матрицы, попарные значения и метрика Recall@K."""
import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer


class BM25Field:
    """BM25 по одному полю (документы - строки лемм через пробел).

    Вес слова в документе считается заранее, оценка запроса - сумма весов его слов.
    Запрос бинарный: повтор слова вес не увеличивает.
    """

    def __init__(self, docs, k1=1.2, b=0.75, min_df=1):
        self.cv = CountVectorizer(tokenizer=str.split, lowercase=False, token_pattern=None, dtype=np.float32, min_df=min_df)
        tf = self.cv.fit_transform(docs).tocsr()
        n, v = tf.shape
        df = np.bincount(tf.indices, minlength=v).astype(np.float32)
        self.idf = np.log(1.0 + (n - df + 0.5) / (df + 0.5)).astype(np.float32)
        doc_len = np.asarray(tf.sum(1)).ravel()
        avg_len = max(doc_len.mean(), 1e-6)
        coo = tf.tocoo()
        denom = coo.data + k1 * (1 - b + b * doc_len[coo.row] / avg_len)
        w = self.idf[coo.col] * coo.data * (k1 + 1) / denom
        self.W = sp.csr_matrix((w.astype(np.float32), (coo.row, coo.col)), shape=(n, v))   # документы x слова
        self.WT = self.W.T.tocsr()                                                       # слова x документы, для умножения Q @ W.T
        self.N, self.V = n, v

    def qmat(self, query_docs):
        """Бинарная матрица запросов в словаре поля (незнакомые слова отбрасываются)."""
        q = self.cv.transform(query_docs).tocsr()
        q.data[:] = 1.0
        return q

    def score(self, q):
        """Разреженная матрица оценок (запросы x документы), ненулевые только у пар с общим словом."""
        return (q @ self.WT).tocsr()


def topk_rows(s, k):
    """Для каждой строки матрицы индексы и оценки топ-k по убыванию."""
    idx_out, sc_out = [], []
    ip, ind, dat = s.indptr, s.indices, s.data
    for r in range(s.shape[0]):
        a, b = ip[r], ip[r + 1]
        d, ix = dat[a:b], ind[a:b]
        if b - a > k:
            sel = np.argpartition(-d, k)[:k]
            d, ix = d[sel], ix[sel]
        order = np.argsort(-d, kind='stable')
        idx_out.append(ix[order])
        sc_out.append(d[order])
    return idx_out, sc_out


def lookup(s, r, c):
    """Значения s[r, c] для массивов индексов, нули там, где элемента нет.

    Ключ строка * N + столбец и бинарный поиск: быстро для миллионов пар.
    """
    s.sort_indices()
    nq, n = s.shape
    keys = np.repeat(np.arange(nq, dtype=np.int64), np.diff(s.indptr)) * n + s.indices
    q = r.astype(np.int64) * n + c
    if len(keys) == 0:
        return np.zeros(len(q), dtype=np.float32)
    pos = np.searchsorted(keys, q)
    pos[pos >= len(keys)] = 0
    return np.where(keys[pos] == q, s.data[pos], 0).astype(np.float32)


def pair_dot(qm, w, rows, cols):
    """Попарное скалярное произведение строк qm[rows] и w[cols]."""
    return np.asarray(qm[rows].multiply(w[cols]).sum(1)).ravel().astype(np.float32)


def recall_at(cands, positives, ks=(50,)):
    """Средний Recall@k. cands - списки индексов по убыванию релевантности, positives - множества положительных."""
    res = {k: [] for k in ks}
    for c, p in zip(cands, positives):
        for k in ks:
            res[k].append(len(p.intersection(c[:k].tolist())) / len(p))
    return {k: float(np.mean(v)) for k, v in res.items()}
