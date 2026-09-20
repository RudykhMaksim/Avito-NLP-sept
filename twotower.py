"""Двухбашенная нейросеть запрос-объявление, обученная на кликах.

BM25 ищет только по совпадению слов, а сеть, обученная на кликах, знает синонимы и предсказывает микрокатегорию по запросу.
Вход: TF-IDF признаки слов, биграмм и символьных n-грамм (устойчиво к опечаткам).
Башни: EmbeddingBag и небольшой резидуальный MLP, на выходе нормированный вектор.
Потери: InfoNCE по батчу (без ложных негативов с тем же запросом или объявлением) плюс классификация микрокатегории.
"""
import time

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


def uni_bi(s):
    """Слова и пары соседних слов строки."""
    w = s.split()
    return w + [a + '_' + b for a, b in zip(w, w[1:])]


def uni(s):
    return s.split()


class Featurizer:
    """Разреженные TF-IDF признаки запросов и объявлений. Словари строятся только по обучающим данным (base)."""

    def __init__(self, q_lem, q_raw, t_lem, t_raw, p_lem):
        mk = lambda **kw: CountVectorizer(lowercase=False, token_pattern=None, dtype=np.float32, **kw)
        self.vq_w = mk(analyzer=uni_bi, min_df=2).fit(q_lem)
        self.vq_c = CountVectorizer(analyzer='char_wb', ngram_range=(3, 5), min_df=3, max_features=200000, dtype=np.float32).fit(q_raw)
        self.vt_w = mk(analyzer=uni_bi, min_df=3).fit(t_lem)
        self.vt_c = CountVectorizer(analyzer='char_wb', ngram_range=(3, 4), min_df=5, max_features=200000, dtype=np.float32).fit(t_raw)
        self.vp_w = mk(analyzer=uni, min_df=5).fit(p_lem)
        self.tf = {}
        for name, v, texts in [('q_w', self.vq_w, q_lem), ('q_c', self.vq_c, q_raw), ('t_w', self.vt_w, t_lem),
                               ('t_c', self.vt_c, t_raw), ('p_w', self.vp_w, p_lem)]:
            self.tf[name] = TfidfTransformer(sublinear_tf=True).fit(v.transform(texts))

    def _block(self, name, vec, texts):
        return self.tf[name].transform(vec.transform(texts)).astype(np.float32)      # tf-idf и L2-нормировка блока

    def queries(self, lem, raw):
        return sp.hstack([self._block('q_w', self.vq_w, lem), self._block('q_c', self.vq_c, raw)], format='csr')

    def items(self, t_lem, t_raw, p_lem):
        """Два блока для двух EmbeddingBag башни объявления: заголовок и параметры."""
        xt = sp.hstack([self._block('t_w', self.vt_w, t_lem), self._block('t_c', self.vt_c, t_raw)], format='csr')
        return xt, self._block('p_w', self.vp_w, p_lem)


def _bag(x):
    """Разреженная матрица в индексы, смещения и веса для nn.EmbeddingBag."""
    return torch.from_numpy(x.indices.astype(np.int64)), torch.from_numpy(x.indptr[:-1].astype(np.int64)), torch.from_numpy(x.data)


class Tower(nn.Module):
    """Взвешенная сумма эмбеддингов признаков (линейная проекция разреженного вектора)."""

    def __init__(self, nfeat, dim):
        super().__init__()
        self.eb = nn.EmbeddingBag(nfeat, dim, mode='sum', sparse=True)
        nn.init.normal_(self.eb.weight, std=0.1)

    def forward(self, x):
        i, o, w = _bag(x)
        return self.eb(i, o, per_sample_weights=w)


class TwoTower(nn.Module):
    def __init__(self, nq, nt, np_, n_mc, dim=128):
        super().__init__()
        self.q = Tower(nq, dim)
        self.it, self.ip = Tower(nt, dim), Tower(np_, dim)
        self.mc = nn.Embedding(n_mc + 1, dim, sparse=True)          # индекс 0 - неизвестная микрокатегория
        nn.init.normal_(self.mc.weight, std=0.1)
        self.q_mlp = nn.Sequential(nn.Linear(dim, dim), nn.Tanh(), nn.Linear(dim, dim))
        self.i_mlp = nn.Sequential(nn.Linear(dim, dim), nn.Tanh(), nn.Linear(dim, dim))
        self.mc_head = nn.Linear(dim, n_mc + 1)                     # микрокатегория по вектору запроса
        self.log_t = nn.Parameter(torch.tensor(np.log(1 / 0.07), dtype=torch.float32))   # обучаемая температура

    def enc_q(self, xq):
        e = self.q(xq)
        return e + self.q_mlp(e)

    def enc_i(self, xt, xp, mc):
        e = self.it(xt) + self.ip(xp) + self.mc(mc)
        return e + self.i_mlp(e)


def train_two_tower(xq_u, xt_u, xp_u, mc_u, qid, iid, n_mc, epochs=6, bs=2048, dim=128, lr_sparse=0.02, lr_dense=2e-3, mc_w=0.3,
                    seed=0, log=print, threads=6):
    """xq_u - признаки уникальных текстов запросов, xt_u, xp_u, mc_u - уникальных объявлений.
    qid и iid - для каждой обучающей пары номер строки в xq_u и xt_u (по ним же находятся ложные негативы)."""
    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)
    torch.set_num_threads(threads)
    m = TwoTower(xq_u.shape[1], xt_u.shape[1], xp_u.shape[1], n_mc, dim)
    sparse_params = [p for n, p in m.named_parameters() if n.split('.')[0] in ('q', 'it', 'ip', 'mc')]
    dense_params = [p for n, p in m.named_parameters() if n.split('.')[0] not in ('q', 'it', 'ip', 'mc')]
    opt_s = torch.optim.SparseAdam(sparse_params, lr=lr_sparse)     # для разреженных градиентов эмбеддингов
    opt_d = torch.optim.Adam(dense_params, lr=lr_dense)
    n, t0 = len(qid), time.time()
    y_mc = mc_u[iid]
    for ep in range(epochs):
        perm = rng.permutation(n)
        tot, nb = 0.0, 0
        for a in range(0, n - bs + 1, bs):
            r = perm[a:a + bs]
            eq = F.normalize(m.enc_q(xq_u[qid[r]]), dim=-1)
            ei = F.normalize(m.enc_i(xt_u[iid[r]], xp_u[iid[r]], torch.from_numpy(mc_u[iid[r]])), dim=-1)
            logits = eq @ ei.T * m.log_t.exp().clamp(max=100)
            qi_, ii_ = torch.from_numpy(qid[r]), torch.from_numpy(iid[r])
            # ложные негативы: тот же запрос или то же объявление на другой позиции батча
            fn = ((qi_[:, None] == qi_[None, :]) | (ii_[:, None] == ii_[None, :])) & ~torch.eye(len(r), dtype=torch.bool)
            lg = logits.masked_fill(fn, -1e4)
            tgt = torch.arange(len(r))
            loss = 0.5 * (F.cross_entropy(lg, tgt) + F.cross_entropy(lg.T, tgt))
            loss_mc = F.cross_entropy(m.mc_head(eq), torch.from_numpy(y_mc[r]))
            opt_s.zero_grad(); opt_d.zero_grad()
            (loss + mc_w * loss_mc).backward()
            opt_s.step(); opt_d.step()
            tot += loss.item(); nb += 1
        log(f'  epoch {ep+1}/{epochs} nce={tot/nb:.4f} mc={loss_mc.item():.3f} temp={1/m.log_t.exp().item():.3f} {time.time()-t0:.0f}s')
    return m


@torch.no_grad()
def encode_items(m, xt, xp, mc, bs=8192):
    m.eval()
    return np.concatenate([F.normalize(m.enc_i(xt[a:a + bs], xp[a:a + bs], torch.from_numpy(mc[a:a + bs])), dim=-1).numpy()
                           for a in range(0, xt.shape[0], bs)])


@torch.no_grad()
def encode_queries(m, xq, bs=8192):
    """Векторы запросов и log-вероятности микрокатегорий."""
    m.eval()
    vec, lp = [], []
    for a in range(0, xq.shape[0], bs):
        e = F.normalize(m.enc_q(xq[a:a + bs]), dim=-1)
        vec.append(e.numpy())
        lp.append(F.log_softmax(m.mc_head(e), dim=-1).numpy())
    return np.concatenate(vec), np.concatenate(lp)


def load_two_tower(path):
    """Загружает обученную модель, размеры берёт из формы весов."""
    sd = torch.load(path)
    nq, dim = sd['q.eb.weight'].shape
    m = TwoTower(nq, sd['it.eb.weight'].shape[0], sd['ip.eb.weight'].shape[0], sd['mc.weight'].shape[0] - 1, dim)
    m.load_state_dict(sd)
    m.eval()
    return m
