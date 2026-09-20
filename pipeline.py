"""Кандидаты из нескольких каналов и признаки пар (запрос, объявление).

Каналы кандидатов, для каждого запроса они объединяются:
1. BM25 по заголовку, параметрам и описанию с приорами (вероятность локации, вид услуги из фильтра).
2. Плотные модели (нейросеть, e5, адаптер e5) с теми же приорами.
3. Память по логам: объявления, на которые кликали по тому же тексту запроса.
4. Чистый текст без приоров (BM25 и косинусы): ловит клики вне выученной связи локаций.
Признаки пар получает ранкер (cg/ranker.py).
"""
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

from .bm25 import BM25Field, topk_rows, lookup
from .priors import LocAffinity, vid_of, km_dist

# слова, которые есть во всех фильтрах и ничего не различают
BOILER = set('вид тип услуга'.split())


class DenseModel:
    """Плотный канал: эмбеддинги объявлений и веса приоров.

    alpha - вес косинуса, loc_w, vid_w, mc_w - веса локации, вида услуги и микрокатегории,
    text_only_k - сколько кандидатов брать по чистому косинусу без приоров.
    """

    def __init__(self, name, item_emb, mc_index=None, alpha=20.0, loc_w=2.0, vid_w=3.0, mc_w=0.5, text_only_k=0):
        self.name, self.E = name, item_emb.astype(np.float32)
        self.mc_index = mc_index
        self.alpha, self.loc_w, self.vid_w, self.mc_w, self.text_only_k = alpha, loc_w, vid_w, mc_w, text_only_k


class Retriever:
    """Индексы по пулу объявлений и статистики кликов base."""

    def __init__(self, pool, base_pairs, base_items, qlem, flem, verbose=True):
        """pool - объявления, среди которых ищем (леммы в колонках t, p, d).
        base_pairs - клики base, base_items - объявления train (по item_id), qlem и flem - леммы запросов и фильтров."""
        t0 = time.time()
        self.pool = pool.reset_index(drop=True)
        self.N = len(self.pool)
        self.qlem, self.flem = qlem, flem
        self.Ft, self.Fp, self.Fd = BM25Field(self.pool.t.values), BM25Field(self.pool.p.values), BM25Field(self.pool.d.values)
        # есть ли слово в параметрах объявления (для доли слов фильтра)
        self.Bp = (self.Fp.W > 0).astype(np.float32).T.tocsr()
        # локации
        self.aff = LocAffinity(base_pairs.search_location_id.values, base_items.item_location_id.reindex(base_pairs.item_id).values)
        self.item_loc = self.pool.item_location_id.values.astype(np.int64)
        self.uloc, self.loc_inv = np.unique(self.item_loc, return_inverse=True)
        ls = pd.Series(self.item_loc)
        self.loc_pool = ls.map(ls.value_counts()).values.astype(np.float32)          # объявлений пула в этой локации
        self.loc_clicks = base_pairs.search_location_id.value_counts()                # кликов из этой локации поиска в логе
        # вид услуги
        vids = self.pool.item_infm_params_text.map(vid_of)
        self.vid_codes = {v: n + 1 for n, v in enumerate(sorted(set(vids) - {''}))}
        self.ivid = vids.map(lambda v: self.vid_codes.get(v, 0)).values.astype(np.int32)
        self.tlen = self.pool.t.str.count(' ').values.astype(np.float32) + 1
        self.dlen = self.pool.d.str.count(' ').values.astype(np.float32) + 1
        self.dense = {}
        self._set_memory(base_pairs)
        self._set_geo(base_pairs, base_items)
        self._set_quality()
        if verbose:
            print(f'Retriever built: N={self.N} {time.time()-t0:.0f}s', flush=True)

    def _set_memory(self, base_pairs):
        """Память по логам: сколько раз по тому же тексту запроса кликали объявление пула."""
        item_index = {x: n for n, x in enumerate(self.pool.item_id.values)}
        lem = base_pairs.search_query.map(self.qlem)
        df = pd.DataFrame({'lem': lem.values, 'sloc': base_pairs.search_location_id.values, 'item_id': base_pairs.item_id.values})
        self.seen_lem = set(df.lem.values)
        pc = df.item_id.value_counts()                                               # популярность объявления в base
        self.pop_base = np.log1p(pc.reindex(self.pool.item_id.values).fillna(0).values).astype(np.float32)
        df['j'] = df.item_id.map(item_index)
        df = df.dropna(subset=['j']).astype({'j': np.int64})
        self.mem_all = df.groupby(['lem', 'j']).size().rename('n').sort_index()
        self.mem_loc = df.groupby(['lem', 'sloc', 'j']).size().rename('n').sort_index()
        self.mem_by_lem = df.groupby('lem').j.apply(lambda s: np.unique(s.values)).to_dict()

    def _set_geo(self, base_pairs, base_items):
        """Центр локации поиска: медианные координаты объявлений, которые из неё кликали."""
        b = base_pairs[['search_location_id', 'item_id']].join(base_items[['item_latitude', 'item_longitude']], on='item_id')
        self.cen = b.groupby('search_location_id').agg(lat=('item_latitude', 'median'), lon=('item_longitude', 'median'))
        self.ilat = self.pool.item_latitude.values.astype(np.float32)
        self.ilon = self.pool.item_longitude.values.astype(np.float32)

    def _set_quality(self):
        """Признаки самого объявления: отзывы, рейтинг, цена, флаги контактов."""
        p = self.pool
        rev = p.item_rating_reviews_count.values.astype(np.float32)
        self.q_reviews = np.where(np.isnan(rev), -1, np.log1p(np.nan_to_num(rev))).astype(np.float32)
        self.q_rating = np.nan_to_num(p.item_rating.values.astype(np.float32), nan=-1.0)
        price = p.item_price.values.astype(np.float32)
        self.q_price = np.log1p(np.maximum(price, 0)).astype(np.float32)
        self.q_price_missing = (price <= 0).astype(np.float32)
        self.q_phone = p.item_is_phone_hidden.values.astype(np.float32)
        self.q_msg = p.item_is_message_forbidden.values.astype(np.float32)

    def add_dense(self, model: DenseModel):
        self.dense[model.name] = model

    def _query_side(self, queries):
        """Леммы запросов, леммы слов фильтра и код вида услуги из фильтра."""
        qd = [self.qlem[s] for s in queries.search_query]
        fl = [' '.join(w for w in self.flem[f].split() if w not in BOILER) for f in queries.search_infm_params_text]
        fvid = np.array([self.vid_codes.get(vid_of(f), 0) for f in queries.search_infm_params_text], dtype=np.int32)
        return qd, fl, fvid

    def _loc_logaff(self, slocs):
        """log(P(локация объявления | локация поиска) + eps) для каждой пары (запрос, объявление пула)."""
        us, inv = np.unique(slocs, return_inverse=True)
        tab = np.stack([np.log(self.aff.get(np.full(len(self.uloc), s), self.uloc) + 1e-3) for s in us]).astype(np.float32)
        return tab[inv][:, self.loc_inv]

    def build(self, queries, dense_q, k_bm25=300, k_dense=150, k_text=40, chunk=250, loc_w=5.0, vid_w=10.0, prune_fn=None, verbose=True):
        """Строит кандидатов и признаки пар для запросов.

        queries - запросы (search_query, search_location_id, search_infm_params_text).
        dense_q - {имя модели: (эмбеддинги запросов, log-вероятности микрокатегорий или None)}.
        prune_fn - функция, отсекающая слабых кандидатов сразу в чанке, чтобы экономить память.
        Возвращает таблицу пар: qi - номер запроса, ji - номер объявления в пуле, дальше признаки.
        """
        t0 = time.time()
        qd, fl, fvid = self._query_side(queries)
        sloc = queries.search_location_id.values.astype(np.int64)
        qlen_all = np.array([len(x) for x in queries.search_query.values], dtype=np.float32)
        nterms_all = np.array([len(x.split()) for x in qd], dtype=np.float32)
        out = []
        for a in range(0, len(queries), chunk):
            b = min(a + chunk, len(queries))
            n = b - a
            qt, qp, qdd = self.Ft.qmat(qd[a:b]), self.Fp.qmat(qd[a:b]), self.Fd.qmat(qd[a:b])
            st, sp_, sd = self.Ft.score(qt), self.Fp.score(qp), self.Fd.score(qdd)
            s0 = (st + sp_ + sd).tocsr()                                       # суммарный BM25 по трём полям
            nt, npp, nd = (np.asarray(qt @ self.Ft.idf).ravel(), np.asarray(qp @ self.Fp.idf).ravel(), np.asarray(qdd @ self.Fd.idf).ravel())

            # канал 1: BM25 с приорами (локация, вид услуги)
            coo = s0.tocoo()
            r, c, d = coo.row, coo.col, coo.data
            p = self.aff.get(sloc[a:b][r], self.item_loc[c])
            fv, iv = fvid[a:b][r], self.ivid[c]
            g0 = d + loc_w * np.log(p + 1e-3) + vid_w * np.where(fv == 0, 0.0, np.where(fv == iv, 0.0, -1.0))
            ci, _ = topk_rows(sp.csr_matrix((g0.astype(np.float32), (r, c)), shape=s0.shape), k_bm25)
            parts_q = [np.concatenate([np.full(len(x), k, dtype=np.int64) for k, x in enumerate(ci)])]
            parts_j = [np.concatenate(ci).astype(np.int64)]
            # канал 4: чистый BM25 без приоров
            if k_text:
                ct, _ = topk_rows(s0, k_text)
                parts_q.append(np.concatenate([np.full(len(x), k, dtype=np.int64) for k, x in enumerate(ct)]))
                parts_j.append(np.concatenate(ct).astype(np.int64))
            # каналы 2 и 4: плотные модели с приорами и по чистому косинусу
            if dense_q:
                la_mat = self._loc_logaff(sloc[a:b])
                fvb = fvid[a:b, None]
                vp_mat = np.where((fvb != 0) & (fvb != self.ivid[None, :]), -1.0, 0.0).astype(np.float32)
                for name, (eq, lp) in dense_q.items():
                    m = self.dense[name]
                    cos = torch.from_numpy(eq[a:b] @ m.E.T)
                    s = cos * m.alpha + m.loc_w * torch.from_numpy(la_mat) + m.vid_w * torch.from_numpy(vp_mat)
                    if lp is not None and m.mc_w:
                        s += m.mc_w * torch.from_numpy(lp[a:b][:, m.mc_index])
                    parts_q.append(np.repeat(np.arange(n), k_dense))
                    parts_j.append(torch.topk(s, k_dense, dim=1).indices.numpy().ravel().astype(np.int64))
                    if m.text_only_k:
                        parts_q.append(np.repeat(np.arange(n), m.text_only_k))
                        parts_j.append(torch.topk(cos, m.text_only_k, dim=1).indices.numpy().ravel().astype(np.int64))
                    del s, cos
            # канал 3: память по логам
            for k in range(n):
                mj = self.mem_by_lem.get(qd[a + k])
                if mj is not None:
                    parts_q.append(np.full(len(mj), k))
                    parts_j.append(mj)

            # объединяем кандидатов и считаем признаки пар
            key = np.unique(np.concatenate(parts_q) * self.N + np.concatenate(parts_j))
            lr = (key // self.N).astype(np.int64)
            ji = (key % self.N).astype(np.int64)
            qi = lr + a
            f = {'qi': qi.astype(np.int32), 'ji': ji.astype(np.int32)}
            f['st'], f['sp'], f['sd'] = lookup(st.tocsr(), lr, ji), lookup(sp_.tocsr(), lr, ji), lookup(sd.tocsr(), lr, ji)
            f['st_n'] = f['st'] / np.maximum(nt[lr], 1e-6)                      # BM25 на сумму IDF слов запроса
            f['sp_n'] = f['sp'] / np.maximum(npp[lr], 1e-6)
            f['sd_n'] = f['sd'] / np.maximum(nd[lr], 1e-6)
            f['s_sum'] = f['st'] + f['sp'] + f['sd']
            rowmax = np.maximum(np.asarray(s0.max(axis=1).todense()).ravel(), 1e-6)
            f['s_rel'] = f['s_sum'] / rowmax[lr]
            f['q_nmatch'] = np.log1p(np.diff(s0.indptr).astype(np.float32))[lr]  # общность запроса: сколько объявлений с ним совпало
            f['aff'] = self.aff.get(sloc[qi], self.item_loc[ji])
            f['exact'] = (sloc[qi] == self.item_loc[ji]).astype(np.float32)
            f['loc_pool'] = self.loc_pool[ji]
            f['sloc_n'] = self.loc_clicks.reindex(sloc[qi]).fillna(0).values.astype(np.float32)
            fvq = fvid[qi]
            f['vid_state'] = np.where(fvq == 0, 0, np.where(fvq == self.ivid[ji], 1, -1)).astype(np.float32)   # 0 - фильтра нет, 1 - совпал, -1 - не совпал
            qf = self.Fp.qmat(fl[a:b])
            nf = np.asarray(qf.sum(1)).ravel()
            sf = (qf @ self.Bp).tocsr()
            f['ftok'] = np.where(nf[lr] > 0, lookup(sf, lr, ji) / np.maximum(nf[lr], 1), -1).astype(np.float32)   # доля слов фильтра в параметрах объявления
            f['nterms'], f['qlen'] = nterms_all[qi], qlen_all[qi]
            f['tlen'], f['dlen'] = self.tlen[ji], self.dlen[ji]
            vpen = np.where(fvq == 0, 0.0, np.where(fvq == self.ivid[ji], 0.0, -1.0))
            la = np.log(f['aff'] + 1e-3)
            f['g0'] = (f['s_sum'] + loc_w * la + vid_w * vpen).astype(np.float32)
            for name, (eq, lp) in dense_q.items():
                m = self.dense[name]
                f[f'cos_{name}'] = (eq[qi] * m.E[ji]).sum(1).astype(np.float32)
                g = m.alpha * f[f'cos_{name}'] + m.loc_w * la + m.vid_w * vpen
                if lp is not None:
                    f[f'lpmc_{name}'] = lp[qi, m.mc_index[ji]].astype(np.float32)   # log-вероятность микрокатегории объявления по запросу
                    f[f'mctop_{name}'] = lp[qi].max(1).astype(np.float32)             # уверенность классификатора
                    g = g + m.mc_w * f[f'lpmc_{name}']
                f[f'gd_{name}'] = g.astype(np.float32)
            lem = np.array(qd, dtype=object)[qi]
            f['mem_n'] = self.mem_all.reindex(pd.MultiIndex.from_arrays([lem, ji])).fillna(0).values.astype(np.float32)
            f['mem_loc_n'] = self.mem_loc.reindex(pd.MultiIndex.from_arrays([lem, sloc[qi], ji])).fillna(0).values.astype(np.float32)
            f['q_seen'] = np.array([x in self.seen_lem for x in qd], dtype=np.float32)[qi]
            f['pop_base'] = self.pop_base[ji]
            cl = self.cen.reindex(sloc[qi])
            gk = np.log1p(km_dist(cl.lat.values, cl.lon.values, self.ilat[ji], self.ilon[ji])).astype(np.float32)
            gk[np.isnan(gk)] = -1
            f['geo_km'] = gk
            f['q_reviews'], f['q_rating'], f['q_price'] = self.q_reviews[ji], self.q_rating[ji], self.q_price[ji]
            f['q_price_missing'], f['q_phone'], f['q_msg'] = self.q_price_missing[ji], self.q_phone[ji], self.q_msg[ji]
            df = pd.DataFrame(f)
            # ранги и отклонения от лучшего значения внутри запроса
            for g in ['g0'] + [f'gd_{nm}' for nm in dense_q]:
                df['r_' + g] = df.groupby('qi')[g].rank(ascending=False, method='first').astype(np.float32)
            for g in ['s_sum'] + [f'cos_{nm}' for nm in dense_q]:
                df[g + '_rel'] = df[g] - df.groupby('qi')[g].transform('max')
                df['r_' + g] = df.groupby('qi')[g].rank(ascending=False, method='first').astype(np.float32)   # ранг по чистому тексту
            df['q_reviews_pct'] = df.groupby('qi')['q_reviews'].rank(pct=True).astype(np.float32)
            out.append(prune_fn(df) if prune_fn is not None else df)
            if verbose and (a // chunk) % 20 == 0:
                print(f'  chunk {a}/{len(queries)} {time.time()-t0:.0f}s', flush=True)
        return pd.concat(out, ignore_index=True)
