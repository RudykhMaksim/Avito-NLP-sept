#!/usr/bin/env python
"""Кандидатогенерация для поиска услуг Авито: до 50 объявлений на запрос (метрика Recall@50).

Запуск: python solution.py --data data --work work --out answer.csv
Готовые стадии берутся из кэша в папке --work.

Как работает решение (подробнее в README.md):
1. Связь между локацией поиска и локацией объявления выучивается по кликам и служит приором.
2. Кандидаты берутся из нескольких каналов: BM25, двухбашенная нейросеть, e5-small с адаптером, память по логам, чистый текст.
3. LightGBM-ранкер выбирает из кандидатов финальные 50.
Ранкер обучается на запросах train, отложенных от обучения нейросетей и статистик.
Используются только open-source компоненты, всё запускается локально.
"""
import argparse
import os
import pickle
import re
import sys
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import scipy.sparse as sp   # noqa: F401

from cg import bm25, e5 as e5mod, pipeline, priors, ranker, text, twotower

SEED = 42
LF = chr(10)                                                 # перевод строки в answer.csv
N_VAL, N_FIT = 3000, 30000                                   # число запросов для валидации и для обучения ранкера
ADAPTER = dict(epochs=8, lr=5e-4, hidden=768, temp=0.03)     # параметры обучения адаптера e5
T0 = time.time()
DEBUG = False                                                # режим --debug: маленькая выборка, только проверка работоспособности


def log(*a):
    print(f'[{time.time()-T0:6.0f}s]', *a, flush=True)


def read_bench_queries(args):
    """Запросы бенчмарка (в режиме debug только первые 100)."""
    bq = pd.read_parquet(os.path.join(args.data, 'benchmark_queries.parquet')).reset_index(drop=True)
    return bq.head(100) if DEBUG else bq


def read_corpus(args, cols):
    """Корпус объявлений бенчмарка (в режиме debug только первые 8000)."""
    c = pq.read_table(os.path.join(args.data, 'benchmark_items.parquet'), columns=cols).to_pandas()
    return c.head(8000) if DEBUG else c


class Work:
    """Папка с кэшем результатов стадий."""

    def __init__(self, root):
        self.root = root
        os.makedirs(root, exist_ok=True)

    def p(self, name):
        return os.path.join(self.root, name)

    def has(self, *names):
        return all(os.path.exists(self.p(n)) for n in names)


ITEM_COLS = ['item_id', 'item_title_raw', 'item_description_raw', 'item_infm_params_text', 'item_category_id', 'item_microcat_id', 'item_price',
             'item_rating', 'item_rating_reviews_count', 'item_location_id', 'item_latitude', 'item_longitude', 'item_is_phone_hidden',
             'item_is_message_forbidden']
QUERY_KEY = ['search_query', 'search_location_id', 'search_is_delivery_search', 'search_infm_params_text', 'search_category']


def stage_prep(args, W):
    """Стадия 1. Пары запрос-объявление из train, уникальные объявления train, леммы текстов."""
    if W.has('train_pairs.parquet', 'train_items.parquet', 'train_items_tp.pkl', 'qlem.pkl'):
        return
    log('стадия 1: чтение train и лемматизация')
    cols = [c for c in pq.ParquetFile(os.path.join(args.data, 'train.parquet')).schema_arrow.names if c != 'item_description_raw']
    tr = pq.read_table(os.path.join(args.data, 'train.parquet'), columns=cols).to_pandas()
    if DEBUG:
        tr = tr.sample(60000, random_state=0).reset_index(drop=True)
    for c in ['item_price', 'item_longitude', 'item_latitude']:
        tr[c] = tr[c].astype('float64')
    tr['search_infm_params_text'] = tr['search_infm_params_text'].fillna('')
    tr['key_id'] = tr.groupby(QUERY_KEY, sort=False).ngroup().astype('int32')          # один key_id - один поисковый запрос
    tr[['key_id', 'search_query', 'search_location_id', 'search_infm_params_text', 'search_category', 'item_id']].to_parquet(W.p('train_pairs.parquet'))
    items = tr.drop_duplicates('item_id')[[c for c in ITEM_COLS if c != 'item_description_raw']].reset_index(drop=True)
    items.to_parquet(W.p('train_items.parquet'))
    # леммы заголовка и параметров объявлений train (для нейросети)
    tp = items[['item_id']].copy()
    tp['t'] = [text.lemmatize_join(s) for s in items.item_title_raw.fillna('')]
    tp['p'] = [text.lemmatize_join(s) for s in items.item_infm_params_text.fillna('')]
    tp.to_pickle(W.p('train_items_tp.pkl'))
    # леммы текстов запросов и фильтров из train и бенчмарка
    bq = read_bench_queries(args)
    qtexts = pd.concat([tr.search_query, bq.search_query]).unique()
    ftexts = pd.concat([tr.search_infm_params_text, bq.search_infm_params_text]).unique()
    pickle.dump(({s: text.lemmatize_join(s) for s in qtexts}, {s: text.lemmatize_join(s) for s in ftexts}), open(W.p('qlem.pkl'), 'wb'))
    log('стадия 1 готова: пар', len(tr), 'уникальных объявлений', len(items))


def stage_split(args, W):
    """Стадия 2. Делит запросы train на val, fit и base и строит пул объявлений.

    val и fit - по одному случайному запросу на уникальный текст, как в бенчмарке. base - остальные клики,
    на них учатся нейросети и статистики. Пул - корпус плюс положительные объявления val и fit.
    """
    if W.has('qsplit.parquet', 'base_pairs.parquet', 'pool.pkl', 'pos.parquet'):
        return
    log('стадия 2: разбиение и пул объявлений')
    P = pd.read_parquet(W.p('train_pairs.parquet'))
    rng = np.random.RandomState(SEED)
    keys = P.drop_duplicates('key_id')[['key_id', 'search_query', 'search_location_id', 'search_infm_params_text', 'search_category']].reset_index(drop=True)
    keys = keys[keys.search_category == 114]
    one = keys.sample(frac=1, random_state=SEED).drop_duplicates('search_query')
    texts = one.search_query.values.copy()
    rng.shuffle(texts)
    one = one.set_index('search_query').loc[texts].reset_index()
    n_val, n_fit = (200, 1200) if DEBUG else (N_VAL, N_FIT)
    val, fit = one.iloc[:n_val].copy(), one.iloc[n_val:n_val + n_fit].copy()
    val['split'], fit['split'] = 'val', 'fit'
    Q = pd.concat([val, fit], ignore_index=True)
    Q.loc[rng.rand(len(Q)) < 0.30, 'search_infm_params_text'] = ''                     # в бенчмарке фильтры есть у 37% запросов
    Q.to_parquet(W.p('qsplit.parquet'))
    held = set(Q.key_id)
    P[~P.key_id.isin(held)].to_parquet(W.p('base_pairs.parquet'))
    pos = P[P.key_id.isin(held)][['key_id', 'item_id']]
    pos.to_parquet(W.p('pos.parquet'))
    corpus = read_corpus(args, ITEM_COLS)
    corpus['src'] = 'corpus'
    need = set(pos.item_id) - set(corpus.item_id)                                      # положительные объявления вне корпуса, описание берём из train
    tr_items = pq.read_table(os.path.join(args.data, 'train.parquet'), columns=ITEM_COLS).to_pandas()
    tr_items = tr_items[tr_items.item_id.isin(need)].drop_duplicates('item_id')
    tr_items['src'] = 'train'
    pool = pd.concat([corpus, tr_items], ignore_index=True)
    del tr_items
    for c in ['item_price', 'item_latitude', 'item_longitude']:
        pool[c] = pool[c].astype('float64')
    for c in ['item_title_raw', 'item_description_raw', 'item_infm_params_text']:
        pool[c] = pool[c].fillna('')
    pool['t'] = [text.lemmatize_join(s) for s in pool.item_title_raw]
    pool['p'] = [text.lemmatize_join(s) for s in pool.item_infm_params_text]
    pool['d'] = [text.lemmatize_join(s) for s in pool.item_description_raw]
    pool.drop(columns=['item_description_raw']).to_pickle(W.p('pool.pkl'))
    log('стадия 2 готова: val', len(val), 'fit', len(fit), 'пул', len(pool), '(корпус', len(corpus), ')')


def stage_twotower(args, W):
    """Стадия 3. Обучает двухбашенную нейросеть на кликах base, считает эмбеддинги пула и запросов."""
    if W.has('tt.pt', 'tt_fz.pkl', 'tt_Ei_pool.npy', 'tt_Eq.pkl'):
        return
    log('стадия 3: обучение двухбашенной нейросети')
    import torch
    base = pd.read_parquet(W.p('base_pairs.parquet'))
    tit = pd.read_parquet(W.p('train_items.parquet'))
    titp = pd.read_pickle(W.p('train_items_tp.pkl'))
    qlem, flem = pickle.load(open(W.p('qlem.pkl'), 'rb'))
    pool = pd.read_pickle(W.p('pool.pkl'))
    bp = base[['search_query', 'item_id']].drop_duplicates().reset_index(drop=True)        # уникальные пары (текст, объявление)
    bitems = tit.merge(titp, on='item_id')
    bitems = bitems[bitems.item_id.isin(set(bp.item_id))].reset_index(drop=True)
    i_row = {x: n for n, x in enumerate(bitems.item_id)}
    utexts = pd.Series(bp.search_query.unique())
    q_row = {x: n for n, x in enumerate(utexts)}
    qid, iid = bp.search_query.map(q_row).values.astype(np.int64), bp.item_id.map(i_row).values.astype(np.int64)
    mcs = sorted(bitems.item_microcat_id.unique())
    mc_map = {m: n + 1 for n, m in enumerate(mcs)}
    mc_u = bitems.item_microcat_id.map(mc_map).values.astype(np.int64)
    low = lambda s: s.fillna('').str.lower().tolist()
    fz = twotower.Featurizer([qlem[s] for s in utexts], utexts.str.lower().tolist(), bitems.t.tolist(), low(bitems.item_title_raw), bitems.p.tolist())
    xq = fz.queries([qlem[s] for s in utexts], utexts.str.lower().tolist())
    xt, xp = fz.items(bitems.t.tolist(), low(bitems.item_title_raw), bitems.p.tolist())
    m = twotower.train_two_tower(xq, xt, xp, mc_u, qid, iid, len(mcs), epochs=1 if DEBUG else 6, bs=2048, seed=0, threads=args.threads, log=log)
    torch.save(m.state_dict(), W.p('tt.pt'))
    pickle.dump((fz, mc_map, len(mcs)), open(W.p('tt_fz.pkl'), 'wb'))
    xt_p, xp_p = fz.items(pool.t.tolist(), low(pool.item_title_raw), pool.p.tolist())
    mc_p = pool.item_microcat_id.map(mc_map).fillna(0).values.astype(np.int64)
    np.save(W.p('tt_Ei_pool.npy'), twotower.encode_items(m, xt_p, xp_p, mc_p))
    # эмбеддинги запросов val, fit и бенчмарка по уникальным текстам
    Q = pd.read_parquet(W.p('qsplit.parquet'))
    bq = read_bench_queries(args)
    qt = pd.Series(pd.concat([Q.search_query, bq.search_query]).unique())
    vec, lp = twotower.encode_queries(m, fz.queries([qlem[s] for s in qt], qt.str.lower().tolist()))
    pickle.dump(({s: n for n, s in enumerate(qt)}, vec, lp), open(W.p('tt_Eq.pkl'), 'wb'))
    log('стадия 3 готова')


def stage_e5(args, W):
    """Стадия 4. Кодирует объявления пула и запросы моделью e5-small."""
    if W.has('e5_pool.npy', 'e5_q.npy', 'e5_q_texts.pkl'):
        return
    log('стадия 4: кодирование e5-small')
    pool = pd.read_pickle(W.p('pool.pkl'))
    Q = pd.read_parquet(W.p('qsplit.parquet'))
    bq = read_bench_queries(args)
    enc = e5mod.load_encoder(args.threads)
    np.save(W.p('e5_pool.npy'), e5mod.encode(enc, e5mod.item_texts(pool.item_title_raw.values, pool.item_infm_params_text.values)))
    log('  объявления пула закодированы')
    qt = pd.Series(pd.concat([Q.search_query, bq.search_query]).unique())
    np.save(W.p('e5_q.npy'), e5mod.encode(enc, ['query: ' + s for s in qt.values], batch_size=256))
    pickle.dump({s: n for n, s in enumerate(qt.values)}, open(W.p('e5_q_texts.pkl'), 'wb'))


def stage_adapter(args, W):
    """Стадия 4б. Обучает адаптер над эмбеддингами e5 на кликах base."""
    if not args.adapter or W.has('adapter_Ei_pool.npy', 'adapter_Eq.pkl'):
        return
    log('стадия 4б: адаптер e5 на кликах')
    import torch
    base = pd.read_parquet(W.p('base_pairs.parquet'))
    tit = pd.read_parquet(W.p('train_items.parquet'))
    pool = pd.read_pickle(W.p('pool.pkl'))
    bp = base[['search_query', 'item_id']].drop_duplicates().reset_index(drop=True)
    if not W.has('e5_train_items.npy', 'e5_train_q.npy'):
        enc = e5mod.load_encoder(args.threads)
        used = tit[tit.item_id.isin(set(bp.item_id))].reset_index(drop=True)
        np.save(W.p('e5_train_items.npy'), e5mod.encode(enc, e5mod.item_texts(used.item_title_raw.values, used.item_infm_params_text.values)))
        pickle.dump(used.item_id.tolist(), open(W.p('e5_train_item_ids.pkl'), 'wb'))
        ut = pd.Series(bp.search_query.unique())
        np.save(W.p('e5_train_q.npy'), e5mod.encode(enc, ['query: ' + s for s in ut.values], batch_size=256))
        pickle.dump({s: n for n, s in enumerate(ut.values)}, open(W.p('e5_train_q_texts.pkl'), 'wb'))
    eq = torch.from_numpy(np.load(W.p('e5_train_q.npy')).astype(np.float32))
    ei = torch.from_numpy(np.load(W.p('e5_train_items.npy')).astype(np.float32))
    qrow = pickle.load(open(W.p('e5_train_q_texts.pkl'), 'rb'))
    irow = {x: n for n, x in enumerate(pickle.load(open(W.p('e5_train_item_ids.pkl'), 'rb')))}
    mc_of = dict(zip(tit.item_id, tit.item_microcat_id))
    mcs = sorted(set(mc_of[i] for i in bp.item_id.unique()))
    mc_map = {m: n + 1 for n, m in enumerate(mcs)}
    qidx, iidx = bp.search_query.map(qrow).values, bp.item_id.map(irow).values
    mc_pair = bp.item_id.map(lambda i: mc_map.get(mc_of[i], 0)).values.astype(np.int64)
    bs = 512 if DEBUG else 4096
    m = e5mod.train_adapter(eq, ei, mc_pair, qidx, iidx, bp.search_query.factorize()[0], bp.item_id.factorize()[0], len(mcs),
                            epochs=1 if DEBUG else ADAPTER['epochs'], lr=ADAPTER['lr'], hidden=ADAPTER['hidden'], temp=ADAPTER['temp'], bs=bs, threads=args.threads, log=log)
    mc_p = pool.item_microcat_id.map(mc_map).fillna(0).values.astype(np.int64)
    np.save(W.p('adapter_Ei_pool.npy'), e5mod.adapt_items(m, np.load(W.p('e5_pool.npy')), mc_p))
    eqall = np.load(W.p('e5_q.npy'))
    vec, lp = e5mod.adapt_queries(m, eqall)
    pickle.dump((vec, lp), open(W.p('adapter_Eq.pkl'), 'wb'))
    pickle.dump(mc_map, open(W.p('adapter_mcmap.pkl'), 'wb'))
    log('стадия 4б готова')


def dense_models(args, W, pool_slice=None):
    """Плотные модели и функции, возвращающие эмбеддинги запросов. pool_slice - часть пула (для бенчмарка только корпус)."""
    sl = pool_slice if pool_slice is not None else slice(None)
    pool = pd.read_pickle(W.p('pool.pkl'))
    models, getters = [], {}
    tt_map = pickle.load(open(W.p('tt_fz.pkl'), 'rb'))[1]
    mc_tt = pool.item_microcat_id.map(tt_map).fillna(0).values.astype(np.int64)[sl]
    models.append(pipeline.DenseModel('tt', np.load(W.p('tt_Ei_pool.npy'))[sl], mc_tt, alpha=20.0, loc_w=2.0, vid_w=3.0, mc_w=0.5, text_only_k=40))
    row, vec, lp = pickle.load(open(W.p('tt_Eq.pkl'), 'rb'))
    getters['tt'] = lambda texts: (vec[[row[s] for s in texts]], lp[[row[s] for s in texts]])
    e5row = pickle.load(open(W.p('e5_q_texts.pkl'), 'rb'))
    e5q = np.load(W.p('e5_q.npy')).astype(np.float32)
    models.append(pipeline.DenseModel('e5', np.load(W.p('e5_pool.npy'))[sl].astype(np.float32), None, alpha=200.0, loc_w=1.0, vid_w=3.0, mc_w=0.0, text_only_k=40))
    getters['e5'] = lambda texts: (e5q[[e5row[s] for s in texts]], None)
    if args.adapter:
        ad_map = pickle.load(open(W.p('adapter_mcmap.pkl'), 'rb'))
        mc_ad = pool.item_microcat_id.map(ad_map).fillna(0).values.astype(np.int64)[sl]
        models.append(pipeline.DenseModel('e5a', np.load(W.p('adapter_Ei_pool.npy'))[sl], mc_ad, alpha=20.0, loc_w=2.0, vid_w=3.0, mc_w=0.5, text_only_k=40))
        avec, alp = pickle.load(open(W.p('adapter_Eq.pkl'), 'rb'))
        getters['e5a'] = lambda texts: (avec[[e5row[s] for s in texts]], alp[[e5row[s] for s in texts]])
    return models, getters


def stage_candidates(args, W):
    """Стадия 5. Кандидаты и признаки для val и fit (пул: корпус плюс положительные объявления val и fit)."""
    if W.has('cands_val.parquet', 'cands_fit.parquet'):
        return
    log('стадия 5: кандидаты и признаки для обучения ранкера')
    pool = pd.read_pickle(W.p('pool.pkl'))
    Q = pd.read_parquet(W.p('qsplit.parquet'))
    base = pd.read_parquet(W.p('base_pairs.parquet'))
    tit = pd.read_parquet(W.p('train_items.parquet')).set_index('item_id')
    qlem, flem = pickle.load(open(W.p('qlem.pkl'), 'rb'))
    pos = pd.read_parquet(W.p('pos.parquet'))
    idx = {i: n for n, i in enumerate(pool.item_id)}
    pos['j'] = pos.item_id.map(idx)
    posset = set(zip(pos.key_id, pos.j))
    retr = pipeline.Retriever(pool, base, tit, qlem, flem)
    models, getters = dense_models(args, W)
    for m in models:
        retr.add_dense(m)
    for name in ['val', 'fit']:
        Qs = Q[Q.split == name].reset_index(drop=True)
        if name == 'fit':
            Qs = Qs.iloc[:args.n_fit].reset_index(drop=True)
        dq = {m.name: getters[m.name](Qs.search_query.tolist()) for m in models}
        C = retr.build(Qs, dq, prune_fn=ranker.prune)                            # слабых кандидатов отсекаем сразу, чтобы экономить память
        kid = Qs.key_id.values[C.qi.values]
        C['key_id'] = kid
        C['y'] = np.fromiter(((k, j) in posset for k, j in zip(kid, C.ji.values)), dtype=np.int8, count=len(C))
        C.to_parquet(W.p(f'cands_{name}.parquet'))
        npos = pos[pos.key_id.isin(set(Qs.key_id))].groupby('key_id').size()
        got = C.groupby('key_id').y.sum().reindex(npos.index).fillna(0)
        log(f'  {name}: запросов {len(Qs)}, кандидатов на запрос {len(C)/len(Qs):.0f}, покрытие кандидатов = {float((got / npos).mean()):.4f}')


def excluded_features(args):
    """Исключаемые признаки. Качество и популярность объявления по умолчанию не используются:
    на платформе без них лучше, а на локальной валидации они завышали оценку. Флаг --with-quality включает их."""
    return [] if args.with_quality else ranker.QUALITY_FEATURES + ranker.POPULARITY_FEATURES


def ranker_name(args):
    """Имя файла ранкера: варианты с признаками качества и без них хранятся отдельно."""
    return 'ranker_q' if args.with_quality else 'ranker'


def stage_ranker(args, W):
    """Стадия 6. Обучает ранкер: сначала на fit с проверкой на val, затем итоговую модель на fit и val."""
    rn = ranker_name(args)
    if W.has(rn + '.txt', rn + '_report.txt'):
        return
    log('стадия 6: обучение ранкера')
    pos = pd.read_parquet(W.p('pos.parquet'))
    npos = pos.groupby('key_id').size()
    CF = pd.read_parquet(W.p('cands_fit.parquet'))
    CV = pd.read_parquet(W.p('cands_val.parquet'))
    feats = ranker.feature_names(CF, exclude=excluded_features(args))
    # оценка: обучение на fit с ранней остановкой, Recall на val
    booster, n_iter = ranker.train(CF, feats, threads=args.threads)
    rec = ranker.recall_at_k(CV, booster.predict(CV[feats], num_iteration=n_iter), npos, ks=(20, 50, 100))
    log('  ВАЛИДАЦИЯ (3000 запросов train, устроенных как бенчмарк): Recall@20/50/100 =', {k: round(v, 4) for k, v in rec.items()}, 'деревьев', n_iter)
    # итоговая модель: fit и val, деревьев на 10% больше
    final, _ = ranker.train(pd.concat([CF, CV], ignore_index=True), feats, n_rounds=int(n_iter * 1.1), threads=args.threads)
    final.save_model(W.p(rn + '.txt'))
    open(W.p(rn + '_report.txt'), 'w').write(f'val recall {rec}\niters {n_iter}\nfeatures {len(feats)}\n')


def stage_bench(args, W):
    """Стадия 7. Строит ответ для бенчмарка: кандидаты, ранкер, топ-50."""
    import lightgbm as lgb
    log('стадия 7: инференс на бенчмарке')
    pool = pd.read_pickle(W.p('pool.pkl'))
    n = int((pool.src == 'corpus').sum())
    corp = pool.iloc[:n].reset_index(drop=True)                                  # корпус стоит первым в пуле, в порядке benchmark_items.parquet
    real = read_corpus(args, ['item_id']).item_id.values
    assert (corp.item_id.values == real).all(), 'порядок корпуса нарушен'
    bq = read_bench_queries(args)
    base = pd.read_parquet(W.p('base_pairs.parquet'))
    tit = pd.read_parquet(W.p('train_items.parquet')).set_index('item_id')
    qlem, flem = pickle.load(open(W.p('qlem.pkl'), 'rb'))
    retr = pipeline.Retriever(corp, base, tit, qlem, flem)
    models, getters = dense_models(args, W, pool_slice=slice(0, n))
    for m in models:
        retr.add_dense(m)
    dq = {m.name: getters[m.name](bq.search_query.tolist()) for m in models}
    C = retr.build(bq, dq, prune_fn=ranker.prune)
    booster = lgb.Booster(model_file=W.p(ranker_name(args) + '.txt'))
    scores = booster.predict(C[booster.feature_name()])
    top = ranker.top_k(C, scores, 50)
    ids = corp.item_id.values
    answer = pd.DataFrame({'query_id': bq.query_id.values, 'answer': [' '.join(ids[top[i]]) if i in top.index else '' for i in range(len(bq))]})
    # только колонки query_id и answer; переводы строк LF, чтобы к последнему item_id строки не добавлялся символ CR
    answer.to_csv(args.out, index=False, lineterminator=LF)
    log('сохранено', args.out, answer.shape)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data')
    ap.add_argument('--work', default='work')
    ap.add_argument('--out', default='answer.csv')
    ap.add_argument('--threads', type=int, default=6)
    ap.add_argument('--debug', action='store_true', help='быстрая проверка работоспособности кода на маленькой выборке (результат не для сдачи)')
    ap.add_argument('--n-fit', type=int, default=20000, help='сколько запросов fit использовать для обучения ранкера')
    ap.add_argument('--no-adapter', dest='adapter', action='store_false', help='не использовать адаптер e5 (быстрее, чуть хуже)')
    ap.add_argument('--with-quality', action='store_true', help='добавить признаки качества и популярности объявления (на платформе хуже, см. README)')
    args = ap.parse_args()
    global DEBUG
    DEBUG = args.debug
    W = Work(args.work)
    for stage in (stage_prep, stage_split, stage_twotower, stage_e5, stage_adapter, stage_candidates, stage_ranker, stage_bench):
        stage(args, W)
    log('готово')


if __name__ == '__main__':
    main()
