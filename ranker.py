"""LightGBM-ранкер (LambdaRank): выбирает финальные объявления по признакам пар из cg.pipeline.

Обучается на запросах train, отложенных от нейросетей и статистик. Метка равна 1, если объявление кликнули по запросу.
"""
import numpy as np
import pandas as pd
import lightgbm as lgb

# признаки самого объявления: качество (отзывы, рейтинг, цена, флаги контактов) и популярность в click-логе.
# По умолчанию отключены (на платформе хуже), включаются флагом --with-quality.
QUALITY_FEATURES = ['q_reviews', 'q_rating', 'q_price', 'q_price_missing', 'q_phone', 'q_msg', 'q_reviews_pct']
POPULARITY_FEATURES = ['pop_base']

# служебные колонки, не признаки
NON_FEATURES = {'qi', 'ji', 'y', 'key_id'}

PARAMS = dict(objective='lambdarank', metric='ndcg', eval_at=[50], lambdarank_truncation_level=100,
              learning_rate=0.05, num_leaves=63, min_data_in_leaf=200, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
              lambda_l2=10, verbose=-1, seed=0, deterministic=True, force_row_wise=True)


def feature_names(C, exclude=()):
    return [c for c in C.columns if c not in NON_FEATURES and c not in set(exclude)]


def prune(C, rank_limit=150, text_limit=40):
    """Отсекает кандидатов без сильных сторон: ранг по формулам каналов хуже rank_limit, ранг по чистому тексту хуже text_limit
    и нет памяти по логам. Правило одинаково для обучения и инференса, метки не используются."""
    rk = [c for c in C.columns if c.startswith('r_gd_') or c == 'r_g0']
    rt = [c for c in C.columns if c.startswith('r_cos_') or c == 'r_s_sum']
    keep = (C[rk].min(axis=1) <= rank_limit) | (C[rt].min(axis=1) <= text_limit) | (C.mem_n > 0)
    return C[keep].reset_index(drop=True)


def _group_sizes(d):
    return d.groupby('key_id', sort=False).size().values


def train(C, feats, n_rounds=None, es_queries=4000, threads=6, seed=0, params=None):
    """Обучает ранкер. Без n_rounds число деревьев подбирается ранней остановкой на es_queries запросах из C.
    В C нужны колонки key_id (запрос) и y (метка). Возвращает модель и число деревьев."""
    p = {**PARAMS, **(params or {}), 'num_threads': threads, 'seed': seed}
    C = C.sort_values('key_id', kind='stable').reset_index(drop=True)
    if n_rounds is not None:
        d = lgb.Dataset(C[feats], C.y, group=_group_sizes(C))
        return lgb.train(p, d, num_boost_round=n_rounds), n_rounds
    rng = np.random.RandomState(seed)
    keys = C.key_id.unique()
    es = set(rng.choice(keys, min(es_queries, max(20, len(keys) // 5)), replace=False))
    m_es = C.key_id.isin(es).values
    tr, ev = C[~m_es], C[m_es]
    dtr = lgb.Dataset(tr[feats], tr.y, group=_group_sizes(tr))
    dev = lgb.Dataset(ev[feats], ev.y, group=_group_sizes(ev), reference=dtr)
    m = lgb.train(p, dtr, num_boost_round=2000, valid_sets=[dev], callbacks=[lgb.early_stopping(50, verbose=False)])
    return m, m.best_iteration


def recall_at_k(C, scores, npos, ks=(50,)):
    """Средний по запросам Recall@k: попадания в топ-k делятся на число всех положительных запроса (в том числе вне кандидатов).
    npos - число положительных по key_id."""
    d = pd.DataFrame({'key_id': C.key_id.values, 's': scores, 'y': C.y.values}).sort_values(['key_id', 's'], ascending=[True, False])
    d['r'] = d.groupby('key_id').cumcount()
    den = npos.reindex(C.key_id.unique())
    return {k: float((d[(d.r < k) & (d.y == 1)].groupby('key_id').size().reindex(den.index).fillna(0) / den).mean()) for k in ks}


def top_k(C, scores, k=50):
    """Для каждого запроса qi индексы объявлений пула с топ-k оценками."""
    d = pd.DataFrame({'qi': C.qi.values, 'ji': C.ji.values, 's': scores}).sort_values(['qi', 's'], ascending=[True, False], kind='stable')
    return d.groupby('qi').head(k).groupby('qi').ji.apply(np.asarray)
