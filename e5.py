"""Модель multilingual-e5-small (open-source, лицензия MIT) и адаптер над её эмбеддингами.

Модель запускается локально на CPU. Веса скачиваются один раз (huggingface-cli download intfloat/multilingual-e5-small),
дальше читаются из локального кэша.
Текст объявления для модели: заголовок, вид услуги, тип услуги. У запроса префикс "query: ", у объявления "passage: ".
Адаптер - две небольшие MLP-башни поверх замороженных эмбеддингов, дообученные на кликах. Старт совпадает с исходной моделью.
"""
import os
import re

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .priors import vid_of

MODEL_NAME = 'intfloat/multilingual-e5-small'
_TIP_RE = re.compile(r'Тип услуги (.*?)(?= Вид услуги| Место оказания| Тип стоимости| Онлайн-запись|$)')


def item_texts(titles, params):
    """Тексты объявлений для модели: заголовок, вид услуги, тип услуги."""
    out = []
    for t, p in zip(titles, params):
        p = p or ''
        m = _TIP_RE.search(p)
        out.append(f'passage: {t}. {vid_of(p)}. {m.group(1).strip() if m else ""}')
    return out


def load_encoder(threads=6, max_len=40):
    os.environ.setdefault('HF_HUB_OFFLINE', '1')                # только локальный кэш, без сети
    from sentence_transformers import SentenceTransformer
    torch.set_num_threads(threads)
    m = SentenceTransformer(MODEL_NAME, device='cpu')
    m.max_seq_length = max_len
    return m


def encode(model, texts, batch_size=128):
    """Нормированные эмбеддинги в float16 (кэш вдвое меньше)."""
    return model.encode(list(texts), batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False).astype(np.float16)


class Adapter(nn.Module):
    D = 384

    def __init__(self, n_mc, hidden=768, temp=0.03):
        super().__init__()
        d = self.D
        self.q_mlp = nn.Sequential(nn.Linear(d, hidden), nn.GELU(), nn.Linear(hidden, d))
        self.i_mlp = nn.Sequential(nn.Linear(d, hidden), nn.GELU(), nn.Linear(hidden, d))
        for mlp in (self.q_mlp, self.i_mlp):
            nn.init.zeros_(mlp[2].weight)
            nn.init.zeros_(mlp[2].bias)                          # старт совпадает с исходной моделью
        self.mc = nn.Embedding(n_mc + 1, d)
        nn.init.zeros_(self.mc.weight)
        self.head = nn.Linear(d, n_mc + 1)                       # микрокатегория по вектору запроса
        self.log_t = nn.Parameter(torch.tensor(np.log(1 / temp), dtype=torch.float32))

    def enc_q(self, x):
        return x + self.q_mlp(x)

    def enc_i(self, x, mc):
        return x + self.i_mlp(x) + self.mc(mc)


def train_adapter(eq, ei, mc_pair, qidx, iidx, qid, iid, n_mc, epochs=8, lr=5e-4, bs=4096, hidden=768, temp=0.03, seed=0, threads=6, log=print):
    """eq и ei - эмбеддинги e5 уникальных запросов и объявлений train, qidx и iidx - строки для каждой пары,
    qid и iid - идентификаторы для поиска ложных негативов, mc_pair - микрокатегория объявления пары."""
    import time
    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)
    torch.set_num_threads(threads)
    m = Adapter(n_mc, hidden, temp)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    n = len(qidx)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * (n // bs))
    t0 = time.time()
    for ep in range(epochs):
        perm = rng.permutation(n)
        tot, nb = 0.0, 0
        for a in range(0, n - bs + 1, bs):
            r = perm[a:a + bs]
            q = F.normalize(m.enc_q(eq[qidx[r]]), dim=-1)
            i = F.normalize(m.enc_i(ei[iidx[r]], torch.from_numpy(mc_pair[r])), dim=-1)
            lg = q @ i.T * m.log_t.exp().clamp(max=200)
            qi_, ii_ = torch.from_numpy(qid[r]), torch.from_numpy(iid[r])
            fn = ((qi_[:, None] == qi_[None, :]) | (ii_[:, None] == ii_[None, :])) & ~torch.eye(len(r), dtype=torch.bool)
            lg = lg.masked_fill(fn, -1e4)
            tgt = torch.arange(len(r))
            loss = 0.5 * (F.cross_entropy(lg, tgt) + F.cross_entropy(lg.T, tgt))
            lmc = F.cross_entropy(m.head(q), torch.from_numpy(mc_pair[r]))
            opt.zero_grad()
            (loss + 0.3 * lmc).backward()
            opt.step(); sched.step()
            tot += loss.item(); nb += 1
        log(f'  adapter epoch {ep+1}/{epochs} nce={tot/nb:.4f} mc={lmc.item():.3f} {time.time()-t0:.0f}s')
    return m.eval()


@torch.no_grad()
def adapt_items(m, e5_items, mc):
    return F.normalize(m.enc_i(torch.from_numpy(e5_items.astype(np.float32)), torch.from_numpy(mc)), dim=-1).numpy()


@torch.no_grad()
def adapt_queries(m, e5_queries):
    """Векторы запросов и log-вероятности микрокатегорий."""
    e = F.normalize(m.enc_q(torch.from_numpy(e5_queries.astype(np.float32))), dim=-1)
    return e.numpy(), F.log_softmax(m.head(e), dim=-1).numpy()
