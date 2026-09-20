"""Приоры на пару (запрос, объявление): вероятность локации, вид услуги из фильтра, расстояние.

Пользователь почти всегда выбирает объявление рядом с локацией поиска, но id локаций поиска и объявления напрямую не совпадают
(поиск по региону, стране, крупному городу). Поэтому по кликам выучивается вероятность локации объявления при заданной локации
поиска: она покрывает 97.5% кликов против 83% при точном совпадении id.
"""
import re

import numpy as np
import pandas as pd

# вид услуги: значение идёт до следующего ключа параметров
_VID_RE = re.compile(r'Вид услуги (.*?)(?= Тип услуги| Онлайн-запись| Где вы| Ваши клиенты| Кто оказывает| Место оказания| Тип стоимости|$)')


def vid_of(text: str) -> str:
    """Значение вида услуги из параметров объявления или фильтров поиска (пустая строка, если нет)."""
    m = _VID_RE.search(text or '')
    return m.group(1).strip() if m else ''


class LocAffinity:
    """Вероятность локации объявления при заданной локации поиска, посчитанная по кликам."""

    SHIFT = 1 << 24

    def __init__(self, search_loc, item_loc):
        df = pd.DataFrame({'s': np.asarray(search_loc, dtype=np.int64), 'i': np.asarray(item_loc, dtype=np.int64)})
        c = df.groupby(['s', 'i']).size().rename('c').reset_index()
        c['p'] = c.c / c.groupby('s').c.transform('sum')
        key = c.s.values * self.SHIFT + c.i.values
        order = np.argsort(key)
        self.key, self.p = key[order], c.p.values[order].astype(np.float32)

    def get(self, s, i):
        """Вероятность P(i|s) для массивов s и i; при точном совпадении id не меньше 0.9."""
        s = np.asarray(s, dtype=np.int64)
        i = np.asarray(i, dtype=np.int64)
        k = s * self.SHIFT + i
        pos = np.searchsorted(self.key, k)
        pos[pos >= len(self.key)] = 0
        p = np.where(self.key[pos] == k, self.p[pos], 0.0).astype(np.float32)
        return np.where(s == i, np.maximum(p, 0.9), p)


def km_dist(lat1, lon1, lat2, lon2):
    """Приблизительное расстояние в километрах (для сравнения «рядом или далеко» этого достаточно)."""
    x = (lon2 - lon1) * np.cos(np.radians((lat1 + lat2) / 2))
    y = lat2 - lat1
    return 111.0 * np.sqrt(x * x + y * y)
