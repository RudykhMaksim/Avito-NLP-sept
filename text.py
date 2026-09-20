"""Нормализация текста и лемматизация (pymorphy3): нижний регистр, ё заменяется на е, слова приводятся к начальной форме."""
import re
from functools import lru_cache

import pymorphy3

_morph = pymorphy3.MorphAnalyzer()
_TOKEN = re.compile(r'[a-zа-я0-9]+')
# служебные слова; остальные частотные слова отсекаются низким IDF в BM25
STOP = set('в во на по для с со и или от до из за под над к ко у о об без при про не что как а но это то же бы ли также'.split())


@lru_cache(maxsize=None)
def lemma(word: str) -> str:
    """Начальная форма слова (числа и латиница не меняются). Результат кэшируется."""
    if word.isdigit() or ('a' <= word[0] <= 'z'):
        return word
    return _morph.parse(word)[0].normal_form.replace('ё', 'е')


def tokens(text: str):
    """Список лемм строки без служебных слов."""
    if not text:
        return []
    return [lemma(w) for w in _TOKEN.findall(text.lower().replace('ё', 'е')) if w not in STOP]


def lemmatize_join(text: str) -> str:
    """Леммы строки через пробел (в таком виде тексты хранятся и индексируются)."""
    return ' '.join(tokens(text))
