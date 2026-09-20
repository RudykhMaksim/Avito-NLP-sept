"""Проверка answer.csv на соответствие ТЗ. Запуск: python check_answer.py [путь_к_файлу]"""
import re, sys
import pandas as pd

path = sys.argv[1] if len(sys.argv) > 1 else 'answer.csv'
# читаем как строки, чтобы id не превратились в числа
ans = pd.read_csv(path, dtype=str, keep_default_na=False, encoding='utf-8')
bq = pd.read_parquet('data/benchmark_queries.parquet', columns=['query_id'])
corpus = set(pd.read_parquet('data/benchmark_items.parquet', columns=['item_id']).item_id)

errors = []
raw = open(path, 'rb').read()
if raw[:3] == bytes([0xEF, 0xBB, 0xBF]):
    errors.append('файл начинается с BOM (нужен чистый UTF-8)')
if bytes([13]) in raw:
    print('ВНИМАНИЕ: в файле переводы строк CRLF - при наивном разборе по LF в конце последнего item_id останется символ CR; лучше LF')
if list(ans.columns) != ['query_id', 'answer']:
    errors.append(f'колонки должны быть [query_id, answer], а не {list(ans.columns)}')
if len(ans) != len(bq) or set(ans.query_id) != set(bq.query_id):
    errors.append(f'набор query_id не совпадает с benchmark_queries: строк {len(ans)} vs {len(bq)}')
if ans.query_id.duplicated().any():
    errors.append('повторяющиеся query_id')
if not ans.query_id.str.fullmatch(r'[0-9A-Za-z]{16}').all():
    errors.append('query_id не длиной 16 символов')

n_bad_len = n_dup = n_missing = n_fmt = n_empty = 0
sizes = []
for a in ans.answer:
    ids = a.split(' ') if a else []
    sizes.append(len(ids))
    n_empty += (len(ids) == 0)
    n_bad_len += (len(ids) > 50)
    n_dup += (len(set(ids)) != len(ids))
    n_fmt += sum(re.fullmatch(r'[0-9a-f]{16}', x) is None for x in ids)
    n_missing += sum(x not in corpus for x in ids)
for name, v in [('строк с >50 объявлений', n_bad_len), ('строк с повторами', n_dup), ('item_id неверного формата', n_fmt),
                ('item_id отсутствуют в корпусе', n_missing), ('пустых ответов', n_empty)]:
    if v:
        errors.append(f'{name}: {v}')
print(f'строк: {len(ans)}, объявлений в строке: min={min(sizes)} mean={sum(sizes)/len(sizes):.1f} max={max(sizes)}')
if errors:
    print('ОШИБКИ:'); [print(' -', e) for e in errors]; sys.exit(1)
print('OK: формат answer.csv соответствует ТЗ')
