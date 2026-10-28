"""Проверка п. 6: какие σ-файлы прочитал `fastbt.s9 cache --scope test` в тесте 26.10 10:33.

Ключ кэша (`fastbt.s9.cache_key`) = sha1(code_hash + main.yaml + sha1 каждого σ-источника + test|2018|2021|2026-02-03).
Совпадение с именем `cache_682e4ce9eb.pkl` из журнала однозначно определяет набор прочитанных файлов.
Ничего не пересчитывает и не открывает замок: только хэши файлов.
PYTHONPATH=. python3 scripts/analysis/verif_vhodov_gl4_kesh.py
"""
import hashlib
from pathlib import Path

from fastbt.cache import code_hash

M = Path('/Users/nilysenok/Desktop/Старое/PhD_Vol_trading')
R, O = M / 'data/regen', M / 'data/fastbt/s9'
Y = Path(__file__).resolve().parents[2] / 'fastbt/designs/final/main.yaml'


def key(h1: Path) -> str:
    h = hashlib.sha1(code_hash().encode())
    h.update(Y.read_bytes())
    for lab, p in (('hyb_h1', h1), ('hyb_h5', R / 'sigma_pred_h5_final2.parquet'),
                   ('harj_h1', R / 'sigma_pred_h1_harj_final2.parquet'),
                   ('naive_h1', O / 'sigma_naive_rvw_test.parquet')):
        h.update(lab.encode() + hashlib.sha1(p.read_bytes()).digest())
    h.update(b'test|2018|2021|2026-02-03')
    return h.hexdigest()[:10]


print('code_hash', code_hash())
for name in ('final3_full', 'final3', 'final2', 'FINAL'):
    print(f'{name:12s} -> cache_{key(R / f"sigma_pred_h1_{name}.parquet")}.pkl')
print('в журнале теста: cache_682e4ce9eb.pkl')
