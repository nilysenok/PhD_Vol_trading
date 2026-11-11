"""S-T (Ж-256 п. 4): диагностические срезы Δ и совпадение исключённых дат — чистые функции по массивам потерь; ничего не пишут.
Основной результат — обычная QLIKE на всех 2198 общих датах; срезы («без t⁺ = 2022-02-24», «без 2022 года», симметричная обрезка 22 + 22 по d_t = ℓ(кандидат) − ℓ(база)) — ТОЛЬКО диагностика.
Публичное API:
- `PAIRS` — 8 сравнений (кандидат, контрольная модель); `KTRIM` = 22; `trim_keep(d, k)` — маска оставшихся дат (как `r_eval_tab.trim_delta`: стабильная сортировка по d_t);
- `slices(tp, ls)`: по паре и срезу — n, n_removed, знаменатель (QLIKE базы на оставшихся датах), QLIKE кандидата на них, Δ = 100·Σ d_t / Σ ℓ_база по оставшимся датам;
- `excluded(tp, ls, h, b)` — исключённые при обрезке даты (22 лучших и 22 худших); `trimsets(tp, ls)`: по каждой паре сравнений число общих исключённых дат из 44 и Жаккар; перечень дат для S-RCK к W5-G и к X1."""
from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd

from models_v2.s18 import r_eval_tab as RT

KTRIM = RT.KTRIM
PAIRS = tuple(("S-RCK", b) for b in ("W5-G", "X1", "K2", "K3", "S-CK")) + (("S-RA2", "W5-G"), ("S-RA2", "S-A2"), ("S-A2", "W5-G"))
LISTED = (("S-RCK", "W5-G"), ("S-RCK", "X1"))
SLICES = ("основной: все общие даты", "без t⁺ = 2022-02-24", "без 2022 года", "симметричная обрезка %d + %d дат" % (KTRIM, KTRIM))


def trim_order(d, k: int = KTRIM) -> tuple:
    o = np.argsort(np.asarray(d, float), kind="stable")
    return o[:k], o[len(d) - k:]


def trim_keep(d, k: int = KTRIM) -> np.ndarray:
    keep = np.ones(len(d), bool)
    lo, hi = trim_order(d, k)
    keep[lo], keep[hi] = False, False
    return keep


def masks(tp, d) -> list:
    tp = pd.DatetimeIndex(tp)
    return [np.ones(len(tp), bool), (tp != pd.Timestamp("2022-02-24")), (tp.year.to_numpy() != 2022), trim_keep(d)]


def slices(tp, ls: dict) -> pd.DataFrame:
    rows = []
    for h, b in PAIRS:
        d = ls[h] - ls[b]
        for i, (name, m) in enumerate(zip(SLICES, masks(tp, d))):
            qb, qh = float(ls[b][m].mean()), float(ls[h][m].mean())
            rows.append(dict(candidate=h, base=b, slice=name, kind="основной" if i == 0 else "диагностика", n=int(m.sum()), n_removed=int((~m).sum()),
                             denominator="QLIKE базы %s на тех же оставшихся датах" % b, q_base=qb, q_candidate=qh, delta_pct=100.0 * float(d[m].sum()) / float(ls[b][m].sum())))
    return pd.DataFrame(rows)


def excluded(tp, ls: dict, h: str, b: str) -> tuple:
    tp, d = pd.DatetimeIndex(tp), ls[h] - ls[b]
    lo, hi = trim_order(d)
    return tuple(sorted(str(tp[i].date()) for i in lo)), tuple(sorted(str(tp[i].date()) for i in hi))


def trimsets(tp, ls: dict) -> pd.DataFrame:
    ex = {p: set(sum(excluded(tp, ls, *p), ())) for p in PAIRS}
    rows = []
    for p, q in combinations(PAIRS, 2):
        c = len(ex[p] & ex[q])
        rows.append(dict(kind="пара сравнений", a="%s | %s" % p, b="%s | %s" % q, n_common=c, n_each=len(ex[p]), jaccard=c / len(ex[p] | ex[q]), side="", dates=""))
    for p in LISTED:
        lo, hi = excluded(tp, ls, *p)
        for side, ds in (("22 лучших для кандидата (наименьшие d_t)", lo), ("22 худших для кандидата (наибольшие d_t)", hi)):
            rows.append(dict(kind="перечень дат", a="%s | %s" % p, b="", n_common=np.nan, n_each=len(ds), jaccard=np.nan, side=side, dates="; ".join(ds)))
    return pd.DataFrame(rows)
