"""К-wf6 (предрегистрация v2, §4 «PBO»): CSCV на общей матрице «точки × блоки» по суммам кварталов wf. Данных не читает.

Вход — суммы ledger по кварталам одной стратегии: S[P, 45] (сумма дневной избыточной), Q2[P, 45] (сумма квадратов),
nb[45] (число баров квартала). 45 кварталов 2015-Q1…2026-Q1 → S = 10 блоков из ЦЕЛЫХ кварталов; границы объявлены
ниже (`BLOKI_KV`), до счёта не подбираются. Шарп блока и объединения блоков — из сумм: mean/sd (ddof = 1)·√252,
то же определение, что `stats.window` / `sr_dnevnoy` (Д7). Выход позиции на границе блоков внутри ряда не взимается
(позиция идёт дальше); платится только там, где ряд кончается (поля F_L/F_X окна) — в суммы квартала это уже входит.

Правило выбора внутри разбиения — аргумент `vybor` (правило процедуры, Ж-080; не argmax; лист 2: `pravilo_procedury`).
`vybor(sh_is, blocks_is) -> список индексов членов | None` получает Шарп точек ТОЛЬКО на блоках in-sample.
Оценка выбранного на out-of-sample — среднее Шарпов членов на блоках out-of-sample; ранг — доля точек с Шарпом ниже
(ансамбль равного веса top-10: ковариаций из сумм нет, поэтому берётся среднее членов — ↺ ниже в описании PR).
logit = ln(w/(1−w)); PBO = доля разбиений с logit ≤ 0 (как `pbo_pravilo.cscv`).
"""
from __future__ import annotations

import itertools
import math
from typing import Callable

import numpy as np

NQ = 45                                               # 2015-Q1 … 2026-Q1
BLOKI_KV = (5, 4, 5, 4, 5, 4, 5, 4, 5, 4)             # кварталов в блоке: 2015-Q1…2016-Q1 | …; сумма = 45
S_BLOKOV = len(BLOKI_KV)
EPS_SD = 1e-12
assert sum(BLOKI_KV) == NQ


def granicy() -> list:
    """[(первый квартал, за последним)] каждого блока."""
    e = np.concatenate([[0], np.cumsum(BLOKI_KV)])
    return [(int(e[i]), int(e[i + 1])) for i in range(S_BLOKOV)]


def bloki(S: np.ndarray, Q2: np.ndarray, nb: np.ndarray) -> tuple:
    """Суммы кварталов → суммы блоков (Sb[P, 10], Qb[P, 10], nbb[10]). Не 45 кварталов или блок без баров — отказ."""
    S, Q2, nb = np.asarray(S, float), np.asarray(Q2, float), np.asarray(nb, float)
    if S.ndim != 2 or S.shape != Q2.shape or S.shape[1] != NQ or nb.shape != (NQ,):
        raise ValueError(f"нужны S, Q2 [точки, {NQ}] и nb [{NQ}] (2015-Q1…2026-Q1)")
    if not (np.isfinite(S).all() and np.isfinite(Q2).all() and np.isfinite(nb).all()) or (nb < 0).any():
        raise ValueError("nan/inf в суммах или отрицательное число баров")
    g = granicy()
    nbb = np.array([nb[a:b].sum() for a, b in g])
    if (nbb < 2).any():
        raise ValueError("в блоке < 2 баров: Шарп блока не определён")
    return (np.stack([S[:, a:b].sum(axis=1) for a, b in g], axis=1),
            np.stack([Q2[:, a:b].sum(axis=1) for a, b in g], axis=1), nbb)


def sharpe_iz_summ(S: np.ndarray, Q2: np.ndarray, n: float) -> np.ndarray:
    """mean/sd (ddof = 1)·√252 по суммам; sd ≤ 1e-12 → 0 (как `kalibr_v5.sharp_okna`)."""
    var = np.maximum(Q2 - S * S / n, 0.0) / (n - 1.0)
    sd = np.sqrt(var)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(sd > EPS_SD, (S / n) / sd, 0.0) * math.sqrt(252.0)


def razbieniya():
    """Все C(10, 5) = 252 разбиения: (блоки in-sample, блоки out-of-sample)."""
    for tr in itertools.combinations(range(S_BLOKOV), S_BLOKOV // 2):
        yield tr, tuple(b for b in range(S_BLOKOV) if b not in tr)


def vybor_argmax(sh_is: np.ndarray, blocks_is: tuple):
    """Справочное и МУТАЦИЯ № 42 (Ж-080): argmax Шарпа in-sample — НЕ правило процедуры; в процедуре не используется."""
    return [int(np.argmax(sh_is))]


def cscv_wf(Sb: np.ndarray, Qb: np.ndarray, nbb: np.ndarray, vybor: Callable) -> dict:
    """PBO по 252 разбиениям. Меньше 2 точек — pbo = nan. vybor вернул None — разбиение пропущено (n_propusk)."""
    P = Sb.shape[0]
    if P < 2:
        return dict(pbo=float("nan"), n=0, n_propusk=0, logits=[])
    logits, skipped = [], 0
    for tr, te in razbieniya():
        tr_i, te_i = list(tr), list(te)
        sh_is = sharpe_iz_summ(Sb[:, tr_i].sum(axis=1), Qb[:, tr_i].sum(axis=1), nbb[tr_i].sum())
        chleny = vybor(sh_is, tr)
        if not chleny:
            skipped += 1
            continue
        sh_oos = sharpe_iz_summ(Sb[:, te_i].sum(axis=1), Qb[:, te_i].sum(axis=1), nbb[te_i].sum())
        a = float(np.mean(sh_oos[list(chleny)]))
        w = min(max(float(np.sum(sh_oos < a)) / (P - 1), 1e-6), 1 - 1e-6)
        logits.append(math.log(w / (1 - w)))
    pbo = sum(1 for x in logits if x <= 0) / len(logits) if logits else float("nan")
    return dict(pbo=pbo, n=len(logits), n_propusk=skipped, logits=logits)


def pbo_wf(S: np.ndarray, Q2: np.ndarray, nb: np.ndarray, vybor: Callable) -> dict:
    """Сквозной вход: суммы кварталов → блоки → CSCV."""
    return cscv_wf(*bloki(S, Q2, nb), vybor)
