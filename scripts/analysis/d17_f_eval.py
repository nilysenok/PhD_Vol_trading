#!/usr/bin/env python3
"""Д17 (агент F): оценка вариантов по правилам третьего круга -> docs/otchety/d17_f/*.csv и таблица в stdout.

Ноги как у финалиста: d812 и D12_h22 — seeds3 (среднее RV по seed 42/43/44 XGB); D12_h5 и F2m_h5 — wrap2 (среднее RV XGB и LGBM),
по каждому seed отдельно и среднее по seed. Поправка Б0.1 — код финалиста (d812, D12_h5: sm; F2m_h5, D12_h22: smrob),
s² по val-остаткам до шага k (l2_lib.s2_est); шаг 1 (2016) — s² по своим же остаткам (вне выборки нечем), только для критериев.
Отбор: средний QLIKE шагов 2–5 (2017–2020); проверка: шаг 6 (2021); критерии c1–c3 на 2016–2021 (как gibrid_designs.flags:
c1 — не хуже HAR-J с той же поправкой > 2% ни в один год; c2 — не хуже лучшей одиночной модели (HAR-J и одиночные бустинги
seed 42, кроме своей ноги) > 2%; c3 — худшее отношение к HAR-J меньше, чем у лучшего одиночного бустинга); c4 для V0/L0 выполнен.
Порог приёма: улучшение OOS 2017–2020 ≥ 1% по среднему трёх seed и знак улучшения во всех трёх seed, 2021 — не хуже > 2%.
HAR-J — l2_lib.har_ln (МНК на ln, компоненты harj_cols) на обучающем окне шага, val шага.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import d17_f as F  # noqa: E402  (L2_FEAT = v3)
L = F.L
OUT = Path(__file__).resolve().parents[2] / 'docs/otchety/d17_f'
CODE = {'d812': 'sm', 'D12_h5': 'sm', 'F2m_h5': 'smrob', 'D12_h22': 'smrob', 'V0_h1': 'sm', 'V0_h5': 'smrob', 'V0_h22': 'smrob'}
LEG = {'d812': 'seeds3', 'D12_h5': 'wrap2', 'F2m_h5': 'wrap2', 'D12_h22': 'seeds3', 'V0_h1': 'wrap2', 'V0_h5': 'wrap2', 'V0_h22': 'wrap2'}


def corrected(val, code):
    """val: step, rv_actual, rv_pred (шаги 1–6) -> {step: QLIKE с Б0.1}."""
    v = val[val.rv_actual > 1e-12]
    Gs = {st: g for st, g in v.groupby('step')}
    E = {st: np.log(g.rv_actual.values + 1e-10) - np.log(g.rv_pred.values) for st, g in Gs.items()}
    out = {}
    for st, g in Gs.items():
        s2 = L.s2_est(code, E, st - 1 if st > 1 else 1)
        out[st] = L.qlike(g.rv_actual.values, g.rv_pred.values * np.exp(s2 / 2))
    return out


_HAR = {}


def har(design):
    d = F.DESIGNS[design]
    if d['h'] in _HAR:
        return _HAR[d['h']]
    F.setup([design], ['base'])
    A = F.G['A'][design]
    hc = L.harj_cols(A)
    rows = []
    for st in range(1, 7):
        tp, va, _, vy, _ = L.split(A, d['h'], st)
        p = L.har_ln(tp, [va], d['h'], hc)[0]
        rows.append(pd.DataFrame({'date': va.date.values, 'ticker': va.ticker.values, 'step': st,
                                  'rv_actual': va[L.tcol(d['h'])].values, 'rv_pred': np.exp(p)}))
    _HAR[d['h']] = pd.concat(rows, ignore_index=True)
    return _HAR[d['h']]


def load(h, design, variant, model):
    p = L.REGEN / f'D17F_h{h}_{design}_{variant}_{model}_val.parquet'
    return pd.read_parquet(p) if p.exists() else None


def leg(h, design, variant, seed=None):
    """Прогноз ноги финалиста; seed=None — seeds3/wrap2 по всем seed."""
    ms = F.DESIGNS[design]['models']
    V = {m: load(h, design, variant, m) for m in ms}
    if any(v is None for v in V.values()):
        return None
    if seed is not None:
        V = {m: v[v.seed == seed] for m, v in V.items()}
    k = ['date', 'ticker', 'step']
    cat = pd.concat(V.values(), ignore_index=True)
    return cat.groupby(k, as_index=False).agg(rv_actual=('rv_actual', 'first'), rv_pred=('rv_pred', 'mean'))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows, years = [], []
    for design, d in F.DESIGNS.items():
        h, code = d['h'], CODE[design]
        if load(h, design, 'base', d['models'][0]) is None:
            continue
        qh = corrected(har(design), code)
        singles = {}
        for m in d['models']:
            v = load(h, design, 'base', m)
            singles[m] = corrected(v[v.seed == 42], code)
        own = 'xgb' if LEG[design] == 'seeds3' else None
        res = {}
        for variant in F.VARIANTS + ('rrv_gk+iterhar', 'rrv_pk+iterhar'):
            if leg(h, design, variant) is None:
                continue
            per_seed = {s: corrected(leg(h, design, variant, s), code) for s in (42, 43, 44)}
            allq = corrected(leg(h, design, variant), code)
            res[variant] = (allq, per_seed)
        base_all, base_seed = res['base']
        for variant, (allq, per_seed) in res.items():
            oos = np.mean([allq[s] for s in range(2, 6)])
            boos = np.mean([base_all[s] for s in range(2, 6)])
            ds = [np.mean([per_seed[sd][s] for s in range(2, 6)]) / np.mean([base_seed[sd][s] for s in range(2, 6)]) - 1 for sd in (42, 43, 44)]
            d21 = allq[6] / base_all[6] - 1
            hy = np.array([allq[s] for s in range(1, 7)])
            hv = np.array([qh[s] for s in range(1, 7)])
            oth = [np.array([singles[m][s] for s in range(1, 7)]) for m in singles if m != own]
            smin = np.minimum.reduce([hv] + oth)
            ml = [(np.array([singles[m][s] for s in range(1, 7)]) / hv).max() for m in singles]
            c1, c2, c3 = bool((hy <= 1.02 * hv).all()), bool((hy <= 1.02 * smin).all()), bool((hy / hv).max() < min(ml))
            acc = variant != 'base' and np.mean(ds) <= -0.01 and all(x < 0 for x in ds) and d21 <= 0.02
            rows.append(dict(h=h, design=design, leg=f"{LEG[design]}/{code}", variant=variant, oos_2017_2020=oos,
                             d_oos_pct=100 * (oos / boos - 1), d_seed42=100 * ds[0], d_seed43=100 * ds[1], d_seed44=100 * ds[2],
                             q_2021=allq[6], d_2021_pct=100 * d21, vs_harj_oos_pct=100 * (oos / np.mean([qh[s] for s in range(2, 6)]) - 1),
                             c1=c1, c2=c2, c3=c3, accepted=bool(acc)))
            for s in range(1, 7):
                years.append(dict(h=h, design=design, variant=variant, year=2015 + s, q=allq[s], q_harj=qh[s]))
    T = pd.DataFrame(rows)
    T.to_csv(OUT / 'itog.csv', index=False, float_format='%.5f')
    pd.DataFrame(years).to_csv(OUT / 'po_godam.csv', index=False, float_format='%.5f')
    pd.set_option('display.width', 250)
    print(T.round(4).to_string(index=False))


if __name__ == '__main__':
    main()
