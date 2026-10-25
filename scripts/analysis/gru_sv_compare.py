#!/usr/bin/env python3
"""Цель (4): GRU на СВ против бустинга и гибрида второго круга — OOS QLIKE разработочного набора (шаги 2-6, val 2017-2021).
Вход: gibrid_v17.py --boost GRU_sv --legs single (каталог data/regen/GRU_sv_gibrid): V0_XGB = GRU с поправкой Б0.1 (ord/rob/sm/smrob, s² шага t-1)
и без неё; выбор поправки по правилу (n_failed, затем средний QLIKE) среди V0-дизайнов GRU.
Сравнение: финалисты финальной сборки второго круга (data/regen/final2/finalists_preliminary.csv, gibrid_v17 по той же процедуре).
DM: основной способ (дневной ряд средней по тикерам разности потерь, NW с лагом h-1), d = L_сравн. - L_GRU (t < 0 — GRU хуже).
Подтверждающий набор (шаги 7-9) не читается: всё фильтруется по step 2-6.
python3 scripts/analysis/gru_sv_compare.py [--h 1 5 22] [--fin <csv финалистов>]"""
import argparse
from pathlib import Path
import numpy as np, pandas as pd

R = Path('/Users/nilysenok/Desktop/Старое/PhD_Vol_trading/data/regen')
G = R / 'GRU_sv_gibrid'
FIN = R / 'final2/finalists_preliminary.csv'  # финальная сборка второго круга; --fin для окончательного списка
qv = lambda y, p: y / np.clip(p, 1e-10, None) - np.log(y / np.clip(p, 1e-10, None)) - 1


def dm(a, b, h):  # a, b: date,ticker,rv_actual,rv_pred; d = L_a - L_b по дню (среднее по тикерам), NW(h-1)
    m = a.merge(b[['date', 'ticker', 'rv_pred']], on=['date', 'ticker'], suffixes=('_a', '_b'))
    m['d'] = qv(m.rv_actual, m.rv_pred_a) - qv(m.rv_actual, m.rv_pred_b)
    d = m.groupby('date').d.mean().values; n = len(d); u = d - d.mean(); L = h - 1
    v = u @ u / n + 2 * sum((1 - l / (L + 1)) * (u[l:] @ u[:-l]) / n for l in range(1, L + 1))
    return d.mean() / np.sqrt(v / n), len(m)


def gru_designs(h):
    D = pd.read_csv(G / f'designs_val_GRU_sv_h{h}.csv'); F = pd.read_csv(G / f'ranking_GRU_sv_h{h}.csv')
    V = F[F.design.str.match(r'single\|V0_XGB\|')].reset_index(drop=True)  # GRU с каждой оценкой s² и без поправки
    return D, F, V


def gru_pred(h, design):  # прогноз GRU выбранного варианта на шагах 2-6: raw x F_oos шага (s² шага t-1)
    v = pd.read_parquet(R / f'GRU_sv_h{h}_val.parquet'); v = v[(v.step >= 2) & (v.step <= 6) & (v.rv_actual > 1e-12)]
    code = design.split('|')[2]
    if code != 'bez':
        bf = pd.read_csv(G / f'bias_factors_GRU_sv_h{h}.csv'); bf = bf[(bf.comp == 'xgb') & (bf.code == code.replace('B-', ''))]
        v = v.merge(bf[['step', 'F_oos']], on='step'); v['rv_pred'] = v.rv_pred * v.F_oos
    return v[['date', 'ticker', 'step', 'rv_actual', 'rv_pred']]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--h', type=int, nargs='*', default=[1, 5, 22])
    ap.add_argument('--fin', default=None); a = ap.parse_args()
    global FIN
    FIN = Path(a.fin) if a.fin else FIN
    rows = []
    for h in a.h:
        D, F, V = gru_designs(h)
        win = V.iloc[0].design
        yrs = lambda dd, nm: dd[(dd.design == nm) & (dd.step.between(2, 6))].set_index('val_year').q_hyb
        print(f'\n### h={h}: варианты GRU (V0, правило: n_failed, затем средний QLIKE шагов 2-6)')
        print(V[['design', 'failed', 'mean_q']].to_string(index=False))
        g = gru_pred(h, win)
        lines = [('GRU, выбранная поправка ' + win.split('|')[2], yrs(D, win), None)]
        lines.append(('GRU без поправки', yrs(D, 'single|V0_XGB|bez|L0'), None))
        cd = D[(D.design == win) & D.step.between(2, 6)].set_index('val_year')
        lines.append((f'HAR-J с той же s² ({win.split("|")[2]})', cd.q_har, None))
        fz = pd.read_csv(FIN); fz = fz[fz.h == h].sort_values('order')
        for _, r in fz.iterrows():  # финалисты второго круга (final2): бустинг (V0_*) и гибриды (схемы смешивания), фолбэк HAR-J — справочно
            d = R / 'final2/gibrid' / r.cand; Dc = pd.read_csv(d / f'designs_val_{r.cand}_h{h}.csv')
            fc = pd.read_csv(d / f'final_choice_{r.cand}_h{h}.csv').design.iloc[0]; t = None
            if r.design == fc:  # прогнозы сохранены только для выбранного дизайна кандидата -> DM
                pc = pd.read_parquet(d / f'hyb_{r.cand}_final_h{h}_val.parquet'); pc = pc[pc.step.between(2, 6) & (pc.rv_actual > 1e-12)]
                t = dm(pc, g, h)
            role = 'фолбэк HAR-J' if isinstance(r.role, str) and r.role else ('одна компонента V0' if '|V0_' in r.design else 'смешивание с HAR-J')
            lines.append((f'финалист {int(r.order)} ({role}) {r.cand}: {r.design}', yrs(Dc, r.design), t))
        base = lines[0][1].mean()
        for lab, s, t in lines:
            rows.append(dict(h=h, line=lab, **{str(int(y)): round(q, 4) for y, q in s.items()}, mean=round(s.mean(), 4),
                             gru_vs_line_pct=round(100 * (base / s.mean() - 1), 2), dm_t=None if t is None else round(t[0], 2),
                             dm_n=None if t is None else t[1]))
    out = pd.DataFrame(rows)
    out.to_csv(R / 'GRU_sv_compare_dev.csv', index=False)
    print(out.to_string(index=False))


if __name__ == '__main__':
    main()
