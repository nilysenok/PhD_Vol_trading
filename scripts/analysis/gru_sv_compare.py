#!/usr/bin/env python3
"""Цель (4): GRU на СВ против бустинга и гибрида второго круга — OOS QLIKE разработочного набора (шаги 2-6, val 2017-2021).
Вход: gibrid_v17.py --boost GRU_sv --legs single (каталог data/regen/GRU_sv_gibrid): V0_XGB = GRU с поправкой Б0.1 (ord/rob/sm/smrob, s² шага t-1)
и без неё; выбор поправки по правилу (n_failed, затем средний QLIKE) среди V0-дизайнов GRU.
Сравнение: финалисты финальной сборки второго круга (data/regen/final2/finalists_preliminary.csv, gibrid_v17 по той же процедуре).
DM: основной способ (дневной ряд средней по тикерам разности потерь, NW с лагом h-1), d = L_сравн. - L_GRU (t > 0 — GRU лучше);
p_one_sided_gru_better = 1 - Phi(t) (H1: GRU лучше). База — справочный GRU_svoos (правило владельца 26.10, 00:05), GRU_sv — строкой.
Подтверждающий набор (шаги 7-9) не читается: всё фильтруется по step 2-6.
python3 scripts/analysis/gru_sv_compare.py [--h 1 5 22] [--fin <csv финалистов>] [--gru GRU_svoos|GRU_sv]"""
import argparse
from pathlib import Path
import numpy as np, pandas as pd
from scipy.special import ndtr

R = Path('/Users/nilysenok/Desktop/Старое/PhD_Vol_trading/data/regen')
FIN = R / 'final2/finalists_preliminary.csv'  # финальная сборка второго круга; --fin для окончательного списка
qv = lambda y, p: y / np.clip(p, 1e-10, None) - np.log(y / np.clip(p, 1e-10, None)) - 1


def dm(a, b, h):  # a, b: date,ticker,rv_actual,rv_pred; d = L_a - L_b по дню (среднее по тикерам), NW(h-1)
    m = a.merge(b[['date', 'ticker', 'rv_pred']], on=['date', 'ticker'], suffixes=('_a', '_b'))
    m['d'] = qv(m.rv_actual, m.rv_pred_a) - qv(m.rv_actual, m.rv_pred_b)
    d = m.groupby('date').d.mean().values; n = len(d); u = d - d.mean(); L = h - 1
    v = u @ u / n + 2 * sum((1 - l / (L + 1)) * (u[l:] @ u[:-l]) / n for l in range(1, L + 1))
    return d.mean() / np.sqrt(v / n), len(m)


def gru_designs(h, tag):
    G = R / f'{tag}_gibrid'
    D = pd.read_csv(G / f'designs_val_{tag}_h{h}.csv'); F = pd.read_csv(G / f'ranking_{tag}_h{h}.csv')
    V = F[F.design.str.match(r'single\|V0_XGB\|')].reset_index(drop=True)  # GRU с каждой оценкой s² и без поправки
    return D, F, V


def gru_pred(h, design, tag):  # прогноз GRU выбранного варианта на шагах 2-6: raw x F_oos шага (s² шага t-1)
    v = pd.read_parquet(R / f'{tag}_h{h}_val.parquet'); v = v[(v.step >= 2) & (v.step <= 6) & (v.rv_actual > 1e-12)]
    code = design.split('|')[2]
    if code != 'bez':
        bf = pd.read_csv(R / f'{tag}_gibrid' / f'bias_factors_{tag}_h{h}.csv'); bf = bf[(bf.comp == 'xgb') & (bf.code == code.replace('B-', ''))]
        v = v.merge(bf[['step', 'F_oos']], on='step'); v['rv_pred'] = v.rv_pred * v.F_oos
    return v[['date', 'ticker', 'step', 'rv_actual', 'rv_pred']]


yrs = lambda dd, nm: dd[(dd.design == nm) & (dd.step.between(2, 6))].set_index('val_year').q_hyb


def cand_line(h, cand, design, label, g):
    d = R / 'final2/gibrid' / cand; Dc = pd.read_csv(d / f'designs_val_{cand}_h{h}.csv')
    fc = pd.read_csv(d / f'final_choice_{cand}_h{h}.csv').design.iloc[0]; t = None
    if design == fc:  # прогнозы сохранены только для выбранного дизайна кандидата -> DM
        pc = pd.read_parquet(d / f'hyb_{cand}_final_h{h}_val.parquet'); pc = pc[pc.step.between(2, 6) & (pc.rv_actual > 1e-12)]
        t = dm(pc, g, h)
    return (label, yrs(Dc, design), t)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--h', type=int, nargs='*', default=[1, 5, 22])
    ap.add_argument('--fin', default=None)
    ap.add_argument('--gru', default='GRU_svoos', help='база: GRU_svoos (справочный, правило 26.10 00:05) | GRU_sv')
    a = ap.parse_args()
    global FIN
    FIN = Path(a.fin) if a.fin else FIN
    goals = pd.read_csv(R / 'final2/owner_goals_dev.csv')
    rows = []
    for h in a.h:
        D, F, V = gru_designs(h, a.gru)
        win = V.iloc[0].design
        print(f'\n### h={h}: варианты {a.gru} (V0, правило: n_failed, затем средний QLIKE шагов 2-6)')
        print(V[['design', 'failed', 'mean_q']].to_string(index=False))
        g = gru_pred(h, win, a.gru)
        fl = V.iloc[0].failed if isinstance(V.iloc[0].failed, str) else 'все критерии'
        lines = [(f'{a.gru} (база), поправка {win.split("|")[2]} по правилу [{fl}]', yrs(D, win), None),
                 (f'{a.gru} без поправки', yrs(D, 'single|V0_XGB|bez|L0'), None)]
        if a.gru != 'GRU_sv':  # внутривыборочный GRU (подбор на годе оценки) — строкой, в выводы не идёт
            Di, _, Vi = gru_designs(h, 'GRU_sv'); wi = Vi.iloc[0].design
            lines.append((f'GRU_sv внутривыборочный (подбор на годе оценки; справка), {wi.split("|")[2]}', yrs(Di, wi),
                          dm(gru_pred(h, wi, 'GRU_sv'), g, h)))
        cd = D[(D.design == win) & D.step.between(2, 6)].set_index('val_year')
        lines.append((f'HAR-J с поправкой ({win.split("|")[2]})', cd.q_har, None))
        fz = pd.read_csv(FIN); fz = fz[fz.h == h].sort_values('order')
        for _, r in fz.iterrows():  # финалисты второго круга (final2); фолбэк HAR-J — справочно
            role = 'фолбэк HAR-J' if isinstance(r.role, str) and r.role else ('одна компонента V0' if '|V0_' in r.design else 'смешивание с HAR-J')
            lines.append(cand_line(h, r.cand, r.design, f'финалист {int(r.order)} ({role}) {r.cand}: {r.design}', g))
        gr = goals[goals.h == h].iloc[0]
        for col, lab in (('g3_pure_v0', 'V0 (чистый бустинг)'), ('g2_best_single_all', 'лучшая одиночная')):
            c, dsg = gr[col].split(': '); dsg = dsg.split(' ')[0]
            lines.append(cand_line(h, c, dsg, f'{lab} {c}: {dsg}', g))
        base = lines[0][1].mean()
        for lab, s_, t in lines:
            rows.append(dict(h=h, line=lab, **{str(int(y)): round(q, 4) for y, q in s_.items()}, mean=round(s_.mean(), 4),
                             worst_year=int(s_.idxmax()), worst_q=round(s_.max(), 4),
                             gru_vs_line_pct=round(100 * (base / s_.mean() - 1), 2), dm_t=None if t is None else round(t[0], 2),
                             p_one_sided_gru_better=None if t is None else round(float(1 - ndtr(t[0])), 4),
                             dm_n=None if t is None else t[1]))
    out = pd.DataFrame(rows)
    out.to_csv(R / f'{a.gru}_compare_dev.csv', index=False)
    pd.set_option('display.width', 250); pd.set_option('display.max_colwidth', 75)
    print(out.to_string(index=False))


if __name__ == '__main__':
    main()
