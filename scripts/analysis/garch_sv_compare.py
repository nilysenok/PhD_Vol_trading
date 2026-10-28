#!/usr/bin/env python3
"""Справочный GJR-GARCH на СВ: OOS QLIKE разработочного набора (шаги 2-6, год оценки 2017-2021) против HAR-J с поправкой Б0.1, финалистов
второго круга, V0, лучшей одиночной и справочного GRU (GRU_svoos). Процедура и функции - gru_sv_compare.py.
Вход: gibrid_v17.py --boost GARCH_svgjr|GARCH_svgarch --legs single (V0_XGB = модель с каждой из оценок s² шага t-1 и без поправки; выбор по правилу).
DM: основной способ (дневной ряд средней по тикерам разности потерь, NW лаг h-1), d = L_сравн. - L_GJR (t > 0 - GJR лучше), одностороннее p = 1 - Phi(t).
Подтверждающий набор (шаги 7-10) и тест не читаются: всё фильтруется по step 2-6.
python3 scripts/analysis/garch_sv_compare.py"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.special import ndtr
sys.path.insert(0, str(Path(__file__).resolve().parent))
import gru_sv_compare as C
import gibrid_har as GH

R, TAG, REF = C.R, 'GARCH_svgjr', 'GARCH_svgarch'


def har_pred(h, code, A, harj, hv):  # HAR-J на шагах 2-6 x F_oos шага (s² шага t-1) по оценке code
    v = hv[hv.step.between(2, 6) & (hv.rv_actual > 1e-12)]
    if code == 'bez': return v
    bf = pd.read_csv(R / f'{TAG}_gibrid' / f'bias_factors_{TAG}_h{h}.csv'); bf = bf[(bf.comp == 'har') & (bf.code == code.replace('B-', ''))]
    v = v.merge(bf[['step', 'F_oos']], on='step'); return v.assign(rv_pred=v.rv_pred * v.F_oos)[['date', 'ticker', 'step', 'rv_actual', 'rv_pred']]


def main():
    goals = pd.read_csv(R / 'final2/owner_goals_dev.csv'); fz_all = pd.read_csv(C.FIN)
    A, harj = GH.load_pit(R / 'features_pit_v3.parquet')
    rows = []
    for h in (1, 5, 22):
        D, F, V = C.gru_designs(h, TAG); win = V.iloc[0].design; code = win.split('|')[2]
        print(f'\n### h={h}: варианты {TAG} (V0, правило: n_failed, затем средний QLIKE шагов 2-6)'); print(V[['design', 'failed', 'mean_q']].to_string(index=False))
        g = C.gru_pred(h, win, TAG)
        Dr, _, Vr = C.gru_designs(h, REF); wr = Vr.iloc[0].design
        Dg, _, Vg = C.gru_designs(h, 'GRU_svoos'); wg = Vg.iloc[0].design
        hc = wg.split('|')[2]; hv, _, _ = GH.har_forecasts(h, A, harj)
        cd = D[(D.design == win) & D.step.between(2, 6)].set_index('val_year')
        lines = [(f'GJR-GARCH (база), поправка {code}', C.yrs(D, win), None),
                 ('GJR-GARCH без поправки', C.yrs(D, 'single|V0_XGB|bez|L0'), None),
                 (f'GARCH(1,1) справочно, {wr.split("|")[2]}', C.yrs(Dr, wr), C.dm(C.gru_pred(h, wr, REF), g, h)),
                 (f'HAR-J с поправкой ({hc})', Dg[(Dg.design == wg) & Dg.step.between(2, 6)].set_index('val_year').q_har,
                  C.dm(har_pred(h, hc, A, harj, hv), g, h)),  # оценка s² - та же, что у справочного GRU (как в gru_sv)
                 (f'справочный GRU (GRU_svoos), {wg.split("|")[2]}', C.yrs(Dg, wg), C.dm(C.gru_pred(h, wg, 'GRU_svoos'), g, h))]
        fz = fz_all[fz_all.h == h].sort_values('order')
        for _, r in fz.iterrows():
            role = 'фолбэк HAR-J' if isinstance(r.role, str) and r.role else ('одна компонента V0' if '|V0_' in r.design else 'смешивание с HAR-J')
            lines.append(C.cand_line(h, r.cand, r.design, f'финалист {int(r.order)} ({role}) {r.cand}: {r.design}', g))
        gr = goals[goals.h == h].iloc[0]
        for col, lab in (('g3_pure_v0', 'V0 (чистый бустинг)'), ('g2_best_single_all', 'лучшая одиночная')):
            c, dsg = gr[col].split(': '); dsg = dsg.split(' ')[0]
            lines.append(C.cand_line(h, c, dsg, f'{lab} {c}: {dsg}', g))
        base = lines[0][1].mean()
        for lab, s_, t in lines:
            rows.append(dict(h=h, line=lab, **{str(int(y)): round(q, 4) for y, q in s_.items()}, mean=round(s_.mean(), 4),
                             worst_year=int(s_.idxmax()), worst_q=round(s_.max(), 4), gjr_vs_line_pct=round(100 * (base / s_.mean() - 1), 2),
                             dm_t=None if t is None else round(t[0], 2), p_one_sided_gjr_better=None if t is None else round(float(1 - ndtr(t[0])), 4),
                             dm_n=None if t is None else t[1]))
    out = pd.DataFrame(rows); out.to_csv(R / 'GARCH_sv_compare_dev.csv', index=False)
    pd.set_option('display.width', 250); pd.set_option('display.max_colwidth', 80); print(out.to_string(index=False))


if __name__ == '__main__':
    main()
