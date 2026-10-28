#!/usr/bin/env python3
"""Финальная сборка второго круга, шаг 3: сводка правила выбора по кандидатам каждого h (разработочный набор: val 2017-2021, шаги 2-6, OOS).
Только разработочный набор: подтверждающий (шаги 7-9) не читается и не печатается (в ranking_* его нет).
Рейтинг — правило гибрида: число нарушенных критериев 1-4 (gibrid_v17, эталон — модели с той же s²), затем средний QLIKE.
Кандидат представлен своим лучшим дизайном по правилу (по всему пространству гибрида, ноги single/top5/seeds3/wrap2).
Финалисты (предварительные, не больше трёх): h=5 и h=1 — лучшие дизайны трёх первых кандидатов; h=1 — tie-break владельца:
гибрид, отстающий от V0 не больше чем на 1%, ставится перед V0; h=22 (один кандидат) — лучшие по правилу в разных семействах, до трёх.
Фолбэк h=22 — HAR-J с поправкой, s² по правилу (лучший дизайн single|V0_HAR|B-*|L0).
Цели владельца (4): (1) гибрид лучше HAR-J с поправкой; (2) не хуже ни одной модели; (3) чистый V0 не хуже HAR-J с поправкой; (4) GRU — нет данных.
python3 scripts/analysis/final2_select.py"""
import json
from pathlib import Path
import numpy as np, pandas as pd

R = Path('/Users/nilysenok/Desktop/Старое/PhD_Vol_trading/data/regen/final2')
CANDS = {1: ['D12_h1_A_d812', 'D12_h1_B_b0.25', 'D12_h1_V0'],
         5: ['F2_h5', 'F2m_h5', 'D12_h5', 'L2v3_h5', 'Bv3'],
         22: ['D12_h22', 'Bv3']}
PURE = {'D12_h1_V0', 'Bv3'}  # чистый бустинг (V0 в смысле владельца)
CR = ('c1', 'c2', 'c3', 'c4')


def rank(b, h):
    p = R / 'gibrid' / b / f'ranking_{b}_h{h}.csv'
    if not p.exists():
        return None
    F = pd.read_csv(p)
    F['cand'] = b
    F['rank_in_cand'] = range(1, len(F) + 1)
    return F


def per_year(b, h, design):
    D = pd.read_csv(R / 'gibrid' / b / f'designs_val_{b}_h{h}.csv')
    g = D[(D.design == design) & (D.step >= 2) & (D.step <= 6)].sort_values('step')
    return g


def main():
    out, fin_rows, goals = [], [], []
    lines = []
    for h, cands in CANDS.items():
        Fs = [f for f in (rank(b, h) for b in cands) if f is not None]
        if not Fs:
            continue
        A = pd.concat(Fs, ignore_index=True)
        best = A.sort_values(['n_failed', 'mean_q'], kind='stable').groupby('cand', sort=False).head(1).reset_index(drop=True)
        best['pure_v0'] = best.cand.isin(PURE)
        order = best.cand.tolist()
        if h == 1 and best.pure_v0.any():
            v0 = best[best.pure_v0].iloc[0]
            hy = best[~best.pure_v0]
            ok = hy[(hy.n_failed <= v0.n_failed) & (hy.mean_q <= v0.mean_q * 1.01)]
            order = ok.cand.tolist() + [v0.cand] + [c for c in hy.cand if c not in set(ok.cand)]
        best = best.set_index('cand').loc[order].reset_index()
        best['h'] = h
        out.append(best)
        if len(cands) == 1 or h == 22:
            F = [f for f in Fs if f.cand.iloc[0] == cands[0]][0]
            fam = F.groupby('family', sort=False).head(1).head(3)
            fins = [(cands[0], d) for d in fam.design]
        else:
            fins = [(r.cand, r.design) for r in best.head(3).itertuples()]
        for i, (c, d) in enumerate(fins, 1):
            r = A[(A.cand == c) & (A.design == d)].iloc[0]
            fin_rows.append(dict(h=h, order=i, cand=c, design=d, failed=r.failed if isinstance(r.failed, str) else '', mean_q=r.mean_q,
                                 rank_in_cand=int(r.rank_in_cand)))
        if h == 22:
            F = [f for f in Fs if f.cand.iloc[0] == 'D12_h22'][0]
            fb = F[F.design.str.match(r'single\|V0_HAR\|B-')].iloc[0]
            fin_rows.append(dict(h=h, order=0, cand='D12_h22', design=fb.design, failed=fb.failed if isinstance(fb.failed, str) else '',
                                 mean_q=fb.mean_q, rank_in_cand=int(fb.rank_in_cand), role='фолбэк владельца (HAR-J с поправкой)'))
        # цели владельца для первого финалиста
        c1, d1 = fins[0]
        code = d1.split('|')[2]
        g = per_year(c1, h, d1)
        F1 = [f for f in Fs if f.cand.iloc[0] == c1][0]
        har = F1[F1.design == f'single|V0_HAR|{code}|L0'].iloc[0] if code != 'bez' else F1[F1.design == 'single|V0_HAR|B-ord|L0'].iloc[0]
        gh = per_year(c1, h, har.design)
        singles = A[A.design.str.contains(r'\|V0_(?:XGB|LGB|HAR|HCJ)\|') & A.design.str.contains(r'\|' + code.replace('-', r'\-') + r'\|')]
        others = singles[~((singles.cand == c1) & (singles.design == d1))]
        bs = others[others.cand == c1].sort_values('mean_q').iloc[0]  # лучшая другая одиночная модель того же кандидата (та же s²)
        bs_all = others.sort_values('mean_q').iloc[0]  # лучшая другая одиночная модель среди всех кандидатов h (та же s²)
        pv = A[A.cand.isin(PURE) & A.design.str.contains(r'\|V0_(?:XGB|LGB)\|')]
        pv_same = pv[pv.design.str.contains(r'\|' + code.replace('-', r'\-') + r'\|')]
        pvb = pv_same.sort_values('mean_q').iloc[0] if len(pv_same) else None
        harp = None
        if pvb is not None:
            Fp = [f for f in Fs if f.cand.iloc[0] == pvb.cand][0]
            harp = Fp[Fp.design == f'single|V0_HAR|{code}|L0'].iloc[0]
        goals.append(dict(
            h=h, finalist=f'{c1}: {d1}', q_final=fins and A[(A.cand == c1) & (A.design == d1)].mean_q.iloc[0],
            q_har_corr=har.mean_q, g1_rel=A[(A.cand == c1) & (A.design == d1)].mean_q.iloc[0] / har.mean_q - 1,
            g1_years_better=int((g.q_hyb.values < gh.q_hyb.values).sum()),
            g1_per_year=' / '.join(f'{a:.4f}:{b:.4f}' for a, b in zip(g.q_hyb, gh.q_hyb)),
            g2_best_single_cand=f'{bs.design} {bs.mean_q:.4f}', g2_rel_cand=A[(A.cand == c1) & (A.design == d1)].mean_q.iloc[0] / bs.mean_q - 1,
            g2_best_single_all=f'{bs_all.cand}: {bs_all.design} {bs_all.mean_q:.4f}',
            g2_rel_all=A[(A.cand == c1) & (A.design == d1)].mean_q.iloc[0] / bs_all.mean_q - 1,
            g3_pure_v0=(f'{pvb.cand}: {pvb.design} {pvb.mean_q:.4f}' if pvb is not None else 'нет'),
            g3_rel=(pvb.mean_q / harp.mean_q - 1) if pvb is not None else np.nan,
            g4='нет данных (GRU на скользящей валидации не считался)'))
    B = pd.concat(out, ignore_index=True)
    B.to_csv(R / 'select_candidates.csv', index=False)
    FN = pd.DataFrame(fin_rows)
    FN.to_csv(R / 'finalists_preliminary.csv', index=False)
    G = pd.DataFrame(goals)
    G.to_csv(R / 'owner_goals_dev.csv', index=False)
    pd.set_option('display.width', 250); pd.set_option('display.max_colwidth', 80)
    print(B[['h', 'cand', 'design', 'failed', 'n_failed', 'mean_q', 'rank_in_cand', 'pure_v0']].to_string(index=False))
    print(FN.to_string(index=False))
    print(G.T.to_string())


if __name__ == '__main__':
    main()
