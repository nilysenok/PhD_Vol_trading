#!/usr/bin/env python3
"""Д26 — круг h=5 на разработке (владелец 26.10, 09:35; агент уровня 2). Отбор — OOS 2017–2020 (шаги 2–5), проверка — 2021 (шаг 6),
c1–c4 — gibrid_v17 на 2017–2021 (эталон с той же s²). Эталон — HAR-J с поправкой s² sm. Б0.1 — s² шага t−1.
Метрики за годы ≥ 2022 не считаются: в оценку и в гибрид идут только строки шагов 1–6 (d26_lib.dev_only).

  --step0          сводка уже посчитанного (D12_h5, G5Xopt, Д16.1 = V9 на HAR-X, итерированный hyb₁ Д22 и его средние, HAR-X⁺₅) по seed
  --d261 ...       Д26.1 (d26_direct.py)    --d262 / --d263 (d26_iter.py)    --d264 / --d265 / --d266 — здесь
Выход: data/regen/D26/ (csv, входы и выходы гибрида), отчёт — docs/otchety/2026-10-26_d26_h5.md.
"""
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import d26_lib as L

R, OUT = L.R, L.OUT
SEEDS = (42, 43, 44)
WORKERS = 6  # не больше 6 процессов (поручение)


def now():
    return time.strftime('%H:%M:%S')


def seed_frames(prefix, m, seeds=SEEDS, single_name=None, seeds_name=None):
    """{seed: val-кадр модели m} из пары файлов «seed 42» + «_seeds4344» (форматы tune_L2 / d16_v9)."""
    a = pd.read_parquet(R / (single_name or f'{prefix}_{m}_h5_val.parquet'))
    b = pd.read_parquet(R / (seeds_name or f'{prefix}_{m}_h5_seeds4344_val.parquet'))
    if 'model' in a:
        a, b = a[a.model == m], b[b.model == m]
    out = {42: a}
    for s in seeds:
        if s != 42:
            out[s] = b[b.seed == s]
    return {s: L.dev_only(d) for s, d in out.items() if s in seeds}


def lnmean(*dfs):
    """Среднее ln-прогнозов (как hyb₁ Д22: среднее ln XGB и ln LGBM)."""
    base = dfs[0][['date', 'ticker', 'year', 'step', 'rv_actual']].copy()
    ln = None
    for d in dfs:
        m = base[['date', 'ticker', 'step']].merge(d[['date', 'ticker', 'step', 'rv_pred']], on=['date', 'ticker', 'step'], how='left')
        assert m.rv_pred.notna().all()
        ln = np.log(m.rv_pred.values) if ln is None else ln + np.log(m.rv_pred.values)
    base['rv_pred'] = np.exp(ln / len(dfs))
    return base


def per_seed_gib(name, slots_by_seed, legs=('single', 'wrap')):
    """slots_by_seed: {seed: (кадр XGB-слота, кадр LGB-слота)} -> {seed: summarize}."""
    from concurrent.futures import ThreadPoolExecutor
    for s, (x, l) in slots_by_seed.items():
        L.write_boost(f'{name}_s{s}', x, l)
    with ThreadPoolExecutor(min(WORKERS, len(slots_by_seed))) as ex:
        paths = dict(zip(slots_by_seed, ex.map(lambda s: L.run_gibrid(f'{name}_s{s}', legs), list(slots_by_seed))))
    S = {s: L.summarize(p) for s, p in paths.items()}
    for s in S:
        print(f'  {now()} гибрид {name} seed {s}: лучший {S[s].iloc[0].design} {S[s].iloc[0].m1720:.4f} / {S[s].iloc[0].q2021:.4f}', flush=True)
    return S


def design_rows(name, S, design=None, src=''):
    """Строка таблицы: дизайн (по правилу на средних по seed или заданный), по seed, условие входа к D12_h5 (фикс. 0,2159)."""
    T, best = L.seeds_table(S)
    import re
    T.to_csv(OUT / ('seeds_' + re.sub(r'[^0-9A-Za-zА-Яа-я_.+-]+', '_', name) + '.csv'), index=False)
    b = T.set_index('design').loc[design] if design else best
    dn = design or best.design
    qs = [b[f'm1720_s{s}'] for s in S]
    e = L.entry(qs)
    return dict(name=name, design=dn, n_failed=int(b.n_failed), failed=b.failed, m1720=b.m1720, q2021=b.q2021,
                **{f'm1720_s{s}': b[f'm1720_s{s}'] for s in S}, **{f'q2021_s{s}': b[f'q2021_s{s}'] for s in S},
                **{str(y): b[str(y)] for y in range(2017, 2022)}, rel_D12=e['rel'], d_mean=e['d_mean'], sigma=e['sigma'], t=e['t'],
                pass3=e['pass3'], pass2s=e['pass2s'], src=src)


# ---------------------------------------------------------------- шаг 0
def step0():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    # D12_h5: финал (wrap2, фикс.) и по seed (single, wrap: среднее XGB и LGBM)
    xs, ls = seed_frames('D12_h5', 'xgb'), seed_frames('D12_h5', 'lgbm')
    S = per_seed_gib('D12_h5', {s: (xs[s], ls[s]) for s in SEEDS})
    rows.append(design_rows('D12_h5 по seed (single|V0_XGB|B-sm)', S, 'single|V0_XGB|B-sm|L0', 'final2 / final3 ranking_h5'))
    rows.append(design_rows('D12_h5 по seed (лучший дизайн по seed)', S, None, 'то же'))
    D12S = {s: lnmean(xs[s], ls[s]) for s in SEEDS}  # среднее XGB и LGBM seed s — компонента Д26.5
    # G5Xopt (Д17.1, GLM с офсетом на HAR-X)
    xs, ls = seed_frames('D17_1_h5_G5Xopt', 'xgb'), seed_frames('D17_1_h5_G5Xopt', 'lgbm')
    S = per_seed_gib('G5Xopt', {s: (xs[s], ls[s]) for s in SEEDS})
    rows.append(design_rows('D17_1_h5_G5Xopt по seed', S, None, 'PR #75, 2026-10-26_d17.md; Д24.3 (PR #94)'))
    # Д16.1 = V9 на HAR-X на точках D12_h5 (это и есть Д26.4 в толковании «основа V9 = HAR-X»)
    a, b = pd.read_parquet(R / 'D16_h5_D12x_val.parquet'), pd.read_parquet(R / 'D16_h5_D12x_s4344_val.parquet')
    ab = pd.concat([a, b])
    sl = {s: (L.dev_only(ab[(ab.seed == s) & (ab.model == 'xgb')]), L.dev_only(ab[(ab.seed == s) & (ab.model == 'lgbm')])) for s in SEEDS}
    S = per_seed_gib('D16_1_h5_D12x', sl)
    rows.append(design_rows('Д16.1 V9 на HAR-X (точки D12_h5) по seed', S, None, '2026-10-26_d16.md (h5_D12x)'))
    # Д22: итерированный hyb₁ (x6), HAR-X⁺₅ (x1) и средние
    import d22_full as F
    A, hist = F.matrix()
    I = F.load_inputs('val')
    sl6, slx = {}, {}
    for s in SEEDS:
        d = L.dev_only(F.assemble(5, 'val', s, I, A, hist))
        base = d[['date', 'ticker', 'year', 'step', 'rv_actual']]
        x6 = base.assign(rv_pred=np.exp(d.x6.values))
        x1 = base.assign(rv_pred=np.exp(d.x1.values))
        avg61 = base.assign(rv_pred=np.exp((d.x6.values + d.x1.values) / 2))
        avg6d = lnmean(x6, D12S[s])
        x6.to_parquet(OUT / f'x6_h5_s{s}_val.parquet', index=False)
        if s == 42:
            rows.append(dict(name='HAR-X⁺₅ (слой 1 Д22, x1), s² sm', design='одиночная', n_failed=np.nan,
                             **{k: v for k, v in L.row('', L.years(x1, 'sm')).items() if k != 'design'}, src='2026-10-26_d22.md, PR #89'))
        sl6[s] = (x6, avg61)
        slx[s] = (x6, avg6d)
    S = per_seed_gib('D22_x6', sl6)
    rows.append(design_rows('итерированный hyb₁ (x6) / среднее с HAR-X⁺₅', S, None, 'Д20 в слоях Д22 (d22_full.assemble)'))
    rows.append(design_rows('итерированный hyb₁ (x6) один', S, 'single|V0_XGB|B-sm|L0', 'то же'))
    rows.append(design_rows('среднее x6 и HAR-X⁺₅', S, 'single|V0_LGB|B-sm|L0', 'то же'))
    S = per_seed_gib('D22_x6d12', slx)
    rows.append(design_rows('среднее x6 и D12_h5 (seed s)', S, 'single|V0_LGB|B-sm|L0', 'то же'))
    T = pd.DataFrame(rows)
    T.to_csv(OUT / 'step0.csv', index=False)
    pd.set_option('display.width', 250)
    print(T[['name', 'design', 'n_failed', 'failed', 'm1720', 'q2021', 'rel_D12', 'd_mean', 'sigma', 't']].to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--step0', action='store_true')
    a = ap.parse_args()
    if a.step0:
        step0()


if __name__ == '__main__':
    main()
