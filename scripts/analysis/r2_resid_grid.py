#!/usr/bin/env python3
"""R2 протокол v3, агент B: сетка остаточного ученика (кластер 4, фаза D, boost.grid = resid_small_v1).
LHS-60: scipy.stats.qmc.LatinHypercube(d=7, seed=20261026), уровень = floor(u·k) по уровням models/designs/r2/r2_B_roli.yaml (lhs.levels),
порядок осей: num_leaves, max_depth, min_data_in_leaf, learning_rate, lambda_l2, feature_fraction, bagging_fraction.
Точки 1–60 — LHS, точка 61 — текущая точка ячейки (контроль; r2_roli.points). Список коммитится до фазы D: models/designs/r2/r2_B_lhs60.csv (+ SHA-256 в журнале).
Счёт точки — r2_roli.build(design, ctx) с design.boost.grid = resid_small_v1, early_stop = qlike_total и ctx['_lhs_point'] = i
(❓ для S/O: ключ схемы boost.grid_point ∈ 1..61, чтобы точка входила в хэш дизайна; до schema_v2 — ctx). ≤ 300 деревьев, patience 30.
Отсечение 60 → 30 по val 2016 (шаг 1), 30 → 15 → 8 по R̄ — у O (протокол v3 п. 4).
python3 scripts/analysis/r2_resid_grid.py lhs              # записать список (проверка: повтор даёт тот же SHA)
python3 scripts/analysis/r2_resid_grid.py smoke [--h 5]    # дым: точки 1, 2, 61 на 3 тикерах, шаги 1–3"""
import argparse, hashlib, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(HERE))
import numpy as np, pandas as pd, yaml

YB = yaml.safe_load(open(REPO / 'models' / 'designs' / 'r2' / 'r2_B_roli.yaml'))
AXES = ('num_leaves', 'max_depth', 'min_data_in_leaf', 'learning_rate', 'lambda_l2', 'feature_fraction', 'bagging_fraction')


def lhs_table():
    from scipy.stats import qmc
    c = YB['lhs']
    lv = c['levels']
    u = qmc.LatinHypercube(d=len(AXES), seed=int(c['seed'])).random(int(c['n']))
    rows = []
    for i, r in enumerate(u, 1):
        row = {'point': i}
        for j, ax in enumerate(AXES):
            k = len(lv[ax])
            row[ax] = lv[ax][min(int(np.floor(r[j] * k)), k - 1)]
        rows.append(row)
    return pd.DataFrame(rows)


def core_fn(design, seed, steps):
    """Для r2_core.run_phase (фаза D, владелец 16:25: 3 seed, LightGBM, ≤ 300 деревьев, ранняя остановка по QLIKE итога, patience 30):
    дизайн ячейки с boost.grid = resid_small_v1 -> {B_<роль>_p<NN>: ряд} по точкам SHARED['B_grid_points'] (по умолчанию 1–61;
    отсечение 60 → 30 → 15 → 8 решает O, r2_halving.py, и передаёт список точек следующей ступени). Точка — в имени ряда
    (каталог кэша — хэш дизайна ячейки), поэтому ключ схемы для точки не нужен."""
    import r2_core as C
    import r2_roli as B
    S = C.SHARED
    d = dict(C.validate(design), seed=int(seed))
    assert d['boost']['grid'] == 'resid_small_v1' and d['boost']['early_stop'] == 'qlike_total' and d['boost']['lib'] == 'lightgbm', \
        'фаза D: grid = resid_small_v1, early_stop = qlike_total, lib = lightgbm'
    h = int(d['h'])
    out = {}
    for pt in S.get('B_grid_points', range(1, 62)):
        leg = S['leg_provider'](d, h)
        leg = leg['base'] if isinstance(leg, dict) else leg
        ctx = dict(A=S['A'], h=h, groups=S['B_groups'], leg=leg, _lhs_point=int(pt), _afp=S.get('B_afp'))   # кэш наборов — как r2_roli
        if d['boost']['resid_source'] == 'cross_fit':
            ctx['leg_refit'] = S['leg_refit_provider'](d, h, S['A'], None)
            ctx['_refit_id'] = B.hashlib.sha256(B.json.dumps(dict(leg=d.get('leg'), tag='base', est='r2_nogi.fit_predict'), sort_keys=True,
                                                             default=str).encode()).hexdigest()[:32]
        out[f'B_{d["boost"]["role"]}_p{int(pt):02d}'] = B.build(d, ctx, steps=steps)
    return out


def cmd_run(a):
    """Контракт драйвера S (modules.BG): run --designs <yaml> --h --seeds --steps --workers [--point pNN]. Дизайны точек LHS
    (boost.grid = resid_small_v1, boost.grid_point = pNN по schema_v2): r2_roli.core_fn с рядом BG; точка — из дизайна (или --point)."""
    import r2_core as C
    import r2_roli as B
    Y = yaml.safe_load(open(a.designs))
    ds = [d for d in (Y.get('designs') or []) if str(C.get(C.validate(d), 'boost.grid')) == 'resid_small_v1']
    print(time.strftime('%F %T'), f'BG run: h={a.h}, дизайнов {len(ds)}, точка {a.point}, seed {a.seeds}, шаги {a.steps}', flush=True)
    if not ds:
        return
    B.prepare_run(a.h, tickers=a.tickers)
    C.share(B_series='BG', B_lhs_point=(int(a.point[1:3]) if a.point else None))
    T = C.run_phase('D', a.h, ds, B.core_fn, seeds=tuple(a.seeds), steps=tuple(a.steps), workers=a.workers, module='BG', check_gate=False)
    bad = T[T.ryad == 'ОШИБКА'] if len(T) else T
    print(time.strftime('%F %T'), f'BG run: рядов {len(T) - len(bad)}, ошибок {len(bad)}', flush=True)
    if len(bad):
        print(bad.to_string(), flush=True)
        raise SystemExit(1)


def cmd_leak(a):
    import r2_roli as B
    B.cmd_leak_designs(a)


def cmd_lhs(a):
    T = lhs_table()
    p = REPO / YB['lhs']['file']
    T.to_csv(p, index=False)
    sha = hashlib.sha256(p.read_bytes()).hexdigest()
    print(f'{p}: {len(T)} точек, SHA-256 {sha}; уникальных комбинаций {len(T.drop(columns="point").drop_duplicates())}')
    for ax in AXES:
        print(f'  {ax}: ' + ', '.join(f'{v}×{n}' for v, n in T[ax].value_counts().sort_index().items()))


def cmd_smoke(a):
    import r2_roli as B
    rows = []
    for h in a.h or YB['smoke']['hs']:
        ctx = B.d0_ctx(h)
        for pt in (1, 2, 61):
            ctx['_lhs_point'] = pt
            t0 = time.time()
            o = B.build(dict(design='grid_smoke', h=h, seed=42, boost=dict(role='residual', grid='resid_small_v1', early_stop='qlike_total',
                                                                          lib='lightgbm')), ctx, steps=YB['smoke']['steps'])
            rows.append(dict(h=h, point=pt, rows=len(o), sec=round(time.time() - t0, 1)))
            print(time.strftime('%T'), f'сетка h={h} точка {pt}: строк {len(o)}, {time.time() - t0:.1f} с', flush=True)
    p = B.R / 'R2' / 'B' / 'smoke_grid_B.csv'
    pd.DataFrame(rows).to_csv(p, index=False)


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest='cmd', required=True)
    sp.add_parser('lhs')
    p = sp.add_parser('smoke'); p.add_argument('--h', type=int, nargs='*')
    p = sp.add_parser('run'); p.add_argument('--designs', required=True); p.add_argument('--h', type=int, required=True)
    p.add_argument('--seeds', type=int, nargs='+', default=[42]); p.add_argument('--steps', type=int, nargs='+', default=[1, 2, 3, 4, 5, 6])
    p.add_argument('--workers', type=int, default=None); p.add_argument('--point', default=None); p.add_argument('--tickers', nargs='*', default=None)
    p = sp.add_parser('leak'); p.add_argument('--designs', required=True); p.add_argument('--h', type=int, required=True)
    a = ap.parse_args()
    {'lhs': cmd_lhs, 'smoke': cmd_smoke, 'run': cmd_run, 'leak': cmd_leak}[a.cmd](a)


if __name__ == '__main__':
    main()
