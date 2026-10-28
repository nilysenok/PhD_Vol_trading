#!/usr/bin/env python3
"""Д24.3 (владелец 26.10, 02:15): финалисты и кандидаты пула с бустингом — 10 seed (42–51) вместо 3; порог «2σ» по 10 seed.
Seed 42/43/44 — сохранённые ноги кандидата ({T}_{m}_h{h}_val, _seeds4344_val); seed 45–51 — переобучение выбранной точки шага тем же кодом
(l2_lib.fit, бины окна шага 1, ES на val-годе, patience 30) только на шагах 1–6 (разработочный набор; шаги 7–9 и тест не нужны и не считаются).
Перед счётом — контроль воспроизведения: seed 42 шага 2 переобучается и сверяется с сохранённым файлом (|Δ ln| ≤ 1e-9).

Метрика seed s: средний OOS QLIKE val 2017–2020 (шаги 2–5) одиночной модели (точка шага, seed s) с поправкой Б0.1 (s² по val шага t−1;
код ord; у gamma с офсетом Д17.1 — raw, поправка внутри офсета). Модель кандидата (XGB или LGBM) — лучшая по среднему 10 seed.
Разность d_s = Q_канд(s) − Q_реф(s): эталон — HAR-J (с той же поправкой; от seed не зависит), основа — модель-основа кандидата с тем же seed.
Правило: d̄ = (1/10) Σ d_s; σ = sqrt(Σ (d_s − d̄)² / (10 − 1)); кандидат проходит, если d̄ < −2σ/√10 (t-статистика по seed < −2).
python3 scripts/analysis/d24_3.py refit --cand D16_5_h22 --workers 2
python3 scripts/analysis/d24_3.py table"""
import argparse, json, os, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
CAND_ARG = sys.argv[sys.argv.index('--cand') + 1] if '--cand' in sys.argv else None
if CAND_ARG == 'D17_1_h5_G5Xopt':
    os.environ['D17_BASE'] = 'harx'
import numpy as np, pandas as pd
import d24_1 as X

L, R, M = X.L, X.R, X.M
SEEDS = tuple(range(42, 52))
NEW = tuple(range(45, 52))
# кандидат -> (h, cfg, матрица, блоки, точки, основа, код s², особый механизм)
C = {
    'D12_h1_A_d812': (1, {'harf': True}, 'features_pit_v3.parquet', '', 'Bv2b', 'D12_h1_V0', 'ord', None),
    'D12_h1_V0': (1, {}, 'features_pit_v3.parquet', '', 'Bv2b', None, 'ord', None),
    'D16_h1_d812x': (1, {'harf': True}, 'features_pit_v3.parquet', '', 'Bv2b', 'D12_h1_A_d812', 'ord', 'harx_feat'),
    'D12_h5': (5, {'v9': True, 'harf': True, 'huber': True}, 'features_pit_v3.parquet', 'd7', 'D12_h5', None, 'ord', None),
    'D17_1_h5_G5Xopt': (5, {'obj': 'gamma', 'v9': True, 'b01': True, 'harf': True}, 'features_pit_v3.parquet', 'd7', 'D17_1_h5_G5Xopt', 'D12_h5', 'raw', 'd17harx'),
    'D16_5_h22': (22, {}, 'features_pit_v3_d165.parquet', '', 'D16_5_h22', 'D15_h22', 'ord', None),
    'D15_h22': (22, {}, 'features_pit_v3.parquet', '', 'D15_h22', None, 'ord', None),
}


def setup(cand):
    h, cfg, matrix, blocks, src, _, _, mech = C[cand]
    n = X.load_cfg(h, cfg, matrix, blocks)
    if mech == 'harx_feat':  # Д16.1 h=1: ln HAR-X шага вместо ln HAR-J в признаке har_ln (как d16_v9.py)
        import d16_v9 as V9
        import harx as HX
        hx = HX.ln_all_steps(h, HX.default_terms(h, True))
        X.G['A'] = X.G['A'].merge(hx, on=['date', 'ticker'], how='left', validate='one_to_one')
        V9.G['control'] = False
    if mech == 'd17harx':  # Д17.1: офсет и признак — ln HAR-X (d17_1.py, D17_BASE=harx), gamma, Б0.1 в офсете
        import d17_1  # noqa: F401  (подменяет L.step_data поверх обёртки d24_1)
        assert d17_1.BASE_KIND == 'harx'
    return h, src, n


def refit(a):
    cand = a.cand
    h, src, n = setup(cand)
    P = X.points(src, h, range(1, 7))
    # контроль: seed 42, шаг 2, обе модели
    for m in M:
        p, rk, pnt = P[(m, 2)][0]
        r = X._unit((m, 2, [(p, 42, 1, pnt)], 'Bv2b'))[0]
        b = pd.read_parquet(R / f'{cand}_{m}_h{h}_val.parquet')
        b = b[(b.step == 2) & ((b.seed == 42) if 'seed' in b else True)]
        if 'rank' in b:
            b = b[b['rank'] == 1]
        mm = r[6].merge(b, on=['date', 'ticker'])
        d = float(np.abs(np.log(mm.rv_pred_x) - np.log(mm.rv_pred_y)).max())
        print(f'{cand} {m}: контроль seed 42 шаг 2 — строк {len(mm)}/{len(b)}, max|Δ ln| = {d:.2e}', flush=True)
        assert len(mm) == len(b) and d <= 1e-9, 'воспроизведение не совпало'
    X._REF.clear()
    tasks = [(m, s, p, sd, 1, pnt) for (m, s), its in P.items() for p, rk, pnt in its[:1] for sd in NEW]
    t0 = time.time()
    print(f'{time.strftime("%T")} {cand}: h={h}, признаков {n}, задач {len(tasks)} (seed {NEW[0]}–{NEW[-1]}, шаги 1–6), воркеров {a.workers}', flush=True)
    res = X.refit_jobs(tasks, a.workers, 'Bv2b')
    for m in M:
        v = pd.concat([x[6] for x in res if x[0] == m], ignore_index=True).sort_values(['seed', 'step', 'date', 'ticker'])
        v.to_parquet(R / f'D24_3_{cand}_{m}_h{h}_seeds4551_val.parquet', index=False)
    pd.DataFrame([dict(model=x[0], step=x[1], seed=x[2], best_iteration=x[5]) for x in res]).to_csv(R / f'D24_3_{cand}_trees.csv', index=False)
    print(f'{time.strftime("%T")} {cand}: готово за {time.time() - t0:.0f} с', flush=True)


def per_seed(cand, code):
    h = C[cand][0]
    out = {}
    for m in M:
        for sd in SEEDS:
            q = X.year_q(X.seed_frame(cand, m, h, sd), code)
            out[(m, sd)] = X.summ(q) | {f'q{2015 + s}': q[s] for s in range(1, 7)}
    return out


def table(a):
    rows, det = [], []
    for cand, (h, _, _, _, _, base, code, _) in C.items():
        if not all((R / f'D24_3_{cand}_{m}_h{h}_seeds4551_val.parquet').exists() for m in M):
            print(f'{cand}: нет seed 45–51, пропуск', flush=True)
            continue
        if a.code and code != 'raw':
            code = a.code
        ps = per_seed(cand, code)
        mbest = min(M, key=lambda m: np.mean([ps[(m, s)]['oos1720'] for s in SEEDS]))
        hc = a.code or 'ord'
        H = X.summ(X.year_q(X.harj_val(h), hc))
        qb = per_seed(base, C[base][6] if (C[base][6] == 'raw' or not a.code) else a.code) if base else None
        mb = min(M, key=lambda m: np.mean([qb[(m, s)]['oos1720'] for s in SEEDS])) if base else None
        for sd in SEEDS:
            det.append(dict(cand=cand, h=h, seed=sd, model=mbest, oos1720=ps[(mbest, sd)]['oos1720'], y2020=ps[(mbest, sd)]['y2020'],
                            y2021=ps[(mbest, sd)]['y2021'], **{f'{m}_oos1720': ps[(m, sd)]['oos1720'] for m in M},
                            base=base, base_oos1720=qb[(mb, sd)]['oos1720'] if base else np.nan, harj_oos1720=H['oos1720']))
        for ref, refname in (('harj', 'HAR-J (эталон)'), ('base', base)):
            if ref == 'base' and not base:
                continue
            for per in ('oos1720', 'y2020', 'y2021'):
                c = np.array([ps[(mbest, s)][per] for s in SEEDS])
                r = np.full(len(SEEDS), H[per]) if ref == 'harj' else np.array([qb[(mb, s)][per] for s in SEEDS])
                d = c - r
                dm, sg = float(d.mean()), float(d.std(ddof=1))
                thr = -2 * sg / np.sqrt(len(SEEDS))
                rows.append(dict(cand=cand, h=h, model=mbest, code=code, ref=refname, ref_model=(mb if ref == 'base' else f'HAR-J {hc}'), period=per,
                                 q_cand=float(c.mean()), q_ref=float(r.mean()), rel=float(c.mean() / r.mean() - 1), d_mean=dm, d_sd=sg, thr=thr,
                                 t=dm / (sg / np.sqrt(len(SEEDS))) if sg > 0 else -np.inf, passed=bool(dm < thr), n_better=int((d < 0).sum()),
                                 pass3=bool(np.mean(d[:3]) <= -0.01 * r[:3].mean() and (d[:3] < 0).all())))
    T = pd.DataFrame(rows)
    sfx = f'_{a.code}' if a.code else ''
    T.to_csv(R / f'D24_3_seeds10{sfx}.csv', index=False, float_format='%.6g')
    pd.DataFrame(det).to_csv(R / f'D24_3_seeds10{sfx}_detail.csv', index=False, float_format='%.6g')
    pd.set_option('display.width', 250)
    print(T[T.period == 'oos1720'][['cand', 'h', 'model', 'ref', 'q_cand', 'q_ref', 'rel', 'd_mean', 'd_sd', 'thr', 't', 'passed', 'n_better']].to_string(), flush=True)


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest='cmd', required=True)
    p = sp.add_parser('refit'); p.add_argument('--cand', required=True); p.add_argument('--workers', type=int, default=2)
    p = sp.add_parser('table'); p.add_argument('--code', default=None, help='замена кода s² (sm/smrob) для справки; gamma — raw')
    a = ap.parse_args()
    {'refit': refit, 'table': table}[a.cmd](a)


if __name__ == '__main__':
    main()
