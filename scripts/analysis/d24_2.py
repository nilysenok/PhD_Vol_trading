#!/usr/bin/env python3
"""Д24.2 (владелец 26.10, 02:15): горизонт как признак — одна модель на объединённой выборке h ∈ {1, 5, 22}
(цель — ln среднего RV за [t+1, t+h]; признак h; Д8.12: ln прогноза HAR-J своего h как признак), сравнение с раздельными моделями.
Учтено: Д13.2 (точка L2_h5, без подбора) — h=1 +13,6%, h=5 +3,2%, h=22 −0,4% (V9); здесь — V0 + Д8.12 и выбор точки на шаге.

Протокол (ночное умолчание; одинаковый для объединённой и раздельных):
  точки-кандидаты шага — выбранные точки поиска B v2b для h=1, 5, 22 на этом шаге (Bv2b_selected_h{h}.csv; шаги 7–9 — точка шага 6);
  объединённая: точка шага (модель отдельно) — минимум среднего по трём h val-QLIKE шага (без поправки), шаги 1–6; seed 42/43/44;
    бины — по обучающему окну шага (без reference окна шага 1), ES — на объединённом val-годе шага, patience 30;
  раздельная на h: та же модель (Д8.12) в выбранной точке B v2b своего h (как D12_h1_A_d812 на h=1), seed 42/43/44, бины окна шага 1.
Оценка — d24_1.gate: OOS 2017–2020, 3 seed, Б0.1 ord, порог ≥ 1% и лучше на каждом seed; 2020 и 2021 отдельно; c1–c3 на 2016–2021.
Выход: data/regen/D24_2_h{h}_{m}_h{h}{,_seeds4344}_val.parquet (шаги 1–9), тест — data/regen/test_locked/D24_2_h{h}/ (не открывается);
раздельные — D24_2s_h{h}_*.
python3 scripts/analysis/d24_2.py run --workers 2
python3 scripts/analysis/d24_2.py gate"""
import argparse, json, multiprocessing as mp, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import d24_1 as X

L, R, M = X.L, X.R, X.M
HS = (1, 5, 22)
CFG = {'harf': True}
G = {}


def bv2b_points():
    import d12_h1 as H1
    grid = H1.grid()
    P = {}
    for h in HS:
        S = pd.read_csv(R / f'Bv2b_selected_h{h}.csv')
        for m in M:
            for s in range(1, 10):
                P[(h, m, s)] = (grid[(m, int(S[(S.model == m) & (S.step == min(s, 9))].point.iloc[0]))], int(S[(S.model == m) & (S.step == s)].point.iloc[0]))
    return P


def pooled_data(m, step):
    Ds = {h: L.step_data(G['A'], G['cols'], h, step, m, CFG, 'Bv2b') for h in HS}
    names = [n for n in Ds[1]['names'] if all(n in Ds[h]['names'] for h in HS)]
    cat = lambda f: np.concatenate([f(h) for h in HS])
    Xs = {k: cat(lambda h: np.column_stack([Ds[h]['X'][k][:, [Ds[h]['names'].index(n) for n in names]],
                                            np.full(len(Ds[h]['X'][k]), np.log(h), np.float32)])).astype(np.float32) for k in ('tr', 'va', 'te')}
    D = dict(X=Xs, y={k: cat(lambda h: Ds[h]['y'][k]) for k in ('tr', 'va', 'te')}, ytr_ln=cat(lambda h: Ds[h]['ytr_ln']),
             off={k: cat(lambda h: Ds[h]['off'][k]) for k in ('tr', 'va', 'te')}, w=None, names=names + ['ln_h'], cats=[], mono=[0] * (len(names) + 1))
    return D, Ds


def split_back(Ds, step, k, pred):
    out, i = {}, 0
    for h in HS:
        d = Ds[h]['parts'][k]; n = len(d)
        out[h] = pd.DataFrame({'date': d['date'].values, 'ticker': d['ticker'].values, 'year': Ds[h]['vy'] if k == 'va' else Ds[h]['ty'], 'step': step,
                               'rv_actual': d[L.tcol(h)].values, 'rv_pred': np.clip(np.exp(pred[i:i + n]), 1e-10, None)})
        i += n
    return out


def job_pooled(a):
    """(модель, шаг): шаги 1–6 — перебор трёх точек-кандидатов (seed 42), выбор по среднему val-QLIKE трёх h, затем seed 43/44;
    шаги 7–9 — точка, выбранная на шаге 6 (передаётся в задаче), seed 42/43/44."""
    m, step, fixed = a
    D, Ds = pooled_data(m, step)
    cands = [fixed] if fixed is not None else [h for h in HS]
    res, scores = {}, {}
    for hp in cands:
        p = G['P'][(hp, m, min(step, 6))][0]  # шаги 7-9: точка шага 6
        pv, pt, bi, _, _ = L.fit(m, p, D, CFG, seed=42, ref=None)
        v = split_back(Ds, step, 'va', pv)
        # шаги 7-9 (подтверждающий набор): точка фиксирована, QLIKE НЕ считается
        scores[hp] = float(np.mean([L.qlike(v[h].rv_actual.values, v[h].rv_pred.values) for h in HS])) if step <= 6 else np.nan
        res[(hp, 42)] = (v, split_back(Ds, step, 'te', pt), bi)
    best = fixed if fixed is not None else min(scores, key=scores.get)
    p = G['P'][(best, m, min(step, 6))][0]
    for sd in (43, 44):
        pv, pt, bi, _, _ = L.fit(m, p, D, CFG, seed=sd, ref=None)
        res[(best, sd)] = (split_back(Ds, step, 'va', pv), split_back(Ds, step, 'te', pt), bi)
    return m, step, best, scores, {sd: res[(best, sd)] for sd in (42, 43, 44)}


def write(tag_h, h, frames):
    """frames: {m: {seed: (val df, test df)}} одного h -> формат gibrid_io."""
    lock = R / 'test_locked' / tag_h
    lock.mkdir(parents=True, exist_ok=True)
    for m, by in frames.items():
        for sfx, sds in (('', (42,)), ('_seeds4344', (43, 44))):
            v = pd.concat([by[sd][0].assign(seed=sd, rank=1) for sd in sds], ignore_index=True).sort_values(['seed', 'step', 'date', 'ticker'])
            t = pd.concat([by[sd][1].assign(seed=sd, rank=1) for sd in sds], ignore_index=True).sort_values(['seed', 'step', 'date', 'ticker'])
            v.to_parquet(R / f'{tag_h}_{m}_h{h}{sfx}_val.parquet', index=False)
            t.to_parquet(lock / f'{tag_h}_{m}_h{h}{sfx}_test.parquet', index=False)


def run(a):
    t0 = time.time()
    X.set_matrix(1, 'features_pit_v3.parquet', '')
    A, FC = L.load_base()
    A2, cols, _, _ = L.build_features(A, FC, 1, CFG)
    G.update(A=A2, cols=cols, P=bv2b_points())
    X.G.update(A=A2, cols=cols, cfg=CFG)
    ctx = mp.get_context('fork')
    # объединённая: шаги 6..1, затем 7–9 с точкой шага 6
    with ctx.Pool(a.workers) as P:
        r16 = P.map(job_pooled, [(m, s, None) for s in range(6, 0, -1) for m in M], chunksize=1)
    sel6 = {m: [r for r in r16 if r[0] == m and r[1] == 6][0][2] for m in M}
    with ctx.Pool(a.workers) as P:
        r79 = P.map(job_pooled, [(m, s, sel6[m]) for s in (7, 8, 9) for m in M], chunksize=1)
    allr = r16 + r79
    pd.DataFrame([dict(model=r[0], step=r[1], point_of_h=r[2], **{f'val_q_pt_h{k}': v for k, v in r[3].items()}, bi42=r[4][42][2]) for r in allr]
                 ).sort_values(['model', 'step']).to_csv(R / 'D24_2_selected.csv', index=False, float_format='%.6g')
    for h in HS:
        fr = {m: {sd: tuple(pd.concat([r[4][sd][j][h] for r in allr if r[0] == m], ignore_index=True) for j in (0, 1)) for sd in (42, 43, 44)} for m in M}
        write(f'D24_2_h{h}', h, fr)
    print(f'{time.strftime("%T")} объединённая готова за {time.time() - t0:.0f} с', flush=True)
    # раздельные (h=5, 22; h=1 — это D12_h1_A_d812: та же модель, точки B v2b h=1)
    for h in (5, 22):
        X.G['h'] = h
        tasks = [(m, s, G['P'][(h, m, min(s, 6) if s <= 6 else 6)][0], sd, 1, G['P'][(h, m, min(s, 6) if s <= 6 else 6)][1]) for m in M for s in range(1, 10) for sd in (42, 43, 44)]
        res = X.refit_jobs(tasks, a.workers, 'Bv2b')
        X.write_legs(res, f'D24_2s_h{h}', h)
        print(f'{time.strftime("%T")} раздельная h={h} готова', flush=True)


def gate(a):
    rows = []
    fin = {1: 'D12_h1_A_d812', 5: 'D12_h5', 22: 'D16_5_h22'}
    for h in HS:
        sep = 'D12_h1_A_d812' if h == 1 else f'D24_2s_h{h}'
        for cur in dict.fromkeys([sep, fin[h]]):
            r = X.gate(f'D24_2_h{h}', cur, h)
            rows.append(dict(time=time.strftime('%F %T'), **r))
            print(json.dumps({k: (round(float(v), 5) if isinstance(v, (float, np.floating)) else (bool(v) if isinstance(v, (bool, np.bool_)) else v))
                              for k, v in r.items() if not k.startswith(('harj_q', 'tag_q'))}, ensure_ascii=False), flush=True)
    pd.DataFrame(rows).to_csv(R / 'D24_2_gate.csv', index=False, float_format='%.6g')


def leak(a):
    """«Будущее не влияет» для объединённой модели: шум с 1 января тестового года шага -> val-прогнозы всех h побайтно равны;
    самопроверка — шум с 1 января val-года -> прогнозы меняются."""
    X.set_matrix(1, 'features_pit_v3.parquet', '')
    A, FC = L.load_base()
    A2, cols, _, _ = L.build_features(A, FC, 1, CFG)
    G.update(A=A2, cols=cols, P=bv2b_points())
    st, ty, rows = a.step, L.TEST_YEARS[a.step - 1], []
    for m in M:
        G['A'] = A2
        b = job_pooled((m, st, None))
        for t0, must in ((f'{ty}-01-01', 'равны'), (f'{ty - 1}-01-01', 'изменились')):
            G['A'] = X.corrupt(A2, t0)
            v = job_pooled((m, st, b[2]))
            for h in HS:
                x, y = v[4][42][0][h], b[4][42][0][h]
                d = float(np.abs(np.log(x.rv_pred.values) - np.log(y.rv_pred.values)).max()) if len(x) == len(y) else np.inf
                ok = bool(d == 0.0) if must == 'равны' else bool(d > 0)
                rows.append(dict(contour=f'D24_2 h={h}', model=m, step=st, noise_from=t0, expect=must, max_abs_dln=d, ok=ok))
                print(f'leak D24_2 h={h} {m} шаг {st}: шум с {t0} -> max|Δ ln| = {d:.3g} (ожидается: {must}) {"OK" if ok else "ПРОВАЛ"}', flush=True)
    X.save_leak(rows)


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest='cmd', required=True)
    p = sp.add_parser('run'); p.add_argument('--workers', type=int, default=2)
    sp.add_parser('gate')
    p = sp.add_parser('leak'); p.add_argument('--step', type=int, default=3)
    a = ap.parse_args()
    {'run': run, 'gate': gate, 'leak': leak}[a.cmd](a)


if __name__ == '__main__':
    main()
