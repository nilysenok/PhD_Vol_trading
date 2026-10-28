#!/usr/bin/env python3
"""Ансамблевая обёртка финального V9 (решение владельца 25.10, 19:40), без поиска; точка — optuna-30 уровня 2 ({TAG}_selected_h{h}.csv).
Компоненты (усреднение на уровне прогноза RV, поправка Б0.1 ord вне выборки — к итоговому прогнозу, как l2_lib.evaluate):
  1a — среднее топ-10 испытаний поиска шага (по val QLIKE шага; прогнозы испытаний — {TAG}_allpts_*, seed 42);
  1b — среднее 5 seed (42-46) лучшей точки;
  2  — среднее V9-LightGBM и V9-XGBoost (лучшие точки поиска);
  3  — среднее моделей шагов t и t-1 (модель шага t-1 — её лучшая точка, seed 42, — предсказывает те же даты; шаг 1 — без пары);
  4  — бэггинг по 5 подмножествам 70% признаков (seed 42-46; har_ln Д8.12 остаётся всегда), среднее.
Внутри модели и шага: среднее членов {лучшая точка} ∪ 1a ∪ 1b ∪ 4; затем 3; затем 2. Оценка — шаги 2-6 (разработочный набор), критерий —
средний val QLIKE с Б0.1 ord. Подтверждающий набор (val шагов 7-9) не пишется и не оценивается; тест — только в test_locked/wrap/, не оценивается.
python3 scripts/analysis/wrap_v9.py --h 22 --cfg '{"v9": true, "linear_lambda": 1}' [--blocks d7] [--tag L2_h22] [--workers 6]"""
import argparse, itertools, json, multiprocessing as mp, sys, time
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import l2_lib as L
from tune_L2 import SPACEB

G = {}
SEEDS = (42, 43, 44, 45, 46)
_split0 = L.split
OVR = {}


def _split(A, h, step):  # тестовая часть шага — год OVR['y'] (для пары t-1 -> t на тестовых годах)
    tp, vi, td, vy, ty = _split0(A, h, step)
    if OVR.get('y'):
        ty = OVR['y']; td = A[A.year == ty].dropna(subset=[L.tcol(h)])
    return tp, vi, td, vy, ty


L.split = _split


def job(a):
    """kind: base / seed / bag / prev. -> dict + val/test кадры (ln -> RV)."""
    kind, model, step, seed, bag = a
    pstep = min(step, 6)
    r = G['sel'][(model, pstep)]
    p = {k: r[k] for k, *_ in SPACEB[model]}
    cfg = dict(G['cfg'])
    if bag is not None:
        rng = np.random.default_rng(bag)
        cols = G['cols']
        keep = set(rng.choice(cols, size=int(round(0.7 * len(cols))), replace=False))
        cfg['drop'] = {model: [c for c in cols if c not in keep]}
    OVR['y'] = L.TEST_YEARS[step] if kind == 'prev' else None  # модель шага `step` предсказывает тест шага step+1
    D = L.step_data(G['A'], G['cols'], G['h'], step, model, cfg, G['btag'])
    ref = L.make_ref(model, G['A'], G['cols'], G['h'], cfg, G['btag']) if step > 1 else None
    pv, pt, bi, _, _ = L.fit(model, p, D, cfg, seed=seed, ref=ref)
    OVR['y'] = None
    out = []
    for k, pred in (('val', pv), ('test', pt)):
        df = D['parts']['va' if k == 'val' else 'te']
        out.append(pd.DataFrame({'date': df['date'].values, 'ticker': df['ticker'].values, 'rv_actual': df[L.tcol(G['h'])].values,
                                 'rv_pred': np.clip(np.exp(pred), 1e-10, None)}))
    return dict(kind=kind, model=model, step=step, seed=seed, bag=bag, bi=bi), out[0], out[1]


def key(df):
    return df.sort_values(['date', 'ticker']).reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--h', type=int, required=True); ap.add_argument('--cfg', default='{}'); ap.add_argument('--blocks', default='')
    ap.add_argument('--tag', default=None); ap.add_argument('--btag', default='Bv2b'); ap.add_argument('--workers', type=int, default=6)
    a = ap.parse_args()
    h, cfg = a.h, json.loads(a.cfg)
    TAG = a.tag or f'L2_h{h}'
    R = L.REGEN
    LOCK = R / 'test_locked' / 'wrap'; LOCK.mkdir(parents=True, exist_ok=True)
    A, FC = L.load_base(None, tuple(x for x in a.blocks.split(',') if x), h)
    A2, cols, _, _ = L.build_features(A, FC, h, cfg)
    sel = L.selected_params(TAG, h)
    G.update(A=A2, cols=cols, cfg=cfg, h=h, btag=a.btag, sel=sel)
    t0 = time.time()
    print(f'wrap {TAG}: cfg={cfg}, блоки={a.blocks or "-"}, признаков {len(cols)}', flush=True)
    M = ('lgbm', 'xgb')
    tasks = [('base', m, s, 42, None) for m in M for s in range(1, 10)]
    tasks += [('seed', m, s, sd, None) for m in M for s in range(1, 10) for sd in SEEDS[1:]]
    tasks += [('bag', m, s, 42, b) for m in M for s in range(1, 10) for b in SEEDS]
    tasks += [('prev', m, s, 42, None) for m in M for s in range(1, 9)]  # тест шага s+1 моделью шага s (val шага s+1 = тест шага s в base)
    with mp.get_context('fork').Pool(a.workers) as P:
        res = P.map(job, tasks, chunksize=1)
    print(f'обучено {len(res)} моделей, {time.time() - t0:.0f} с', flush=True)
    F = {}
    for info, fv, ft in res:
        F[(info['kind'], info['model'], info['step'], info['seed'], info['bag'])] = (key(fv), key(ft))
    # сверка: base seed 42 воспроизводит основной прогноз поиска (val, шаги 1-6)
    for m in M:
        V = pd.read_parquet(R / f'{TAG}_{m}_h{h}_val.parquet')
        V = V[(V.step <= 6) & (V.seed == 42)]
        d = max(float(np.max(np.abs(key(V[V.step == s])['rv_pred'].values / F[('base', m, s, 42, None)][0]['rv_pred'].values - 1))) for s in range(1, 7))
        print(f'сверка base {m}: макс. отн. расхождение {d:.2e}', flush=True)
    # топ-10 испытаний (val — из allpts в data/regen; тест — из test_locked/{TAG}, без оценки)
    T = pd.read_csv(R / f'{TAG}_trials.csv')
    T = T[~T.pruned]
    TOP = {}
    for m in M:
        AV = pd.read_parquet(R / f'{TAG}_allpts_{m}_h{h}_val.parquet')
        pt = R / 'test_locked' / TAG / f'{TAG}_allpts_{m}_h{h}_test.parquet'
        AT = pd.read_parquet(pt) if pt.exists() else None
        for s in range(1, 7):
            tr = T[(T.model == m) & (T.step == s)].sort_values('qlike_val').trial.tolist()[:10]
            TOP[(m, s)] = ([key(AV[(AV.step == s) & (AV.trial == t)]) for t in tr],
                           [key(AT[(AT.step == s) & (AT.trial == t)]) for t in tr] if AT is not None else None)

    def member_mean(m, s, comps, k):
        i = 0 if k == 'val' else 1
        base = F[('base', m, s, 42, None)][i]
        mem = [base['rv_pred'].values]
        if '1a' in comps and s <= 6 and TOP[(m, s)][i] is not None:
            for d in TOP[(m, s)][i]:
                assert np.array_equal(d['date'].values, base['date'].values) and np.array_equal(d['ticker'].values, base['ticker'].values)
                mem.append(d['rv_pred'].values)
            mem = mem[1:]  # топ-10 уже содержит лучшую точку (ранг 1)
        if '1b' in comps:
            mem += [F[('seed', m, s, sd, None)][i]['rv_pred'].values for sd in SEEDS[1:]]
        if '4' in comps:
            mem += [F[('bag', m, s, 42, b)][i]['rv_pred'].values for b in SEEDS]
        return np.mean(mem, axis=0)

    def forecast(m, s, comps, k):
        f = member_mean(m, s, comps, k)
        if '3' in comps and s >= 2:
            # модель шага s-1 на тех же датах: val шага s = тест шага s-1 (base); тест шага s — prev шага s-1
            prev = F[('base', m, s - 1, 42, None)][1] if k == 'val' else F[('prev', m, s - 1, 42, None)][1]
            cur = F[('base', m, s, 42, None)][0 if k == 'val' else 1]
            assert np.array_equal(prev['date'].values, cur['date'].values) and np.array_equal(prev['ticker'].values, cur['ticker'].values)
            f = 0.5 * (f + prev['rv_pred'].values)
        return f

    def build(comps, k, prim):
        steps = range(1, 7) if k == 'val' else range(1, 10)
        ms = M if '2' in comps else (prim,)
        out = []
        for s in steps:
            b = F[('base', ms[0], s, 42, None)][0 if k == 'val' else 1]
            out.append(b[['date', 'ticker', 'rv_actual']].assign(step=s, rv_pred=np.mean([forecast(m, s, comps, k) for m in ms], axis=0)))
        return pd.concat(out, ignore_index=True)

    ev0 = {m: L.evaluate(build((), 'val', m)) for m in M}
    prim = min(M, key=lambda m: ev0[m]['ord'][0])
    q0 = ev0[prim]['ord'][0]
    rows = [dict(comp='V9 (исходный, лучшая точка, seed 42)', model=prim, q=q0, best4=min(ev0[prim][c][0] for c in L.CODES),
                 per_step=' / '.join(f'{x:.4f}' for x in ev0[prim]['ord'][1]), taken='база')]
    for m in M:
        print(f'база {m}: ord {ev0[m]["ord"][0]:.4f}', flush=True)
    ind = {}
    for c in ('1a', '1b', '2', '3', '4'):
        e = L.evaluate(build((c,), 'val', prim))
        ind[c] = e['ord'][0]
        rows.append(dict(comp=c, model='lgbm+xgb' if c == '2' else prim, q=e['ord'][0], best4=min(e[x][0] for x in L.CODES),
                         per_step=' / '.join(f'{x:.4f}' for x in e['ord'][1]), taken='да' if e['ord'][0] <= q0 else 'нет'))
        print(f'{c}: {e["ord"][0]:.4f} ({e["ord"][0] / q0 - 1:+.2%})', flush=True)
    keep = [c for c in ind if ind[c] <= q0]
    hist = []
    while True:
        if not keep:
            qa = q0; break
        e = L.evaluate(build(tuple(keep), 'val', prim))
        qa = e['ord'][0]
        hist.append(('+'.join(keep), qa, min(e[x][0] for x in L.CODES), ' / '.join(f'{x:.4f}' for x in e['ord'][1])))
        print(f'вместе {"+".join(keep)}: {qa:.4f} ({qa / q0 - 1:+.2%})', flush=True)
        if qa <= q0:
            break
        keep.remove(max(keep, key=lambda c: ind[c]))
    for comb, q, b4, ps in hist:
        rows.append(dict(comp='итог: ' + comb, model='lgbm+xgb' if '2' in comb.split('+') else prim, q=q, best4=b4, per_step=ps,
                         taken='да' if q <= q0 else 'нет'))
    tab = pd.DataFrame(rows)
    tab['vs_base'] = tab.q / q0 - 1
    tab.to_csv(R / f'wrap_h{h}_table.csv', index=False, float_format='%.6g')
    final = tuple(keep)
    V = build(final, 'val', prim); V['wrap'] = '+'.join(final) or 'none'
    V.to_parquet(R / f'wrap_h{h}_val.parquet', index=False)
    TT = build(final, 'test', prim); TT['wrap'] = '+'.join(final) or 'none'
    TT.to_parquet(LOCK / f'wrap_h{h}_test.parquet', index=False)
    print(f'wrap h={h} готов {time.strftime("%F %T")}, {time.time() - t0:.0f} с: итог {"+".join(final) or "-"} {qa:.4f} против V9 {q0:.4f} ({prim})', flush=True)
    print(tab.to_string(), flush=True)


if __name__ == '__main__':
    main()
