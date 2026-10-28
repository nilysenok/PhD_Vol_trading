#!/usr/bin/env python3
"""Финальная сборка второго круга, шаг 5: шаг 10 — обучение на годах <= 2024, ранняя остановка и оценка s2 по val 2025, прогноз 01-02.2026.
Точка — точка шага 6 (как на шагах 7-9; топ-5 на шагах без поиска не строится — нога top5 = точка шага 6). Метрики НЕ считаются:
тестовые строки 2026 не фильтруются по наличию цели (цели в конце ряда нет), rv_actual может быть NaN.
--kind h1  — кандидаты Д12 на h=1 (d12_h1.CANDS, точки ЛГК Bv2b, сетка Bv2b_grid_*); boost = D12_h1_<канд>;
--kind opt — итоговые комбинации optuna-30 (tune_L2_optuna.py; точка шага 6 из {boost}_selected_h{h}.csv), --cfg/--blocks как при подборе.
Выход: data/regen/final2/step10/{boost}_{m}_h{h}_s10_val.parquet (val 2025), data/regen/test_locked/final2/step10/{boost}_{m}_h{h}_s10_test.parquet,
data/regen/final2/step10/{boost}_h{h}_s10_trees.csv (best_iteration по модели и seed).
L2_FEAT=<data/regen>/features_pit_v3.parquet python3 scripts/analysis/final2_step10.py --kind opt --boost F2_h5 --h 5 --cfg '{...}' --blocks d7"""
import argparse, json, multiprocessing as mp, sys
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import l2_lib as L
from tune_L2 import SPACEB

R = L.REGEN
G = {}
_split0 = L.split


def _split10(A, h, step):
    if step != 10:
        return _split0(A, h, step)
    t = L.tcol(h)
    tr = A[A.year < 2026]
    return tr[tr.year != 2025].dropna(subset=[t]), tr[tr.year == 2025].dropna(subset=[t]), A[A.year == 2026], 2025, 2026


L.split = _split10


def job(a):
    m, seed = a
    cfg = G['cfg']
    D = L.step_data(G['A'], G['cols'], G['h'], 10, m, cfg, 'Bv2b')
    if cfg.get('beta') is not None:
        D['off'] = {k: cfg['beta'] * v for k, v in D['off'].items()}
    if cfg.get('mono_sign') == -1:
        D['mono'] = [-x for x in D['mono']]
    ref = L.make_ref(m, G['A'], G['cols'], G['h'], cfg, 'Bv2b')
    pv, pt, bi, _, _ = L.fit(m, G['P'][m], D, cfg, seed=seed, ref=ref)
    out = []
    for part, pred, yr in (('va', pv, 2025), ('te', pt, 2026)):
        d = D['parts'][part]
        out.append(pd.DataFrame({'date': d['date'].values, 'ticker': d['ticker'].values, 'year': yr, 'step': 10, 'seed': seed,
                                 'rv_actual': d[L.tcol(G['h'])].values, 'rv_pred': np.clip(np.exp(pred), 1e-10, None)}))
    return m, seed, bi, out[0], out[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--kind', required=True, choices=['h1', 'opt']); ap.add_argument('--boost', required=True); ap.add_argument('--h', type=int, required=True)
    ap.add_argument('--cfg', default='{}'); ap.add_argument('--blocks', default=''); ap.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44])
    ap.add_argument('--workers', type=int, default=6)
    a = ap.parse_args()
    h = a.h
    if a.kind == 'h1':
        import d12_h1 as H1
        cfg = H1.CANDS[a.boost.replace('D12_h1_', '')]
        S = pd.read_csv(R / 'Bv2b_selected_h1.csv')
        grid = H1.grid()
        P = {m: grid[(m, int(S[(S.model == m) & (S.step == 6)].point.iloc[0]))] for m in ('xgb', 'lgbm')}
        A, FC = L.load_base(R / 'features_pit_v3.parquet')
    else:
        cfg = json.loads(a.cfg)
        sel = L.selected_params(a.boost, h)
        P = {m: {k: sel[(m, 6)][k] for k, *_ in SPACEB[m]} for m in ('xgb', 'lgbm')}
        A, FC = L.load_base(None, tuple(x for x in a.blocks.split(',') if x), h)
    assert 'features_pit_v3' in str(L.FEAT) or a.kind == 'h1', 'нужен L2_FEAT=features_pit_v3'
    A2, cols, _, _ = L.build_features(A, FC, h, cfg)
    G.update(A=A2, cols=cols, cfg=cfg, h=h, P=P)
    with mp.get_context('fork').Pool(a.workers) as pool:
        res = pool.map(job, [(m, s) for m in ('xgb', 'lgbm') for s in a.seeds], chunksize=1)
    od, lock = R / 'final2' / 'step10', R / 'test_locked' / 'final2' / 'step10'
    od.mkdir(parents=True, exist_ok=True); lock.mkdir(parents=True, exist_ok=True)
    for m in ('xgb', 'lgbm'):
        rr = [r for r in res if r[0] == m]
        pd.concat([r[3] for r in rr]).to_parquet(od / f'{a.boost}_{m}_h{h}_s10_val.parquet', index=False)
        pd.concat([r[4] for r in rr]).to_parquet(lock / f'{a.boost}_{m}_h{h}_s10_test.parquet', index=False)
    tr = pd.DataFrame([dict(boost=a.boost, h=h, model=r[0], seed=r[1], best_iteration=r[2], params=json.dumps({k: float(v) for k, v in P[r[0]].items()}))
                       for r in res])
    tr.to_csv(od / f'{a.boost}_h{h}_s10_trees.csv', index=False)
    print(f'{a.boost} h={h}: шаг 10 готов; деревья ' + ', '.join(f'{r[0]}/s{r[1]}: {r[2]}' for r in res)
          + f'; тест 2026 строк {len(res[0][4])} (метрики не считались)', flush=True)


if __name__ == '__main__':
    main()
