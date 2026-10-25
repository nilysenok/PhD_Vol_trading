#!/usr/bin/env python3
"""Д16.1: V9 на основе HAR-X — бустинг учит ln RV − ln HAR-X (h=5/22) или видит ln HAR-X как признак (h=1, Д8.12), на точках финалистов
без нового подбора и в той же обёртке:
  h1  d812x  — D12_h1_A_d812: XGB, {"harf"}, точки top5 ЛГК B v2b (Bv2b_xgb_h1_top5_val), нога top5, поправка sm;
  h5  D12x   — D12_h5: V9 + Д8.12 + Д7 + Huber, точки D12_h5_selected_h5, нога wrap2 (среднее XGB и LGBM, seed 42), sm;
  h5  F2mx   — F2m_h5: V9 + Д8.12 + Д7 + mono −1, точки F2m_h5_selected_h5, нога wrap2, smrob;
  h22 D12x   — D12_h22: V9 (+ linear_tree λ=1 у LGBM), точки D12_h22_selected_h22, нога seeds3 XGB (42/43/44), smrob.
Механизм: l2_lib.har_ln подменяется — вместо МНК HAR-J возвращается ln HAR-X модели того же шага (harx.ln_all_steps; на обучении —
внутри выборки, как у HAR-J в V9). Остальное (step_data, бины окна шага 1, ES на val-годе, patience 30) — l2_lib без изменений.
Контроль: --control — та же задача без подмены (должна совпасть с сохранёнными файлами финалистов).
Выход: data/regen/D16_h{h}_{cand}_{model}_val.parquet (шаги 1-6 или 1-10 при --final; тест -> test_locked/D16_h{h}_{cand}/)."""
import os, sys, time, json, argparse, multiprocessing as mp
from pathlib import Path
os.environ.setdefault('L2_FEAT', '/Users/nilysenok/Desktop/Старое/PhD_Vol_trading/data/regen/features_pit_v3.parquet')
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import l2_lib as L
import harx as HX

R = L.REGEN
CANDS = {
    'h1_d812x': dict(h=1, cfg={'harf': True}, blocks=(), pts='bv2b_top5', models=('xgb',), seeds=(42,), leg='top5'),
    'h5_D12x': dict(h=5, cfg={'v9': True, 'harf': True, 'huber': True}, blocks=('d7',), pts='D12_h5', models=('xgb', 'lgbm'), seeds=(42,), leg='wrap2'),
    'h5_F2mx': dict(h=5, cfg={'v9': True, 'harf': True, 'mono': True, 'mono_sign': -1}, blocks=('d7',), pts='F2m_h5', models=('xgb', 'lgbm'), seeds=(42,), leg='wrap2'),
    'h22_D12x': dict(h=22, cfg={'v9': True, 'linear_lambda': 1}, blocks=(), pts='D12_h22', models=('xgb',), seeds=(42, 43, 44), leg='seeds3'),
}
G = {}
_har_ln0 = L.har_ln


def har_ln_x(tp, parts, h, hc, w=None, huber=False, y=None):
    """Подмена: ln HAR-X модели шага (шаг по последнему году обучения: vy = max(year)+1, k = vy − 2015)."""
    if G.get('control'):
        return _har_ln0(tp, parts, h, hc, w, huber, y)
    k = int(tp['year'].max()) + 1 - 2015
    col = f'hx{k}'
    return [d[col].values for d in parts]


L.har_ln = har_ln_x
_split0 = L.split


def _split10(A, h, step):
    """Шаг 10 (как final2_step10.py): обучение <= 2024, val 2025, тест 2026 (цель в конце ряда может отсутствовать)."""
    if step != 10:
        return _split0(A, h, step)
    t = L.tcol(h)
    tr = A[A.year < 2026]
    return tr[tr.year != 2025].dropna(subset=[t]), tr[tr.year == 2025].dropna(subset=[t]), A[A.year == 2026], 2025, 2026


L.split = _split10
PCOLS = ('num_leaves', 'learning_rate', 'bagging_fraction', 'feature_fraction', 'min_data_in_leaf', 'reg_alpha', 'reg_lambda',
         'max_depth', 'subsample', 'colsample_bytree', 'min_child_weight')


def points(c, steps, seeds=None):
    """[(model, step, params, seed, rank)]. seeds - замена seed (порог по 3 seed: 43 44; для h=1 - только точка ранга 1)."""
    out = []
    if seeds and c['pts'] == 'bv2b_top5':
        S = pd.read_csv(R / 'Bv2b_selected_h1.csv')
        g = pd.concat([pd.read_csv(R / f'Bv2b_grid_{x}.csv') for x in ('xgb40', 'xgb100', 'lgbm100')])
        grid = {(r['model'], int(r['point'])): {k: r[k] for k in PCOLS} for _, r in g.iterrows()}
        for m in c['models']:
            for s in steps:
                pnt = int(S[(S.model == m) & (S.step == min(s, 9))].point.iloc[0])
                out += [(m, s, grid[(m, pnt)], sd, 1) for sd in seeds]
        return out
    if c['pts'] == 'bv2b_top5':
        g = pd.concat([pd.read_csv(R / f'Bv2b_grid_{x}.csv') for x in ('xgb40', 'xgb100', 'lgbm100')])
        grid = {(r['model'], int(r['point'])): {k: r[k] for k in PCOLS} for _, r in g.iterrows()}
        S = pd.read_csv(R / 'Bv2b_selected_h1.csv')
        for m in c['models']:
            T = pd.read_parquet(R / f'Bv2b_{m}_h1_top5_val.parquet').groupby(['step', 'rank']).point.first()
            for s in steps:
                if s <= 6:
                    out += [(m, s, grid[(m, int(T[(s, r)]))], 42, r) for r in range(1, 6)]
                else:
                    out.append((m, s, grid[(m, int(S[(S.model == m) & (S.step == min(s, 9))].point.iloc[0]))], 42, 1))
    else:
        P = L.selected_params(c['pts'], c['h'])
        for m in c['models']:
            for s in steps:
                for sd in (seeds or c['seeds']):
                    out.append((m, s, P[(m, min(s, 6))], sd, 1))
    return out


def job(a):
    cand, m, s, p, seed, rank = a
    c = CANDS[cand]
    A, cols = G['A'][cand], G['cols'][cand]
    cfg = c['cfg']
    btag = 'Bv2b'
    D = L.step_data(A, cols, c['h'], s, m, cfg, btag)
    if cfg.get('mono_sign') == -1:
        D['mono'] = [-x for x in D['mono']]
    ref = L.make_ref(m, A, cols, c['h'], cfg, btag) if s > 1 else None
    pv, pt, bi, _, _ = L.fit(m, p, D, cfg, seed=seed, ref=ref)
    fr = []
    for part, pred in (('va', pv), ('te', pt)):
        d = D['parts'][part]
        fr.append(pd.DataFrame({'date': d['date'].values, 'ticker': d['ticker'].values, 'year': d['year'].values, 'step': s, 'model': m,
                                'seed': seed, 'rank': rank, 'rv_actual': d[L.tcol(c['h'])].values, 'rv_pred': np.clip(np.exp(pred), 1e-10, None)}))
    return cand, m, s, seed, rank, bi, fr[0], fr[1]


def prepare(cands, harx_tag):
    G['A'], G['cols'] = {}, {}
    cache = {}
    for cand in cands:
        c = CANDS[cand]
        A, FC = L.load_base(None, c['blocks'], c['h'])
        A2, cols, _, _ = L.build_features(A, FC, c['h'], c['cfg'])
        if c['h'] not in cache:
            terms = HX.default_terms(c['h'], harx_tag != 'HARX0')
            cache[c['h']] = HX.ln_all_steps(c['h'], terms)
        A2 = A2.merge(cache[c['h']], on=['date', 'ticker'], how='left', validate='one_to_one')
        assert len(A2) == len(A) and A2['hx1'].notna().all()
        G['A'][cand], G['cols'][cand] = A2, cols


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cands', nargs='+', default=list(CANDS))
    ap.add_argument('--steps', type=int, nargs='+', default=[1, 2, 3, 4, 5, 6])
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--control', action='store_true', help='без подмены (сверка с сохранёнными финалистами)')
    ap.add_argument('--harx', default='HARX', help='HARX (слагаемые владельца + Д17.3) | HARX0')
    ap.add_argument('--suffix', default='')
    ap.add_argument('--seeds', type=int, nargs='*', default=None)
    a = ap.parse_args()
    G['control'] = a.control
    t0 = time.time()
    prepare(a.cands, a.harx)
    jobs = [(cand, *pt) for cand in a.cands for pt in points(CANDS[cand], a.steps, a.seeds)]
    print(f'{len(jobs)} задач, подготовка {time.time() - t0:.0f} c', flush=True)
    res = []
    with mp.get_context('fork').Pool(a.workers) as pool:
        for r in pool.imap_unordered(job, jobs):
            res.append(r)
            print(f'{r[0]} {r[1]} шаг {r[2]} seed {r[3]} rank {r[4]}: деревьев {r[5]} ({time.time() - t0:.0f} c)', flush=True)
    for cand in a.cands:
        tag = f'D16_{cand}{a.suffix}' + ('_control' if a.control else '')
        V = pd.concat([r[6] for r in res if r[0] == cand], ignore_index=True)
        T = pd.concat([r[7] for r in res if r[0] == cand], ignore_index=True)
        V.to_parquet(R / f'{tag}_val.parquet', index=False)
        d = R / 'test_locked' / tag
        d.mkdir(parents=True, exist_ok=True)
        T.to_parquet(d / f'{tag}_test.parquet', index=False)
        pd.DataFrame([dict(cand=r[0], model=r[1], step=r[2], seed=r[3], rank=r[4], trees=r[5]) for r in res if r[0] == cand]).to_csv(
            R / 'D16' / f'{tag}_trees.csv', index=False)
    print(f'готово {time.time() - t0:.0f} c', flush=True)


if __name__ == '__main__':
    main()
