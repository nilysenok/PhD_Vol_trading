#!/usr/bin/env python3
"""Д17 (третий круг): рычаги на точках текущих финалов без нового подбора.
Д17.1 — остаточный ученик как GLM с офсетом: objective gamma на уровне RV (LGBM gamma / XGB reg:gamma), init_score / base_margin =
ln(прогноз основы) + s²/2 (поправка Б0.1 основы; s² — дисперсия остатков ln основы на обучающем окне шага, внутри выборки — без val-года),
ранняя остановка по нативной gamma-метрике на val-годе шага; итоговый прогноз БЕЗ отдельной поправки (код 'raw').
Основа: HAR-J (8 регрессоров, ln, МНК на окне — l2_lib.har_ln); D17_BASE=harx — HAR-X (harx.py агента d16-harx) на всех строках шага из
data/regen/D17/harx_base_h{h}.parquet (d17_1_harx_base.py); --base-file — произвольный внешний ln-прогноз (date, ticker, step, base_ln).
Точки: h=1 — B v2b (лучшая точка шага; top5 по флагу), h=5/22 — D12_h{h}_selected_h{h}.csv (optuna-30 v3).
Правило третьего круга (владелец 26.10, 00:55): отбор по OOS шагов 2-5 (val 2017-2020), шаг 6 (2021) — отложенный, «не хуже текущего».
Выход: val -> data/regen/D17/h{h}/{name}_val.parquet (шаги 1-6, seed 42/43/44; seed 42 — шаги 1-9 при --steps9);
тест -> data/regen/test_locked/D17_h{h}/{name}_test.parquet (не открывается и не оценивается).
Общие файлы конвейера (l2_lib, tune_L2_optuna, gibrid_v17) не меняются: офсет Б0.1 (b01) и клип метки gamma — обёртка step_data ниже
(подменяет L.step_data в процессе; так же работает d17_1_tune.py — optuna-30 через tune_L2_optuna).
Выход для интеграции: data/regen/D17_1_h{h}_<канд>_{xgb,lgbm}_h{h}{,_seeds4344}_val.parquet (формат gibrid_io.load_boost).
python3 scripts/analysis/d17_1.py --h 22 --cands G22 [--seeds 42 43 44] [--workers 8]"""
import argparse, json, multiprocessing as mp, os, sys, time
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import l2_lib as L

R = L.REGEN
OUT = R / 'D17'
CANDS = {
    1: {'G1': {'obj': 'gamma', 'v9': True, 'b01': True},                     # Д17.1: HAR-J как init_score к чистому бустингу (замена Д8.12)
        'G1f': {'obj': 'gamma', 'v9': True, 'b01': True, 'harf': True}},    # то же + ln HAR-J как признак
    5: {'G5': {'obj': 'gamma', 'v9': True, 'b01': True, 'harf': True}},     # Д17.1 вместо V9-остатка (Huber не совместим с gamma)
    22: {'G22': {'obj': 'gamma', 'v9': True, 'b01': True, 'linear_lambda': 1},
         'G22n': {'obj': 'gamma', 'v9': True, 'b01': True}},                # без linear_tree
}
_SD0 = L.step_data
BASE_KIND = os.environ.get('D17_BASE', 'harj')  # harj | harx
_HB = {}


def harx_base(h, step):
    if h not in _HB:
        _HB[h] = pd.read_parquet(R / 'D17' / f'harx_base_h{h}.parquet')
    B = _HB[h]
    b = B[B.step == step]
    return {k: b[b.part == k][['date', 'ticker', 'base_ln']] for k in ('tr', 'va', 'te')}


def step_data(A, cols, h, step, model, cfg, base_tag, **kw):
    """Обёртка l2_lib.step_data: b01 — офсет ln основы + s²/2 (s² — остатки ln основы на обучающем окне шага, внутри выборки);
    gamma — метка клипуется снизу 1e-8."""
    D = _SD0(A, cols, h, step, model, cfg, base_tag, **kw)
    D['s2'] = np.nan
    if BASE_KIND == 'harx' and cfg.get('v9'):  # основа HAR-X: ln-прогноз на строках шага (обучение — внутри выборки), d17_1_harx_base.py
        B = harx_base(h, step)
        for k, d in D['parts'].items():
            v = d[['date', 'ticker']].merge(B[k], on=['date', 'ticker'], how='left')['base_ln'].values
            assert not np.isnan(v).any(), (h, step, k)
            if cfg.get('harf') and D['names'][-1] == 'har_ln':  # признак harf — тоже ln основы
                D['X'][k][:, -1] = v.astype(np.float32)
            D['off'][k] = v
    if cfg.get('b01'):
        s2 = float(np.var(np.log(np.clip(D['y']['tr'], 1e-10, None)) - D['off']['tr']))
        D['off'] = {k: v + s2 / 2 for k, v in D['off'].items()}
        D['s2'] = s2
    if cfg.get('obj') == 'gamma':
        D['y'] = {k: np.clip(v, 1e-8, None) for k, v in D['y'].items()}
    return D


L.step_data = step_data
G = {}


def params(h):
    if h == 1:
        import d12_h1 as D1
        gr = D1.grid()
        S = pd.read_csv(R / 'Bv2b_selected_h1.csv')
        T = {m: pd.read_parquet(R / f'Bv2b_{m}_h1_top5_val.parquet').groupby(['step', 'rank']).point.first() for m in ('xgb', 'lgbm')}
        P = {}
        for m in ('xgb', 'lgbm'):
            for s in range(1, 10):
                sp = int(S[(S.model == m) & (S.step == s)].point.iloc[0])
                P[(m, s, 1)] = gr[(m, sp)]
                if s <= 6:
                    for r in range(2, 6):
                        P[(m, s, r)] = gr[(m, int(T[m][(s, r)]))]
        return P, 'Bv2b'
    sp = L.selected_params(R / f'D12_h{h}_selected_h{h}.csv', h)
    return {(m, s, 1): sp[(m, min(s, 6))] for m in ('xgb', 'lgbm') for s in range(1, 10)}, 'Bv2b'


def ext_base(D, h, step):
    """Внешняя основа (ln прогноз на строках окна): файл с колонками date, ticker, step, base_ln (обучение — внутри выборки)."""
    B = G['base']
    out = {}
    for k, d in D['parts'].items():
        b = B[B.step == step]
        v = d[['date', 'ticker']].merge(b[['date', 'ticker', 'base_ln']], on=['date', 'ticker'], how='left')['base_ln'].values
        if np.isnan(v).any():
            v = np.where(np.isnan(v), D['off'][k], v)  # нет строки во внешней основе -> HAR-J
        out[k] = v
    return out


def job(a):
    name, m, s, rank, seed = a
    cfg = CANDS[G['h']][name]
    A, cols = G['A'], G['cols'][name]
    h = G['h']
    p = G['P'][(m, s, rank)]
    D = L.step_data(A, cols, h, s, m, cfg, G['btag'])
    s2 = D['s2']
    if G.get('base') is not None and cfg.get('v9'):  # внешняя основа (HAR-X): офсет заменяется, Б0.1 — по её остаткам на окне
        D['off'] = ext_base(D, h, s)
        if cfg.get('b01'):
            s2 = float(np.var(np.log(np.clip(D['y']['tr'], 1e-10, None)) - D['off']['tr']))
            D['off'] = {k: v + s2 / 2 for k, v in D['off'].items()}
    ref = L.make_ref(m, A, cols, h, cfg, G['btag']) if s > 1 else None
    pv, pt, bi, _, _ = L.fit(m, p, D, cfg, seed=seed, ref=ref)
    fr = []
    for k, pr, yr in (('va', pv, D['vy']), ('te', pt, D['ty'])):
        d = D['parts'][k]
        fr.append(pd.DataFrame({'date': d.date.values, 'ticker': d.ticker.values, 'year': yr, 'step': s, 'model': m, 'rank': rank, 'seed': seed,
                                'rv_actual': d[L.tcol(h)].values, 'rv_pred': np.clip(np.exp(pr), 1e-10, None), 'base_ln': D['off'][k]}))
    return name, m, s, rank, seed, bi, s2, fr[0], fr[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--h', type=int, required=True); ap.add_argument('--cands', nargs='+', required=True)
    ap.add_argument('--seeds', nargs='+', type=int, default=[42, 43, 44]); ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--top5', action='store_true', help='h=1: ещё ранги 2-5 (seed 42, шаги 1-6)')
    ap.add_argument('--steps9', action='store_true', help='seed 42: шаги 7-9 (точка шага 6) — для финальной сборки; val 7-9 не оценивается')
    ap.add_argument('--base-file', default=None, help='внешняя основа (HAR-X): parquet date,ticker,step,base_ln')
    ap.add_argument('--suffix', default='')
    ap.add_argument('--matrix', default=None)
    a = ap.parse_args()
    h = a.h
    t0 = time.time()
    blocks = ('d7',) if h == 5 else ()
    A, FC = L.load_base(Path(a.matrix) if a.matrix else None, blocks, h)
    P, btag = params(h)
    if BASE_KIND == 'harx':
        harx_base(h, 1)  # основа читается один раз до fork (copy-on-write)
    G.update(h=h, A=A, P=P, btag=btag, cols={}, base=pd.read_parquet(a.base_file) if a.base_file else None)
    for c in a.cands:
        A2, cols, _, _ = L.build_features(A, FC, h, CANDS[h][c])
        G['cols'][c] = cols
    tasks = []
    for c in a.cands:
        for m in ('lgbm', 'xgb'):
            for s in range(9 if a.steps9 else 6, 0, -1):
                for sd in a.seeds:
                    if s <= 6 or sd == 42:
                        tasks.append((c, m, s, 1, sd))
                if a.top5 and s <= 6:
                    tasks += [(c, m, s, r, 42) for r in range(2, 6)]
    print(f'{time.strftime("%T")} h={h} {L.FEAT.name}{"+d7" if blocks else ""} кандидаты {a.cands} моделей {len(tasks)}', flush=True)
    with mp.get_context('fork').Pool(a.workers) as Pl:
        res = Pl.map(job, tasks, chunksize=1)
    od = OUT / f'h{h}'; od.mkdir(parents=True, exist_ok=True)
    lock = R / 'test_locked' / f'D17_h{h}'; lock.mkdir(parents=True, exist_ok=True)
    for c in a.cands:
        rr = [r for r in res if r[0] == c]
        V = pd.concat([r[7] for r in rr], ignore_index=True)
        T = pd.concat([r[8] for r in rr], ignore_index=True)
        nm = c + a.suffix
        V.to_parquet(od / f'{nm}_val.parquet', index=False)
        T.to_parquet(lock / f'{nm}_test.parquet', index=False)
        meta = pd.DataFrame([dict(model=r[1], step=r[2], rank=r[3], seed=r[4], best_iteration=r[5], s2_base=r[6]) for r in rr])
        meta.to_csv(od / f'{nm}_meta.csv', index=False)
        boost = f'D17_1_h{h}_{nm}'
        lk = R / 'test_locked' / boost; lk.mkdir(parents=True, exist_ok=True)
        for m in ('xgb', 'lgbm'):
            for k, F, d in (('val', V, R), ('test', T, lk)):
                F1 = F[(F.model == m) & (F['rank'] == 1)].drop(columns=['model', 'base_ln'])
                F1[F1.seed == 42].sort_values(['step', 'date', 'ticker']).to_parquet(d / f'{boost}_{m}_h{h}_{k}.parquet', index=False)
                F1[F1.seed != 42].sort_values(['seed', 'step', 'date', 'ticker']).to_parquet(d / f'{boost}_{m}_h{h}_seeds4344_{k}.parquet', index=False)
                if a.top5:  # нога top5 (формат D12_h1: ранги 1-5, шаги 1-6, seed 42; шаги 7-9 гибрид берёт из single)
                    F5 = F[(F.model == m) & (F.seed == 42) & (F.step <= 6)].drop(columns=['model', 'base_ln'])
                    F5.sort_values(['step', 'rank', 'date', 'ticker']).to_parquet(d / f'{boost}_{m}_h{h}_top5_{k}.parquet', index=False)
        print(f'{c}: сохранено {od / (nm + "_val.parquet")}', flush=True)
    print(f'{time.strftime("%T")} готово за {time.time() - t0:.0f} с', flush=True)


if __name__ == '__main__':
    main()
