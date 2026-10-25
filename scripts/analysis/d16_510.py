#!/usr/bin/env python3
"""Третий круг (сводный блок владельца 26.10, 00:55): Д16.10 — размер обёртки по валидации; Д16.5 — чистый бустинг с регрессорами HAR-X как признаками.
Правила третьего круга: отбор — OOS 2017–2020 (шаги 2–5, Б0.1 вне выборки: s² по val-году шага t−1); проверка — 2021 (шаг 6);
критерии c1–c4 — на 2016–2021 (шаги 1–6; на шаге 1 предыдущего val-года нет — s² шага 1 по его же остаткам, одинаково для всех моделей);
порог по 3 seed (42/43/44): вариант принимается, если среднее по трём seed лучше текущего ≥ 1% и лучше на каждом seed.
Подтверждающий набор (шаги 7–9) и тест здесь НЕ оцениваются: шаги 7–9 и тест пишутся в файлы ног (тест — только в data/regen/test_locked/).

Д16.10 (`refit10`, `wrap10`): кандидаты — текущие финалисты (h=1 Д8.12, h=5 D12_h5/F2m_h5, h=22 D12_h22) и V0 каждого h (h=1 D12_h1_V0;
h=5/22 — V0-улучшенный Д15, строка «бустинг»). Сетка одинаковая для всех: точек P ∈ {3, 5, 10} (топ-P точек поиска шага, seed 42),
seed S ∈ {3, 5} (лучшая точка, seed 42…42+S−1); члены обёртки модели = топ-P ∪ seeds; плюс вариант «2» — среднее LGBM и XGB того же размера.
Недостающие члены: seed 45/46 лучшей точки (шаги 1–9) и, для h=1, точки ранга 6–10 ЛГК-поиска B v2b (шаги 1–6) — `refit10`;
точки 6–10 optuna-кандидатов — из сохранённых `{T}_allpts_*` (без обучения). На шагах 7–9 топ-P нет — точка шага 6 (как final2_wrap).
python3 scripts/analysis/d16_510.py refit10 --h 22 [--cands D12_h22 D15_h22] [--workers 4]
python3 scripts/analysis/d16_510.py wrap10 --h 22

Д16.5 (`harx_matrix`, `tune165`, `gate165`): признаки HAR-X (модуль harx.py агента B) добавляются к матрице v3; база — чистый бустинг Д15
(h=5 — D15_h5: LGBM, цель ln(RV_avg/RV_m) × RV_m, mono −1; h=22 — база V0; h=1 — V0), прогноз HAR в признаки НЕ входит; optuna-30 (d15_tune)."""
import argparse, json, multiprocessing as mp, os, sys, time
from pathlib import Path

R0 = Path('/Users/nilysenok/Desktop/Старое/PhD_Vol_trading/data/regen')
os.environ.setdefault('L2_FEAT', str(R0 / 'features_pit_v3.parquet'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import l2_lib as L
from tune_L2 import SPACEB

R = L.REGEN
K = ['date', 'ticker', 'step']
M = ('lgbm', 'xgb')
# кандидат -> (вид, cfg, блоки, V0?, префикс файлов с allpts)
C10 = {
    1: {'D12_h1_A_d812': ('h1', {'harf': True}, '', False), 'D12_h1_V0': ('h1', {}, '', True)},
    5: {'D12_h5': ('opt', {'v9': True, 'harf': True, 'huber': True}, 'd7', False),
        'F2m_h5': ('opt', {'v9': True, 'harf': True, 'mono': True, 'mono_sign': -1}, 'd7', False),
        'D15_h5': ('opt', {'rel': True, 'mono': True, 'mono_sign': -1}, '', True),
        'L2v3_h5': ('opt', {'v9': True, 'harf': True}, 'd7', False)},  # добавлен по указанию дирижёра 01:12 (прошёл подтверждающий)
    22: {'D12_h22': ('opt', {'v9': True, 'linear_lambda': 1}, '', False), 'D15_h22': ('opt', {}, '', True)},
}
PGRID, SGRID = (3, 5, 10), (3, 5)
G = {}


def prm_cast(r, m):
    return {k: (int(r[k]) if sc in ('int', 'logint') else float(r[k])) for k, _, _, sc in SPACEB[m]}


def step_data(m, s):
    cfg = G['cfg']
    D = L.step_data(G['A'], G['cols'], G['h'], s, m, cfg, 'Bv2b')
    if cfg.get('mono_sign') == -1:
        D['mono'] = [-x for x in D['mono']]
    if cfg.get('rel'):  # как d15_tune: 17 строк 2014 г. с rv_m = NaN -> смещение = медиана (только обучение)
        for k, o in D['off'].items():
            if np.isnan(o).any():
                D['off'][k] = np.where(np.isnan(o), np.nanmedian(o), o)
    return D


_REF = {}


def job(a):
    """Одна единица = (модель, шаг) со всеми её членами: step_data и бины окна шага 1 строятся один раз и переиспользуются между seed/точками
    (указание владельца 26.10: такты за память без изменения расчёта; L.fit детерминирован при nthread=1)."""
    m, s, items = a
    D = step_data(m, s)
    if s > 1 and m not in _REF:
        _REF[m] = L.make_ref(m, G['A'], G['cols'], G['h'], G['cfg'], 'Bv2b')
    res = []
    for p, seed, rank, pnt in items:
        pv, pt, bi, _, _ = L.fit(m, p, D, G['cfg'], seed=seed, ref=_REF.get(m) if s > 1 else None)
        out = []
        for part, pred in (('va', pv), ('te', pt)):
            d = D['parts'][part]
            out.append(pd.DataFrame({'date': d['date'].values, 'ticker': d['ticker'].values, 'year': d['year'].values, 'step': s, 'point': pnt,
                                     'seed': seed, 'rank': rank, 'rv_actual': d[L.tcol(G['h'])].values, 'rv_pred': np.clip(np.exp(pred), 1e-10, None)}))
        res.append((m, s, seed, rank, bi, out[0], out[1]))
    return res


def load_cand(T, h):
    kind, cfg, blocks, _ = C10[h][T]
    if kind == 'h1':
        A, FC = L.load_base(R / 'features_pit_v3.parquet')
    else:
        A, FC = L.load_base(None, tuple(x for x in blocks.split(',') if x), h)
    A2, cols, _, _ = L.build_features(A, FC, h, cfg)
    G.update(A=A2, cols=cols, cfg=cfg, h=h)


def tasks10(T, h, steps_max):
    kind = C10[h][T][0]
    out = []
    if kind == 'h1':
        import d12_h1 as H1
        grid = H1.grid()
        S = pd.read_csv(R / 'Bv2b_selected_h1.csv')
        for m in M:
            ap = pd.read_parquet(R / f'Bv2b_allpts_{m}_h1_val.parquet')
            ap = ap[ap.step <= 6]
            q = ap.groupby(['step', 'point']).apply(lambda g: L.qlike(g.rv_actual.values, g.rv_pred.values), include_groups=False).rename('q').reset_index()
            for s in range(1, steps_max + 1):
                sp = int(S[(S.model == m) & (S.step == s)].point.iloc[0])
                out += [(m, s, grid[(m, sp)], sd, 1, sp) for sd in (45, 46)]
                if s <= 6:
                    rk = q[q.step == s].sort_values('q').point.tolist()
                    assert rk[0] == sp
                    out += [(m, s, grid[(m, int(pt))], 42, r + 1, int(pt)) for r, pt in enumerate(rk) if 5 <= r < 10]
    else:
        sel = pd.read_csv(R / f'{T}_selected_h{h}.csv')
        for r in sel[sel.step <= steps_max].to_dict('records'):
            out += [(r['model'], int(r['step']), prm_cast(r, r['model']), sd, 1, int(r['trial'])) for sd in (45, 46)]
    return out


def tasks_lag(T, h):
    """Честная оценка размера (↺, после смещения seed 42): на val-году шага s (2-6) члены обёртки — точки и seed шага s−1
    (топ-10 точек поиска шага s−1, seed 42; лучшая точка шага s−1, seed 43-46), обученные на окне шага s."""
    kind = C10[h][T][0]
    out = []
    if kind == 'h1':
        import d12_h1 as H1
        grid = H1.grid()
        for m in M:
            ap = pd.read_parquet(R / f'Bv2b_allpts_{m}_h1_val.parquet')
            ap = ap[ap.step <= 5]
            q = ap.groupby(['step', 'point']).apply(lambda g: L.qlike(g.rv_actual.values, g.rv_pred.values), include_groups=False).rename('q').reset_index()
            for s in range(2, 7):
                rk = [int(x) for x in q[q.step == s - 1].sort_values('q').point.tolist()[:10]]
                out += [(m, s, grid[(m, pt)], 42, r + 1, pt) for r, pt in enumerate(rk)]
                out += [(m, s, grid[(m, rk[0])], sd, 1, rk[0]) for sd in (43, 44, 45, 46)]
    else:
        tr = pd.read_csv(R / f'{T}_trials.csv')
        for m in M:
            t = tr[(tr.model == m) & (~tr.pruned)].sort_values(['step', 'qlike_val'])
            for s in range(2, 7):
                rk = t[t.step == s - 1].head(10).to_dict('records')
                out += [(m, s, prm_cast(r, m), 42, i + 1, int(r['trial'])) for i, r in enumerate(rk)]
                out += [(m, s, prm_cast(rk[0], m), sd, 1, int(rk[0]['trial'])) for sd in (43, 44, 45, 46)]
    return out


def refitlag(a):
    h = a.h
    for T in (a.cands or list(C10[h])):
        t0 = time.time()
        load_cand(T, h)
        tk = tasks_lag(T, h)
        _REF.clear()
        units = {}
        for m, st, p, sd, rk, pnt in tk:
            units.setdefault((m, st), []).append((p, sd, rk, pnt))
        print(f'{time.strftime("%T")} {T} h={h}: точки шага t−1, задач {len(tk)}, единиц {len(units)}, воркеров {a.workers}', flush=True)
        with mp.get_context('fork').Pool(a.workers) as P:
            res = [x for r in P.map(job, [(m, st, it) for (m, st), it in units.items()], chunksize=1) for x in r]
        for m in M:
            pd.concat([x[5] for x in res if x[0] == m]).to_parquet(R / f'D16_10_h{h}_{T}_{m}_lag_val.parquet', index=False)  # только val шагов 2-6
        pd.DataFrame([dict(cand=T, model=x[0], step=x[1], seed=x[2], rank=x[3], best_iteration=x[4]) for x in res]).to_csv(
            R / f'D16_10_h{h}_{T}_lag_trees.csv', index=False)
        print(f'{time.strftime("%T")} {T} h={h}: готово за {time.time() - t0:.0f} с', flush=True)


def members_lag(T, h, m, k='val'):
    """Члены для честной оценки: шаги 2-6 — точки/seed шага t−1 (refitlag), шаг 1 — свои (s² для шага 2)."""
    Mm = members(T, h, m, 'val')
    lg = pd.read_parquet(R / f'D16_10_h{h}_{T}_{m}_lag_val.parquet')
    st1 = lambda d: d[d.step == 1]
    pts = pd.concat([st1(Mm['pts']), lg[lg.seed == 42][K + ['rank', 'rv_pred']]], ignore_index=True)
    seeds = pd.concat([st1(Mm['seeds']), lg[lg['rank'] == 1][K + ['seed', 'rv_pred']]], ignore_index=True)
    return dict(base=Mm['base'][Mm['base'].step <= 6], pts=pts, seeds=seeds)


def refit10(a):
    h = a.h
    lock = R / 'test_locked' / f'D16_10_h{h}'
    lock.mkdir(parents=True, exist_ok=True)
    for T in (a.cands or list(C10[h])):
        t0 = time.time()
        load_cand(T, h)
        tk = tasks10(T, h, a.steps_max)
        _REF.clear()
        print(f'{time.strftime("%T")} {T} h={h}: задач {len(tk)}, воркеров {a.workers}', flush=True)
        units = {}
        for m, st, p, sd, rk, pnt in tk:
            units.setdefault((m, st), []).append((p, sd, rk, pnt))
        U = sorted(units.items(), key=lambda kv: -len(kv[1]))
        with mp.get_context('fork').Pool(a.workers) as P:
            res = [x for r in P.map(job, [(m, st, it) for (m, st), it in U], chunksize=1) for x in r]
        for m in M:
            for nm, f in (('seeds4546', lambda x: x[2] in (45, 46)), ('top610', lambda x: x[2] == 42 and x[3] >= 6)):
                rr = [x for x in res if x[0] == m and f(x)]
                if not rr:
                    continue
                pd.concat([x[5] for x in rr]).to_parquet(R / f'D16_10_h{h}_{T}_{m}_{nm}_val.parquet', index=False)
                pd.concat([x[6] for x in rr]).to_parquet(lock / f'D16_10_h{h}_{T}_{m}_{nm}_test.parquet', index=False)
        pd.DataFrame([dict(cand=T, model=x[0], step=x[1], seed=x[2], rank=x[3], best_iteration=x[4]) for x in res]).to_csv(
            R / f'D16_10_h{h}_{T}_refit_trees.csv', index=False)
        print(f'{time.strftime("%T")} {T} h={h}: готово за {time.time() - t0:.0f} с', flush=True)


# ---------------------------------------------------------------- оценка третьего круга
def s2(e, code):
    if code in ('rob', 'smrob'):
        return float((1.4826 * np.median(np.abs(e - np.median(e)))) ** 2)
    return float(np.var(e))


def year_q(V, code='ord'):
    """V: [step, rv_actual, rv_pred] шагов 1..6 -> {шаг: QLIKE с Б0.1}; шаг k>=2 — s² по val шагов до k−1 (sm/smrob — три последних);
    шаг 1 — s² по своим остаткам (вне выборки нельзя; одинаково для всех моделей, только для критериев на 2016)."""
    v = V[(V.rv_actual > 1e-12) & (V.step <= 6)]
    Gs = {st: g for st, g in v.groupby('step')}
    E = {st: np.log(g.rv_actual.values + 1e-10) - np.log(g.rv_pred.values) for st, g in Gs.items()}
    out = {}
    for st, g in Gs.items():
        if code == 'raw':
            F = 1.0
        else:
            src = [st] if st == 1 else [j for j in range(max(1, st - 3) if code in ('sm', 'smrob') else st - 1, st) if j in E]
            F = np.exp(s2(np.concatenate([E[j] for j in src]), code) / 2)
        out[st] = L.qlike(g.rv_actual.values, g.rv_pred.values * F)
    return out


def summ(q):
    return dict(oos1720=float(np.mean([q[s] for s in (2, 3, 4, 5)])), y2021=q[6], dev1721=float(np.mean([q[s] for s in range(2, 7)])),
                **{f'q{2015 + s}': q[s] for s in range(1, 7)})


def harj_val(h):
    """HAR-J на v3 (эталон H1 и c1): МНК на ln, обучение на окне без val-года; val шагов 1-6."""
    p = R / f'D16_10_harj_h{h}_val.parquet'
    if p.exists():
        return pd.read_parquet(p)
    A, FC = L.load_base(R / 'features_pit_v3.parquet')
    hc = L.harj_cols(A)
    out = []
    for s in range(1, 7):
        tp, vi, _, _, _ = L.split(A, h, s)
        pv = L.har_ln(tp, [vi], h, hc)[0]
        out.append(pd.DataFrame({'date': vi['date'].values, 'ticker': vi['ticker'].values, 'step': s, 'rv_actual': vi[L.tcol(h)].values,
                                 'rv_pred': np.exp(pv)}))
    d = pd.concat(out, ignore_index=True)
    d.to_parquet(p, index=False)
    return d


def members(T, h, m, k):
    """-> dict: 'pts' — DataFrame[K, rank, rv_pred] топ-10 точек seed 42 (шаги 1-6); 'seeds' — [K, seed, rv_pred] лучшая точка seed 42..46 (шаги 1-9);
    base — [K, year, rv_actual]."""
    kind = C10[h][T][0]
    d = R if k == 'val' else R / 'test_locked' / T
    dl = R if k == 'val' else R / 'test_locked' / f'D16_10_h{h}'
    if kind == 'h1' and k == 'test':
        d = R / 'test_locked' / 'D12_h1' / T
    single = pd.read_parquet(d / f'{T}_{m}_h{h}_{k}.parquet')
    single = single[single.seed == 42] if 'seed' in single else single
    base = single[K + ['year', 'rv_actual']]
    s42 = single[K + ['rv_pred']].assign(seed=42)
    s4344 = pd.read_parquet(d / f'{T}_{m}_h{h}_seeds4344_{k}.parquet')[K + ['seed', 'rv_pred']]
    s4546 = pd.read_parquet(dl / f'D16_10_h{h}_{T}_{m}_seeds4546_{k}.parquet')[K + ['seed', 'rv_pred']]
    seeds = pd.concat([s42, s4344, s4546], ignore_index=True)
    t5 = pd.read_parquet(d / f'{T}_{m}_h{h}_top5_{k}.parquet')[K + ['rank', 'rv_pred']]
    if kind == 'h1':
        t610 = pd.read_parquet(dl / f'D16_10_h{h}_{T}_{m}_top610_{k}.parquet')[K + ['rank', 'rv_pred']]
    else:
        ap = pd.read_parquet(d / f'{T}_allpts_{m}_h{h}_{k}.parquet')
        tr = pd.read_csv(R / f'{T}_trials.csv')
        tr = tr[(tr.model == m) & (~tr.pruned)].sort_values(['step', 'qlike_val'])
        tr['rank'] = tr.groupby('step').cumcount() + 1
        rk = tr[(tr['rank'] >= 6) & (tr['rank'] <= 10)][['step', 'trial', 'rank']]
        t610 = ap.merge(rk, on=['step', 'trial'])[K + ['rank', 'rv_pred']]
    pts = pd.concat([t5, t610], ignore_index=True)
    return dict(base=base, pts=pts, seeds=seeds)


def wrap_pred(Mm, P, S):
    """Среднее RV членов: топ-P точек (seed 42, шаги 1-6; на шагах без топа — точка шага 6 = seed 42 single) ∪ лучшая точка seed 42..42+S−1."""
    pts = Mm['pts'][Mm['pts']['rank'] <= P][K + ['rv_pred']]
    has = set(pts.step.unique())
    sd = Mm['seeds'][Mm['seeds'].seed < 42 + S]
    seeds = sd[~((sd.seed == 42) & sd.step.isin(has))][K + ['rv_pred']]  # seed 42 лучшей точки = ранг 1 топа
    f = pd.concat([pts, seeds], ignore_index=True).groupby(K, as_index=False).rv_pred.mean()
    out = Mm['base'].merge(f, on=K, how='left', validate='one_to_one')
    assert out.rv_pred.notna().all()
    return out


def avg2(a, b):
    m = a.merge(b[K + ['rv_pred']], on=K, suffixes=('_l', '_x'), validate='one_to_one')
    return m.assign(rv_pred=(m.rv_pred_l + m.rv_pred_x) / 2).drop(columns=['rv_pred_l', 'rv_pred_x'])


def crit(qd, qh, qs):
    """c1–c4 на шагах 1–6 (2016–2021) для дизайна V0 (одна компонента): qd — дизайн, qh — HAR-J, qs — {модель: одиночная (seed 42)} с той же s²."""
    st = range(1, 7)
    hy = np.array([qd[s] for s in st]); har = np.array([qh[s] for s in st])
    sing = np.minimum.reduce([np.array([q[s] for s in st]) for q in qs.values()])
    worst = (hy / har).max()
    worst_single = min((np.array([q[s] for s in st]) / har).max() for q in qs.values())
    return dict(c1=bool((hy <= 1.02 * har).all()), c2=bool((hy <= 1.02 * sing).all()), c3=bool(worst < worst_single), c4=True,
                worst_ratio=worst, worst_ratio_single=worst_single)


def wrap10(a):
    h = a.h
    code = a.code
    H = year_q(harj_val(h), code)
    rows = []
    out_files = {}
    for T in (a.cands or list(C10[h])):
        MM = {m: (members_lag if a.lag else members)(T, h, m, 'val') for m in M}
        single = {m: wrap_pred(MM[m], 1, 1) for m in M}
        qs = {m: year_q(single[m], code) for m in M}
        prim = min(M, key=lambda m: summ(qs[m])['oos1720'])
        cur = str(pd.read_csv(R / 'final2' / f'wrap2_{T}_h{h}.csv', dtype={'final': str}).final.iloc[0]) if (R / 'final2' / f'wrap2_{T}_h{h}.csv').exists() else 'n/a'
        variants = [('single', 1, 1, False)] + [(f'P{P}S{S}', P, S, two) for P in PGRID for S in SGRID for two in (False, True)]
        # текущая обёртка final2 (топ-5 и/или 3 seed и/или 2) — для сравнения
        cp = {'none': (1, 1, False), '2': (1, 1, True), '1a': (5, 1, False), '1b': (1, 3, False), '1a+1b': (5, 3, False), '1a+2': (5, 1, True),
              '1b+2': (1, 3, True), '1a+1b+2': (5, 3, True)}.get(cur)
        if cp:
            variants.append(('final2:' + cur, *cp))
        preds = {}
        for nm, P, S, two in variants:
            W = {m: wrap_pred(MM[m], P, S) for m in M}
            pr = avg2(W['lgbm'], W['xgb']) if two else W[prim]
            preds[nm] = pr
            q = year_q(pr, code)
            rows.append(dict(cand=T, h=h, v0=C10[h][T][3], variant=nm, P=P, S=S, lgb_xgb=two, model='lgbm+xgb' if two else prim, code=code,
                             final2_wrap=cur, **summ(q), **crit(q, H, qs if two else {m: qs[m] for m in M if m != prim})))
        # справка: одиночная лучшая точка с другим seed (43-46) — seed 42 выбранной точки смещён в свою пользу (точка подобрана на том же val-году)
        fq = []
        for sd in (43, 44, 45, 46):
            ss = MM[prim]['seeds']
            one = MM[prim]['base'].merge(ss[ss.seed == sd][K + ['rv_pred']], on=K, how='inner')
            fq.append(year_q(one, code))
        qf = {st: float(np.mean([f[st] for f in fq])) for st in fq[0]}
        rows.append(dict(cand=T, h=h, v0=C10[h][T][3], variant='single_s43-46', P=1, S=1, lgb_xgb=False, model=prim, code=code, final2_wrap=cur,
                         **summ(qf), **crit(qf, H, {m: qs[m] for m in M if m != prim})))
        i0 = [i for i, r in enumerate(rows) if r['cand'] == T]
        sub = pd.DataFrame([rows[i] for i in i0], index=i0)
        # выбор (↺ к первому прогону): размер — argmin OOS 2017-2020 по сетке {3,5,10}x{3,5} (± среднее LGBM и XGB);
        # принимается вместо текущей обёртки final2 (у D15 — без обёртки), если не хуже её и на OOS 2017-2020, и на 2021 (проверка);
        # одиночная seed 42 в отбор как «размер 1» не входит отдельно: она смещена (точка подобрана на том же val-году) — см. single_s43-46.
        pool = sub[sub.variant.str.match(r'^(P\d+S\d+|single)$' if a.lag else r'^P\d+S\d+$')]  # lag: смещения нет — «без обёртки» в выборе
        gi = pool.oos1720.idxmin()
        g = sub.loc[gi]
        ri = sub[sub.variant.str.startswith('final2')].index[0] if cp else sub[sub.variant == 'single'].index[0]
        ref = sub.loc[ri]
        adopt = bool(g.oos1720 <= ref.oos1720 and g.y2021 <= ref.y2021)
        bi = gi if adopt else ri
        best = sub.loc[bi]
        ok21 = bool(g.y2021 <= ref.y2021)
        for i in i0:
            rows[i]['grid_best'] = i == gi
            rows[i]['reference'] = i == ri
            rows[i]['chosen'] = i == bi
            rows[i]['adopt_grid'] = adopt if i == gi else None
        out_files[T] = (best.variant, int(best.P), int(best.S), bool(best.lgb_xgb), prim, ok21)
    df = pd.DataFrame(rows)
    df['vs_single'] = df.oos1720 / df.groupby('cand').oos1720.transform(lambda s: s.iloc[0]) - 1
    sfx = '_lag' if a.lag else ''
    df.to_csv(R / f'D16_10_h{h}_wrap_grid{sfx}.csv', index=False, float_format='%.6g')
    # ноги wrap10 (val шагов 1-9; тест — test_locked/D16_10_h{h}/): слот основной модели = выбранная обёртка, другой слот — та же внутримодельная
    lock = R / 'test_locked' / f'D16_10_h{h}'
    for T, (nm, P, S, two, prim, ok21) in out_files.items():
        for k in ('val', 'test'):
            MM = {m: members(T, h, m, k) for m in M}
            W = {m: wrap_pred(MM[m], P, S) for m in M}
            slots = dict(W)
            if two:
                slots[prim] = avg2(W['lgbm'], W['xgb'])
            dd = R if k == 'val' else lock
            for m in M:
                slots[m].assign(wrap=nm + ('+2' if two and m == prim else '')).sort_values(K).to_parquet(dd / f'D16_10_h{h}_{T}_{m}_h{h}_wrap10{sfx}_{k}.parquet', index=False)
        print(f'{T} h={h}: итог {nm} ({"lgbm+xgb" if two else prim}); лучший по сетке на 2021 не хуже текущей: {ok21}', flush=True)
    pd.set_option('display.width', 250)
    print(df[df.grid_best | df.reference | df.chosen | df.variant.str.startswith('single')][['cand', 'variant', 'model', 'oos1720', 'y2021', 'dev1721', 'c1', 'c2', 'c3', 'grid_best', 'reference', 'chosen']].to_string(), flush=True)


# ---------------------------------------------------------------- Д16.5
def harx_matrix(a):
    """Матрица features_pit_v3_d165.parquet = v3 + регрессоры HAR-X (слагаемые harx.build_terms, без прогноза HAR и без ln_sig1) + config.json.
    Слагаемые одинаковы для всех h (уровень тикер-даты, только данные <= t; тест «будущее не влияет» — у агента B, leak_test_harx.csv)."""
    import hashlib, importlib.util
    spec = importlib.util.spec_from_file_location('harx', a.harx_py)
    HX = importlib.util.module_from_spec(spec); spec.loader.exec_module(HX)
    A, _ = HX.load()
    X = HX.build_terms(A, a.cols)
    X = pd.concat([A[['date', 'ticker']], X], axis=1)
    X['date'] = pd.to_datetime(X['date'])
    base = pd.read_parquet(R / 'features_pit_v3.parquet')
    cfg = json.load(open(R / 'features_pit_v3.config.json'))
    newc = {c: f'hx_{c}' for c in a.cols}
    dt = base['date'].dtype
    b = base.assign(date=pd.to_datetime(base['date']))
    out = b.merge(X.rename(columns=newc), on=['date', 'ticker'], how='left', validate='one_to_one')
    out['date'] = out['date'].astype(dt)
    na = out[list(newc.values())].isna().mean()
    p = R / 'features_pit_v3_d165.parquet'
    out.to_parquet(p, index=False)
    sha = hashlib.sha256(open(a.harx_py, 'rb').read()).hexdigest()
    cfg2 = dict(cfg, feature_cols=list(cfg['feature_cols']) + list(newc.values()), d165=dict(harx_py=str(a.harx_py), harx_sha256=sha, cols=list(a.cols)))
    json.dump(cfg2, open(p.with_suffix('.config.json'), 'w'), ensure_ascii=False, indent=1)
    print(f'{p.name}: строк {len(out)} (v3 {len(base)}), добавлено {list(newc.values())}; harx.py sha256 {sha[:12]}; доля NaN: '
          + ', '.join(f'{k} {v:.3f}' for k, v in na.items()), flush=True)


def gate(tag, cur, h, lo=2, hi=5):
    """Порог третьего круга: метрика seed s — средний OOS QLIKE шагов lo..hi лучшего из XGB/LGBM с Б0.1 ord; seed 42 — выбранная точка, 43/44 — та же точка."""
    def met(T):
        out = {}
        for sd in (42, 43, 44):
            ev = {}
            for m in M:
                v = pd.read_parquet(R / f'{T}_{m}_h{h}_val.parquet') if sd == 42 else pd.read_parquet(R / f'{T}_{m}_h{h}_seeds4344_val.parquet')
                v = v[(v.seed == sd)] if 'seed' in v else v
                q = year_q(v[v.step <= 6], 'ord')
                ev[m] = (float(np.mean([q[s] for s in range(lo, hi + 1)])), q[6], float(np.mean([q[s] for s in range(1, 7) if s >= 2])))
            bm = min(ev, key=lambda m: ev[m][0])
            out[sd] = (ev[bm][0], ev[bm][1], bm, ev)
        return out
    a, b = met(tag), met(cur)
    ma, mb = np.mean([a[s][0] for s in a]), np.mean([b[s][0] for s in b])
    each = all(a[s][0] < b[s][0] for s in a)
    acc = (ma <= 0.99 * mb) and each
    return dict(tag=tag, cur=cur, h=h, mean_tag=ma, mean_cur=mb, rel=ma / mb - 1, each_seed=each, accepted=acc,
                **{f'{x}_s{s}': d[s][0] for x, d in (('tag', a), ('cur', b)) for s in d},
                **{f'{x}_2021_s{s}': d[s][1] for x, d in (('tag', a), ('cur', b)) for s in d},
                **{f'{x}_model_s{s}': d[s][2] for x, d in (('tag', a), ('cur', b)) for s in d})


def gate165(a):
    r = gate(a.tag, a.cur, a.h)
    H = year_q(harj_val(a.h), 'ord')
    for T, pre in ((a.tag, 'tag'), (a.cur, 'cur')):  # c1–c4 на 2016–2021: лучшая модель seed 42 против HAR-J и другой модели (Б0.1 ord)
        qs = {}
        for m in M:
            v = pd.read_parquet(R / f'{T}_{m}_h{a.h}_val.parquet')
            v = v[v.seed == 42] if 'seed' in v else v
            qs[m] = year_q(v[v.step <= 6], 'ord')
        bm = r[f'{pre}_model_s42']
        c = crit(qs[bm], H, {m: qs[m] for m in M if m != bm})
        r.update({f'{pre}_{k}': v for k, v in c.items()})
        r.update({f'{pre}_q{2015 + s}': qs[bm][s] for s in range(1, 7)})
    r.update({f'harj_q{2015 + s}': H[s] for s in range(1, 7)}, harj_oos1720=float(np.mean([H[s] for s in (2, 3, 4, 5)])))
    p = R / f'D16_5_h{a.h}_gate.csv'
    df = pd.DataFrame([r])
    if p.exists():
        df = pd.concat([pd.read_csv(p), df], ignore_index=True)
    df.to_csv(p, index=False, float_format='%.6g')
    print(json.dumps({k: (round(float(v), 5) if isinstance(v, (float, np.floating)) else (bool(v) if isinstance(v, (bool, np.bool_)) else v)) for k, v in r.items()}, ensure_ascii=False), flush=True)


def designs(a):
    """models/designs/d16_10_h{h}.yaml и d16_5_h{h}.yaml — по сводкам D16_10_h{h}_wrap_grid.csv и D16_5_h{h}_gate.csv."""
    import yaml
    od = Path(__file__).resolve().parents[2] / 'models' / 'designs'
    for h in a.hs:
        g = R / f'D16_10_h{h}_wrap_grid.csv'
        if g.exists():
            d = pd.read_csv(g, dtype={'final2_wrap': str})
            c = d[d.chosen == True]
            gb = d[d.grid_best == True].set_index('cand')
            y = dict(name=f'd16_10_h{h}', block='Д16.10 размер обёртки (третий круг, 26.10)', h=h, grid=dict(points=list(PGRID), seeds=list(SGRID), lgbm_xgb=[False, True]),
                     select='размер = мин. OOS QLIKE 2017–2020 (шаги 2–5, Б0.1 ord вне выборки) по сетке; принимается вместо текущей обёртки final2 (D15 — без обёртки), если не хуже её на 2017–2020 и на 2021; c1–c4 на 2016–2021',
                     candidates={r.cand: dict(v0=bool(r.v0), chosen=r.variant, model=r.model, points=int(r.P), seeds=int(r.S), lgbm_xgb=bool(r.lgb_xgb),
                                              cfg=C10[h][r.cand][1], blocks=C10[h][r.cand][2], final2_wrap=str(r.final2_wrap), grid_best=dict(variant=gb.loc[r.cand, 'variant'], model=gb.loc[r.cand, 'model'], oos1720=round(float(gb.loc[r.cand, 'oos1720']), 6), y2021=round(float(gb.loc[r.cand, 'y2021']), 6)),
                                              oos1720=round(float(r.oos1720), 6), y2021=round(float(r.y2021), 6), c1=bool(r.c1), c2=bool(r.c2), c3=bool(r.c3),
                                              val=f'data/regen/D16_10_h{h}_{r.cand}_{{lgbm,xgb}}_h{h}_wrap10_val.parquet',
                                              test=f'data/regen/test_locked/D16_10_h{h}/D16_10_h{h}_{r.cand}_{{lgbm,xgb}}_h{h}_wrap10_test.parquet')
                                 for r in c.itertuples()})
            yaml.safe_dump(y, open(od / f'd16_10_h{h}.yaml', 'w'), allow_unicode=True, sort_keys=False)
        g = R / f'D16_5_h{h}_gate.csv'
        if g.exists():
            r = pd.read_csv(g).iloc[-1]
            cfg = {5: {'rel': True, 'mono': True, 'mono_sign': -1}, 22: {}, 1: {}}[h]
            y = dict(name=f'd16_5_h{h}', block='Д16.5 чистый бустинг + регрессоры HAR-X как признаки (без прогноза HAR)', h=h, tag=r.tag, compared_with=r.cur,
                     matrix='data/regen/features_pit_v3_d165.parquet', harx_terms=['ln_rvi2', 'ln_rvm_imoex', 'ln_rvm_mkt', 'lhar_1', 'lhar_5', 'lhar_22', 'pc1'],
                     cfg=cfg, search='optuna TPE-30 seed 42, пространство B v2 (d15_tune.py)', selected=f'data/regen/{r.tag}_selected_h{h}.csv',
                     gate='среднее по seed 42/43/44 OOS 2017–2020 лучше текущей ≥ 1% и лучше на каждом seed',
                     mean_tag=round(float(r.mean_tag), 6), mean_cur=round(float(r.mean_cur), 6), rel=round(float(r.rel), 5), accepted=bool(r.accepted),
                     test=f'data/regen/test_locked/{r.tag}/')
            yaml.safe_dump(y, open(od / f'd16_5_h{h}.yaml', 'w'), allow_unicode=True, sort_keys=False)


def kandidat(a):
    """yaml кандидата для сборки третьего круга (формат docs/otchety/2026-10-26_final3.md): data/regen/final3/kandidaty/<boost>.yaml."""
    import yaml
    od = R / 'final3' / 'kandidaty'
    od.mkdir(parents=True, exist_ok=True)
    y = dict(boost=a.boost, h=a.h, semeystvo=a.semeystvo, rychag=a.rychag, opisanie=a.opisanie, regen='data/regen', legs=['single', 'top5', 'seeds3'],
             odinochnaya=False, chistyy=bool(a.chistyy), test_dir=f'data/regen/test_locked/{a.boost}', osnova=a.osnova, n_variantov=a.n)
    if a.step10:  # шаг 10 — final2_step10.py (формат {boost}_{m}_h{h}_s10_val.parquet)
        assert all((R / 'final2' / 'step10' / f'{a.boost}_{m}_h{a.h}_s10_val.parquet').exists() for m in M)
        y.update(step10_dir='data/regen/final2/step10', step10_lock='data/regen/test_locked/final2/step10')
    for m in M:  # контроль наличия: val шагов 1-9 и тест шагов 1-9
        v = pd.read_parquet(R / f'{a.boost}_{m}_h{a.h}_val.parquet', columns=['step'])
        t = pd.read_parquet(R / 'test_locked' / a.boost / f'{a.boost}_{m}_h{a.h}_test.parquet', columns=['step'])
        assert sorted(v.step.unique()) == list(range(1, 10)) and sorted(t.step.unique()) == list(range(1, 10)), (m, a.boost)
    yaml.safe_dump(y, open(od / f'{a.boost}.yaml', 'w'), allow_unicode=True, sort_keys=False)
    print(f'{od / (a.boost + ".yaml")} записан', flush=True)


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest='cmd', required=True)
    p = sp.add_parser('refit10'); p.add_argument('--h', type=int, required=True); p.add_argument('--cands', nargs='*'); p.add_argument('--workers', type=int, default=4)
    p.add_argument('--steps-max', type=int, default=9)
    p = sp.add_parser('wrap10'); p.add_argument('--h', type=int, required=True); p.add_argument('--cands', nargs='*'); p.add_argument('--code', default='ord')
    p.add_argument('--lag', action='store_true')
    p = sp.add_parser('refitlag'); p.add_argument('--h', type=int, required=True); p.add_argument('--cands', nargs='*'); p.add_argument('--workers', type=int, default=2)
    p = sp.add_parser('harx_matrix'); p.add_argument('--harx-py', required=True)
    p.add_argument('--cols', nargs='+', default=['ln_rvi2', 'ln_rvm_imoex', 'ln_rvm_mkt', 'lhar_1', 'lhar_5', 'lhar_22', 'pc1'])
    p = sp.add_parser('gate165'); p.add_argument('--h', type=int, required=True); p.add_argument('--tag', required=True); p.add_argument('--cur', required=True)
    p = sp.add_parser('kandidat'); p.add_argument('--boost', required=True); p.add_argument('--h', type=int, required=True)
    p.add_argument('--semeystvo', required=True); p.add_argument('--rychag', default='Д16.5'); p.add_argument('--opisanie', required=True)
    p.add_argument('--osnova', required=True); p.add_argument('--chistyy', type=int, default=1); p.add_argument('--n', type=int, default=1)
    p.add_argument('--step10', type=int, default=0)
    p = sp.add_parser('designs'); p.add_argument('--hs', type=int, nargs='+', default=[1, 5, 22])
    a = ap.parse_args()
    {'refitlag': refitlag, 'kandidat': kandidat, 'designs': designs, 'refit10': refit10, 'wrap10': wrap10, 'harx_matrix': harx_matrix, 'gate165': gate165}[a.cmd](a)


if __name__ == '__main__':
    main()
