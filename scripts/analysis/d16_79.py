#!/usr/bin/env python3
"""Третий круг, Д16.7 и Д16.9 (h=5, h=22): обратная подгонка HAR-X <-> остаточный бустинг и переменные коэффициенты HAR-X.
Точки бустинга — без нового подбора: h=22 — D12_h22 (V9 + linear_tree λ=1), h=5 — F2m_h5 (V9 + Д8.12 + Д7 + mono −1); seed 42/43/44.
HAR-X — модуль агента B (harx.py: слагаемые build_terms, эластичная сеть fit_en с нештрафуемыми членами HAR-J); гиперпараметры сети шага —
по внутренней валидации шага (обучение < vy−1, оценка на vy−1, MSE логов), как harx.forecast_step.

Д16.7 (вариант bf): итерация 0 — HAR-X на ln y, бустинг на ln y − ln HAR-X (= Д16.1 V9 на HAR-X при тех же точках); итерация i >= 1 —
HAR-X (те же α, l1_ratio шага) на ln y − f_{i−1}(x) (f — внутривыборочный прогноз бустинга на обучающем окне), бустинг заново на
ln y − ln HAR-X_i (ES на val-годе шага). До 3 итераций. Правило остановки по шагу: после итерации i >= 1 стоп, если |Δ val-QLIKE| < 0,2%
к итерации i−1 (val-год шага — тот же, что у ранней остановки бустинга). Справочно сохраняются все K = 0..3 (выбор K — по OOS шагов 2–5).

Д16.9 (вариант pw): кусочно-линейный HAR-X по листьям дерева состояний (MOB: дерево глубины 2–3 по признакам состояния; в листьях — ридж-
поправка к коэффициентам глобального HAR-X на регрессорах HAR-X [ln HAR-J-члены, слагаемые HAR-X], штраф λ на отклонения, свободный член
без штрафа). Расщепление — минимум суммы SSE ридж-регрессий в дочерних узлах; пороги — дециль 0,1..0,9 узла; лист >= 10% строк окна.
Глубина ∈ {0, 2, 3} и λ — по внутренней валидации шага (как сеть HAR-X); 0 = HAR-X без дерева. ↺ LightGBM linear_tree не используется:
в нём линейная модель листа строится на признаках расщеплений своей ветви, разделить «признаки состояния — для порогов, регрессоры HAR-X —
линейные» нельзя; ручная реализация честно делает именно это.
Признаки состояния (только данные <= t): перцентиль RV_m тикера по расширяющемуся окну, ln RVI² со срезом 18:50 (rvi_1850, 10-мин. свечи),
дни после 22.06.2020 и после 24.02.2022 (0 до даты), дни с последнего дня J/RV >= 0,3 тикера (потолок 250).
Затем — остаточный бустинг на ln y − ln PW-HARX (те же точки, 3 seed) и PW-HARX без бустинга (Б0.1).

Запуск: python3 scripts/analysis/d16_79.py --h 22 --task bf|pw|pwlin|leak [--steps 1-6|1-9] [--seeds 42 43 44] [--workers 4]
Тест (шаги 1–9) — только data/regen/test_locked/D16_7_h{h}/, D16_9_h{h}/ (не открывать); val шагов 7–9 не оценивается и не печатается."""
import os
os.environ.setdefault('L2_FEAT', '/Users/nilysenok/Desktop/Старое/PhD_Vol_trading/data/regen/features_pit_v3.parquet')
for _k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_k] = '1'
import argparse, json, multiprocessing as mp, sys, time
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
if not (HERE / 'harx.py').exists() and os.environ.get('HARX_DIR'):  # до слияния модуля агента B — путь к его копии
    sys.path.insert(1, os.environ['HARX_DIR'])
import l2_lib as L
import harx as HX

OUT = L.REGEN / 'D16_79'
CONF = {22: dict(cfg={'v9': True, 'linear_lambda': 1}, blocks='', tag='D12_h22'),
        5: dict(cfg={'v9': True, 'harf': True, 'mono': True, 'mono_sign': -1}, blocks='d7', tag='F2m_h5'),
        1: dict(cfg={'v9': True}, blocks='', tag='D12_h1_B_b1', params='Bv2b')}  # h=1: финал (Д8.12) не V9 -> V9 (B_b1) на точках Bv2b
# основы, прошедшие подтверждающий во втором круге (указание дирижёра 01:19): выбор через D16_BASE=<тег>; выходы с суффиксом _<тег>
CONF.update({'D12_h5': dict(cfg={'v9': True, 'harf': True, 'huber': True}, blocks='d7', tag='D12_h5'),
             'L2v3_h5': dict(cfg={'v9': True, 'harf': True}, blocks='d7', tag='L2v3_h5')})
BASE = os.environ.get('D16_BASE', '')
SFX = f'_{BASE}' if BASE else ''


def conf(h):
    return CONF[BASE] if BASE else CONF[h]


TOL = 0.002
NIT = 3
LAMS = (10.0, 100.0, 1000.0, 10000.0)
DEPTHS = (0, 2, 3)
G = {}


# ---------------------------------------------------------------- признаки состояния
def state_features(GA, rvi=None):
    """По датам <= t строки. GA — матрица harx.load() (features_pit_v3)."""
    tk = GA['ticker']
    order = GA.sort_values(['ticker', 'date']).index
    pct = GA.loc[order, 'rv_m'].groupby(tk.loc[order]).transform(lambda s: s.expanding(min_periods=60).rank(pct=True)).reindex(GA.index)
    s = L.rvi_1850() if rvi is None else rvi
    r = pd.merge_asof(pd.DataFrame({'date': pd.to_datetime(sorted(GA['date'].unique()))}),
                      s.rename('rvi').rename_axis('date').reset_index().assign(date=lambda d: pd.to_datetime(d['date'])), on='date')
    rv = GA['date'].map(dict(zip(r['date'], np.log(np.clip(r['rvi'].values, 1e-6, None) ** 2))))
    d = pd.to_datetime(GA['date']).values
    da22 = np.clip((d - np.datetime64('2022-02-24')) / np.timedelta64(1, 'D'), 0, None)
    da20 = np.clip((d - np.datetime64('2020-06-22')) / np.timedelta64(1, 'D'), 0, None)
    j = ((GA['jump'] / GA['rv'].replace(0, np.nan)) >= 0.3).astype(float)

    def since(x):
        idx = np.arange(len(x)); last = pd.Series(np.where(x.values > 0, idx, np.nan), index=x.index).ffill()
        return (pd.Series(idx, index=x.index) - last).fillna(250).clip(upper=250)
    dsj = j.loc[order].groupby(tk.loc[order]).transform(since).reindex(GA.index)
    S = pd.DataFrame({'st_rvm_pct': pct.fillna(-1.0), 'st_ln_rvi2': rv.fillna(np.log(30.0 ** 2)), 'st_days_after_20200622': da20,
                      'st_days_after_20220224': da22, 'st_days_since_jump': dsj}, index=GA.index)
    return S


def leak_test(GA, cuts=('2018-06-30', '2020-12-31', '2023-06-30'), seed=20261026):
    """Строки date > t: числовые входы — бутстрап значений до t, ряд RVI после t — шум; признаки состояния строк <= t должны совпасть побайтно."""
    rng = np.random.default_rng(seed)
    rvi = L.rvi_1850()
    base = state_features(GA, rvi)
    rows = []
    for t in cuts:
        t = pd.Timestamp(t)
        B = GA.copy(); fut = pd.to_datetime(B.date) > t
        for c in ('rv_m', 'rv', 'jump'):
            B.loc[fut, c] = rng.choice(B.loc[~fut, c].dropna().values, fut.sum())
        r2 = rvi.copy(); f = pd.to_datetime(r2.index) > t
        r2[f] = rng.uniform(10, 120, f.sum())
        X2 = state_features(B, r2)
        for n in base.columns:
            a1, a2 = base.loc[~fut, n].values, X2.loc[~fut, n].values
            rows.append(dict(t=str(t.date()), feature=n, n=int((~fut).sum()), mismatch=int((~((a1 == a2) | (np.isnan(a1) & np.isnan(a2)))).sum())))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- HAR-X шага
def zfill(ref):
    """Слагаемые HAR-X по всем строкам; пропуски — средним слагаемого по строкам ref (обучение подгонки), затем 0 (как harx.forecast_step)."""
    Zr = G['Zr']
    with np.errstate(all='ignore'):
        mu = np.nanmean(Zr[ref], axis=0)
    Z = np.where(np.isnan(Zr), mu, Zr)
    return np.nan_to_num(Z, nan=0.0)


def hx_hparams(step):
    """α, l1_ratio сети HAR-X шага — внутренняя валидация (как harx.forecast_step)."""
    GA, H, y, ok = G['GA'], G['H'], G['y'], G['ok']
    vy = 2015 + step
    it, iv = np.where((GA.year < vy - 1) & ok)[0], np.where((GA.year == vy - 1) & ok)[0]
    Z = zfill(it)
    res = []
    for l1 in HX.L1S:
        for al in HX.ALPHAS:
            p, _ = HX.fit_en(H[it], Z[it], y[it], al, l1)
            res.append((al, l1, float(np.mean((y[iv] - p(H[iv], Z[iv])) ** 2))))
    p, _ = HX.fit_en(H[it], Z[it], y[it], 0.0, 1.0)
    res.append((0.0, 0.0, float(np.mean((y[iv] - p(H[iv], Z[iv])) ** 2))))
    return min(res, key=lambda r: r[2])


# ---------------------------------------------------------------- Д16.9: дерево состояний с ридж-поправкой HAR-X в листьях
def _ridge(X, r, lam):
    """r ~ [1, X] (X стандартизован), штраф lam на коэффициенты при X. -> (beta, sse)."""
    Xc = np.column_stack([np.ones(len(X)), X])
    P = np.eye(Xc.shape[1]) * lam; P[0, 0] = 0
    b = np.linalg.solve(Xc.T @ Xc + P, Xc.T @ r)
    e = r - Xc @ b
    return b, float(e @ e)


def grow(S, X, r, lam, depth, minleaf):
    """Жадное MOB-дерево: узел = {'leaf': beta} | {'f', 'thr', 'l', 'r'}."""
    b, sse = _ridge(X, r, lam)
    node = {'leaf': b, 'n': len(r), 'sse': sse}
    if depth == 0 or len(r) < 2 * minleaf:
        return node
    best = None
    for f in range(S.shape[1]):
        s = S[:, f]
        for thr in np.unique(np.quantile(s, np.arange(0.1, 0.91, 0.1))):
            m = s <= thr
            nl = int(m.sum())
            if nl < minleaf or len(r) - nl < minleaf:
                continue
            tot = _ridge(X[m], r[m], lam)[1] + _ridge(X[~m], r[~m], lam)[1]
            if best is None or tot < best[0]:
                best = (tot, f, float(thr))
    if best is None or best[0] > sse * (1 - 1e-3):
        return node
    _, f, thr = best
    m = S[:, f] <= thr
    return {'f': f, 'thr': thr, 'n': len(r), 'l': grow(S[m], X[m], r[m], lam, depth - 1, minleaf), 'r': grow(S[~m], X[~m], r[~m], lam, depth - 1, minleaf)}


def tree_pred(node, S, X):
    out = np.zeros(len(X))
    if 'leaf' in node:
        return np.column_stack([np.ones(len(X)), X]) @ node['leaf']
    m = S[:, node['f']] <= node['thr']
    out[m] = tree_pred(node['l'], S[m], X[m]); out[~m] = tree_pred(node['r'], S[~m], X[~m])
    return out


def tree_str(node, names):
    if 'leaf' in node:
        return f"[n={node['n']}]"
    return f"({names[node['f']]}<={node['thr']:.4g} ? {tree_str(node['l'], names)} : {tree_str(node['r'], names)})"


def pw_fit(itr, al, l1, lam, depth):
    """Глобальный HAR-X (α, l1) на itr + дерево состояний на остатках. -> predict(idx) (ln), дерево."""
    H, y, S = G['H'], G['y'], G['S']
    Z = zfill(itr)
    p, _ = HX.fit_en(H[itr], Z[itr], y[itr], al, l1)
    XL = np.column_stack([H, Z])
    mu, sd = XL[itr].mean(0), XL[itr].std(0); sd[sd < 1e-12] = 1
    Xs = (XL - mu) / sd
    if depth == 0:
        return (lambda idx: p(H[idx], Z[idx])), None
    r = y[itr] - p(H[itr], Z[itr])
    tr = grow(S[itr], Xs[itr], r, lam, depth, max(3000, int(0.1 * len(itr))))
    return (lambda idx: p(H[idx], Z[idx]) + tree_pred(tr, S[idx], Xs[idx])), tr


def pw_step(step):
    """-> (step, ln PW-HARX по всем строкам GA с годом <= ty (NaN — прочие), журнал)."""
    GA, y, ok = G['GA'], G['y'], G['ok']
    vy, ty = 2015 + step, 2016 + step
    al, l1, _ = G['HP'][step]
    it, iv = np.where((GA.year < vy - 1) & ok)[0], np.where((GA.year == vy - 1) & ok)[0]
    res = []
    for d in DEPTHS:
        for lam in (LAMS if d else (0.0,)):
            pr, _ = pw_fit(it, al, l1, lam, d)
            res.append((d, lam, float(np.mean((y[iv] - pr(iv)) ** 2))))
    d, lam, mse = min(res, key=lambda r: r[2])
    itr = np.where((GA.year < vy) & ok)[0]
    pr, tr = pw_fit(itr, al, l1, lam, d)
    allx = np.where(GA.year <= ty)[0]
    out = np.full(len(GA), np.nan); out[allx] = pr(allx)
    log = dict(step=step, depth=d, lam=lam, inner_mse=mse, inner_mse_hx=[r for r in res if r[0] == 0][0][2],
               tree=tree_str(tr, list(G['S_names'])) if tr else '', grid=json.dumps([[a, b, round(c, 6)] for a, b, c in res]))
    return step, out, log


# ---------------------------------------------------------------- бустинг
def _pos(d):
    return d[['date', 'ticker']].merge(G['key'], on=['date', 'ticker'], how='left')['gi'].values


def task(a):
    variant, model, step, seed = a
    h, cfg = G['h'], G['cfg']
    A, cols = G['A'], G['cols']
    p = G['P'][(model, step)]
    D = L.step_data(A, cols, h, step, model, cfg, 'Bv2b')
    if cfg.get('mono_sign') == -1:
        D['mono'] = [-m for m in D['mono']]
    ref = G['REF'][model] if step > 1 else None  # бины окна шага 1 — один раз на модель (в родителе, раздаются через fork)
    pos = {k: _pos(D['parts'][k]) for k in ('tr', 'va', 'te')}
    assert all(not np.isnan(pos[k].astype(float)).any() for k in pos), 'строки шага не найдены в матрице HAR-X'
    H = G['H']
    Z = zfill(pos['tr'])  # пропуски слагаемых — среднее обучающего окна шага (как harx.forecast_step)
    ytr = D['ytr_ln']
    nte = len(D['X']['te'])
    Xte2 = np.vstack([D['X']['te'], D['X']['tr']])
    recs = []

    def boost(off):
        D2 = {**D, 'off': {**off, 'te': np.concatenate([off['te'], off['tr']])}, 'X': {**D['X'], 'te': Xte2}}
        pv, ptt, bi, _, _ = L.fit(model, p, D2, cfg, seed=seed, ref=ref)
        return pv, ptt[:nte], ptt[nte:], bi
    if variant == 'bf':
        al, l1, _ = G['HP'][step]
        pr, _ = HX.fit_en(H[pos['tr']], Z[pos['tr']], ytr, al, l1)
        for i in range(NIT + 1):
            off = {k: pr(H[pos[k]], Z[pos[k]]) for k in ('tr', 'va', 'te')}
            pv, pt, ptr, bi = boost(off)
            recs.append((i, pv, pt, off, bi))
            if i < NIT:
                pr, _ = HX.fit_en(H[pos['tr']], Z[pos['tr']], ytr - (ptr - off['tr']), al, l1)
    elif variant == 'hj':  # сверка обвязки: смещение — ln HAR-J из step_data (должно воспроизвести выходы базы seed 42)
        pv, pt, _, bi = boost(D['off'])
        recs.append((0, pv, pt, D['off'], bi))
    else:  # pw: смещение — ln PW-HARX шага
        lp = G['PW'][step]
        off = {k: lp[pos[k]] for k in ('tr', 'va', 'te')}
        pv, pt, _, bi = boost(off)
        recs.append((0, pv, pt, off, bi))
    frames = []
    for i, pv, pt, off, bi in recs:
        for k, pr_, yr in (('va', pv, D['vy']), ('te', pt, D['ty'])):
            d = D['parts'][k]
            frames.append(pd.DataFrame({'date': d.date.values, 'ticker': d.ticker.values, 'year': yr, 'step': step, 'model': model, 'seed': seed,
                                        'it': i, 'part': k, 'rv_actual': D['y'][k], 'rv_pred': np.clip(np.exp(pr_), 1e-10, None), 'off_ln': off[k], 'bi': bi}))
    return pd.concat(frames, ignore_index=True)


def stop_select(V):
    """Правило остановки по шагу (val-QLIKE без поправки, val-год шага): итерация i>=1 — стоп при |Δ| < 0,2%; иначе 3. -> {(model, seed, step): it}."""
    res = {}
    for (m, sd, st), g in V[(V.part == 'va') & (V.rv_actual > 1e-12)].groupby(['model', 'seed', 'step']):
        q = {i: L.qlike(gi.rv_actual.values, gi.rv_pred.values) for i, gi in g.groupby('it')}
        k = NIT
        for i in range(1, NIT + 1):
            if abs(q[i] / q[i - 1] - 1) < TOL:
                k = i; break
        res[(m, sd, st)] = k
    return res


def metrics(V, lo=2, hi=6):
    """val (шаги 1..6) одной модели-варианта -> QLIKE по шагам (ord/sm/smrob, s² по шагу t−1)."""
    ev = L.evaluate(V[['step', 'rv_actual', 'rv_pred']], lo, hi)
    return {c: ev[c][1] for c in ('raw', 'ord', 'sm', 'smrob')}


SEL, CHK = (2, 3, 4, 5), 6  # третий круг: отбор по шагам 2-5 (2017-2020), проверка — шаг 6 (2021)


def _ev(V):
    """V: val одной модели (шаги 1-6) -> {код: [q шагов 2..6]}."""
    return metrics(V[V.step <= 6])


def summary_rows(name, frames):
    """frames: {seed: {model: val df}} -> строки по seed и 'mean' (среднее метрик по seed) и 'seeds3' (среднее прогнозов по seed)."""
    rows = []
    per = {}
    for sd, fm in frames.items():
        for m, V in fm.items():
            ev = _ev(V)
            per[(sd, m)] = ev
            for c in ('ord', 'sm', 'smrob'):
                q = ev[c]
                rows.append(dict(name=name, seed=sd, model=m, code=c, **{f'q{2015 + s}': q[s - 2] for s in range(2, 7)},
                                 sel_2017_20=float(np.mean([q[s - 2] for s in SEL])), chk_2021=q[CHK - 2], dev_2017_21=float(np.mean(q))))
    R = pd.DataFrame(rows)
    M = R.groupby(['name', 'model', 'code'], as_index=False)[[c for c in R.columns if c.startswith(('q', 'sel', 'chk', 'dev'))]].mean().assign(seed='mean')
    seeds = sorted(frames)
    s3 = []
    for m in frames[seeds[0]]:
        Vs = pd.concat([frames[sd][m] for sd in seeds]).groupby(['date', 'ticker', 'step'], as_index=False).agg(rv_actual=('rv_actual', 'first'), rv_pred=('rv_pred', 'mean'))
        ev = _ev(Vs)
        for c in ('ord', 'sm', 'smrob'):
            q = ev[c]
            s3.append(dict(name=name, seed='seeds3', model=m, code=c, **{f'q{2015 + s}': q[s - 2] for s in range(2, 7)},
                           sel_2017_20=float(np.mean([q[s - 2] for s in SEL])), chk_2021=q[CHK - 2], dev_2017_21=float(np.mean(q))))
    return pd.concat([R, M, pd.DataFrame(s3)], ignore_index=True)


def base_frames(tag, h):
    """База — та же точка, V9 на HAR-J (выходы второго круга): seed 42 — single, 43/44 — seeds4344."""
    out = {}
    for m in ('lgbm', 'xgb'):
        s = pd.read_parquet(L.REGEN / f'{tag}_{m}_h{h}_val.parquet')
        ss = pd.read_parquet(L.REGEN / f'{tag}_{m}_h{h}_seeds4344_val.parquet')
        out.setdefault(42, {})[m] = s
        for sd in (43, 44):
            out.setdefault(sd, {})[m] = ss[ss.seed == sd]
    return out


def var_frames(V, pick):
    """V — выход task (часть va); pick(model, seed, step) -> итерация. -> {seed: {model: df}}."""
    out = {}
    for (m, sd), g in V.groupby(['model', 'seed']):
        sel = g[g.it == g.step.map(lambda st: pick(m, sd, st))]
        out.setdefault(int(sd), {})[m] = sel
    return out


def evaluate_h(h):
    tag = conf(h)['tag']
    R = [summary_rows(f'base_{tag}_V9_HARJ', base_frames(tag, h))]
    picks = {}
    fb = OUT / f'bf{SFX}_h{h}_s1-9.parquet'
    if fb.exists():
        V = pd.read_parquet(fb)
        Va = V[V.part == 'va']
        stp = stop_select(Va[Va.step <= 6])
        for K in range(NIT + 1):
            picks[f'bf_K{K}'] = (lambda K: (lambda m, sd, st: K))(K)
            R.append(summary_rows(f'bf_K{K}' + (' (=Д16.1 V9 на HAR-X)' if K == 0 else ''), var_frames(Va[Va.step <= 6], picks[f'bf_K{K}'])))
        picks['bf_stop'] = lambda m, sd, st: stp[(m, sd, st)]
        R.append(summary_rows('bf_stop', var_frames(Va[Va.step <= 6], picks['bf_stop'])))
        pd.DataFrame([dict(model=m, seed=sd, step=st, it=k) for (m, sd, st), k in stp.items()]).to_csv(OUT / f'bf{SFX}_h{h}_stop_iters.csv', index=False)
    fp = OUT / f'pw{SFX}_h{h}_s1-9.parquet'
    if fp.exists():
        V = pd.read_parquet(fp); Va = V[(V.part == 'va') & (V.step <= 6)]
        R.append(summary_rows('pw_V9 (Д16.9 + остаточный бустинг)', var_frames(Va, lambda m, sd, st: 0)))
    S = pd.concat(R, ignore_index=True)
    S.to_csv(OUT / f'summary{SFX}_h{h}.csv', index=False)
    # линейные: HAR-J, HAR-X (агент B), PW-HARX (Д16.9 без бустинга) — Б0.1 по шагу t-1
    lin = []
    GA, harj = HX.load()
    hv, ht, _ = HX.har_steps(h, GA, harj)
    hv, _ = HX.add_bias(hv, ht)
    srcs = {'HAR-J': hv, 'HAR-X (HARX, агент B)': pd.read_parquet(L.REGEN / f'HARX_h{h}_val.parquet')}
    if (L.REGEN / f'D16_9_h{h}_pwlin_val.parquet').exists():
        srcs['PW-HARX (Д16.9, без бустинга)'] = pd.read_parquet(L.REGEN / f'D16_9_h{h}_pwlin_val.parquet')
    for n, v in srcs.items():
        for c in ('sm', 'smrob', 'ord'):
            q = HX.dev_table(v, f'rv_pred_{c}')
            lin.append(dict(name=n, code=c, **{f'q{2015 + s}': q[s] for s in range(2, 7)}, sel_2017_20=np.mean([q[s] for s in SEL]), chk_2021=q[CHK]))
    pd.DataFrame(lin).to_csv(OUT / f'linear_h{h}.csv', index=False)
    return S, pd.DataFrame(lin), picks


def write_final(h, src, pick, name):
    """Выходы кандидата: data/regen/{name}_{m}_h{h}_val.parquet (seed 42, шаги 1-9), _seeds4344_val; тест — test_locked/{name}/ (не открывать)."""
    V = pd.read_parquet(OUT / f'{src}{SFX}_h{h}_s1-9.parquet')
    if src == 'bf' and pick == 'stop':
        stp = stop_select(V[V.part == 'va'])  # шаги 7-9: правило по val-году шага, как ранняя остановка; метрики не печатаются
        f = lambda m, sd, st: stp[(m, sd, st)]
    else:
        f = lambda m, sd, st: pick
    keep = V.it.values == np.array([f(m, sd, st) for m, sd, st in zip(V.model, V.seed, V.step)])
    V = V[keep]
    lock = L.REGEN / 'test_locked' / name; lock.mkdir(parents=True, exist_ok=True)
    cols = ['date', 'ticker', 'year', 'step', 'rv_actual', 'rv_pred', 'har_ln']
    for m in ('xgb', 'lgbm'):
        for part, d, sfx in (('va', L.REGEN, 'val'), ('te', lock, 'test')):
            g = V[(V.model == m) & (V.part == part)].rename(columns={'off_ln': 'har_ln'})
            g[g.seed == 42][cols].sort_values(['step', 'date', 'ticker']).to_parquet(d / f'{name}_{m}_h{h}_{sfx}.parquet', index=False)
            g[g.seed != 42][cols + ['seed']].assign(trial=-1, rank=1).sort_values(['seed', 'step', 'date', 'ticker']).to_parquet(
                d / f'{name}_{m}_h{h}_seeds4344_{sfx}.parquet', index=False)


def leak_pw(h, cuts=('2019-06-30', '2021-12-31'), seed=20261026):
    """Детектор (а) для модели Д16.9 (PW-HARX): после t входы модели (ln HAR-J-члены, слагаемые HAR-X, rv/rv_m/jump для признаков
    состояния, RVI) — бутстрап значений до t, цели строк с позицией > pos(t)−h — бутстрап; модель шага, чьи данные (обучение + цели на h
    вперёд) кончились не позже t, обязана дать побайтно те же ln-прогнозы на строках с датой <= t."""
    rng = np.random.default_rng(seed)
    GA = G['GA']
    base = {s: pw_step(s)[1] for s in range(1, 10)}
    keep = {k: G[k] for k in ('H', 'Zr', 'y', 'S')}
    d = pd.to_datetime(GA.date).values.astype('datetime64[ns]')
    ud = np.unique(d)
    rows = []
    rvi = L.rvi_1850()
    for t in cuts:
        t = np.datetime64(pd.Timestamp(t)).astype('datetime64[ns]')
        pt = np.searchsorted(ud, t, side='right') - 1
        fut, futy = d > t, np.searchsorted(ud, d) > pt - h
        B = GA.copy()
        for c in ('rv_m', 'rv', 'jump'):
            B.loc[fut, c] = rng.choice(B.loc[~fut, c].dropna().values, fut.sum())
        tc = f'rv_target_h{h}'
        B.loc[futy, tc] = rng.choice(B.loc[~futy, tc].dropna().values, futy.sum())  # цели в мире B — тоже шум (для самопроверки)
        r2 = rvi.copy(); f = pd.to_datetime(r2.index) > pd.Timestamp(t); r2[f] = rng.uniform(10, 120, f.sum())
        G['S'] = state_features(B, r2).values
        H2, Z2, y2 = keep['H'].copy(), keep['Zr'].copy(), keep['y'].copy()
        H2[fut] = H2[~fut][rng.integers(0, (~fut).sum(), fut.sum())]
        Z2[fut] = Z2[~fut][rng.integers(0, (~fut).sum(), fut.sum())]
        y2[futy] = y2[~futy][rng.integers(0, (~futy).sum(), futy.sum())]
        G.update(H=H2, Zr=Z2, y=y2)
        for s in range(1, 10):
            vy = 2015 + s
            last = np.searchsorted(ud, np.datetime64(f'{vy - 1}-12-31').astype('datetime64[ns]'), side='right') - 1
            if last + h > pt:
                continue
            o = pw_step(s)[1]
            m = (d <= t) & ~np.isnan(base[s])
            rows.append(dict(h=h, t=str(t)[:10], step=s, n=int(m.sum()), mismatch=int((base[s][m] != o[m]).sum())))
        G.update(keep)
    return pd.DataFrame(rows)


def hyb_table(h, boosts):
    """Гибрид (gibrid_v17, OOS): лучший дизайн по правилу у каждого выхода — QLIKE по годам 2017-2021, отбор 2017-2020, проверка 2021, c1-c4."""
    rows = []
    for b in boosts:
        d = OUT / f'gibrid_{b}'
        if not (d / f'ranking_{b}_h{h}.csv').exists():
            continue
        F = pd.read_csv(d / f'ranking_{b}_h{h}.csv')
        Dv = pd.read_csv(d / f'designs_val_{b}_h{h}.csv')
        # ключ третьего круга (final3, п. 3): число нарушенных c1-c4 (на шагах 2-6), затем средний OOS QLIKE 2017-2020 (шаги 2-5)
        sel = Dv[Dv.step.between(SEL[0], SEL[-1])].groupby('design').q_hyb.mean().rename('sel_q')
        F = F.merge(sel, left_on='design', right_index=True).sort_values(['n_failed', 'sel_q'], kind='stable').reset_index(drop=True)
        for role, r in (('по правилу', F.iloc[0]), ('лучший с бустингом', F[~F.family.isin(['V0_HAR', 'V0_HCJ'])].iloc[0])):
            g = Dv[(Dv.design == r.design) & Dv.step.between(2, 6)].sort_values('step')
            q = dict(zip(g.step, g.q_hyb)); qh = dict(zip(g.step, g.q_har))
            rows.append(dict(h=h, boost=b, role=role, design=r.design, failed='' if pd.isna(r.failed) else r.failed,
                             **{f'q{2015 + s}': q[s] for s in range(2, 7)}, sel_2017_20=np.mean([q[s] for s in SEL]), chk_2021=q[CHK],
                             dev_2017_21=np.mean([q[s] for s in range(2, 7)]), har_sel_2017_20=np.mean([qh[s] for s in SEL]), har_2021=qh[CHK]))
    T = pd.DataFrame(rows)
    T.to_csv(OUT / f'hybrid{SFX}_h{h}.csv', index=False)
    return T


def setup(h):
    c = conf(h)
    A, FC = L.load_base(None, tuple(x for x in c['blocks'].split(',') if x), h)
    A2, cols, _, _ = L.build_features(A, FC, h, c['cfg'])
    GA, harj = HX.load()
    terms = HX.default_terms(h)  # как выход HARX агента B (слагаемые владельца + кандидаты Д17.3 в той же сети)
    if os.environ.get('D16_TERMS'):
        terms = os.environ['D16_TERMS'].split(',')
    Xt = HX.build_terms(GA, terms)  # пропуски заполняются в zfill средним обучающих строк подгонки
    tcol = f'rv_target_h{h}'
    S = state_features(GA)
    G.update(h=h, cfg=c['cfg'], tag=c['tag'], A=A2, cols=cols, P=L.selected_params(str(L.REGEN / f"{c.get('params', c['tag'])}_selected_h{h}.csv"), h),
             GA=GA, harj=harj, terms=terms, H=GA_H(GA, harj), Zr=Xt[terms].values.astype('float64'),
             y=np.log(GA[tcol].values + 1e-10), ok=GA[tcol].notna().values, S=S.values, S_names=list(S.columns),
             key=GA[['date', 'ticker']].assign(gi=np.arange(len(GA))))


def GA_H(GA, harj):
    import gibrid_har as GH
    return GH.clean(np.log(GA[harj].values.clip(1e-10)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--h', type=int, required=True); ap.add_argument('--task', required=True, choices=['bf', 'hj', 'pw', 'pwlin', 'leak', 'leakpw', 'summ', 'final'])
    ap.add_argument('--final', nargs=3, default=None, metavar=('SRC', 'PICK', 'NAME'), help='final: bf|pw, K|stop, имя выхода')
    ap.add_argument('--steps', default='1-6'); ap.add_argument('--seeds', type=int, nargs='*', default=[42, 43, 44])
    ap.add_argument('--workers', type=int, default=4); ap.add_argument('--models', nargs='*', default=['lgbm', 'xgb'])
    a = ap.parse_args()
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    h = a.h
    if a.task == 'leak':
        GA, _ = HX.load()
        r = leak_test(GA)
        r.to_csv(OUT / 'leak_test_state.csv', index=False)
        print(r.to_string()); print('всего расхождений:', int(r.mismatch.sum())); return
    if a.task == 'summ':
        S, Lin, _ = evaluate_h(h)
        pd.set_option('display.width', 250)
        print(S[S.seed.isin(['mean', 'seeds3'])].round(4).to_string()); print(Lin.round(4).to_string())
        nm = (lambda x: f'{x}_{BASE}') if BASE else (lambda x: f'{x}_h{h}')
        T = hyb_table(h, [conf(h)['tag'], nm('D16_7') + '_K0', nm('D16_7'), nm('D16_9')])
        print(T.round(4).to_string()); return
    if a.task == 'final':
        src, pick, name = a.final
        write_final(h, src, pick if pick == 'stop' else int(pick), name); print('записано', name); return
    setup(h)
    if a.task == 'leakpw':
        hp = pd.read_csv(OUT / f'hx_hparams_h{h}.csv')
        G['HP'] = {int(r.step): (r.alpha, r.l1, r.inner_mse) for r in hp.itertuples()}
        r = leak_pw(h)
        r.to_csv(OUT / f'leak_test_pw_h{h}.csv', index=False)
        print(r.to_string()); print('всего расхождений:', int(r.mismatch.sum())); return
    lo, hi = map(int, a.steps.split('-'))
    steps = list(range(lo, hi + 1))
    hpf = OUT / f'hx_hparams_h{h}.csv'
    if hpf.exists() and set(steps) <= set(pd.read_csv(hpf).step):
        hp = pd.read_csv(hpf)
    else:
        with mp.get_context('fork').Pool(a.workers) as P:
            hp = pd.DataFrame([dict(step=s, alpha=r[0], l1=r[1], inner_mse=r[2]) for s, r in zip(range(1, 10), P.map(hx_hparams, range(1, 10)))])
        hp.to_csv(hpf, index=False)
    G['HP'] = {int(r.step): (r.alpha, r.l1, r.inner_mse) for r in hp.itertuples()}
    print(f'h={h}: слагаемые {G["terms"]}; HAR-X α/l1 по шагам {[(s, G["HP"][s][:2]) for s in steps]}; {time.time() - t0:.0f} c', flush=True)
    if a.task in ('pw', 'pwlin'):
        pwf = OUT / f'pw_h{h}.npz'
        if pwf.exists():
            z = np.load(pwf); G['PW'] = {int(k[1:]): z[k] for k in z.files}
        else:
            with mp.get_context('fork').Pool(a.workers) as P:
                R = P.map(pw_step, range(1, 10))
            G['PW'] = {s: o for s, o, _ in R}
            np.savez(pwf, **{f's{s}': o for s, o, _ in R})
            pd.DataFrame([l for _, _, l in R]).to_csv(OUT / f'pw_h{h}_trees.csv', index=False)
        print(f'PW-HARX готов, {time.time() - t0:.0f} c', flush=True)
        if a.task == 'pwlin':  # Д16.9 без бустинга: val/test шагов 1-9, rv_pred без поправки + Б0.1 (как harx.add_bias)
            GA = G['GA']; tc = f'rv_target_h{h}'
            vals, tests = [], []
            for s in range(1, 10):
                vy, ty = 2015 + s, 2016 + s
                for yr, lst in ((vy, vals), (ty, tests)):
                    m = (GA.year == yr).values & (G['ok'] if yr == vy else True)
                    lst.append(pd.DataFrame({'date': GA.date.values[m], 'ticker': GA.ticker.values[m], 'year': yr, 'step': s,
                                             'rv_actual': GA[tc].values[m], 'rv_pred': np.clip(np.exp(G['PW'][s][m]), 1e-10, None)}))
            v, t = HX.add_bias(pd.concat(vals, ignore_index=True), pd.concat(tests, ignore_index=True))
            v.to_parquet(L.REGEN / f'D16_9_h{h}_pwlin_val.parquet', index=False)
            lock = L.REGEN / 'test_locked' / f'D16_9_h{h}'; lock.mkdir(parents=True, exist_ok=True)
            t.to_parquet(lock / f'D16_9_h{h}_pwlin_test.parquet', index=False)
            print('PW-HARX без бустинга записан', flush=True); return
    G['REF'] = {}
    for m in a.models:  # как L.make_ref в каждой задаче, но один раз
        r = L.make_ref(m, G['A'], G['cols'], h, G['cfg'], 'Bv2b')
        if m == 'lgbm':
            r.construct()
        G['REF'][m] = r
    tasks = [(a.task, m, s, sd) for s in sorted(steps, reverse=True) for sd in a.seeds for m in a.models]
    with mp.get_context('fork').Pool(a.workers) as P:
        R = P.map(task, tasks, chunksize=1)
    V = pd.concat(R, ignore_index=True)
    fn = OUT / f'{a.task}{SFX}_h{h}_s{lo}-{hi}.parquet'
    V.to_parquet(fn, index=False)
    print(f'записано {fn}, {time.time() - t0:.0f} c', flush=True)


if __name__ == '__main__':
    main()
