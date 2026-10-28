#!/usr/bin/env python3
"""Д25.4 — объединитель без параметров на h=22: медиана и усечённое среднее (без минимума и максимума) OOS-прогнозов допущенных
кандидатов пула h=22 (раздел 4 черновика FINAL_DESIGN) + HAR-J с поправкой (sm); рядом — ridge и QLIKE-вариант объединителя Д22
(d22_full.fit_comb) на том же наборе. Также общая библиотека Д25.3–Д25.5 (OOS-прогнозы дизайна кандидата, метрики, c1–c4, детектор).

OOS-прогноз дизайна кандидата: gibrid_compute.compute -> gibrid_designs.select/v0 с ev='oos' (параметры, схема и s² шага t−1 -> val-год шага t);
это ровно то, что final3 оценивает в designs_val (q_hyb). Шаги 7–9 (подтверждающий) в метриках не участвуют: все таблицы фильтруются
step <= 6 до счёта; файлы val шагов 1–9 пишутся только для yaml.
python3 scripts/analysis/d25_4.py run | leak"""
import os, sys, json, hashlib, argparse
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gibrid_har as GH
import gibrid_harcj as HC
import gibrid_schemes as S
import gibrid_designs as D
from gibrid_compute import compute
from gibrid_io import load_boost

ROOT = Path('/Users/nilysenok/Desktop/Старое/PhD_Vol_trading')
R = ROOT / 'data' / 'regen'
FEAT = R / 'features_pit_v3.parquet'
OUT = R / 'D25b'
LOCK = R / 'test_locked'
KAND = R / 'final3' / 'kandidaty'
RANK, HOLD, CRIT = (2, 3, 4, 5), 6, (2, 3, 4, 5, 6)
DEVMAX = 6  # шаги 7–9 — подтверждающий набор: метрики не считаются
KEY = ['date', 'ticker', 'step']
SFX = {'single': '', 'top5': '_top5', 'seeds3': '_seeds4344', 'wrap2': '_wrap2'}

# допущенные кандидаты пула h=22 (раздел 4 черновика, рейтинг h=22, «не допущен» = —), дизайн-представитель по правилу; + HAR-J с поправкой.
# Зафиксировано до счёта (журнал dev_log_d25b.md, 04:40).
SET22 = [('D23_1_h22e', 'single|g3|B-sm|L0'), ('D19_4y_h22', 'single|g3|B-sm|L0'), ('D19_1x_h22', 'single|g3|B-sm|L0'),
         ('D18x_h22', 'single|g3|B-sm|L0'), ('D19_4_h22', 'single|g3|B-sm|L0'), ('D24_6_d18r', 'single|g3|B-sm|L0.5'),
         ('HARJ_corr', 'single|V0_HAR|B-sm|L0')]
# финалисты (CONFIRM_OPENED.json) + финал h=22 (HAR-J с поправкой)
FINALISTS = {1: [('D16_5_h1', 'single|step|B-sm|L0|log'), ('D22_h1', 'top5|V0_XGB|B-sm|L0'), ('D16_h1_d812x', 'top5|V0_XGB|B-sm|L0')],
             5: [('D17_1_h5_G5Xopt', 'top5|V0_LGB|bez|L0'), ('D12_h5', 'wrap2|V0_XGB|B-sm|L0'), ('D24_6_rgarch', 'single|g3|B-sm|L0')],
             22: [('D23_1_h22e', 'single|g3|B-sm|L0'), ('D19_4y_h22', 'single|g3|B-sm|L0'), ('D19_1x_h22', 'single|g3|B-sm|L0'),
                  ('HARJ_corr', 'single|V0_HAR|B-sm|L0')]}
FINAL = {1: 'D22_h1', 5: 'D12_h5', 22: 'HARJ_corr'}
KRUG2 = {'D12_h1_A_d812', 'D12_h1_B_b0.25', 'D12_h1_V0', 'F2m_h5', 'D12_h5', 'L2v3_h5', 'D12_h22'}
HARJ_HOST = {1: 'D12_h1_A_d812', 5: 'D12_h5', 22: 'D12_h22'}
BEST2021 = {1: 0.2382, 5: 0.1553, 22: 0.1404}  # «текущий лучший» кругов 1–2 (final3, раздел 4: D12_h1_V0 / F2m_h5 / D12_h22), val 2021


def qlike(y, p):
    p = np.clip(p, 1e-10, None)
    return float(np.mean(y / p - np.log(y / p) - 1))


def rp(x, default):
    if not x:
        return Path(default)
    x = Path(x)
    return x if x.is_absolute() else ROOT / x


def cand_cfg(boost, h):
    """-> dict(gboost, regen, features, test_dir) кандидата (yaml третьего круга или кандидат второго круга / HAR-J)."""
    if boost == 'HARJ_corr':
        boost = HARJ_HOST[h]
    if boost in KRUG2:
        return dict(gboost=boost, regen=R, features=FEAT, test_dir=LOCK / boost, h=h)
    if boost.startswith('D25_') and not (KAND / f'{boost}.yaml').exists():  # варианты Д25 до подачи yaml
        return dict(gboost=boost, regen=OUT, features=FEAT, test_dir=LOCK / boost, h=h)
    import yaml
    y = yaml.safe_load((KAND / f'{boost}.yaml').read_text())
    return dict(gboost=boost, regen=rp(y.get('regen'), R), features=rp(y.get('features_har'), FEAT),
                test_dir=rp(y.get('test_dir'), LOCK / boost), h=h, yaml=y)


_A = {}


def matrix(features=FEAT):
    k = str(features)
    if k not in _A:
        A, harj = GH.load_pit(features)
        _A[k] = (HC.add_cj(A), harj)
    return _A[k]


_HV = {}


def har_inputs(h, features=FEAT, test=False):
    k = (h, str(features), test)
    if k not in _HV:
        A, harj = matrix(features)
        hv, ht, _ = GH.har_forecasts(h, A, harj)
        _HV[k] = (hv, ht if test else ht.iloc[:0], dict(hcj=HC.forecasts(h, A), state=HC.state(A)))
    return _HV[k]


def design_oos(boost, h, design, test=False, bx_override=None):
    """OOS-прогноз дизайна на val (шаги 1–9; шаг 1 у схем — NaN) и, при test=True, тест шагов 1–9 (параметры шага, без метрик).
    -> (val DataFrame[date,ticker,year,step,rv_actual,rv_pred,har], test DataFrame | None, ch-таблица шагов)."""
    c = cand_cfg(boost, h)
    var, sel, lab, lam, *bl = design.split('|')
    S.MODE = bl[0] if bl else 'rv'
    code = 'bez' if lab == 'bez' else lab[2:]
    hv, ht, ext = har_inputs(h, c['features'], test)
    if test:
        os.environ['GIBRID_TEST_DIR'] = str(c['test_dir'])
    if bx_override is not None:
        bx, bl_ = bx_override
    else:
        bx, bl_ = load_boost(c['gboost'], h, c['regen'], var, test=test)
    if test:
        os.environ.pop('GIBRID_TEST_DIR', None)
    res = compute(hv, ht, h, bx, bl_, code, h, [], ext)
    rs = D.shrink(res, float(lam[1:]))
    if sel.startswith('V0_'):
        ch, ov, ot = D.v0(rs, sel[3:].lower(), h)
        ov = ov.assign(rv_pred_oos=ov.rv_pred.values)
    else:
        ch, ov, ot = D.select(rs, sel, h, 'oos')
    har = pd.concat([r['vs'][KEY + ['har']] for r in rs])
    v = ov.drop(columns=['rv_pred']).rename(columns={'rv_pred_oos': 'rv_pred'}).merge(har, on=KEY, how='left')
    S.MODE = 'rv'
    return v.reset_index(drop=True), (ot.reset_index(drop=True) if test else None), ch


def dev(df):
    return df[(df.step >= 2) & (df.step <= DEVMAX)]


def year_table(v, col='rv_pred'):
    """QLIKE по шагам 2–6 (только разработочный набор; шаги 7–9 отбрасываются до счёта)."""
    d = dev(v)
    d = d[d.rv_actual > 1e-12]
    return {int(s): qlike(g.rv_actual.values, g[col].values) for s, g in d.groupby('step')}


def summ(q):
    return dict(m1720=float(np.mean([q[s] for s in RANK])), q2021=q[HOLD], m1721=float(np.mean([q[s] for s in CRIT])))


_ET = {}


def etalon_steps(h):
    """HAR-J с поправкой sm (единый эталон), QLIKE шагов 2–6."""
    if h not in _ET:
        v, _, _ = design_oos('HARJ_corr', h, 'single|V0_HAR|B-sm|L0')
        _ET[h] = year_table(v)
    return _ET[h]


def crit(q, h, singles=None, w=None):
    """c1–c4 по правилу gibrid_designs.flags на 2017–2021 для самостоятельной модели: эталон — HAR-J с поправкой sm; singles — {имя: {шаг: q}}
    одиночные модели той же s² (c2/c3); веса w (шаг -> массив) для c4 (у самостоятельной модели весов нет -> c4 выполнен)."""
    et = etalon_steps(h)
    rows = []
    for s in CRIT:
        r = dict(step=s, q_hyb=q[s], q_har=et[s], scheme='own', v1_ml='x', w_har=0.0, w_xgb=0.0, w_lgb=0.0)
        sg = singles or {}
        r['q_xgb_single'] = sg.get('xgb', {}).get(s, np.inf)
        r['q_lgb_single'] = sg.get('lgb', {}).get(s, np.inf)
        if w is not None:
            r.update(w_har=w[s][0], w_xgb=w[s][1], w_lgb=w[s][2])
        rows.append(r)
    fl = D.flags(pd.DataFrame(rows))
    return {k: fl[k] for k in ('c1', 'c2', 'c3', 'c4')}


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()[:12]


# ---------------------------------------------------------------- Д25.4
def components22(test=False):
    """OOS-прогнозы набора SET22 (val шагов 1–9) в широком виде: KEY + rv_actual + столбец на кандидата."""
    W, T = None, None
    for b, d in SET22:
        v, t, _ = design_oos(b, 22, d, test=test)
        v = v[KEY + ['year', 'rv_actual', 'rv_pred']].rename(columns={'rv_pred': b})
        W = v if W is None else W.merge(v[KEY + [b]], on=KEY, how='inner')
        if test:
            t = t[KEY + ['year', 'rv_actual', 'rv_pred']].rename(columns={'rv_pred': b})
            T = t if T is None else T.merge(t[KEY + [b]], on=KEY, how='inner')
    return W, T


def comb_free(M, kind):
    P = np.log(np.clip(M, 1e-10, None))
    if kind == 'median':
        return np.exp(np.median(P, axis=1))
    if kind == 'trim':
        s = np.sort(P, axis=1)
        return np.exp(s[:, 1:-1].mean(axis=1))
    if kind == 'mean':
        return np.exp(P.mean(axis=1))
    raise ValueError(kind)


def s2_sm_factor(v, col, st):
    """Поправка Б0.1 sm на объединённый прогноз: s² по OOS-остаткам трёх последних val-лет до шага st (шаги st−3..st−1); шаг 2 — по шагу 1."""
    e = []
    for j in range(max(1, st - 3), st):
        g = v[(v.step == j) & (v.rv_actual > 1e-12)]
        g = g[np.isfinite(g[col])]
        if len(g):
            e.append(np.log(g.rv_actual.values + 1e-10) - np.log(g[col].values))
    return float(np.exp(np.var(np.concatenate(e)) / 2)) if e else 1.0


def comb_weights(W, names, obj, years_lt):
    """Объединитель Д22 (d22_full.fit_comb: ridge / qlike, β в симплексе, сдвиг к x1 = HAR-J с поправкой) на строках лет < years_lt."""
    import d22_full as DF
    base = 'HARJ_corr'
    oth = [n for n in names if n != base]
    T = W[(W.year < years_lt) & (W.rv_actual > 1e-12)].dropna(subset=names)
    if len(T) == 0:
        return {base: 1.0}
    X = pd.DataFrame({'y': np.log(T.rv_actual.values + 1e-10), 'x1': np.log(T[base].values)})
    xs = ['x1']
    for i, n in enumerate(oth):
        X[f'g{i}'] = np.log(T[n].values)
        xs.append(f'g{i}')
    saveG = DF.GROUP
    DF.GROUP = tuple(['x1'] + xs[1:])
    try:
        best = None
        In = X[T.year.values < years_lt - 1]
        Ev = X[T.year.values == years_lt - 1]
        lam = 1.0
        if len(In) and len(Ev):
            for l_ in DF.LAMS:
                b = DF.fit_comb(In, xs, l_, obj)
                res = In.y.values - DF.predict(In, b)
                q = DF.qlike_ln(np.exp(Ev.y.values), DF.predict(Ev, b), np.exp(np.var(res) / 2) if obj == 'ridge' else 1.0)
                if best is None or q < best[0]:
                    best = (q, l_)
            lam = best[1]
        b = DF.fit_comb(X, xs, lam, obj)
    finally:
        DF.GROUP = saveG
    out = {base: b['x1']}
    out.update({n: b.get(f'g{i}', 0.0) for i, n in enumerate(oth)})
    return out


WTS = []
LEAK_PLANT = False


def combine(W, kind, names):
    """Прогноз объединителя по шагам (val-год шага; для ridge/QLIKE — веса по годам < val-года с эмбарго 22 дня, затем поправка sm)."""
    out = W[KEY + ['year', 'rv_actual']].copy()
    out['rv_pred'] = np.nan
    M = W[names].values
    if kind in ('median', 'trim', 'mean'):
        out['raw'] = comb_free(M, kind)
    else:
        out['raw'] = np.nan
        for st in sorted(W.step.unique()):
            m = (W.step == st).values
            Y = int(W.loc[m, 'year'].iloc[0])
            # эмбарго: строки года Y−1, чей горизонт заходит в год Y, исключаются (последние 22 торговых дня года Y−1)
            Wt = W[(W.year < Y)].copy()
            last = Wt[Wt.year == Y - 1].groupby('ticker').date.transform(lambda s: s.rank(ascending=False))
            Wt = Wt.drop(last[last <= 22].index)
            if LEAK_PLANT:  # самопроверка детектора: обучение и на val-годе Y (будущее)
                Wt, Yfit = W[W.year <= Y], Y + 1
            else:
                Yfit = Y
            b = comb_weights(Wt, names, 'ridge' if kind == 'ridge' else 'qlike', Yfit)
            out.loc[m, 'raw'] = np.exp(sum(b.get(n, 0.0) * np.log(W.loc[m, n].values) for n in names if b.get(n, 0.0) != 0.0))
            if st <= DEVMAX:  # веса шагов 7–9 обучены на годах подтверждающего — не сохраняются
                WTS.append(dict(kind=kind, step=st, year=Y, **{n: b.get(n, 0.0) for n in names}))
    # компоненты уже с поправкой своих дизайнов; объединённый прогноз — без дополнительной поправки (столбец rv_pred)
    # и с поправкой sm по собственным OOS-остаткам шагов < t (столбец rv_pred_sm) — справка
    out['rv_pred'] = out['raw']
    out['rv_pred_sm'] = np.nan
    for st in sorted(out.step.unique()):
        m = (out.step == st).values
        out.loc[m, 'rv_pred_sm'] = out.loc[m, 'raw'] * s2_sm_factor(out.assign(_r=out.raw / 1.0), '_r', st) if st > 1 else out.loc[m, 'raw']
    return out


def run(args):
    OUT.mkdir(parents=True, exist_ok=True)
    W, _ = components22()
    W.to_parquet(OUT / 'd25_4_components_h22_val.parquet', index=False)
    names = [b for b, _ in SET22]
    rows = []
    for b in names:
        q = year_table(W.rename(columns={b: 'rv_pred'}))
        rows.append(dict(variant=b, **summ(q), **{f'y{2015 + s}': q[s] for s in CRIT}))
    et = summ(etalon_steps(22))
    # одиночные модели для c2/c3 объединителя — компоненты набора: по годам лучшая из компонент без HAR-J (ночное умолчание, см. отчёт)
    qc = {b: year_table(W.rename(columns={b: 'rv_pred'})) for b in names if b != 'HARJ_corr'}
    best_comp = {s: min(qc[b][s] for b in qc) for s in CRIT}
    sg = {'xgb': best_comp, 'lgb': best_comp}
    res = {}
    for kind in ('median', 'trim', 'mean', 'ridge', 'qlike'):
        o = combine(W, kind, names)
        res[kind] = o
        o.to_parquet(OUT / f'd25_4_{kind}_h22_val.parquet', index=False)
        for col in ('rv_pred', 'rv_pred_sm'):
            q = year_table(o, col)
            rows.append(dict(variant=f'{kind}{"" if col == "rv_pred" else "+sm"}', **summ(q), **{f'y{2015 + s}': q[s] for s in CRIT},
                             **crit(q, 22, sg)))
    T = pd.DataFrame(rows)
    T['d_etalon_1720'] = T.m1720 / et['m1720'] - 1
    pd.DataFrame(WTS).to_csv(OUT / 'd25_4_weights.csv', index=False)
    T.to_csv(OUT / 'd25_4_summary.csv', index=False)
    print(T.to_string())


def leak(args):
    """Детектор «будущее не влияет»: портим rv_actual и прогнозы компонент с даты среза; прогнозы объединителей до среза не должны меняться.
    Самопроверка: подложенная утечка (вес по году Y, а не < Y) обязана сработать."""
    W = pd.read_parquet(OUT / 'd25_4_components_h22_val.parquet')
    names = [b for b, _ in SET22]
    cut = pd.Timestamp('2019-07-01')
    Wb = W.copy()
    fut = Wb.date >= cut
    rng = np.random.default_rng(0)
    Wb.loc[fut, 'rv_actual'] *= rng.uniform(0.2, 5, fut.sum())
    for n in names:
        Wb.loc[fut, n] *= rng.uniform(0.5, 2, fut.sum())
    ok = True
    for kind in ('median', 'trim', 'ridge', 'qlike'):
        a, b = combine(W, kind, names), combine(Wb, kind, names)
        pre = (a.date < cut).values
        d = float(np.nanmax(np.abs(np.log(a.rv_pred_sm.values[pre]) - np.log(b.rv_pred_sm.values[pre]))))
        print(f'{kind}: max |Δ ln ŷ| до среза = {d:.2e}', 'OK' if d < 1e-9 else 'УТЕЧКА')
        ok &= d < 1e-9
    # подложенная утечка: веса ridge по строкам года Y включительно
    global LEAK_PLANT
    LEAK_PLANT = True
    a, b = combine(W, 'ridge', names), combine(Wb, 'ridge', names)
    LEAK_PLANT = False
    pre = (a.date < cut).values
    d = float(np.nanmax(np.abs(np.log(a.rv_pred.values[pre]) - np.log(b.rv_pred.values[pre]))))
    print(f'подложенная утечка: max |Δ| = {d:.2e}', 'сработал (OK)' if d > 1e-9 else 'НЕ СРАБОТАЛ')
    ok &= d > 1e-9
    print('ДЕТЕКТОР Д25.4:', 'OK' if ok else 'ОШИБКА')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['run', 'leak'])
    a = ap.parse_args()
    {'run': run, 'leak': leak}[a.cmd](a)
