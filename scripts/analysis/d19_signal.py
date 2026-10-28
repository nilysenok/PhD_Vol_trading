"""Д19: гибрид «сигнал + динамика» для h=5/22.
r1 = ln(hyb1) - ln(HAR1) — только OOS-прогнозы (val шага s = модель шага s; тест — test_locked/final2/final_h1_test, только сборка).
Д19.1: ln y_h = ln HAR_h + rho_h * r1, rho_h — OLS без константы по OOS-истории (val шагов < s; для шага 1 — его val-год).
Д19.2: rho_h — импульсный отклик дневной HAR-J шага (рекурсия), без подгонки.
Д19.3: LGBM-классификатор P(ln RV_h > ln HAR_h), ln y = ln HAR_h + delta*(2p-1), delta по val шага t-1.
Выход: data/regen/D19/{boost}_{xgb|lgbm}_h{h}[_seeds4344]_val.parquet (один прогноз в обоих слотах, до поправки s² — её делает gibrid_v17),
       data/regen/test_locked/D19_h{h}/{boost}_{xgb|lgbm}_h{h}_test.parquet (шаги 1-9, без метрик), data/regen/D19/rho_*.csv.
python3 scripts/analysis/d19_signal.py [--only 1 2 3]"""
import argparse, sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import LinearRegression
sys.path.insert(0, str(Path(__file__).parent))
import gibrid_har as GH

R = Path('/Users/nilysenok/Desktop/Старое/PhD_Vol_trading/data/regen')
OUT, TL = R / 'D19', R / 'test_locked'
FEAT = R / 'features_pit_v3.parquet'  # как в финалах: har_raw final2 совпадает с HAR-J на v3 до 0 (на v2 - нет)
K = ['date', 'ticker', 'step']


def har_steps(h, a, harj):
    """HAR-J шага: val (OOS, год шага), тест, коэффициенты дневной модели (h=1) и in-sample прогноз на окне (для Д19.3)."""
    return GH.har_forecasts(h, a, harj)


def har1_coefs(a, harj):
    """Коэффициенты HAR-J h=1 по шагам (как gibrid_har.har_forecasts)."""
    co = {}
    for step, ty in enumerate([y for y in range(2017, 2026) if y in a.year.values], 1):
        tr = a[a.year < ty]; vy = sorted(tr.year.unique())[-1]
        tp = tr[tr.year != vy].dropna(subset=['rv_target_h1'])
        X = GH.clean(np.log(tp[harj].values.clip(1e-10)))
        m = LinearRegression().fit(X, np.log(tp['rv_target_h1'].values + 1e-10))
        co[step] = dict(zip(harj, m.coef_))
    return co


def irf_rho(c, h):
    """Шок +1 в ln RV дня 1 (то, что предсказывает r1); дни 2..h — рекурсия дневной HAR (ln rv_w ~ среднее логов за 5 дней, rv_m — за 22).
    rho_h = среднее откликов за дни 1..h. Скачки не шокируются."""
    e = [1.0]
    for k in range(1, h):
        past = e[::-1]
        e.append(c['rv_d'] * past[0] + c['rv_w'] * sum(past[:5]) / 5 + c['rv_m'] * sum(past[:22]) / 22)
    return float(np.mean(e))


def r1_val(seed='top5'):
    """r1 на val (OOS): hyb1 = нога финалиста h=1 (D12_h1_A_d812 XGB top5; шаги без топ-5 — single) или seed-вариант; HAR1 — HAR-J h=1 шага."""
    b = 'D12_h1_A_d812_xgb_h1'
    s = pd.read_parquet(R / f'{b}_val.parquet')
    if seed == 'top5':
        t = pd.read_parquet(R / f'{b}_top5_val.parquet').groupby(K, as_index=False).rv_pred.mean()
        p = pd.concat([t, s[~s.step.isin(t.step.unique())][K + ['rv_pred']]])
    elif seed == 's42':
        p = s[K + ['rv_pred']]
    else:
        ss = pd.read_parquet(R / f'{b}_seeds4344_val.parquet'); p = ss[ss.seed == int(seed[1:])][K + ['rv_pred']]
    return p.rename(columns={'rv_pred': 'hyb1'})


def build(h, A, harj, hv1, ht1, co, seeds, only):
    hv, ht, s2 = har_steps(h, A, harj)
    hv = hv.rename(columns={'rv_pred': 'har'}); ht = ht.rename(columns={'rv_pred': 'har'})
    t1 = pd.read_parquet(TL / 'final2/final_h1_test.parquet')  # тест: только сборка, метрик нет
    t1 = t1[K + ['xgb_raw', 'har_raw']].rename(columns={'xgb_raw': 'hyb1', 'har_raw': 'har1'})
    rows, out = [], {}
    for sd in seeds:
        V = hv.merge(r1_val(sd), on=K).merge(hv1[K + ['har1']], on=K)
        V['r1'] = np.log(V.hyb1) - np.log(V.har1)
        V['res'] = np.log(V.rv_actual.clip(1e-10)) - np.log(V.har)
        T = ht.merge(t1, on=K)
        T['r1'] = np.log(T.hyb1) - np.log(T.har1)
        pv, pt = {1: [], 2: []}, {1: [], 2: []}
        for s in range(1, 10):
            ok = V.rv_actual > 1e-12
            # Д19.1: val шага s — rho по OOS-истории шагов < s (шаг 1 — свой val-год); тест шага s — по шагам <= s
            hist_v = V[ok & ((V.step < s) | ((s == 1) & (V.step == 1)))]
            hist_t = V[ok & (V.step <= s)]
            ols = lambda d: float((d.r1 * d.res).sum() / (d.r1 ** 2).sum())
            rv_, rt_ = ols(hist_v), ols(hist_t)
            rho2 = irf_rho(co[s], h)
            for m, (rvv, rtt) in ((1, (rv_, rt_)), (2, (rho2, rho2))):
                v = V[V.step == s]; t = T[T.step == s]
                pv[m].append(v[K + ['year', 'rv_actual']].assign(rv_pred=np.exp(np.log(v.har) + rvv * v.r1)))
                pt[m].append(t[K + ['year', 'rv_actual']].assign(rv_pred=np.exp(np.log(t.har) + rtt * t.r1)))
            rows.append(dict(h=h, seed=sd, step=s, rho1_val=rv_, rho1_test=rt_, rho2=rho2,
                             b_d=co[s]['rv_d'], b_w=co[s]['rv_w'], b_m=co[s]['rv_m'], n_hist=len(hist_v)))
        for m in (1, 2):
            if m in only:
                out[(m, sd)] = (pd.concat(pv[m]), pd.concat(pt[m]))
    if 3 in only:
        out[(3, 'top5')] = d193(h, A, harj, hv, ht)
    return pd.DataFrame(rows), out, s2


def d193(h, A, harj, hv, ht):
    """Д19.3: LGBM binary на P(ln RV_h > ln HAR_h(in-sample шага)); гиперпараметры — D12_h{h} lgbm по шагам; признаки — features_pit_v3 (числовые, без целей)."""
    import lightgbm as lgb
    sel = pd.read_csv(R / f'D12_h{h}_selected_h{h}.csv'); sel = sel[sel.model == 'lgbm'].set_index('step')
    tgt = f'rv_target_h{h}'
    drop = {'date', 'ticker', 'year', 'source'} | {c for c in A.columns if 'target' in c or c.startswith('source')}
    F = [c for c in A.columns if c not in drop and pd.api.types.is_numeric_dtype(A[c])]
    years = [y for y in range(2017, 2026) if y in A.year.values]
    P = {}
    for step, ty in enumerate(years, 1):
        tr = A[A.year < ty]; vy = sorted(tr.year.unique())[-1]
        tp = tr[tr.year != vy].dropna(subset=[tgt]); vi = tr[tr.year == vy]; td = A[A.year == ty]
        X = lambda d: GH.clean(np.log(d[harj].values.clip(1e-10)))
        m = LinearRegression().fit(X(tp), np.log(tp[tgt].values + 1e-10))
        lab = (np.log(tp[tgt].values + 1e-10) > m.predict(X(tp))).astype(int)
        p = sel.loc[step]
        prm = dict(objective='binary', num_leaves=int(p.num_leaves), learning_rate=float(p.learning_rate), bagging_fraction=float(p.bagging_fraction),
                   bagging_freq=1, feature_fraction=float(p.feature_fraction), min_data_in_leaf=int(p.min_data_in_leaf), reg_alpha=float(p.reg_alpha),
                   reg_lambda=float(p.reg_lambda), seed=42, num_threads=3, verbose=-1)
        bst = lgb.train(prm, lgb.Dataset(tp[F].values.astype(float), lab), num_boost_round=max(int(p.best_iteration), 1))
        for nm, d in (('val', vi), ('test', td)):
            P[(nm, step)] = pd.DataFrame({'date': d.date.values, 'ticker': d.ticker.values, 'step': step, 'p': bst.predict(d[F].values.astype(float))})
        print(f'  Д19.3 h={h} шаг {step}: доля 1 = {lab.mean():.3f}', flush=True)
    return _delta_finish(h, hv, ht, P, 'D19_3')


def _delta_finish(h, hv, ht, P, tag):
    """ln y = ln HAR_h + delta*(2p-1); delta из {0,1; 0,2; 0,3}: val шага s — по val шага s-1 (шаг 1 — свой), тест шага s — по val шага s."""
    pv = pd.concat([P[('val', s)] for s in range(1, 10)]); pt = pd.concat([P[('test', s)] for s in range(1, 10)])
    V = hv.merge(pv, on=K); T = ht.merge(pt, on=K)
    V = V[V.rv_actual > 1e-12]

    def ql(d, dl):  # QLIKE с оптимальным масштабом (выбор delta не зависит от поправки s²)
        f = np.exp(np.log(d.har) + dl * (2 * d.p - 1)); f = f * np.mean(d.rv_actual / f); x = d.rv_actual / f
        return float(np.mean(x - np.log(x) - 1))
    DL = (0.1, 0.2, 0.3)
    best = {s: min(DL, key=lambda dl: ql(V[V.step == s], dl)) for s in range(1, 10)}
    use = {s: best[max(s - 1, 1)] for s in range(1, 10)}  # val шага s — delta с val шага s-1 (шаг 1 — свой)
    use_t = best  # тест шага s — delta с val шага s
    ov = pd.concat([V[V.step == s].assign(rv_pred=lambda d, s=s: np.exp(np.log(d.har) + use[s] * (2 * d.p - 1))) for s in range(1, 10)])
    ot = pd.concat([T[T.step == s].assign(rv_pred=lambda d, s=s: np.exp(np.log(d.har) + use_t[s] * (2 * d.p - 1))) for s in range(1, 10)])
    pd.DataFrame([dict(h=h, step=s, delta_val=use[s], delta_test=use_t[s]) for s in range(1, 10)]).to_csv(OUT / f'delta_{tag}_h{h}.csv', index=False)
    c = K + ['year', 'rv_actual', 'rv_pred']
    return ov[c], ot[c]


def d194(A, harj, hv1, seed='top5'):
    """Д19.4 = Д18.1 x Д19.1 (h=22, уровень RV): y = (5*hyb5 + 17*exp(ln HAR_6_22 + rho*r1))/22.
    hyb5 — нога финалиста h=5 D12_h5 (wrap2, XGB-слот, до поправки; тест — xgb_raw final_h5_test); HAR_6_22 — HAR-J шага на цель
    «среднее RV за t+6..t+22» = (22*mean_h22 - 5*mean_h5)/17; rho — OLS без константы как Д19.1, на дальнюю цель, OOS-история шагов < s."""
    A = A.copy()
    A['rv_target_h622'] = (22 * A['rv_target_mean_h22'] - 5 * A['rv_target_mean_h5']) / 17
    A.loc[A['rv_target_h622'] <= 0, 'rv_target_h622'] = np.nan
    hv, ht, _ = GH.har_forecasts(622, A, harj)
    hv = hv.rename(columns={'rv_pred': 'har622', 'rv_actual': 'rv622'}); ht = ht.rename(columns={'rv_pred': 'har622'}).drop(columns=['rv_actual'])
    h22v, h22t, _ = GH.har_forecasts(22, A, harj)  # только для rv_actual (цель h=22) и состава строк
    h5v = pd.read_parquet(R / 'D12_h5_xgb_h5_wrap2_val.parquet')[K + ['rv_pred']].rename(columns={'rv_pred': 'hyb5'})
    t5 = pd.read_parquet(TL / 'final2/final_h5_test.parquet')[K + ['xgb_raw']].rename(columns={'xgb_raw': 'hyb5'})
    t1 = pd.read_parquet(TL / 'final2/final_h1_test.parquet')[K + ['xgb_raw', 'har_raw']].rename(columns={'xgb_raw': 'hyb1', 'har_raw': 'har1'})
    V = h22v[K + ['year', 'rv_actual']].merge(hv[K + ['har622', 'rv622']], on=K, how='left').merge(h5v, on=K).merge(r1_val(seed), on=K).merge(hv1[K + ['har1']], on=K)
    V['r1'] = np.log(V.hyb1) - np.log(V.har1)
    T = h22t[K + ['year', 'rv_actual']].merge(ht[K + ['har622']], on=K).merge(t5, on=K).merge(t1, on=K)
    T['r1'] = np.log(T.hyb1) - np.log(T.har1)
    V['res'] = np.log(V.rv622) - np.log(V.har622)
    ols = lambda d: float((d.r1 * d.res).sum() / (d.r1 ** 2).sum())
    ok = V.res.notna() & np.isfinite(V.res)
    pv, pt, rows = [], [], []
    for s in range(1, 10):
        rv_ = ols(V[ok & ((V.step < s) | ((s == 1) & (V.step == 1)))]); rt_ = ols(V[ok & (V.step <= s)])
        for d, r, lst in ((V[V.step == s], rv_, pv), (T[T.step == s], rt_, pt)):
            far = np.exp(np.log(d.har622) + r * d.r1)
            lst.append(d[K + ['year', 'rv_actual']].assign(rv_pred=(5 * d.hyb5 + 17 * far) / 22))
        rows.append(dict(h=22, step=s, rho4_val=rv_, rho4_test=rt_))
    ov, ot = pd.concat(pv), pd.concat(pt)
    assert ov.rv_pred.notna().all() and ot.rv_pred.notna().all()
    return pd.DataFrame(rows), ov, ot


def d193x(A, harj, hv, ht):
    """Д19.3 с признаками Д16.5 (h=22): матрица features_pit_v3_d165 + build_features(cfg={}) + step_data('Bv2b') как d16_510.py;
    гиперпараметры LGBM — D16_5_h22_selected_h22.csv по шагам; метка — ln RV_22 > in-sample HAR-J шага."""
    import os
    os.environ['L2_FEAT'] = str(R / 'features_pit_v3_d165.parquet')
    import l2_lib as L
    import lightgbm as lgb
    h = 22
    B, FC = L.load_base(None, (), h)
    B2, cols, _, _ = L.build_features(B, FC, h, {})
    sel = pd.read_csv(R / 'D16_5_h22_selected_h22.csv'); sel = sel[sel.model == 'lgbm'].set_index('step')
    P = {}
    for step in range(1, 10):
        D = L.step_data(B2, cols, h, step, 'lgbm', {}, 'Bv2b')
        tp = D['parts']['tr']
        X = lambda d: GH.clean(np.log(d[harj].values.clip(1e-10)))
        yl = np.log(tp[L.tcol(h)].values + 1e-10)
        lab = (yl > LinearRegression().fit(X(tp), yl).predict(X(tp))).astype(int)
        p = sel.loc[step]
        prm = dict(objective='binary', num_leaves=int(p.num_leaves), learning_rate=float(p.learning_rate), bagging_fraction=float(p.bagging_fraction),
                   bagging_freq=1, feature_fraction=float(p.feature_fraction), min_data_in_leaf=int(p.min_data_in_leaf), reg_alpha=float(p.reg_alpha),
                   reg_lambda=float(p.reg_lambda), seed=42, num_threads=3, verbose=-1)
        bst = lgb.train(prm, lgb.Dataset(D['X']['tr'], lab), num_boost_round=max(int(p.best_iteration), 1))
        for nm, k in (('val', 'va'), ('test', 'te')):
            d = D['parts'][k]
            P[(nm, step)] = pd.DataFrame({'date': d.date.values, 'ticker': d.ticker.values, 'step': step, 'p': bst.predict(D['X'][k])})
        print(f'  Д19.3x h=22 шаг {step}: признаков {len(D["names"])}, доля 1 = {lab.mean():.3f}', flush=True)
    return _delta_finish(h, hv, ht, P, 'D19_3x')


def d165_leg(k):
    """Нога Д16.5 h=22 (решение владельца 01:40): XGB single (лучшая V0-нога Д16.5 в гибриде, gibrid_ref/D16_5_h22); val — OOS шагов, тест — только сборка."""
    f = R / 'D16_5_h22_xgb_h22_val.parquet' if k == 'val' else TL / 'D16_5_h22/D16_5_h22_xgb_h22_test.parquet'
    return pd.read_parquet(f)[K + ['rv_pred']].rename(columns={'rv_pred': 'x165'})


def d19x(A, harj, hv1, seed='top5'):
    """Варианты h=22 с ногой Д16.5 вместо V9:
    D19_1x — XGB-слот = Д19.1, LGB-слот = Д16.5 (гибрид смешивает HAR-J, Д19.1 и Д16.5);
    D19_4y — XGB-слот = Д19.4, LGB-слот = Д16.5;
    D19_4x — дальняя часть с Д16.5: far = exp(ln HAR_6_22 + rho*r1 + kappa*rx), rx = ln Д16.5 - ln HAR-J_22 (OOS),
             rho, kappa — OLS без константы по OOS-истории шагов < t; y = (5*hyb5 + 17*far)/22; XGB-слот = y, LGB-слот = Д16.5."""
    sfx = '' if seed == 'top5' else f'_{seed}'
    _, ov4, ot4 = d194(A, harj, hv1, seed)
    A2 = A.copy()
    A2['rv_target_h622'] = (22 * A2['rv_target_mean_h22'] - 5 * A2['rv_target_mean_h5']) / 17
    A2.loc[A2['rv_target_h622'] <= 0, 'rv_target_h622'] = np.nan
    hv, ht, _ = GH.har_forecasts(622, A2, harj)
    hv = hv.rename(columns={'rv_pred': 'har622', 'rv_actual': 'rv622'}); ht = ht.rename(columns={'rv_pred': 'har622'}).drop(columns=['rv_actual'])
    h22v, h22t, _ = GH.har_forecasts(22, A2, harj)
    h22v = h22v.rename(columns={'rv_pred': 'har22'}); h22t = h22t.rename(columns={'rv_pred': 'har22'})
    h5v = pd.read_parquet(R / 'D12_h5_xgb_h5_wrap2_val.parquet')[K + ['rv_pred']].rename(columns={'rv_pred': 'hyb5'})
    t5 = pd.read_parquet(TL / 'final2/final_h5_test.parquet')[K + ['xgb_raw']].rename(columns={'xgb_raw': 'hyb5'})
    t1 = pd.read_parquet(TL / 'final2/final_h1_test.parquet')[K + ['xgb_raw', 'har_raw']].rename(columns={'xgb_raw': 'hyb1', 'har_raw': 'har1'})
    V = (h22v[K + ['year', 'rv_actual', 'har22']].merge(hv[K + ['har622', 'rv622']], on=K, how='left').merge(h5v, on=K)
         .merge(r1_val(seed), on=K).merge(hv1[K + ['har1']], on=K).merge(d165_leg('val'), on=K))
    T = h22t[K + ['year', 'rv_actual', 'har22']].merge(ht[K + ['har622']], on=K).merge(t5, on=K).merge(t1, on=K).merge(d165_leg('test'), on=K)
    for d in (V, T):
        d['r1'] = np.log(d.hyb1) - np.log(d.har1); d['rx'] = np.log(d.x165) - np.log(d.har22)
    V['res'] = np.log(V.rv622) - np.log(V.har622)
    ok = V.res.notna() & np.isfinite(V.res)

    def ols2(d):
        X = d[['r1', 'rx']].values
        return np.linalg.lstsq(X, d.res.values, rcond=None)[0]
    pv, pt, rows = [], [], []
    for s_ in range(1, 10):
        bv = ols2(V[ok & ((V.step < s_) | ((s_ == 1) & (V.step == 1)))]); bt = ols2(V[ok & (V.step <= s_)])
        for d, b, lst in ((V[V.step == s_], bv, pv), (T[T.step == s_], bt, pt)):
            far = np.exp(np.log(d.har622) + b[0] * d.r1 + b[1] * d.rx)
            lst.append(d[K + ['year', 'rv_actual', 'x165']].assign(rv_pred=(5 * d.hyb5 + 17 * far) / 22))
        rows.append(dict(h=22, seed=seed, step=s_, rho_val=bv[0], kappa_val=bv[1], rho_test=bt[0], kappa_test=bt[1]))
    pd.DataFrame(rows).to_csv(OUT / f'rho4x_h22{sfx}.csv', index=False)
    print(seed, pd.DataFrame(rows).round(3).to_string(), flush=True)
    c = K + ['year', 'rv_actual', 'rv_pred']
    x4v, x4t = pd.concat(pv), pd.concat(pt)
    legv = x4v[K + ['year', 'rv_actual']].merge(d165_leg('val'), on=K).rename(columns={'x165': 'rv_pred'})
    legt = x4t[K + ['year', 'rv_actual']].merge(d165_leg('test'), on=K).rename(columns={'x165': 'rv_pred'})
    v1 = pd.read_parquet(OUT / f'D19_1_h22{sfx}_xgb_h22_val.parquet')
    t1x = pd.read_parquet(TL / 'D19_h22/D19_1_h22_xgb_h22_test.parquet')
    td = TL / 'D19_h22'
    for b, (xv, xt) in (('D19_1x_h22', (v1, t1x)), ('D19_4y_h22', (ov4, ot4)), ('D19_4x_h22', (x4v[c], x4t[c]))):
        lv = xv[K].merge(d165_leg('val'), on=K).rename(columns={'x165': 'rv_pred'}).merge(xv[K + ['year', 'rv_actual']], on=K)[c]
        lt = xt[K].merge(d165_leg('test'), on=K).rename(columns={'x165': 'rv_pred'}).merge(xt[K + ['year', 'rv_actual']], on=K)[c]
        assert len(lv) == len(xv) and len(lt) == len(xt)
        xv[c].to_parquet(OUT / f'{b}{sfx}_xgb_h22_val.parquet', index=False); lv.to_parquet(OUT / f'{b}{sfx}_lgbm_h22_val.parquet', index=False)
        if seed == 'top5':
            xt[c].to_parquet(td / f'{b}_xgb_h22_test.parquet', index=False); lt.to_parquet(td / f'{b}_lgbm_h22_test.parquet', index=False)


INC = {5: ('F2m_h5', 'wrap2|V0_LGB|B-sm|L0', 0.2122287402429689, 0.155332001485612),  # текущий лучший кругов 1-2 (final3_dry/incumbent.csv)
       22: ('D12_h22', 'seeds3|V0_XGB|B-sm|L0', 0.2222331795994113, 0.140402478566046)}


def evaluate():
    """Сводка по правилу третьего круга (final3.designs/summarize): представитель = лучший дизайн (без V0_HAR); отдельно — лучший V0_XGB
    (одиночная модель-кандидат); проверка отложенного 2021 против текущего лучшего кругов 1-2."""
    import final3 as F3
    rows = []
    for d in sorted((OUT / 'gibrid').iterdir()):
        b = d.name; h = int(b.split('_h')[1].split('_')[0])
        D = F3.designs(dict(gib_dir=d, gboost=b, h=h))
        if D is None:
            continue
        S = F3.summarize(dict(), D)
        comp = {}  # компоненты с той же s² (цель владельца h=22: гибрид не хуже лучшей из своих компонент), по дизайну
        for nm, g in D.groupby('design', sort=False):
            g = g.set_index('step')
            comp[nm] = {f'{c}_{t}': (g.loc[list(F3.RANK_STEPS), c].mean() if t == '1720' else g.loc[F3.HOLD, c])
                        for c in ('q_har', 'q_xgb_single', 'q_lgb_single') for t in ('1720', '2021')}
        for tag, x in (('лучший', S.iloc[0]), ('V0_XGB', S[S.design.str.contains(r'\|V0_XGB\|')].iloc[0])):
            inc = INC[h]
            rows.append(dict(boost=b, h=h, kind=tag, design=x.design, n_failed=x.n_failed, failed=x.failed, mean_1720=x.mean_1720, q2021=x.q2021,
                             mean_1721=x.mean_1721, har_1720=x.har_1720, **{f'q_{y}': x[f'q_{y}'] for y in range(2017, 2022)},
                             inc=inc[0], inc_1720=inc[2], inc_2021=inc[3], pass_2021=bool(x.q2021 <= inc[3]), beats_1720=bool(x.mean_1720 <= inc[2]),
                             **comp[x.design]))
            r = rows[-1]
            r['best_comp_1720'] = min(r['q_har_1720'], r['q_xgb_single_1720'], r['q_lgb_single_1720'])
            r['best_comp_2021'] = min(r['q_har_2021'], r['q_xgb_single_2021'], r['q_lgb_single_2021'])
            r['ne_huzhe_komp'] = bool(r['mean_1720'] <= r['best_comp_1720'] and r['q2021'] <= r['best_comp_2021'])
    E = pd.DataFrame(rows)
    E.to_csv(OUT / 'eval_D19.csv', index=False)
    with pd.option_context('display.width', 250, 'display.max_columns', 30):
        print(E[['boost', 'kind', 'design', 'n_failed', 'failed', 'mean_1720', 'q2021', 'pass_2021', 'best_comp_1720', 'best_comp_2021', 'ne_huzhe_komp']].round(4).to_string())


def leak_test():
    """Будущее не влияет: (а) r1 val — прогноз года шага моделью шага (year == 2015+step); (б) rho1/rho4 шага s не меняются при порче
    строк шагов >= s; (в) коэффициенты HAR-J h=1 шага s (rho2) не меняются при порче всех данных с val-года шага s."""
    rng = np.random.default_rng(0)
    A, harj = GH.load_pit(FEAT)
    hv1, _, _ = GH.har_forecasts(1, A, harj); hv1 = hv1.rename(columns={'rv_pred': 'har1'})
    r = r1_val('top5').merge(hv1, on=K)
    assert (r.year == 2015 + r.step).all(); res = ['(а) r1: year == 2015+step для всех строк val: OK']
    hv, _, _ = GH.har_forecasts(5, A, harj)
    V = hv.merge(r.drop(columns=['year', 'rv_actual']), on=K)
    V = V[V.rv_actual > 1e-12]; V['r1'] = np.log(V.hyb1) - np.log(V.har1); V['res'] = np.log(V.rv_actual) - np.log(V.rv_pred)
    ols = lambda d: float((d.r1 * d.res).sum() / (d.r1 ** 2).sum())
    for s_ in range(2, 10):
        a0 = ols(V[V.step < s_]); W = V.copy(); f = W.step >= s_
        W.loc[f, 'r1'] = rng.normal(size=f.sum()); W.loc[f, 'res'] = rng.normal(size=f.sum()) * 5
        assert ols(W[W.step < s_]) == a0
    res.append('(б) rho1 шагов 2-9 (val) не меняется при порче r1 и остатков шагов >= s: OK')
    c0 = har1_coefs(A, harj)
    years = [y for y in range(2017, 2026) if y in A.year.values]
    num = [c for c in A.columns if pd.api.types.is_numeric_dtype(A[c]) and c != 'year']
    for s_, ty in enumerate(years, 1):
        B = A.copy(); f = B.year >= ty - 1
        B.loc[f, num] = B.loc[f, num].values * rng.uniform(0.2, 5, size=(f.sum(), len(num)))
        c1 = har1_coefs(B, harj)
        assert all(abs(c1[s_][k] - c0[s_][k]) < 1e-12 for k in c0[s_]), s_
        assert irf_rho(c1[s_], 22) == irf_rho(c0[s_], 22)
    res.append('(в) коэффициенты HAR-J h=1 и rho2 шага s не меняются при порче всех данных с val-года шага s: OK')
    x = pd.read_parquet(R / 'D16_5_h22_xgb_h22_val.parquet'); h5 = pd.read_parquet(R / 'D12_h5_xgb_h5_wrap2_val.parquet')
    assert (x.year == 2015 + x.step).all() and (h5.year == 2015 + h5.step).all()
    res.append('(г) ноги Д16.5 (rx, Д19.4x) и D12_h5 (hyb5, Д19.4): val — год шага моделью шага (year == 2015+step): OK')
    for nm in ('rho4_h22.csv', 'rho4x_h22.csv'):  # (д) rho/kappa val шага s = rho/kappa test шага s-1 (обе оценки — по шагам < s)
        q = pd.read_csv(OUT / nm); c = 'rho4' if nm.startswith('rho4_') else 'rho'
        assert np.allclose(q[f'{c}_val'].values[1:], q[f'{c}_test'].values[:-1])
    res.append('(д) Д19.1/19.4/19.4x: rho (и kappa) для val шага s совпадает с оценкой по шагам <= s-1 (та же OLS без шага s и дальше): OK')
    open(OUT / 'leak_test.txt', 'w').write('\n'.join(res) + '\n'); print('\n'.join(res))


YAML = {
    'D19_1_h5': ('сигнал h=1 + динамика (Д19)', 'Д19.1', 'ln y5 = ln HAR-J_5 + rho5*r1, r1 = ln(гибрид h=1 Д8.12) - ln(HAR-J_1) из OOS-прогнозов шагов; '
                 'rho5 - OLS без константы по OOS-истории шагов < t (0,75-0,85); один прогноз в обоих слотах', 3),
    'D19_1_h22': ('сигнал h=1 + динамика (Д19)', 'Д19.1', 'ln y22 = ln HAR-J_22 + rho22*r1, r1 как Д19.1; rho22 - OLS без константы по OOS-истории шагов < t '
                  '(0,56-0,71); один прогноз в обоих слотах', 4),
    'D19_4_h22': ('структура h=22: финалист h=5 + дальний HAR с сигналом h=1 (Д18.1 x Д19.1)', 'Д19.4',
                  'y22 = (5*hyb5 + 17*exp(ln HAR-J_{6-22} + rho*r1))/22 на уровне RV; hyb5 - нога D12_h5 (wrap2, XGB, до поправки); '
                  'HAR-J_{6-22} - HAR-J шага на среднее RV t+6..t+22; rho - OLS без константы по OOS-истории шагов < t; r1 как Д19.1', 4),
    'D19_1x_h22': ('сигнал h=1 + динамика (Д19)', 'Д19.1 x Д16.5',
                   'гибрид трёх ног h=22: HAR-J, XGB-слот = Д19.1 (ln HAR-J_22 + rho22*r1), LGB-слот = Д16.5 (чистый бустинг с регрессорами HAR-X, XGB single); '
                   'решение владельца 01:40 (нога Д16.5 вместо V9)', 8),
    'D19_4y_h22': ('структура h=22: финалист h=5 + дальний HAR с сигналом h=1 (Д18.1 x Д19.1)', 'Д19.4 x Д16.5',
                   'гибрид трёх ног h=22: HAR-J, XGB-слот = Д19.4 ((5*hyb5 + 17*exp(ln HAR-J_{6-22} + rho*r1))/22), LGB-слот = Д16.5 (XGB single); '
                   'решение владельца 01:40 (нога Д16.5 вместо V9)', 8)}


def write_yaml(names):
    """yaml кандидатов третьего круга (формат 2026-10-26_final3.md) -> data/regen/final3/kandidaty/<boost>.yaml."""
    d = R / 'final3' / 'kandidaty'
    for b in names:
        fam, ry, op, nv = YAML[b]; h = int(b.split('_h')[1])
        txt = (f'boost: {b}\nh: {h}\nsemeystvo: "{fam}"\nrychag: {ry}\nopisanie: "{op}"\nregen: data/regen/D19\nlegs: [single]\n'
               f'odinochnaya: false\nchistyy: false\ntest_dir: data/regen/test_locked/D19_h{h}\nn_variantov: {nv}\n')
        (d / f'{b}.yaml').write_text(txt); print('yaml:', d / f'{b}.yaml')


def tables():
    """Таблицы отчёта (eval_D19.csv, rho*.csv, delta*.csv) -> data/regen/D19/table_main.md, table_rho.md."""
    D = str(OUT) + "/"
    E = pd.read_csv(D + 'eval_D19.csv')
    f = lambda x: f'{x:.4f}'.replace('.', ',')
    E['base'] = E.boost.str.replace(r'_s4[234]$', '', regex=True)
    E['seed'] = E.boost.str.extract(r'_(s4[234])$')[0].fillna('top5')
    out = []
    out.append('| h | вариант | дизайн (лучший по правилу) | нарушено | 2017 | 2018 | 2019 | 2020 | 2017–2020 | 2021 | 2021 ≤ лучшего кругов 1–2 | seed 42/43/44: 2017–2020 | seed 42/43/44: 2021 (проход) | одиночная модель V0_XGB: нарушено; 2017–2020 / 2021 | лучшая компонента (та же s²) 2017–2020 / 2021 | не хуже компонент |')
    out.append('|' + '---|' * 16)
    order = ['D19_1_h5', 'D19_2_h5', 'D19_3_h5', 'D19_1_h22', 'D19_2_h22', 'D19_3_h22', 'D19_4_h22', 'D19_1x_h22', 'D19_3x_h22', 'D19_4x_h22', 'D19_4y_h22']
    for b in order:
        g = E[E.base == b]
        x = g[(g.seed == 'top5') & (g.kind == 'лучший')].iloc[0]
        v = g[(g.seed == 'top5') & (g.kind == 'V0_XGB')].iloc[0]
        sd = g[(g.seed != 'top5') & (g.kind == 'лучший')].sort_values('seed')
        s1720 = ' / '.join(f(q) for q in sd.mean_1720) if len(sd) else 'н/п'
        s21 = ' / '.join(f'{f(q)}{"✓" if p else "✗"}' for q, p in zip(sd.q2021, sd.pass_2021)) if len(sd) else 'н/п'
        out.append(f'| {x.h} | {b} | {x.design.replace("|", "/")} | {x.n_failed} ({x.failed if isinstance(x.failed, str) else "—"}) | '
                   + ' | '.join(f(x[f"q_{y}"]) for y in range(2017, 2021)) + f' | {f(x.mean_1720)} | {f(x.q2021)} | {"да" if x.pass_2021 else "нет"} (≤ {f(x.inc_2021)}) | '
                   f'{s1720} | {s21} | {v.n_failed} ({v.failed if isinstance(v.failed, str) else "—"}); {f(v.mean_1720)} / {f(v.q2021)} | '
                   f'{f(x.best_comp_1720)} / {f(x.best_comp_2021)} | {"да" if x.ne_huzhe_komp else "нет"} |')
    open(D + 'table_main.md', 'w').write('\n'.join(out) + '\n')
    r = []
    for h in (5, 22):
        q = pd.read_csv(D + f'rho_h{h}.csv')
        for sd in ('top5', 's42', 's43', 's44'):
            g = q[q.seed == sd].sort_values('step')
            r.append(f'| {h} | Д19.1 ρ (val шага; hyb₁ {sd}) | ' + ' | '.join(f'{v:.3f}'.replace('.', ',') for v in g.rho1_val) + ' |')
        g = q[q.seed == 'top5'].sort_values('step')
        r.append(f'| {h} | Д19.2 ρ (отклик HAR-J шага) | ' + ' | '.join(f'{v:.3f}'.replace('.', ',') for v in g.rho2) + ' |')
    for sd, sfx in (('top5', ''), ('s42', '_s42'), ('s43', '_s43'), ('s44', '_s44')):
        q = pd.read_csv(D + f'rho4_h22{sfx}.csv')
        r.append(f'| 22 | Д19.4 ρ (дальняя цель 6–22; {sd}) | ' + ' | '.join(f'{v:.3f}'.replace('.', ',') for v in q.rho4_val) + ' |')
    for sd, sfx in (('top5', ''),):
        q = pd.read_csv(D + f'rho4x_h22{sfx}.csv')
        r.append(f'| 22 | Д19.4x ρ при r̂₁ ({sd}) | ' + ' | '.join(f'{v:.3f}'.replace('.', ',') for v in q.rho_val) + ' |')
        r.append(f'| 22 | Д19.4x κ при ln Д16.5 − ln HAR₂₂ ({sd}) | ' + ' | '.join(f'{v:.3f}'.replace('.', ',') for v in q.kappa_val) + ' |')
    for h in (5, 22):
        q = pd.read_csv(D + f'delta_D19_3_h{h}.csv')
        r.append(f'| {h} | Д19.3 δ (val шага, выбран на шаге t−1) | ' + ' | '.join(f'{v:.1f}'.replace('.', ',') for v in q.delta_val) + ' |')
    q = pd.read_csv(D + 'delta_D19_3x_h22.csv')
    r.append('| 22 | Д19.3x δ | ' + ' | '.join(f'{v:.1f}'.replace('.', ',') for v in q.delta_val) + ' |')
    hdr = '| h | параметр | ' + ' | '.join(f'шаг {s} ({2015 + s})' for s in range(1, 10)) + ' |\n|' + '---|' * 11 + '\n'
    open(D + 'table_rho.md', 'w').write(hdr + '\n'.join(r) + '\n')
    print(open(D + 'table_main.md').read()); print(hdr + '\n'.join(r))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--only', type=int, nargs='*', default=[1, 2, 3]); ap.add_argument('--h', type=int, nargs='*', default=[5, 22])
    ap.add_argument('--eval', action='store_true'); ap.add_argument('--leak', action='store_true'); ap.add_argument('--yaml', nargs='*')
    a = ap.parse_args()
    if a.eval:
        evaluate(); return tables()
    if a.yaml:
        return write_yaml(a.yaml)
    if a.leak:
        return leak_test()
    OUT.mkdir(parents=True, exist_ok=True)
    A, harj = GH.load_pit(FEAT)
    hv1, ht1, _ = GH.har_forecasts(1, A, harj)
    hv1 = hv1.rename(columns={'rv_pred': 'har1'})
    co = har1_coefs(A, harj)
    seeds = ['top5', 's42', 's43', 's44']
    if 6 in a.only:  # Д19.3 на признаках Д16.5 (h=22)
        hv, ht, _ = GH.har_forecasts(22, A, harj)
        ov, ot = d193x(A, harj, hv.rename(columns={'rv_pred': 'har'}), ht.rename(columns={'rv_pred': 'har'}))
        for slot in ('xgb', 'lgbm'):
            ov.to_parquet(OUT / f'D19_3x_h22_{slot}_h22_val.parquet', index=False); ot.to_parquet(TL / f'D19_h22/D19_3x_h22_{slot}_h22_test.parquet', index=False)
        return
    if 5 in a.only:  # варианты с ногой Д16.5 (h=22)
        for sd in seeds:
            d19x(A, harj, hv1, sd)
        return
    if 4 in a.only:
        for sd in seeds:
            rho, ov, ot = d194(A, harj, hv1, sd)
            sfx = '' if sd == 'top5' else f'_{sd}'
            rho.to_csv(OUT / f'rho4_h22{sfx}.csv', index=False); print(sd, rho.round(3).to_string())
            td = TL / 'D19_h22'; td.mkdir(parents=True, exist_ok=True)
            for slot in ('xgb', 'lgbm'):
                ov.to_parquet(OUT / f'D19_4_h22{sfx}_{slot}_h22_val.parquet', index=False)
                if sd == 'top5':
                    ot.to_parquet(td / f'D19_4_h22_{slot}_h22_test.parquet', index=False)
        return
    for h in a.h:
        rho, out, _ = build(h, A, harj, hv1, ht1, co, seeds, set(a.only))
        rho.to_csv(OUT / f'rho_h{h}.csv', index=False)
        td = TL / f'D19_h{h}'; td.mkdir(parents=True, exist_ok=True)
        for (m, sd), (ov, ot) in out.items():
            b = f'D19_{m}_h{h}'
            sfx = '' if sd == 'top5' else f'_{sd}'
            for slot in ('xgb', 'lgbm'):
                ov.to_parquet(OUT / f'{b}{sfx}_{slot}_h{h}_val.parquet', index=False)
                if sd == 'top5':
                    ot.to_parquet(td / f'{b}_{slot}_h{h}_test.parquet', index=False)
        print(rho[rho.seed == 'top5'].round(3).to_string(), flush=True)


if __name__ == '__main__':
    main()
