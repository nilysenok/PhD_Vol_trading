"""Д18: структурный гибрид по горизонту (третий круг, новый дизайн).
Идея: на уровне RV среднее за [t+1, t+h] раскладывается арифметически: 22*m(1..22) = k*m(1..k) + (22-k)*m(k+1..22).
Ближнюю часть даёт финалист меньшего горизонта (hyb_1 / hyb_5, только OOS-файлы финалистов), дальнюю - HAR-J на цель подгоризонта.
  Д18.1 h=22: (5*hyb5 + 17*HAR[6..22])/22          Д18.2 h=5: (1*hyb1 + 4*HAR[2..5])/5
  Д18.3 h=22: (1*hyb1 + 21*HAR[2..22])/22 и выбор k in {1,5} по OOS-QLIKE шага t-1
  Д18.4 h=5/22: HAR-J, где ln RV_d заменён на ln hyb1 (OOS-прогнозы прошлых шагов на обучающем окне; шаг 1 - только val-год)
HAR-J на подгоризонт: те же регрессоры, что gibrid_har.load_pit (rv_d, rv_w, rv_m, jump*), OLS в логах на обучающем окне шага
(годы < val-года шага), ЭМБАРГО: строка t обучающего окна берётся, только если дата t+hi (конец цели) < начала val-года.
Поправка Б0.1: F = exp(s2/2), s2 по остаткам ln val-лет (sm - дисперсия трёх последних, smrob - (1,4826 MAD)^2), на val шага t - s2 шага t-1.
Компоненты смешиваются БЕЗ поправки (сырые), поправка - на итоговый прогноз (как V0 гибрида: одна оценка s2 на прогноз).
Вход (чтение): data/regen/features_pit_v3.parquet; OOS val финалистов D12_h1_A_d812 (top5, xgb) и D12_h5 (wrap2, xgb);
тест - test_locked/final2/final_h{1,5}_test.parquet (xgb_raw; только сборка, метрики теста не считаются).
Подтверждающий (val шагов 7-9) - прогнозы строятся (нужны пулу), метрики НЕ считаются.
Выход: data/regen/D18_har_sub_h{5,22}_{lo}_{hi}_{val,test}.parquet (для Д19), data/regen/D18/*, кандидаты {boost}_{xgb|lgbm}_h{h}_{val}.parquet,
test_locked/<boost>/..._test.parquet.
python3 scripts/analysis/d18_struct.py [--selftest]"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import LinearRegression

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gibrid_har as GH
from gibrid_io import load_boost, MAIN_REGEN

R = MAIN_REGEN
OUT = R / 'D18'
LOCK = R / 'test_locked'
DEV = 6


def qlike(y, p):
    p = np.clip(p, 1e-10, None)
    return float(np.mean(y / p - np.log(y / p) - 1))


def load():
    a, harj = GH.load_pit(R / 'features_pit_v3.parquet')
    a = a.sort_values(['ticker', 'date']).reset_index(drop=True)
    g = a.groupby('ticker', sort=False)
    for j in range(1, 23):
        a[f'_f{j}'] = g['rv'].shift(-j)
        a[f'_fd{j}'] = g['date'].shift(-j)
    assert np.allclose(a['_f1'].fillna(-1), a['rv_target_h1'].fillna(-1))
    m22 = a[[f'_f{j}' for j in range(1, 23)]].mean(axis=1, skipna=False)
    assert np.nanmax(np.abs(m22 - a.rv_target_mean_h22)) < 1e-12
    return a, harj


def sub_target(a, lo, hi):
    return a[[f'_f{j}' for j in range(lo, hi + 1)]].mean(axis=1, skipna=False), a[f'_fd{hi}']


def steps(a):
    return [(s, ty, ty - 1) for s, ty in enumerate([y for y in range(2017, 2026) if y in a.year.values], 1)]


def fit_step(a, X, y, end_date, vy, train_mask=None):
    """OLS на окне шага: годы < vy, эмбарго - дата конца цели < 1 янв vy."""
    tr = (a.year < vy) & y.notna() & (end_date < pd.Timestamp(f'{vy}-01-01'))
    if train_mask is not None:
        tr &= train_mask
    m = LinearRegression().fit(X[tr.values], np.log(y[tr].values + 1e-10))
    return m, int(tr.sum())


def har_sub(a, harj, lo, hi, regs=None, Xdf=None, test_X=None):
    """Прогнозы HAR-J на цель m(lo..hi): val (val-год шага) и тест (тест-год шага) для шагов 1-9. Xdf - своя матрица ln-регрессоров
    (Д18.4: по шагам; функция шага), иначе ln HAR-J."""
    y, ed = sub_target(a, lo, hi)
    Xh = GH.clean(np.log(a[harj].values.clip(1e-10)))
    V, T, info = [], [], []
    for s, ty, vy in steps(a):
        X = Xh if Xdf is None else Xdf(s)
        if Xdf is not None:
            mask = pd.Series(np.isfinite(X).all(axis=1), index=a.index)  # даты без OOS-прогноза hyb1 - вне окна
            mtr = mask & (a.year >= 2016) if s > 1 else mask & (a.year == 2016)
            if s == 1:  # шаг 1: только val-год (OOS-прогнозов hyb1 раньше нет) - in-sample, шаг 1 метрик не даёт
                tr = mtr & y.notna()
                m = LinearRegression().fit(np.nan_to_num(X[tr.values]), np.log(y[tr].values + 1e-10)); n = int(tr.sum())
            else:
                m, n = fit_step(a, np.nan_to_num(X), y, ed, vy, mtr)
            Xv = np.nan_to_num(X)
        else:
            m, n = fit_step(a, X, y, ed, vy)
            Xv = X
        info.append(dict(lo=lo, hi=hi, step=s, n_train=n, coef=json.dumps([round(float(c), 5) for c in m.coef_]), b=float(m.intercept_)))
        for yr, lst, XX in ((vy, V, Xv), (ty, T, Xv if test_X is None else test_X(s))):
            sel = (a.year == yr).values
            Xs = XX[sel]
            p = np.clip(np.exp(m.predict(np.nan_to_num(Xs))), 1e-10, None)
            p[~np.isfinite(Xs).all(axis=1)] = np.nan  # нет прогноза hyb1 на дату - нет прогноза
            lst.append(pd.DataFrame({'date': a.date.values[sel], 'ticker': a.ticker.values[sel], 'year': yr, 'step': s,
                                     'rv_actual': y.values[sel], 'rv_pred': p}))
    return pd.concat(V, ignore_index=True), pd.concat(T, ignore_index=True), pd.DataFrame(info)


def load_hyb():
    """Сырые ноги финалистов (до поправки): hyb1 = D12_h1_A_d812 top5 XGB, hyb5 = D12_h5 wrap2 XGB; тест - xgb_raw файлов сборки."""
    out = {}
    for h, boost, var in ((1, 'D12_h1_A_d812', 'top5'), (5, 'D12_h5', 'wrap2')):
        v = load_boost(boost, h, R, var)[0]['val'][['date', 'ticker', 'step', 'rv_pred']]
        t = pd.read_parquet(LOCK / 'final2' / f'final_h{h}_test.parquet', columns=['date', 'ticker', 'step', 'xgb_raw', 'design'])
        assert t.design.iloc[0].startswith(var + '|V0_XGB')
        t = t[t.step <= 9].rename(columns={'xgb_raw': 'rv_pred'})[['date', 'ticker', 'step', 'rv_pred']]
        out[h] = (v, t)
    return out


def load_hyb_seed(seed):
    """Seed-варианты: одиночная точка XGB seed 42 (файл _val) / 43, 44 (_seeds4344_val) - только val."""
    out = {}
    for h, boost in ((1, 'D12_h1_A_d812'), (5, 'D12_h5')):
        if seed == 42:
            v = pd.read_parquet(R / f'{boost}_xgb_h{h}_val.parquet')
        else:
            v = pd.read_parquet(R / f'{boost}_xgb_h{h}_seeds4344_val.parquet'); v = v[v.seed == seed]
        out[h] = (v[['date', 'ticker', 'step', 'rv_pred']], None)
    return out


def s2_tab(df):
    """s2 по шагам (по val-остаткам ln итогового прогноза): sm, smrob - три последних val-года."""
    E = {s: np.log(g.rv_actual.values + 1e-10) - np.log(g.rv_pred.values) for s, g in df.groupby('step')}
    res = {}
    for code in ('sm', 'smrob'):
        for s in E:
            e = np.concatenate([E[j] for j in range(max(1, s - 2), s + 1) if j in E])
            res[(code, s)] = float((1.4826 * np.median(np.abs(e - np.median(e)))) ** 2) if code == 'smrob' else float(np.var(e))
    return res


def correct(val, test, code):
    """OOS: val шага t * exp(s2_{t-1}/2) (шаг 1: F=1); тест шага t * exp(s2_t/2)."""
    S = s2_tab(val[val.step <= 9])
    v, t = val.copy(), test.copy() if test is not None else None
    v['rv_pred'] *= v.step.map(lambda s: np.exp(S[(code, s - 1)] / 2) if s > 1 else 1.0)
    if t is not None:
        t['rv_pred'] *= t.step.map(lambda s: np.exp(S[(code, s)] / 2))
    return v, t


def by_year(v):
    d = v[(v.step >= 2) & (v.step <= DEV) & (v.rv_actual > 1e-12)]
    q = {str(int(s) + 2015): qlike(g.rv_actual.values, g.rv_pred.values) for s, g in d.groupby('step')}
    q['2017-2020'] = float(np.mean([q[str(y)] for y in range(2017, 2021)]))
    return q


def combine(parts, wts, key=('date', 'ticker', 'step'), tgt=None):
    """Взвешенное среднее сырых прогнозов (уровень RV); rv_actual - из tgt."""
    m = tgt[list(key) + ['year', 'rv_actual']].copy()
    m['rv_pred'] = 0.0
    for p, w in zip(parts, wts):
        x = m[list(key)].merge(p[list(key) + ['rv_pred']], on=list(key), how='left')
        m['rv_pred'] += w * x.rv_pred.values
    return m.dropna(subset=['rv_pred', 'rv_actual']).reset_index(drop=True)


def target_df(a, h, which):
    """Каркас (date,ticker,year,step,rv_actual) на val-годах (which='val') или тест-годах шагов 1-9."""
    y = a.rv_target_h1 if h == 1 else a[f'rv_target_mean_h{h}']
    L = []
    for s, ty, vy in steps(a):
        yr = vy if which == 'val' else ty
        sel = (a.year == yr).values
        L.append(pd.DataFrame({'date': a.date.values[sel], 'ticker': a.ticker.values[sel], 'year': yr, 'step': s, 'rv_actual': y.values[sel]}))
    return pd.concat(L, ignore_index=True)


def hyb1_regressor(a, harj, hyb):
    """Д18.4: матрица ln-регрессоров с ln hyb1 вместо ln rv_d. На val-годах - OOS-прогноз hyb1 шага этого val-года (одна дата - один шаг);
    тест шага s - тестовый hyb1 шага s."""
    v1, t1 = hyb[1]
    ser = a[['date', 'ticker']].merge(v1[['date', 'ticker', 'rv_pred']], on=['date', 'ticker'], how='left').rv_pred.values
    base = GH.clean(np.log(a[harj].values.clip(1e-10)))
    i = harj.index('rv_d')

    def Xdf(s):
        X = base.copy(); X[:, i] = np.log(np.clip(ser, 1e-10, None)); return X

    def test_X(s):
        tt = t1[t1.step == s]
        x = a[['date', 'ticker']].merge(tt[['date', 'ticker', 'rv_pred']], on=['date', 'ticker'], how='left').rv_pred.values
        X = base.copy(); X[:, i] = np.log(np.clip(x, 1e-10, None)); return X
    return Xdf, test_X


def selftest(a, harj):
    """«Будущее не влияет»: всё, начиная с 1 янв val-года шага, умножается на случайные множители -> коэффициенты HAR подгоризонта
    шага не меняются (эмбарго). Контроль: без эмбарго (фильтр по дате конца цели снят) коэффициенты меняются."""
    rs = np.random.default_rng(0)
    out = []
    for lo, hi in ((6, 22), (2, 5), (2, 22)):
        y, ed = sub_target(a, lo, hi)
        X = GH.clean(np.log(a[harj].values.clip(1e-10)))
        for s, ty, vy in steps(a)[:6]:
            m0, _ = fit_step(a, X, y, ed, vy)
            b = a.copy()
            fut = (b.date >= pd.Timestamp(f'{vy}-01-01')).values
            k = rs.uniform(0.2, 5, fut.sum())
            b.loc[fut, 'rv'] *= k
            for c in harj:
                b.loc[fut, c] *= k
            g = b.groupby('ticker', sort=False)
            for j in range(1, 23):
                b[f'_f{j}'] = g['rv'].shift(-j)
            y2, ed2 = sub_target(b, lo, hi)
            X2 = GH.clean(np.log(b[harj].values.clip(1e-10)))
            m1, _ = fit_step(b, X2, y2, ed2, vy)
            ctl = lambda yy, XX: LinearRegression().fit(XX[((a.year < vy) & yy.notna()).values], np.log(yy[(a.year < vy) & yy.notna()].values + 1e-10)).coef_
            d_emb = float(np.abs(m0.coef_ - m1.coef_).max())
            d_ctl = float(np.abs(ctl(y, X) - ctl(y2, X2)).max())
            out.append(dict(lo=lo, hi=hi, step=s, max_dcoef_embargo=d_emb, max_dcoef_bez_embargo=d_ctl, ok=d_emb < 1e-10 and d_ctl > 1e-6))
    # Д18.4 (ln hyb1 вместо ln RV_d, цель m(1..h)): то же возмущение будущего, шаги 2-6 (шаг 1 - in-sample на val-году по построению)
    hyb = load_hyb()
    for h in (5, 22):
        for s, ty, vy in steps(a)[1:6]:
            coefs = []
            for b in (a, None):
                if b is None:
                    b = a.copy()
                    fut = (b.date >= pd.Timestamp(f'{vy}-01-01')).values
                    k = rs.uniform(0.2, 5, fut.sum())
                    b.loc[fut, 'rv'] *= k
                    for c in harj:
                        b.loc[fut, c] *= k
                    g = b.groupby('ticker', sort=False)
                    for j in range(1, 23):
                        b[f'_f{j}'] = g['rv'].shift(-j)
                y, ed = sub_target(b, 1, h)
                X = hyb1_regressor(b, harj, hyb)[0](s)
                mtr = pd.Series(np.isfinite(X).all(axis=1), index=b.index) & (b.year >= 2016)
                coefs.append(fit_step(b, np.nan_to_num(X), y, ed, vy, mtr)[0].coef_)
            d = float(np.abs(coefs[0] - coefs[1]).max())
            out.append(dict(lo=1, hi=h, step=s, max_dcoef_embargo=d, max_dcoef_bez_embargo=np.nan, ok=d < 1e-10, model='D18.4'))
    return pd.DataFrame(out)


def step10(a, harj):
    """Шаг 10 (формат final2_step10.py / final2_build.leg10): обучение HAR подгоризонта на годах < 2025 (эмбарго), val 2025, тест 01-02.2026;
    ноги hyb - шаг 10 финалистов (final2/step10, test_locked/final2/step10; leg10 с их ногой). Метрики НЕ считаются; rv_actual 2026 может быть NaN.
    Выход: data/regen/D18/step10/D18_h22_{xgb|lgbm}_h22_s10_val.parquet, test_locked/D18_h22/step10/..._s10_test.parquet (seed = 42)."""
    import final2_build as FB
    L1 = FB.leg10('D12_h1_A_d812', 1, 'top5')['xgb']
    L5 = FB.leg10('D12_h5', 5, 'wrap2')['xgb']
    tgt = a.rv_target_mean_h22

    def har10(lo, hi):
        y, ed = sub_target(a, lo, hi)
        X = GH.clean(np.log(a[harj].values.clip(1e-10)))
        tr = (a.year < 2025) & y.notna() & (ed < pd.Timestamp('2025-01-01'))
        m = LinearRegression().fit(X[tr.values], np.log(y[tr].values + 1e-10))
        res = []
        for yr in (2025, 2026):
            sel = (a.year == yr).values
            if yr == 2025:
                sel &= tgt.notna().values  # val 2025 - строки с целью h=22 (как har_step10: dropna по цели)
            res.append(pd.DataFrame({'date': a.date.values[sel], 'ticker': a.ticker.values[sel], 'year': yr, 'step': 10,
                                     'rv_actual': tgt.values[sel], 'rv_pred': np.clip(np.exp(m.predict(X[sel])), 1e-10, None)}))
        return res
    H6, H2 = har10(6, 22), har10(2, 22)
    sd, lk = OUT / 'step10', LOCK / 'D18_h22' / 'step10'
    sd.mkdir(parents=True, exist_ok=True); lk.mkdir(parents=True, exist_ok=True)
    k = ['date', 'ticker']
    for m, hy, Hs, w in (('xgb', L5, H6, 5 / 22), ('lgbm', L1, H2, 1 / 22)):
        for i, (dst, sfx) in enumerate(((sd, 'val'), (lk, 'test'))):
            x = Hs[i].merge(hy[i][k + ['rv_pred']].rename(columns={'rv_pred': 'hyb'}), on=k, how='left', validate='one_to_one')
            assert x.hyb.notna().all(), (m, sfx, int(x.hyb.isna().sum()))
            x = x.assign(rv_pred=w * x.hyb + (1 - w) * x.rv_pred, seed=42).drop(columns='hyb')
            x.to_parquet(dst / f'D18_h22_{m}_h22_s10_{sfx}.parquet', index=False)
            print(m, sfx, len(x), x.date.min().date(), x.date.max().date(), flush=True)


def d165(a, harj, leg, slot):
    """Решение владельца 01:40: Д18.1 с бустинг-компонентой h=22 = Д16.5 (чистый бустинг с регрессорами HAR-X), нога leg, слот slot (xgb|lgb)
    - представитель Д16.5 по правилу его гибрида. Структуры (веса фиксированы):
      D18_1d = (5*hyb5 + 17*D16.5)/22      - D16.5 (среднее за 1..22) как оценка дальней части 6..22 вместо HAR[6..22];
      D18_1e = (5*hyb5 + 17*(HAR[6..22] + D16.5)/2)/22 - дальняя часть = равное среднее HAR и бустинга;
      кандидат D18x_h22 для гибрида: XGB-слот = D18_1 (основной), LGB-слот = D16.5 -> пространство гибрида смешивает структурный прогноз с бустингом h=22."""
    import final2_build as FB
    hyb = load_hyb()
    T = (target_df(a, 22, 'val'), target_df(a, 22, 'test'))
    H6 = (pd.read_parquet(R / 'D18_har_sub_h22_6_22_val.parquet'), pd.read_parquet(LOCK / 'D18_har_sub' / 'D18_har_sub_h22_6_22_test.parquet'))
    si = 0 if slot == 'xgb' else 1
    bv = load_boost('D16_5_h22', 22, R, leg)[si]['val']
    bt = load_boost('D16_5_h22', 22, R, leg, test=True)[si]['test']
    bt = bt[bt.step <= 9]
    C = {'D16_5': (combine([bv], [1.0], tgt=T[0]), combine([bt], [1.0], tgt=T[1])),
         'D18_1': (combine([hyb[5][0], H6[0]], [5 / 22, 17 / 22], tgt=T[0]), combine([hyb[5][1], H6[1]], [5 / 22, 17 / 22], tgt=T[1])),
         'D18_1d': (combine([hyb[5][0], bv], [5 / 22, 17 / 22], tgt=T[0]), combine([hyb[5][1], bt], [5 / 22, 17 / 22], tgt=T[1])),
         'D18_1e': (combine([hyb[5][0], H6[0], bv], [5 / 22, 8.5 / 22, 8.5 / 22], tgt=T[0]),
                    combine([hyb[5][1], H6[1], bt], [5 / 22, 8.5 / 22, 8.5 / 22], tgt=T[1]))}
    rows = []
    for nm, (v, t) in C.items():
        for code in ('sm', 'smrob'):
            rows.append(dict(name=nm, d165_leg=f'{leg}/{slot}', s2=code, **by_year(correct(v, t, code)[0])))
    Q = pd.DataFrame(rows); Q.to_csv(OUT / 'qlike_d165.csv', index=False); print(Q.to_string(), flush=True)
    # seed-варианты: hyb одиночные точки seed s, D16.5 одиночная точка seed s (XGB и LGBM - оба слота, как в файлах)
    srows = []
    sm = 'xgb' if slot == 'xgb' else 'lgbm'
    for seed in (42, 43, 44):
        hs = load_hyb_seed(seed)
        f = R / (f'D16_5_h22_{sm}_h22_val.parquet' if seed == 42 else f'D16_5_h22_{sm}_h22_seeds4344_val.parquet')
        b = pd.read_parquet(f); b = b[b.seed == seed] if 'seed' in b and seed != 42 else b
        b = b[['date', 'ticker', 'step', 'rv_pred']]
        for nm, parts, w in (('D16_5_single', [b], [1.0]), ('D18_1d', [hs[5][0], b], [5 / 22, 17 / 22]),
                             ('D18_1e', [hs[5][0], H6[0], b], [5 / 22, 8.5 / 22, 8.5 / 22])):
            v = combine(parts, w, tgt=T[0])
            for code in ('sm', 'smrob'):
                srows.append(dict(name=nm, seed=seed, s2=code, **by_year(correct(v, None, code)[0])))
    S = pd.DataFrame(srows); S.to_csv(OUT / 'qlike_d165_seeds.csv', index=False); print(S.to_string(), flush=True)
    save_cand('D18x_h22', 22, C['D18_1'][0], C['D16_5'][0], C['D18_1'][1], C['D16_5'][1])
    save_cand('D18d_h22', 22, C['D18_1d'][0], C['D16_5'][0], C['D18_1d'][1], C['D16_5'][1])
    save_cand('D18e_h22', 22, C['D18_1e'][0], C['D16_5'][0], C['D18_1e'][1], C['D16_5'][1])
    # шаг 10 (формат leg10): XGB-слот - структурный прогноз, LGB-слот - D16.5 той же ноги
    L5 = FB.leg10('D12_h5', 5, 'wrap2')['xgb']
    L16 = FB.leg10('D16_5_h22', 22, leg)[slot]
    sd = OUT / 'step10'
    H10 = {sfx: pd.read_parquet(sd / f'D18_h22_xgb_h22_s10_{sfx}.parquet' if sfx == 'val' else LOCK / 'D18_h22' / 'step10' / f'D18_h22_xgb_h22_s10_{sfx}.parquet')
           for sfx in ('val', 'test')}
    k = ['date', 'ticker']
    for i, sfx in enumerate(('val', 'test')):
        base = H10[sfx][k + ['year', 'step', 'rv_actual', 'rv_pred']].rename(columns={'rv_pred': 'd18_1'})
        x = base.merge(L5[i][k + ['rv_pred']].rename(columns={'rv_pred': 'hyb'}), on=k, validate='one_to_one') \
                .merge(L16[i][k + ['rv_pred']].rename(columns={'rv_pred': 'b'}), on=k, validate='one_to_one')
        assert len(x) == len(base)
        far = (x.d18_1 - 5 / 22 * x.hyb) * 22 / 17  # HAR[6..22] шага 10 из сохранённого D18_1
        outs = {'D18x_h22': (x.d18_1, x.b), 'D18d_h22': (5 / 22 * x.hyb + 17 / 22 * x.b, x.b),
                'D18e_h22': (5 / 22 * x.hyb + 8.5 / 22 * far + 8.5 / 22 * x.b, x.b)}
        for bn, (px, pl) in outs.items():
            dst = sd if sfx == 'val' else LOCK / bn / 'step10'
            dst.mkdir(parents=True, exist_ok=True)
            for m, p in (('xgb', px), ('lgbm', pl)):
                x[k + ['year', 'step', 'rv_actual']].assign(rv_pred=p.values, seed=42).to_parquet(dst / f'{bn}_{m}_h22_s10_{sfx}.parquet', index=False)
    print('D16.5-варианты: кандидаты и шаг 10 записаны', flush=True)


def seeds_x(a):
    """Три seed для D18x (запрос дирижёра): XGB-слот = Д18.1 с hyb5 = одиночная точка XGB seed s, LGB-слот = Д16.5 single XGB seed s.
    Только val (для gibrid_v17 --regen data/regen/D18/seeds, тест не нужен). Выход: D18/seeds/D18xs{s}_h22_{xgb|lgbm}_h22_val.parquet."""
    T = target_df(a, 22, 'val')
    H6 = pd.read_parquet(R / 'D18_har_sub_h22_6_22_val.parquet')
    d = OUT / 'seeds'; d.mkdir(parents=True, exist_ok=True)
    cols = ['date', 'ticker', 'year', 'step', 'rv_actual', 'rv_pred']
    for seed in (42, 43, 44):
        hs = load_hyb_seed(seed)
        b = pd.read_parquet(R / ('D16_5_h22_xgb_h22_val.parquet' if seed == 42 else 'D16_5_h22_xgb_h22_seeds4344_val.parquet'))
        b = (b[b.seed == seed] if seed != 42 else b)[['date', 'ticker', 'step', 'rv_pred']]
        combine([hs[5][0], H6], [5 / 22, 17 / 22], tgt=T)[cols].to_parquet(d / f'D18xs{seed}_h22_xgb_h22_val.parquet', index=False)
        combine([b], [1.0], tgt=T)[cols].to_parquet(d / f'D18xs{seed}_h22_lgbm_h22_val.parquet', index=False)
    print('seed-файлы D18x записаны', flush=True)


def summary():
    """Представители кандидатов по правилу третьего круга (gibrid_v17 --dev-eval oos): ключ - число нарушенных c1-c4 (шаги 2-6), затем
    средний OOS QLIKE 2017-2020 (шаги 2-5); QLIKE по годам 2017-2021 из designs_val (шаги 7-9 скрыты гибридом)."""
    rows = []
    for d in sorted((OUT / 'gibrid').iterdir()):
        if not d.is_dir():
            continue
        b = d.name
        for f in d.glob(f'designs_val_{b}_h*.csv'):
            h = int(f.stem.rsplit('_h', 1)[1])
            D = pd.read_csv(f)
            D = D[(D.step >= 2) & (D.step <= DEV)]
            for nm, g in D.groupby('design', sort=False):
                q = dict(zip(g.val_year.astype(int), g.q_hyb))
                rows.append(dict(boost=b, h=h, design=nm, n_failed=sum(not bool(g[c + '_h'].iloc[0]) for c in ('c1', 'c2', 'c3', 'c4')),
                                 failed=','.join(c for c in ('c1', 'c2', 'c3', 'c4') if not bool(g[c + '_h'].iloc[0])),
                                 **{str(y): q.get(y) for y in range(2017, 2022)}, q1720=np.mean([q[y] for y in range(2017, 2021)])))
    A = pd.DataFrame(rows).sort_values(['boost', 'n_failed', 'q1720'])
    A.to_csv(OUT / 'gibrid_all_designs.csv', index=False)
    rep = A.groupby('boost').head(1)
    rep.to_csv(OUT / 'gibrid_representatives.csv', index=False)
    print(rep.to_string(), flush=True)
    return A


def save_cand(boost, h, val_x, val_l, test_x, test_l):
    cols = ['date', 'ticker', 'year', 'step', 'rv_actual', 'rv_pred']
    td = LOCK / boost
    td.mkdir(parents=True, exist_ok=True)
    for m, v, t in (('xgb', val_x, test_x), ('lgbm', val_l, test_l)):
        v[cols].to_parquet(R / f'{boost}_{m}_h{h}_val.parquet', index=False)
        t[cols].to_parquet(td / f'{boost}_{m}_h{h}_test.parquet', index=False)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    a, harj = load()
    print('HAR-J регрессоры:', harj, flush=True)
    if '--selftest' in sys.argv:
        st = selftest(a, harj); st.to_csv(OUT / 'selftest_future.csv', index=False); print(st.to_string()); return
    if '--step10' in sys.argv:
        step10(a, harj); return
    if '--seeds-x' in sys.argv:
        seeds_x(a); return
    if '--summary' in sys.argv:
        summary(); return
    if '--d165' in sys.argv:  # --d165 <нога> <слот xgb|lgb> - представитель Д16.5 по его гибриду
        i = sys.argv.index('--d165'); d165(a, harj, sys.argv[i + 1], sys.argv[i + 2]); return
    # 1) HAR на подгоризонты - выкладываются сразу (их ждёт Д19)
    H, infos = {}, []
    for h, lo, hi in ((22, 6, 22), (22, 2, 22), (5, 2, 5)):
        v, t, inf = har_sub(a, harj, lo, hi)
        v.to_parquet(R / f'D18_har_sub_h{h}_{lo}_{hi}_val.parquet', index=False)
        (LOCK / 'D18_har_sub').mkdir(parents=True, exist_ok=True)
        t.to_parquet(LOCK / 'D18_har_sub' / f'D18_har_sub_h{h}_{lo}_{hi}_test.parquet', index=False)
        H[(lo, hi)] = (v, t); infos.append(inf)
        print(f'HAR[{lo}..{hi}] выложен', flush=True)
    pd.concat(infos).to_csv(OUT / 'har_sub_coef.csv', index=False)
    hyb = load_hyb()
    T = {h: (target_df(a, h, 'val'), target_df(a, h, 'test')) for h in (5, 22)}
    # канонический HAR-J h (без эмбарго, как гибрид) - эталон-фолбэк
    base = {}
    for h in (5, 22):
        hv, ht, _ = GH.har_forecasts(h, a.sort_values('date').reset_index(drop=True), harj)
        base[h] = (hv, ht)
    C = {}  # имя -> (val сырые, test сырые)
    C['D18_1_h22'] = (combine([hyb[5][0], H[(6, 22)][0]], [5 / 22, 17 / 22], tgt=T[22][0]),
                      combine([hyb[5][1], H[(6, 22)][1]], [5 / 22, 17 / 22], tgt=T[22][1]))
    C['D18_3_h22'] = (combine([hyb[1][0], H[(2, 22)][0]], [1 / 22, 21 / 22], tgt=T[22][0]),
                      combine([hyb[1][1], H[(2, 22)][1]], [1 / 22, 21 / 22], tgt=T[22][1]))
    C['D18_2_h5'] = (combine([hyb[1][0], H[(2, 5)][0]], [1 / 5, 4 / 5], tgt=T[5][0]),
                     combine([hyb[1][1], H[(2, 5)][1]], [1 / 5, 4 / 5], tgt=T[5][1]))
    Xdf, test_X = hyb1_regressor(a, harj, hyb)
    for h in (5, 22):
        v, t, inf = har_sub(a, harj, 1, h, Xdf=Xdf, test_X=test_X)
        inf.to_csv(OUT / f'd18_4_coef_h{h}.csv', index=False)
        C[f'D18_4_h{h}'] = (v, t)
    C['HARJ_h22'] = base[22]; C['HARJ_h5'] = base[5]
    C['hyb5_D12_h5'] = (combine([hyb[5][0]], [1.0], tgt=T[5][0]), combine([hyb[5][1]], [1.0], tgt=T[5][1]))
    # Д18.3: выбор k по OOS-QLIKE шага t-1 (после поправки той же s2); шаг 1 -> k=5 (основной дизайн, без данных)
    rows, CORR = [], {}
    for nm, (v, t) in C.items():
        for code in ('sm', 'smrob'):
            vc, tc = correct(v, t, code)
            CORR[(nm, code)] = (vc, tc)
            rows.append(dict(name=nm, s2=code, **by_year(vc)))
    for code in ('sm', 'smrob'):
        v5, t5 = CORR[('D18_1_h22', code)]; v1, t1 = CORR[('D18_3_h22', code)]
        q = lambda v: {s: qlike(g.rv_actual.values, g.rv_pred.values) for s, g in v[v.rv_actual > 1e-12].groupby('step')}
        q5, q1 = q(v5), q(v1)
        kk = {s: (5 if s == 1 or q5[s - 1] <= q1[s - 1] else 1) for s in sorted(q5)}
        kt = {s: (5 if q5[s] <= q1[s] else 1) for s in sorted(q5)}  # тест шага s: выбор по val шага s (последний известный)
        vs = pd.concat([(v5 if kk[s] == 5 else v1)[lambda d: d.step == s] for s in kk])
        ts = pd.concat([(t5 if kt[s] == 5 else t1)[lambda d: d.step == s] for s in kt])
        CORR[('D18_3s_h22', code)] = (vs, ts)
        rows.append(dict(name='D18_3s_h22', s2=code, k_by_step=json.dumps(kk), **by_year(vs)))
    Q = pd.DataFrame(rows)
    Q.to_csv(OUT / 'qlike_by_year.csv', index=False)
    print(Q.to_string(), flush=True)
    # seed-варианты (одиночная точка XGB seed 42/43/44 в обеих ногах hyb)
    srows = []
    for seed in (42, 43, 44):
        hs = load_hyb_seed(seed)
        for nm, parts, w, h in (('D18_1_h22', [hs[5][0], H[(6, 22)][0]], [5 / 22, 17 / 22], 22),
                                ('D18_3_h22', [hs[1][0], H[(2, 22)][0]], [1 / 22, 21 / 22], 22),
                                ('D18_2_h5', [hs[1][0], H[(2, 5)][0]], [1 / 5, 4 / 5], 5),
                                ('hyb5_single', [hs[5][0]], [1.0], 5)):
            v = combine(parts, w, tgt=T[h][0])
            for code in ('sm', 'smrob'):
                srows.append(dict(name=nm, seed=seed, s2=code, **by_year(correct(v, None, code)[0])))
    S = pd.DataFrame(srows); S.to_csv(OUT / 'qlike_seeds.csv', index=False); print(S.to_string(), flush=True)
    # кандидаты для пула/gibrid_v17: сырые прогнозы (поправку делает гибрид); XGB-слот - основной дизайн, LGB-слот - второй вариант
    save_cand('D18_h22', 22, C['D18_1_h22'][0], C['D18_3_h22'][0], C['D18_1_h22'][1], C['D18_3_h22'][1])
    save_cand('D18_h5', 5, C['D18_2_h5'][0], C['D18_4_h5'][0], C['D18_2_h5'][1], C['D18_4_h5'][1])
    for nm in C:
        C[nm][0].to_parquet(OUT / f'{nm}_raw_val.parquet', index=False)
    print('готово', flush=True)


if __name__ == '__main__':
    main()
