"""Д23 (владелец 26.10, 01:55 по Маку; третий круг, оценка как у Д18-Д22): коррекция свободного члена, корреляции, веса.

Д23.1 коррекция свободного члена по недавней ошибке финалиста h=1.
  e(s) = ln RV(s+1) - ln y1(s), y1 - OOS-прогноз финалиста h=1 (D12_h1_A_d812, нога top5 XGB, сырой, до поправки Б0.1).
  Ошибка строки s известна только после закрытия дня s+1, поэтому к 18:50 дня t известны e(s) при s <= t-1:
    ebar_m(t) = среднее e(t-1), ..., e(t-m) по тикеру (строки тикера), m in {5, 10}; рыночная - то же по средней за дату e.
  h=22 дополнительно: ebar22(t) = e22(t-22), e22(s) = ln m22(s) - ln y22(s) (окно [s+1, s+22] закрывается в s+22).
  Прогноз: ln y_h = ln y_h^base + kappa_h' * x(t), kappa_h - OLS без константы по OOS-истории шагов < t (val-годы 2016..vy-1),
  эмбарго: строка берётся, только если конец её цели < 1 янв. val-года шага (ночное умолчание: "окно шага t-1" = OOS-история до шага t-1).
  Тест шага t: kappa по val-годам 2016..vy (последнее известное), x - из тестового ряда.
  Поправка Б0.1 - на итог (делает gibrid_v17, s2 шага t-1). Компоненты (база) - сырые.
  Ряды: val - дата года Y -> val-прогноз шага Y-2015 (2016-2024; 2025 - шаг 10); тест - дата года Y -> тест шага Y-2016 (2017-2026), 2016 - val шага 1.
Д23.2-23.4 - подкоманды boost / harx (см. ниже).
Подтверждающий (val 2022-2024) и тест: прогнозы строятся для пула и сборки, метрики НЕ считаются (by_year - только шаги 2-6).
python3 scripts/analysis/d23_ec.py ebar | kappa | selftest | gibrid <boost> <h> | summary
"""
import sys, json, subprocess, time
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gibrid_har as GH
from gibrid_io import load_boost, MAIN_REGEN
from sklearn.linear_model import LinearRegression

R = MAIN_REGEN
OUT = R / 'D23'
LOCK = R / 'test_locked'
DEV = 6
K = ['date', 'ticker']
H1B, H1LEG = 'D12_h1_A_d812', 'top5'


def qlike(y, p):
    p = np.clip(p, 1e-10, None)
    return float(np.mean(y / p - np.log(y / p) - 1))


# ---------------------------------------------------------------- каркас
def load():
    a, harj = GH.load_pit(R / 'features_pit_v3.parquet')
    a = a.sort_values(['ticker', 'date']).reset_index(drop=True)
    g = a.groupby('ticker', sort=False)
    for h in (1, 5, 22):
        a[f'_fd{h}'] = g['date'].shift(-h)  # дата конца цели h (строки тикера, как цели матрицы)
    a['y1'] = a['rv_target_h1']
    a['y5'] = a['rv_target_mean_h5']
    a['y22'] = a['rv_target_mean_h22']
    return a, harj


def continuous(val, test):
    """val/test: (date,ticker,step,rv_pred) по шагам -> два непрерывных ряда по дате (см. шапку)."""
    v = val.copy(); v['year'] = pd.to_datetime(v.date).dt.year
    v = v[v.step == v.year - 2015]
    t = test.copy(); t['year'] = pd.to_datetime(t.date).dt.year
    t = t[t.step == t.year - 2016]
    t = pd.concat([v[v.year == 2016], t], ignore_index=True)
    for d in (v, t):
        assert not d.duplicated(K).any()
    return v[K + ['rv_pred']], t[K + ['rv_pred']]


def h1_series(seed=None):
    """Финалист h=1 (сырой): val шагов 1-9 + шаг 10 (val 2025), тест шагов 1-10. seed=None - нога top5 XGB."""
    import final2_build as FB
    if seed is None:
        v = load_boost(H1B, 1, R, H1LEG)[0]['val'][['date', 'ticker', 'step', 'rv_pred']]
        t = pd.read_parquet(LOCK / 'final2' / 'final_h1_test.parquet', columns=['date', 'ticker', 'step', 'xgb_raw', 'design'])
        assert t.design.iloc[0].startswith(H1LEG + '|V0_XGB')
        t = t.rename(columns={'xgb_raw': 'rv_pred'})[['date', 'ticker', 'step', 'rv_pred']]
        v10 = FB.leg10(H1B, 1, H1LEG)['xgb'][0][['date', 'ticker', 'step', 'rv_pred']]
        v = pd.concat([v, v10], ignore_index=True)
        return continuous(v, t)
    f = R / (f'{H1B}_xgb_h1_val.parquet' if seed == 42 else f'{H1B}_xgb_h1_seeds4344_val.parquet')
    v = pd.read_parquet(f)
    if seed != 42:
        v = v[v.seed == seed]
    return continuous(v[['date', 'ticker', 'step', 'rv_pred']], v.iloc[:0][['date', 'ticker', 'step', 'rv_pred']])


def errors_h1(a, ser):
    """e(s) на строках a (порядок ticker, date); ряд ser (date,ticker,rv_pred)."""
    p = a[K].merge(ser, on=K, how='left').rv_pred.values
    return np.log(a.y1.values + 1e-10) - np.log(np.clip(p, 1e-10, None))


def ebar_from_e(a, e, ms=(5, 10)):
    """ebar_m(t) = среднее e(t-1..t-m) по тикеру; рыночная - по средней за дату e (даты < t). NaN -> 0 (нет истории)."""
    out = pd.DataFrame(index=a.index)
    s = pd.Series(e, index=a.index)
    g = s.groupby(a.ticker.values, sort=False)
    sh = g.shift(1)  # e(t-1): известна в t (её цель - RV дня t)
    em = s.groupby(a.date.values).mean().sort_index()  # средняя за дату s ошибка (цели - RV следующих дней)
    for m in ms:
        out[f'ebar{m}'] = sh.groupby(a.ticker.values, sort=False).transform(lambda x: x.rolling(m, min_periods=1).mean()).fillna(0).values
        mk = em.shift(1).rolling(m, min_periods=1).mean().fillna(0)
        out[f'ebar{m}_mkt'] = a.date.map(mk).values
    return out


def har_series(a, harj, h):
    """HAR-J h (сырой, OLS в логах на годах < vy, как gibrid_har.har_forecasts) - шаги 1-10: val vy=2015+s, тест ty=2016+s."""
    V, T = [], []
    X = GH.clean(np.log(a[harj].values.clip(1e-10)))
    y = a[f'y{h}']
    for s in range(1, 11):
        vy, ty = 2015 + s, 2016 + s
        tr = ((a.year < vy) & y.notna()).values
        m = LinearRegression().fit(X[tr], np.log(y[tr].values + 1e-10))
        for yr, lst in ((vy, V), (ty, T)):
            sel = (a.year == yr).values
            lst.append(pd.DataFrame({'date': a.date.values[sel], 'ticker': a.ticker.values[sel], 'step': s,
                                     'rv_pred': np.clip(np.exp(m.predict(X[sel])), 1e-10, None)}))
    return continuous(pd.concat(V, ignore_index=True), pd.concat(T, ignore_index=True))


def e22_of(a, ser):
    p = a[K].merge(ser, on=K, how='left').rv_pred.values
    e = np.log(a.y22.values + 1e-10) - np.log(np.clip(p, 1e-10, None))
    return pd.Series(e, index=a.index).groupby(a.ticker.values, sort=False).shift(22).fillna(0).values  # e22(t-22): окно закрылось в t


def ebar_tables(a, harj):
    """-> {'val': DataFrame, 'test': DataFrame} на строках a: ebar5/10 (+_mkt) из финалиста h=1, ebar22_h22 (HAR-J h=22),
    ebar22_hyb (D18e_h22, XGB-слот, сырой)."""
    v1, t1 = h1_series()
    hv, ht = har_series(a, harj, 22)
    dv, dt = d18e_series()
    res = {}
    for kind, s1, s22, sd in (('val', v1, hv, dv), ('test', t1, ht, dt)):
        E = ebar_from_e(a, errors_h1(a, s1))
        E['ebar22_h22'] = e22_of(a, s22)
        E['ebar22_hyb'] = e22_of(a, sd)
        E.insert(0, 'ticker', a.ticker.values); E.insert(0, 'date', a.date.values)
        res[kind] = E
    return res


def d18e_series():
    """Лучший одиночный гибрид h=22 (ночное умолчание): D18e_h22, XGB-слот (V0_XGB/B-rob - представитель, нарушен 1 критерий), сырой."""
    v = pd.read_parquet(R / 'D18e_h22_xgb_h22_val.parquet')
    v10 = pd.read_parquet(R / 'D18' / 'step10' / 'D18e_h22_xgb_h22_s10_val.parquet')
    t = pd.read_parquet(LOCK / 'D18e_h22' / 'D18e_h22_xgb_h22_test.parquet')
    t10 = pd.read_parquet(LOCK / 'D18e_h22' / 'step10' / 'D18e_h22_xgb_h22_s10_test.parquet')
    c = ['date', 'ticker', 'step', 'rv_pred']
    return continuous(pd.concat([v[c], v10[c]]), pd.concat([t[c], t10[c]]))


def cmd_ebar():
    a, harj = load()
    E = ebar_tables(a, harj)
    cols = ['date', 'ticker', 'ebar5', 'ebar10', 'ebar5_mkt', 'ebar10_mkt', 'ebar22_h22', 'ebar22_hyb']
    v = E['val'][cols]
    v = v[pd.to_datetime(v.date).dt.year.between(2016, 2025)]  # val-ряд: 2016-2025 (2026 - только тест)
    v.to_parquet(R / 'D23_ebar.parquet', index=False)
    (LOCK / 'D23_ebar').mkdir(parents=True, exist_ok=True)
    t = E['test'][cols]
    t[pd.to_datetime(t.date).dt.year >= 2016].to_parquet(LOCK / 'D23_ebar' / 'D23_ebar_test.parquet', index=False)
    print(v.describe().T.to_string(), flush=True)


# ---------------------------------------------------------------- kappa
def base_series(name, a, harj):
    """База кандидата -> {slot: (val ряд, test ряд)} (сырые, непрерывные). slot xgb/lgb (gibrid), для сида - отдельные."""
    import final2_build as FB
    c = ['date', 'ticker', 'step', 'rv_pred']
    if name == 'h1':
        out = {}
        T = pd.read_parquet(LOCK / 'final2' / 'final_h1_test.parquet')
        for si, (m, col) in enumerate((('xgb', 'xgb_raw'), ('lgb', 'lgb_raw'))):
            v = load_boost(H1B, 1, R, H1LEG)[si]['val'][c]
            v10 = FB.leg10(H1B, 1, H1LEG)[m][0][c]
            t = T.drop(columns='rv_pred').rename(columns={col: 'rv_pred'})[c]
            out[m] = continuous(pd.concat([v, v10]), t)
        return out
    if name == 'h5':
        out = {}
        T = pd.read_parquet(LOCK / 'final2' / 'final_h5_test.parquet')
        for si, (m, col) in enumerate((('xgb', 'xgb_raw'), ('lgb', 'lgb_raw'))):
            v = load_boost('D12_h5', 5, R, 'wrap2')[si]['val'][c]
            v10 = FB.leg10('D12_h5', 5, 'wrap2')[m][0][c]
            t = T.drop(columns='rv_pred').rename(columns={col: 'rv_pred'})[c]
            out[m] = continuous(pd.concat([v, v10]), t)
        return out
    if name == 'harj22':
        s = har_series(a, harj, 22)
        return {'xgb': s, 'lgb': s}
    if name == 'd18e':
        out = {'xgb': d18e_series()}
        v = pd.read_parquet(R / 'D18e_h22_lgbm_h22_val.parquet')
        v10 = pd.read_parquet(R / 'D18' / 'step10' / 'D18e_h22_lgbm_h22_s10_val.parquet')
        t = pd.read_parquet(LOCK / 'D18e_h22' / 'D18e_h22_lgbm_h22_test.parquet')
        t10 = pd.read_parquet(LOCK / 'D18e_h22' / 'step10' / 'D18e_h22_lgbm_h22_s10_test.parquet')
        out['lgb'] = continuous(pd.concat([v[c], v10[c]]), pd.concat([t[c], t10[c]]))
        return out
    raise KeyError(name)


def seed_base(name, seed):
    """Сид-варианты базы (только val, шаги 1-9): h1 - одиночная точка XGB финалиста; h5 - одиночная точка XGB D12_h5."""
    boost, h = {'h1': (H1B, 1), 'h5': ('D12_h5', 5)}[name]
    f = R / (f'{boost}_xgb_h{h}_val.parquet' if seed == 42 else f'{boost}_xgb_h{h}_seeds4344_val.parquet')
    v = pd.read_parquet(f)
    if seed != 42:
        v = v[v.seed == seed]
    c = ['date', 'ticker', 'step', 'rv_pred']
    return continuous(v[c], v.iloc[:0][c])


def kappa_apply(a, h, base_v, base_t, Xv, Xt, cols):
    """-> (val ряд с коррекцией по шагам 1-10, тест ряд по шагам 1-10, таблица kappa). Xv/Xt - DataFrame регрессоров на строках a."""
    y = np.log(a[f'y{h}'].values + 1e-10)
    bv = a[K].merge(base_v, on=K, how='left').rv_pred.values
    bt = a[K].merge(base_t, on=K, how='left').rv_pred.values
    r = y - np.log(np.clip(bv, 1e-10, None))
    XV, XT = Xv[cols].values, Xt[cols].values
    ok = np.isfinite(r) & np.isfinite(XV).all(axis=1) & np.isfinite(bv)
    yr = a.year.values
    fd = a[f'_fd{h}'].values
    V, T, info = [], [], []
    for s in range(1, 11):
        vy, ty = 2015 + s, 2016 + s
        res = {}
        for tag, last, app_y, B, XX in (('val', vy - 1, vy, bv, XV), ('test', vy, ty, bt, XT)):
            tr = ok & (yr >= 2016) & (yr <= last) & (fd < np.datetime64(f'{last + 1}-01-01'))
            kap = np.linalg.lstsq(XX[tr], r[tr], rcond=None)[0] if tr.sum() > 50 else np.zeros(len(cols))
            sel = (yr == app_y) & np.isfinite(B)
            p = B[sel] * np.exp(np.nan_to_num(XX[sel]) @ kap)
            df = pd.DataFrame({'date': a.date.values[sel], 'ticker': a.ticker.values[sel], 'year': app_y, 'step': s,
                               'rv_actual': a[f'y{h}'].values[sel], 'rv_pred': np.clip(p, 1e-10, None)})
            (V if tag == 'val' else T).append(df)
            res[tag] = kap
        info.append(dict(step=s, **{f'kappa_val_{c}': float(k) for c, k in zip(cols, res['val'])},
                         **{f'kappa_test_{c}': float(k) for c, k in zip(cols, res['test'])}))
    return pd.concat(V, ignore_index=True), pd.concat(T, ignore_index=True), pd.DataFrame(info)


def plain(a, h, base_v, base_t):
    """База без коррекции в том же формате (шаги 1-10)."""
    Z = pd.DataFrame({'z': np.zeros(len(a))})
    v, t, _ = kappa_apply(a, h, base_v, base_t, Z, Z, ['z'])
    return v, t


def s2_tab(df):
    E = {s: np.log(g.rv_actual.values + 1e-10) - np.log(g.rv_pred.values) for s, g in df[df.rv_actual > 1e-12].groupby('step')}
    res = {}
    for code in ('sm', 'smrob'):
        for s in E:
            e = np.concatenate([E[j] for j in range(max(1, s - 2), s + 1) if j in E])
            res[(code, s)] = float((1.4826 * np.median(np.abs(e - np.median(e)))) ** 2) if code == 'smrob' else float(np.var(e))
    return res


def correct(v, code):
    """OOS: val шага t * exp(s2_{t-1}/2) (шаг 1: F=1); s2 - только по шагам <= 6 нужны для оценки 2-6."""
    S = s2_tab(v[v.step <= 6])
    v = v.copy()
    v['rv_pred'] *= v.step.map(lambda s: np.exp(S[(code, s - 1)] / 2) if 1 < s <= 6 else 1.0)
    return v


def by_year(v):
    d = v[(v.step >= 2) & (v.step <= DEV) & (v.rv_actual > 1e-12)].dropna(subset=['rv_pred', 'rv_actual'])
    q = {str(int(s) + 2015): qlike(g.rv_actual.values, g.rv_pred.values) for s, g in d.groupby('step')}
    q['2017-2020'] = float(np.mean([q[str(y)] for y in range(2017, 2021)]))
    return q


def best_s2(v):
    """Лучшая из sm/smrob по 2017-2020 (как Д18) -> (код, QLIKE по годам)."""
    c = {code: by_year(correct(v, code)) for code in ('sm', 'smrob')}
    k = min(c, key=lambda x: c[x]['2017-2020'])
    return k, c[k]


VARS = {  # имя -> регрессоры x(t)
    'tk5': ['ebar5'], 'tk10': ['ebar10'], 'mk5': ['ebar5_mkt'], 'mk10': ['ebar10_mkt'],
    'tm5': ['ebar5', 'ebar5_mkt'], 'tm10': ['ebar10', 'ebar10_mkt'],
}
VARS22 = {'e22': ['E22'], 'e22tm5': ['E22', 'ebar5', 'ebar5_mkt'], 'e22tk10': ['E22', 'ebar10']}
BASES = {'h1': 1, 'h5': 5, 'harj22': 22, 'd18e': 22}


def regressors(a, harj, E, base, slot_series):
    """Регрессоры варианта: ebar из финалиста h=1 (E[val/test]) + для h=22 E22 - ошибка собственного прогноза базы h=22 22 дня назад."""
    Xv, Xt = E['val'].copy(), E['test'].copy()
    if BASES[base] == 22:
        Xv['E22'] = e22_of(a, slot_series[0]); Xt['E22'] = e22_of(a, slot_series[1])
    return Xv, Xt


def cmd_kappa():
    a, harj = load()
    E = ebar_tables(a, harj)
    rows, store = [], {}
    for base, h in BASES.items():
        B = base_series(base, a, harj)
        for slot, (bv, bt) in B.items():
            if base in ('harj22',) and slot == 'lgb':
                continue
            pv, pt = plain(a, h, bv, bt)
            k, q = best_s2(pv)
            rows.append(dict(base=base, h=h, slot=slot, var='base', s2=k, **q))
            store[(base, slot, 'base')] = (pv, pt, None)
            Xv, Xt = regressors(a, harj, E, base, (bv, bt))
            vs = dict(VARS, **(VARS22 if h == 22 else {}))
            for vn, cols in vs.items():
                v, t, info = kappa_apply(a, h, bv, bt, Xv, Xt, cols)
                k, q = best_s2(v)
                rows.append(dict(base=base, h=h, slot=slot, var=vn, s2=k, kappa=json.dumps({c: [round(x, 3) for x in info[f'kappa_val_{c}'].values[1:6]] for c in cols}), **q))
                store[(base, slot, vn)] = (v, t, info)
                print(base, slot, vn, k, round(q['2017-2020'], 5), round(q['2021'], 5), flush=True)
    Q = pd.DataFrame(rows)
    Q.to_csv(OUT / 'd23_1_qlike.csv', index=False)
    pd.to_pickle(store, OUT / 'd23_1_store.pkl')
    # сиды: база - одиночная точка XGB seed 42/43/44 (h1, h5); регрессоры - те же (ebar финалиста h=1)
    srows = []
    for base in ('h1', 'h5'):
        h = BASES[base]
        for seed in (42, 43, 44):
            bv, _ = seed_base(base, seed)
            pv, _ = plain(a, h, bv, bv.iloc[:0])
            for vn in ['base'] + list(VARS):
                if vn == 'base':
                    v = pv
                else:
                    v = kappa_apply(a, h, bv, bv.iloc[:0], E['val'], E['test'], VARS[vn])[0]
                for code in ('sm', 'smrob'):
                    srows.append(dict(base=base, seed=seed, var=vn, s2=code, **by_year(correct(v, code))))
    S = pd.DataFrame(srows)
    S.to_csv(OUT / 'd23_1_seeds.csv', index=False)
    pd.set_option('display.width', 250)
    print(Q.drop(columns=['kappa']).to_string(), flush=True)


def selftest():
    """«Будущее не влияет» для ebar и kappa:
    (1) ebar(t) при возмущении RV всех дней > c (и прогнозов для строк >= c) не меняется на датах <= c;
        контроль - ebar без сдвига (e(t) вместо e(t-1)) меняется на дате c (цель строки c - RV дня c+1);
    (2) e22: возмущение RV дней > c не меняет ebar22 на датах <= c;
    (3) kappa шага t не меняется при возмущении всего, начиная с 1 янв. val-года шага (эмбарго); контроль без эмбарго - меняется."""
    a, harj = load()
    v1, _ = h1_series()
    rs = np.random.default_rng(1)
    out = []
    for c in ('2017-06-15', '2019-03-11', '2020-12-30', '2021-09-01'):
        cd = pd.Timestamp(c)
        b = a.copy()
        fut = (b.date > cd).values
        b.loc[fut, 'rv'] *= rs.uniform(0.2, 5, fut.sum())
        b['y1'] = b.groupby('ticker', sort=False)['rv'].shift(-1)  # цель h=1 = RV следующего дня тикера (как в матрице)
        g = b.groupby('ticker', sort=False)
        b['y22'] = pd.concat([g['rv'].shift(-j) for j in range(1, 23)], axis=1).mean(axis=1, skipna=False)
        s1 = v1.copy(); s1.loc[s1.date >= cd, 'rv_pred'] *= rs.uniform(0.2, 5, (s1.date >= cd).sum())
        e0, e1 = errors_h1(a, v1), errors_h1(b, s1)
        E0, E1 = ebar_from_e(a, e0), ebar_from_e(b, e1)
        past = (a.date <= cd).values
        d = float(np.abs(E0.values[past] - E1.values[past]).max())
        at_c = (a.date == cd).values
        ctl = lambda e: pd.Series(e, index=a.index).groupby(a.ticker.values, sort=False).transform(lambda x: x.rolling(5, min_periods=1).mean()).values
        dc = float(np.nanmax(np.abs(ctl(e0)[at_c] - ctl(e1)[at_c])))
        hv, _ = har_series(a, harj, 22)
        d22 = float(np.abs(e22_of(a, hv)[past] - e22_of(b, hv)[past]).max())
        ctl22 = lambda bb: pd.Series(np.log(bb.y22.values + 1e-10), index=a.index).groupby(a.ticker.values, sort=False).shift(21).values
        dc22 = float(np.nanmax(np.abs(ctl22(a)[past] - ctl22(b)[past])))
        out.append(dict(test='ebar', cut=c, max_d_past=d, max_d_ctl_bez_sdviga=dc, max_d22_past=d22, max_d22_ctl_sdvig21=dc22,
                        ok=d == 0 and d22 < 1e-12 and dc > 1e-6 and dc22 > 1e-6))
    # (3) kappa: всё с 1 янв vy - возмущено -> kappa шага не меняется
    E = ebar_tables(a, harj)
    B = base_series('h1', a, harj)['xgb']
    for s in range(2, 7):
        vy = 2015 + s
        b = a.copy()
        fut = (b.date >= pd.Timestamp(f'{vy}-01-01')).values
        b.loc[fut, 'rv'] *= rs.uniform(0.2, 5, fut.sum())
        b['y1'] = b.groupby('ticker', sort=False)['rv'].shift(-1)  # цель строки 30.12 (RV первого дня vy) тоже возмущена
        Ev = E['val'].copy(); Ev.loc[fut, ['ebar5', 'ebar5_mkt']] *= 3
        k0 = kappa_apply(a, 1, B[0], B[1], E['val'], E['test'], ['ebar5', 'ebar5_mkt'])[2]
        k1 = kappa_apply(b, 1, B[0], B[1], Ev, E['test'], ['ebar5', 'ebar5_mkt'])[2]
        d = float(np.abs(k0.loc[k0.step == s, ['kappa_val_ebar5', 'kappa_val_ebar5_mkt']].values - k1.loc[k1.step == s, ['kappa_val_ebar5', 'kappa_val_ebar5_mkt']].values).max())
        b2 = b.copy(); b2['_fd1'] = pd.Timestamp('1990-01-01')  # контроль: эмбарго снято
        a2 = a.copy(); a2['_fd1'] = pd.Timestamp('1990-01-01')
        k0c = kappa_apply(a2, 1, B[0], B[1], E['val'], E['test'], ['ebar5', 'ebar5_mkt'])[2]
        k1c = kappa_apply(b2, 1, B[0], B[1], Ev, E['test'], ['ebar5', 'ebar5_mkt'])[2]
        dc = float(np.abs(k0c.loc[k0c.step == s, ['kappa_val_ebar5']].values - k1c.loc[k1c.step == s, ['kappa_val_ebar5']].values).max())
        out.append(dict(test='kappa', cut=f'{vy}-01-01 (шаг {s})', max_d_past=d, max_d_ctl_bez_sdviga=dc, ok=d == 0 and dc > 1e-9))
    st = pd.DataFrame(out)
    st.to_csv(OUT / 'selftest_future.csv', index=False)
    print(st.to_string(), flush=True)
    return st



# ---------------------------------------------------------------- Д23.3 / Д23.2 / Д23.4: бустинг h=1 (Д8.12) с весами, окнами, признаками
BCANDS = {
    'hl2': {'harf': True, 'hl': 2}, 'hl4': {'harf': True, 'hl': 4},  # Д23.3: веса 0,5^(возраст/полураспад), возраст от последней даты окна
    'r3': {'harf': True, 'win': 3}, 'r5': {'harf': True, 'win': 5},  # Д23.3: скользящее окно 3/5 лет перед val-годом (HAR-J признака - расширяющееся)
    'fx': {'harf': True, 'xf': 'all'},  # Д23.2 + Д23.4: корреляционные и ликвидностные признаки (см. xfeat)
}


def _install_win():
    """Обёртка l2_lib.step_data: cfg['win'] = k - обучающие строки только годов >= vy-k (ln HAR-J-признак считается на полном окне)."""
    import l2_lib as L
    if getattr(L, '_d23_win', False):
        return
    sd0 = L.step_data

    def sd(A, cols, h, step, model, cfg, base_tag, **kw):
        D = sd0(A, cols, h, step, model, cfg, base_tag, **kw)
        k = cfg.get('win')
        if k:
            keep = (D['parts']['tr'].year >= D['vy'] - int(k)).values
            D['parts']['tr'] = D['parts']['tr'][keep]
            D['X']['tr'] = D['X']['tr'][keep]
            D['y']['tr'] = D['y']['tr'][keep]
            D['ytr_ln'] = D['ytr_ln'][keep]
            D['off']['tr'] = D['off']['tr'][keep]
            if D['w'] is not None:
                D['w'] = D['w'][keep]
        return D
    L.step_data = sd
    L._d23_win = True


def xfeat_cols():
    p = R / 'D23' / 'd23_xfeat.parquet'
    if not p.exists():
        return [], None
    X = pd.read_parquet(p)
    return [c for c in X.columns if c not in ('date', 'ticker')], X


def boost_run(cands, workers=3, legs=('single', 'top5', 'seeds3'), check=False):
    """Как d12_h1.py (точки ЛГК B v2b, ноги single/top5/seeds3, l2_lib.fit), кандидаты BCANDS; выход - формат gibrid_io:
    data/regen/D23_h1_<канд>_{xgb,lgbm}_h1{,_top5,_seeds4344}_val.parquet, тест - test_locked/D23_h1_<канд>/ (без метрик)."""
    import multiprocessing as mp
    import d12_h1 as D12
    import l2_lib as L
    _install_win()
    A, FC = L.load_base(R / 'features_pit_v3.parquet')
    xc, X = xfeat_cols()
    if X is not None and any(BCANDS[c].get('xf') for c in cands):
        A = A.merge(X, on=['date', 'ticker'], how='left', validate='one_to_one')
    D12.G.update(A=A, grid=D12.grid(), cols={})
    for c in cands + ['A_d812']:
        D12.CANDS[c] = BCANDS[c] if c in BCANDS else D12.CANDS[c]
        _, cols, _, _ = L.build_features(A, FC, 1, D12.CANDS[c])
        if D12.CANDS[c].get('xf'):
            cols = cols + [x for x in xc if x not in cols]
        D12.G['cols'][c] = cols
    pl = D12.plan(list(legs))
    if check:  # сверка: A_d812, шаг 2, xgb, ранг 1, seed 42 -> сохранённый D12_h1_A_d812_xgb_h1_val
        t = [x for x in pl if x[0] == 'xgb' and x[1] == 2 and x[4] == 1 and x[3] == 42][0]
        r = D12.job(('A_d812',) + t)
        ref = pd.read_parquet(R / 'D12_h1_A_d812_xgb_h1_val.parquet')
        m = r[7].merge(ref[ref.step == 2], on=['date', 'ticker'], suffixes=('', '_ref'))
        print('сверка шаг 2 xgb:', len(m), float(np.abs(np.log(m.rv_pred) - np.log(m.rv_pred_ref)).max()), flush=True)
        return
    tasks = [(c,) + t for c in cands for t in pl]
    t0 = time.time()
    print(f'{time.strftime("%T")} кандидаты {cands}, моделей {len(tasks)}, воркеров {workers}', flush=True)
    with mp.get_context('fork').Pool(workers) as P:
        res = P.map(D12.job, tasks, chunksize=1)
    print(f'{time.strftime("%T")} обучено за {time.time() - t0:.0f} с', flush=True)
    for c in cands:
        boost = f'D23_h1_{c}'
        td = LOCK / boost
        td.mkdir(parents=True, exist_ok=True)
        for m in ('xgb', 'lgbm'):
            rr = [r for r in res if r[0] == c and r[1] == m]
            sel = lambda f: (pd.concat([x[7] for x in rr if f(x)], ignore_index=True), pd.concat([x[8] for x in rr if f(x)], ignore_index=True))
            files = {'': sel(lambda x: x[5] == 1 and x[4] == 42)}
            if 'top5' in legs:
                files['_top5'] = sel(lambda x: x[2] <= 6 and x[4] == 42)
            if 'seeds3' in legs:
                files['_seeds4344'] = sel(lambda x: x[4] in (43, 44))
            for sfx, (v, t) in files.items():
                v.to_parquet(R / f'{boost}_{m}_h1{sfx}_val.parquet', index=False)
                t.to_parquet(td / f'{boost}_{m}_h1{sfx}_test.parquet', index=False)
        print(c, 'записан', flush=True)


def leg_eval(boost, h=1, legs=('single', 'top5', 'seeds3')):
    """OOS QLIKE по годам (шаги 2-6) ног кандидата, s2 sm/smrob (лучшая по 2017-2020)."""
    rows = []
    for leg in legs:
        L2 = load_boost(boost, h, R, leg)
        for si, m in enumerate(('xgb', 'lgb')):
            v = L2[si]['val']
            v = v[v.step <= 6]
            k, q = best_s2(v)
            rows.append(dict(boost=boost, leg=leg, model=m, s2=k, **q))
    return pd.DataFrame(rows)


def seed_eval(boost, h=1):
    """Одиночная точка XGB/LGBM seed 42/43/44 (шаги 2-6), s2 sm/smrob - лучшая."""
    rows = []
    for m in ('xgb', 'lgbm'):
        for seed in (42, 43, 44):
            f = R / (f'{boost}_{m}_h{h}_val.parquet' if seed == 42 else f'{boost}_{m}_h{h}_seeds4344_val.parquet')
            v = pd.read_parquet(f)
            if seed != 42:
                v = v[v.seed == seed]
            v = v[v.step <= 6]
            k, q = best_s2(v)
            rows.append(dict(boost=boost, model=m, seed=seed, s2=k, **q))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- Д23.2 / Д23.4: признаки из баров (крейт moex_iss, бинарник d23_corr)
LEADERS = ('SBER', 'LKOH')  # ночное умолчание: GAZP нет в 17 тикерах и в барах data/iss_check (проверено) - только SBER и LKOH
XF_BOOST = ['corr_d', 'corr_w', 'corr_m', 'disp_lnrv', 'lead_SBER', 'lead_LKOH', 'amihud', 'lasthour']
XF_HARX = ['corr_d', 'corr_w', 'corr_m', 'disp_lnrv', 'ln_lead_SBER', 'ln_lead_LKOH', 'ln_amihud', 'lasthour']


def cmd_xfeat():
    """d23_corr_raw.parquet (крейт) + RV лидеров дня t (столбец rv матрицы, срез 18:50) -> D23/d23_xfeat.parquet на строках матрицы.
    Детектор: признаки крейта со срезом баров (leak_corr_a/b: бары позже 15.06.2018 / 30.12.2022 отброшены) совпадают на датах <= среза."""
    a = pd.read_parquet(R / 'features_pit_v3.parquet', columns=['date', 'ticker', 'rv', 'volume'])
    c = pd.read_parquet(OUT / 'd23_corr_raw.parquet'); c['date'] = pd.to_datetime(c.date)
    rows = []
    for f, cut in (('leak_corr_a.parquet', '2018-06-15'), ('leak_corr_b.parquet', '2022-12-30')):
        x = pd.read_parquet(OUT / f); x['date'] = pd.to_datetime(x.date)
        m = c[c.date <= cut].merge(x, on=['date', 'ticker'], suffixes=('', '_cut'), how='outer', indicator=True)
        d = {k: float(np.nanmax(np.abs(m[k] - m[k + '_cut']))) for k in ('corr_d', 'corr_w', 'corr_m', 'disp_lnrv', 'amihud', 'lasthour')}
        rows.append(dict(cut=cut, rows=len(m), only_one_side=int((m._merge != 'both').sum()), **d, ok=all(v == 0 for v in d.values()) and (m._merge == 'both').all()))
    st = pd.DataFrame(rows); st.to_csv(OUT / 'selftest_xfeat.csv', index=False); print(st.to_string(), flush=True)
    X = a[['date', 'ticker']].merge(c, on=['date', 'ticker'], how='left', validate='one_to_one')
    lead = a[a.ticker.isin(LEADERS)].pivot(index='date', columns='ticker', values='rv')
    for t in LEADERS:
        X[f'lead_{t}'] = X.date.map(lead[t]).values  # RV лидера дня t: для прогноза t+1 - лаг 1
        X[f'ln_lead_{t}'] = np.log(X[f'lead_{t}'].clip(lower=1e-12))
    X['ln_amihud'] = np.log(X.amihud.clip(lower=1e-6))
    X.to_parquet(OUT / 'd23_xfeat.parquet', index=False)
    print(X.drop(columns=['date', 'ticker']).describe().T.round(4).to_string(), flush=True)
    print('Д23.4: в матрице v3 есть volume; нет числа сделок (в барах ISS тоже нет), Amihud, доли RV последнего часа (rv_evening - треть сессии)', flush=True)


def harx_ext(hs=(1, 5, 22)):
    """HAR-X (harx.py, без правки) + слагаемые Д23.2/23.4 в той же эластичной сети: свой реестр = столбцы XF_HARX в матрице A
    (build_terms берёт существующий столбец как есть). Сравнение - HAR-X по умолчанию (default_terms) на том же A.
    Выход: data/regen/D23_HARXc_h{h}_val.parquet (шаги 1-10, rv_pred_{ord,rob,sm,smrob}), тест - test_locked/D23_HARXc/; коэффициенты - D23/."""
    import harx as HX
    A, harj = HX.load()
    X = pd.read_parquet(OUT / 'd23_xfeat.parquet')
    A = A.merge(X[['date', 'ticker'] + XF_HARX], on=['date', 'ticker'], how='left', validate='one_to_one')
    lt = HX.leak_test(A, XF_HARX)
    if HX._SIG.get('v', 0) is None:  # leak_test кладёт None в кэш σ̂, если он был пуст - вернуть пустой
        HX._SIG.pop('v')
    lt.to_csv(OUT / 'leak_test_harx_terms.csv', index=False)
    print('leak_test слагаемых Д23 (столбцы как есть):', int(lt.mismatch.sum()), flush=True)
    (LOCK / 'D23_HARXc').mkdir(parents=True, exist_ok=True)
    rows = []
    for h in hs:
        base = HX.default_terms(h)
        out = {}
        for tag, terms in (('HARX', base), ('HARXc', base + XF_HARX)):
            log = []
            Xall = HX.build_terms(A, terms)
            v, t = HX.harx_forecasts(h, A, harj, terms, Xall, log)
            v, t = HX.add_bias(v, t)
            out[tag] = v
            pd.DataFrame(log).to_csv(OUT / f'{tag}_h{h}_coefs.csv', index=False)
            if tag == 'HARXc':
                v.to_parquet(R / f'D23_HARXc_h{h}_val.parquet', index=False)
                t.to_parquet(LOCK / 'D23_HARXc' / f'D23_HARXc_h{h}_test.parquet', index=False)
            for code in ('sm', 'smrob'):
                q = by_year(v[v.step <= 6].assign(rv_pred=v[f'rv_pred_{code}']))
                rows.append(dict(h=h, model=tag, s2=code, **q))
        print(f'HAR-X h={h} готов', flush=True)
    Q = pd.DataFrame(rows); Q.to_csv(OUT / 'd23_2_harx.csv', index=False)
    print(Q.to_string(), flush=True)


# ---------------------------------------------------------------- Д23.1: кандидаты для пула (формат gibrid_v17 / final3)
def save_cand(boost, h, slots, ref_val, ref_test):
    """slots: {'xgb': (val, test), 'lgbm': (val, test)} - ряды kappa_apply (шаги 1-10). Строки val/тест шагов 1-9 - как у файлов базы
    (ref_val/ref_test: (date, ticker, step)); шаг 10 - D23/step10 и test_locked/<boost>/step10 (формат final2_step10, seed=42)."""
    cols = ['date', 'ticker', 'year', 'step', 'rv_actual', 'rv_pred']
    td = LOCK / boost
    (td / 'step10').mkdir(parents=True, exist_ok=True)
    k = ['date', 'ticker', 'step']
    for m, (v, t) in slots.items():
        vv = ref_val[k].merge(v[cols], on=k, how='left', validate='one_to_one')
        tt = ref_test[k].merge(t[cols], on=k, how='left', validate='one_to_one')
        assert vv.rv_pred.notna().all() and tt.rv_pred.notna().all(), (boost, m)
        vv[cols].to_parquet(R / f'{boost}_{m}_h{h}_val.parquet', index=False)
        tt[cols].to_parquet(td / f'{boost}_{m}_h{h}_test.parquet', index=False)
        v[v.step == 10][cols].assign(seed=42).to_parquet(OUT / 'step10' / f'{boost}_{m}_h{h}_s10_val.parquet', index=False)
        t[t.step == 10][cols].assign(seed=42).to_parquet(td / 'step10' / f'{boost}_{m}_h{h}_s10_test.parquet', index=False)
    print(boost, 'записан', flush=True)


def cmd_cand(spec):
    """spec: '<boost>:<база>:<вариант>' например D23_1_h22e:d18e:mk10 - оба слота базы с kappa варианта."""
    boost, base, var = spec.split(':')
    a, harj = load()
    E = ebar_tables(a, harj)
    h = BASES[base]
    B = base_series(base, a, harj)
    cols = dict(VARS, **VARS22)[var]
    slots, info = {}, []
    for slot, m in (('xgb', 'xgb'), ('lgb', 'lgbm')):
        Xv, Xt = regressors(a, harj, E, base, B[slot])
        v, t, inf = kappa_apply(a, h, B[slot][0], B[slot][1], Xv, Xt, cols)
        slots[m] = (v, t)
        info.append(inf.assign(slot=m))
    pd.concat(info).to_csv(OUT / f'kappa_{boost}.csv', index=False)
    ref = {'d18e': ('D18e_h22_xgb_h22_val.parquet', 'D18e_h22/D18e_h22_xgb_h22_test.parquet'),
           'harj22': ('D18e_h22_xgb_h22_val.parquet', 'D18e_h22/D18e_h22_xgb_h22_test.parquet'),
           'h1': (f'{H1B}_xgb_h1_val.parquet', None), 'h5': ('D12_h5_xgb_h5_val.parquet', None)}[base]
    rv = pd.read_parquet(R / ref[0])
    rt = pd.read_parquet(LOCK / ref[1]) if ref[1] else pd.read_parquet(LOCK / 'final2' / f'final_h{h}_test.parquet').query('step <= 9')
    save_cand(boost, h, slots, rv, rt)


def seeds22(var='mk10'):
    """3 seed для h=22 (D18e + kappa): D18_1e сида s = (5*hyb5_s + 8.5*HAR[6..22] + 8.5*D16.5_s)/22 (как Д18, одиночные точки XGB seed s),
    регрессоры ebar - из одиночной точки XGB финалиста h=1 того же сида. Только val, шаги 2-6."""
    a, harj = load()
    H6 = pd.read_parquet(R / 'D18_har_sub_h22_6_22_val.parquet')[K + ['step', 'rv_pred']].rename(columns={'rv_pred': 'har'})
    rows = []
    for seed in (42, 43, 44):
        def one(boost, h):
            f = R / (f'{boost}_xgb_h{h}_val.parquet' if seed == 42 else f'{boost}_xgb_h{h}_seeds4344_val.parquet')
            v = pd.read_parquet(f)
            return (v if seed == 42 else v[v.seed == seed])[K + ['step', 'rv_pred']]
        x = H6.merge(one('D12_h5', 5).rename(columns={'rv_pred': 'h5'}), on=K + ['step']).merge(
            one('D16_5_h22', 22).rename(columns={'rv_pred': 'b'}), on=K + ['step'])
        x['rv_pred'] = (5 * x.h5 + 8.5 * x.har + 8.5 * x.b) / 22
        bv, _ = continuous(x[K + ['step', 'rv_pred']], x.iloc[:0][K + ['step', 'rv_pred']])
        s1, _ = h1_series(seed)
        Ev = ebar_from_e(a, errors_h1(a, s1))
        pv, _ = plain(a, 22, bv, bv.iloc[:0])
        cv = kappa_apply(a, 22, bv, bv.iloc[:0], Ev, Ev, dict(VARS, **VARS22).get(var, VARS.get(var)))[0]
        for nm, v in (('base', pv), (var, cv)):
            for code in ('sm', 'smrob'):
                rows.append(dict(seed=seed, var=nm, s2=code, **by_year(correct(v, code))))
    S = pd.DataFrame(rows); S.to_csv(OUT / f'd23_1_seeds_h22_{var}.csv', index=False)
    print(S.to_string(), flush=True)


def gibrid_summary():
    """Представители кандидатов по правилу третьего круга (gibrid_v17 --dev-eval oos): ключ - число нарушенных c1-c4 (шаги 2-6), затем
    средний OOS QLIKE 2017-2020 (шаги 2-5); QLIKE по годам 2017-2021 из designs_val (шаги 7-9 гибридом скрыты). Как Д18."""
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
    pd.set_option('display.width', 250)
    print(rep.to_string(), flush=True)
    return A


def run_gibrid(boost, h, legs):
    """gibrid_v17 (OOS, эталон - та же s2) по кандидату; подтверждающий не открывается."""
    F = R / 'features_pit_v3.parquet'
    cmd = ['python3', '-W', 'ignore', str(Path(__file__).resolve().parent / 'gibrid_v17.py'), '--boost', boost, '--h', str(h), '--features', str(F),
           '--legs', *legs, '--out', str(OUT / 'gibrid' / boost), '--devlog', str(OUT / 'gibrid' / f'devlog_{boost}.md'),
           '--dev-eval', 'oos', '--etalon', 'same']
    with open(OUT / 'logs' / f'gibrid_{boost}.log', 'w') as f:
        subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, check=True)


# ---------------------------------------------------------------- Д23.3 h=5: D12_h5 (V9 + Д8.12 + Д7 + Huber, optuna-30) с весами/окнами
G5 = {}
D12H5_CFG = {'v9': True, 'harf': True, 'huber': True}


BCANDS5 = {'hl2': {'hl': 2}, 'hl4': {'hl': 4}, 'r3': {'win': 3}, 'r5': {'win': 5}}


def boost5_run(vars_, workers=3, check=False):
    """Точки D12_h5 (D12_h5_trials.csv: топ-5 неснятых испытаний шага 1-6; шаги 7-9 - точка шага 6; seed 43/44 - выбранная точка),
    ноги single/top5/seeds3 -> data/regen/D23_h5_<вар>_{xgb,lgbm}_h5{,_top5,_seeds4344}_val.parquet, тест - test_locked/D23_h5_<вар>/."""
    import multiprocessing as mp
    import l2_lib as L
    from tune_L2 import SPACEB
    _install_win()
    A, FC = L.load_base(None, ('d7',), 5)
    A2, cols, _, _ = L.build_features(A, FC, 5, D12H5_CFG)
    T = pd.read_csv(R / 'D12_h5_trials.csv')
    T = T[~T.pruned.astype(bool)]
    S = pd.read_csv(R / 'D12_h5_selected_h5.csv')
    P = {}  # trial-номера уникальны только внутри (модель, шаг): ключ - (модель, шаг, trial)
    for r in T.itertuples():
        P[(r.model, int(r.step), int(r.trial))] = {k: getattr(r, k) for k, *_ in SPACEB[r.model]}
    sel6 = {m: int(S[(S.model == m) & (S.step == 6)].trial.iloc[0]) for m in ('xgb', 'lgbm')}
    G5.update(A=A2, cols=cols)
    plan = []
    for m in ('xgb', 'lgbm'):
        for s in range(1, 10):
            if s <= 6:
                top = T[(T.model == m) & (T.step == s)].sort_values('qlike_val').trial.astype(int).tolist()[:5]
                selt = int(S[(S.model == m) & (S.step == s)].trial.iloc[0])
                assert top[0] == selt, (m, s, top[0], selt)
                plan += [(m, s, t, 42, r + 1) for r, t in enumerate(top)]
                plan += [(m, s, selt, sd, 1) for sd in (43, 44)]
            else:
                plan += [(m, s, sel6[m], sd, 1) for sd in (42, 43, 44)]
    # параметры по (модель, шаг, trial): шаги 7-9 - точка шага 6
    def par(m, s, t):
        return P[(m, min(s, 6), t)]
    G5['par'] = par
    if check:
        t2 = int(S[(S.model == 'xgb') & (S.step == 2)].trial.iloc[0])
        r = _job5_p(('base', 'xgb', 2, t2, 42, 1))
        ref = pd.read_parquet(R / 'D12_h5_xgb_h5_val.parquet')
        m_ = r[7].merge(ref[ref.step == 2], on=['date', 'ticker'], suffixes=('', '_ref'))
        print('сверка D12_h5 шаг 2 xgb:', len(m_), float(np.abs(np.log(m_.rv_pred) - np.log(m_.rv_pred_ref)).max()), flush=True)
        return
    tasks = [(v,) + t for v in vars_ for t in plan]
    t0 = time.time()
    print(f'{time.strftime("%T")} h=5 варианты {vars_}, моделей {len(tasks)}, воркеров {workers}', flush=True)
    with mp.get_context('fork').Pool(workers) as Pl:
        res = Pl.map(_job5_p, tasks, chunksize=1)
    print(f'{time.strftime("%T")} обучено за {time.time() - t0:.0f} с', flush=True)
    for v in vars_:
        boost = f'D23_h5_{v}'
        td = LOCK / boost
        td.mkdir(parents=True, exist_ok=True)
        for m in ('xgb', 'lgbm'):
            rr = [x for x in res if x[0] == v and x[1] == m]
            sel = lambda f: (pd.concat([x[7] for x in rr if f(x)], ignore_index=True), pd.concat([x[8] for x in rr if f(x)], ignore_index=True))
            files = {'': sel(lambda x: x[5] == 1 and x[4] == 42), '_top5': sel(lambda x: x[2] <= 6 and x[4] == 42),
                     '_seeds4344': sel(lambda x: x[4] in (43, 44))}
            for sfx, (vv, tt) in files.items():
                vv.sort_values(['step', 'date', 'ticker']).to_parquet(R / f'{boost}_{m}_h5{sfx}_val.parquet', index=False)
                tt.sort_values(['step', 'date', 'ticker']).to_parquet(td / f'{boost}_{m}_h5{sfx}_test.parquet', index=False)
        print(boost, 'записан', flush=True)


def _job5_p(t):
    """Как _job5, параметры - G5['par'](модель, шаг, trial); вариант 'base' - без рычага (сверка)."""
    import l2_lib as L
    var, model, step, trial, seed, rank = t
    cfg = dict(D12H5_CFG, **BCANDS5.get(var, {}))
    p = G5['par'](model, step, trial)
    D = L.step_data(G5['A'], G5['cols'], 5, step, model, cfg, 'Bv2b')
    ref = L.make_ref(model, G5['A'], G5['cols'], 5, cfg, 'Bv2b') if step > 1 else None
    pv, pt, bi, _, _ = L.fit(model, p, D, cfg, seed=seed, ref=ref)
    out = []
    for k, pr in (('va', pv), ('te', pt)):
        df, yr = (D['parts']['va'], D['vy']) if k == 'va' else (D['parts']['te'], D['ty'])
        out.append(pd.DataFrame({'date': df['date'].values, 'ticker': df['ticker'].values, 'year': yr, 'step': step, 'trial': trial,
                                 'seed': seed, 'rank': rank, 'rv_actual': df[L.tcol(5)].values, 'rv_pred': np.clip(np.exp(pr), 1e-10, None)}))
    return var, model, step, trial, seed, rank, bi, out[0], out[1]


# ---------------------------------------------------------------- Д23.3: ансамбль по окнам и сводка рычагов бустинга
def make_wens(h, members, out):
    """Среднее RV прогнозов кандидатов-членов (одни и те же точки/seed/шаги) -> кандидат out (val и тест, все ноги)."""
    keys = ['date', 'ticker', 'year', 'step', 'seed', 'rank', 'rv_actual']
    td = LOCK / out
    td.mkdir(parents=True, exist_ok=True)
    for m in ('xgb', 'lgbm'):
        for sfx in ('', '_top5', '_seeds4344'):
            for kind in ('val', 'test'):
                fr = []
                for b in members:
                    p = (R / f'{b}_{m}_h{h}{sfx}_val.parquet') if kind == 'val' else (LOCK / b / f'{b}_{m}_h{h}{sfx}_test.parquet')
                    d = pd.read_parquet(p)
                    pk = 'point' if 'point' in d else 'trial'
                    fr.append(d.rename(columns={pk: 'pt'}))
                k = [c for c in keys if c in fr[0]] + ['pt']
                M = fr[0][k + ['rv_pred']]
                for i, d in enumerate(fr[1:], 1):
                    M = M.merge(d[k + ['rv_pred']], on=k, suffixes=('', f'_{i}'), validate='one_to_one')
                assert len(M) == len(fr[0]), (m, sfx, kind, len(M), len(fr[0]))
                M['rv_pred'] = M[[c for c in M.columns if c.startswith('rv_pred')]].mean(axis=1)
                M = M[k + ['rv_pred']].rename(columns={'pt': 'point'})
                if kind == 'val':
                    M.to_parquet(R / f'{out}_{m}_h{h}{sfx}_val.parquet', index=False)
                else:
                    M.to_parquet(td / f'{out}_{m}_h{h}{sfx}_test.parquet', index=False)
    print(out, 'записан', flush=True)


def eval_boost(h, boosts, base):
    """Ноги и сиды кандидатов против базы (одни и те же точки): 2017-2020, 2021, s2 - лучшая из sm/smrob."""
    rows = [leg_eval(b, h).assign(kind='leg') for b in [base] + boosts]
    srows = [seed_eval(b, h).assign(kind='seed') for b in [base] + boosts]
    L_ = pd.concat(rows, ignore_index=True)
    S_ = pd.concat(srows, ignore_index=True)
    L_.to_csv(OUT / f'd23_boost_legs_h{h}.csv', index=False)
    S_.to_csv(OUT / f'd23_boost_seeds_h{h}.csv', index=False)
    pd.set_option('display.width', 250)
    print(L_.round(5).to_string(), flush=True)
    pv = S_.pivot_table(index=['boost', 'model'], columns='seed', values='2017-2020')
    pv['mean3'] = pv.mean(axis=1)
    print(pv.round(5).to_string(), flush=True)
    return L_, S_

if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'ebar'
    if cmd == 'wens':  # wens <h> <out> <члены...>
        make_wens(int(sys.argv[2]), sys.argv[4:], sys.argv[3])
    elif cmd == 'evalb':  # evalb <h> <база> <кандидаты...>
        eval_boost(int(sys.argv[2]), sys.argv[4:], sys.argv[3])
    elif cmd == 'boost5':  # boost5 <вар...> [--workers N] [--check]
        args = sys.argv[2:]
        w = int(args[args.index('--workers') + 1]) if '--workers' in args else 3
        boost5_run([x for x in args if x in BCANDS5], workers=w, check='--check' in args)
    elif cmd == 'summary':
        gibrid_summary()
    elif cmd == 'gibrid':  # gibrid <boost> <h> <ноги...>
        run_gibrid(sys.argv[2], int(sys.argv[3]), sys.argv[4:] or ['single'])
    elif cmd == 'cand':  # cand <boost>:<база>:<вариант>
        cmd_cand(sys.argv[2])
    elif cmd == 'seeds22':
        seeds22(sys.argv[2] if len(sys.argv) > 2 else 'mk10')
    elif cmd == 'boost':  # boost <канд...> [--workers N] [--check]
        args = sys.argv[2:]
        w = int(args[args.index('--workers') + 1]) if '--workers' in args else 3
        boost_run([x for x in args if x in BCANDS], workers=w, check='--check' in args)
    else:
        {'ebar': cmd_ebar, 'kappa': cmd_kappa, 'selftest': selftest, 'xfeat': cmd_xfeat, 'harx': harx_ext}[cmd]()
