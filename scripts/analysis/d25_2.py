"""Д25.2: факторная декомпозиция h=22 (владелец 26.10, 04:50; ночные умолчания — в docs/otchety/2026-10-26_d25a.md).
Рынок: RV_m(t) = rv_IMOEX(t)^2/252 (rv_IMOEX в features_pit_v3 — годовая волатильность, sqrt), RVI — close_RVI (% годовых) -> (RVI/100)^2/252.
beta_i(t): OLS ln rv_i на ln RV_m по окну 504 торговых дня тикера, заканчивающемуся в t (дневные значения известны к закрытию t), min 60 набл.
Рынок h=22: HAR без J на ln RV_m (ln среднего RV_m за 1/5/22 дня) [+ ln(RVI^2/252) — привязка Д25.1]; цель — среднее ln RV_m за [t+1, t+22];
  шаг s: обучение — даты года < val-года шага, у которых цель закончилась до начала val-года (эмбарго 22 торговых дня), val — год 2015+s, тест — 2016+s.
Собственная часть: u_i(tau) = ln rv_i(tau) - beta_i(t) ln RV_m(tau); цель бустинга — среднее u_i за [t+1, t+22] при beta_i(t).
Бустинг: признаки, шаги, исключения и гиперпараметры D15_h22 (= D15v_h22_base: cfg={}, пространство B v2, Bv2b; D15_h22_selected_h22.csv),
  ES на val-годе (как протокол), seed 42/43/44. Новый optuna-30 не делался (ночное умолчание, срок 06:00).
Прогноз: ln y = beta_i(t)*yhat_mkt + uhat; (А) yhat_mkt = HAR(+RVI) рынка; (Б) yhat_mkt = среднее ln RV_m за [t-21, t] (наивный рынок, без HAR).
Выход: data/regen/D25/D25_2{A,An,B}_h22_{xgb|lgbm}_h22[_seeds4344]_val.parquet (+ _s42/_s43/_s44 — один seed в single), mkt_*.csv, beta_*.parquet.
python3 scripts/analysis/d25_2.py [--fit] [--leak] [--eval]"""
import argparse, os, sys, time
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
R = Path('/Users/nilysenok/Desktop/Старое/PhD_Vol_trading/data/regen')
os.environ['L2_FEAT'] = str(R / 'features_pit_v3.parquet')
OUT = R / 'D25'
FEAT = R / 'features_pit_v3.parquet'
K = ['date', 'ticker', 'step']
H = 22
WB, MINB = 504, 60
EPS = 1e-9


# ---------------------------------------------------------------- рынок и beta
def market(A):
    """Дневной ряд рынка по датам: RV_m, RVI (дневная дисперсия). Один и тот же на все тикеры даты -> берётся медиана по тикерам."""
    g = A.groupby('date')
    m = pd.DataFrame({'rvm': g['rv_IMOEX'].median() ** 2 / 252, 'rvi': (g['close_RVI'].median() / 100) ** 2 / 252}).sort_index()
    m = m[m.rvm > 0].copy()
    m['lm'] = np.log(m.rvm.clip(EPS))
    m['x1'] = m.lm
    m['x5'] = np.log(m.rvm.rolling(5, min_periods=1).mean())
    m['x22'] = np.log(m.rvm.rolling(22, min_periods=1).mean())
    m['xrvi'] = np.log(m.rvi.ffill().clip(EPS))
    m['naive'] = m.lm.rolling(22, min_periods=5).mean()  # (Б): среднее ln RV_m за [t-21, t]
    m['tgt'] = m.lm[::-1].rolling(H, min_periods=H).mean()[::-1].shift(-1)  # среднее ln RV_m за [t+1, t+22]
    m['year'] = pd.DatetimeIndex(m.index).year
    return m


def market_har(m, rvi):
    """HAR рынка по шагам. -> DataFrame date, step, part(val/test), mhat; коэффициенты."""
    xs = ['x1', 'x5', 'x22'] + (['xrvi'] if rvi else [])
    rows, co = [], []
    for s in range(1, 10):
        vy, ty = 2015 + s, 2016 + s
        tr = m[(m.year < vy)].dropna(subset=xs + ['tgt'])
        first_v = m.index[m.year == vy].min()
        pos = m.index.get_loc(first_v)
        cut = m.index[pos - H]  # цель строки t заканчивается в t+22 (рыночные даты); t <= cut -> цель известна к началу val-года
        tr = tr[tr.index < cut]
        X = np.column_stack([np.ones(len(tr)), tr[xs].values])
        b = np.linalg.lstsq(X, tr.tgt.values, rcond=None)[0]
        co.append(dict(step=s, rvi=rvi, n=len(tr), last_train=tr.index.max(), **dict(zip(['c'] + xs, b))))
        for part, yr in (('val', vy), ('test', ty)):
            d = m[m.year == yr].dropna(subset=xs)
            rows.append(pd.DataFrame({'date': d.index, 'step': s, 'part': part, 'mhat': np.column_stack([np.ones(len(d)), d[xs].values]) @ b}))
    return pd.concat(rows, ignore_index=True), pd.DataFrame(co)


def beta_frame(A, m, win=WB, shift_future=0):
    """beta_i(t) по окну [t-win+1, t] торговых дней тикера. shift_future>0 — ПОДЛОЖЕННАЯ УТЕЧКА (окно до t+shift_future) для самопроверки."""
    out = []
    for tk, g in A.groupby('ticker'):
        g = g.sort_values('date')
        x = g.date.map(m.lm).values
        y = np.log(g.rv.clip(EPS).values)
        ok = np.isfinite(x) & np.isfinite(y)
        d = pd.DataFrame({'x': np.where(ok, x, 0), 'y': np.where(ok, y, 0), 'n': ok.astype(float)})
        d['xx'], d['xy'] = d.x * d.x, d.x * d.y
        r = d.rolling(win, min_periods=1).sum()
        if shift_future:
            r = r.shift(-shift_future)
        n = r.n
        vx = r.xx - r.x ** 2 / n
        b = (r.xy - r.x * r.y / n) / vx
        b[(n < MINB) | ~np.isfinite(b)] = np.nan
        xbar = r.x / n  # среднее ln RV_m по тому же окну (центр рынка, только прошлое)
        # собственная цель: среднее ln rv_i за [t+1, t+22] и среднее ln RV_m за те же дни тикера
        ly = pd.Series(np.where(ok, y, np.nan))
        lx = pd.Series(np.where(np.isfinite(x), x, np.nan))
        fy = ly[::-1].rolling(H, min_periods=H).mean()[::-1].shift(-1).values
        fx = lx[::-1].rolling(H, min_periods=H).mean()[::-1].shift(-1).values
        out.append(pd.DataFrame({'date': g.date.values, 'ticker': tk, 'beta': b.values, 'xbar': xbar.values, 'fy': fy, 'fx': fx}))
    B = pd.concat(out, ignore_index=True)
    B['beta'] = B.beta.fillna(1.0)  # тикеры без 60 наблюдений до даты — beta = 1 (ночное умолчание)
    # ↺ ночное умолчание: рыночный фактор центрируется средним ln RV_m окна beta (xbar_i(t), только прошлое):
    # u = ln rv - beta*(ln RV_m - xbar). Без центра u определяется уровнем beta*ln RV_m (ln RV_m ~ -10, beta 0,37-0,63 по шагам ->
    # сдвиг u ~2 при неизменной волатильности); первый прогон без центра: V0 QLIKE 1,03 (2017-2020).
    B['utgt'] = B.fy - B.beta * (B.fx - B.xbar)
    return B


# ---------------------------------------------------------------- бустинг собственной части
G = {}


def _init():
    import l2_lib as L
    A, FC = L.load_base(None, (), H)
    A2, cols, _, _ = L.build_features(A, FC, H, {})
    raw = pd.read_parquet(FEAT)
    m = market(raw)
    Bt = beta_frame(raw, m)
    A2 = A2.merge(Bt[['date', 'ticker', 'beta', 'xbar', 'utgt']], on=['date', 'ticker'], how='left')
    A2['naive'] = A2.date.map(m.naive)
    G.update(L=L, A=A2, cols=cols, m=m, B=Bt)


def _task(a):
    model, step, seed = a
    import tune_L2_optuna as T
    L = G['L']
    sel = pd.read_csv(R / 'D15_h22_selected_h22.csv')
    r = sel[(sel.model == model) & (sel.step == step) & (sel.seed == 42)].iloc[0]
    p = {k: (int(r[k]) if sc in ('int', 'logint') else float(r[k])) for k, _, _, sc in T.SPACEB[model]}
    D = L.step_data(G['A'], G['cols'], H, step, model, {}, 'Bv2b')
    ref = L.make_ref(model, G['A'], G['cols'], H, {}, 'Bv2b') if step > 1 else None
    # обучение: строки с известной собственной целью
    tp = D['parts']['tr']
    keep = np.isfinite(tp.utgt.values)
    D['X']['tr'] = D['X']['tr'][keep]; D['parts']['tr'] = tp[keep]; D['ytr_ln'] = tp.utgt.values[keep]
    D['off']['tr'] = np.zeros(keep.sum()); D['w'] = None if D['w'] is None else D['w'][keep]
    # val (ES, как протокол): метка = собственная цель val-года (у последних 22 дней 9-го шага цели нет -> ln-цель уровня, их мало)
    va = D['parts']['va']
    uv = va.utgt.values
    uv = np.where(np.isfinite(uv), uv, np.log(D['y']['va'] + 1e-10) - va.beta.values * (va.naive.values - va.xbar.values))
    D['off']['va'] = np.zeros(len(va)); D['off']['te'] = np.zeros(len(D['parts']['te']))
    D['y']['va'] = np.exp(uv) - 1e-10
    pv, pt, bi, _, _ = L.fit(model, p, D, {}, seed=seed, ref=ref)
    res = []
    for k, part, pr in (('val', 'va', pv), ('test', 'te', pt)):
        d = D['parts'][part]
        res.append(pd.DataFrame({'date': d.date.values, 'ticker': d.ticker.values, 'step': step, 'year': D['vy'] if k == 'val' else D['ty'],
                                 'rv_actual': d[L.tcol(H)].values, 'uhat': pr, 'beta': d.beta.values, 'xbar': d.xbar.values, 'naive': d.naive.values,
                                 'model': model, 'seed': seed, 'part': k}))
    print(f'  {model} шаг {step} seed {seed}: деревьев {bi}', flush=True)
    return pd.concat(res)


def fit(workers=2):
    import multiprocessing as mp
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    _init()
    m = G['m']
    G['B'].to_parquet(OUT / 'D25_2_beta.parquet', index=False)
    mk = {}
    for tag, rvi in (('A', True), ('An', False)):
        mh, co = market_har(m, rvi)
        co.to_csv(OUT / f'D25_2_mkt_coef_{tag}.csv', index=False)
        mk[tag] = mh
        print(co.round(3).to_string(), flush=True)
    tasks = [(mo, s, sd) for sd in (42, 43, 44) for mo in ('xgb', 'lgbm') for s in range(1, 10)]
    with mp.get_context('fork').Pool(workers) as P:
        U = pd.concat(P.map(_task, tasks, chunksize=1), ignore_index=True)
    U.to_parquet(OUT / 'D25_2_uhat.parquet', index=False)
    print(f'бустинг: {time.time() - t0:.0f} с', flush=True)
    write(U, mk)


def write(U, mk):
    c = K + ['year', 'rv_actual', 'rv_pred']
    V = U[U.part == 'val']
    for tag in ('A', 'An', 'B'):
        if tag == 'B':
            X = V.assign(mhat=V.naive)
        else:
            X = V.merge(mk[tag][mk[tag].part == 'val'][['date', 'step', 'mhat']], on=['date', 'step'], how='left')
        assert X.mhat.notna().all(), tag
        X = X.assign(rv_pred=np.exp(X.beta * (X.mhat - X.xbar) + X.uhat).clip(1e-10))
        b = f'D25_2{tag}_h22'
        for mo, slot in (('xgb', 'xgb'), ('lgbm', 'lgbm')):
            x = X[X.model == mo]
            x[x.seed == 42][c].sort_values(['step', 'date', 'ticker']).to_parquet(OUT / f'{b}_{slot}_h22_val.parquet', index=False)
            x[x.seed != 42][c + ['seed']].to_parquet(OUT / f'{b}_{slot}_h22_seeds4344_val.parquet', index=False)
            for sd in (42, 43, 44):
                x[x.seed == sd][c].sort_values(['step', 'date', 'ticker']).to_parquet(OUT / f'{b}_s{sd}_{slot}_h22_val.parquet', index=False)
    print('val-файлы записаны', flush=True)


# ---------------------------------------------------------------- утечки
def leak():
    rng = np.random.default_rng(0)
    raw = pd.read_parquet(FEAT)
    m = market(raw)
    res = []
    # (а) beta(t) не меняется при порче rv_i и RV_m после t
    B0 = beta_frame(raw, m)
    dates = sorted(raw.date.unique())
    for cut in (dates[800], dates[1500]):
        A1 = raw.copy(); f = A1.date > cut
        A1.loc[f, 'rv'] = A1.loc[f, 'rv'].values * rng.uniform(0.1, 10, f.sum())
        m1 = m.copy(); fm = m1.index > cut
        m1.loc[fm, 'lm'] = m1.loc[fm, 'lm'] + rng.normal(size=fm.sum())
        B1 = beta_frame(A1, m1)
        g0, g1 = B0[B0.date <= cut], B1[B1.date <= cut]
        assert np.allclose(g0.beta.values, g1.beta.values, atol=1e-12)
        # подложенная утечка: окно до t+22 должно поменяться
        L0, L1 = beta_frame(raw, m, shift_future=22), beta_frame(A1, m1, shift_future=22)
        caught = not np.allclose(L0[L0.date <= cut].beta.values, L1[L1.date <= cut].beta.values, atol=1e-12)
        assert caught, 'подложенная утечка не поймана'
    res.append('(а) beta_i(t) не меняется при порче rv_i и ln RV_m после t (2 точки среза): OK; подложенная утечка (окно до t+22) поймана: OK')
    # (б) рынок шага s не меняется при порче всех дат с val-года шага (включая цель); подложенная утечка (без эмбарго) ловится
    for rvi in (True, False):
        mh0, co0 = market_har(m, rvi)
        for s in range(1, 10):
            vy = 2015 + s
            m1 = m.copy(); f = m1.year >= vy
            for col in ('lm', 'x1', 'x5', 'x22', 'xrvi'):
                m1.loc[f, col] = m1.loc[f, col] + rng.normal(size=f.sum())
            # цель строк до val-года, заходящая в val-год, тоже портится
            m1['tgt'] = m1.lm[::-1].rolling(H, min_periods=H).mean()[::-1].shift(-1)
            _, co1 = market_har(m1, rvi)
            xs = [c for c in co0.columns if c.startswith(('c', 'x')) and c != 'n']
            assert np.allclose(co0.loc[s - 1, xs].astype(float), co1.loc[s - 1, xs].astype(float), atol=1e-10), (rvi, s)
    res.append('(б) коэффициенты HAR рынка (с RVI и без) шага s не меняются при порче всех дат с val-года шага и цели, заходящей в него: OK')

    def har_noemb(mm, s):
        vy = 2015 + s; tr = mm[mm.year < vy].dropna(subset=['x1', 'x5', 'x22', 'tgt'])
        X = np.column_stack([np.ones(len(tr)), tr[['x1', 'x5', 'x22']].values]); return np.linalg.lstsq(X, tr.tgt.values, rcond=None)[0]
    m1 = m.copy(); f = m1.year >= 2018
    m1.loc[f, 'lm'] = m1.loc[f, 'lm'] + 3
    m1['tgt'] = m1.lm[::-1].rolling(H, min_periods=H).mean()[::-1].shift(-1)
    assert not np.allclose(har_noemb(m, 3), har_noemb(m1, 3)), 'подложенная утечка (без эмбарго) не поймана'
    res.append('(б\') подложенная утечка: HAR рынка без эмбарго меняется при порче val-года (цель последних 22 дней заходит в него): поймана: OK')
    # (в) собственная цель обучения шага s не заходит в val-год: у строк обучения (годы < val-года) с эмбарго нет — как D15_h22 —
    # поэтому проверяем: цель строки t использует только даты (t, t+22] и beta(t); порча дат > t+22 не меняет utgt(t)
    for cut in (dates[900],):
        A1 = raw.copy(); i = dates.index(cut)
        f = A1.date > dates[i + H]
        A1.loc[f, 'rv'] = A1.loc[f, 'rv'].values * 7
        m1 = m.copy(); m1.loc[m1.index > dates[i + H], 'lm'] += 2
        B1 = beta_frame(A1, m1)
        g0, g1 = B0[B0.date <= cut].utgt.values, B1[B1.date <= cut].utgt.values
        ok = np.isfinite(g0)
        assert np.allclose(g0[ok], g1[ok], atol=1e-12)
    res.append('(в) собственная цель строки t зависит только от (t, t+22] и beta(t): порча дат > t+22 не меняет её: OK')
    # (г) прогнозы бустинга: val шага s — год 2015+s моделью шага s (как D15_h22)
    U = pd.read_parquet(OUT / 'D25_2_uhat.parquet') if (OUT / 'D25_2_uhat.parquet').exists() else None
    if U is not None:
        v = U[U.part == 'val']
        assert (v.year == 2015 + v.step).all()
        res.append('(г) бустинг собственной части: val шага s — год 2015+s моделью шага s (обучение — годы < val-года, split l2_lib): OK')
    # выравнивание рынка по дате (справка)
    cs = raw.groupby('date').rv.mean()
    j = pd.concat([m.rvm, cs], axis=1, keys=['m', 'cs']).dropna()
    cor = {k: float(np.corrcoef(np.log(j.m.shift(k).dropna()), np.log(j.cs[j.m.shift(k).notna()]))[0, 1]) for k in (-1, 0, 1)}
    res.append(f'(справка) corr(ln RV_m(t+k), ln среднего rv тикеров(t)) k=-1/0/+1: {cor[-1]:.3f} / {cor[0]:.3f} / {cor[1]:.3f}')
    (OUT / 'leak_d25_2.txt').write_text('\n'.join(res) + '\n'); print('\n'.join(res))


# ---------------------------------------------------------------- оценка
def evaluate():
    import final3 as F3
    rows = []
    for d in sorted((OUT / 'gibrid').iterdir()):
        if not d.is_dir() or not d.name.startswith(('D25_2', 'D15_h22')):
            continue
        D = F3.designs(dict(gib_dir=d, gboost=d.name, h=22))
        if D is None:
            continue
        S = F3.summarize(dict(), D)
        for tag, sub in (('лучший', S), ('V0_XGB', S[S.design.str.contains(r'\|V0_XGB\|')]), ('V0_LGB', S[S.design.str.contains(r'\|V0_LGB\|')])):
            if len(sub):
                x = sub.iloc[0]
                rows.append(dict(boost=d.name, kind=tag, design=x.design, n_failed=x.n_failed, failed=x.failed, mean_1720=x.mean_1720, q2021=x.q2021,
                                 mean_1721=x.mean_1721))
    E = pd.DataFrame(rows); E.to_csv(OUT / 'eval_D25_2.csv', index=False)
    with pd.option_context('display.width', 250, 'display.max_columns', 20):
        print(E.round(4).to_string())


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--fit', action='store_true'); ap.add_argument('--leak', action='store_true')
    ap.add_argument('--eval', action='store_true'); ap.add_argument('--workers', type=int, default=2)
    a = ap.parse_args()
    if a.fit:
        fit(a.workers)
    if a.leak:
        leak()
    if a.eval:
        evaluate()


if __name__ == '__main__':
    main()
