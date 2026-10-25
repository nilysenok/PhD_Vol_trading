"""Д16: HAR-X = HAR-J + линейные слагаемые (эластичная сеть по времени), пул 17 тикеров, лог, расширяющееся окно, шаги 1-10.
Слагаемые по умолчанию (определение владельца 26.10): ln RVI^2 (срез 18:50), ln RV_m IMOEX, ln средней RV_m по 17 тикерам, отрицательные
доходности тикера за 1/5/22 дня (LHAR), PC1 глобальной волатильности (Д7.2), ln прогноза финального гибрида h=1 (для h=5/22).
Расширяемость (Д17): build_terms() возвращает таблицу слагаемых; TERMS - реестр {имя: функция(A)->Series по индексу A}; любой столбец A
тоже можно назвать слагаемым; harx_forecasts(h, A, harj, terms=[...]) принимает список имён.
Схема шага k (как gibrid_har.har_forecasts): тест-год ty = 2016+k, val-год vy = ty-1, обучение - годы < vy. HAR-J-члены не штрафуются
(Фриш-Во: y и X слагаемых очищаются от [1, ln HAR-J] на обучении, затем ElasticNet на остатках - точное решение задачи со штрафом
только на слагаемые). alpha и l1_ratio - по внутренней валидации шага по времени: обучение < vy-1, оценка на vy-1 (MSE логов),
затем переобучение на < vy. Val-год шага оценивается вне выборки (ни веса, ни гиперпараметры его не видели).
Выход: rv_pred - без поправки (поправку Б0.1 применяет гибрид/оценка), плюс rv_pred_{sm,smrob,ord} - Б0.1 (val: оценка шага k-1, F=1
на шаге 1; тест: оценка шага k).
Запуск: python3 d16_harx.py [--h 1 5 22] [--terms ...] [--tag HARX] [--leak-test]"""
import argparse, os, sys, json, time
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import ElasticNet, LinearRegression

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gibrid_har as GH
import d7_block as D7

DATA = Path('/Users/nilysenok/Desktop/Старое/PhD_Vol_trading/data')
REGEN = DATA / 'regen'
FEAT = REGEN / 'features_pit_v3.parquet'
ALPHAS = (1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3, 1.0)
L1S = (0.1, 0.5, 0.9, 1.0)
CODES = ('ord', 'rob', 'sm', 'smrob')


# ---------------------------------------------------------------- слагаемые
def _by_date(A):
    """Уникальные даты с внешними столбцами; строки-заполнители (как Д7) исключаются и протягиваются последним реальным днём."""
    cols = ['close_RVI', 'rv_IMOEX', 'imoex_low_bars', D7.REF] + D7.GLOBAL + ['vix']
    x = A[['date'] + [c for c in dict.fromkeys(cols) if c in A]].drop_duplicates('date').sort_values('date').set_index('date')
    fill = (x[D7.REF] == x[D7.REF].iloc[0]).values
    v = x[~fill].copy()
    v.loc[v['imoex_low_bars'] == 1, 'rv_IMOEX'] = np.nan
    v['rv_IMOEX'] = v['rv_IMOEX'].ffill(limit=2)
    return x, v


def t_rvi(A):
    x, v = _by_date(A)
    s = np.log(v['close_RVI'].clip(lower=1e-6) ** 2).reindex(x.index, method='ffill')
    return pd.Series(s.reindex(A['date']).values, index=A.index)


def t_rvm_imoex(A):
    x, v = _by_date(A)
    s = np.log((v['rv_IMOEX'] ** 2).rolling(22, min_periods=13).mean().clip(lower=1e-10)).reindex(x.index, method='ffill')
    return pd.Series(s.reindex(A['date']).values, index=A.index)


def t_rvm_mkt(A):
    s = np.log(A.groupby('date')['rv_m'].mean().clip(lower=1e-12))
    return pd.Series(s.reindex(A['date']).values, index=A.index)


def _ret(A):
    """Лог-доходность close-to-close тикера (close - срез матрицы); консолидация IRAO 20.01.2015 (×100, corporate_actions: applied) учтена."""
    d = A[['date', 'ticker', 'close']].copy()
    ca = pd.read_csv(DATA / 'corporate_actions.csv')
    for _, r in ca[ca.status == 'applied'].iterrows():
        m = (d.ticker == r.ticker) & (pd.to_datetime(d.date) < pd.Timestamp(r.date))
        d.loc[m, 'close'] = d.loc[m, 'close'] * float(r.factor)
    d = d.sort_values(['ticker', 'date'])
    r = np.log(d['close'].clip(lower=1e-12)).groupby(d['ticker']).diff().clip(-0.5, 0.5)
    return r.reindex(A.index)


def _lhar(w):
    def f(A):
        r = _ret(A).fillna(0.0)
        tk = A['ticker']
        order = A.sort_values(['ticker', 'date']).index
        m = r.loc[order].groupby(tk.loc[order]).transform(lambda s: s.rolling(w, min_periods=1).mean())
        return np.minimum(m, 0.0).reindex(A.index)
    return f


def t_pc1(A):
    x, v = _by_date(A)
    pc = D7._gvol(v)['d7_gvol_pc1'].reindex(x.index, method='ffill')
    return pd.Series(pc.reindex(A['date']).values, index=A.index).fillna(0.0)


_SIG = {}


def t_sig1(A):
    """ln(σ̂²_1 / F_шага) - ln HAR-J_1 того же шага: превышение прогноза финального гибрида h=1 (OOS по шагам, без его поправки Б0.1)
    над HAR-J h=1 (OOS той же схемы шагов); там, где σ̂ нет (до 2017), - 0 (= «без информации сверх HAR-J»)."""
    if 'v' not in _SIG:
        s = pd.read_parquet(REGEN / 'sigma_pred_h1_final2.parquet')
        st = pd.read_csv(REGEN / 'sigma_pred_h1_final2_steps.csv')
        s['ty'] = pd.to_datetime(s.date).dt.year
        s = s.merge(st[['test_year', 'F']].rename(columns={'test_year': 'ty'}), on='ty')
        s['ln1'] = np.log(s.sigma_pred_h1.astype('float64') ** 2 / s.F)
        harj = [c for c in GH_HARJ if c in A]
        _, ht, _ = har_steps(1, A, harj)
        ht = ht[['date', 'ticker', 'rv_pred']].rename(columns={'rv_pred': 'har1'})
        m = s.merge(ht, on=['date', 'ticker'])
        m['dev'] = m.ln1 - np.log(m.har1)
        _SIG['v'] = m[['date', 'ticker', 'dev']]
    k = A[['date', 'ticker']].merge(_SIG['v'], on=['date', 'ticker'], how='left')
    return pd.Series(k['dev'].fillna(0.0).values, index=A.index)


def _lagcol(col, lag=0, log=True):
    def f(A):
        x = A[col].astype('float64')
        if lag:
            x = x.groupby(A['ticker']).shift(lag)
        return np.log(x.clip(lower=1e-12)) if log else x
    return f


def t_harq(A):
    """Д17.3 HARQ (Bollerslev-Patton-Quaedvlieg): член RV_d·√RQ_d в логарифмической форме - ln RV_d × √(RQ_d)/RV_d,
    где RQ = (n/3)·Σr⁴ = (n/3)·rkurt·RV² (rkurt = Σr⁴/RV²) => √RQ/RV = √(n·rkurt/3); день RV_d (лаг 1, как rv_d)."""
    q = np.sqrt((A['n_bars'].astype('float64') * A['rkurt'].astype('float64') / 3).clip(lower=0)).clip(upper=10)
    q = q.groupby(A['ticker']).shift(1)
    return np.log(A['rv_d'].astype('float64').clip(lower=1e-12)) * q


def _rvlong(w):
    def f(A):
        order = A.sort_values(['ticker', 'date']).index
        m = A.loc[order, 'rv_d'].groupby(A.loc[order, 'ticker']).transform(lambda s: s.rolling(w, min_periods=int(0.6 * w)).mean())
        return np.log(m.clip(lower=1e-12)).reindex(A.index)
    return f


TERMS = {'ln_rvi2': t_rvi, 'ln_rvm_imoex': t_rvm_imoex, 'ln_rvm_mkt': t_rvm_mkt,
         'lhar_1': _lhar(1), 'lhar_5': _lhar(5), 'lhar_22': _lhar(22), 'pc1': t_pc1, 'ln_sig1': t_sig1,
         # Д17.3 (кандидаты в той же сети): HARQ, RSV+/RSV- раздельно (день t, срез 18:50), RV за 66/132 дня (для h=22)
         'harq': t_harq, 'ln_rsv_pos': _lagcol('rsv_pos'), 'ln_rsv_neg': _lagcol('rsv_neg'), 'ln_rv66': _rvlong(66), 'ln_rv132': _rvlong(132)}
D17_TERMS = {1: ['harq', 'ln_rsv_pos', 'ln_rsv_neg'], 5: ['harq', 'ln_rsv_pos', 'ln_rsv_neg'], 22: ['harq', 'ln_rsv_pos', 'ln_rsv_neg', 'ln_rv66', 'ln_rv132']}
BASE_TERMS = ['ln_rvi2', 'ln_rvm_imoex', 'ln_rvm_mkt', 'lhar_1', 'lhar_5', 'lhar_22', 'pc1', 'ln_sig1']
GH_HARJ = None


def default_terms(h, d17=True):
    """HAR-X: слагаемые владельца (для h=1 без ln_sig1) + кандидаты Д17.3 в той же сети (d17=False - только слагаемые владельца, HARX0)."""
    return [t for t in BASE_TERMS if not (h == 1 and t == 'ln_sig1')] + (D17_TERMS[h] if d17 else [])


def build_terms(A, names):
    """Таблица слагаемых (индекс A). Имя - из TERMS или существующий столбец A (как есть)."""
    out = {}
    for n in names:
        out[n] = TERMS[n](A) if n in TERMS else A[n].astype('float64')
    X = pd.DataFrame(out, index=A.index)
    return X.replace([np.inf, -np.inf], np.nan)


# ---------------------------------------------------------------- шаги
def steps(A):
    ys = [y for y in range(2017, 2027) if y in A.year.values]
    for k, ty in enumerate(ys, 1):
        yield k, ty, ty - 1


def har_steps(h, A, harj):
    """HAR-J по шагам 1-10 (как gibrid_har.har_forecasts, + шаг 10: val 2025, тест 2026)."""
    target = f'rv_target_h{h}'
    vals, tests, s2 = [], [], {}
    X = lambda d: GH.clean(np.log(d[harj].values.clip(1e-10)))
    for k, ty, vy in steps(A):
        tp = A[A.year < vy].dropna(subset=[target]); vi = A[A.year == vy].dropna(subset=[target]); td = A[A.year == ty]
        y = np.log(tp[target].values + 1e-10)
        m = LinearRegression().fit(X(tp), y)
        for df, lst, yr in ((vi, vals, vy), (td, tests, ty)):
            lst.append(pd.DataFrame({'date': df['date'].values, 'ticker': df['ticker'].values, 'year': yr, 'step': k,
                                     'rv_actual': df[target].values, 'rv_pred': np.clip(np.exp(m.predict(X(df))), 1e-10, None)}))
    return pd.concat(vals, ignore_index=True), pd.concat(tests, ignore_index=True), s2


def _fwl(H, Z, y):
    """Очистка от столбцов H (с константой) по МНК на обучении; возвращает функцию-проектор и коэффициенты."""
    Hc = np.column_stack([np.ones(len(H)), H])
    B, *_ = np.linalg.lstsq(Hc, np.column_stack([y, Z]), rcond=None)
    return B


def fit_en(H, Z, y, alpha, l1):
    """y ~ [1, H] без штрафа + Z со штрафом ElasticNet (Z стандартизуется по обучению). Возвращает predict(H, Z)."""
    mu, sd = Z.mean(0), Z.std(0)
    keep = sd > 1e-12
    Zs = np.zeros_like(Z)
    Zs[:, keep] = (Z[:, keep] - mu[keep]) / sd[keep]
    B = _fwl(H, Zs, y)
    Hc = np.column_stack([np.ones(len(H)), H])
    R = np.column_stack([y, Zs]) - Hc @ B
    g = np.zeros(Z.shape[1])
    if keep.any():
        if alpha > 0:
            en = ElasticNet(alpha=alpha, l1_ratio=l1, fit_intercept=False, max_iter=20000, tol=1e-7, selection='cyclic', precompute=True)
            en.fit(R[:, 1:][:, keep], R[:, 0])
            g[keep] = en.coef_
        else:
            g[keep] = np.linalg.lstsq(R[:, 1:][:, keep], R[:, 0], rcond=None)[0]
    b = B[:, 0] - B[:, 1:] @ g  # коэффициенты при [1, H]

    def pred(H2, Z2):
        Z2s = np.zeros_like(Z2)
        Z2s[:, keep] = (Z2[:, keep] - mu[keep]) / sd[keep]
        return np.column_stack([np.ones(len(H2)), H2]) @ b + Z2s @ g
    coef = {'b': b, 'g_std': g, 'g_raw': np.where(keep, g / np.where(keep, sd, 1), 0.0)}
    return pred, coef


INNER = 1  # число внутренних валидационных лет (1 - год vy-1; n - средняя MSE по годам vy-1..vy-n, обучение каждого - на годах до него)


def forecast_step(A, h, step, harj, terms, Xt=None, log=None, grid=True, all_rows=False, inner=None):
    """ОДИН шаг: матрица A (load()), горизонт h, шаг k (1-10) -> (val_df, test_df) OOS. Val-год vy = 2015+k, тест ty = 2016+k,
    обучение - годы < vy; гиперпараметры сети - по году vy-1 (обучение < vy-1). Столбцы: date, ticker, year, step, rv_actual, rv_pred (без Б0.1)."""
    target = f'rv_target_h{h}'
    if Xt is None:
        Xt = build_terms(A, terms)
    Xt = Xt[terms]  # пропуски - средним слагаемого по обучающим строкам этой подгонки (не нулём: у ln-слагаемых 0 - выброс)
    Hf = lambda idx: GH.clean(np.log(A.loc[idx, harj].values.clip(1e-10)))
    Zf = lambda idx, ref: Xt.loc[idx].fillna(Xt.loc[ref].mean()).fillna(0.0).values.astype('float64')
    ok = A[target].notna()
    k, ty, vy = step, 2016 + step, 2015 + step
    tp = A.index[(A.year < vy) & ok]; vi = A.index[(A.year == vy) & ok]; td = A.index[A.year == ty]
    best = (0.0, 1.0, np.nan)
    if grid and terms:
        n_in = INNER if inner is None else inner
        acc = {}
        for j in range(1, n_in + 1):
            i_tr = A.index[(A.year < vy - j) & ok]; i_va = A.index[(A.year == vy - j) & ok]
            if len(i_va) == 0 or len(i_tr) == 0:
                continue
            yt, yv = np.log(A.loc[i_tr, target].values + 1e-10), np.log(A.loc[i_va, target].values + 1e-10)
            Htr, Ztr, Hva, Zva = Hf(i_tr), Zf(i_tr, i_tr), Hf(i_va), Zf(i_va, i_tr)
            for l1, al in [(l1, al) for l1 in L1S for al in ALPHAS] + [(0.0, 0.0)]:
                p, _ = fit_en(Htr, Ztr, yt, al, l1 if al > 0 else 1.0)
                acc.setdefault((al, l1), []).append(float(np.mean((yv - p(Hva, Zva)) ** 2)))
        res = [(al, l1, float(np.mean(v))) for (al, l1), v in acc.items()]
        best = min(res, key=lambda r: r[2])
    y = np.log(A.loc[tp, target].values + 1e-10)
    pred, coef = fit_en(Hf(tp), Zf(tp, tp), y, best[0], best[1])
    s2tr = float(np.var(y - pred(Hf(tp), Zf(tp, tp))))
    if log is not None:
        log.append(dict(h=h, step=k, val_year=vy, alpha=best[0], l1_ratio=best[1], inner_mse=best[2], s2_train=s2tr,
                        n_nonzero=int((np.abs(coef['g_std']) > 1e-10).sum()),
                        **{f'g_{t}': v for t, v in zip(terms, coef['g_raw'])},
                        **{f'b_{c}': v for c, v in zip(['const'] + list(harj), coef['b'])}))
    out = []
    for idx, yr in ((vi, vy), (td, ty)):
        out.append(pd.DataFrame({'date': A.loc[idx, 'date'].values, 'ticker': A.loc[idx, 'ticker'].values, 'year': yr, 'step': k,
                                 'rv_actual': A.loc[idx, target].values,
                                 'rv_pred': np.clip(np.exp(pred(Hf(idx), Zf(idx, tp))), 1e-10, None)}))
    if all_rows:  # ln-прогноз модели шага на всех строках A (обучение - внутри выборки), для смещения V9 / признака Д8.12
        return out[0], out[1], pd.Series(pred(Hf(A.index), Zf(A.index, tp)), index=A.index)
    return out[0], out[1]


def ln_all_steps(h, terms=None, steps_=range(1, 11), A=None, harj=None, Xt=None):
    """(date, ticker, hx1..hx10): ln HAR-X модели шага k на всех строках матрицы - для V9 (смещение) и Д8.12 (признак) на HAR-X."""
    if A is None:
        A, harj = load()
    terms = default_terms(h) if terms is None else terms
    Xt = build_terms(A, terms) if Xt is None else Xt
    out = A[['date', 'ticker']].copy()
    for k in steps_:
        out[f'hx{k}'] = forecast_step(A, h, k, harj, terms, Xt, all_rows=True)[2].values
    return out


def harx_forecasts(h, A, harj, terms, Xt=None, log=None, grid=True):
    """HAR-X по шагам 1-10. Xt - таблица слагаемых (если None - строится). log - список для коэффициентов/гиперпараметров."""
    target = f'rv_target_h{h}'
    if Xt is None:
        Xt = build_terms(A, terms)
    Xt = Xt[terms]  # пропуски - средним слагаемого по обучающим строкам этой подгонки (не нулём: у ln-слагаемых 0 - выброс)
    Hf = lambda idx: GH.clean(np.log(A.loc[idx, harj].values.clip(1e-10)))
    Zf = lambda idx, ref: Xt.loc[idx].fillna(Xt.loc[ref].mean()).fillna(0.0).values.astype('float64')
    vals, tests = [], []
    for k, ty, vy in steps(A):
        v, t = forecast_step(A, h, k, harj, terms, Xt, log, grid)
        vals.append(v); tests.append(t)
    return pd.concat(vals, ignore_index=True), pd.concat(tests, ignore_index=True)


# ---------------------------------------------------------------- поправка Б0.1 и метрики
def s2_est(code, E, st):
    e = np.concatenate([E[j] for j in range(max(1, st - 2) if code in ('sm', 'smrob') else st, st + 1) if j in E])
    if code in ('rob', 'smrob'):
        return float((1.4826 * np.median(np.abs(e - np.median(e)))) ** 2)
    return float(np.var(e))


def add_bias(val, test):
    """rv_pred_{code}: val шага k - F по оценке шага k-1 (шаг 1: F=1), тест шага k - F по оценке шага k. Остатки - val, rv>1e-12."""
    v = val[val.rv_actual > 1e-12]
    E = {st: np.log(g.rv_actual.values + 1e-10) - np.log(g.rv_pred.values) for st, g in v.groupby('step')}
    for code in CODES:
        Fv = {st: (np.exp(s2_est(code, E, st - 1) / 2) if st > 1 else 1.0) for st in val.step.unique()}
        Ft = {st: np.exp(s2_est(code, E, st) / 2) for st in test.step.unique()}
        fv, ft = val.step.map(Fv), test.step.map(Ft)
        val[f'rv_pred_{code}'] = val.rv_pred * fv
        test[f'rv_pred_{code}'] = test.rv_pred * ft
    return val, test


def qlike(y, p):
    r = y / np.clip(p, 1e-12, None)
    return float(np.mean(r - np.log(r) - 1))


def dev_table(val, col, lo=2, hi=6):
    v = val[(val.step >= lo) & (val.step <= hi) & (val.rv_actual > 1e-12)]
    return {int(st): qlike(g.rv_actual.values, g[col].values) for st, g in v.groupby('step')}


def load(feat=FEAT):
    global GH_HARJ
    A, harj = GH.load_pit(feat)
    GH_HARJ = harj
    return A, harj


# ---------------------------------------------------------------- тест «будущее не влияет»
def leak_test(A, names, cuts=('2019-06-30', '2021-12-31', '2023-06-30'), seed=20261026):
    """Числовые входы строк date > t заменяются бутстрапом значений столбца до t; слагаемые на строках date <= t должны совпасть побайтно.
    ln_sig1 читает внешний OOS-файл (по построению шага), его строки > t тоже заменяются шумом."""
    rng = np.random.default_rng(seed)
    base = build_terms(A, names)
    rows = []
    for t in cuts:
        t = pd.Timestamp(t)
        B = A.copy()
        fut = B.date > t
        for c in B.columns:
            if c in ('date', 'ticker', 'year') or not np.issubdtype(B[c].dtype, np.number):
                continue
            pool = B.loc[~fut, c].dropna().values
            if len(pool):
                B.loc[fut, c] = rng.choice(pool, fut.sum())
        saved = _SIG.pop('v', None)
        if saved is not None:  # будущие строки σ̂ -> шум
            s = saved.copy(); f = s.date > t
            s.loc[f, 'dev'] = rng.normal(0, 1, f.sum())
            _SIG['v'] = s
        X2 = build_terms(B, names)
        _SIG['v'] = saved
        past = ~fut
        for n in names:
            a1, a2 = base.loc[past, n].values, X2.loc[past, n].values
            rows.append(dict(t=str(t.date()), term=n, n=int(past.sum()), mismatch=int((~((a1 == a2) | (np.isnan(a1) & np.isnan(a2)))).sum())))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--h', type=int, nargs='*', default=[1, 5, 22])
    ap.add_argument('--terms', nargs='*', default=None, help='по умолчанию - BASE_TERMS (для h=1 без ln_sig1)')
    ap.add_argument('--extra', nargs='*', default=[], help='добавить к умолчанию (Д17)')
    ap.add_argument('--tag', default='HARX')
    ap.add_argument('--leak-test', action='store_true')
    ap.add_argument('--no-grid', action='store_true', help='OLS без штрафа (alpha=0)')
    ap.add_argument('--no-d17', action='store_true', help='без кандидатов Д17.3 (HARX0)')
    ap.add_argument('--inner', type=int, default=1, help='внутренних валидационных лет для alpha/l1 (рычаг; по умолчанию 1)')
    a = ap.parse_args()
    t0 = time.time()
    A, harj = load()
    global INNER
    INNER = a.inner
    (REGEN / 'D16').mkdir(exist_ok=True)
    if a.leak_test:
        names = list(TERMS)
        r = leak_test(A, names)
        print(r.to_string()); r.to_csv(REGEN / 'D16' / 'leak_test_harx.csv', index=False)
        print('всего расхождений:', int(r.mismatch.sum())); return
    (REGEN / 'test_locked' / a.tag).mkdir(parents=True, exist_ok=True)
    allt = list(dict.fromkeys(BASE_TERMS + sum(D17_TERMS.values(), []) + a.extra + (a.terms or [])))
    Xall = build_terms(A, allt)
    summ = []
    for h in a.h:
        terms = (a.terms if a.terms is not None else default_terms(h, not a.no_d17)) + [e for e in a.extra if e not in (a.terms or [])]
        if h == 1:
            terms = [t for t in terms if t != 'ln_sig1']
        log = []
        val, test = harx_forecasts(h, A, harj, terms, Xall, log, grid=not a.no_grid)
        val, test = add_bias(val, test)
        val.to_parquet(REGEN / f'{a.tag}_h{h}_val.parquet', index=False)
        test.to_parquet(REGEN / 'test_locked' / a.tag / f'{a.tag}_h{h}_test.parquet', index=False)
        pd.DataFrame(log).to_csv(REGEN / 'D16' / f'{a.tag}_h{h}_coefs.csv', index=False)
        hv, ht, _ = har_steps(h, A, harj)
        hv, ht = add_bias(hv, ht)
        for code in ('sm', 'smrob'):
            qx, qh = dev_table(val, f'rv_pred_{code}'), dev_table(hv, f'rv_pred_{code}')
            summ.append(dict(h=h, code=code, terms=' '.join(terms), **{f'harx_{s}': qx[s] for s in qx}, **{f'harj_{s}': qh[s] for s in qh},
                             harx_mean=np.mean([qx[s] for s in range(2, 6)]), harj_mean=np.mean([qh[s] for s in range(2, 6)]),
                             harx_2021=qx[6], harj_2021=qh[6]))
        print(f'h={h} готов ({time.time() - t0:.0f} c)', flush=True)
    S = pd.DataFrame(summ)
    S['ratio_val17_20'] = S.harx_mean / S.harj_mean
    S['ratio_2021'] = S.harx_2021 / S.harj_2021
    S.to_csv(REGEN / 'D16' / f'{a.tag}_summary.csv', index=False)
    print(S[['h', 'code', 'harx_mean', 'harj_mean', 'ratio_val17_20', 'harx_2021', 'harj_2021', 'ratio_2021']].to_string())


if __name__ == '__main__':
    main()
