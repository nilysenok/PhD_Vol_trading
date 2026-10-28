"""Статистики протокола теста главы 3: DM по дневному ряду, MCS, охват (HLN), Минцер-Зарновиц (HAC). Данные не читает."""
import numpy as np, pandas as pd
from scipy import stats
import warnings
warnings.filterwarnings('ignore', message='.*encountered in matmul')  # ложные флаги FP у BLAS (Accelerate, macOS); результаты проверены selftest


def qloss(y, p):
    r = y / p
    return r - np.log(r) - 1


def nw_var(d, lag):
    """Долгосрочная дисперсия среднего (Ньюи-Уэст, Бартлетт): (g0 + 2 sum (1 - k/(L+1)) g_k) / n."""
    d = np.asarray(d, float) - np.mean(d)
    n = len(d)
    g = np.sum(d * d) / n
    for k in range(1, lag + 1):
        g += 2 * (1 - k / (lag + 1)) * np.sum(d[k:] * d[:-k]) / n
    return g / n


def daily(dates, v):
    """Дневной ряд средней по тикерам, упорядочен по дате."""
    return pd.Series(np.asarray(v, float), index=np.asarray(dates)).groupby(level=0).mean().sort_index().values


def dm_series(d, h):
    """DM по ряду d (d > 0: финальная лучше): t, одностороннее p (H1: d > 0), среднее, n. NW лаг h-1."""
    d = np.asarray(d, float)
    v = nw_var(d, h - 1) if len(d) > h else np.nan
    t = d.mean() / np.sqrt(v) if v > 1e-300 else 0.0  # nan > x -> False: t = 0
    return dict(t=float(t), p_one=float(stats.norm.sf(t)), mean_d=float(d.mean()), n=len(d))


def dm_main(df, ref, h, pred='rv_pred'):
    """Основной DM: дневной ряд средней по тикерам разности потерь QLIKE (эталон - модель), NW лаг h-1."""
    d = qloss(df.rv_actual.values, df[ref].values) - qloss(df.rv_actual.values, df[pred].values)
    return dm_series(daily(df.date.values, d), h)


def mcs(L, names, h, B=10000, stat='R', seed=0, block=None):
    """Model Confidence Set (Hansen-Lunde-Nason 2011), блочный бутстрап (скользящие блоки, циклически). L: T x M потерь.
    stat='R' (T_R) или 'max' (T_max). Возвращает {имя: p_MCS}; модель в MCS_alpha, если p_MCS >= alpha. Длина блока - max(10, h)."""
    L = np.asarray(L, float)
    T, M = L.shape
    b = block or max(10, h)
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(T / b))
    cs = np.vstack([np.zeros(M), np.cumsum(np.vstack([L, L[:b]]), 0)])
    st = rng.integers(0, T, (B, nb))
    bm = sum(cs[st[:, j] + b] - cs[st[:, j]] for j in range(nb)) / (nb * b)  # B x M: бутстрап-средние потерь
    lbar, act, pm, pmax = L.mean(0), list(range(M)), {}, 0.0
    while len(act) > 1:
        sub, lb = lbar[act], bm[:, act]
        dij = sub[:, None] - sub[None, :]
        dbs = lb[:, :, None] - lb[:, None, :]  # B x m x m
        if stat == 'R':
            se = dbs.std(0)
            se = np.where(se > 1e-14, se, np.inf)
            tij = dij / se
            s, null = np.abs(tij).max(), (np.abs(dbs - dij) / se).max((1, 2))
            drop = int(tij.max(1).argmax())
        else:
            di, dbi = dij.mean(1), dbs.mean(2)
            se = dbi.std(0)
            se = np.where(se > 1e-14, se, np.inf)
            ti = di / se
            s, null = ti.max(), ((dbi - di) / se).max(1)
            drop = int(ti.argmax())
        pmax = max(pmax, float((null >= s).mean()))
        pm[names[act[drop]]] = pmax
        act.pop(drop)
    pm[names[act[0]]] = 1.0
    return pm


def hln(eA, eB, dates, h):
    """Охват (Harvey-Leybourne-Newbold 1998): H0 - A охватывает B. d = e_A(e_A - e_B), дневное среднее по тикерам, NW лаг h-1,
    поправка HLN, t(n-1), одностороннее p (H1: mean d > 0, B добавляет информацию). lam - вес B в оптимальной комбинации."""
    eA, eB = np.asarray(eA, float), np.asarray(eB, float)
    d = daily(dates, eA * (eA - eB))
    n = len(d)
    v = nw_var(d, h - 1)
    c = np.sqrt(max((n + 1 - 2 * h + h * (h - 1) / n) / n, 1e-12))
    t = c * d.mean() / np.sqrt(v) if v > 1e-300 else 0.0
    lam = float((eA * (eA - eB)).mean() / ((eA - eB) ** 2).mean())
    return dict(lam=lam, t=float(t), p_one=float(stats.t.sf(t, n - 1)), n=n)


def mz(y, p, dates, h):
    """Минцер-Зарновиц y = a + b p: МНК на пуле, HAC типа Дрисколла-Крея (скоры суммируются по тикерам внутри дня,
    NW лаг h-1 по дням). Wald-тест a = 0, b = 1 (хи-квадрат, 2 ст. св.)."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    X = np.column_stack([np.ones_like(p), p])
    XtXi = np.linalg.inv(X.T @ X)
    beta = XtXi @ X.T @ y
    e = y - X @ beta
    s = pd.DataFrame(X * e[:, None], index=np.asarray(dates)).groupby(level=0).sum().sort_index().values
    lag = h - 1
    S = s.T @ s
    for k in range(1, lag + 1):
        G = s[k:].T @ s[:-k]
        S += (1 - k / (lag + 1)) * (G + G.T)
    V = XtXi @ S @ XtXi
    r = beta - np.array([0.0, 1.0])
    W = float(r @ np.linalg.solve(V, r))
    return dict(alpha=float(beta[0]), se_a=float(np.sqrt(V[0, 0])), beta=float(beta[1]), se_b=float(np.sqrt(V[1, 1])),
                R2=float(1 - e @ e / ((y - y.mean()) ** 2).sum()), wald=W, p_wald=float(stats.chi2.sf(W, 2)), n=len(y), days=len(s))


def vol_terciles(feat, rows, col='rv_m'):
    """Терцили RV_m по расширяющемуся окну: порог на дату t - квантили 1/3 и 2/3 RV_m всех наблюдений матрицы (тикеры rows) с датой <= t
    (без заглядывания вперёд). Возвращает 0/1/2 для строк rows; NaN - если RV_m нет."""
    f = feat[feat.ticker.isin(rows.ticker.unique())][['date', 'ticker', col]].dropna().sort_values('date')
    dts, v = f.date.values, f[col].values
    qs = {t: np.quantile(v[:np.searchsorted(dts, t, side='right')], [1 / 3, 2 / 3]) for t in np.unique(rows.date.values)}
    q = np.vstack([qs[t] for t in rows.date.values])
    x = rows[['date', 'ticker']].merge(f, on=['date', 'ticker'], how='left')[col].values
    return np.where(np.isnan(x), np.nan, (x > q[:, 0]).astype(float) + (x > q[:, 1]))
