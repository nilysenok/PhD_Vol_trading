"""Шум вместо будущего и перестановка цели. Все замены детерминированы (seed), форма/dtype/NaN-маска сохраняются.
Правило цели: rv_target_h на дату s использует RV за [s+1, s+h], поэтому строки календаря с индексом > pos(t)-h
зависят от данных после t и заменяются шумом вместе с признаками строк > pos(t)."""
import zlib
import numpy as np, pandas as pd

SEED = 20251025
KEEP = ('date', 'ticker', 'year', 'step', 'source', 'datetime')


def calendar(df):
    return np.sort(pd.to_datetime(df['date']).unique())


def pos_t(cal, t):
    """Индекс последнего торгового дня календаря <= t."""
    return int(np.searchsorted(cal, np.datetime64(pd.Timestamp(t)), side='right')) - 1


def row_idx(cal, dates):
    return np.searchsorted(cal, pd.to_datetime(dates).values)


def _crc(s):
    return zlib.crc32(str(s).encode()) % 100003


def _resample(out, c, mask, pool, rng):
    vals = pool[rng.integers(0, len(pool), int(mask.sum()))]
    col = out[c].values.copy()
    nan = np.isnan(col) if col.dtype.kind == 'f' else None
    col[mask] = vals.astype(col.dtype)
    if nan is not None:
        col[nan & mask] = np.nan  # маска NaN сохраняется (хвост выборки без цели остаётся без цели)
    out[c] = col


def noise_table(df, t, targets, cal, seed=SEED, keep=KEEP):
    """Числовые столбцы строк с date > t -> бутстрап из значений столбца до t; целевые столбцы (targets: {имя: h}) —
    для строк с индексом календаря > pos(t)-h. Нечисловые столбцы и KEEP не трогаются."""
    out = df.copy()
    p, idx = pos_t(cal, t), row_idx(cal, df['date'])
    for k, c in enumerate(df.columns):
        if c in keep or df[c].dtype.kind not in 'fiu':
            continue
        h = targets.get(c, 0)
        mask, ok = idx > p - h, (idx <= p - h) & df[c].notna().values
        if mask.any() and ok.any():
            _resample(out, c, mask, df[c].values[ok], np.random.default_rng([seed, k]))
    return out


def noise_bars(bars, t, seed=SEED, vol_mult=1.0):
    """Дневные бары одного тикера (open/high/low/close/volume): после t — синтетический лог-блуждающий ряд с
    волатильностью до t; high>=max(o,c), low<=min(o,c); объём — бутстрап."""
    b = bars.sort_values('date').reset_index(drop=True).copy()
    m = (pd.to_datetime(b['date']) > pd.Timestamp(t)).values
    if not m.any() or m.all():
        return b
    rng = np.random.default_rng(seed)
    c0 = b['close'].values.astype(float)
    sd = vol_mult * float(np.nanstd(np.diff(np.log(c0[~m]))))
    cl = c0[~m][-1] * np.exp(np.cumsum(rng.normal(0, sd, m.sum())))
    op = np.r_[c0[~m][-1], cl[:-1]] * np.exp(rng.normal(0, sd / 3, m.sum()))
    hi = np.maximum(op, cl) * (1 + np.abs(rng.normal(0, sd / 2, m.sum())))
    lo = np.minimum(op, cl) * (1 - np.abs(rng.normal(0, sd / 2, m.sum())))
    for c, v in (('close', cl), ('open', op), ('high', hi), ('low', lo)):
        b.loc[m, c] = v
    pool = b.loc[~m, 'volume'].values
    b.loc[m, 'volume'] = pool[rng.integers(0, len(pool), m.sum())]
    return b


def noise_sigma(vp, t, cols=('sigma_h1', 'sigma_h5', 'sigma_h22'), seed=SEED):
    """σ_pred для стратегий: значения после t — бутстрап из значений до t по тикеру."""
    out = vp.copy()
    for tk, g in vp.groupby('ticker'):
        m = (pd.to_datetime(g['date']) > pd.Timestamp(t)).values
        for k, c in enumerate(cols):
            if c in vp.columns and m.any() and (~m).any():
                rng = np.random.default_rng([seed, k, _crc(tk)])
                pool = g[c].values[~m]
                out.loc[g.index[m], c] = pool[rng.integers(0, len(pool), m.sum())]
    return out


def permute_within_ticker(df, cols, seed=42):
    """Случайная перестановка непустых значений столбцов внутри тикера (NaN остаются на местах)."""
    out = df.copy()
    for k, c in enumerate(cols):
        v = out[c].values.copy()
        for tk, ix in out.groupby('ticker').indices.items():
            rng = np.random.default_rng([seed, k, _crc(tk)])
            ok = ix[~np.isnan(v[ix])]
            v[ok] = v[rng.permutation(ok)]
        out[c] = v
    return out
