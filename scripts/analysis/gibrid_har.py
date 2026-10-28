"""HAR-J по шагам walk_forward_full.py (стр. 61-100, 164-205): прогнозы val/test и дисперсия остатков обучающего окна.
Вход: moex_volatility/data/prepared (чтение). Выход: val_df, test_df, s2 (шаг -> дисперсия остатков ln)."""
import json, warnings
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import LinearRegression

warnings.filterwarnings('ignore')  # как walk_forward_full.py:20
ROOT = Path(__file__).resolve().parents[2]
MV = ROOT / 'moex_volatility'


def load_all():
    P = MV / 'data/prepared'
    parts = [pd.read_parquet(P / f'{n}.parquet') for n in ('train', 'val', 'test')]
    for y in range(2020, 2027):
        fp = P / f'walkforward_{y}.parquet'
        if fp.exists():
            d = pd.read_parquet(fp)
            if len(d) > 100:
                parts.append(d)
    a = pd.concat(parts).sort_values('date').reset_index(drop=True)
    a['year'] = pd.to_datetime(a['date']).dt.year
    cfg = json.load(open(P / 'config.json'))
    jump = [c for c in a.columns if 'jump' in c.lower() and 'target' not in c.lower() and 'idx_' not in c.lower()]
    harj = [f for f in cfg['har_features'] + [c for c in jump if c not in cfg['har_features']] if f in a.columns]
    return a, harj


def load_pit(path):
    """Матрица features_pit_v2 (B v2): RV со срезом 18:50. Цель h=1 - rv_target_h1; h=5/22 - rv_target_mean_h{h} (среднее RV за [t+1, t+h],
    уровень; модель в логах) - кладётся в столбец rv_target_h{h}, чтобы процедура шагов HAR-J/HAR-CJ не менялась. Признаки HAR-J - как load_all."""
    a = pd.read_parquet(path).sort_values('date').reset_index(drop=True)
    a['year'] = pd.to_datetime(a['date']).dt.year
    for h in (5, 22):
        a[f'rv_target_h{h}'] = a[f'rv_target_mean_h{h}']
    jump = [c for c in a.columns if 'jump' in c.lower() and 'target' not in c.lower() and 'idx_' not in c.lower()]
    harj = [f for f in ['rv_d', 'rv_w', 'rv_m'] + [c for c in jump if c not in ('rv_d', 'rv_w', 'rv_m')] if f in a.columns]
    return a, harj


def clean(X):  # walk_forward_full.py:51
    return np.nan_to_num(X, nan=0, posinf=0, neginf=0)


def har_forecasts(h, a, harj):
    target = f'rv_target_h{h}'
    vals, tests, s2 = [], [], {}
    for step, ty in enumerate([y for y in range(2017, 2026) if y in a.year.values], 1):
        tr = a[a.year < ty]
        vy = sorted(tr.year.unique())[-1]
        tp = tr[tr.year != vy].dropna(subset=[target])
        vi = tr[tr.year == vy].dropna(subset=[target])
        td = a[a.year == ty].dropna(subset=[target])
        X = lambda d: clean(np.log(d[harj].values.clip(1e-10)))
        y = np.log(tp[target].values + 1e-10)
        m = LinearRegression().fit(X(tp), y)
        s2[step] = float(np.var(y - m.predict(X(tp))))
        for df, lst, yr in ((vi, vals, vy), (td, tests, ty)):
            p = np.clip(np.exp(m.predict(X(df))), 1e-10, None)
            lst.append(pd.DataFrame({'date': df['date'].values, 'ticker': df['ticker'].values, 'year': yr,
                                     'step': step, 'rv_actual': df[target].values, 'rv_pred': p}))
    return pd.concat(vals, ignore_index=True), pd.concat(tests, ignore_index=True), s2


def check_saved(h, test_df):
    """Сверка с сохранёнными har_h{h}_annual.parquet (идентичность/масштаб, не качество)."""
    s = pd.read_parquet(MV / f'data/predictions/walk_forward/har_h{h}_annual.parquet')
    m = test_df.merge(s[['date', 'ticker', 'rv_pred']], on=['date', 'ticker'], suffixes=('', '_saved'))
    r = np.abs(m.rv_pred / m.rv_pred_saved - 1)
    return dict(h=h, n_new=len(test_df), n_saved_2017_25=int((s.year <= 2025).sum()), n_matched=len(m),
                max_rel_diff=float(r.max()), median_rel_diff=float(r.median()),
                corr_log=float(np.corrcoef(np.log(m.rv_pred), np.log(m.rv_pred_saved))[0, 1]))
