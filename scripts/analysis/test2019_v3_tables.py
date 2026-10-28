"""Таблицы 3.4–3.6 теста 2019 на features_pit_v3 (исходное разбиение <= 2017 / 2018 / 2019) из data/regen/test2019_v3/.
Метрики как final_test.py: QLIKE = mean(y/p − ln(y/p) − 1), RMSE и MAE на уровне RV; фильтр rv_actual > 1e-12.
DM (потери QLIKE, d = L_эталон − L_модель, t > 0 — модель лучше): номинальный (пул, без HAC); NW лаг h−1 на пуле (дата, тикер);
основной — дневной ряд средней по тикерам разности потерь, NW лаг h−1. Эталоны: HAR-J как в диссертации и HAR-J с Б0.1.
LSTM/GRU (test2019_v3_nn.py) и GARCH — если файлы есть; иначе строки пустые. Строки «(диссертация)» — старые числа текста (правки_5)."""
from pathlib import Path
import numpy as np, pandas as pd
from final_test import qloss, metrics, nw_var
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
TAB = ROOT / 'docs/tables'
HS = (1, 5, 22)
OLD = {'XGBoost': (.277, .396, .470), 'LightGBM': (.290, .382, .455), 'HAR-J': (.305, .424, .467), 'Гибридная': (.276, .373, .441),
       'LSTM': (.372, .435, .485), 'GRU': (.399, .411, .503), 'GJR-GARCH': (.501, .567, .601)}
COLS = {'Гибридная': 'hyb', 'Гибридная с Б0.1': 'hyb_B01', 'XGBoost': 'xgb', 'LightGBM': 'lgb', 'HAR-J': 'har', 'HAR-J с Б0.1': 'har_B01',
        'XGBoost с Б0.1': 'xgb_B01', 'LightGBM с Б0.1': 'lgb_B01', 'LSTM': 'lstm', 'GRU': 'gru', 'GJR-GARCH': 'gjr', 'GARCH(1,1)': 'garch',
        'GJR-GARCH с масштабом': 'gjr_sc', 'GARCH(1,1) с масштабом': 'garch_sc'}
REFS = {'HAR-J как в диссертации': 'har', 'HAR-J с Б0.1': 'har_B01'}


def build(src, h):
    k = ['date', 'ticker']
    rd = lambda n: pd.read_parquet(src / f'{n}_h{h}.parquet')
    t = lambda d: d[d.part == 'test']
    df = t(rd('hyb'))
    for n, c in (('har', 'har'), ('xgb', 'xgb'), ('lgbm', 'lgb')):
        df = df.merge(t(rd(n))[k + ['rv_pred']].rename(columns={'rv_pred': c}), on=k, how='left')
    if (src / f'garch_h{h}.parquet').exists():
        g = t(rd('garch'))
        for m in ('garch', 'gjr'):
            gm = g[g.model == m][k + ['rv_pred', 'rv_pred_scaled']].rename(columns={'rv_pred': m, 'rv_pred_scaled': m + '_sc'})
            df = df.merge(gm, on=k, how='left')
    for n in ('lstm', 'gru'):
        if (src / f'{n}_h{h}.parquet').exists():
            df = df.merge(t(rd(n))[k + ['rv_pred']].rename(columns={'rv_pred': n}), on=k, how='left')
    return df[df.rv_actual > 1e-12].sort_values(k).reset_index(drop=True)


def dm(df, ref, col, h):
    s = df[df[col].notna() & df[ref].notna()]
    d = qloss(s.rv_actual.values, s[ref].values) - qloss(s.rv_actual.values, s[col].values)
    daily = pd.Series(d, index=s.date.values).groupby(level=0).mean().sort_index().values
    out = [('номинальный', len(d), d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))), (f'NW лаг {h - 1} на пуле', len(d), d.mean() / np.sqrt(nw_var(d, h - 1))),
           (f'основной (дневной ряд, NW лаг {h - 1})', len(daily), daily.mean() / np.sqrt(nw_var(daily, h - 1)))]
    return [dict(способ=m, n=n, t=round(float(t), 2), p_two=float(2 * stats.norm.sf(abs(t))), mean_d=float(d.mean())) for m, n, t in out]


def r4(v): return np.nan if v is None or not np.isfinite(v) else round(float(v), 4)
def pct(a, b): return '' if not (np.isfinite(a) and np.isfinite(b)) else f'{100 * (a / b - 1):+.0f}%'.replace('+0%', '0%').replace('-0%', '0%')


def run(src):
    src = Path(src); M, DM = {}, []
    for h in HS:
        df = build(src, h)
        for m, c in COLS.items():
            if c in df and df[c].notna().any():
                s = df[df[c].notna()]
                M[(m, h)] = dict(**metrics(s.rv_actual.values, s[c].values), n=len(s))
                if m not in REFS.values():
                    for rn, rc in REFS.items():
                        if c != rc:
                            DM += [dict(H=h, модель=m, эталон=rn, **r) for r in dm(df, rc, c, h)]
    q = lambda m, h, k='QLIKE': M[(m, h)][k] if (m, h) in M else np.nan
    old = lambda m: {f'H = {h}': v for h, v in zip(HS, OLD[m])}
    T4 = [{'Модель': m, **{f'H = {h}': r4(q(m, h)) for h in HS}} for m in ('XGBoost', 'LightGBM', 'HAR-J', 'HAR-J с Б0.1', 'XGBoost с Б0.1', 'LightGBM с Б0.1')]
    T4 += [{'Модель': f'{m} (диссертация)', **old(m)} for m in ('XGBoost', 'LightGBM', 'HAR-J')]
    r5 = lambda m, x, hj: {'Модель': m, 'QLIKE H = 1': r4(x), 'Δ vs XGBoost': '-' if m.startswith('XGBoost') else pct(x, q('XGBoost', 1) if '(дисс' not in m else OLD['XGBoost'][0]),
                           'Δ vs HAR-J': '-' if m.startswith('HAR-J') else pct(x, hj)}
    T5 = [r5(m, q(m, 1), q('HAR-J', 1)) for m in ('XGBoost', 'LightGBM', 'HAR-J', 'LSTM', 'GRU')]
    T5 += [r5(f'{m} (диссертация)', OLD[m][0], OLD['HAR-J'][0]) for m in ('XGBoost', 'LightGBM', 'HAR-J', 'LSTM', 'GRU')]
    T6 = [{'Модель': m, **{f'H = {h}': r4(q(m, h)) for h in HS}} for m in ('Гибридная', 'LightGBM', 'XGBoost', 'HAR-J', 'LSTM', 'GRU', 'GJR-GARCH',
                                                                         'Гибридная с Б0.1', 'HAR-J с Б0.1', 'GARCH(1,1)', 'GJR-GARCH с масштабом')]
    T6 += [{'Модель': f'{m} (диссертация)', **old(m)} for m in ('Гибридная', 'LightGBM', 'XGBoost', 'HAR-J', 'LSTM', 'GRU', 'GJR-GARCH')]
    TAB.mkdir(parents=True, exist_ok=True)
    for n, T in (('3_4', T4), ('3_5', T5), ('3_6', T6)):
        pd.DataFrame(T).to_csv(TAB / f'table_{n}_test2019_v3.csv', index=False)
    pd.DataFrame([dict(Модель=m, H=h, **v) for (m, h), v in M.items()]).to_csv(TAB / 'table_3_x_test2019_v3_metrics.csv', index=False)
    D = pd.DataFrame(DM)
    if len(D):
        D['p_two'] = D.p_two.map(lambda p: '<0,001' if p < 1e-3 else round(p, 3))
    D.to_csv(TAB / 'table_3_x_test2019_v3_dm.csv', index=False)
    print(pd.DataFrame(T6).to_string(), flush=True)
    return M, D
