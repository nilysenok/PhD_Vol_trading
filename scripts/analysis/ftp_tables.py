"""Таблицы протокола теста главы 3 (MZ, доля тикеров, подпериоды, терцили, MCS, охват). Вход - словарь {h: DataFrame} (как в final_test.run):
date, ticker, year, rv_actual, rv_pred, har_raw, har_corr, <лучшая одиночная>; терцили - ещё столбец terc (0/1/2/NaN).
Новые таблицы: разделитель ';', десятичная запятая, UTF-8 с BOM (Excel); 'p<0,001' - строкой. Прежние таблицы 3.x не менялись."""
import numpy as np, pandas as pd
import ftp_stats as S

BEST = 'lgb_corr'  # лучшая одиночная бустинга: LightGBM-нога с поправкой - лучшая одиночная на разработочном наборе при всех h
MODS = {'Финальная': 'rv_pred', 'HAR-J (а)': 'har_raw', 'HAR-J (б)': 'har_corr', 'Лучшая одиночная бустинга': BEST}
PERIODS = {'2017-2019': (2017, 2019), '2020-2021': (2020, 2021), '2022': (2022, 2022), '2023-2025': (2023, 2025)}
TERC = {0: 'низкая', 1: 'средняя', 2: 'высокая'}
REFS = {'HAR-J (а)': 'har_raw', 'HAR-J (б)': 'har_corr'}


def pf(p):
    return '<0,001' if p < 1e-3 else round(float(p), 3)


def sg(v):
    return float(f'{v:.4g}')


def write(T, tables, tag='test'):
    tables.mkdir(parents=True, exist_ok=True)
    for n, df in T.items():
        df.to_csv(tables / f'table_3_{n}_{tag}.csv', index=False, sep=';', decimal=',', encoding='utf-8-sig')
        print('записано', tables / f'table_3_{n}_{tag}.csv', len(df), 'строк')


def _q(g, c):
    return float(S.qloss(g.rv_actual.values, g[c].values).mean())


def _dm_cols(g, h):
    out = {}
    for rn, rc in REFS.items():
        r = S.dm_main(g, rc, h)
        out[f'DM t vs {rn}'], out[f'p vs {rn}'] = round(r['t'], 2), pf(r['p_one'])
    return out


def mz_table(D):
    rows = []
    for h, df in D.items():
        for scale in ('RV', 'ln RV'):
            for m, c in MODS.items():
                y, p = (df.rv_actual.values, df[c].values) if scale == 'RV' else (np.log(df.rv_actual.values), np.log(df[c].values))
                r = S.mz(y, p, df.date.values, h)
                rows.append({'h': h, 'Шкала': scale, 'Модель': m, 'α': sg(r['alpha']), 'se α': sg(r['se_a']), 'β': round(r['beta'], 3),
                             'se β': round(r['se_b'], 3), 'R²': round(r['R2'], 3), 'Wald α=0, β=1': round(r['wald'], 2), 'p Wald': pf(r['p_wald']), 'дней': r['days']})
    return pd.DataFrame(rows)


def tickers_table(D, alpha=0.05):
    rows = []
    for h, df in D.items():
        sub = []
        for tk, g in df.groupby('ticker'):
            qf, qb = _q(g, 'rv_pred'), _q(g, 'har_corr')
            d = S.qloss(g.rv_actual.values, g.har_corr.values) - S.qloss(g.rv_actual.values, g.rv_pred.values)
            r = S.dm_series(S.daily(g.date.values, d), h)
            sub.append({'h': h, 'Тикер': tk, 'QLIKE финальной': round(qf, 4), 'QLIKE HAR-J (б)': round(qb, 4), 'финальная лучше': int(qf < qb),
                        'DM t': round(r['t'], 2), 'p (финальная лучше)': pf(r['p_one']), 'значимо лучше': int(r['p_one'] < alpha), 'значимо хуже': int(r['p_one'] > 1 - alpha)})
        rows += sub
        k = len(sub)
        rows.append({'h': h, 'Тикер': f'ДОЛЯ из {k}', 'QLIKE финальной': np.nan, 'QLIKE HAR-J (б)': np.nan,
                     'финальная лучше': round(sum(r['финальная лучше'] for r in sub) / k, 3), 'DM t': np.nan, 'p (финальная лучше)': '',
                     'значимо лучше': round(sum(r['значимо лучше'] for r in sub) / k, 3), 'значимо хуже': round(sum(r['значимо хуже'] for r in sub) / k, 3)})
    return pd.DataFrame(rows)


def _block(h, label, key, g):
    return {'h': h, key: label, 'n': len(g), 'дней': g.date.nunique(), **{f'QLIKE {m}': round(_q(g, c), 4) for m, c in MODS.items()}, **_dm_cols(g, h)}


def subperiods_table(D):
    return pd.DataFrame([_block(h, nm, 'Период', df[df.year.between(a, b)]) for h, df in D.items() for nm, (a, b) in PERIODS.items()])


def terciles_table(D):
    return pd.DataFrame([_block(h, nm, 'Терциль RV_m', df[df.terc == k]) for h, df in D.items() for k, nm in TERC.items() if (df.terc == k).any()])


def mcs_table(D, B=10000):
    rows = []
    for h, df in D.items():
        L = np.column_stack([S.daily(df.date.values, S.qloss(df.rv_actual.values, df[c].values)) for c in MODS.values()])
        P = {s: S.mcs(L, list(MODS), h, B=B, stat=s) for s in ('R', 'max')}
        for i, m in enumerate(MODS):
            rows.append({'h': h, 'Модель': m, 'средний дневной QLIKE': round(L[:, i].mean(), 4), 'дней': len(L),
                         **{f'p_MCS T_{s}': round(P[s][m], 3) for s in P},
                         **{f'в MCS α={a} T_{s}': int(P[s][m] >= a) for s in P for a in (0.10, 0.25)}})
    return pd.DataFrame(rows).rename(columns=lambda c: c.replace('0.1', '0,1').replace('0.25', '0,25'))


def encompassing_table(D):
    rows = []
    for h, df in D.items():
        dts = df.date.values
        for scale in ('RV', 'ln RV'):
            f = (lambda x: x) if scale == 'RV' else np.log
            ef, eb = f(df.rv_actual.values) - f(df.rv_pred.values), f(df.rv_actual.values) - f(df.har_corr.values)
            for nm, (a, b) in {'финальная охватывает HAR-J (б)': (ef, eb), 'HAR-J (б) охватывает финальную': (eb, ef)}.items():
                r = S.hln(a, b, dts, h)
                rows.append({'h': h, 'Шкала': scale, 'H0': nm, 'вес второго прогноза λ': round(r['lam'], 3), 't (HLN)': round(r['t'], 2),
                             'p (H1: второй добавляет)': pf(r['p_one']), 'H0 отвергнута 5%': int(r['p_one'] < 0.05), 'дней': r['n']})
    return pd.DataFrame(rows)


def protocol(D, B=10000):
    """Сначала быстрые таблицы, затем MCS и охват. Возвращает словарь имя -> DataFrame."""
    T = {'mz': mz_table(D), 'tickers': tickers_table(D), 'subperiods': subperiods_table(D), 'terciles': terciles_table(D)}
    T['encompassing'] = encompassing_table(D)
    T['mcs'] = mcs_table(D, B)
    return T
