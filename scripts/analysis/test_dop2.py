"""Правило данных «закрытие биржи» и знаковый тест по тикерам — поручение владельца 26.10, 09:55 (п. 1 и 3).
Срез по поручению владельца после единственного запуска; правило сформулировано после первого прогона теста.
final_test.py не вызывается, сохранённые таблицы теста не перезаписываются; модели не обучаются, прогнозы не меняются.

Правило: точка отсчёта t тикера исключается из оценки (все модели, все h), если окно цели [t+1, t+h] — следующие h строк тикера
в матрице признаков (так построена цель: rv_actual = среднее rv следующих h строк тикера, m0_features_pit.py) — по календарю задевает
дни остановки торгов MOEX 28.02–23.03.2022, а для тикера, открывшегося позже 24.03.2022, ещё и его неторговые дни до первого торгового дня:
исключить, если t <= C1(тикер) и d_h(t) >= 28.02.2022, где d_h(t) — дата h-й следующей строки тикера, C1 — день перед первым торговым днём
тикера после 23.03.2022. Первый торговый день — по 10-минутным барам: data/iss_check/rv_1850.parquet (n_bars > 0) и
data/iss_check/bars_10m/<тикер>.parquet (объём > 0), обе проверки обязаны совпасть.

Выход (docs/tables/, ';', десятичная запятая, UTF-8 с BOM):
  table_3_zakrytie_isklyucheno.csv — исключённые точки по h × тикер, первый торговый день после 23.03.2022;
  table_3_{vse,kontrol,neotkr}_{svodka,dm,mcs,tickers,subperiods,znak}_test_{bez_zakrytiya,s_zakrytiem}.csv.
  vse — весь тест 2017–2026; kontrol — 2022–2026 (2022–2024 — подтверждающий, использовался для выбора из трёх финалистов);
  neotkr — 2025 – 01/02.2026 (правило там ничего не меняет). znak — доля тикеров с выигрышем и биномиальный знаковый тест
  (иллюстрация: тикеры коррелированы, тест не независимый).
Контроль: версия bez_zakrytiya для kontrol и neotkr совпадает с table_3_{kontrol,neotkr}_*_test.csv (PR #114), для vse — с table_3_9 и table_3_mcs.
python3 scripts/analysis/test_dop2.py
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ftp_tables as FT
from final_test import ETALON, dm as dm3
from test_dop import load, write_named, TABLES

DATA = Path('/Users/nilysenok/Desktop/Старое/PhD_Vol_trading/data')
C0, C1 = pd.Timestamp('2022-02-28'), pd.Timestamp('2022-03-23')
NOTE = 'срез по поручению владельца после единственного запуска; правило сформулировано после первого прогона теста'
PER = {'vse': ('весь тест 2017–2026', 2017, 2026,
               {'2017–2019': (2017, 2019), '2020–2021': (2020, 2021), '2022': (2022, 2022), '2023–2026': (2023, 2026)}),
       'kontrol': ('контрольный 2022–2026 (2022–2024 — подтверждающий, использовался для выбора из трёх финалистов)', 2022, 2026,
                   {'2022': (2022, 2022), '2023–2024': (2023, 2024), '2025–2026': (2025, 2026)}),
       'neotkr': ('неоткрывавшийся 2025–2026 (01–02.2026)', 2025, 2026, {'2025': (2025, 2025), '2026 (01–02)': (2026, 2026)})}
RULE = {'bez_zakrytiya': 'без правила', 's_zakrytiem': 'с правилом «закрытие биржи»'}


def first_trade():
    r = pd.read_parquet(DATA / 'iss_check/rv_1850.parquet', columns=['date', 'ticker', 'n_bars'])
    r['date'] = pd.to_datetime(r['date'])
    out = {}
    for tk in sorted(FT_TICKERS):
        a = r[(r.ticker == tk) & (r.date > C1) & (r.n_bars > 0)].date.min()
        b = pd.read_parquet(DATA / f'iss_check/bars_10m/{tk}.parquet', columns=['begin', 'volume'])
        b = b[(b.begin > C1 + pd.Timedelta(days=1)) & (b.volume > 0)].begin.min().normalize()
        assert a == b, f'{tk}: первый торговый день по rv_1850 {a} и по барам {b} не совпали'
        last = r[(r.ticker == tk) & (r.date < C0) & (r.n_bars > 0)].date.max()
        out[tk] = (last, a)
    return out


def masks(D, ft):
    """{h: bool-маска исключения по строкам D[h]} и таблица исключённых точек."""
    M = pd.read_parquet(DATA / 'regen/features_pit_v3.parquet', columns=['date', 'ticker'])
    M['date'] = pd.to_datetime(M['date'])
    cal = {tk: np.sort(g.date.values) for tk, g in M.groupby('ticker')}
    ex, rows = {}, []
    for h, df in D.items():
        m = np.zeros(len(df), bool)
        for tk, g in df.groupby('ticker'):
            c = cal[tk]
            pos = np.searchsorted(c, g.date.values)
            assert (c[pos] == g.date.values).all(), f'{tk}: дата прогноза не найдена в матрице'
            dh = np.where(pos + h < len(c), c[np.minimum(pos + h, len(c) - 1)], np.datetime64('NaT'))
            c1 = ft[tk][1] - pd.Timedelta(days=1)
            e = (g.date.values <= np.datetime64(c1)) & (dh >= np.datetime64(C0))
            m[g.index[e]] = True
            d = g.date[e]
            rows.append({'h': h, 'Тикер': tk, 'исключено точек': int(e.sum()),
                         'первая исключённая t': d.min().date() if len(d) else '', 'последняя исключённая t': d.max().date() if len(d) else '',
                         'последний торговый день до 28.02.2022': ft[tk][0].date(), 'первый торговый день после 23.03.2022': ft[tk][1].date(),
                         'открылся 24.03.2022': 'да' if ft[tk][1] == pd.Timestamp('2022-03-24') else 'нет'})
        ex[h] = m
    T = pd.DataFrame(rows)
    tot = T.groupby('h', as_index=False)['исключено точек'].sum().assign(Тикер='ВСЕГО')
    return ex, pd.concat([T, tot], ignore_index=True).sort_values(['h'], kind='stable')


def dm_rows(D, label, rule, h1):
    rows = []
    for h, df in D.items():
        for rn, rc in FT.REFS.items():
            with np.errstate(invalid='ignore', divide='ignore'):
                for r in dm3(df, rc, h):
                    ok = r['p_one_H1'] == r['p_one_H1']
                    rows.append({'Период': label, 'правило': rule, 'h': h, 'эталон': rn, 'способ': r['способ'], 'n': r['n'], 't': round(r['t'], 2),
                                 'p двусторонний': FT.pf(r['p_two']) if ok else np.nan, 'p односторонний (H1: финал лучше)': FT.pf(r['p_one_H1']) if ok else np.nan,
                                 'средняя разность потерь': r['mean_d'],
                                 'H1 принята 5% (основной, против HAR-J (б))': (int(rn == 'HAR-J (б)' and r['способ'].startswith('основной') and r['p_one_H1'] < 0.05)
                                                                               if h1 else '')})
    return pd.DataFrame(rows)


def znak(D, label, rule):
    rows = []
    for h, df in D.items():
        for rn, rc in FT.REFS.items():
            q = pd.DataFrame([{'f': FT._q(g, 'rv_pred'), 'e': FT._q(g, rc)} for _, g in df.groupby('ticker')])
            k, l = int((q.f < q.e).sum()), int((q.f > q.e).sum())
            n = k + l
            p2 = stats.binomtest(k, n, 0.5).pvalue if n else np.nan
            p1 = stats.binomtest(k, n, 0.5, alternative='greater').pvalue if n else np.nan
            rows.append({'Период': label, 'правило': rule, 'h': h, 'эталон': rn, 'тикеров': len(q), 'финал лучше': k, 'финал хуже': l, 'равно': len(q) - n,
                         'доля с выигрышем': round(k / len(q), 3), 'p знаковый двусторонний': FT.pf(p2) if n else np.nan,
                         'p знаковый односторонний (финал лучше)': FT.pf(p1) if n else np.nan,
                         'оговорка': 'иллюстрация: тикеры коррелированы, тест не независимый' + ('' if n else '; финал = эталон')})
    return pd.DataFrame(rows)


def svodka(D, label, rule, mc, h1, zn):
    rows = []
    for h, df in D.items():
        q = {m: FT._q(df, c) for m, c in FT.MODS.items()}
        r = {'Период': label, 'правило': rule, 'h': h, 'n': len(df), 'дней': df.date.nunique(), **{f'QLIKE {m}': round(v, 4) for m, v in q.items()},
             'финал к HAR-J (б), %': round(100 * (q['Финальная'] / q['HAR-J (б)'] - 1), 1),
             'финал к HAR-J (а), %': round(100 * (q['Финальная'] / q['HAR-J (а)'] - 1), 1), **FT._dm_cols(df, h)}
        if h == 22:
            r['DM t vs HAR-J (б)'], r['p vs HAR-J (б)'] = np.nan, 'финал = эталон'
        for m in FT.MODS:
            r[f'p_MCS T_R {m}'] = mc[(mc.h == h) & (mc['Модель'] == m)]['p_MCS T_R'].iloc[0]
        z = zn[(zn.h == h) & (zn['эталон'] == 'HAR-J (б)')].iloc[0]
        r['тикеров с выигрышем (против (б))'] = f"{z['финал лучше']} из {z['тикеров']}"
        r['знаковый p одностор. (против (б))'] = z['p знаковый односторонний (финал лучше)']
        r['H1 (основной DM против HAR-J (б), 5%)'] = ('не проверяема (финал = эталон)' if h == 22 else
                                                     ('принята' if FT.S.dm_main(df, ETALON, h)['p_one'] < 0.05 else 'не принята')) if h1 else ''
        r['пометка'] = NOTE
        rows.append(r)
    return pd.DataFrame(rows)


def control(tag, T):
    """bez_zakrytiya против сохранённых таблиц PR #114 (kontrol, neotkr) и запуска (vse)."""
    if tag in ('kontrol', 'neotkr'):
        for part in ('dm', 'mcs', 'tickers', 'subperiods'):
            old = pd.read_csv(TABLES / f'table_3_{tag}_{part}_test.csv', sep=';', decimal=',', encoding='utf-8-sig')
            new = T[part]
            col = {'dm': 't', 'mcs': 'p_MCS T_R', 'tickers': 'QLIKE финальной', 'subperiods': 'QLIKE Финальная'}[part]
            a, b = pd.to_numeric(old[col], errors='coerce').values, pd.to_numeric(new[col], errors='coerce').values
            assert np.allclose(a, b, equal_nan=True), f'контроль {tag} {part}'
    else:
        t9 = pd.read_csv(TABLES / 'table_3_9_test_raw.csv')
        assert np.allclose(t9.t.values, T['dm'].t.values, equal_nan=True), 'контроль vse dm'
        old = pd.read_csv(TABLES / 'table_3_mcs_test_raw.csv', sep=';', decimal=',', encoding='utf-8-sig')
        assert np.allclose(old['p_MCS T_R'].values, T['mcs']['p_MCS T_R'].values), 'контроль vse mcs'
    print(f'контроль {tag} без правила = сохранённые таблицы — OK')


FT_TICKERS = ()


def main():
    global FT_TICKERS
    FT.set_etalon(ETALON)
    D = {h: load(h) for h in (1, 5, 22)}
    FT_TICKERS = tuple(sorted(D[1].ticker.unique()))
    assert len(FT_TICKERS) == 17
    ft = first_trade()
    ex, X = masks(D, ft)
    write_named(X.assign(пометка=NOTE), 'table_3_zakrytie_isklyucheno.csv')
    for rule, rlab in RULE.items():
        DR = {h: (df[~ex[h]] if rule == 's_zakrytiem' else df).reset_index(drop=True) for h, df in D.items()}
        for tag, (label, a, b, sub) in PER.items():
            P = {h: df[df.year.between(a, b)].reset_index(drop=True) for h, df in DR.items()}
            h1 = tag == 'kontrol'
            T = {'dm': dm_rows(P, label, rlab, h1), 'mcs': FT.mcs_table(P, 10000).assign(Период=label, правило=rlab),
                 'tickers': FT.tickers_table(P).assign(Период=label, правило=rlab),
                 'subperiods': pd.DataFrame([FT._block(h, nm, 'Подпериод', df[df.year.between(x, y)]) for h, df in P.items()
                                             for nm, (x, y) in sub.items()]).assign(Период=label, правило=rlab),
                 'znak': znak(P, label, rlab)}
            T['svodka'] = svodka(P, label, rlab, T['mcs'], h1, T['znak'])
            if rule == 'bez_zakrytiya':
                control(tag, T)
            for part, df in T.items():
                write_named(df, f'table_3_{tag}_{part}_test_{rule}.csv')


if __name__ == '__main__':
    main()
