"""Дополнительные таблицы теста главы 3 по поручению владельца 26.10, 09:05. Расчёт ПОСЛЕ единственного запуска теста
(final_test.py --scope test, 26.10 08:57, коммит 002a74c) из СОХРАНЁННЫХ тестовых прогнозов и функций ftp_*.py. final_test.py не вызывается,
сохранённые таблицы теста не перезаписываются; модели не обучаются, прогнозы не меняются.

Вход: <общий data>/regen/test_locked/final3/final_h{1,5,22}_test.parquet (SHA-256 сверяется с FINAL_DESIGN.md, поле 10), фильтр и сортировка —
как в final_test.run (rv_actual > 1e-12, по date, ticker). Модели и эталоны — ftp_tables.MODS/REFS с эталоном final_test.ETALON (har_corr).
Выход (docs/tables/, ';', десятичная запятая, UTF-8 с BOM):
  table_3_dop_dm_h1h5_test.csv   — DM тремя способами для h=1 и h=5 против обоих эталонов: ПЕРЕКЛАДКА сохранённой table_3_9_test_raw.csv (без счёта);
  table_3_dop_subperiods_test.csv — подпериоды: 2022; 2023–2026; весь тест без февраля–марта 2022 (по дате строки, 01.02–31.03.2022);
                                    справочно — весь тест. Столбцы — как table_3_subperiods_test (ftp_tables._block: QLIKE, основной DM).
Контроль: строка «2022» обязана совпасть с сохранённой table_3_subperiods_test_raw.csv, строка «весь тест» — t основного DM из table_3_9_test_raw.csv.
Поручение 26.10 (после слияния PR #112): терцили RV_m по features_pit_v3 (финалы построены на v3; тест брал по умолчанию features_pit_v2):
  table_3_dop_terciles_v3_test.csv — ftp_stats.vol_terciles + ftp_tables.terciles_table на rv_m из features_pit_v3.parquet (SHA-256 — FINAL, поле 4).
  Контроль: тот же код на features_pit_v2.parquet обязан воспроизвести сохранённую table_3_terciles_test_raw.csv.
Поручение владельца 26.10, 09:15 — срез по поручению владельца после единственного запуска (--part kontrol):
  контрольный период 2022–2026 (2022–2024 — подтверждающий набор, использовался для выбора из трёх финалистов; 2025–2026 не открывались)
  и неоткрывавшийся период 2025 – 01/02.2026 (по году строки; в прогнозах последняя дата 02.02.2026 при h=1, 29.01 при h=5, 12.01 при h=22);
  разработочный 2017–2021 — справочно. Функции — те же, что в тесте: final_test.dm (DM тремя способами), ftp_tables.mcs_table (B = 10 000),
  ftp_tables.tickers_table, ftp_tables._block (подпериоды). H1 — основной DM против HAR-J с поправкой на контрольном периоде.
  Выход: table_3_kontrol_{svodka,dm,mcs,tickers,subperiods}_test.csv, table_3_neotkr_{dm,mcs,tickers,subperiods}_test.csv.
  Контроль: на всём тесте final_test.dm и ftp_tables.mcs_table воспроизводят сохранённые table_3_9_test_raw.csv и table_3_mcs_test_raw.csv.
python3 scripts/analysis/test_dop.py [--part all|terciles_v3|kontrol]
"""
import argparse, hashlib, io, csv, sys
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ftp_tables as FT
from final_test import ETALON, dm as dm3
from tables_test_format import conv, to_bytes

KONTROL = 'контрольный 2022–2026 (2022–2024 — подтверждающий, использовался для выбора из трёх финалистов)'
NEOTKR = 'неоткрывавшийся 2025–2026 (01–02.2026)'
RAZRAB = 'разработочный 2017–2021 (справочно; участвовал в отборе)'
VES = 'весь тест 2017–2026 (справочно; заранее заданный основной в final_test.py)'
PERIODS = {KONTROL: (2022, 2026), NEOTKR: (2025, 2026), RAZRAB: (2017, 2021), VES: (2017, 2026)}
SUB = {KONTROL: {'2022': (2022, 2022), '2023–2024': (2023, 2024), '2025–2026': (2025, 2026)},
       NEOTKR: {'2025': (2025, 2025), '2026 (01–02)': (2026, 2026)}}

ROOT = Path(__file__).resolve().parents[2]
SRC = Path('/Users/nilysenok/Desktop/Старое/PhD_Vol_trading/data/regen/test_locked/final3')
TABLES = ROOT / 'docs/tables'
SHA = {1: '4e56f6252a48cb47f6027a0ff86c1f197102a5e02f61123265287b119cf4e7d0',
       5: '4694991372bd95d3e2a7a091c2548444d77050ca4c75cc40170f5bd78b8652fc',
       22: '91f2a48e5f4ad431ac75c85bdd3dd7a58a449e4e0c9d731fed6dbd0a4937b059'}
GAP = (pd.Timestamp('2022-02-01'), pd.Timestamp('2022-03-31'))
FEAT = {'v2': Path('/Users/nilysenok/Desktop/Старое/PhD_Vol_trading/data/regen/features_pit_v2.parquet'),
        'v3': Path('/Users/nilysenok/Desktop/Старое/PhD_Vol_trading/data/regen/features_pit_v3.parquet')}
SHA_V3 = '8c51bfa1eb6ee862'  # префикс из FINAL_DESIGN.md, поле 4


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def load(h):
    f = SRC / f'final_h{h}_test.parquet'
    assert sha(f) == SHA[h], f'{f.name}: SHA-256 не совпал с FINAL_DESIGN.md'
    df = pd.read_parquet(f)
    return df[df.rv_actual > 1e-12].sort_values(['date', 'ticker']).reset_index(drop=True)


def write(df, name):
    s = io.StringIO()
    df.to_csv(s, index=False, sep=';', decimal=',')
    rows = [[conv(c) for c in r] for r in csv.reader(io.StringIO(s.getvalue(), newline=''), delimiter=';')]
    p = TABLES / f'table_3_dop_{name}_test.csv'
    p.write_bytes(to_bytes(rows))
    print('записано', p, len(df), 'строк')


def dm_h1h5():
    D = pd.read_csv(TABLES / 'table_3_9_test_raw.csv', dtype=str, keep_default_na=False)
    D = D[D.H.isin(['1', '5'])]
    return pd.DataFrame({'h': D.H, 'эталон': D['эталон'], 'способ': D['способ'], 'n': D.n, 't': D.t,
                         'p двусторонний': D.p_two, 'p односторонний (H1: финал лучше)': D.p_one_H1,
                         'средняя разность потерь QLIKE (эталон − финал)': D.mean_d})


def subperiods(D):
    per = {'2022': lambda d: d.year == 2022,
           '2023-2026': lambda d: d.year.between(2023, 2026),
           'весь тест без 02-03.2022': lambda d: ~d.date.between(*GAP),
           'весь тест (справочно)': lambda d: d.year > 0}
    return pd.DataFrame([FT._block(h, nm, 'Период', df[f(df)]) for h, df in D.items() for nm, f in per.items()])


def control(S):
    old = pd.read_csv(TABLES / 'table_3_subperiods_test_raw.csv', sep=';', decimal=',', encoding='utf-8-sig')
    for h in (1, 5, 22):
        a = S[(S.h == h) & (S['Период'] == '2022')].iloc[0]
        b = old[(old.h == h) & (old['Период'] == '2022')].iloc[0]
        for c in [c for c in old.columns if c.startswith('QLIKE') or c.startswith('DM t') or c in ('n', 'дней')]:
            assert np.isclose(float(a[c]), float(b[c])), f'контроль 2022 h={h} {c}: {a[c]} != {b[c]}'
    t9 = pd.read_csv(TABLES / 'table_3_9_test_raw.csv')
    for h in (1, 5, 22):
        a = S[(S.h == h) & (S['Период'] == 'весь тест (справочно)')].iloc[0]
        b = t9[(t9.H == h) & (t9['эталон'] == 'HAR-J как в диссертации') & t9['способ'].str.startswith('основной')].iloc[0]
        assert np.isclose(a['DM t vs HAR-J (а)'], b.t), f'контроль 3.9 h={h}: {a["DM t vs HAR-J (а)"]} != {b.t}'
    print('контроль: 2022 = table_3_subperiods_test, весь тест = table_3_9 (основной DM против HAR-J (а)) — OK')


def terciles(D, ver):
    """Как final_test.run: feat (date, ticker, rv_m), terc = ftp_stats.vol_terciles(feat, df), затем ftp_tables.terciles_table."""
    feat = pd.read_parquet(FEAT[ver], columns=['date', 'ticker', 'rv_m'])
    feat['date'] = pd.to_datetime(feat['date'])
    return FT.terciles_table({h: df.assign(terc=FT.S.vol_terciles(feat, df)) for h, df in D.items()})


def control_terciles(T2):
    old = pd.read_csv(TABLES / 'table_3_terciles_test_raw.csv', sep=';', decimal=',', encoding='utf-8-sig', dtype=str, keep_default_na=False)
    s = io.StringIO()
    T2.to_csv(s, index=False, sep=';', decimal=',')
    new = pd.read_csv(io.StringIO(s.getvalue()), sep=';', dtype=str, keep_default_na=False)
    assert old.shape == new.shape and (old.values == new.values).all(), 'контроль терцилей v2: не воспроизвели table_3_terciles_test_raw.csv'
    print('контроль: терцили по features_pit_v2 = table_3_terciles_test_raw.csv (все ячейки) — OK')


def cut(D, a, b):
    return {h: df[df.year.between(a, b)].reset_index(drop=True) for h, df in D.items()}


def dm_rows(D, label):
    rows = []
    for h, df in D.items():
        for rn, rc in FT.REFS.items():
            with np.errstate(invalid='ignore', divide='ignore'):
                for r in dm3(df, rc, h):
                    rows.append({'Период': label, 'h': h, 'эталон': rn, 'способ': r['способ'], 'n': r['n'], 't': round(r['t'], 2),
                                 'p двусторонний': FT.pf(r['p_two']) if r['p_two'] == r['p_two'] else np.nan,
                                 'p односторонний (H1: финал лучше)': FT.pf(r['p_one_H1']) if r['p_one_H1'] == r['p_one_H1'] else np.nan,
                                 'средняя разность потерь': r['mean_d'],
                                 'H1 принята 5% (основной, против HAR-J (б))': int(rn == 'HAR-J (б)' and r['способ'].startswith('основной')
                                                                                  and r['p_one_H1'] < 0.05) if label == KONTROL else ''})
    return pd.DataFrame(rows)


def svodka(P, M):
    """Основная таблица гипотез: QLIKE (пул), % к эталонам, основной DM против (б) и (а), p_MCS T_R."""
    rows = []
    for label, D in P.items():
        for h, df in D.items():
            q = {m: FT._q(df, c) for m, c in FT.MODS.items()}
            dmc = FT._dm_cols(df, h)
            mc = M.get(label)
            r = {'Период': label, 'h': h, 'n': len(df), 'дней': df.date.nunique(), **{f'QLIKE {m}': round(v, 4) for m, v in q.items()},
                 'финал к HAR-J (б), %': round(100 * (q['Финальная'] / q['HAR-J (б)'] - 1), 1),
                 'финал к HAR-J (а), %': round(100 * (q['Финальная'] / q['HAR-J (а)'] - 1), 1), **dmc}
            if h == 22:
                r['DM t vs HAR-J (б)'], r['p vs HAR-J (б)'] = np.nan, 'финал = эталон'
            for m in FT.MODS:
                r[f'p_MCS T_R {m}'] = mc[(mc.h == h) & (mc['Модель'] == m)]['p_MCS T_R'].iloc[0] if mc is not None else np.nan
            h1 = ''
            if label == KONTROL:
                h1 = 'не проверяема (финал = эталон)' if h == 22 else ('принята' if FT.S.dm_main(df, ETALON, h)['p_one'] < 0.05 else 'не принята')
            r['H1 (основной DM против HAR-J (б), 5%)'] = h1
            rows.append(r)
    return pd.DataFrame(rows)


def control_full(D):
    t9 = pd.read_csv(TABLES / 'table_3_9_test_raw.csv')
    new = dm_rows(D, VES)
    for h in (1, 5):
        for e_old, e_new in (('HAR-J как в диссертации', 'HAR-J (а)'), ('HAR-J с поправкой', 'HAR-J (б)')):
            a = t9[(t9.H == h) & (t9['эталон'] == e_old)].t.values
            b = new[(new.h == h) & (new['эталон'] == e_new)].t.values
            assert np.allclose(a, b), f'контроль DM h={h} {e_old}: {a} != {b}'
    print('контроль: final_test.dm на всём тесте = table_3_9_test_raw.csv — OK')


def control_mcs(M):
    old = pd.read_csv(TABLES / 'table_3_mcs_test_raw.csv', sep=';', decimal=',', encoding='utf-8-sig')
    assert np.allclose(old['p_MCS T_R'].values, M['p_MCS T_R'].values) and np.allclose(old['p_MCS T_max'].values, M['p_MCS T_max'].values), 'контроль MCS'
    print('контроль: ftp_tables.mcs_table на всём тесте = table_3_mcs_test_raw.csv — OK')


def kontrol(D):
    P = {lab: cut(D, a, b) for lab, (a, b) in PERIODS.items()}
    M = {lab: FT.mcs_table(P[lab], 10000) for lab in (KONTROL, NEOTKR, VES)}
    control_full(D)
    control_mcs(M[VES])
    write_named(svodka(P, M), 'table_3_kontrol_svodka_test.csv')
    for lab, tag in ((KONTROL, 'kontrol'), (NEOTKR, 'neotkr')):
        write_named(dm_rows(P[lab], lab), f'table_3_{tag}_dm_test.csv')
        write_named(M[lab].assign(Период=lab), f'table_3_{tag}_mcs_test.csv')
        write_named(FT.tickers_table(P[lab]).assign(Период=lab), f'table_3_{tag}_tickers_test.csv')
        write_named(pd.DataFrame([FT._block(h, nm, 'Подпериод', df[df.year.between(a, b)]) for h, df in P[lab].items()
                                  for nm, (a, b) in SUB[lab].items()]).assign(Период=lab), f'table_3_{tag}_subperiods_test.csv')


def write_named(df, fname):
    s = io.StringIO()
    df.to_csv(s, index=False, sep=';', decimal=',')
    rows = [[conv(c) for c in r] for r in csv.reader(io.StringIO(s.getvalue(), newline=''), delimiter=';')]
    p = TABLES / fname
    p.write_bytes(to_bytes(rows))
    print('записано', p, len(df), 'строк')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--part', choices=['all', 'terciles_v3', 'kontrol'], default='all')
    a = ap.parse_args()
    FT.set_etalon(ETALON)
    D = {h: load(h) for h in (1, 5, 22)}
    if a.part == 'kontrol':
        return kontrol(D)
    if a.part == 'all':
        write(dm_h1h5(), 'dm_h1h5')
        S = subperiods(D)
        control(S)
        write(S, 'subperiods')
    assert sha(FEAT['v3']).startswith(SHA_V3), 'features_pit_v3: SHA-256 не совпал с FINAL_DESIGN.md'
    control_terciles(terciles(D, 'v2'))
    write(terciles(D, 'v3'), 'terciles_v3')


if __name__ == '__main__':
    main()
