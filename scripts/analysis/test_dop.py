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
python3 scripts/analysis/test_dop.py
"""
import hashlib, io, csv, sys
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ftp_tables as FT
from final_test import ETALON
from tables_test_format import conv, to_bytes

ROOT = Path(__file__).resolve().parents[2]
SRC = Path('/Users/nilysenok/Desktop/Старое/PhD_Vol_trading/data/regen/test_locked/final3')
TABLES = ROOT / 'docs/tables'
SHA = {1: '4e56f6252a48cb47f6027a0ff86c1f197102a5e02f61123265287b119cf4e7d0',
       5: '4694991372bd95d3e2a7a091c2548444d77050ca4c75cc40170f5bd78b8652fc',
       22: '91f2a48e5f4ad431ac75c85bdd3dd7a58a449e4e0c9d731fed6dbd0a4937b059'}
GAP = (pd.Timestamp('2022-02-01'), pd.Timestamp('2022-03-31'))


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


def main():
    FT.set_etalon(ETALON)
    D = {h: load(h) for h in (1, 5, 22)}
    write(dm_h1h5(), 'dm_h1h5')
    S = subperiods(D)
    control(S)
    write(S, 'subperiods')


if __name__ == '__main__':
    main()
