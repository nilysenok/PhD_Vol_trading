#!/usr/bin/env python3
"""Таблицы 3.4–3.9 v2 для текста главы 3 — из таблиц второго открытия теста (R2, 26.10 21:36) и справочных GRU/LSTM (PR #228).
Агент A2, 26.10.2026. Тест и модели не запускаются: читаются только CSV в docs/tables (test_locked не читается).

python3 scripts/analysis/tables_v2_r2.py            # пишет docs/tables/table_3_{4..9}_v2.csv и сверяет их с источниками

Основная версия: d0, контрольный период 2022–2026, без правила «закрытие биржи». Формат — как table_3_{4..9}_test.csv: «;», десятичная
запятая, UTF-8 с BOM; под таблицей строки «Примечание» и «Источник». Числа QLIKE, t, p, n переносятся строкой из источника (точка → запятая).
Производные (помечены в примечании): табл. 3.5 — Δ, % = (QLIKE строки / QLIKE XGBoost или HAR-J − 1)·100 по значениям таблицы;
табл. 3.7 — простое среднее годовых QLIKE 2022–2026 из table_3_final_po_godam_test_r2.csv (2026 = 01–02.2026 весит как год).
Старые table_3_{4..9}_test.csv не трогаются.
"""
import csv, io, sys
from pathlib import Path
import pandas as pd

REPO = Path(__file__).resolve().parent.parent.parent
T = REPO / 'docs/tables'
BOM = '﻿'
HS = (1, 5, 22)
PER = '2022–2026 (основной)'
OSN = ('Основная версия: второе открытие теста (R2, 26.10.2026 21:36, один запуск final_test_r2.py), все модели в версии d0 '
       '(данные по день d включительно); контрольный период 2022–2026 (2022–2024 — подтверждающий период первого раунда, на нём '
       'выбирали из трёх финалистов; 2025 – 01/02.2026 — подпериод); без правила «закрытие биржи». Финал = F_h(d0) на всех H '
       '(победителей второго раунда нет); при H = 22 финал совпадает с HAR-J с поправкой. «HAR-J» — без поправки, «HAR-J с поправкой» — '
       '× exp(κ·s²), s² по правилу t−1 с эмбарго.')


def c(x):
    """Ячейка источника -> формат владельца (точка -> запятая), NaN -> пусто."""
    s = '' if x is None or (isinstance(x, float) and x != x) else str(x)
    return '' if s == 'nan' else s.replace('.', ',')


def write(name, header, rows, note, src):
    buf = io.StringIO()
    w = csv.writer(buf, delimiter=';', lineterminator='\n')
    w.writerow(header)
    w.writerows(rows)
    w.writerow([f'Примечание. {note}'])
    w.writerow([f'Источник: {src}'])
    (T / name).write_text(BOM + buf.getvalue(), encoding='utf-8')
    return name


def periody():
    p = pd.read_csv(T / 'table_3_periody_test_r2.csv', dtype=str)
    return p[p['Период'] == PER]


def q(p, model, h):
    x = p[(p['Модель'] == model) & (p.h == str(h))]
    assert len(x) == 1, (model, h)
    return x.QLIKE.iloc[0]


def ref_ng():
    r = pd.read_csv(T / 'table_3_ref_ng_test_r2.csv', dtype=str)
    return r[r['Период'] == PER]


def t34(p):
    rows = [[m] + [c(q(p, m, h)) for h in HS] for m in ('XGBoost', 'LightGBM', 'HAR-J')]
    return write('table_3_4_v2.csv', ['Модель', 'H = 1', 'H = 5', 'H = 22'], rows,
                 'QLIKE, пул 2022–2026. «XGBoost», «LightGBM» — слоты ноги финального дизайна без поправки (V0 первого раунда; по ВЫР-2 '
                 'день d видят, не перестраивались). ' + OSN, 'table_3_periody_test_r2.csv (Период «2022–2026 (основной)»)')


def t35(p, r):
    vals = {m: q(p, m, 1) for m in ('XGBoost', 'LightGBM', 'HAR-J')}
    for m in ('LSTM', 'GRU'):
        x = r[(r['Модель'] == m) & (r.h == '1')]
        assert len(x) == 1
        vals[m] = x.QLIKE.iloc[0]
    d = lambda a, b: '—' if a == b else f'{(float(vals[a]) / float(vals[b]) - 1) * 100:+.1f}'.replace('.', ',')
    rows = [[m, c(v), d(m, 'XGBoost'), d(m, 'HAR-J')] for m, v in vals.items()]
    return write('table_3_5_v2.csv', ['Модель', 'QLIKE H = 1', 'Δ vs XGBoost, %', 'Δ vs HAR-J, %'], rows,
                 'QLIKE при H = 1, пул 2022–2026. Δ = (QLIKE строки / QLIKE XGBoost или HAR-J − 1)·100 по значениям таблицы; плюс — хуже. '
                 'LSTM и GRU — справочные (ансамбль seed 42/43/44, Б0.1; test_locked/ref_nn_garch, по ВЫР-2 день d видят, не '
                 'перестраивались), в выборе финала и в H1 не участвуют; QLIKE — scripts/analysis/ref_ng_test_r2.py (PR #228), кадр и '
                 'строки — те же, что у открытия. ' + OSN,
                 'table_3_periody_test_r2.csv (XGBoost, LightGBM, HAR-J); table_3_ref_ng_test_r2.csv (LSTM, GRU)')


def t36(p):
    rows = [[m] + [c(q(p, m, h)) for h in HS] for m in ('Гибридная', 'LightGBM', 'XGBoost', 'HAR-J', 'HAR-J с поправкой')]
    return write('table_3_6_v2.csv', ['Модель', 'H = 1', 'H = 5', 'H = 22'], rows,
                 'QLIKE, пул 2022–2026. «Гибридная» — финал F_h(d0). ' + OSN, 'table_3_periody_test_r2.csv (Период «2022–2026 (основной)»)')


def t37():
    g = pd.read_csv(T / 'table_3_final_po_godam_test_r2.csv', dtype=str)
    g = g[g['Год'].isin([str(y) for y in range(2022, 2027)])]
    rows = []
    for m in ('HAR-J', 'HAR-J с поправкой', 'XGBoost', 'LightGBM', 'Гибридная'):
        row = [m]
        for h in HS:
            x = g[(g['Модель'] == m) & (g.h == str(h))]
            assert len(x) == 5, (m, h)
            row.append(c(round(x.QLIKE.astype(float).mean(), 4)))
        rows.append(row)
    return write('table_3_7_v2.csv', ['Модель', 'H = 1', 'H = 5', 'H = 22'], rows,
                 'QLIKE, простое среднее пяти годовых значений 2022, 2023, 2024, 2025, 2026 (2026 — 01–02.2026, весит как год) из '
                 'table_3_final_po_godam_test_r2.csv; у прежней табл. 3.7 — среднее десяти лет 2017–2026. ' + OSN,
                 'table_3_final_po_godam_test_r2.csv (годы 2022–2026)')


def t38():
    s = pd.read_csv(T / 'table_3_kontrol_svodka_test_r2_bez_zakrytiya.csv', sep=';', encoding='utf-8-sig', dtype=str)
    rows = []
    for h in HS:
        x = s[s.h == str(h)].iloc[0]
        t, pv = x['DM t vs HAR-J (б)'], x['p vs HAR-J (б)']
        rows.append([h, '', ''] if pd.isna(t) else [h, t, pv])
    return write('table_3_8_v2.csv', ['H', 't-стат.', 'p-value'], rows,
                 'Основной DM: дневной ряд средней по 17 тикерам, NW лаг H−1, против HAR-J с поправкой; p одностороннее (H1: финал лучше). '
                 'H1 не принята при H = 1 и 5; при H = 22 не проверяема (финал = эталон, разность потерь тождественно 0). ' + OSN,
                 'table_3_kontrol_svodka_test_r2_bez_zakrytiya.csv')


def t39():
    d = pd.read_csv(T / 'table_3_kontrol_dm_test_r2_bez_zakrytiya.csv', sep=';', encoding='utf-8-sig', dtype=str)
    name = {'HAR-J (а)': 'HAR-J как в диссертации', 'HAR-J (б)': 'HAR-J с поправкой'}
    rows = []
    for h in HS:
        for e in ('HAR-J (а)', 'HAR-J (б)'):
            x = d[(d.h == str(h)) & (d['эталон'] == e)]
            assert len(x) == 3, (h, e)
            for _, r in x.iterrows():
                t = r['t'] if r['t'] == r['t'] else ''
                main = r['способ'].startswith('основной')
                h1 = main and e == 'HAR-J (б)' and r['H1 принята 5% (основной, против HAR-J (б))'] == '1'
                rows.append([h, name[e], r['способ'], r['n'], t, r['p двусторонний'] if t else '',
                             r['p односторонний (H1: финал лучше)'] if t else '', r['средняя разность потерь'], h1])
    return write('table_3_9_v2.csv', ['H', 'эталон', 'способ', 'n', 't', 'p_two', 'p_one_H1', 'mean_d', 'H1_принята_5pct'], rows,
                 'DM тремя способами, как в первом отчёте: номинальный на пуле без HAC; NW на пуле, лаг H−1; основной — дневной ряд средней '
                 'по 17 тикерам, NW лаг H−1. d = QLIKE эталона − QLIKE финала, t > 0 — финал лучше; H1 — только основной способ против HAR-J '
                 'с поправкой, p < 0,05. Дополнительно (по тикерам с Холмом, по непересекающимся точкам) — table_3_dm5_test_r2.csv. ' + OSN,
                 'table_3_kontrol_dm_test_r2_bez_zakrytiya.csv')


def check(names):
    """Сверка: каждая числовая ячейка, перенесённая строкой, есть в источнике; число строк и столбцов."""
    for n in names:
        rows = list(csv.reader(io.StringIO((T / n).read_text(encoding='utf-8')[1:]), delimiter=';'))
        assert rows[-2][0].startswith('Примечание') and rows[-1][0].startswith('Источник'), n
        w = len(rows[0])
        assert all(len(r) == w for r in rows[1:-2]), n
    p = periody()
    for n, models in (('table_3_4_v2.csv', ('XGBoost', 'LightGBM', 'HAR-J')), ('table_3_6_v2.csv', ('Гибридная', 'LightGBM', 'XGBoost', 'HAR-J', 'HAR-J с поправкой'))):
        rows = list(csv.reader(io.StringIO((T / n).read_text(encoding='utf-8')[1:]), delimiter=';'))[1:-2]
        for r in rows:
            assert [float(v.replace(',', '.')) for v in r[1:]] == [float(q(p, r[0], h)) for h in HS], (n, r)
    return len(names)


def main():
    p, r = periody(), ref_ng()
    names = [t34(p), t35(p, r), t36(p), t37(), t38(), t39()]
    k = check(names)
    print(f'записано и сверено: {k} таблиц — ' + ', '.join(names))


if __name__ == '__main__':
    sys.exit(main())
