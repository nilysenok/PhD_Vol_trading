"""Таблицы 3.4–3.9 для текста диссертации (поручение владельца 26.10, 09:50, п. 2). Вид строк и столбцов — как в
«Диссертация_Лысенок_правки_5.docx» (табл. 3.4–3.8; 3.9 — DM тремя способами, как table_3_9_test). Основной период — контрольный
2022–2026 (2022–2024 — подтверждающий набор, сноска), рядом — весь тест 2017–2026 и не открывавшиеся 2025 – 01.2026.

СРЕЗ ПО ПОРУЧЕНИЮ ВЛАДЕЛЬЦА ПОСЛЕ ЕДИНСТВЕННОГО ЗАПУСКА ТЕСТА (final_test.py --scope test, 26.10 08:57, коммит 002a74c).
Тест не запускается, модели не обучаются, прогнозы не меняются, сохранённые таблицы теста не перезаписываются.

Источники чисел — только сохранённые CSV теста в docs/tables/ (table_3_*_test.csv, table_3_kontrol_*, table_3_neotkr_*).
Чего в них нет (QLIKE XGBoost/LightGBM без поправки на 2022–2026 и 2025–2026; среднее годовых QLIKE на этих периодах для табл. 3.7),
считается из сохранённых тестовых прогнозов <общий data>/regen/test_locked/final3/final_h{1,5,22}_test.parquet функциями
test_dop.load (SHA-256 = FINAL_DESIGN, поле 10; фильтр rv_actual > 1e-12), test_dop.cut и ftp_tables._q и пишется в
docs/tables/dlya_teksta/istochnik_srez_test.csv. Контроль этого расчёта: те же функции воспроизводят сохранённые
table_3_final_po_godam_test (пул и каждый год), table_3_7_test (среднее годовых) и table_3_kontrol_svodka_test (QLIKE финала и HAR-J, n).

Выход: docs/tables/dlya_teksta/table_3_{4..9}.csv (';', десятичная запятая, UTF-8 с BOM; под таблицей строки «Примечание» и «Источник»),
istochnik_srez_test.csv, README.md (источники колонок и отчёт о сверке). Сверка: каждое число выходной таблицы заново читается из
своего исходного CSV и сравнивается (QLIKE — до 4 знаков, t — до 2, p — до 3); Δ табл. 3.5 пересчитываются из QLIKE той же строки.

python3 scripts/analysis/tables_dlya_teksta.py           # расчёт среза, таблицы, сверка, README
python3 scripts/analysis/tables_dlya_teksta.py --check   # только сверка записанных таблиц (прогнозы не читаются)
"""
import argparse, csv, io, sys
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_dop as TD
import ftp_tables as FT
from tables_test_format import conv, to_bytes

TABLES = TD.TABLES
OUT = TABLES / 'dlya_teksta'
IST = 'dlya_teksta/istochnik_srez_test.csv'
HS = (1, 5, 22)
# (ключ, подпись в шапке, метка периода в CSV test_dop, годы)
PER = [('K', '2022–2026¹', TD.KONTROL, (2022, 2026)),
       ('V', '2017–2026²', TD.VES, (2017, 2026)),
       ('N', '2025 – 01.2026³', TD.NEOTKR, (2025, 2026))]
COLS = {'Гибридная': 'rv_pred', 'HAR-J': 'har_raw', 'HAR-J с поправкой': 'har_corr', 'XGBoost': 'xgb_raw', 'LightGBM': 'lgb_raw'}
SVODKA = {'Гибридная': 'QLIKE Финальная', 'HAR-J': 'QLIKE HAR-J (а)', 'HAR-J с поправкой': 'QLIKE HAR-J (б)'}
DMFILE = {'K': 'table_3_kontrol_dm_test.csv', 'N': 'table_3_neotkr_dm_test.csv'}
ET9 = {'HAR-J как в диссертации': ('HAR-J (а)', 'HAR-J как в диссертации'), 'HAR-J с поправкой (эталон)': ('HAR-J (б)', 'HAR-J с поправкой')}
SRK = 'Срез по поручению владельца после единственного запуска теста (26.10 08:57): столбцы 2022–2026 и 2025 – 01.2026.'
NOTE = ('¹ Контрольный период (основной): годы, не участвовавшие в отборе дизайнов на разработочном наборе 2017–2021; '
        '2022–2024 — подтверждающий набор (3 открытия, использовался для выбора финала из трёх финалистов), 2025 – 01.2026 не открывались. '
        '² Весь тест — заранее заданный в final_test.py основной расчёт (справочно); 2026 — 01.01–03.02. '
        '³ Ни разу не открывавшиеся данные: строки 2025 и 2026 гг. (последняя дата 02.02.2026 при H = 1, 29.01 при H = 5, 12.01 при H = 22).')


# ---------- чтение исходных CSV и запись происхождения ----------
_cache = {}


def rows_of(rel):
    if rel not in _cache:
        t = (TABLES / rel).read_bytes().decode('utf-8-sig')
        _cache[rel] = list(csv.DictReader(io.StringIO(t, newline=''), delimiter=';'))
    return _cache[rel]


def src(rel, where, col):
    m = [r for r in rows_of(rel) if all(r[k] == v for k, v in where.items())]
    assert len(m) == 1, f'{rel} {where}: найдено строк {len(m)}'
    return m[0][col]


def num(s):
    return float(s.replace(',', '.'))


def disp(s, kind):
    """Строка исходного CSV -> вид в таблице. q4: QLIKE 4 знака; t2: t 2 знака; p3: p 3 знака или '<0,001'; пусто -> '—'."""
    if s == '':
        return '—'
    if kind == 'p3' and s.startswith('<'):
        return s
    nd = {'q4': 4, 't2': 2, 'p3': 3}[kind]
    return f'{num(s):.{nd}f}'.replace('.', ',')


class Tab:
    def __init__(self, name, header):
        self.name, self.header, self.rows, self.prov = name, header, [], []

    def cell(self, rel, where, col, kind):
        s = src(rel, where, col)
        self.prov.append((len(self.rows), len(self._cur), rel, where, col, kind))
        self._cur.append(disp(s, kind))

    def lit(self, v):
        self._cur.append(v)

    def start(self, *lits):
        self._cur = list(lits)

    def end(self):
        assert len(self._cur) == len(self.header), (self.name, self._cur)
        self.rows.append(self._cur)


# ---------- срез из сохранённых прогнозов (то, чего нет в CSV) ----------
def srez():
    FT.set_etalon(TD.ETALON)
    D = {h: TD.load(h) for h in HS}
    po = pd.read_csv(TABLES / 'table_3_final_po_godam_test_raw.csv')
    t7 = pd.read_csv(TABLES / 'table_3_7_test_raw.csv')
    sv = rows_of('table_3_kontrol_svodka_test.csv')
    rows, nchk = [], 0
    for key, _, lab, (a, b) in PER:
        P = TD.cut(D, a, b)
        for h, df in P.items():
            years = sorted(int(y) for y in df.year.unique())
            for m, c in COLS.items():
                yq = {y: FT._q(df[df.year == y], c) for y in years}
                pool, mean_y = FT._q(df, c), float(np.mean(list(yq.values())))
                for y, q in yq.items():  # каждый год = сохранённая table_3_final_po_godam_test
                    ref = po[(po.h == h) & (po['Модель'] == m) & (po['Год'] == str(y))].QLIKE.iloc[0]
                    assert np.isclose(q, ref, rtol=1e-12, atol=0), f'год {y} h={h} {m}: {q} != {ref}'
                    nchk += 1
                if key == 'V':
                    ref = po[(po.h == h) & (po['Модель'] == m) & (po['Год'] == 'все годы (пул)')].QLIKE.iloc[0]
                    assert np.isclose(pool, ref, rtol=1e-12, atol=0), f'пул h={h} {m}'
                    ref7 = t7[t7['Модель'] == m][f'H = {h}'].iloc[0]
                    assert round(mean_y, 4) == ref7, f'3.7 h={h} {m}: {mean_y} != {ref7}'
                    nchk += 2
                if key in ('K', 'N') and m in SVODKA:
                    r = [x for x in sv if x['Период'] == lab and x['h'] == str(h)][0]
                    assert round(pool, 4) == num(r[SVODKA[m]]) and len(df) == int(r['n']), f'svodka {key} h={h} {m}'
                    nchk += 2
                rows.append({'Период': lab, 'h': h, 'Модель': m, 'столбец прогноза': c, 'n': len(df),
                             'годы': f'{years[0]}–{years[-1]}', 'QLIKE пул': pool, 'QLIKE среднее по годам': mean_y})
    s = io.StringIO()
    pd.DataFrame(rows).to_csv(s, index=False, sep=';', decimal=',')
    cells = [[conv(c) for c in r] for r in csv.reader(io.StringIO(s.getvalue(), newline=''), delimiter=';')]
    OUT.mkdir(parents=True, exist_ok=True)
    (TABLES / IST).write_bytes(to_bytes(cells))
    _cache.pop(IST, None)
    print(f'срез: {len(rows)} строк -> {IST}; контроль против сохранённых CSV: {nchk} сравнений OK')
    return nchk


# ---------- таблицы ----------
def qlike_cell(T, key, lab, h, m, mode):
    """mode: 'pool' (3.4–3.6) или 'years' (3.7)."""
    if mode == 'years':
        if key == 'V':
            return T.cell('table_3_7_test.csv', {'Модель': m}, f'H = {h}', 'q4')
        return T.cell(IST, {'Период': lab, 'h': str(h), 'Модель': m}, 'QLIKE среднее по годам', 'q4')
    if key == 'V':
        return T.cell('table_3_final_po_godam_test.csv', {'h': str(h), 'Модель': m, 'Год': 'все годы (пул)'}, 'QLIKE', 'q4')
    if m in SVODKA:
        return T.cell('table_3_kontrol_svodka_test.csv', {'Период': lab, 'h': str(h)}, SVODKA[m], 'q4')
    return T.cell(IST, {'Период': lab, 'h': str(h), 'Модель': m}, 'QLIKE пул', 'q4')


def t_models(name, models, mode):
    T = Tab(name, ['Модель'] + [f'{p}, H = {h}' for _, p, _, _ in PER for h in HS])
    for m in models:
        T.start(m)
        for key, _, lab, _ in PER:
            for h in HS:
                qlike_cell(T, key, lab, h, m, mode)
        T.end()
    return T


def delta(q, qref):
    d = round(100 * (num(q) / num(qref) - 1))
    return f'+{d}%' if d > 0 else (f'-{-d}%' if d < 0 else '0%')


def t35():
    T = Tab('3_5', ['Модель'] + [f'{p}, {c}' for _, p, _, _ in PER for c in ('QLIKE H = 1', 'Δ vs XGBoost', 'Δ vs HAR-J')])
    for m in ('XGBoost', 'LightGBM', 'HAR-J'):
        T.start(m)
        for key, _, lab, _ in PER:
            qlike_cell(T, key, lab, 1, m, 'pool')
            T.lit('Δ')
            T.lit('Δ')
        T.end()
    for i, r in enumerate(T.rows):  # Δ из QLIKE той же таблицы (как в диссертации: целые %)
        for k in range(len(PER)):
            j = 1 + 3 * k
            qx, qh = T.rows[0][j], T.rows[2][j]
            r[j + 1] = '—' if i == 0 else delta(r[j], qx)
            r[j + 2] = '—' if i == 2 else delta(r[j], qh)
    for m in ('LSTM', 'GRU'):
        T.rows.append([m] + ['н/д'] * (len(T.header) - 1))
    return T


def dm_where(key, h, et, main_only):
    e_kn, e_v = ET9[et]
    if key == 'V':
        return 'table_3_9_test.csv', {'H': str(h), 'эталон': e_v}, ('t', 'p_one_H1')
    return DMFILE[key], {'h': str(h), 'эталон': e_kn}, ('t', 'p односторонний (H1: финал лучше)')


def method_rows(rel, where):
    return [r for r in rows_of(rel) if all(r[k] == v for k, v in where.items())]


def t38():
    T = Tab('3_8', ['H'] + [f'{p}, {c}' for _, p, _, _ in PER for c in ('t-стат.', 'p-value')])
    for h in HS:
        T.start(str(h))
        for key, _, lab, _ in PER:
            if key == 'V':
                T.cell('table_3_8_test.csv', {'H': str(h)}, 't-стат.', 't2')
                T.cell('table_3_8_test.csv', {'H': str(h)}, 'p-value', 'p3')
            else:
                sp = [r['способ'] for r in method_rows(DMFILE[key], {'h': str(h), 'эталон': 'HAR-J (б)'}) if r['способ'].startswith('основной')][0]
                w = {'h': str(h), 'эталон': 'HAR-J (б)', 'способ': sp}
                T.cell(DMFILE[key], w, 't', 't2')
                T.cell(DMFILE[key], w, 'p односторонний (H1: финал лучше)', 'p3')
        T.end()
    return T


def t39():
    T = Tab('3_9', ['H', 'Эталон', 'Способ'] + [f'{p}, {c}' for _, p, _, _ in PER for c in ('t', 'p (одностор.)')])
    for h in HS:
        for et in ET9:
            rel, w, _ = dm_where('V', h, et, False)
            for sp in [r['способ'] for r in method_rows(rel, w)]:
                T.start(str(h), et, sp)
                for key, _, lab, _ in PER:
                    rel, w, (ct, cp) = dm_where(key, h, et, False)
                    w = {**w, 'способ': sp}
                    T.cell(rel, w, ct, 't2')
                    T.cell(rel, w, cp, 'p3')
                T.end()
    return T


# ---------- примечания и источники ----------
X_NOTE = ('«XGBoost» и «LightGBM» — слоты ноги финального дизайна без поправки (столбцы xgb_raw, lgb_raw; при H = 5 «XGBoost» = среднее '
          'V9-XGBoost и V9-LightGBM), а не модели XGBoost и LightGBM из прежнего текста; «HAR-J» — HAR-J без поправки (har_raw), как в диссертации.')
SRC_POOL = ('2022–2026 и 2025 – 01.2026: Гибридная, HAR-J, HAR-J с поправкой — table_3_kontrol_svodka_test.csv; XGBoost, LightGBM — '
            'dlya_teksta/istochnik_srez_test.csv (из сохранённых прогнозов test_locked/final3). 2017–2026: table_3_final_po_godam_test.csv, строка «все годы (пул)».')
FOOT = {
    '3_4': ('QLIKE по всем парам (дата, тикер) периода. ' + X_NOTE, SRC_POOL),
    '3_5': ('QLIKE при H = 1, пул; Δ = QLIKE строки / QLIKE XGBoost (HAR-J) − 1, целые % по значениям таблицы. LSTM и GRU в финальный тест '
            '(test_locked/final3) не входят — н/д. ' + X_NOTE, SRC_POOL),
    '3_6': ('QLIKE по всем парам (дата, тикер) периода. «Гибридная» — финальный дизайн (rv_pred): H = 1 — D22_h1, H = 5 — D12_h5, '
            'H = 22 — HAR-J с поправкой (совпадает со строкой «HAR-J с поправкой»). LSTM, GRU, GJR-GARCH в финальный тест не входят. ' + X_NOTE, SRC_POOL),
    '3_7': ('Простое среднее годовых QLIKE периода (как в table_3_7_test: неполный 2026 весит как год). Строки — как table_3_7_test. ' + X_NOTE,
            '2022–2026 и 2025 – 01.2026: dlya_teksta/istochnik_srez_test.csv (столбец «QLIKE среднее по годам», из сохранённых прогнозов); '
            '2017–2026: table_3_7_test.csv.'),
    '3_8': ('Основной DM: дневной ряд средней по тикерам разности потерь QLIKE (эталон − финал), NW лаг H − 1; эталон — HAR-J с поправкой; '
            'p односторонний (H1: финал лучше). H = 22: финал = эталон, тест не определён (—).',
            '2022–2026: table_3_kontrol_dm_test.csv; 2025 – 01.2026: table_3_neotkr_dm_test.csv (эталон HAR-J (б), основной способ); 2017–2026: table_3_8_test.csv.'),
    '3_9': ('DM тремя способами (d = потери эталона − потери финала, t > 0 — финал лучше), p односторонний (H1: финал лучше). Лаг NW — H − 1. '
            'Против HAR-J с поправкой при H = 22 тест не определён (финал = эталон, —). Двусторонние p и n — в исходных CSV.',
            '2022–2026: table_3_kontrol_dm_test.csv; 2025 – 01.2026: table_3_neotkr_dm_test.csv; 2017–2026: table_3_9_test.csv.'),
}
TITLE = {
    '3_4': 'Таблица 3.4. QLIKE моделей бустинга и HAR-J (контрольный период 2022–2026; справочно 2017–2026 и 2025 – 01.2026)',
    '3_5': 'Таблица 3.5. QLIKE при H = 1: бустинг и HAR-J; рекуррентные сети в финальный тест не входят',
    '3_6': 'Таблица 3.6. Сводное сравнение моделей (контрольный период 2022–2026; справочно 2017–2026 и 2025 – 01.2026)',
    '3_7': 'Таблица 3.7. Среднее годовых QLIKE моделей (2022–2026; справочно 2017–2026 и 2025 – 01.2026)',
    '3_8': 'Таблица 3.8. Тест Диболда–Мариано: финальная (гибридная) модель против HAR-J с поправкой',
    '3_9': 'Таблица 3.9. Тест Диболда–Мариано тремя способами против двух эталонов',
}


def build():
    return [t_models('3_4', ('XGBoost', 'LightGBM', 'HAR-J'), 'pool'), t35(),
            t_models('3_6', ('Гибридная', 'LightGBM', 'XGBoost', 'HAR-J', 'HAR-J с поправкой'), 'pool'),
            t_models('3_7', ('HAR-J', 'HAR-J с поправкой', 'XGBoost', 'LightGBM', 'Гибридная'), 'years'), t38(), t39()]


def write(T):
    note, source = FOOT[T.name]
    tail = [[], [f'Примечание. {note} {NOTE} {SRK}'], [f'Источник: составлено автором; {source}']]
    (OUT / f'table_{T.name}.csv').write_bytes(to_bytes([T.header] + T.rows + tail))


# ---------- сверка ----------
def read_out(name):
    t = (OUT / f'table_{name}.csv').read_bytes()
    assert t.startswith('﻿'.encode()), f'{name}: нет BOM'
    rows = list(csv.reader(io.StringIO(t.decode('utf-8-sig'), newline=''), delimiter=';'))
    return rows[1:rows.index([])]


def same(out, s, kind):
    if s == '':
        return out == '—'
    if kind == 'p3' and s.startswith('<'):
        return out == s
    nd = {'q4': 4, 't2': 2, 'p3': 3}[kind]
    return num(out) == round(num(s), nd)


def check(tabs):
    _cache.clear()
    rep = []
    for T in tabs:
        body = read_out(T.name)
        assert len(body) == len(T.rows), T.name
        seen, k = set(), 0
        for i, j, rel, where, col, kind in T.prov:
            out = body[i][j]
            s = src(rel, where, col)
            assert same(out, s, kind), f'{T.name} [{i},{j}] {out!r} != {rel} {where} {col} {s!r}'
            seen.add((i, j))
            k += 1
        nd = 0
        if T.name == '3_5':
            for i, r in enumerate(body[:3]):
                for kk in range(len(PER)):
                    j = 1 + 3 * kk
                    ex = ('—' if i == 0 else delta(r[j], body[0][j]), '—' if i == 2 else delta(r[j], body[2][j]))
                    assert (r[j + 1], r[j + 2]) == ex, f'3.5 Δ [{i},{j}]'
                    seen |= {(i, j + 1), (i, j + 2)}
                    nd += 2
        # каждая числовая ячейка обязана иметь источник
        for i, r in enumerate(body):
            for j, c in enumerate(r):
                if any(ch.isdigit() for ch in c) and (i, j) not in seen:
                    lead = 1 if T.name in ('3_4', '3_5', '3_6', '3_7', '3_8') else 3
                    assert j < lead, f'{T.name} [{i},{j}] {c!r}: число без источника'
        dash = sum(c == '—' for r in body for c in r)
        rep.append((T.name, len(body), k, nd, dash))
        print(f'сверка table_{T.name}.csv: {len(body)} строк, {k} чисел = исходный CSV, Δ пересчитано {nd}, «—» {dash} — OK')
    return rep


def readme(tabs, rep, nchk):
    L = ['# Таблицы 3.4–3.9 для текста диссертации',
         '',
         f'**{SRK}** Поручение владельца 26.10, 09:50, п. 2; решения 09:00, 09:05, 09:15 — `docs/zadaniya/2026-10-25_pereschet.md`. '
         'Тест (`final_test.py --scope test`) не запускался повторно; модели не обучались; сохранённые таблицы теста не менялись.',
         '',
         'Скрипт: `python3 scripts/analysis/tables_dlya_teksta.py` (перезапускаемый; `--check` — только сверка). '
         'Формат: «;», десятичная запятая, UTF-8 с BOM; под таблицей строки «Примечание» и «Источник». '
         'Вид строк и столбцов — как в «Диссертация_Лысенок_правки_5.docx» (табл. 3.4–3.8), табл. 3.9 — как `table_3_9_test.csv`; '
         'периоды стоят рядом: столбцы «<период>, H = …».',
         '',
         '## Периоды',
         '',
         '- **2022–2026¹ — основной (контрольный).** Годы, не участвовавшие в отборе дизайнов на разработочном наборе. '
         'Сноска: 2022–2024 — подтверждающий набор (3 открытия, 9 просмотров; использовался для выбора финала из трёх финалистов); '
         'ни разу не открывались только 2025 и 01–02.2026.',
         '- **2017–2026² — весь тест**, заранее заданный в `final_test.py` основной расчёт (справочно). 2026 — 01.01–03.02.',
         '- **2025 – 01.2026³ — не открывавшиеся** (строки 2025 и 2026 гг.; последняя дата 02.02.2026 при H = 1, 29.01 при H = 5, 12.01 при H = 22).',
         '',
         'Эталон DM (3.8, H1) — HAR-J с поправкой (`har_corr`, s² = sm). В 3.9 — оба эталона: HAR-J с поправкой и HAR-J как в диссертации (`har_raw`). '
         'QLIKE = mean(y/p − ln(y/p) − 1) по парам (дата, тикер), фильтр rv_actual > 1e-12.',
         '']
    SRCS = {
        '3_4': [('Модель', 'XGBoost, LightGBM, HAR-J — как в диссертации'),
                ('2022–2026, H = 1/5/22', 'HAR-J: `table_3_kontrol_svodka_test.csv` («QLIKE HAR-J (а)»); XGBoost, LightGBM: `dlya_teksta/istochnik_srez_test.csv` («QLIKE пул»)'),
                ('2017–2026, H = 1/5/22', '`table_3_final_po_godam_test.csv`, Год = «все годы (пул)», QLIKE (до 4 знаков)'),
                ('2025 – 01.2026, H = 1/5/22', 'как 2022–2026, строки периода «неоткрывавшийся 2025–2026»')],
        '3_5': [('Модель', 'XGBoost, LightGBM, HAR-J; LSTM, GRU — н/д (в тесте final3 нет)'),
                ('<период>, QLIKE H = 1', 'как табл. 3.4, H = 1'),
                ('<период>, Δ vs XGBoost / Δ vs HAR-J', 'пересчёт из QLIKE той же строки и строки XGBoost / HAR-J, целые %')],
        '3_6': [('Модель', 'Гибридная (финал, `rv_pred`), LightGBM, XGBoost, HAR-J, HAR-J с поправкой — как `table_3_6_test.csv`'),
                ('2022–2026 и 2025 – 01.2026', 'Гибридная, HAR-J, HAR-J с поправкой: `table_3_kontrol_svodka_test.csv` («QLIKE Финальная», «QLIKE HAR-J (а)», «QLIKE HAR-J (б)»); XGBoost, LightGBM: `dlya_teksta/istochnik_srez_test.csv`'),
                ('2017–2026', '`table_3_final_po_godam_test.csv`, «все годы (пул)»')],
        '3_7': [('Модель', 'как `table_3_7_test.csv`'),
                ('2022–2026 и 2025 – 01.2026', '`dlya_teksta/istochnik_srez_test.csv`, «QLIKE среднее по годам»'),
                ('2017–2026', '`table_3_7_test.csv`')],
        '3_8': [('H', '1, 5, 22'),
                ('2022–2026, t-стат. / p-value', '`table_3_kontrol_dm_test.csv`: эталон HAR-J (б), способ «основной…», t и «p односторонний»'),
                ('2017–2026', '`table_3_8_test.csv`'),
                ('2025 – 01.2026', '`table_3_neotkr_dm_test.csv`, как 2022–2026')],
        '3_9': [('H, Эталон, Способ', 'строки `table_3_9_test.csv` (способ — его подпись; лаг NW = H − 1)'),
                ('2022–2026, t / p', '`table_3_kontrol_dm_test.csv` (эталон HAR-J (а) = как в диссертации, HAR-J (б) = с поправкой), t и «p односторонний»'),
                ('2017–2026', '`table_3_9_test.csv`: t, p_one_H1'),
                ('2025 – 01.2026', '`table_3_neotkr_dm_test.csv`, как 2022–2026')],
    }
    for T in tabs:
        note, _ = FOOT[T.name]
        L += [f'## `table_{T.name}.csv` — {TITLE[T.name]}', '', '| колонка | источник |', '|---|---|']
        L += [f'| {a} | {b} |' for a, b in SRCS[T.name]]
        L += ['', f'Примечание: {note}', '']
    L += ['## `istochnik_srez_test.csv` — что посчитано из прогнозов',
          '',
          'Только то, чего нет в сохранённых CSV: QLIKE (пул и среднее годовых) для пяти строк × три периода × три H. '
          'Вход — `/Users/nilysenok/Desktop/Старое/PhD_Vol_trading/data/regen/test_locked/final3/final_h{1,5,22}_test.parquet` '
          '(SHA-256 сверен с FINAL_DESIGN, поле 10, в `test_dop.load`). Функции: `test_dop.load`, `test_dop.cut`, `ftp_tables._q` (без правок). '
          'В таблицы из этого файла идут: XGBoost и LightGBM на 2022–2026 и 2025 – 01.2026 (3.4–3.6) и среднее годовых на этих периодах (3.7); '
          'остальные строки файла — для контроля.',
          '',
          f'Контроль расчёта ({nchk} сравнений, все OK): каждый годовой QLIKE = `table_3_final_po_godam_test` (отн. 1e-12); пул 2017–2026 = её строка «все годы (пул)»; '
          'среднее годовых 2017–2026 = `table_3_7_test` (4 знака); QLIKE финала, HAR-J (а), (б) и n на 2022–2026 и 2025 – 01.2026 = `table_3_kontrol_svodka_test`.',
          '',
          '## Отчёт о сверке',
          '',
          'Каждое число выходной таблицы заново прочитано из своего исходного CSV (по строке и колонке) и сравнено: QLIKE — после округления источника до 4 знаков, '
          't — до 2, p — до 3 («<0,001» — как строка); пустая ячейка источника = «—». Δ табл. 3.5 пересчитаны из QLIKE таблицы. '
          'Числовых ячеек без источника нет.',
          '',
          '| таблица | строк | чисел = исходный CSV | Δ пересчитано | «—» | итог |', '|---|---|---|---|---|---|']
    L += [f'| table_{n}.csv | {r} | {k} | {d} | {s} | OK |' for n, r, k, d, s in rep]
    L += ['', '«—» в 3.8/3.9: DM против HAR-J с поправкой при H = 22 не определён (финал = эталон); в 3.5 — Δ строки к самой себе. '
          '«н/д» в 3.5 — LSTM и GRU (нет в финальном тесте).', '']
    (OUT / 'README.md').write_text('\n'.join(L), encoding='utf-8')
    print('записано', OUT / 'README.md')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--check', action='store_true', help='только сверка записанных таблиц')
    a = ap.parse_args()
    nchk = None if a.check else srez()
    tabs = build()
    if not a.check:
        for T in tabs:
            write(T)
            print('записано', OUT / f'table_{T.name}.csv')
    rep = check(tabs)
    if not a.check:
        readme(tabs, rep, nchk)


if __name__ == '__main__':
    main()
