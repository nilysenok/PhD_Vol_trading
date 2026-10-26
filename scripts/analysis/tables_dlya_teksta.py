"""Таблицы 3.4–3.9 для текста диссертации (поручение владельца 26.10, 09:50, п. 2; дополнение 09:55 — правило «закрытие биржи»).
Вид строк и столбцов — как в «Диссертация_Лысенок_правки_5.docx» (табл. 3.4–3.8; 3.9 — DM тремя способами, как table_3_9_test).
Основной период — контрольный 2022–2026 (2022–2024 — подтверждающий набор, сноска), рядом — весь тест 2017–2026 и не открывавшиеся
2025 – 01.2026. Периоды стоят рядом блоками столбцов.

СРЕЗ ПО ПОРУЧЕНИЮ ВЛАДЕЛЬЦА ПОСЛЕ ЕДИНСТВЕННОГО ЗАПУСКА ТЕСТА (final_test.py --scope test, 26.10 08:57, коммит 002a74c).
Тест не запускается, модели не обучаются, прогнозы не меняются, сохранённые таблицы теста не перезаписываются.

Версии (--versiya):
  1 — без правила (PR #118): docs/tables/dlya_teksta/v1/table_3_{4..9}.csv. Источники — table_3_*_test, table_3_kontrol_*_test,
      table_3_neotkr_*_test; недостающее — dlya_teksta/istochnik_srez_test.csv.
  2 — основная БЕЗ правила (заранее объявленный дизайн; решение владельца 26.10, 10:20), рядом — с правилом «закрытие биржи» как проверка
      устойчивости (FINAL_DESIGN, разд. 13; ПРАВИЛО СФОРМУЛИРОВАНО ПОСЛЕ ПЕРВОГО ПРОГОНА ТЕСТА); в примечаниях — эпизод 02–03.2022 при h=5
      и доли исключённых точек (числа — из исходных CSV); в 3.8 — доля тикеров с выигрышем и знаковый тест (иллюстрация: тикеры коррелированы); приложение — исключённые точки по h и тикерам.
      docs/tables/dlya_teksta/table_3_{4..9}.csv, table_3_pril_zakrytie.csv. Источники — table_3_{kontrol,vse,neotkr}_*_test_{s_zakrytiem,
      bez_zakrytiya}.csv, table_3_zakrytie_isklyucheno.csv (test_dop2.py, PR #120); недостающее — dlya_teksta/istochnik_srez_zakrytie_test.csv.
Недостающее (QLIKE XGBoost/LightGBM без поправки на срезах; среднее годовых QLIKE для 3.7) считается из сохранённых прогнозов
<общий data>/regen/test_locked/final3/final_h{1,5,22}_test.parquet: test_dop.load (SHA-256 = FINAL_DESIGN, поле 10), test_dop.cut, ftp_tables._q;
точки правила — ровно строки table_3_zakrytie_isklyucheno.csv (тикер, t от первой до последней исключённой). Контроль расчёта — против
сохранённых CSV (годовые QLIKE, пул, среднее годовых, сводки обеих версий, n).
Сверка: каждое число выходной таблицы заново читается из своего исходного CSV и сравнивается (QLIKE — 4 знака, t — 2, p — 3, целые и даты —
точно); Δ табл. 3.5 пересчитываются из QLIKE той же строки. README.md — источники колонок и отчёт о сверке обеих версий.

python3 scripts/analysis/tables_dlya_teksta.py [--versiya 1|2|all]   # срез, таблицы, сверка, README (по умолчанию all)
python3 scripts/analysis/tables_dlya_teksta.py --check               # только сверка записанных таблиц обеих версий (прогнозы не читаются)
"""
import argparse, csv, io, sys
from collections import namedtuple
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_dop as TD
import ftp_tables as FT
from tables_test_format import conv, to_bytes

TABLES = TD.TABLES
OUT = TABLES / 'dlya_teksta'
IST1 = 'dlya_teksta/istochnik_srez_test.csv'
IST2 = 'dlya_teksta/istochnik_srez_zakrytie_test.csv'
ISKL = 'table_3_zakrytie_isklyucheno.csv'
HS = (1, 5, 22)
COLS = {'Гибридная': 'rv_pred', 'HAR-J': 'har_raw', 'HAR-J с поправкой': 'har_corr', 'XGBoost': 'xgb_raw', 'LightGBM': 'lgb_raw'}
SVODKA = {'Гибридная': 'QLIKE Финальная', 'HAR-J': 'QLIKE HAR-J (а)', 'HAR-J с поправкой': 'QLIKE HAR-J (б)'}
ET9 = {'HAR-J как в диссертации': ('HAR-J (а)', 'HAR-J как в диссертации'), 'HAR-J с поправкой (эталон)': ('HAR-J (б)', 'HAR-J с поправкой')}
RULE = {'s': 's_zakrytiem', 'b': 'bez_zakrytiya'}
YEARS = {'kontrol': (2022, 2026), 'vse': (2017, 2026), 'neotkr': (2025, 2026)}
LAB1 = {'kontrol': TD.KONTROL, 'vse': TD.VES, 'neotkr': TD.NEOTKR}
Blk = namedtuple('Blk', 'label tag rule')  # rule: None (версия 1), 's' (с правилом), 'b' (без правила)
BLOCKS = {1: [Blk('2022–2026¹', 'kontrol', None), Blk('2017–2026²', 'vse', None), Blk('2025 – 01.2026³', 'neotkr', None)],
          2: [Blk('2022–2026¹', 'kontrol', 'b'), Blk('2022–2026¹ с правилом⁴ (устойчивость)', 'kontrol', 's'),
              Blk('2017–2026²', 'vse', 'b'), Blk('2017–2026² с правилом⁴ (устойчивость)', 'vse', 's'),
              Blk('2025 – 01.2026³ (правило не меняет)', 'neotkr', 'b')]}
DIR = {1: OUT / 'v1', 2: OUT}
SRK = {1: 'Срез по поручению владельца после единственного запуска теста (26.10 08:57): столбцы 2022–2026 и 2025 – 01.2026.',
       2: ('Срез по поручению владельца после единственного запуска теста (26.10 08:57). Основная версия — без правила (заранее объявленный дизайн); '
           'столбцы «с правилом» — проверка устойчивости, правило «закрытие биржи» сформулировано после первого прогона теста.')}
NOTE = ('¹ Контрольный период (основной): годы, не участвовавшие в отборе дизайнов на разработочном наборе 2017–2021; '
        '2022–2024 — подтверждающий набор (3 открытия, использовался для выбора финала из трёх финалистов), 2025 – 01.2026 не открывались. '
        '² Весь тест — заранее заданный в final_test.py основной расчёт (справочно); 2026 — 01.01–03.02. '
        '³ Ни разу не открывавшиеся данные: строки 2025 и 2026 гг. (последняя дата 02.02.2026 при H = 1, 29.01 при H = 5, 12.01 при H = 22).')
def pct(a, b):
    return f'{100 * a / b:.2f}'.replace('.', ',') + '%'


def facts():
    """Числа для примечаний и README — из исходных CSV (с проверкой: n без правила − n с правилом = исключённые точки).
    Возвращает (доли исключённых точек, эпизод h=5 на контрольном периоде)."""
    sh = []
    for h in HS:
        x = int(src(ISKL, {'h': str(h), 'Тикер': 'ВСЕГО'}, 'исключено точек'))
        nk, nv = (int(src(f'table_3_{t}_svodka_test_bez_zakrytiya.csv', {'h': str(h)}, 'n')) for t in ('kontrol', 'vse'))
        for t, n in (('kontrol', nk), ('vse', nv)):
            assert n - int(src(f'table_3_{t}_svodka_test_s_zakrytiem.csv', {'h': str(h)}, 'n')) == x, f'n {t} h={h}'
        sh.append(f'H = {h} — {x} ({pct(x, nk)} точек 2022–2026 из {nk}; {pct(x, nv)} точек 2017–2026 из {nv})')
    dol = 'Исключено точек (тикер × дата): ' + '; '.join(sh) + '.'
    g = {}
    for rk in ('b', 's'):
        sv, zn = f'table_3_kontrol_svodka_test_{RULE[rk]}.csv', f'table_3_kontrol_znak_test_{RULE[rk]}.csv'
        w = {'h': '5'}
        g[rk] = (disp(src(sv, w, 'QLIKE Финальная'), 'q4'), disp(src(sv, w, 'QLIKE HAR-J (б)'), 'q4'), src(sv, w, 'финал к HAR-J (б), %'),
                 disp(src(sv, w, 'DM t vs HAR-J (б)'), 't2'), disp(src(sv, w, 'p vs HAR-J (б)'), 'p3'),
                 src(zn, {**w, 'эталон': 'HAR-J (б)'}, 'финал лучше'))
    sg = lambda s: '−' + s[1:] if s.startswith('-') else '+' + s
    f = lambda a: (f'QLIKE финала {a[0]} против эталона {a[1]} ({sg(a[2])}%), основной DM t = {a[3].replace("-", "−")}, p = {a[4]}, '
                   f'тикеров с выигрышем {a[5]} из 17')
    epi = ('При H = 5 выигрыш финала у HAR-J с поправкой на контрольном периоде сосредоточен в эпизоде февраля–марта 2022 г. '
           f'(окна цели, задевающие остановку торгов): без правила — {f(g["b"])}; с правилом — {f(g["s"])}.')
    return dol, epi


def note4():
    dol, epi = facts()
    return (' ⁴ Правило «закрытие биржи» (FINAL_DESIGN, разд. 13; поручение владельца 26.10, 09:55) — проверка устойчивости, сформулировано после '
            'первого прогона теста, когда его результаты были известны: из оценки всех моделей исключена точка t тикера, если окно цели [t+1, t+H] '
            f'задевает остановку торгов MOEX 28.02–23.03.2022 (приложение table_3_pril_zakrytie). {dol} {epi}')


# ---------- чтение исходных CSV ----------
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


ND = {'q4': 4, 't2': 2, 'p3': 3}


def disp(s, kind):
    """Строка исходного CSV -> вид в таблице. q4/t2/p3 — знаков после запятой ('<0,001' как есть); int — целое; date — ДД.ММ.ГГГГ; пусто -> '—'."""
    if s == '':
        return '—'
    if kind == 'int':
        return str(int(num(s)))
    if kind == 'date':
        y, m, d = s.split('-')
        return f'{d}.{m}.{y}'
    if kind == 'p3' and s.startswith('<'):
        return s
    return f'{num(s):.{ND[kind]}f}'.replace('.', ',')


def same(out, s, kind):
    if s == '':
        return out == '—'
    if kind in ('int', 'date') or (kind == 'p3' and s.startswith('<')):
        return out == disp(s, kind)
    return num(out) == round(num(s), ND[kind])


class Tab:
    def __init__(self, name, header):
        self.name, self.header, self.rows, self.prov = name, header, [], []

    def start(self, *lits):
        self._cur = list(lits)

    def cell(self, rel, where, col, kind):
        self.prov.append((len(self.rows), len(self._cur), rel, where, col, kind))
        self._cur.append(disp(src(rel, where, col), kind))

    def lit(self, v):
        self._cur.append(v)

    def end(self):
        assert len(self._cur) == len(self.header), (self.name, self._cur)
        self.rows.append(self._cur)


# ---------- срез из сохранённых прогнозов ----------
def write_df(df, rel):
    s = io.StringIO()
    df.to_csv(s, index=False, sep=';', decimal=',')
    cells = [[conv(c) for c in r] for r in csv.reader(io.StringIO(s.getvalue(), newline=''), delimiter=';')]
    (TABLES / rel).parent.mkdir(parents=True, exist_ok=True)
    (TABLES / rel).write_bytes(to_bytes(cells))
    _cache.pop(rel, None)


def qrow(df, m, c):
    years = sorted(int(y) for y in df.year.unique())
    yq = {y: FT._q(df[df.year == y], c) for y in years}
    return years, yq, FT._q(df, c), float(np.mean(list(yq.values())))


def srez1(D):
    po = pd.read_csv(TABLES / 'table_3_final_po_godam_test_raw.csv')
    t7 = pd.read_csv(TABLES / 'table_3_7_test_raw.csv')
    sv = rows_of('table_3_kontrol_svodka_test.csv')
    rows, nchk = [], 0
    for tag in ('kontrol', 'vse', 'neotkr'):
        lab = LAB1[tag]
        for h, df in TD.cut(D, *YEARS[tag]).items():
            for m, c in COLS.items():
                years, yq, pool, mean_y = qrow(df, m, c)
                for y, q in yq.items():
                    ref = po[(po.h == h) & (po['Модель'] == m) & (po['Год'] == str(y))].QLIKE.iloc[0]
                    assert np.isclose(q, ref, rtol=1e-12, atol=0), f'год {y} h={h} {m}'
                    nchk += 1
                if tag == 'vse':
                    ref = po[(po.h == h) & (po['Модель'] == m) & (po['Год'] == 'все годы (пул)')].QLIKE.iloc[0]
                    assert np.isclose(pool, ref, rtol=1e-12, atol=0) and round(mean_y, 4) == t7[t7['Модель'] == m][f'H = {h}'].iloc[0], f'vse h={h} {m}'
                    nchk += 2
                elif m in SVODKA:
                    r = [x for x in sv if x['Период'] == lab and x['h'] == str(h)][0]
                    assert round(pool, 4) == num(r[SVODKA[m]]) and len(df) == int(r['n']), f'svodka {tag} h={h} {m}'
                    nchk += 2
                rows.append({'Период': lab, 'h': h, 'Модель': m, 'столбец прогноза': c, 'n': len(df),
                             'годы': f'{years[0]}–{years[-1]}', 'QLIKE пул': pool, 'QLIKE среднее по годам': mean_y})
    write_df(pd.DataFrame(rows), IST1)
    print(f'срез v1: {len(rows)} строк -> {IST1}; контроль: {nchk} сравнений OK')
    return nchk


def isklyuchit(D):
    """Маски исключения по table_3_zakrytie_isklyucheno.csv: строки тикера с t от первой до последней исключённой включительно."""
    ex, nchk = {}, 0
    for h, df in D.items():
        m = np.zeros(len(df), bool)
        for r in rows_of(ISKL):
            if r['h'] != str(h) or r['Тикер'] == 'ВСЕГО' or int(r['исключено точек']) == 0:
                continue
            e = (df.ticker.values == r['Тикер']) & df.date.between(pd.Timestamp(r['первая исключённая t']), pd.Timestamp(r['последняя исключённая t'])).values
            assert e.sum() == int(r['исключено точек']), f'h={h} {r["Тикер"]}: {e.sum()} != {r["исключено точек"]}'
            m |= e
            nchk += 1
        assert m.sum() == int(src(ISKL, {'h': str(h), 'Тикер': 'ВСЕГО'}, 'исключено точек')), f'h={h}: всего'
        ex[h] = m
    return ex, nchk


def srez2(D):
    ex, nchk = isklyuchit(D)
    t7 = pd.read_csv(TABLES / 'table_3_7_test_raw.csv')
    v1 = {(r['Период'], r['h'], r['Модель']): r for r in rows_of(IST1)}
    rows, keep = [], {}
    for rk, rname in RULE.items():
        DR = {h: (df[~ex[h]] if rk == 's' else df).reset_index(drop=True) for h, df in D.items()}
        for tag in ('kontrol', 'vse', 'neotkr'):
            sv = f'table_3_{tag}_svodka_test_{rname}.csv'
            for h, df in TD.cut(DR, *YEARS[tag]).items():
                assert len(df) == int(src(sv, {'h': str(h)}, 'n')), f'n {sv} h={h}'
                assert round(FT._q(df, 'lgb_corr'), 4) == num(src(sv, {'h': str(h)}, 'QLIKE Лучшая одиночная бустинга')), f'{sv} h={h} лучшая'
                nchk += 2
                for m, c in COLS.items():
                    years, yq, pool, mean_y = qrow(df, m, c)
                    if m in SVODKA:
                        assert round(pool, 4) == num(src(sv, {'h': str(h)}, SVODKA[m])), f'{sv} h={h} {m}'
                        nchk += 1
                    if rk == 'b':  # без правила = срез версии 1
                        r1 = v1[(LAB1[tag], str(h), m)]
                        assert np.isclose(pool, num(r1['QLIKE пул']), rtol=1e-12) and np.isclose(mean_y, num(r1['QLIKE среднее по годам']), rtol=1e-12)
                        nchk += 2
                        if tag == 'vse':
                            assert round(mean_y, 4) == t7[t7['Модель'] == m][f'H = {h}'].iloc[0]
                            nchk += 1
                    keep[(tag, rk, h, m)] = (pool, mean_y)
                    rows.append({'Период': tag, 'правило': rname, 'h': h, 'Модель': m, 'столбец прогноза': c, 'n': len(df),
                                 'годы': f'{years[0]}–{years[-1]}', 'QLIKE пул': pool, 'QLIKE среднее по годам': mean_y})
    for (tag, rk, h, m), v in keep.items():  # на неоткрывавшемся правило ничего не меняет
        if tag == 'neotkr':
            assert v == keep[('neotkr', 'b', h, m)], f'neotkr h={h} {m}'
    write_df(pd.DataFrame(rows), IST2)
    print(f'срез v2: {len(rows)} строк -> {IST2}; контроль: {nchk} сравнений OK')
    return nchk


# ---------- источники ячеек ----------
def q_pool(T, b, h, m):
    if b.rule is None:
        if b.tag == 'vse':
            return T.cell('table_3_final_po_godam_test.csv', {'h': str(h), 'Модель': m, 'Год': 'все годы (пул)'}, 'QLIKE', 'q4')
        if m in SVODKA:
            return T.cell('table_3_kontrol_svodka_test.csv', {'Период': LAB1[b.tag], 'h': str(h)}, SVODKA[m], 'q4')
        return T.cell(IST1, {'Период': LAB1[b.tag], 'h': str(h), 'Модель': m}, 'QLIKE пул', 'q4')
    if m in SVODKA:
        return T.cell(f'table_3_{b.tag}_svodka_test_{RULE[b.rule]}.csv', {'h': str(h)}, SVODKA[m], 'q4')
    return T.cell(IST2, {'Период': b.tag, 'правило': RULE[b.rule], 'h': str(h), 'Модель': m}, 'QLIKE пул', 'q4')


def q_years(T, b, h, m):
    if b.tag == 'vse' and b.rule in (None, 'b'):
        return T.cell('table_3_7_test.csv', {'Модель': m}, f'H = {h}', 'q4')
    if b.rule is None:
        return T.cell(IST1, {'Период': LAB1[b.tag], 'h': str(h), 'Модель': m}, 'QLIKE среднее по годам', 'q4')
    return T.cell(IST2, {'Период': b.tag, 'правило': RULE[b.rule], 'h': str(h), 'Модель': m}, 'QLIKE среднее по годам', 'q4')


def dm_file(b):
    if b.rule is None:
        return {'kontrol': 'table_3_kontrol_dm_test.csv', 'neotkr': 'table_3_neotkr_dm_test.csv'}.get(b.tag)
    return f'table_3_{b.tag}_dm_test_{RULE[b.rule]}.csv'


def dm_cells(T, b, h, et, sp):
    e_kn, e_v = ET9[et]
    if b.rule is None and b.tag == 'vse':
        rel, w, cp = 'table_3_9_test.csv', {'H': str(h), 'эталон': e_v, 'способ': sp}, 'p_one_H1'
    else:
        rel, w, cp = dm_file(b), {'h': str(h), 'эталон': e_kn, 'способ': sp}, 'p односторонний (H1: финал лучше)'
    T.cell(rel, w, 't', 't2')
    T.cell(rel, w, cp, 'p3')


def methods(h, et):
    return [r['способ'] for r in rows_of('table_3_9_test.csv') if r['H'] == str(h) and r['эталон'] == ET9[et][1]]


# ---------- таблицы ----------
def t_models(v, name, models, mode):
    B = BLOCKS[v]
    T = Tab(name, ['Модель'] + [f'{b.label}, H = {h}' for b in B for h in HS])
    for m in models:
        T.start(m)
        for b in B:
            for h in HS:
                (q_pool if mode == 'pool' else q_years)(T, b, h, m)
        T.end()
    return T


def delta(q, qref):
    d = round(100 * (num(q) / num(qref) - 1))
    return f'+{d}%' if d > 0 else (f'-{-d}%' if d < 0 else '0%')


def t35(v):
    B = BLOCKS[v]
    T = Tab('3_5', ['Модель'] + [f'{b.label}, {c}' for b in B for c in ('QLIKE H = 1', 'Δ vs XGBoost', 'Δ vs HAR-J')])
    for m in ('XGBoost', 'LightGBM', 'HAR-J'):
        T.start(m)
        for b in B:
            q_pool(T, b, 1, m)
            T.lit('Δ')
            T.lit('Δ')
        T.end()
    for i, r in enumerate(T.rows):
        for k in range(len(B)):
            j = 1 + 3 * k
            r[j + 1] = '—' if i == 0 else delta(r[j], T.rows[0][j])
            r[j + 2] = '—' if i == 2 else delta(r[j], T.rows[2][j])
    for m in ('LSTM', 'GRU'):
        T.rows.append([m] + ['н/д'] * (len(T.header) - 1))
    return T


def t38(v):
    B = BLOCKS[v]
    cols = ('t-стат.', 'p-value') if v == 1 else ('t-стат.', 'p-value', 'тикеров с выигрышем из 17', 'p знакового теста')
    T = Tab('3_8', ['H'] + [f'{b.label}, {c}' for b in B for c in cols])
    sp = {h: [s for s in methods(h, 'HAR-J с поправкой (эталон)') if s.startswith('основной')][0] for h in HS}
    for h in HS:
        T.start(str(h))
        for b in B:
            if b.rule is None and b.tag == 'vse':
                T.cell('table_3_8_test.csv', {'H': str(h)}, 't-стат.', 't2')
                T.cell('table_3_8_test.csv', {'H': str(h)}, 'p-value', 'p3')
            else:
                dm_cells(T, b, h, 'HAR-J с поправкой (эталон)', sp[h])
            if v == 2:
                zf, w = f'table_3_{b.tag}_znak_test_{RULE[b.rule]}.csv', {'h': str(h), 'эталон': 'HAR-J (б)'}
                if h == 22:  # финал = эталон: все тикеры «равно», тест не определён
                    assert src(zf, w, 'равно') == src(zf, w, 'тикеров') and src(zf, w, 'p знаковый односторонний (финал лучше)') == ''
                    T.lit('—')
                    T.cell(zf, w, 'p знаковый односторонний (финал лучше)', 'p3')
                else:
                    T.cell(zf, w, 'финал лучше', 'int')
                    T.cell(zf, w, 'p знаковый односторонний (финал лучше)', 'p3')
        T.end()
    return T


def t39(v):
    B = BLOCKS[v]
    T = Tab('3_9', ['H', 'Эталон', 'Способ'] + [f'{b.label}, {c}' for b in B for c in ('t', 'p (одностор.)')])
    for h in HS:
        for et in ET9:
            for sp in methods(h, et):
                T.start(str(h), et, sp)
                for b in B:
                    dm_cells(T, b, h, et, sp)
                T.end()
    return T


def t_pril():
    T = Tab('3_pril_zakrytie', ['Тикер'] + [f'H = {h}, {c}' for h in HS for c in ('исключено точек', 'первая исключённая t', 'последняя исключённая t')]
            + ['Последний торговый день до 28.02.2022', 'Первый торговый день после 23.03.2022'])
    tickers = [r['Тикер'] for r in rows_of(ISKL) if r['h'] == '1' and r['Тикер'] != 'ВСЕГО']
    for tk in tickers + ['ВСЕГО']:
        T.start(tk if tk != 'ВСЕГО' else 'Всего')
        for h in HS:
            w = {'h': str(h), 'Тикер': tk}
            T.cell(ISKL, w, 'исключено точек', 'int')
            T.cell(ISKL, w, 'первая исключённая t', 'date')
            T.cell(ISKL, w, 'последняя исключённая t', 'date')
        w = {'h': '1', 'Тикер': tk}
        T.cell(ISKL, w, 'последний торговый день до 28.02.2022', 'date')
        T.cell(ISKL, w, 'первый торговый день после 23.03.2022', 'date')
        T.end()
    return T


def build(v):
    tabs = [t_models(v, '3_4', ('XGBoost', 'LightGBM', 'HAR-J'), 'pool'), t35(v),
            t_models(v, '3_6', ('Гибридная', 'LightGBM', 'XGBoost', 'HAR-J', 'HAR-J с поправкой'), 'pool'),
            t_models(v, '3_7', ('HAR-J', 'HAR-J с поправкой', 'XGBoost', 'LightGBM', 'Гибридная'), 'years'), t38(v), t39(v)]
    return tabs + ([t_pril()] if v == 2 else [])


# ---------- примечания и источники ----------
X_NOTE = ('«XGBoost» и «LightGBM» — слоты ноги финального дизайна без поправки (столбцы xgb_raw, lgb_raw; при H = 5 «XGBoost» = среднее '
          'V9-XGBoost и V9-LightGBM), а не модели XGBoost и LightGBM из прежнего текста; «HAR-J» — HAR-J без поправки (har_raw), как в диссертации.')
ZNAK = (' Тикеров с выигрышем — из 17, где QLIKE финала ниже эталона; p знакового теста — биномиальный односторонний. Знаковый тест — '
        'только иллюстрация: тикеры коррелированы, наблюдения не независимы.')
SRC_POOL = {1: ('2022–2026 и 2025 – 01.2026: Гибридная, HAR-J, HAR-J с поправкой — table_3_kontrol_svodka_test.csv; XGBoost, LightGBM — '
                'dlya_teksta/istochnik_srez_test.csv (из сохранённых прогнозов test_locked/final3). 2017–2026: table_3_final_po_godam_test.csv, строка «все годы (пул)».'),
            2: ('Гибридная, HAR-J, HAR-J с поправкой — table_3_{kontrol,vse,neotkr}_svodka_test_{s_zakrytiem,bez_zakrytiya}.csv; XGBoost, LightGBM — '
                'dlya_teksta/istochnik_srez_zakrytie_test.csv (из сохранённых прогнозов test_locked/final3).')}
SRC_YEARS = {1: ('2022–2026 и 2025 – 01.2026: dlya_teksta/istochnik_srez_test.csv (столбец «QLIKE среднее по годам», из сохранённых прогнозов); '
                 '2017–2026: table_3_7_test.csv.'),
             2: '2017–2026 без правила: table_3_7_test.csv; остальные столбцы: dlya_teksta/istochnik_srez_zakrytie_test.csv («QLIKE среднее по годам»).'}
SRC_38 = {1: '2022–2026: table_3_kontrol_dm_test.csv; 2025 – 01.2026: table_3_neotkr_dm_test.csv (эталон HAR-J (б), основной способ); 2017–2026: table_3_8_test.csv.',
          2: 'DM — table_3_{kontrol,vse,neotkr}_dm_test_{s_zakrytiem,bez_zakrytiya}.csv (эталон HAR-J (б), основной способ); тикеры и знаковый тест — '
             'table_3_{kontrol,vse,neotkr}_znak_test_{s_zakrytiem,bez_zakrytiya}.csv (эталон HAR-J (б)).'}
SRC_39 = {1: '2022–2026: table_3_kontrol_dm_test.csv; 2025 – 01.2026: table_3_neotkr_dm_test.csv; 2017–2026: table_3_9_test.csv.',
          2: 'table_3_{kontrol,vse,neotkr}_dm_test_{s_zakrytiem,bez_zakrytiya}.csv.'}


def foot(v, name):
    F = {
        '3_4': ('QLIKE по всем парам (дата, тикер) периода. ' + X_NOTE, SRC_POOL[v]),
        '3_5': ('QLIKE при H = 1, пул; Δ = QLIKE строки / QLIKE XGBoost (HAR-J) − 1, целые % по значениям таблицы. LSTM и GRU в финальный тест '
                '(test_locked/final3) не входят — н/д. ' + X_NOTE, SRC_POOL[v]),
        '3_6': ('QLIKE по всем парам (дата, тикер) периода. «Гибридная» — финальный дизайн (rv_pred): H = 1 — D22_h1, H = 5 — D12_h5, '
                'H = 22 — HAR-J с поправкой (совпадает со строкой «HAR-J с поправкой»). LSTM, GRU, GJR-GARCH в финальный тест не входят. ' + X_NOTE, SRC_POOL[v]),
        '3_7': ('Простое среднее годовых QLIKE периода (как в table_3_7_test: неполный 2026 весит как год). Строки — как table_3_7_test. ' + X_NOTE, SRC_YEARS[v]),
        '3_8': ('Основной DM: дневной ряд средней по тикерам разности потерь QLIKE (эталон − финал), NW лаг H − 1; эталон — HAR-J с поправкой; '
                'p односторонний (H1: финал лучше). H = 22: финал = эталон, тест не определён (—).' + (ZNAK if v == 2 else ''), SRC_38[v]),
        '3_9': ('DM тремя способами (d = потери эталона − потери финала, t > 0 — финал лучше), p односторонний (H1: финал лучше). Лаг NW — H − 1. '
                'Против HAR-J с поправкой при H = 22 тест не определён (финал = эталон, —). Двусторонние p и n — в исходных CSV.', SRC_39[v]),
        '3_pril_zakrytie': ('Приложение к таблицам 3.4–3.9: точки, исключённые правилом «закрытие биржи», по H и тикерам. Окно цели — следующие H торговых '
                            'строк тикера; у LSRG, MTLR, OGKB (открылись 28.03.2022) в закрытие входят и 24–25.03.2022, лишних точек это не добавляет.',
                            'table_3_zakrytie_isklyucheno.csv (test_dop2.py).'),
    }
    return F[name]


TITLE = {
    '3_4': 'Таблица 3.4. QLIKE моделей бустинга и HAR-J (контрольный период 2022–2026; справочно 2017–2026 и 2025 – 01.2026)',
    '3_5': 'Таблица 3.5. QLIKE при H = 1: бустинг и HAR-J; рекуррентные сети в финальный тест не входят',
    '3_6': 'Таблица 3.6. Сводное сравнение моделей (контрольный период 2022–2026; справочно 2017–2026 и 2025 – 01.2026)',
    '3_7': 'Таблица 3.7. Среднее годовых QLIKE моделей (2022–2026; справочно 2017–2026 и 2025 – 01.2026)',
    '3_8': 'Таблица 3.8. Тест Диболда–Мариано: финальная (гибридная) модель против HAR-J с поправкой',
    '3_9': 'Таблица 3.9. Тест Диболда–Мариано тремя способами против двух эталонов',
    '3_pril_zakrytie': 'Приложение к табл. 3.4–3.9. Точки, исключённые правилом «закрытие биржи», по H и тикерам',
}


def write(v, T):
    note, source = foot(v, T.name)
    n = NOTE + (note4() if v == 2 else '') if T.name != '3_pril_zakrytie' else facts()[0]
    tail = [[], [f'Примечание. {note} {n} {SRK[v]}'.replace('  ', ' ')], [f'Источник: составлено автором; {source}']]
    DIR[v].mkdir(parents=True, exist_ok=True)
    (DIR[v] / f'table_{T.name}.csv').write_bytes(to_bytes([T.header] + T.rows + tail))


# ---------- сверка ----------
def read_out(v, name):
    t = (DIR[v] / f'table_{name}.csv').read_bytes()
    assert t.startswith('﻿'.encode()), f'{name}: нет BOM'
    rows = list(csv.reader(io.StringIO(t.decode('utf-8-sig'), newline=''), delimiter=';'))
    return rows[1:rows.index([])]


def check(v, tabs):
    _cache.clear()
    rep = []
    for T in tabs:
        body = read_out(v, T.name)
        assert body == T.rows, f'v{v} {T.name}: записанная таблица не равна построенной из источников'
        seen, k = set(), 0
        for i, j, rel, where, col, kind in T.prov:
            s = src(rel, where, col)
            assert same(body[i][j], s, kind), f'v{v} {T.name} [{i},{j}] {body[i][j]!r} != {rel} {where} {col} {s!r}'
            seen.add((i, j))
            k += 1
        nd = 0
        if T.name == '3_5':
            for i, r in enumerate(body[:3]):
                for kk in range(len(BLOCKS[v])):
                    j = 1 + 3 * kk
                    ex = ('—' if i == 0 else delta(r[j], body[0][j]), '—' if i == 2 else delta(r[j], body[2][j]))
                    assert (r[j + 1], r[j + 2]) == ex, f'3.5 Δ [{i},{j}]'
                    seen |= {(i, j + 1), (i, j + 2)}
                    nd += 2
        lead = 3 if T.name == '3_9' else 1
        for i, r in enumerate(body):
            for j, c in enumerate(r):
                assert not (any(ch.isdigit() for ch in c) and (i, j) not in seen and j >= lead), f'{T.name} [{i},{j}] {c!r}: число без источника'
        dash = sum(c == '—' for r in body for c in r)
        rep.append((T.name, len(body), k, nd, dash))
        print(f'сверка v{v} table_{T.name}.csv: {len(body)} строк, {k} чисел = исходный CSV, Δ {nd}, «—» {dash} — OK')
    if v == 2:  # на неоткрывавшемся версии совпадают: каждое число столбцов «2025 – 01.2026» равно числу файла без правила
        nn = 0
        for T in tabs:
            for i, j, rel, where, col, kind in T.prov:
                if '_neotkr_' in rel and RULE['b'] in rel:
                    assert src(rel.replace(RULE['b'], RULE['s']), where, col) == src(rel, where, col), f'neotkr {rel} {where} {col}'
                    nn += 1
                elif rel == IST2 and where.get('Период') == 'neotkr':
                    assert src(rel, {**where, 'правило': RULE['s']}, col) == src(rel, where, col)
                    nn += 1
        facts()  # контроль: n без правила − n с правилом = исключённые точки (2022–2026 и 2017–2026)
        print(f'сверка v2: 2025 – 01.2026 с правилом = без правила, {nn} чисел — OK')
        rep.append(('neotkr', nn))
    return rep


def readme(rep, nchk):
    L = ['# Таблицы 3.4–3.9 для текста диссертации',
         '',
         f'**{SRK[2]}** Поручение владельца 26.10, 09:50, п. 2, и дополнение 09:55 (правило «закрытие биржи»); решения 09:00, 09:05, 09:15 — '
         '`docs/zadaniya/2026-10-25_pereschet.md`; правило — FINAL_DESIGN, разд. 13, и `docs/tables/README_zakrytie.md`. '
         'Тест (`final_test.py --scope test`) повторно не запускался; модели не обучались; сохранённые таблицы теста не менялись.',
         '',
         'Скрипт: `python3 scripts/analysis/tables_dlya_teksta.py [--versiya 1|2|all]` (перезапускаемый; `--check` — только сверка обеих версий). '
         'Формат: «;», десятичная запятая, UTF-8 с BOM; под таблицей строки «Примечание» и «Источник». '
         'Вид строк и столбцов — как в «Диссертация_Лысенок_правки_5.docx» (табл. 3.4–3.8), табл. 3.9 — как `table_3_9_test.csv`; '
         'периоды стоят рядом блоками столбцов «<период>, H = …».',
         '',
         '## Версии',
         '',
         '- **Версия 2 — `table_3_{4..9}.csv` и приложение `table_3_pril_zakrytie.csv`.** По решению владельца 10:20 **основная — без правила** '
         '(заранее объявленный дизайн); столбцы «с правилом «закрытие биржи»» стоят рядом как **проверка устойчивости**. '
         'Порядок блоков: 2022–2026 · 2022–2026 с правилом · 2017–2026 · 2017–2026 с правилом · 2025 – 01.2026 (правило там ничего не меняет, блок один). '
         '**Правило сформулировано после первого прогона теста** (поручение 09:55), когда результаты теста были известны. '
         'В 3.8 добавлены тикеры с выигрышем и знаковый тест по тикерам — только иллюстрация: тикеры коррелированы.',
         '- **Версия 1 — `v1/table_3_{4..9}.csv`** (PR #118): без правила, блоки 2022–2026 · 2017–2026 · 2025 – 01.2026. Числа совпадают с блоками «без правила» версии 2.',
         '',
         '## Периоды и правило',
         '',
         '- **2022–2026¹ — основной (контрольный).** Годы, не участвовавшие в отборе дизайнов на разработочном наборе. '
         'Сноска: 2022–2024 — подтверждающий набор (3 открытия, 9 просмотров; использовался для выбора финала из трёх финалистов); '
         'ни разу не открывались только 2025 и 01–02.2026.',
         '- **2017–2026² — весь тест**, заранее заданный в `final_test.py` основной расчёт (справочно). 2026 — 01.01–03.02.',
         '- **2025 – 01.2026³ — не открывавшиеся** (строки 2025 и 2026 гг.; последняя дата 02.02.2026 при H = 1, 29.01 при H = 5, 12.01 при H = 22).',
         '- **⁴ Правило «закрытие биржи»:** точка t тикера исключается для всех моделей, если окно цели [t+1, t+H] задевает остановку торгов '
         '28.02–23.03.2022; проверка устойчивости, сформулировано после первого прогона теста. По тикерам — `table_3_pril_zakrytie.csv`.',
         '',
         '## Исключённые точки и эпизод февраля–марта 2022',
         '',
         f'- **Доли исключённых точек.** {facts()[0]} Знаменатель — число точек (тикер × дата) периода без правила '
         '(`table_3_{kontrol,vse}_svodka_test_bez_zakrytiya.csv`, n); разность n без правила и с правилом равна числу исключённых точек — проверено.',
         f'- **Эпизод h = 5.** {facts()[1]} Источники: `table_3_kontrol_svodka_test_{{bez_zakrytiya,s_zakrytiem}}.csv`, '
         '`table_3_kontrol_znak_test_{bez_zakrytiya,s_zakrytiem}.csv`.',
         '',
         'Эталон DM (3.8, H1) — HAR-J с поправкой (`har_corr`, s² = sm). В 3.9 — оба эталона. QLIKE = mean(y/p − ln(y/p) − 1) по парам (дата, тикер), фильтр rv_actual > 1e-12.',
         '',
         '## Источники колонок (версия 2)',
         '',
         '`<tag>` — `kontrol` (2022–2026), `vse` (2017–2026), `neotkr` (2025 – 01.2026); `<R>` — `s_zakrytiem` (с правилом) или `bez_zakrytiya` (без правила).',
         '']
    S2 = {
        '3_4': [('Модель', 'XGBoost, LightGBM, HAR-J — как в диссертации'),
                ('<блок>, H = 1/5/22: HAR-J', '`table_3_<tag>_svodka_test_<R>.csv`, «QLIKE HAR-J (а)»'),
                ('<блок>, H = 1/5/22: XGBoost, LightGBM', '`dlya_teksta/istochnik_srez_zakrytie_test.csv`, «QLIKE пул»')],
        '3_5': [('Модель', 'XGBoost, LightGBM, HAR-J; LSTM, GRU — н/д (в тесте final3 нет)'),
                ('<блок>, QLIKE H = 1', 'как табл. 3.4, H = 1'),
                ('<блок>, Δ vs XGBoost / Δ vs HAR-J', 'пересчёт из QLIKE той же строки и строки XGBoost / HAR-J, целые %')],
        '3_6': [('Модель', 'Гибридная (финал, `rv_pred`), LightGBM, XGBoost, HAR-J, HAR-J с поправкой — как `table_3_6_test.csv`'),
                ('<блок>: Гибридная, HAR-J, HAR-J с поправкой', '`table_3_<tag>_svodka_test_<R>.csv`, «QLIKE Финальная», «QLIKE HAR-J (а)», «QLIKE HAR-J (б)»'),
                ('<блок>: XGBoost, LightGBM', '`dlya_teksta/istochnik_srez_zakrytie_test.csv`, «QLIKE пул»')],
        '3_7': [('Модель', 'как `table_3_7_test.csv`'),
                ('2017–2026 без правила', '`table_3_7_test.csv`'),
                ('остальные блоки', '`dlya_teksta/istochnik_srez_zakrytie_test.csv`, «QLIKE среднее по годам»')],
        '3_8': [('H', '1, 5, 22'),
                ('<блок>, t-стат. / p-value', '`table_3_<tag>_dm_test_<R>.csv`: эталон HAR-J (б), способ «основной…», t и «p односторонний»'),
                ('<блок>, тикеров с выигрышем из 17 / p знакового теста', '`table_3_<tag>_znak_test_<R>.csv`: эталон HAR-J (б), «финал лучше» и «p знаковый односторонний»')],
        '3_9': [('H, Эталон, Способ', 'строки `table_3_9_test.csv` (способ — его подпись; лаг NW = H − 1)'),
                ('<блок>, t / p', '`table_3_<tag>_dm_test_<R>.csv` (HAR-J (а) = как в диссертации, HAR-J (б) = с поправкой), t и «p односторонний»')],
        '3_pril_zakrytie': [('все колонки', '`table_3_zakrytie_isklyucheno.csv` (`test_dop2.py`, PR #120): исключено точек, первая и последняя исключённая t по H; '
                             'последний торговый день до и первый после закрытия')],
    }
    for name in ('3_4', '3_5', '3_6', '3_7', '3_8', '3_9', '3_pril_zakrytie'):
        L += [f'### `table_{name}.csv` — {TITLE[name]}', '', '| колонка | источник |', '|---|---|']
        L += [f'| {a} | {b} |' for a, b in S2[name]]
        L += ['', f'Примечание: {foot(2, name)[0]}', '']
    L += ['## Источники колонок (версия 1, `v1/`)',
          '',
          '- 3.4–3.6, 2022–2026 и 2025 – 01.2026: Гибридная, HAR-J, HAR-J с поправкой — `table_3_kontrol_svodka_test.csv`; XGBoost, LightGBM — '
          '`dlya_teksta/istochnik_srez_test.csv`. 2017–2026 — `table_3_final_po_godam_test.csv`, «все годы (пул)».',
          '- 3.7: 2017–2026 — `table_3_7_test.csv`; остальные — `dlya_teksta/istochnik_srez_test.csv`, «QLIKE среднее по годам».',
          '- 3.8: `table_3_kontrol_dm_test.csv`, `table_3_neotkr_dm_test.csv` (эталон HAR-J (б), основной способ); 2017–2026 — `table_3_8_test.csv`.',
          '- 3.9: `table_3_kontrol_dm_test.csv`, `table_3_neotkr_dm_test.csv`; 2017–2026 — `table_3_9_test.csv`.',
          '',
          '## Что посчитано из прогнозов',
          '',
          'Только то, чего нет в сохранённых CSV: QLIKE (пул и среднее годовых) для пяти строк × периоды × H — `istochnik_srez_test.csv` (без правила) и '
          '`istochnik_srez_zakrytie_test.csv` (обе версии). Вход — `/Users/nilysenok/Desktop/Старое/PhD_Vol_trading/data/regen/test_locked/final3/final_h{1,5,22}_test.parquet` '
          '(SHA-256 сверен с FINAL_DESIGN, поле 10, в `test_dop.load`). Функции: `test_dop.load`, `test_dop.cut`, `ftp_tables._q` (без правок). '
          'Точки правила — ровно строки `table_3_zakrytie_isklyucheno.csv` (тикер, t от первой до последней исключённой; число точек сверено по каждому тикеру и итогу). '
          'В таблицы идут XGBoost и LightGBM (3.4–3.6) и средние годовых (3.7); остальные строки — для контроля.',
          '',
          f'Контроль расчёта (v1: {nchk[1]}, v2: {nchk[2]} сравнений, все OK): годовые QLIKE = `table_3_final_po_godam_test`; пул и среднее годовых 2017–2026 = '
          '`table_3_final_po_godam_test` и `table_3_7_test`; QLIKE финала, HAR-J (а), (б), лучшей одиночной и n обеих версий = `table_3_<tag>_svodka_test_<R>.csv` '
          'и `table_3_kontrol_svodka_test.csv`; без правила = срез v1; на 2025 – 01.2026 с правилом = без правила.',
          '',
          '## Отчёт о сверке',
          '',
          'Каждое число выходной таблицы заново прочитано из своего исходного CSV (по строке и колонке) и сравнено: QLIKE — после округления источника до 4 знаков, '
          't — до 2, p — до 3 («<0,001» — как строка), целые и даты — точно; пустая ячейка источника = «—». Записанный файл совпадает с таблицей, '
          'построенной из источников. Δ табл. 3.5 пересчитаны из QLIKE таблицы. Числовых ячеек без источника нет.',
          '',
          '| версия | таблица | строк | чисел = исходный CSV | Δ пересчитано | «—» | итог |', '|---|---|---|---|---|---|---|']
    for v in (2, 1):
        for x in rep[v]:
            if len(x) == 5:
                n, r, k, d, s = x
                L.append(f'| {v} | {"v1/" if v == 1 else ""}table_{n}.csv | {r} | {k} | {d} | {s} | OK |')
    nn = [x for x in rep[2] if len(x) == 2][0][1]
    L += ['', f'Версия 2: в блоке 2025 – 01.2026 каждое из {nn} чисел файла «с правилом» равно числу файла «без правила» — OK.',
          '', '«—»: DM и знаковый тест против HAR-J с поправкой при H = 22 не определены (финал = эталон); в 3.5 — Δ строки к самой себе; '
          'в приложении — даты в строке «Всего». «н/д» в 3.5 — LSTM и GRU (нет в финальном тесте).', '']
    (OUT / 'README.md').write_text('\n'.join(L), encoding='utf-8')
    print('записано', OUT / 'README.md')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--versiya', choices=['1', '2', 'all'], default='all')
    ap.add_argument('--check', action='store_true', help='только сверка записанных таблиц обеих версий')
    a = ap.parse_args()
    vs = (1, 2) if a.check or a.versiya == 'all' else (int(a.versiya),)
    nchk = {}
    if not a.check:
        FT.set_etalon(TD.ETALON)
        D = {h: TD.load(h) for h in HS}
        if 1 in vs:
            nchk[1] = srez1(D)
        if 2 in vs:
            nchk[2] = srez2(D)
    rep = {}
    for v in vs:
        tabs = build(v)
        if not a.check:
            for T in tabs:
                write(v, T)
            print(f'записаны таблицы версии {v} в {DIR[v]}')
        rep[v] = check(v, tabs)
    if not a.check and vs == (1, 2):
        readme(rep, nchk)


if __name__ == '__main__':
    main()
