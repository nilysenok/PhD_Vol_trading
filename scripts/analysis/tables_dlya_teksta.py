"""Таблицы 3.4–3.9 для текста диссертации (поручение владельца 26.10, 09:50, п. 2; дополнение 09:55 — правило «закрытие биржи»).
Вид строк и столбцов — как в «Диссертация_Лысенок_правки_5.docx» (табл. 3.4–3.8; 3.9 — DM тремя способами, как table_3_9_test).
Основной период — контрольный 2022–2026 (2022–2024 — подтверждающий набор, сноска), рядом — весь тест 2017–2026 и не открывавшиеся
2025 – 01.2026. Периоды стоят рядом блоками столбцов.

СРЕЗ ПО ПОРУЧЕНИЮ ВЛАДЕЛЬЦА ПОСЛЕ ЕДИНСТВЕННОГО ЗАПУСКА ТЕСТА (final_test.py --scope test, 26.10 08:57, коммит 002a74c).
Тест не запускается, прогнозы финала не меняются, сохранённые таблицы теста не перезаписываются. Модели здесь не обучаются; исключение — справочные
строки табл. 3.4 «LightGBM / XGBoost (чистый бустинг, с поправкой)» (поручение владельца 10:55): прогнозы обучены отдельно scripts/analysis/ref_v0_pure.py
(путь (в)) и читаются здесь из test_locked/ref_v0_pure со сверкой SHA-256; прежние строки (ноги гибрида) — приложение prilozhenie_nogi_gibrida.csv.

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


# ---------- табл. 3.4 «Сравнение моделей» (поручение владельца 10:35) ----------
IST3 = 'dlya_teksta/istochnik_sravnenie_test.csv'
REFDIR = Path('/Users/nilysenok/Desktop/Старое/PhD_Vol_trading/data/regen/test_locked/ref_nn_garch')
REF_SHA = {  # docs/otchety/2026-10-26_ref_nn_garch.md, «Хэши SHA-256 (итог, 08:15)»
    ('LSTM', 1): 'ae726e37ec98ad2547bc9d4f67f98dc69eff06ad6e484103d864614a9bb57e0d', ('LSTM', 5): 'f349ad30f6e0f56b7e71b427d687db9e05cbf2d4d65939c03aa5a58a14a9ec6e',
    ('LSTM', 22): '0a4e35a5da578519f0bb3d3ffbbaa03da2dd9423c56d7f6efeb5b46afbbaff3c', ('GRU', 1): '5f5443d7f4f05e21874118a8f6f3bc6a2f4a3be08442d23a9a696fab0328d2a3',
    ('GRU', 5): '02bf4ff2badcad5ef01d8b5198910f2c116371b38e00ac86c340e1172c293370', ('GRU', 22): 'a330b1183a95fb4dc7bb3c59fa76eb5307099804248dbf852ab8dfe0bf576924',
    ('GJR', 1): '714678801fd872bf8563295af10f24562d0639b0bf142ff21083ef3a6c3fe59f', ('GJR', 5): 'f2a486533ab7d48d48b242c3c62ccb005094d707bbb2b054a5fc7638efa3ae8b',
    ('GJR', 22): 'b435823e167ddfc6d16bb3b81890b3b321ece952f9b7096563c8d2c0f78588be', ('GARCH', 1): 'db4e7b93ac65227a99a8553535690cce8b7e173f05320297fa2cef52f9d56d9d',
    ('GARCH', 5): 'cb039e04ea69a5b01ee14ab3526ed400a9053b4baa89a8ec35e21b2df9be4c00', ('GARCH', 22): 'e5a8691ec20a507226b2754733174ecc71598646664a9a5d4fe5ea5972044831'}
# чистый бустинг (поручение владельца 10:55): scripts/analysis/ref_v0_pure.py --train, путь (в) — обучен заново по точкам ног финала
PUREDIR = Path('/Users/nilysenok/Desktop/Старое/PhD_Vol_trading/data/regen/test_locked/ref_v0_pure')
PURE_SHA = {  # test_locked/ref_v0_pure/SHA256.txt (26.10, 11:01–11:06)
    ('xgb', 1): '61ed017c31a88ff4988a0ea6bbc11dce1575d94b467e74cd078b5f6b8dc45a69', ('lgbm', 1): 'e8a8a87a19b92b5b0edcc9a4bdc9f3e54b69b3352a84957468959501946f2860',
    ('xgb', 5): 'a94bd99a819cb50d214b263ac391a4c593a05a00fc0d0925703b03ce4ee5fed6', ('lgbm', 5): '51aa7c0ac2284f7d59aeca4cda4638855e61407a221e0dfc8ebae57a78408919',
    ('xgb', 22): '18c5fc20882b550bc5b853afb278721a7da2f5062eaecb4e3106a85def7fa608', ('lgbm', 22): '95ed0609319b1e72f588c5acab9424dbcbf8390d66df4f49e296432fc75f2eb8'}
PURE_L, PURE_X = 'LightGBM (чистый бустинг, с поправкой)', 'XGBoost (чистый бустинг, с поправкой)'
NOGA_L, NOGA_X = 'нога гибрида (LightGBM на HAR-X)', 'нога гибрида (XGBoost на HAR-X)'
SRAV = {'Финальная (гибридная)': ('rv_pred', 'QLIKE Финальная'), 'HAR-J с поправкой (эталон)': ('har_corr', 'QLIKE HAR-J (б)'),
        'HAR-J как в диссертации': ('har_raw', 'QLIKE HAR-J (а)'), PURE_L: ('pure_lgbm', None), PURE_X: ('pure_xgb', None),
        'GRU': ('ref_GRU', None), 'LSTM': ('ref_LSTM', None), 'GJR-GARCH с калибровкой': ('ref_GJR', None), 'GARCH(1,1) с калибровкой': ('ref_GARCH', None)}
NOGI = {NOGA_L: ('lgb_corr', 'QLIKE Лучшая одиночная бустинга'), NOGA_X: ('xgb_corr', None)}  # прежние строки 3.4 — приложение (поручение 10:55)
BL34 = [Blk('2022–2026¹', 'kontrol', 'b'), Blk('2022–2026¹ с правилом⁴ (устойчивость)', 'kontrol', 's'),
        Blk('2017–2026²', 'vse', 'b'), Blk('2017–2026² с правилом⁴ (устойчивость)', 'vse', 's')]
SPRAV = ('Справочные модели (GRU, LSTM, GARCH, GJR) открыты на тесте для отчёта по поручению владельца 10:35 и в выборе финала не участвовали; '
         'срез после единственного запуска теста.')


def srez3(D):
    """QLIKE строк табл. 3.4 «Сравнение моделей» на строках финального теста (фильтр и сортировка test_dop.load); справочные модели
    присоединяются по (дата, тикер) без досчёта: нет прогноза — строка не входит в QLIKE модели, покрытие пишется."""
    import hashlib
    ex, nchk = isklyuchit(D)
    po = pd.read_csv(TABLES / 'table_3_final_po_godam_test_raw.csv')
    rows = []
    for h, base in D.items():
        base = base.copy()
        for m in ('GRU', 'LSTM', 'GJR', 'GARCH'):
            f = REFDIR / f'REF_{m}_h{h}_test.parquet'
            assert hashlib.sha256(f.read_bytes()).hexdigest() == REF_SHA[(m, h)], f'{f.name}: SHA-256 не совпал с отчётом ref_nn_garch'
            r = pd.read_parquet(f, columns=['date', 'ticker', 'rv_actual', 'rv_pred'])
            r['date'] = pd.to_datetime(r['date'])
            j = base[['date', 'ticker', 'rv_actual']].merge(r, how='left', on=['date', 'ticker'], suffixes=('', '_ref'), validate='1:1')
            ok = j.rv_actual_ref.notna()
            assert np.allclose(j.rv_actual[ok], j.rv_actual_ref[ok], rtol=1e-9, atol=0), f'{m} h={h}: rv_actual не совпал'
            base[f'ref_{m}'] = j.rv_pred.where(j.rv_pred > 0).values
            nchk += 1
        for m in ('xgb', 'lgbm'):  # чистый бустинг (поручение 10:55): rv_pred = rv_raw × F, F — Б0.1 sm своей модели
            f = PUREDIR / f'REF_V0_{m}_h{h}_test.parquet'
            assert hashlib.sha256(f.read_bytes()).hexdigest() == PURE_SHA[(m, h)], f'{f.name}: SHA-256 не совпал с ref_v0_pure/SHA256.txt'
            r = pd.read_parquet(f, columns=['date', 'ticker', 'step', 'rv_actual', 'rv_raw', 'F', 'rv_pred'])
            assert np.allclose(r.rv_pred, r.rv_raw * r.F, rtol=1e-15, atol=0)
            j = base[['date', 'ticker', 'step', 'rv_actual']].merge(r, how='left', on=['date', 'ticker'], suffixes=('', '_ref'), validate='1:1')
            assert j.rv_pred.notna().all() and (j.step == j.step_ref).all(), f'{f.name}: нет прогноза для части строк теста или шаг не совпал'
            assert np.allclose(j.rv_actual, j.rv_actual_ref, rtol=1e-9, atol=0), f'{f.name}: rv_actual не совпал'
            base[f'pure_{m}'] = j.rv_pred.values
            nchk += 1
        for rk, rname in RULE.items():
            DR = (base[~ex[h]] if rk == 's' else base).reset_index(drop=True)
            for tag in ('kontrol', 'vse'):
                a, b = YEARS[tag]
                P = DR[DR.year.between(a, b)]
                sv = f'table_3_{tag}_svodka_test_{rname}.csv'
                for name, (c, scol) in {**SRAV, **NOGI}.items():
                    v = P[c].notna()
                    q = FT._q(P[v], c) if v.any() else np.nan
                    if scol:  # финал, эталоны, нога LightGBM с поправкой — те же числа, что в сохранённых сводках
                        assert v.all() and round(q, 4) == num(src(sv, {'h': str(h)}, scol)) and len(P) == int(src(sv, {'h': str(h)}, 'n')), f'{sv} h={h} {name}'
                        nchk += 1
                    if c.startswith('pure_'):  # чистый бустинг — полное покрытие строк периода
                        assert v.all() and len(P) == int(src(sv, {'h': str(h)}, 'n')), f'{sv} h={h} {name}: покрытие'
                        nchk += 1
                    if c == 'xgb_corr' and tag == 'vse' and rk == 'b':
                        assert np.isclose(q, po[(po.h == h) & (po['Модель'] == 'XGBoost с поправкой') & (po['Год'] == 'все годы (пул)')].QLIKE.iloc[0], rtol=1e-12)
                        nchk += 1
                    miss = P[~v].groupby('year').size()
                    rows.append({'Период': tag, 'правило': rname, 'h': h, 'Модель': name, 'столбец прогноза': c, 'строк периода': len(P),
                                 'строк с прогнозом': int(v.sum()), 'покрытие, %': round(100 * v.mean(), 2), 'QLIKE пул': q,
                                 'нет прогноза (год: строк)': ', '.join(f'{y}: {k}' for y, k in miss.items())})
    write_df(pd.DataFrame(rows), IST3)
    print(f'срез 3.4 «Сравнение моделей»: {len(rows)} строк -> {IST3}; контроль: {nchk} сравнений OK')
    return nchk


def pokrytie():
    """Покрытие справочных моделей по IST3 (без правила) — текст для примечания и README."""
    out = []
    for m in ('GRU', 'LSTM', 'GJR-GARCH с калибровкой', 'GARCH(1,1) с калибровкой'):
        part = []
        for h in HS:
            for tag, lab in (('kontrol', '2022–2026'), ('vse', '2017–2026')):
                r = [x for x in rows_of(IST3) if x['Модель'] == m and x['h'] == str(h) and x['Период'] == tag and x['правило'] == RULE['b']][0]
                gap = f', нет: {r["нет прогноза (год: строк)"]}' if r['нет прогноза (год: строк)'] else ''
                part.append(f'H = {h}, {lab}: {r["строк с прогнозом"]} из {r["строк периода"]}{gap}')
        out.append(f'{m} — ' + '; '.join(part))
    full = all(x['покрытие, %'] in ('100', '100,0') for x in rows_of(IST3))
    return ('Покрытие справочных моделей (строк финального теста с прогнозом): ' + ('полное — прогнозы есть для всех строк обоих периодов при всех H, '
            'включая шаг 10 (01–02.2026).' if full else ' | '.join(out) + '. Пропуски не досчитывались: QLIKE модели — по строкам с прогнозом.'))


def t34n(name_='3_4', rows=None):
    T = Tab(name_, ['Модель'] + [f'{b.label}, H = {h}' for b in BL34 for h in HS])
    for name, (c, scol) in (rows or SRAV).items():
        T.start(name)
        for b in BL34:
            for h in HS:
                if scol:
                    T.cell(f'table_3_{b.tag}_svodka_test_{RULE[b.rule]}.csv', {'h': str(h)}, scol, 'q4')
                else:
                    T.cell(IST3, {'Период': b.tag, 'правило': RULE[b.rule], 'h': str(h), 'Модель': name}, 'QLIKE пул', 'q4')
        T.end()
    return T


def build(v):
    tabs = [t_models(v, '3_4' if v == 1 else '3_4_v_prezhn', ('XGBoost', 'LightGBM', 'HAR-J'), 'pool'), t35(v),
            t_models(v, '3_6', ('Гибридная', 'LightGBM', 'XGBoost', 'HAR-J', 'HAR-J с поправкой'), 'pool'),
            t_models(v, '3_7', ('HAR-J', 'HAR-J с поправкой', 'XGBoost', 'LightGBM', 'Гибридная'), 'years'), t38(v), t39(v)]
    return ([t34n()] if v == 2 else []) + tabs + ([t_pril(), t34n('pril_nogi', NOGI)] if v == 2 else [])


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


PUREV = Path('/Users/nilysenok/Desktop/Старое/PhD_Vol_trading/data/regen/ref_v0_pure')


def pure_kontrol():
    """Контроль чистого бустинга из CSV ref_v0_pure.py (--repro, --leak): всё пройдено -> текст для примечания и README."""
    rp = pd.concat([pd.read_csv(PUREV / f'repro_h{h}.csv') for h in HS])
    lk = pd.concat([pd.read_csv(PUREV / f'leak_h{h}.csv') for h in HS])
    assert rp.ok.all(), 'ref_v0_pure: контроль идентичности не пройден'
    cl, dt = lk[~lk.leak_injected], lk[lk.leak_injected]
    assert cl.val_identical.all() and cl.best_it_identical.all() and (cl.mismatch_le_T == 0).all() and (dt.mismatch_le_T > 0).all(), 'ref_v0_pure: детектор'
    e = lambda x: f'{x:.0e}'.replace('e-', '·10^−').replace('e+00', '')
    d1 = rp[rp.check.str.startswith('драйвер')].max_rel_diff.max()
    d2 = rp[rp.check.str.startswith('поправка')].max_rel_diff.max()
    return (f'Контроль: драйвер с конфигурацией ноги (с прогнозом HAR в признаках и офсетом) повторяет сохранённые val-прогнозы ноги на шагах 2 и 10 '
            f'(макс. отн. расхождение {e(d1)}), поправка Б0.1 того же кода на val ноги повторяет отношения xgb_corr/xgb_raw и lgb_corr/lgb_raw '
            f'финального тестового файла на шагах 1–10 (до {e(d2)}); детектор «будущее не влияет» (шаги 2, 9, 10, оба бустинга, все H: шум во всех '
            f'столбцах строк после даты T внутри тестового года) — прогнозы val и теста до T совпали побайтно, подложенная утечка поймана во всех '
            f'{len(dt)} случаях.')


def pure_floor():
    """Строки теста, где прогноз чистого бустинга (до поправки) упёрся в нижнюю границу 1e-10, — текст для примечания (из файлов прогнозов)."""
    out = []
    for h in HS:
        for m, nm in (('lgbm', 'LightGBM'), ('xgb', 'XGBoost')):
            d = pd.read_parquet(PUREDIR / f'REF_V0_{m}_h{h}_test.parquet', columns=['date', 'ticker', 'step', 'rv_raw'])
            f = d[d.rv_raw <= 1.0000001e-10]
            if len(f):
                g = f.assign(y=f.date.dt.year).groupby(['step', 'y', 'ticker']).size()
                out.append(f'{nm} при H = {h} — {len(f)} строк теста (' + ', '.join(f'шаг {s} ({y}) {t} {n}' for (s, y, t), n in g.items()) + ')')
    if not out:
        return ''
    return ('⚠ Прогноз на нижней границе 1e-10: ' + '; '.join(out) + '. Причина — linear_tree (λ = 1, как у ноги D12_h22): линейные листья '
            'экстраполируют на значения признаков вне диапазона обучающего окна, без офсета HAR это не гасится; у val-лет таких строк нет. QLIKE этой '
            'строки определяется этими точками и как оценка модели не информативен. Менять спецификацию после просмотра теста — выбор по тесту, '
            'поэтому число оставлено как есть; другая спецификация — только по решению владельца и с пометкой. ')


PURE_NOTE = ('«LightGBM / XGBoost (чистый бустинг, с поправкой)» — справочные строки по поручению владельца 10:55, получены путём (в): сохранённых '
             'тестовых прогнозов чистого бустинга с гиперпараметрами и признаками ног финала нет (V0-дизайны в test_locked — с другими точками или '
             'признаками), сохранённых обученных моделей нет; модели обучены заново (scripts/analysis/ref_v0_pure.py) по точкам гиперпараметров ног финала: '
             'H = 1 — нога D22_h1 (среднее топ-5 точек шага на шагах 1–6, точка шага 6 на шагах 7–10), H = 5 — нога D12_h5, H = 22 — нога D12_h22 '
             '(её xgb_corr/lgb_corr стояли в строках H = 22); те же признаки и исключения признаков, без прогноза HAR-J/HAR-X в признаках, без офсета и '
             'остаточной схемы; цель — ln RV (H = 1) или ln среднего RV за [t+1, t+H] (H = 5, 22); винзоризация цели (H = 1), Huber (H = 5) и linear_tree '
             '(LightGBM, H = 22) — как у ноги; те же 10 шагов скользящей валидации (обучение до Y−2, ранняя остановка на Y−1, шаг 10 — обучение до 2024, '
             'val 2025, прогноз 01–02.2026), seed 42; поправка Б0.1: s² sm по val-остаткам своей модели, на тесте шага k — оценка шага k, как у финала. '
             'Без подбора и без сравнения вариантов; первое открытие этих прогнозов на тесте (26.10). XGBoost при H = 5 — сам XGBoost (в финале '
             'XGB-слот при H = 5 — среднее V9-XGBoost и V9-LightGBM). ↺ У XGBoost при H = 5 (pseudo-Huber) цель центрирована средним ln цели обучающего '
             'окна шага: без этого XGBoost 2.1.4 оценивает base_score ≈ −555 и останавливается на итерации 0 на всех шагах (прогноз — константа); '
             'выявлено на val до открытия теста. Прогнозы и SHA-256 — test_locked/ref_v0_pure. Прежние строки «LightGBM / XGBoost одиночный с поправкой» '
             '(ноги гибрида) — приложение prilozhenie_nogi_gibrida.csv. ')
NOGI_NOTE = ('Ноги гибридного финала × F (lgb_corr, xgb_corr файлов test_locked/final3) — прежние строки табл. 3.4 «LightGBM / XGBoost одиночный '
             'с поправкой», перенесены в приложение по поручению владельца 10:55 (в табл. 3.4 их заменил чистый бустинг). Название — по поручению; '
             'основа ноги: H = 1 — HAR-X⁺ (Д8.12, D22_h1, нога top5: прогноз ln HAR-X⁺ в признаках); H = 5 — HAR-J (V9, D12_h5: бустинг на остатке '
             'ln RV − ln HAR-J и прогноз HAR-J в признаках; нога wrap2: XGBoost-слот = среднее V9-XGBoost и V9-LightGBM, LightGBM-слот — V9-LightGBM); '
             'H = 22 — HAR-J (V9, D12_h22, нога single; финал H = 22 — HAR-J с поправкой). При H = 1 и 5 XGBoost-нога совпадает с финалом. '
             'LightGBM-нога с поправкой — «лучшая одиночная бустинга», зафиксированная до теста. Те же ряды, что «LightGBM» и «XGBoost» табл. 3.5–3.7, '
             'но с поправкой. QLIKE по всем парам (дата, тикер) периода финального теста (фильтр rv_actual > 1e-12).')


def foot(v, name):
    if v == 2 and name == '3_4':
        return (f'{SPRAV} QLIKE по всем парам (дата, тикер) периода финального теста (фильтр rv_actual > 1e-12), без выбора. «Финальная» — финальный дизайн '
                '(rv_pred): H = 1 — D22_h1, H = 5 — D12_h5, H = 22 — HAR-J с поправкой. ' + PURE_NOTE + pure_floor() + pure_kontrol() + ' GRU и LSTM — ансамбль трёх seed с '
                'поправкой Б0.1; GJR-GARCH и GARCH(1,1) — калибровка a·σ²·F (OLS по логарифмам по тикеру на годе шага t−1); основной rv_pred '
                'справочных прогнозов, объявленный в отчёте 2026-10-26_ref_nn_garch до теста. ' + pokrytie(),
                'финал, HAR-J с поправкой, HAR-J как в диссертации — table_3_{kontrol,vse}_svodka_test_{bez_zakrytiya,s_zakrytiem}.csv; '
                'чистый бустинг и справочные — dlya_teksta/istochnik_sravnenie_test.csv (из прогнозов test_locked/ref_v0_pure — обучены по поручению 10:55, '
                'SHA-256 сверен, — и сохранённых test_locked/ref_nn_garch).')
    if v == 2 and name == 'pril_nogi':
        return (NOGI_NOTE, 'нога LightGBM — table_3_{kontrol,vse}_svodka_test_{bez_zakrytiya,s_zakrytiem}.csv («QLIKE Лучшая одиночная бустинга»); '
                'нога XGBoost — dlya_teksta/istochnik_sravnenie_test.csv (из сохранённых прогнозов test_locked/final3).')
    F = {
        '3_4': ('QLIKE по всем парам (дата, тикер) периода. ' + X_NOTE, SRC_POOL[v]),
        '3_4_v_prezhn': ('Прежняя табл. 3.4 (до поручения 10:35). QLIKE по всем парам (дата, тикер) периода. ' + X_NOTE, SRC_POOL[v]),
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
    '3_4': 'Таблица 3.4. Сравнение моделей: QLIKE на контрольном периоде 2022–2026 и на 2017–2026',
    '3_4_v_prezhn': 'Прежняя таблица 3.4. QLIKE моделей бустинга и HAR-J (контрольный период 2022–2026; справочно 2017–2026 и 2025 – 01.2026)',
    '3_5': 'Таблица 3.5. QLIKE при H = 1: бустинг и HAR-J; рекуррентные сети в финальный тест не входят',
    '3_6': 'Таблица 3.6. Сводное сравнение моделей (контрольный период 2022–2026; справочно 2017–2026 и 2025 – 01.2026)',
    '3_7': 'Таблица 3.7. Среднее годовых QLIKE моделей (2022–2026; справочно 2017–2026 и 2025 – 01.2026)',
    '3_8': 'Таблица 3.8. Тест Диболда–Мариано: финальная (гибридная) модель против HAR-J с поправкой',
    '3_9': 'Таблица 3.9. Тест Диболда–Мариано тремя способами против двух эталонов',
    '3_pril_zakrytie': 'Приложение к табл. 3.4–3.9. Точки, исключённые правилом «закрытие биржи», по H и тикерам',
    'pril_nogi': 'Приложение к табл. 3.4. Ноги гибридного финала (XGBoost/LightGBM на HAR-X) с поправкой: QLIKE на 2022–2026 и 2017–2026',
}
FNAME = {'pril_nogi': 'prilozhenie_nogi_gibrida.csv'}


def fname(name):
    return FNAME.get(name, f'table_{name}.csv')


def write(v, T):
    note, source = foot(v, T.name)
    n = NOTE + (note4() if v == 2 else '') if T.name != '3_pril_zakrytie' else facts()[0]
    tail = [[], [f'Примечание. {note} {n} {SRK[v]}'.replace('  ', ' ')], [f'Источник: составлено автором; {source}']]
    DIR[v].mkdir(parents=True, exist_ok=True)
    (DIR[v] / fname(T.name)).write_bytes(to_bytes([T.header] + T.rows + tail))


# ---------- сверка ----------
def read_out(v, name):
    t = (DIR[v] / fname(name)).read_bytes()
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
        print(f'сверка v{v} {fname(T.name)}: {len(body)} строк, {k} чисел = исходный CSV, Δ {nd}, «—» {dash} — OK')
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
         'Тест (`final_test.py --scope test`) повторно не запускался; сохранённые таблицы теста не менялись; модели не обучались, кроме справочных строк '
         'чистого бустинга табл. 3.4 (поручение владельца 10:55, `scripts/analysis/ref_v0_pure.py`, путь (в) — см. ниже).',
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
         f'- **Табл. 3.4 «Сравнение моделей» (поручение владельца 10:35) — `table_3_4.csv`.** {SPRAV} Прежняя табл. 3.4 '
         '(бустинг без поправки и HAR-J) переименована в `table_3_4_v_prezhn.csv`, числа не менялись.',
         '- **Поручение владельца 10:55 — чистый бустинг в табл. 3.4.** Строки «LightGBM (чистый бустинг, с поправкой)» и «XGBoost (чистый бустинг, '
         'с поправкой)» заменили «LightGBM / XGBoost одиночный с поправкой» (это были ноги гибрида). Прежние строки переименованы в «нога гибрида '
         '(LightGBM на HAR-X)» и «нога гибрида (XGBoost на HAR-X)» и перенесены в приложение `prilozhenie_nogi_gibrida.csv` (числа не менялись). '
         '**Путь получения чистого бустинга — (в):** (а) сохранённых тестовых прогнозов с теми же точками и признаками нет (V0-дизайны в `test_locked` — '
         'D12_h1_V0, D17F_*_V0*, D25_7_V0_h1 — с другими точками или признаками); (б) сохранённых обученных моделей нет; (в) обучены заново по точкам ног '
         'финала, seed 42, без подбора и без сравнения вариантов. **Это первое открытие этих прогнозов на тесте (26.10).** Строки справочные. '
         + pure_floor(),
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
        '3_4': [('Модель', 'финал, HAR-J с поправкой (эталон), HAR-J как в диссертации, LightGBM и XGBoost (чистый бустинг, с поправкой; поручение 10:55), '
                 'GRU, LSTM, GJR-GARCH и GARCH(1,1) с калибровкой — по поручению владельца 10:35, без выбора'),
                ('блоки', '2022–2026 · 2022–2026 с правилом · 2017–2026 · 2017–2026 с правилом; H = 1/5/22'),
                ('финал, HAR-J (б), HAR-J (а)', '`table_3_<tag>_svodka_test_<R>.csv`: «QLIKE Финальная», «QLIKE HAR-J (б)», «QLIKE HAR-J (а)»'),
                ('чистый бустинг LightGBM / XGBoost', '`dlya_teksta/istochnik_sravnenie_test.csv`, «QLIKE пул»; прогнозы — '
                 '`<общий data>/regen/test_locked/ref_v0_pure/REF_V0_{lgbm,xgb}_h{1,5,22}_test.parquet`, столбец rv_pred = rv_raw × F (SHA-256 — '
                 '`ref_v0_pure/SHA256.txt`, сверяется в `srez3`)'),
                ('GRU, LSTM, GJR, GARCH', '`dlya_teksta/istochnik_sravnenie_test.csv`, «QLIKE пул» (там же покрытие по строкам и годам)')],
        'pril_nogi': [('Модель', 'нога гибрида (LightGBM на HAR-X), нога гибрида (XGBoost на HAR-X) — прежние строки табл. 3.4 «LightGBM / XGBoost '
                       'одиночный с поправкой» (поручение 10:55)'),
                      ('блоки', 'как табл. 3.4'),
                      ('нога LightGBM', '`table_3_<tag>_svodka_test_<R>.csv`, «QLIKE Лучшая одиночная бустинга» (lgb_corr)'),
                      ('нога XGBoost', '`dlya_teksta/istochnik_sravnenie_test.csv`, «QLIKE пул» (xgb_corr из test_locked/final3)')],
        '3_4_v_prezhn': [('Модель', 'XGBoost, LightGBM, HAR-J — как в диссертации (прежняя табл. 3.4, файл `table_3_4.csv` до поручения 10:35)'),
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
    for name in ('3_4', 'pril_nogi', '3_4_v_prezhn', '3_5', '3_6', '3_7', '3_8', '3_9', '3_pril_zakrytie'):
        L += [f'### `{fname(name)}` — {TITLE[name]}', '', '| колонка | источник |', '|---|---|']
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
          '`istochnik_sravnenie_test.csv` (табл. 3.4 «Сравнение моделей» и приложение с ногами гибрида): QLIKE пул и покрытие одиннадцати строк × '
          '2022–2026 / 2017–2026 × обе версии × H. Чистый бустинг — `<общий data>/regen/test_locked/ref_v0_pure/REF_V0_{lgbm,xgb}_h{1,5,22}_test.parquet` '
          '(SHA-256 сверен; присоединён по (дата, тикер), шаг и rv_actual совпали, покрытие полное). ' + pure_kontrol() + ' '
          'Хэши тестовых файлов: ' + '; '.join(f'`REF_V0_{m}_h{h}_test.parquet` `{s}`' for (m, h), s in PURE_SHA.items()) + '. '
          'Справочные — `/Users/nilysenok/Desktop/Старое/PhD_Vol_trading/data/regen/test_locked/ref_nn_garch/REF_{GRU,LSTM,GJR,GARCH}_h{1,5,22}_test.parquet` '
          '(SHA-256 сверен с `docs/otchety/2026-10-26_ref_nn_garch.md`), столбец rv_pred; присоединены к строкам финального теста по (дата, тикер), '
          'rv_actual совпал. Нет прогноза — строка не входит в QLIKE модели, покрытие указано, ничего не досчитывалось. ' + pokrytie(),
          '',
          f'Контроль расчёта (v1: {nchk[1]}, v2: {nchk[2]}, табл. 3.4: {nchk[3]} сравнений, все OK): для табл. 3.4 и приложения с ногами QLIKE финала, обоих HAR-J, '
          'ноги LightGBM с поправкой и n обеих версий = `table_3_<tag>_svodka_test_<R>.csv`, нога XGBoost с поправкой 2017–2026 = `table_3_final_po_godam_test`; '
          'чистый бустинг — SHA-256, полное покрытие, совпадение шага и rv_actual, rv_pred = rv_raw × F. Годовые QLIKE = `table_3_final_po_godam_test`; пул и среднее годовых 2017–2026 = '
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
                L.append(f'| {v} | {"v1/" if v == 1 else ""}{fname(n)} | {r} | {k} | {d} | {s} | OK |')
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
            nchk[3] = srez3(D)
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
