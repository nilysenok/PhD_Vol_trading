#!/usr/bin/env python3
"""R2: второе открытие теста — только строки гибрида второго раунда (протокол docs/zadaniya/2026-10-26_vtoroy_raund_gibrida.md,
«Итог раунда и второе открытие», п. 3). Автор — агент О. До открытия проверяется только --selftest (синтетика, тест не читается).

python3 scripts/analysis/final_test_r2.py --selftest
python3 scripts/analysis/final_test_r2.py --scope test --src data/regen/test_locked/R2/final --first-close-tag s_zakrytiem
                                                                                  # ОДИН раз, дирижёр, после «да» владельца

Вход:
  <src>/final_r2_h{h}_test.parquet — date, ticker, year, step, rv_pred, design (собирает r2_sborka_test.py; колонки rv_actual НЕТ — assert);
  h без файла — на h остаётся F_h, строк R2 нет;
  замороженный первый тест: test_locked/final3/final_h{h}_test.parquet (SHA-256 сверяется с test_dop.SHA = FD поле 10) — не пересчитывается.
Счёт: строки (date, ticker, step) R2 и первого теста совпадают один в один. Кадр = первый тест (фильтр rv_actual > 1e-12, порядок date,
ticker — как final_test.run / test_dop.load), rv_pred := R2, hyb_r1 := гибрид раунда 1.
Таблицы — тем же кодом, что первый тест (импорт final_test.run, ftp_tables, test_dop2; final_test.py и test_dop2.py как скрипты не
запускаются). MCS — на наборе {замороженные + R2 + гибрид раунда 1}.
  основная версия (без правила): table_3_*_test_r2.csv (как final_test.run: годы, 3.2–3.9, MZ, тикеры, подпериоды, терцили, MCS, охват)
     и table_3_{vse,kontrol,neotkr}_{svodka,dm,mcs,tickers,subperiods,znak}_test_r2_bez_zakrytiya.csv (как test_dop2: 2017–2026,
     2022–2026 — основной, 2025 – 01/02.2026);
  версия с правилом «закрытие биржи» (рядом, проверка устойчивости; решение владельца 10:20): table_3_*_test_r2_zakrytie.csv и
     table_3_{vse,kontrol,neotkr}_*_test_r2_s_zakrytiem.csv; исключённые точки — table_3_zakrytie_isklyucheno_test_r2.csv.
  Дополнительно: table_3_periody_* (QLIKE/RMSE/MAE по трём периодам) и table_3_dm_vs_F_* (справочно — против F_h).
Правило закрытия биржи — маски test_dop2.masks + first_trade (тот же код, что у первого теста): точка t исключается, если окно цели
[t+1, t+h] задевает остановку торгов MOEX 28.02–23.03.2022; по тексту дирижёра (26.10) это ровно h точек на тикер — последние h торговых
дней до 25.02.2022; расхождение кода с текстом печатается и пишется в таблицу исключённых точек (раздел 13 FD агент О не читал).
Сверка: ячейки эталонов (строки и колонки без гибрида) побайтно совпадают с таблицами первого теста:
  основная — с table_3_*_test_raw.csv (вывод final_test.run байт в байт; table_3_*_test.csv переформатированы tables_test_format.py)
  и table_3_*_test_bez_zakrytiya.csv; с правилом — с table_3_*_test_<--first-close-tag>.csv
  (s_zakrytiem). Расхождение — СТОП (STOP_R2.txt, код выхода 3). Таблицы первого теста открывает только этот скрипт и только при открытии.
Замок: <src>/OPENED_R2.json пишется ДО чтения теста; если он есть — отказ (одно открытие)."""
import argparse, hashlib, json, subprocess, sys, tempfile
from contextlib import contextmanager
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import final_test as FTST  # noqa: E402  код таблиц первого теста (импорт; скрипт не запускается)
import ftp_tables as FT  # noqa: E402
import test_dop as TD  # noqa: E402  запись таблиц (write_named) и SHA замороженных файлов
import test_dop2 as TD2  # noqa: E402  правило закрытия биржи и таблицы периодов — тем же кодом

REPO = HERE.parent.parent
MAIN = Path('/Users/nilysenok/Desktop/Старое/PhD_Vol_trading')
FROZEN = MAIN / 'data/regen/test_locked/final3'
TABLES = REPO / 'docs/tables'
KEY = ['date', 'ticker', 'step']
R2_COLS = ['date', 'ticker', 'year', 'step', 'rv_pred', 'design']
PERIODY = {'2022–2026 (основной)': (2022, 2026), '2025 – 01/02.2026': (2025, 2026), '2017–2026 (справочно)': (2017, 2026)}
HALT = (pd.Timestamp('2022-02-28'), pd.Timestamp('2022-03-23'))
TAG, TAG_Z = 'test_r2', 'test_r2_zakrytie'
RULES = {'bez': 'bez_zakrytiya', 'pravilo': 's_zakrytiem'}
NOTE_R2 = 'второе открытие (R2): строки гибрида раунда 2; эталоны — из первого прогона без пересчёта'


class Stop(Exception):
    pass


def now():
    return subprocess.run(['date', '+%Y-%m-%d %H:%M:%S'], capture_output=True, text=True).stdout.strip()


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def zakrytie(df, h):
    """Правило по тексту дирижёра (контроль к test_dop2.masks): на тикер последние h торговых дат < 28.02.2022. Строк внутри остановки
    быть не должно (assert). Дата строки — точка отсчёта t."""
    d = pd.to_datetime(df.date)
    inside = d.between(*HALT)
    assert not inside.any(), f'h={h}: {int(inside.sum())} строк внутри остановки торгов 28.02–23.03.2022'
    pre = d < HALT[0]
    last = df[pre].assign(_d=d[pre]).groupby('ticker')._d.transform(lambda s: s.rank(method='first', ascending=False) <= h)
    m = pd.Series(False, index=df.index)
    m.loc[last.index] = last.astype(bool)
    return m.values


def merge_h(src, frozen, h, check_sha=True):
    """Кадр h: первый тест + R2. Проверки: SHA замороженного файла; в R2 нет rv_actual; строки совпадают один в один."""
    f = src / f'final_r2_h{h}_test.parquet'
    r2 = pd.read_parquet(f)
    assert 'rv_actual' not in r2.columns, f'{f.name}: в файле R2 не должно быть rv_actual'
    assert set(R2_COLS) <= set(r2.columns), f'{f.name}: нет колонок {set(R2_COLS) - set(r2.columns)}'
    ff = frozen / f'final_h{h}_test.parquet'
    if check_sha and sha(ff) != TD.SHA[h]:
        raise Stop(f'{ff}: SHA-256 не совпал с FINAL_DESIGN (поле 10)')
    fz = pd.read_parquet(ff)
    if len(r2) != len(fz) or r2.duplicated(KEY).any():
        raise Stop(f'h={h}: строк R2 {len(r2)}, первого теста {len(fz)} (или дубли ключа)')
    m = fz.rename(columns={'rv_pred': 'hyb_r1', 'design': 'design_r1'}).merge(r2[KEY + ['rv_pred', 'design']], on=KEY, how='left', validate='1:1')
    if m.rv_pred.isna().any():
        raise Stop(f'h={h}: {int(m.rv_pred.isna().sum())} строк первого теста без прогноза R2')
    return m[m.rv_actual > 1e-12].sort_values(['date', 'ticker']).reset_index(drop=True)  # как test_dop.load / final_test.run


def masks_first(frames, ft=None):
    """Маски исключения тем же кодом, что первый тест (test_dop2.masks); контроль — правило по тексту дирижёра (zakrytie)."""
    TD2.FT_TICKERS = tuple(sorted(next(iter(frames.values())).ticker.unique()))
    ex, X = TD2.masks(frames, ft if ft is not None else TD2.first_trade())
    diff = {}
    for h, df in frames.items():
        try:
            diff[h] = int((ex[h] != zakrytie(df, h)).sum())
        except AssertionError as e:
            diff[h] = str(e)
        print(f'правило закрытия биржи, h={h}: исключено {int(ex[h].sum())} строк; расхождение с текстом дирижёра (h точек на тикер): {diff[h]}',
              flush=True)
    return ex, X.assign(rashozhdenie_s_tekstom=X.h.map(diff))


@contextmanager
def r2_env(hs, out_tables):
    """Только h с победителем R2; MCS на наборе {замороженные + R2 + гибрид раунда 1}; запись test_dop.write_named — в out_tables."""
    hs0, mods0, tab0 = FTST.HS, dict(FT.MODS), TD.TABLES
    try:
        FTST.HS = hs
        FT.set_etalon(FTST.ETALON)
        FT.MODS['Гибридная раунда 1'] = 'hyb_r1'
        TD.TABLES = Path(out_tables)
        yield
    finally:
        FTST.HS, TD.TABLES = hs0, tab0
        FT.MODS.clear(); FT.MODS.update(mods0)


def period_tables(fr, rule, out_tables, suffix, B, r2=True):
    """Таблицы периодов vse / kontrol / neotkr — как test_dop2.main (те же функции). suffix: test_r2_<правило> (у первого теста — test_<правило>)."""
    rlab = TD2.RULE[rule]
    tab0 = TD.TABLES
    TD.TABLES = Path(out_tables)
    Path(out_tables).mkdir(parents=True, exist_ok=True)
    try:
        for tag, (label, a, b, sub) in TD2.PER.items():
            P = {h: df[df.year.between(a, b)].reset_index(drop=True) for h, df in fr.items()}
            h1 = tag == 'kontrol'
            T = {'dm': TD2.dm_rows(P, label, rlab, h1), 'mcs': FT.mcs_table(P, B).assign(Период=label, правило=rlab),
                 'tickers': FT.tickers_table(P).assign(Период=label, правило=rlab),
                 'subperiods': pd.DataFrame([FT._block(h, nm, 'Подпериод', df[df.year.between(x, y)]) for h, df in P.items()
                                             for nm, (x, y) in sub.items()]).assign(Период=label, правило=rlab),
                 'znak': TD2.znak(P, label, rlab)}
            S = TD2.svodka(P, label, rlab, T['mcs'], h1, T['znak'])
            if r2:  # у первого теста финал h=22 = эталон, и svodka это прописывает; у R2 — нет: DM и H1 h=22 считаются как у прочих h
                for i in S.index[S.h == 22]:
                    S.loc[i, ['DM t vs HAR-J (б)', 'p vs HAR-J (б)']] = [FT._dm_cols(P[22], 22)['DM t vs HAR-J (б)'], FT._dm_cols(P[22], 22)['p vs HAR-J (б)']]
                    if h1:
                        S.loc[i, 'H1 (основной DM против HAR-J (б), 5%)'] = 'принята' if FT.S.dm_main(P[22], FTST.ETALON, 22)['p_one'] < 0.05 else 'не принята'
                S['пометка'] = NOTE_R2
            T['svodka'] = S
            for part, df in T.items():
                TD.write_named(df, f'table_3_{tag}_{part}_{suffix}.csv')
    finally:
        TD.TABLES = tab0


def periody(frames):
    rows = []
    for h, df in frames.items():
        for nm, (a, b) in PERIODY.items():
            g = df[df.year.between(a, b)]
            if g.empty:
                continue
            for m, c in {**FTST.MODELS, 'Гибридная раунда 1 (F_h)': 'hyb_r1'}.items():
                rows.append(dict(h=h, Период=nm, Модель=m, n=len(g), **{k: FTST.fmt(v, k) for k, v in FTST.metrics(g.rv_actual.values, g[c].values).items()}))
    return pd.DataFrame(rows)


def dm_vs_F(frames):
    out = []
    for h, df in frames.items():
        out += [dict(H=h, эталон='гибрид раунда 1 (F_h), справочно', **r) for r in FTST.dm(df, 'hyb_r1', h)]
    return pd.DataFrame(out)


# ---------------------------------------------------------------- сверка эталонов с первым тестом
# (имя таблицы, разделитель, ключ, колонка-признак гибрида, значения гибрида, исключаемые колонки (префиксы), только эти колонки)
HYB = ('Гибридная', 'Гибридная (финал)', 'Финальная', 'Гибридная раунда 1')
SVERKA = [
    ('3_2', ',', ['Год'], None, (), (), None),
    ('3_3', ',', ['Горизонт'], None, (), (), None),
    ('3_4', ',', ['Модель'], None, (), (), None), ('3_4_rmse', ',', ['Модель'], None, (), (), None), ('3_4_mae', ',', ['Модель'], None, (), (), None),
    ('3_6', ',', ['Модель'], 'Модель', HYB, (), None), ('3_6_rmse', ',', ['Модель'], 'Модель', HYB, (), None),
    ('3_6_mae', ',', ['Модель'], 'Модель', HYB, (), None),
    ('3_7', ',', ['Модель'], 'Модель', HYB, (), None), ('3_7_rmse', ',', ['Модель'], 'Модель', HYB, (), None),
    ('3_7_mae', ',', ['Модель'], 'Модель', HYB, (), None),
    ('3_final_po_godam', ',', ['h', 'Модель', 'Год'], 'Модель', HYB, (), None),
    ('3_raskladka', ',', ['h', 'Год', 'Строка'], 'Строка', HYB, (), None),
    ('3_mz', ';', ['h', 'Шкала', 'Модель'], 'Модель', HYB, (), None),
    ('3_mcs', ';', ['h', 'Модель'], 'Модель', HYB, ('p_MCS', 'в MCS'), None),
]
ETAL_Q = ['QLIKE HAR-J (а)', 'QLIKE HAR-J (б)', 'QLIKE Лучшая одиночная бустинга']
PER_KEYS = {'svodka': (['Период', 'правило', 'h'], None, ['n', 'дней'] + ETAL_Q),
            'subperiods': (['Период', 'правило', 'h', 'Подпериод'], None, ['n', 'дней'] + ETAL_Q),
            'tickers': (['Период', 'правило', 'h', 'Тикер'], None, ['QLIKE HAR-J (б)']),
            'mcs': (['Период', 'правило', 'h', 'Модель'], 'Модель', ['средний дневной QLIKE', 'дней'])}  # dm, znak — зависят от финала


def compare(a, b, key, col, only, excl, hs, name):
    bad, n = [], 0
    if col:
        a, b = a[~a[col].isin(HYB)], b[~b[col].isin(HYB)]
    if 'h' in key:
        a = a[a.h.isin([str(h) for h in hs])]
    if 'Горизонт' in key:
        a = a[a['Горизонт'].isin([f'H = {h}' for h in hs])]
    cols = only or [c for c in b.columns if c in a.columns and c not in key and not any(c.startswith(e) for e in excl)]
    m = a.merge(b, on=key, how='inner', suffixes=('_a', '_b'))
    if len(m) != len(b) or len(m) == 0 or len(m) != len(a):
        return [f'{name}: строк эталонов в R2 {len(b)}, в первом тесте {len(a)}, совпало ключей {len(m)}'], 0
    for c in cols:
        if c + '_a' not in m:
            bad.append(f'{name}: нет колонки {c}')
            continue
        diff = m[m[c + '_a'].fillna('') != m[c + '_b'].fillna('')]
        n += len(m)
        if len(diff):
            bad.append(f'{name}/{c}: {len(diff)} ячеек, напр. {diff.iloc[0][key].tolist()} {diff.iloc[0][c + "_a"]} != {diff.iloc[0][c + "_b"]}')
    return bad, n


def sverka(first_dir, r2_dir, hs, tag_first='test', tag_r2=TAG):
    """Таблицы final_test.run: строковое (побайтное) совпадение ячеек эталонов."""
    bad, n = [], 0
    for name, sep, key, col, hyb, excl, only in SVERKA:
        a = pd.read_csv(first_dir / f'table_{name}_{tag_first}.csv', sep=sep, dtype=str, encoding='utf-8-sig')
        b = pd.read_csv(r2_dir / f'table_{name}_{tag_r2}.csv', sep=sep, dtype=str, encoding='utf-8-sig')
        bb, k = compare(a, b, key, col, only, excl, hs, name)
        bad += bb; n += k
    return bad, n


def sverka_per(first_dir, r2_dir, hs, rule, first_rule_tag):
    """Таблицы периодов test_dop2 (vse / kontrol / neotkr): ячейки эталонов против table_3_<период>_<часть>_test_<first_rule_tag>.csv."""
    bad, n = [], 0
    for tag in TD2.PER:
        for part, (key, col, only) in PER_KEYS.items():
            a = pd.read_csv(first_dir / f'table_3_{tag}_{part}_test_{first_rule_tag}.csv', sep=';', dtype=str, encoding='utf-8-sig')
            b = pd.read_csv(r2_dir / f'table_3_{tag}_{part}_test_r2_{rule}.csv', sep=';', dtype=str, encoding='utf-8-sig')
            bb, k = compare(a, b, key, col, only, (), hs, f'{tag}_{part}')
            bad += bb; n += k
    return bad, n


def run_r2(src, frozen, first_tables, out_tables, feat, B, lock=True, need_fd=True, close='obe', first_tag='test_raw',
           first_close_tag='s_zakrytiem', ft=None, check_sha=True):
    """close: 'obe' (по умолчанию) — основная версия и версия с правилом закрытия биржи; 'bez' / 'pravilo' — одна.
    ft — {тикер: (последний торговый день до 28.02.2022, первый после 23.03.2022)} для синтетики; None — test_dop2.first_trade()."""
    src, frozen, first_tables, out_tables = map(Path, (src, frozen, first_tables, out_tables))
    if need_fd and not (REPO / 'docs/otchety/FINAL_DESIGN_round2.md').exists():
        raise Stop('нет docs/otchety/FINAL_DESIGN_round2.md — открытие только после него и «да» владельца')
    hs = tuple(h for h in FTST.HS if (src / f'final_r2_h{h}_test.parquet').exists())
    if not hs:
        raise Stop(f'в {src} нет ни одного final_r2_h{{h}}_test.parquet — открывать нечего (на всех h остаётся F_h)')
    lk = src / 'OPENED_R2.json'
    if lock:
        if lk.exists():
            raise Stop(f'{lk} уже есть: второе открытие было ({lk.read_text()[:200]})')
        lk.write_text(json.dumps(dict(time=now(), h=hs, close=close, src={f.name: sha(f) for f in sorted(src.glob('final_r2_h*_test.parquet'))},
                                      code=sha(__file__)), ensure_ascii=False, indent=1))
    out_tables.mkdir(parents=True, exist_ok=True)
    frames = {h: merge_h(src, frozen, h, check_sha) for h in hs}
    versions = []
    if close in ('obe', 'bez'):
        versions.append(('bez', TAG, frames, first_tag, 'bez_zakrytiya'))
    if close in ('obe', 'pravilo'):
        ex, X = masks_first(frames, ft)
        with r2_env(hs, out_tables):
            TD.write_named(X.assign(пометка=NOTE_R2), 'table_3_zakrytie_isklyucheno_test_r2.csv')
        versions.append(('pravilo', TAG_Z, {h: df[~ex[h]].reset_index(drop=True) for h, df in frames.items()}, None, first_close_tag))
    res = {}
    for ver, tag, fr, ftag3, ftagP in versions:
        mdir = Path(tempfile.mkdtemp(prefix=f'r2_merged_{ver}_'))
        for h, m in fr.items():
            m.to_parquet(mdir / f'final_h{h}_test.parquet', index=False)
        with r2_env(hs, out_tables):
            FTST.run(mdir, out_tables, tag=tag, feat=feat, B=B)
            period_tables(fr, RULES[ver], out_tables, f'test_r2_{RULES[ver]}', B)
        periody(fr).to_csv(out_tables / f'table_3_periody_{tag}.csv', index=False)
        dm_vs_F(fr).to_csv(out_tables / f'table_3_dm_vs_F_{tag}.csv', index=False)
        bad, n = [], 0
        if ftag3:
            b3, n3 = sverka(first_tables, out_tables, hs, tag_first=ftag3, tag_r2=tag)
            bad += b3; n += n3
        if ftagP:
            bp, npp = sverka_per(first_tables, out_tables, hs, RULES[ver], ftagP)
            bad += bp; n += npp
        else:
            print(f'{ver}: таблицы периодов не сверялись — не задан тег первого теста', flush=True)
        if bad:
            (out_tables / 'STOP_R2.txt').write_text(f'версия {ver}\n' + '\n'.join(bad) + '\n')
            raise Stop(f'версия {ver}: эталоны не совпали с первым тестом:\n' + '\n'.join(bad[:10]))
        print(f'{ver}: сверка эталонов с первым тестом ({ftag3 or "—"} / {ftagP or "—"}): {n} ячеек, расхождений нет; h = {hs}', flush=True)
        res[ver] = n
    return hs, res


# ---------------------------------------------------------------- самопроверка (синтетика; тест не читается)
def selftest():
    import ftp_selftest as FS
    rng = np.random.default_rng(26)
    tmp = Path(tempfile.mkdtemp(prefix='r2_selftest_'))
    fz, t1, src = tmp / 'final3', tmp / 'tables1', tmp / 'R2final'
    for d in (fz, src, tmp / 'data' / 'regen'):
        d.mkdir(parents=True)
    late = 'T16'  # тикер открылся 28.03.2022: строк за 24–25.03 нет (как LSRG, MTLR, OGKB)
    gg = {}
    for h in FTST.HS:
        g = FS.make(rng, start='2017-01-03', end='2026-02-27')
        g = g[~g.date.between(*HALT)]  # как в данных: дней остановки торгов нет
        g = g[~((g.ticker == late) & g.date.between('2022-03-24', '2022-03-25'))].reset_index(drop=True)
        g['step'] = g.year - 2016
        g.drop(columns='rv_m').assign(design='r1').to_parquet(fz / f'final_h{h}_test.parquet', index=False)
        gg[h] = g
    # синтетический календарь «матрицы признаков» (date, ticker) и первые торговые дни — для test_dop2.masks
    g.loc[:, ['date', 'ticker']].to_parquet(tmp / 'data' / 'regen' / 'features_pit_v3.parquet', index=False)
    ft = {tk: (x[x < HALT[0]].max(), x[x > HALT[1]].min()) for tk, x in g.groupby('ticker').date}
    assert ft[late][1] == pd.Timestamp('2022-03-28') and ft['T0'][1] == pd.Timestamp('2022-03-24')
    data0, TD2.DATA = TD2.DATA, tmp / 'data'
    try:
        # 1. правило: код первого теста (test_dop2.masks) = текст дирижёра — ровно h точек на тикер, последние h торговых дней до 25.02.2022
        D = {h: d[d.rv_actual > 1e-12].sort_values(['date', 'ticker']).reset_index(drop=True) for h, d in gg.items()}
        ex, X = masks_first(D, ft)
        for h, d in D.items():
            assert ex[h].sum() == h * d.ticker.nunique() and (ex[h] == zakrytie(d, h)).all(), h
            want = sorted(d[d.date <= '2022-02-25'].date.unique())[-h:]
            assert all(sorted(s) == want for _, s in d[ex[h]].groupby('ticker').date), h
        try:  # строки внутри остановки — не тихо: assert
            zakrytie(FS.make(rng, start='2022-02-01', end='2022-03-31', nt=2), 5)
            raise AssertionError('строки внутри остановки не пойманы')
        except AssertionError as e:
            assert 'внутри остановки' in str(e), e
        # 2. «первый тест» на синтетике: final_test.run и таблицы периодов test_dop2 в обеих версиях
        feat = pd.concat([FS.make(rng, start='2014-01-02', end='2016-12-30'), g])[['date', 'ticker', 'rv_m']]
        FTST.run(fz, t1, tag='test_raw', feat=feat, B=300)  # как table_3_*_test_raw.csv первого теста
        FT.set_etalon(FTST.ETALON)
        period_tables(D, 'bez_zakrytiya', t1, 'test_bez_zakrytiya', 300, r2=False)
        period_tables({h: d[~ex[h]].reset_index(drop=True) for h, d in D.items()}, 's_zakrytiem', t1, 'test_s_zakrytiem', 300, r2=False)
        for h in (5, 22):  # победитель R2 только на h=5 и 22; h=1 — остаётся F_h
            x = pd.read_parquet(fz / f'final_h{h}_test.parquet')
            x['rv_pred'] = x.rv_pred * np.exp(rng.normal(0, 0.1, len(x)))
            x[R2_COLS].assign(design='r2').to_parquet(src / f'final_r2_h{h}_test.parquet', index=False)

        def copy_src(name, base=src):
            d = tmp / name
            d.mkdir()
            for f in base.glob('final_r2_h*_test.parquet'):
                (d / f.name).write_bytes(f.read_bytes())
            return d
        srcb, srcp, src2, src3 = copy_src('R2_bez'), copy_src('R2_pravilo'), copy_src('R2_m1'), copy_src('R2_m2')
        kw = dict(need_fd=False, ft=ft, check_sha=False)
        # 3. обе версии, сверка эталонов каждой с «первым тестом» (основная — 3.x и периоды bez_zakrytiya; с правилом — s_zakrytiem)
        hs, res = run_r2(src, fz, t1, tmp / 'tables2', feat, B=300, **kw)
        assert hs == (5, 22) and res['bez'] > 0 and res['pravilo'] > 0, res
        t2 = tmp / 'tables2'
        for tag in (TAG, TAG_Z):
            for n in ('3_2', '3_7', '3_9', '3_mcs', '3_mz', '3_periody', '3_dm_vs_F'):
                assert (t2 / f'table_{n}_{tag}.csv').exists(), (n, tag)
        for rule in RULES.values():
            for per in TD2.PER:
                for part in ('svodka', 'dm', 'mcs', 'tickers', 'subperiods', 'znak'):
                    assert (t2 / f'table_3_{per}_{part}_test_r2_{rule}.csv').exists(), (per, part, rule)
        assert (t2 / 'table_3_zakrytie_isklyucheno_test_r2.csv').exists()
        p = pd.read_csv(t2 / f'table_3_periody_{TAG}.csv')
        assert set(p.Период) == set(PERIODY) and set(p.h) == {5, 22}
        mc = pd.read_csv(t2 / f'table_3_mcs_{TAG}.csv', sep=';', decimal=',')
        assert {'Финальная', 'Гибридная раунда 1', 'HAR-J (б)'} <= set(mc['Модель'])
        sv = pd.read_csv(t2 / 'table_3_kontrol_svodka_test_r2_bez_zakrytiya.csv', sep=';', decimal=',', encoding='utf-8-sig')
        assert sv[sv.h == 22]['DM t vs HAR-J (б)'].notna().all(), 'DM h=22 у R2 должен считаться'
        fpg = pd.read_csv(t2 / f'table_3_final_po_godam_{TAG}.csv')
        f1 = pd.read_csv(t1 / 'table_3_final_po_godam_test_raw.csv')
        a = fpg[(fpg['Модель'] == 'Гибридная') & (fpg.h == 5)].QLIKE.values
        b = f1[(f1['Модель'] == 'Гибридная') & (f1.h == 5)].QLIKE.values
        assert not np.allclose(a, b), 'строки гибрида должны быть R2, а не раунда 1'
        pa, pz = (pd.read_csv(t2 / f'table_3_final_po_godam_{t}.csv') for t in (TAG, TAG_Z))
        mm = pa.merge(pz, on=['h', 'Модель', 'Год'], suffixes=('_a', '_z'))
        y22, oth = mm[mm['Год'] == '2022'], mm[~mm['Год'].isin(['2022', 'все годы (пул)'])]
        assert (y22.QLIKE_a != y22.QLIKE_z).any() and (oth.QLIKE_a == oth.QLIKE_z).all(), 'правило меняет только 2022 (и пул)'
        # 4. флаг: только одна версия
        _, rb = run_r2(srcb, fz, t1, tmp / 'tables_b', feat, B=100, close='bez', **kw)
        assert set(rb) == {'bez'} and not list((tmp / 'tables_b').glob(f'*_{TAG_Z}.csv')) and not list((tmp / 'tables_b').glob('*s_zakrytiem*'))
        _, rp = run_r2(srcp, fz, t1, tmp / 'tables_p', feat, B=100, close='pravilo', **kw)
        assert set(rp) == {'pravilo'} and rp['pravilo'] > 0 and not (tmp / 'tables_p' / f'table_3_2_{TAG}.csv').exists()
        # 5. одно открытие: замок
        try:
            run_r2(src, fz, t1, tmp / 'tables3', feat, B=100, **kw)
            raise AssertionError('замок не сработал')
        except Stop as e:
            assert 'уже есть' in str(e)
        # 6. rv_actual в файле R2 — отказ; неполный файл — СТОП
        bad = tmp / 'bad'; bad.mkdir()
        x = pd.read_parquet(fz / 'final_h5_test.parquet')
        x[R2_COLS + ['rv_actual']].to_parquet(bad / 'final_r2_h5_test.parquet', index=False)
        try:
            run_r2(bad, fz, t1, tmp / 'tables4', feat, B=100, **kw)
            raise AssertionError('rv_actual в R2 не пойман')
        except AssertionError as e:
            assert 'rv_actual' in str(e), e
        short = tmp / 'short'; short.mkdir()
        x[R2_COLS].iloc[:-5].to_parquet(short / 'final_r2_h5_test.parquet', index=False)
        try:
            run_r2(short, fz, t1, tmp / 'tables5', feat, B=100, **kw)
            raise AssertionError('неполный файл R2 не пойман')
        except Stop as e:
            assert 'строк' in str(e)
        # 7. эталон первого теста не совпал — СТОП: в 3.x (основная) и в таблице периодов с правилом
        for nm, (fname, colname, srcx) in {'3_2': ('table_3_2_test_raw.csv', 'QLIKE H = 5', src2),
                                           'kontrol_svodka': ('table_3_kontrol_svodka_test_s_zakrytiem.csv', 'QLIKE HAR-J (б)', src3)}.items():
            t1b = tmp / f'tables1_{nm}'
            t1b.mkdir()
            for f in t1.glob('*.csv'):
                (t1b / f.name).write_bytes(f.read_bytes())
            sep = ',' if nm == '3_2' else ';'
            d = pd.read_csv(t1b / fname, dtype=str, sep=sep, encoding='utf-8-sig')
            d.loc[d.index[-1], colname] = '9,9999'
            d.to_csv(t1b / fname, index=False, sep=sep)
            try:
                run_r2(srcx, fz, t1b, tmp / f'tables6_{nm}', feat, B=100, **kw)
                raise AssertionError(f'расхождение эталона {nm} не поймано')
            except Stop as e:
                assert nm in str(e) and (tmp / f'tables6_{nm}' / 'STOP_R2.txt').exists(), e
        # 8. без FINAL_DESIGN_round2.md — отказ (в репозитории его пока нет); SHA замороженного файла не совпал — СТОП
        if not (REPO / 'docs/otchety/FINAL_DESIGN_round2.md').exists():
            try:
                run_r2(src2, fz, t1, tmp / 'tables7', feat, B=100, lock=False, ft=ft, check_sha=False)
                raise AssertionError('открытие без FINAL_DESIGN_round2 не остановлено')
            except Stop as e:
                assert 'FINAL_DESIGN_round2' in str(e)
        try:
            run_r2(src2, fz, t1, tmp / 'tables8', feat, B=100, lock=False, need_fd=False, ft=ft)
            raise AssertionError('SHA замороженного файла не проверен')
        except Stop as e:
            assert 'SHA-256' in str(e)
    finally:
        TD2.DATA = data0
    print('selftest final_test_r2: OK (правило закрытия биржи: test_dop2.masks = текст дирижёра, h точек на тикер, тикер с 28.03; обе версии '
          'и флаг bez/pravilo/obe; сверка эталонов 3.x и таблиц периодов bez_zakrytiya / s_zakrytiem; СТОП при расхождении; замок, rv_actual, '
          'неполный файл, SHA):', tmp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scope', choices=['test'])
    ap.add_argument('--selftest', action='store_true')
    ap.add_argument('--src', default=str(MAIN / 'data/regen/test_locked/R2/final'))
    ap.add_argument('--features', default=str(MAIN / 'data/regen/features_pit_v2.parquet'), help='как final_test.py: RV_m для терцилей')
    ap.add_argument('--mcs-b', type=int, default=10000)
    ap.add_argument('--close-rule', choices=['obe', 'bez', 'pravilo'], default='obe',
                    help='правило закрытия биржи: obe — основная (без правила) и с правилом рядом (по умолчанию); bez / pravilo — одна версия')
    ap.add_argument('--first-close-tag', default='s_zakrytiem',
                    help='тег таблиц первого теста с правилом: table_3_{vse,kontrol,neotkr}_*_test_<тег>.csv (дирижёр 26.10: s_zakrytiem)')
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if a.scope != 'test':
        sys.exit('нужен --scope test (или --selftest)')
    feat = pd.read_parquet(a.features, columns=['date', 'ticker', 'rv_m'])
    feat['date'] = pd.to_datetime(feat['date'])
    src = Path(a.src if Path(a.src).is_absolute() else MAIN / a.src)
    try:
        run_r2(src, FROZEN, TABLES, TABLES, feat, a.mcs_b, close=a.close_rule, first_close_tag=a.first_close_tag)
    except Stop as e:
        print('СТОП:', e, flush=True)
        sys.exit(3)


if __name__ == '__main__':
    main()
