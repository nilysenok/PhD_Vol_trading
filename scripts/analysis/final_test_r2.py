#!/usr/bin/env python3
"""R2: второе открытие теста — только строки гибрида второго раунда (протокол docs/zadaniya/2026-10-26_vtoroy_raund_gibrida.md,
«Итог раунда и второе открытие», п. 3). Автор — агент О. До открытия проверяется только --selftest (синтетика, тест не читается).

python3 scripts/analysis/final_test_r2.py --selftest
python3 scripts/analysis/final_test_r2.py --scope test --src data/regen/test_locked/R2/final     # ОДИН раз, дирижёр, после «да» владельца

Вход:
  <src>/final_r2_h{h}_test.parquet — date, ticker, year, step, rv_pred, design (собирает r2_sborka_test.py; колонки rv_actual НЕТ — assert);
  h без файла — на h остаётся F_h, строк R2 нет;
  замороженный первый тест: test_locked/final3/final_h{h}_test.parquet (rv_actual, эталоны, компоненты, гибрид раунда 1) — не пересчитывается.
Счёт: строки (date, ticker, step) R2 и первого теста должны совпасть один в один. Кадр = первый тест, rv_pred := R2, hyb_r1 := гибрид раунда 1.
Таблицы — тем же кодом, что первый тест (final_test.run и ftp_tables импортируются, final_test.py как скрипт не запускается), суффикс _test_r2:
QLIKE / RMSE / MAE по h и годам; DM тремя способами против HAR-J как в диссертации и против HAR-J с поправкой; MCS на наборе
{замороженные + R2 + гибрид раунда 1}; MZ, доля тикеров, подпериоды, терцили. Дополнительно: table_3_periody_test_r2.csv
(2022–2026 основной, 2025 – 01/02.2026 отдельно, 2017–2026 справочно) и table_3_dm_vs_F_test_r2.csv (справочно — против F_h).
Сверка: ячейки эталонов (строки без гибрида) в таблицах R2 должны побайтно совпасть с таблицами первого теста; расхождение — СТОП
(файл STOP_R2.txt, код выхода 3). Замок: <src>/OPENED_R2.json пишется ДО чтения теста; если он есть — отказ (одно открытие).
Правило закрытия биржи (текст дирижёра 26.10; раздел 13 FD агент О не читал), «обе версии»: --close-rule obe (по умолчанию) пишет
основную версию без правила (_test_r2) и рядом версию с правилом (_test_r2_zakrytie) — исключены точки отсчёта, окно цели которых
задевает остановку торгов 28.02–23.03.2022 (ровно h точек на тикер). Сверка эталонов версии с правилом — с таблицами первого теста
под тегом --first-close-tag."""
import argparse, hashlib, json, subprocess, sys, tempfile
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import final_test as FTST  # noqa: E402  код таблиц первого теста (импорт; скрипт не запускается)
import ftp_tables as FT  # noqa: E402

REPO = HERE.parent.parent
MAIN = Path('/Users/nilysenok/Desktop/Старое/PhD_Vol_trading')
FROZEN = MAIN / 'data/regen/test_locked/final3'
TABLES = REPO / 'docs/tables'
KEY = ['date', 'ticker', 'step']
R2_COLS = ['date', 'ticker', 'year', 'step', 'rv_pred', 'design']
PERIODY = {'2022–2026 (основной)': (2022, 2026), '2025 – 01/02.2026': (2025, 2026), '2017–2026 (справочно)': (2017, 2026)}
# Правило закрытия биржи (текст дирижёра 26.10 для R2; раздел 13 FD агенту О не читать): из оценки исключаются точки отсчёта t, окно цели
# [t+1, t+h] которых содержит хотя бы один день остановки торгов MOEX 28.02–23.03.2022. В данных дней остановки нет (LSRG, MTLR, OGKB
# открылись 28.03, строк за 24–25.03 у них нет), поэтому исключаются ровно h точек на тикер: последние h торговых дней до 25.02.2022
# включительно. Основная версия — без правила; с правилом — рядом, проверка устойчивости (решение владельца 10:20).
HALT = (pd.Timestamp('2022-02-28'), pd.Timestamp('2022-03-23'))
TAG, TAG_Z = 'test_r2', 'test_r2_zakrytie'


def zakrytie(df, h):
    """Маска исключаемых строк (True — исключить): на тикер последние h торговых дат < HALT[0] (т. е. <= 25.02.2022) и строки внутри
    остановки (в данных их быть не должно — тогда assert). Дата строки — точка отсчёта t."""
    d = pd.to_datetime(df.date)
    inside = d.between(*HALT)
    assert not inside.any(), f'h={h}: {int(inside.sum())} строк внутри остановки торгов 28.02–23.03.2022'
    pre = d < HALT[0]
    last = df[pre].assign(_d=d[pre]).groupby('ticker')._d.transform(lambda s: s.rank(method='first', ascending=False) <= h)
    m = pd.Series(False, index=df.index)
    m.loc[last.index] = last.astype(bool)
    return m.values


class Stop(Exception):
    pass


def now():
    return subprocess.run(['date', '+%Y-%m-%d %H:%M:%S'], capture_output=True, text=True).stdout.strip()


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def merge_h(src, frozen, h):
    """Кадр h: первый тест + R2. Проверки: в R2 нет rv_actual; строки совпадают один в один; нет пропусков."""
    f = src / f'final_r2_h{h}_test.parquet'
    r2 = pd.read_parquet(f)
    assert 'rv_actual' not in r2.columns, f'{f.name}: в файле R2 не должно быть rv_actual'
    assert set(R2_COLS) <= set(r2.columns), f'{f.name}: нет колонок {set(R2_COLS) - set(r2.columns)}'
    fz = pd.read_parquet(frozen / f'final_h{h}_test.parquet')
    if len(r2) != len(fz) or r2.duplicated(KEY).any():
        raise Stop(f'h={h}: строк R2 {len(r2)}, первого теста {len(fz)} (или дубли ключа)')
    m = fz.rename(columns={'rv_pred': 'hyb_r1', 'design': 'design_r1'}).merge(r2[KEY + ['rv_pred', 'design']], on=KEY, how='left', validate='1:1')
    if m.rv_pred.isna().any():
        raise Stop(f'h={h}: {int(m.rv_pred.isna().sum())} строк первого теста без прогноза R2')
    return m


def periody(frames):
    rows = []
    for h, df in frames.items():
        df = df[df.rv_actual > 1e-12]
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
        df = df[df.rv_actual > 1e-12].sort_values(['date', 'ticker']).reset_index(drop=True)
        out += [dict(H=h, эталон='гибрид раунда 1 (F_h), справочно', **r) for r in FTST.dm(df, 'hyb_r1', h)]
    return pd.DataFrame(out)


# ---------------------------------------------------------------- сверка эталонов с первым тестом
# (имя таблицы, разделитель, ключ, колонка-признак гибрида, значения-гибрида, исключаемые колонки)
SVERKA = [
    ('3_2', ',', ['Год'], None, (), ()),
    ('3_3', ',', ['Горизонт'], None, (), ()),
    ('3_4', ',', ['Модель'], None, (), ()), ('3_4_rmse', ',', ['Модель'], None, (), ()), ('3_4_mae', ',', ['Модель'], None, (), ()),
    ('3_6', ',', ['Модель'], 'Модель', ('Гибридная',), ()), ('3_6_rmse', ',', ['Модель'], 'Модель', ('Гибридная',), ()),
    ('3_6_mae', ',', ['Модель'], 'Модель', ('Гибридная',), ()),
    ('3_7', ',', ['Модель'], 'Модель', ('Гибридная',), ()), ('3_7_rmse', ',', ['Модель'], 'Модель', ('Гибридная',), ()),
    ('3_7_mae', ',', ['Модель'], 'Модель', ('Гибридная',), ()),
    ('3_final_po_godam', ',', ['h', 'Модель', 'Год'], 'Модель', ('Гибридная',), ()),
    ('3_raskladka', ',', ['h', 'Год', 'Строка'], 'Строка', ('Гибридная (финал)',), ()),
    ('3_mz', ';', ['h', 'Шкала', 'Модель'], 'Модель', ('Финальная', 'Гибридная раунда 1'), ()),
    ('3_mcs', ';', ['h', 'Модель'], 'Модель', ('Финальная', 'Гибридная раунда 1'), ('p_MCS', 'в MCS')),
]


def sverka(first_dir, r2_dir, hs, tag_first='test', tag_r2='test_r2'):
    """Ячейки эталонов: строковое (побайтное) совпадение значений; колонки и строки — те, что есть в обеих таблицах для этих h."""
    bad, n = [], 0
    for name, sep, key, col, hyb, excl in SVERKA:
        a = pd.read_csv(first_dir / f'table_{name}_{tag_first}.csv', sep=sep, dtype=str, encoding='utf-8-sig')
        b = pd.read_csv(r2_dir / f'table_{name}_{tag_r2}.csv', sep=sep, dtype=str, encoding='utf-8-sig')
        if col:
            a, b = a[~a[col].isin(hyb)], b[~b[col].isin(hyb)]
        if 'h' in key:
            a = a[a.h.isin([str(h) for h in hs])]
        cols = [c for c in b.columns if c in a.columns and c not in key and not any(c.startswith(e) for e in excl)]
        if name in ('3_3',):
            a = a[a['Горизонт'].isin([f'H = {h}' for h in hs])]
        m = a.merge(b, on=key, how='inner', suffixes=('_a', '_b'))
        if len(m) != len(b) or len(m) == 0:
            bad.append(f'{name}: строк эталонов в R2 {len(b)}, совпало ключей {len(m)}')
            continue
        for c in cols:
            diff = m[m[c + '_a'].fillna('') != m[c + '_b'].fillna('')]
            n += len(m)
            if len(diff):
                bad.append(f'{name}/{c}: {len(diff)} ячеек, напр. {diff.iloc[0][key].tolist()} {diff.iloc[0][c + "_a"]} != {diff.iloc[0][c + "_b"]}')
    return bad, n


def run_r2(src, frozen, first_tables, out_tables, feat, B, lock=True, need_fd=True, close='obe', first_tag='test', first_tag_close=None):
    """close: 'bez' — только основная версия; 'pravilo' — только с правилом закрытия биржи; 'obe' — обе (по умолчанию).
    Таблицы: _test_r2 (основная) и _test_r2_zakrytie (с правилом). first_tag_close — тег таблиц первого теста с правилом для сверки
    эталонов (None — сверка этой версии не делается, сообщение в выводе)."""
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
    frames = {h: merge_h(src, frozen, h) for h in hs}
    versions = {'bez': (TAG, frames, first_tag)}
    if close in ('obe', 'pravilo'):
        fz = {}
        for h, m in frames.items():
            ex = zakrytie(m, h)
            fz[h] = m[~ex].reset_index(drop=True)
            print(f'правило закрытия биржи, h={h}: исключено {int(ex.sum())} строк ({m[ex].ticker.nunique()} тикеров × {h})', flush=True)
        versions['pravilo'] = (TAG_Z, fz, first_tag_close)
        if close == 'pravilo':
            versions.pop('bez')
    res = {}
    for ver, (tag, fr, ftag) in versions.items():
        mdir = Path(tempfile.mkdtemp(prefix=f'r2_merged_{ver}_'))
        for h, m in fr.items():
            m.to_parquet(mdir / f'final_h{h}_test.parquet', index=False)
        hs0, mods0 = FTST.HS, dict(FT.MODS)
        try:
            FTST.HS = hs  # только h с победителем R2
            FT.MODS['Гибридная раунда 1'] = 'hyb_r1'  # MCS (и MZ) — на наборе {замороженные + R2 + гибрид раунда 1}
            FTST.run(mdir, out_tables, tag=tag, feat=feat, B=B)
        finally:
            FTST.HS = hs0
            FT.MODS.clear(); FT.MODS.update(mods0)
        periody(fr).to_csv(out_tables / f'table_3_periody_{tag}.csv', index=False)
        dm_vs_F(fr).to_csv(out_tables / f'table_3_dm_vs_F_{tag}.csv', index=False)
        if ftag is None:
            print(f'{ver}: сверка эталонов не делалась — не задан тег таблиц первого теста для этой версии (--first-close-tag)', flush=True)
            res[ver] = None
            continue
        bad, n = sverka(first_tables, out_tables, hs, tag_first=ftag, tag_r2=tag)
        if bad:
            (out_tables / 'STOP_R2.txt').write_text(f'версия {ver}\n' + '\n'.join(bad) + '\n')
            raise Stop(f'версия {ver}: эталоны не совпали с первым тестом:\n' + '\n'.join(bad[:10]))
        print(f'{ver}: сверка эталонов с первым тестом ({ftag}): {n} ячеек, расхождений нет; h = {hs}', flush=True)
        res[ver] = n
    return hs, res


# ---------------------------------------------------------------- самопроверка (синтетика; тест не читается)
def selftest():
    import ftp_selftest as FS
    rng = np.random.default_rng(26)
    tmp = Path(tempfile.mkdtemp(prefix='r2_selftest_'))
    fz, t1, src = tmp / 'final3', tmp / 'tables1', tmp / 'R2final'
    for d in (fz, src):
        d.mkdir()
    fzz = tmp / 'final3_zakrytie'
    fzz.mkdir()
    for h in FTST.HS:
        g = FS.make(rng, start='2017-01-03', end='2026-02-27')
        g = g[~g.date.between(*HALT)].reset_index(drop=True)  # как в данных: дней остановки торгов нет
        g['step'] = g.year - 2016
        g.drop(columns='rv_m').assign(design='r1').to_parquet(fz / f'final_h{h}_test.parquet', index=False)
        # правило закрытия биржи: ровно h точек на тикер — последние h торговых дней до 25.02.2022 включительно
        ex = zakrytie(g, h)
        exd = g[ex].groupby('ticker').date.apply(lambda s: sorted(s))
        want = sorted(g[g.date <= '2022-02-25'].date.unique())[-h:]
        assert ex.sum() == h * g.ticker.nunique() and all(list(v) == want for v in exd), h
        g[~ex].drop(columns='rv_m').assign(design='r1').to_parquet(fzz / f'final_h{h}_test.parquet', index=False)
    try:  # строки внутри остановки — не тихо: assert
        zakrytie(FS.make(rng, start='2022-02-01', end='2022-03-31', nt=2), 5)
        raise AssertionError('строки внутри остановки не пойманы')
    except AssertionError as e:
        assert 'внутри остановки' in str(e), e
    feat = pd.concat([FS.make(rng, start='2014-01-02', end='2016-12-30'), g])[['date', 'ticker', 'rv_m']]
    FTST.run(fz, t1, tag='test', feat=feat, B=300)  # «первый тест» на синтетике: основная версия
    FTST.run(fzz, t1, tag='test_zakrytie', feat=feat, B=300)  # и версия с правилом закрытия биржи
    for h in (5, 22):  # победитель R2 только на h=5 и 22; h=1 — остаётся F_h
        x = pd.read_parquet(fz / f'final_h{h}_test.parquet')
        x['rv_pred'] = x.rv_pred * np.exp(rng.normal(0, 0.1, len(x)))
        x[R2_COLS].assign(design='r2').to_parquet(src / f'final_r2_h{h}_test.parquet', index=False)

    def copy_src(name):
        d = tmp / name
        d.mkdir()
        for f in src.glob('final_r2_h*_test.parquet'):
            (d / f.name).write_bytes(f.read_bytes())
        return d
    srcb, srcp = copy_src('R2_bez'), copy_src('R2_pravilo')
    hs, res = run_r2(src, fz, t1, tmp / 'tables2', feat, B=300, need_fd=False, first_tag_close='test_zakrytie')
    assert hs == (5, 22) and res['bez'] > 0 and res['pravilo'] > 0, res  # обе версии сверены с «первым тестом»
    t2 = tmp / 'tables2'
    for tag in (TAG, TAG_Z):
        for n in ('3_2', '3_7', '3_9', '3_mcs', '3_mz', '3_periody', '3_dm_vs_F'):
            assert (t2 / f'table_{n}_{tag}.csv').exists(), (n, tag)
    pa = pd.read_csv(t2 / f'table_3_final_po_godam_{TAG}.csv')
    pz = pd.read_csv(t2 / f'table_3_final_po_godam_{TAG_Z}.csv')
    k = ['h', 'Модель', 'Год']
    mm = pa.merge(pz, on=k, suffixes=('_a', '_z'))
    y22 = mm[mm['Год'] == '2022']
    oth = mm[~mm['Год'].isin(['2022', 'все годы (пул)'])]
    assert (y22.QLIKE_a != y22.QLIKE_z).any() and (oth.QLIKE_a == oth.QLIKE_z).all(), 'правило меняет только 2022 (и пул)'
    _, rb = run_r2(srcb, fz, t1, tmp / 'tables_b', feat, B=100, need_fd=False, close='bez')
    assert set(rb) == {'bez'} and not list((tmp / 'tables_b').glob(f'*_{TAG_Z}.csv'))
    _, rp = run_r2(srcp, fz, t1, tmp / 'tables_p', feat, B=100, need_fd=False, close='pravilo', first_tag_close='test_zakrytie')
    assert set(rp) == {'pravilo'} and rp['pravilo'] > 0 and not (tmp / 'tables_p' / f'table_3_2_{TAG}.csv').exists()
    p = pd.read_csv(t2 / 'table_3_periody_test_r2.csv')
    assert set(p.Период) == set(PERIODY) and set(p.h) == {5, 22}
    mc = pd.read_csv(t2 / 'table_3_mcs_test_r2.csv', sep=';', decimal=',')
    assert {'Финальная', 'Гибридная раунда 1', 'HAR-J (б)'} <= set(mc['Модель'])
    fpg = pd.read_csv(t2 / 'table_3_final_po_godam_test_r2.csv')
    f1 = pd.read_csv(t1 / 'table_3_final_po_godam_test.csv')
    a = fpg[(fpg['Модель'] == 'Гибридная') & (fpg.h == 5)].QLIKE.values
    b = f1[(f1['Модель'] == 'Гибридная') & (f1.h == 5)].QLIKE.values
    assert not np.allclose(a, b), 'строки гибрида должны быть R2, а не раунда 1'
    # одно открытие: замок
    try:
        run_r2(src, fz, t1, tmp / 'tables3', feat, B=100, need_fd=False)
        raise AssertionError('замок не сработал')
    except Stop as e:
        assert 'уже есть' in str(e)
    # rv_actual в файле R2 — отказ
    bad = tmp / 'bad'; bad.mkdir()
    x = pd.read_parquet(fz / 'final_h5_test.parquet')
    x[R2_COLS + ['rv_actual']].to_parquet(bad / 'final_r2_h5_test.parquet', index=False)
    try:
        run_r2(bad, fz, t1, tmp / 'tables4', feat, B=100, need_fd=False)
        raise AssertionError('rv_actual в R2 не пойман')
    except AssertionError as e:
        assert 'rv_actual' in str(e), e
    # не все строки — СТОП
    short = tmp / 'short'; short.mkdir()
    x[R2_COLS].iloc[:-5].to_parquet(short / 'final_r2_h5_test.parquet', index=False)
    try:
        run_r2(short, fz, t1, tmp / 'tables5', feat, B=100, need_fd=False)
        raise AssertionError('неполный файл R2 не пойман')
    except Stop as e:
        assert 'строк' in str(e)
    # эталон первого теста не совпал — СТОП
    t1b = tmp / 'tables1b'
    t1b.mkdir()
    for f in t1.glob('*.csv'):
        (t1b / f.name).write_bytes(f.read_bytes())
    d = pd.read_csv(t1b / 'table_3_2_test.csv', dtype=str)
    d.loc[0, 'QLIKE H = 5'] = '9.9999'
    d.to_csv(t1b / 'table_3_2_test.csv', index=False)
    src2 = tmp / 'R2final2'; src2.mkdir()
    for f in src.glob('final_r2_h*_test.parquet'):
        (src2 / f.name).write_bytes(f.read_bytes())
    try:
        run_r2(src2, fz, t1b, tmp / 'tables6', feat, B=100, need_fd=False)
        raise AssertionError('расхождение эталона не поймано')
    except Stop as e:
        assert '3_2' in str(e) and (tmp / 'tables6' / 'STOP_R2.txt').exists()
    # без FINAL_DESIGN_round2.md — отказ (в репозитории его пока нет)
    if not (REPO / 'docs/otchety/FINAL_DESIGN_round2.md').exists():
        try:
            run_r2(src2, fz, t1, tmp / 'tables7', feat, B=100, lock=False)
            raise AssertionError('открытие без FINAL_DESIGN_round2 не остановлено')
        except Stop as e:
            assert 'FINAL_DESIGN_round2' in str(e)
    print('selftest final_test_r2: OK (обе версии правила закрытия биржи: h точек на тикер, сверка эталонов каждой версии, флаг bez/pravilo/obe; '
          'замок, rv_actual, неполный файл, СТОП при расхождении):', tmp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scope', choices=['test'])
    ap.add_argument('--selftest', action='store_true')
    ap.add_argument('--src', default=str(MAIN / 'data/regen/test_locked/R2/final'))
    ap.add_argument('--features', default=str(MAIN / 'data/regen/features_pit_v2.parquet'), help='как final_test.py: RV_m для терцилей')
    ap.add_argument('--mcs-b', type=int, default=10000)
    ap.add_argument('--close-rule', choices=['obe', 'bez', 'pravilo'], default='obe',
                    help='правило закрытия биржи: obe — основная (без правила) и с правилом рядом (по умолчанию); bez / pravilo — одна версия')
    ap.add_argument('--first-close-tag', default=None,
                    help='тег таблиц первого теста с правилом закрытия (table_3_*_<тег>.csv) для сверки эталонов версии с правилом; '
                         'не задан — эта версия не сверяется (сообщение в выводе)')
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if a.scope != 'test':
        sys.exit('нужен --scope test (или --selftest)')
    feat = pd.read_parquet(a.features, columns=['date', 'ticker', 'rv_m'])
    feat['date'] = pd.to_datetime(feat['date'])
    src = Path(a.src if Path(a.src).is_absolute() else MAIN / a.src)
    try:
        run_r2(src, FROZEN, TABLES, TABLES, feat, a.mcs_b, close=a.close_rule, first_tag_close=a.first_close_tag)
    except Stop as e:
        print('СТОП:', e, flush=True)
        sys.exit(3)


if __name__ == '__main__':
    main()
