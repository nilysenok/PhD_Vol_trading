#!/usr/bin/env python3
"""R2: второе открытие теста — все строки в версии d0 (протокол v3 docs/zadaniya/2026-10-26_vtoroy_raund_gibrida.md, разделы 2.8, 9, 11).
Авторы — агент О (PR #144, #145, #147), под v3 доведён агентом A. До открытия проверяется только --selftest (синтетика, тест не читается).

python3 scripts/analysis/final_test_r2.py --selftest
python3 scripts/analysis/final_test_r2.py --scope test          # ОДИН раз, дирижёр, после «да» владельца на FINAL_DESIGN_round2.md

Строки (все — версия d0, единое информационное множество «по день d включительно»):
  HAR-J (а) d0, HAR-J (б) d0 — сырой ряд ВЫР-1, s² по t−1 с эмбарго (решение владельца 17:35, PR #194/#195): собирает
    r2_sborka_test.py → test_locked/R2/final/etalon_d0_s2emb_h{h}_test.parquet (har_raw, har_corr); прежний (б) ВЫР-1
    (vyr/etalon_d0_h{h}_test.parquet, s² как R1) — только справочно в таблице периодов и как источник цели для сверки;
  гибрид — победитель R2 (test_locked/R2/final/final_r2_h{h}_test.parquet от r2_sborka_test.py; колонки rv_actual НЕТ — проверка до
    замка) или, если на h победителя нет, F_h(d0) с тем же s² (R2/final/final_d0_s2emb_h{h}_test.parquet; у h=22 это эталон (б) d0);
  F_h(d0) — отдельной строкой «Гибридная раунда 1 d0» (в MCS, периодах и DM против F_h(d0) справочно); прежний F_h(d0) ВЫР-1 — справочно;
  SHA собранных файлов — r2_A_sborka.yaml (sobrano), пишется после сборки, до «да»; без них — СТОП до замка;
  V0 XGBoost / LightGBM (raw и с поправкой) — кадр первого теста test_locked/final3/final_h{h}_test.parquet как есть: по ВЫР-2 они день d
    видят и не перестраивались; HAR-CJ d0 — ВЫР-2 (test_locked/vyr/ref_harcj_d0/), справочно, в таблице периодов;
  HAR-X с поправкой в раскладке (3_raskladka) — кадр первого теста, версия R1 (❓ A-2: d0-ряда на тесте нет).
Кадр: ключи (date, ticker, step), цель и фильтр rv_actual > 1e-12 — как у первого теста (final_test.run / test_dop.load); цель эталонов
  ВЫР-1 обязана совпасть с целью кадра (СТОП). Старая строка гибрида первого прогона (до выравнивания) в кадр не входит.
Замороженные файлы: SHA-256 из models/designs/r2/r2_A_sborka.yaml (zamorozheno) сверяются ДО замка; несовпадение — СТОП, открытие не
  тратится. Так же до замка: FINAL_DESIGN_round2.md заполнен (нет «ждёт ворот E»), файлы R2 без rv_actual и с числом строк кадра.
Таблицы — тем же кодом, что первый тест (импорт final_test.run, ftp_tables, test_dop, test_dop2; final_test.py и test_dop2.py как скрипты не
  запускаются). MCS — на наборе {гибрид, HAR-J (а), HAR-J (б), лучшая одиночная бустинга, F_h(d0)}.
  основная версия (без правила): table_3_*_test_r2.csv и table_3_{vse,kontrol,neotkr}_{svodka,dm,mcs,tickers,subperiods,znak}_test_r2_bez_zakrytiya.csv;
  с правилом «закрытие биржи» рядом: table_3_*_test_r2_zakrytie.csv и table_3_{vse,kontrol,neotkr}_*_test_r2_s_zakrytiem.csv;
    исключённые точки — table_3_zakrytie_isklyucheno_test_r2.csv; правило — test_dop2.masks + first_trade, контроль — текст дирижёра
    (на тикер последние h торговых дней до 25.02.2022);
  дополнительно: table_3_periody_* (QLIKE / RMSE / MAE: 2022–2026 основной, 2025 – 01/02.2026, 2017–2026 справочно) и
    table_3_dm_vs_F_* (против F_h(d0), только h с победителем R2);
  h=22, вторичные свидетельства (блок владельца 20:10, объявлены в FINAL_DESIGN_round2 до открытия) — table_3_h22_vtorichnye_*:
    основной — панельный DM против HAR-J (б) на 2022–2026 (тот же, что в 3.8); вторичные — тот же DM на 2017–2026 (оговорка: 2017–2021
    участвовали в отборе), MCS на обоих периодах, DM по неперекрывающимся точкам отсчёта (первая торговая дата периода и каждая 22-я
    далее, средняя по тикерам, без NW).
Сверка кода с первым тестом: строки моделей, которые в d0 не менялись (V0 XGBoost / LightGBM, лучшая одиночная бустинга), побайтно
  совпадают с table_3_*_test_raw.csv и таблицами периодов _test_bez_zakrytiya / _test_<--first-close-tag>; строки эталонов и гибрида
  в d0 другие и не сверяются. Расхождение — СТОП (STOP_R2.txt, код выхода 3).
Замок: <src>/OPENED_R2.json пишется ДО чтения теста; если он есть — отказ (одно открытие)."""
import argparse, hashlib, json, subprocess, sys, tempfile
from contextlib import contextmanager
from pathlib import Path
import numpy as np, pandas as pd
import pyarrow.parquet as pq
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import final_test as FTST  # noqa: E402  код таблиц первого теста (импорт; скрипт не запускается)
import ftp_tables as FT  # noqa: E402
import test_dop as TD  # noqa: E402  запись таблиц (write_named) и SHA кадра первого теста
import test_dop2 as TD2  # noqa: E402  правило закрытия биржи и таблицы периодов — тем же кодом

REPO = HERE.parent.parent
MAIN = Path('/Users/nilysenok/Desktop/Старое/PhD_Vol_trading')
LOCKED = MAIN / 'data/regen/test_locked'
TABLES = REPO / 'docs/tables'
FD = REPO / 'docs/otchety/FINAL_DESIGN_round2.md'
SPEC = REPO / 'models/designs/r2/r2_A_sborka.yaml'
PLACEHOLDER = 'ждёт ворот E'
KEY = ['date', 'ticker', 'step']
R2_COLS = ['date', 'ticker', 'year', 'step', 'rv_pred', 'design']
PERIODY = {'2022–2026 (основной)': (2022, 2026), '2025 – 01/02.2026': (2025, 2026), '2017–2026 (справочно)': (2017, 2026)}
HALT = (pd.Timestamp('2022-02-28'), pd.Timestamp('2022-03-23'))
TAG, TAG_Z = 'test_r2', 'test_r2_zakrytie'
RULES = {'bez': 'bez_zakrytiya', 'pravilo': 's_zakrytiem'}
F_NAME = 'Гибридная раунда 1 d0'  # F_h(d0)
REF_NG = {'GRU': 'GRU', 'LSTM': 'LSTM', 'GARCH': 'GARCH(1,1)', 'GJR': 'GJR-GARCH'}  # справочные (ref_nn_garch), не в H1 / DM / MCS
SAME = ('XGBoost', 'LightGBM', 'XGBoost с поправкой', 'LightGBM с поправкой', 'Лучшая одиночная бустинга')  # в d0 не менялись
NOTE_R2 = ('второе открытие (R2), версия d0: эталоны (а), (б) и F_h(d0) — ВЫР-1, справочные — ВЫР-2, гибрид — победитель R2 или F_h(d0); '
           'числа первого прогона (до выравнивания) — только журнал')


class Stop(Exception):
    pass


def now():
    return subprocess.run(['date', '+%Y-%m-%d %H:%M:%S'], capture_output=True, text=True).stdout.strip()


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def frozen_files(h):
    """fz — кадр первого теста; et, fd — ВЫР-1 (s² как R1, справочно и цель); et2, fd2 — эталоны и F_h(d0) с s² по t−1 и эмбарго
    (r2_sborka_test.py, PR #194/#195, решение владельца 17:35); cj — HAR-CJ d0 (ВЫР-2); gru, lstm, garch, gjr — справочные NN и GARCH
    (test_locked/ref_nn_garch, PR #97; по ВЫР-2, PR #154, день d видят — без изменений)."""
    return {'fz': f'final3/final_h{h}_test.parquet', 'et': f'vyr/etalon_d0_h{h}_test.parquet', 'fd': f'vyr/final_d0_h{h}_test.parquet',
            'cj': f'vyr/ref_harcj_d0/REF_HARCJ_d0_h{h}_test.parquet',
            'et2': f'R2/final/etalon_d0_s2emb_h{h}_test.parquet', 'fd2': f'R2/final/final_d0_s2emb_h{h}_test.parquet',
            **{m.lower(): f'ref_nn_garch/REF_{m}_h{h}_test.parquet' for m in REF_NG}}


def frozen_sha(spec=SPEC):
    """SHA замороженных до R2 (zamorozheno) и собранных r2_sborka_test.py (sobrano, пишется после сборки, до «да»)."""
    y = yaml.safe_load(Path(spec).read_text())
    z = {**(y.get('zamorozheno') or {}), **(y.get('sobrano') or {})}
    assert all(z[f'final3/final_h{h}_test.parquet'] == TD.SHA[h] for h in FTST.HS), 'r2_A_sborka.yaml: SHA кадра != test_dop.SHA'
    return z


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


# ---------------------------------------------------------------- проверки до замка (данные не разбираются)
def precheck(src, locked, shas, check_sha=True, fd=FD, need_fd=True):
    """FINAL_DESIGN_round2 заполнен; SHA замороженных; файлы R2: колонки без rv_actual, число строк = кадр первого теста. -> h с R2."""
    if need_fd:
        if not Path(fd).exists():
            raise Stop(f'нет {fd} — открытие только после него и «да» владельца')
        if PLACEHOLDER in Path(fd).read_text():
            raise Stop(f'{Path(fd).name}: есть поля «{PLACEHOLDER}» — шаблон не заполнен, открытие запрещено')
    for h in FTST.HS:
        for r in frozen_files(h).values():
            p = locked / r
            if not p.exists():
                raise Stop(f'{r}: нет замороженного файла')
            if check_sha and not shas.get(r):
                raise Stop(f'{r}: SHA-256 не записан в r2_A_sborka.yaml (zamorozheno / sobrano)')
            if check_sha and sha(p) != shas.get(r):
                raise Stop(f'{r}: SHA-256 не совпал с замороженным (r2_A_sborka.yaml)')
    r2_hs = []
    for h in FTST.HS:
        f = src / f'final_r2_h{h}_test.parquet'
        if not f.exists():
            continue
        names = pq.read_schema(f).names
        if 'rv_actual' in names:
            raise Stop(f'{f.name}: в файле R2 не должно быть rv_actual')
        if not set(R2_COLS) <= set(names):
            raise Stop(f'{f.name}: нет колонок {set(R2_COLS) - set(names)}')
        n, n0 = pq.read_metadata(f).num_rows, pq.read_metadata(locked / frozen_files(h)['fz']).num_rows
        if n != n0:
            raise Stop(f'h={h}: строк R2 {n}, кадра первого теста {n0}')
        r2_hs.append(h)
    return tuple(r2_hs)


# ---------------------------------------------------------------- кадр d0
def test_part(p, cols):
    d = pd.read_parquet(p, columns=KEY + ['part'] + cols)
    d = d[d.part == 'test'].drop(columns='part')
    if d.duplicated(KEY).any():
        raise Stop(f'{p.name}: дубли ключа в part = test')
    return d


def frame_d0(src, locked, h):
    """Кадр h в версии d0: ключи и цель — кадр первого теста; эталоны (а), (б) и F_h(d0) — ВЫР-1; HAR-CJ d0 — ВЫР-2; гибрид — R2 или F_h(d0)."""
    F = {k: locked / r for k, r in frozen_files(h).items()}
    fz = pd.read_parquet(F['fz'])
    fz = fz.drop(columns=[c for c in ('rv_pred', 'har_raw', 'har_corr') if c in fz]).rename(columns={'design': 'design_r1'})
    if fz.duplicated(KEY).any():
        raise Stop(f'h={h}: дубли ключа в кадре первого теста')
    et = test_part(F['et'], ['rv_actual', 'har_raw', 'har_corr']).rename(columns={'rv_actual': 'y_et', 'har_raw': 'har_raw_r1',
                                                                                  'har_corr': 'har_corr_bez_emb'})
    fd = test_part(F['fd'], ['rv_pred']).rename(columns={'rv_pred': 'F_d0_bez_emb'})
    et2 = pd.read_parquet(F['et2'], columns=KEY + ['har_raw', 'har_corr'])  # эталоны с s² по t−1 и эмбарго — основные строки
    fd2 = pd.read_parquet(F['fd2'], columns=KEY + ['rv_pred']).rename(columns={'rv_pred': 'F_d0'})
    cj = pd.read_parquet(F['cj'], columns=KEY + ['rv_pred']).rename(columns={'rv_pred': 'harcj_d0'})
    m = fz.merge(et, on=KEY, how='left', validate='1:1').merge(fd, on=KEY, how='left', validate='1:1')
    m = m.merge(et2, on=KEY, how='left', validate='1:1').merge(fd2, on=KEY, how='left', validate='1:1')
    m = m.merge(cj.drop_duplicates(KEY), on=KEY, how='left', validate='1:1')
    for mod in REF_NG:  # справочные NN / GARCH: как есть (rv_pred файла), покрытие может быть неполным — считаются по своим строкам
        r = pd.read_parquet(F[mod.lower()], columns=KEY + ['rv_pred']).rename(columns={'rv_pred': f'ref_{mod.lower()}'})
        m = m.merge(r.drop_duplicates(KEY), on=KEY, how='left', validate='1:1')
    for c in ('har_raw', 'har_corr', 'F_d0', 'har_corr_bez_emb', 'F_d0_bez_emb'):
        if m[c].isna().any():
            raise Stop(f'h={h}: {int(m[c].isna().sum())} строк кадра без {c} (d0)')
    if not np.allclose(m.har_raw, m.har_raw_r1, rtol=1e-12):
        raise Stop(f'h={h}: сырой эталон (а) сборки != ВЫР-1')
    m = m.drop(columns='har_raw_r1')
    ok = m.rv_actual > 1e-12
    dy = np.abs(m.y_et[ok] / m.rv_actual[ok] - 1)
    if not (dy < 1e-9).all():
        raise Stop(f'h={h}: цель эталонов ВЫР-1 не совпала с целью кадра первого теста ({int((~(dy < 1e-9)).sum())} строк)')
    f = src / f'final_r2_h{h}_test.parquet'
    if f.exists():
        r2 = pd.read_parquet(f, columns=R2_COLS)
        if r2.duplicated(KEY).any():
            raise Stop(f'h={h}: дубли ключа в R2')
        m = m.merge(r2[KEY + ['rv_pred', 'design']], on=KEY, how='left', validate='1:1')
        if m.rv_pred.isna().any():
            raise Stop(f'h={h}: {int(m.rv_pred.isna().sum())} строк кадра без прогноза R2')
    else:
        m = m.assign(rv_pred=m.F_d0, design='F_h(d0)')
    return m.drop(columns='y_et')[ok.values].sort_values(['date', 'ticker']).reset_index(drop=True)  # как test_dop.load / final_test.run


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
    """MCS и периоды на наборе {замороженные d0 + гибрид + F_h(d0)}; запись test_dop.write_named — в out_tables."""
    hs0, mods0, tab0 = FTST.HS, dict(FT.MODS), TD.TABLES
    try:
        FTST.HS = hs
        FT.set_etalon(FTST.ETALON)
        FT.MODS[F_NAME] = 'F_d0'
        TD.TABLES = Path(out_tables)
        yield
    finally:
        FTST.HS, TD.TABLES = hs0, tab0
        FT.MODS.clear(); FT.MODS.update(mods0)


def period_tables(fr, rule, out_tables, suffix, B, r2_hs=(), note=NOTE_R2):
    """Таблицы периодов vse / kontrol / neotkr — как test_dop2.main (те же функции). suffix: test_r2_<правило> (у первого теста — test_<правило>).
    svodka test_dop2 считает финал h=22 эталоном; это верно, пока на h=22 гибрид = F_22(d0) = эталон (б) d0. Если на h=22 есть
    победитель R2, DM и H1 на h=22 считаются как у прочих h."""
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
            if 22 in r2_hs and 22 in P:
                dmc = FT._dm_cols(P[22], 22)
                for i in S.index[S.h == 22]:
                    S.loc[i, ['DM t vs HAR-J (б)', 'p vs HAR-J (б)']] = [dmc['DM t vs HAR-J (б)'], dmc['p vs HAR-J (б)']]
                    if h1:
                        S.loc[i, 'H1 (основной DM против HAR-J (б), 5%)'] = 'принята' if FT.S.dm_main(P[22], FTST.ETALON, 22)['p_one'] < 0.05 else 'не принята'
            if note:
                S['пометка'] = note
            T['svodka'] = S
            for part, df in T.items():
                TD.write_named(df, f'table_3_{tag}_{part}_{suffix}.csv')
    finally:
        TD.TABLES = tab0


def periody(frames):
    rows = []
    mods = {**FTST.MODELS, F_NAME + ' (F_h(d0))': 'F_d0', 'HAR-CJ d0 (справочно)': 'harcj_d0',
            **{f'{REF_NG[m]} (справочно)': f'ref_{m.lower()}' for m in REF_NG},
            'HAR-J (б) d0, s² без эмбарго — ВЫР-1 (справочно)': 'har_corr_bez_emb', 'F_h(d0), s² без эмбарго — ВЫР-1 (справочно)': 'F_d0_bez_emb'}
    for h, df in frames.items():
        for nm, (a, b) in PERIODY.items():
            g0 = df[df.year.between(a, b)]
            for m, c in mods.items():
                g = g0[g0[c].notna()]
                if g.empty:
                    continue
                rows.append(dict(h=h, Период=nm, Модель=m, n=len(g), **{k: FTST.fmt(v, k) for k, v in FTST.metrics(g.rv_actual.values, g[c].values).items()}))
    return pd.DataFrame(rows)


def dm_vs_F(frames, r2_hs):
    out = []
    for h, df in frames.items():
        if h in r2_hs:  # без победителя гибрид = F_h(d0): разность тождественно 0
            out += [dict(H=h, эталон='гибрид раунда 1 d0 (F_h(d0)), справочно', **r) for r in FTST.dm(df, 'F_d0', h)]
    return pd.DataFrame(out, columns=['H', 'эталон', 'способ', 'n', 't', 'p_two', 'p_one_H1', 'mean_d'])


# ---------------------------------------------------------------- h=22: вторичные свидетельства (блок владельца 20:10, объявлены в FD до открытия)
V22 = {'kontrol': ('2022–2026 (контрольный)', 2022, 2026, ''),
       'vse': ('2017–2026', 2017, 2026, 'годы 2017–2021 участвовали в отборе (LOYO) — не независимая оценка')}
SHAG22 = 22


def tochki22(dates, shag=SHAG22):
    """Неперекрывающиеся точки отсчёта: отсортированные торговые даты периода (объединение по тикерам кадра), первая дата периода
    и далее каждая 22-я (позиции 0, 22, 44, …). Окна цели [t+1, t+22] соседних точек не пересекаются."""
    u = np.sort(pd.unique(pd.to_datetime(dates)))
    return u[::shag]


def dm_neperekr(df, ref, shag=SHAG22):
    """DM по неперекрывающимся точкам: в каждой точке — средняя по тикерам разность потерь QLIKE (эталон − гибрид); t без NW (лаг 0)."""
    t0 = set(tochki22(df.date.values, shag))
    g = df[pd.to_datetime(df.date).isin(t0)]
    d = FT.S.qloss(g.rv_actual.values, g[ref].values) - FT.S.qloss(g.rv_actual.values, g.rv_pred.values)
    return FT.S.dm_series(FT.S.daily(g.date.values, d), 1), int(g.date.nunique())


def vtorichnye22(fr, B, rlab):
    """h=22: основной — панельный DM (дневной ряд средней по тикерам, NW лаг 21) против HAR-J (б) на 2022–2026; вторичные — DM на
    2017–2026 тем же способом, MCS, DM по неперекрывающимся месячным точкам (каждая 22-я торговая дата, без NW). -> таблица."""
    if 22 not in fr:
        return pd.DataFrame()
    rows = []
    for tag, (label, a, b, ogov) in V22.items():
        g = fr[22][fr[22].year.between(a, b)].reset_index(drop=True)
        r = FT.S.dm_main(g, FTST.ETALON, 22)
        rows.append(dict(svidetelstvo='основное' if tag == 'kontrol' else 'вторичное', metod='панельный DM (дневной ряд, NW лаг 21) против HAR-J (б)',
                         period=label, n=r['n'], t=round(r['t'], 2), p_odnostor=FT.pf(r['p_one']), mean_d=r['mean_d'], ogovorka=ogov))
        r2, nd = dm_neperekr(g, FTST.ETALON)
        rows.append(dict(svidetelstvo='вторичное', metod='DM по неперекрывающимся месячным точкам (каждая 22-я торговая дата), без NW',
                         period=label, n=nd, t=round(r2['t'], 2), p_odnostor=FT.pf(r2['p_one']), mean_d=r2['mean_d'], ogovorka=ogov))
        mc = FT.mcs_table({22: g}, B)
        for _, x in mc.iterrows():
            rows.append(dict(svidetelstvo='вторичное', metod=f'MCS: {x["Модель"]}', period=label, n=int(x['дней']), t=np.nan,
                             p_odnostor=np.nan, mean_d=x['средний дневной QLIKE'], p_MCS_T_R=x['p_MCS T_R'], p_MCS_T_max=x['p_MCS T_max'],
                             ogovorka=ogov))
    return pd.DataFrame(rows).assign(h=22, pravilo=rlab, primechanie='основной результат h=22 — панельный DM на 2022–2026; прочее — вторичное')


# ---------------------------------------------------------------- сверка кода с первым тестом (только строки, не менявшиеся в d0)
# (имя таблицы, разделитель, ключ, колонка модели, исключаемые префиксы колонок, только эти колонки)
SVERKA = [(n, ',', ['Модель'], 'Модель', (), None) for n in ('3_4', '3_4_rmse', '3_4_mae', '3_6', '3_6_rmse', '3_6_mae', '3_7', '3_7_rmse', '3_7_mae')] + [
    ('3_final_po_godam', ',', ['h', 'Модель', 'Год'], 'Модель', (), None),
    ('3_mz', ';', ['h', 'Шкала', 'Модель'], 'Модель', (), None),
    ('3_mcs', ';', ['h', 'Модель'], 'Модель', ('p_MCS', 'в MCS'), None),
    ('3_subperiods', ';', ['h', 'Период'], None, (), ['n', 'дней', 'QLIKE Лучшая одиночная бустинга']),
    ('3_terciles', ';', ['h', 'Терциль RV_m'], None, (), ['n', 'дней', 'QLIKE Лучшая одиночная бустинга']),
]
PER_KEYS = {'svodka': (['Период', 'правило', 'h'], None, ['n', 'дней', 'QLIKE Лучшая одиночная бустинга']),
            'subperiods': (['Период', 'правило', 'h', 'Подпериод'], None, ['n', 'дней', 'QLIKE Лучшая одиночная бустинга']),
            'mcs': (['Период', 'правило', 'h', 'Модель'], 'Модель', ['средний дневной QLIKE', 'дней'])}  # dm, znak, tickers — от эталонов


def compare(a, b, key, col, only, excl, hs, name):
    bad, n = [], 0
    if col:
        a, b = a[a[col].isin(SAME)], b[b[col].isin(SAME)]
    if 'h' in key:
        a = a[a.h.isin([str(h) for h in hs])]
    cols = only or [c for c in b.columns if c in a.columns and c not in key and not any(c.startswith(e) for e in excl)]
    m = a.merge(b, on=key, how='inner', suffixes=('_a', '_b'))
    if len(m) != len(b) or len(m) == 0 or len(m) != len(a):
        return [f'{name}: строк в R2 {len(b)}, в первом тесте {len(a)}, совпало ключей {len(m)}'], 0
    for c in cols:
        if c + '_a' not in m:
            bad.append(f'{name}: нет колонки {c}')
            continue
        diff = m[m[c + '_a'].fillna('') != m[c + '_b'].fillna('')]
        n += len(m)
        if len(diff):
            bad.append(f'{name}/{c}: {len(diff)} ячеек, напр. {diff.iloc[0][key].tolist()} {diff.iloc[0][c + "_a"]} != {diff.iloc[0][c + "_b"]}')
    return bad, n


def sverka(first_dir, r2_dir, hs, tag_first='test_raw', tag_r2=TAG):
    """Таблицы final_test.run: строковое (побайтное) совпадение ячеек моделей, не менявшихся в d0."""
    bad, n = [], 0
    for name, sep, key, col, excl, only in SVERKA:
        a = pd.read_csv(first_dir / f'table_{name}_{tag_first}.csv', sep=sep, dtype=str, encoding='utf-8-sig')
        b = pd.read_csv(r2_dir / f'table_{name}_{tag_r2}.csv', sep=sep, dtype=str, encoding='utf-8-sig')
        bb, k = compare(a, b, key, col, only, excl, hs, name)
        bad += bb; n += k
    return bad, n


def sverka_per(first_dir, r2_dir, hs, rule, first_rule_tag):
    """Таблицы периодов test_dop2 (vse / kontrol / neotkr) против table_3_<период>_<часть>_test_<first_rule_tag>.csv."""
    bad, n = [], 0
    for tag in TD2.PER:
        for part, (key, col, only) in PER_KEYS.items():
            a = pd.read_csv(first_dir / f'table_3_{tag}_{part}_test_{first_rule_tag}.csv', sep=';', dtype=str, encoding='utf-8-sig')
            b = pd.read_csv(r2_dir / f'table_3_{tag}_{part}_test_r2_{rule}.csv', sep=';', dtype=str, encoding='utf-8-sig')
            bb, k = compare(a, b, key, col, only, (), hs, f'{tag}_{part}')
            bad += bb; n += k
    return bad, n


def run_r2(src, locked, first_tables, out_tables, feat, B, lock=True, need_fd=True, close='obe', first_tag='test_raw',
           first_close_tag='s_zakrytiem', ft=None, check_sha=True, shas=None, fd=FD):
    """close: 'obe' (по умолчанию) — основная версия и версия с правилом закрытия биржи; 'bez' / 'pravilo' — одна.
    ft — {тикер: (последний торговый день до 28.02.2022, первый после 23.03.2022)} для синтетики; None — test_dop2.first_trade().
    shas — {путь от test_locked: SHA-256}; None — r2_A_sborka.yaml (zamorozheno)."""
    src, locked, first_tables, out_tables = map(Path, (src, locked, first_tables, out_tables))
    hs = tuple(FTST.HS)
    r2_hs = precheck(src, locked, shas if shas is not None else (frozen_sha() if check_sha else {}), check_sha, fd, need_fd)
    lk = src / 'OPENED_R2.json'
    if lock:
        if lk.exists():
            raise Stop(f'{lk} уже есть: второе открытие было ({lk.read_text()[:200]})')
        src.mkdir(parents=True, exist_ok=True)
        lk.write_text(json.dumps(dict(time=now(), h=hs, r2_h=r2_hs, close=close, code=sha(__file__),
                                      src={f.name: sha(f) for f in sorted(src.glob('final_r2_h*_test.parquet'))},
                                      fd=sha(fd) if Path(fd).exists() else None), ensure_ascii=False, indent=1))
    out_tables.mkdir(parents=True, exist_ok=True)
    frames = {h: frame_d0(src, locked, h) for h in hs}
    for h, df in frames.items():
        print(f'h={h}: гибрид — {", ".join(sorted(df.design.unique()))}; строк {len(df)}', flush=True)
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
            period_tables(fr, RULES[ver], out_tables, f'test_r2_{RULES[ver]}', B, r2_hs)
        periody(fr).to_csv(out_tables / f'table_3_periody_{tag}.csv', index=False)
        dm_vs_F(fr, r2_hs).to_csv(out_tables / f'table_3_dm_vs_F_{tag}.csv', index=False)
        with r2_env(hs, out_tables):
            vtorichnye22(fr, B, TD2.RULE[RULES[ver]]).to_csv(out_tables / f'table_3_h22_vtorichnye_{tag}.csv', index=False)
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
            raise Stop(f'версия {ver}: строки, не менявшиеся в d0, не совпали с первым тестом:\n' + '\n'.join(bad[:10]))
        print(f'{ver}: сверка кода с первым тестом ({ftag3 or "—"} / {ftagP or "—"}; V0 и лучшая одиночная): {n} ячеек, расхождений нет; '
              f'h = {hs}, победитель R2 на h = {r2_hs}', flush=True)
        res[ver] = n
    return hs, r2_hs, res


# ---------------------------------------------------------------- самопроверка (синтетика; тест не читается)
def selftest():
    import ftp_selftest as FS
    rng = np.random.default_rng(26)
    tmp = Path(tempfile.mkdtemp(prefix='r2_selftest_'))
    lkd = tmp / 'locked'
    fz, vy, cjd, t1, src = lkd / 'final3', lkd / 'vyr', lkd / 'vyr' / 'ref_harcj_d0', tmp / 'tables1', lkd / 'R2' / 'final'
    for d in (fz, cjd, src, tmp / 'data' / 'regen'):
        d.mkdir(parents=True)
    late = 'T16'  # тикер открылся 28.03.2022: строк за 24–25.03 нет (как LSRG, MTLR, OGKB)
    gg = {}
    noise = lambda x, s=0.15: x * np.exp(rng.normal(0, s, len(x)))
    for h in FTST.HS:
        g = FS.make(rng, start='2017-01-03', end='2026-02-27')
        g = g[~g.date.between(*HALT)]  # как в данных: дней остановки торгов нет
        g = g[~((g.ticker == late) & g.date.between('2022-03-24', '2022-03-25'))].reset_index(drop=True)
        g['step'] = g.year - 2016
        g.loc[g.date > '2026-02-20', 'rv_actual'] = np.nan  # шаг 10: строки без цели, как в данных
        g.drop(columns='rv_m').assign(design='r1').to_parquet(fz / f'final_h{h}_test.parquet', index=False)
        gg[h] = g
        # ВЫР-1: эталоны d0 и F_h(d0) — part = test (шаги 1–10) и val (шаги 7–10, в кадр не входят)
        et = g[KEY + ['year', 'rv_actual']].assign(har_raw=noise(g.har_raw.values), har_corr=noise(g.har_corr.values), F_sm=1.1)
        val = et[et.step.between(7, 10)].assign(step=lambda x: x.step - 1, har_raw=-1.0, har_corr=-1.0, part='val')
        pd.concat([et.assign(part='test'), val]).to_parquet(vy / f'etalon_d0_h{h}_test.parquet', index=False)
        fdv = et.har_corr.values if h == 22 else noise(g.rv_pred.values)  # F_22(d0) = эталон (б) d0
        fd = g[KEY + ['year', 'rv_actual']].assign(rv_pred=fdv, leg_raw=fdv, F_sm=1.1, design=f'F_{h} d0')
        pd.concat([fd.assign(part='test'), fd[fd.step >= 7].assign(part='val', step=lambda x: x.step - 1, rv_pred=-1.0)]).to_parquet(
            vy / f'final_d0_h{h}_test.parquet', index=False)
        cj = g[KEY + ['year', 'rv_actual']].assign(rv_pred_raw=1.0, F=1.0, rv_pred=noise(g.har_corr.values, 0.3), n_train=1)
        cj.drop(index=cj.index[1000:1300]).to_parquet(cjd / f'REF_HARCJ_d0_h{h}_test.parquet', index=False)  # справочная: неполное покрытие допустимо
        # r2_sborka_test: эталоны и F_h(d0) с s² по t−1 и эмбарго (сырой эталон тот же, поправка другая); цели нет
        e2 = et[KEY + ['year', 'har_raw']].assign(har_corr=noise(et.har_corr.values, 0.05), F_sm=1.12, F_sm_bez_emb=1.1)
        e2.to_parquet(src / f'etalon_d0_s2emb_h{h}_test.parquet', index=False)
        f2 = e2.har_corr.values if h == 22 else noise(fdv, 0.05)
        fd[KEY + ['year', 'leg_raw', 'design']].assign(rv_pred=f2, F_sm=1.12, F_sm_bez_emb=1.1).to_parquet(
            src / f'final_d0_s2emb_h{h}_test.parquet', index=False)
        (lkd / 'ref_nn_garch').mkdir(exist_ok=True)
        for i, mod in enumerate(REF_NG):  # справочные NN / GARCH: как есть; у GARCH — неполное покрытие
            rf = g[KEY + ['year', 'rv_actual']].assign(rv_pred_raw=1.0, rv_pred=noise(g.har_corr.values, 0.2 + 0.05 * i))
            (rf.drop(index=rf.index[2000:2100]) if 'GARCH' in mod or mod == 'GJR' else rf).to_parquet(
                lkd / 'ref_nn_garch' / f'REF_{mod}_h{h}_test.parquet', index=False)
    shas = {r: sha(lkd / r) for h in FTST.HS for r in frozen_files(h).values()}
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
        # 2. «первый тест» на синтетике (до выравнивания): final_test.run и таблицы периодов test_dop2 в обеих версиях
        feat = pd.concat([FS.make(rng, start='2014-01-02', end='2016-12-30'), g])[['date', 'ticker', 'rv_m']]
        FTST.run(fz, t1, tag='test_raw', feat=feat, B=300)  # как table_3_*_test_raw.csv первого теста
        FT.set_etalon(FTST.ETALON)
        period_tables(D, 'bez_zakrytiya', t1, 'test_bez_zakrytiya', 300, note=None)
        period_tables({h: d[~ex[h]].reset_index(drop=True) for h, d in D.items()}, 's_zakrytiem', t1, 'test_s_zakrytiem', 300, note=None)
        for h in (5, 22):  # победитель R2 на h=5 и 22; h=1 — гибрид = F_1(d0)
            x = pd.read_parquet(fz / f'final_h{h}_test.parquet')
            x['rv_pred'] = noise(x.rv_pred.values, 0.1)
            x[R2_COLS].assign(design=f'R2_h{h}').to_parquet(src / f'final_r2_h{h}_test.parquet', index=False)

        def copy_src(name, base=src, only=None):
            d = tmp / name
            d.mkdir()
            for f in base.glob('final_r2_h*_test.parquet'):
                if only is None or f.name in only:
                    (d / f.name).write_bytes(f.read_bytes())
            return d
        srcb, srcp, src2, src3, src4 = (copy_src(n) for n in ('R2_bez', 'R2_pravilo', 'R2_m1', 'R2_m2', 'R2_m3'))
        src5 = copy_src('R2_h5', only={'final_r2_h5_test.parquet'})
        fd_ok = tmp / 'FINAL_DESIGN_round2.md'
        fd_ok.write_text('# заполнен\n')
        kw = dict(ft=ft, shas=shas, fd=fd_ok)
        # 3. обе версии; все h в d0; сверка строк, не менявшихся в d0, с «первым тестом»
        hs, r2_hs, res = run_r2(src, lkd, t1, tmp / 'tables2', feat, B=300, **kw)
        assert hs == (1, 5, 22) and r2_hs == (5, 22) and res['bez'] > 0 and res['pravilo'] > 0, (hs, r2_hs, res)
        t2 = tmp / 'tables2'
        for tag in (TAG, TAG_Z):
            for n in ('3_2', '3_7', '3_9', '3_mcs', '3_mz', '3_periody', '3_dm_vs_F'):
                assert (t2 / f'table_{n}_{tag}.csv').exists(), (n, tag)
        for rule in RULES.values():
            for per in TD2.PER:
                for part in ('svodka', 'dm', 'mcs', 'tickers', 'subperiods', 'znak'):
                    assert (t2 / f'table_3_{per}_{part}_test_r2_{rule}.csv').exists(), (per, part, rule)
        assert (t2 / 'table_3_zakrytie_isklyucheno_test_r2.csv').exists()
        # 3a. эталоны — d0 (ВЫР-1), а не первого прогона: 3.2 (HAR-J (а) по годам) пересчитана и равна прямому счёту по эталону d0
        a32, b32 = (pd.read_csv(p, dtype=str) for p in (t1 / 'table_3_2_test_raw.csv', t2 / f'table_3_2_{TAG}.csv'))
        assert (a32['QLIKE H = 5'] != b32['QLIKE H = 5']).any(), 'эталон (а) должен быть d0'
        e5 = pd.read_parquet(vy / 'etalon_d0_h5_test.parquet')
        e5 = e5[(e5.part == 'test') & (e5.year == 2019) & (e5.rv_actual > 1e-12)]
        q = FTST.fmt(FTST.metrics(e5.rv_actual.values, e5.har_raw.values)['QLIKE'], 'QLIKE')
        assert float(b32[b32['Год'] == '2019']['QLIKE H = 5'].iloc[0]) == q, 'HAR-J (а) d0, 2019, h=5'
        # 3b. гибрид: h=1 — F_1(d0), h=5 и 22 — R2; F_h(d0) отдельной строкой
        p = pd.read_csv(t2 / f'table_3_periody_{TAG}.csv')
        assert set(p.Период) == set(PERIODY) and set(p.h) == {1, 5, 22}
        hyb, fh = (p[p['Модель'] == m].set_index(['h', 'Период']).QLIKE for m in ('Гибридная', F_NAME + ' (F_h(d0))'))
        assert (hyb.loc[1] == fh.loc[1]).all() and (hyb.loc[5] != fh.loc[5]).any() and (hyb.loc[22] != fh.loc[22]).any()
        q_b, q_b0 = (p[p['Модель'] == m].set_index(['h', 'Период']).QLIKE for m in (FTST.ETALON_NAME, 'HAR-J (б) d0, s² без эмбарго — ВЫР-1 (справочно)'))
        assert (q_b != q_b0).any(), 'HAR-J (б) — из сборки (s² t−1 с эмбарго), ВЫР-1 — только справочно'
        e2 = pd.read_parquet(src / 'etalon_d0_s2emb_h5_test.parquet').merge(D[5][KEY + ['rv_actual']], on=KEY)
        e2 = e2[e2.year.between(2022, 2026)]
        assert q_b.loc[(5, '2022–2026 (основной)')] == FTST.fmt(FTST.metrics(e2.rv_actual.values, e2.har_corr.values)['QLIKE'], 'QLIKE')
        q_f0 = p[p['Модель'] == 'F_h(d0), s² без эмбарго — ВЫР-1 (справочно)'].set_index(['h', 'Период']).QLIKE
        assert (fh.loc[1] != q_f0.loc[1]).any(), 'F_h(d0) — из сборки'
        cjn = p[p['Модель'] == 'HAR-CJ d0 (справочно)']
        assert len(cjn) == 9 and (cjn.n < p[p['Модель'] == 'Гибридная'].n.values).any()
        for mod, nm in REF_NG.items():  # справочные NN / GARCH — в таблице периодов, прямой счёт по файлу
            rn = p[p['Модель'] == f'{nm} (справочно)']
            assert len(rn) == 9, nm
            rf = pd.read_parquet(lkd / 'ref_nn_garch' / f'REF_{mod}_h5_test.parquet').merge(D[5][KEY], on=KEY)
            rf = rf[rf.year.between(2022, 2026) & (rf.rv_actual > 1e-12)]
            q = FTST.fmt(FTST.metrics(rf.rv_actual.values, rf.rv_pred.values)['QLIKE'], 'QLIKE')
            assert rn.set_index(['h', 'Период']).QLIKE.loc[(5, '2022–2026 (основной)')] == q, nm
        assert not {f'MCS: {nm}' for nm in REF_NG.values()} & set(pd.read_csv(t2 / f'table_3_h22_vtorichnye_{TAG}.csv').metod), 'справочные — не в MCS'
        dmf = pd.read_csv(t2 / f'table_3_dm_vs_F_{TAG}.csv')
        assert set(dmf.H) == {5, 22}, 'DM против F_h(d0) — только h с победителем'
        mc = pd.read_csv(t2 / f'table_3_mcs_{TAG}.csv', sep=';', decimal=',')
        assert {'Финальная', F_NAME, 'HAR-J (б)'} <= set(mc['Модель'])
        sv = pd.read_csv(t2 / 'table_3_kontrol_svodka_test_r2_bez_zakrytiya.csv', sep=';', decimal=',', encoding='utf-8-sig')
        assert sv[sv.h == 22]['DM t vs HAR-J (б)'].notna().all(), 'DM h=22 у победителя R2 должен считаться'
        assert NOTE_R2 in set(sv['пометка'])
        # 3d. h=22: основной — панельный DM 2022–2026; вторичные — DM 2017–2026 (с оговоркой), MCS, DM по неперекрывающимся точкам
        v = pd.read_csv(t2 / f'table_3_h22_vtorichnye_{TAG}.csv')
        osn = v[v.svidetelstvo == 'основное']
        assert len(osn) == 1 and osn.period.iloc[0].startswith('2022') and osn.metod.iloc[0].startswith('панельный')
        assert (v.svidetelstvo == 'вторичное').sum() == len(v) - 1
        F22 = frame_d0(src, lkd, 22)
        K = F22[F22.year.between(2022, 2026)]
        r = FT.S.dm_main(K, FTST.ETALON, 22)
        assert abs(osn.t.iloc[0] - round(r['t'], 2)) < 1e-9
        r = FT.S.dm_main(F22, FTST.ETALON, 22)
        vs = v[(v.period == '2017–2026') & v.metod.str.startswith('панельный')]
        assert vs.svidetelstvo.iloc[0] == 'вторичное' and abs(vs.t.iloc[0] - round(r['t'], 2)) < 1e-9 and 'отборе' in vs.ogovorka.iloc[0]
        u = np.sort(K.date.unique())
        pts = u[::22]
        pos = np.searchsorted(u, pts)
        assert pos[0] == 0 and (np.diff(pos) == 22).all(), 'точки: позиции 0, 22, 44, … торгового календаря периода'
        g = K[K.date.isin(pts)]
        s = pd.Series(FT.S.qloss(g.rv_actual.values, g[FTST.ETALON].values) - FT.S.qloss(g.rv_actual.values, g.rv_pred.values),
                      index=g.date.values).groupby(level=0).mean()
        t_ref = s.mean() / np.sqrt(s.var(ddof=0) / len(s))  # без NW
        row = v[v.period.str.startswith('2022') & v.metod.str.startswith('DM по неперекр')]
        assert row.n.iloc[0] == len(pts) == int(np.ceil(len(u) / 22)) and abs(row.t.iloc[0] - round(t_ref, 2)) < 1e-9
        assert set(v[v.metod.str.startswith('MCS')].period) == {'2022–2026 (контрольный)', '2017–2026'}
        assert {'MCS: Финальная', 'MCS: HAR-J (б)', f'MCS: {F_NAME}'} <= set(v.metod)
        assert (t2 / f'table_3_h22_vtorichnye_{TAG_Z}.csv').exists()
        vz = pd.read_csv(t2 / f'table_3_h22_vtorichnye_{TAG_Z}.csv')
        assert set(vz.pravilo) == {TD2.RULE['s_zakrytiem']} and (vz.t.dropna() != v.t.dropna()).any()
        fpg = pd.read_csv(t2 / f'table_3_final_po_godam_{TAG}.csv')
        f1 = pd.read_csv(t1 / 'table_3_final_po_godam_test_raw.csv')
        a = fpg[(fpg['Модель'] == 'Гибридная') & (fpg.h == 5)].QLIKE.values
        b = f1[(f1['Модель'] == 'Гибридная') & (f1.h == 5)].QLIKE.values
        assert not np.allclose(a, b), 'строки гибрида должны быть R2, а не раунда 1'
        pa, pz = (pd.read_csv(t2 / f'table_3_final_po_godam_{t}.csv') for t in (TAG, TAG_Z))
        mm = pa.merge(pz, on=['h', 'Модель', 'Год'], suffixes=('_a', '_z'))
        y22, oth = mm[mm['Год'] == '2022'], mm[~mm['Год'].isin(['2022', 'все годы (пул)'])]
        assert (y22.QLIKE_a != y22.QLIKE_z).any() and (oth.QLIKE_a == oth.QLIKE_z).all(), 'правило меняет только 2022 (и пул)'
        # 3c. без победителя на h=22: гибрид = F_22(d0) = эталон (б) d0, svodka — «финал = эталон»
        _, r5, _ = run_r2(src5, lkd, t1, tmp / 'tables_h5', feat, B=100, close='bez', **kw)
        assert r5 == (5,)
        sv5 = pd.read_csv(tmp / 'tables_h5' / 'table_3_kontrol_svodka_test_r2_bez_zakrytiya.csv', sep=';', decimal=',', encoding='utf-8-sig')
        assert sv5[sv5.h == 22]['p vs HAR-J (б)'].eq('финал = эталон').all()
        # 4. флаг: только одна версия
        _, _, rb = run_r2(srcb, lkd, t1, tmp / 'tables_b', feat, B=100, close='bez', **kw)
        assert set(rb) == {'bez'} and not list((tmp / 'tables_b').glob(f'*_{TAG_Z}.csv')) and not list((tmp / 'tables_b').glob('*s_zakrytiem*'))
        _, _, rp = run_r2(srcp, lkd, t1, tmp / 'tables_p', feat, B=100, close='pravilo', **kw)
        assert set(rp) == {'pravilo'} and rp['pravilo'] > 0 and not (tmp / 'tables_p' / f'table_3_2_{TAG}.csv').exists()
        # 5. одно открытие: замок
        try:
            run_r2(src, lkd, t1, tmp / 'tables3', feat, B=100, **kw)
            raise AssertionError('замок не сработал')
        except Stop as e:
            assert 'уже есть' in str(e)

        def stop_before_lock(srcx, what, **k):
            try:
                run_r2(srcx, lkd, t1, tmp / 'tables_x', feat, B=100, **{**kw, **k})
                raise AssertionError(f'не пойман: {what}')
            except Stop as e:
                assert what in str(e), (what, e)
            assert not (srcx / 'OPENED_R2.json').exists(), f'{what}: открытие потрачено'
        # 6. до замка: rv_actual в файле R2, неполный файл, SHA замороженного (кадр, эталон d0, F_h(d0), HAR-CJ d0), шаблон FD
        bad = tmp / 'bad'; bad.mkdir()
        x = pd.read_parquet(fz / 'final_h5_test.parquet')
        x[R2_COLS + ['rv_actual']].to_parquet(bad / 'final_r2_h5_test.parquet', index=False)
        stop_before_lock(bad, 'rv_actual')
        short = tmp / 'short'; short.mkdir()
        x[R2_COLS].iloc[:-5].to_parquet(short / 'final_r2_h5_test.parquet', index=False)
        stop_before_lock(short, 'строк R2')
        for r in ('final3/final_h1_test.parquet', 'vyr/etalon_d0_h5_test.parquet', 'vyr/final_d0_h22_test.parquet',
                  'vyr/ref_harcj_d0/REF_HARCJ_d0_h1_test.parquet', 'R2/final/etalon_d0_s2emb_h5_test.parquet',
                  'R2/final/final_d0_s2emb_h1_test.parquet'):
            stop_before_lock(src4, 'SHA-256', shas={**shas, r: '0' * 64})
        stop_before_lock(src4, 'не записан', shas={r: s for r, s in shas.items() if 's2emb_h22' not in r})
        fd_t = tmp / 'FD_template.md'
        fd_t.write_text(f'| s² | {PLACEHOLDER} |\n')
        stop_before_lock(src4, PLACEHOLDER, fd=fd_t)
        stop_before_lock(src4, 'нет', fd=tmp / 'net.md')
        if FD.exists() and PLACEHOLDER in FD.read_text():  # шаблон в репозитории открытие блокирует
            stop_before_lock(src4, PLACEHOLDER, fd=FD)
        z = frozen_sha()  # yaml: 12 замороженных до R2, SHA кадра = test_dop.SHA; собранные (sobrano) — после сборки
        assert len([r for r in z if 's2emb' not in r]) == 24  # 12 ВЫР / кадр + 12 справочных NN / GARCH
        # 7. строки, не менявшиеся в d0, не совпали с первым тестом — СТОП: в 3.x (основная) и в таблице периодов с правилом
        for nm, (fname, key, row, colname, srcx) in {
                '3_4': ('table_3_4_test_raw.csv', 'Модель', 'XGBoost', 'H = 5', src2),
                'kontrol_svodka': ('table_3_kontrol_svodka_test_s_zakrytiem.csv', 'h', '5', 'QLIKE Лучшая одиночная бустинга', src3)}.items():
            t1b = tmp / f'tables1_{nm}'
            t1b.mkdir()
            for f in t1.glob('*.csv'):
                (t1b / f.name).write_bytes(f.read_bytes())
            sep = ',' if nm == '3_4' else ';'
            d = pd.read_csv(t1b / fname, dtype=str, sep=sep, encoding='utf-8-sig')
            d.loc[d[key] == row, colname] = '9,9999'
            d.to_csv(t1b / fname, index=False, sep=sep)
            try:
                run_r2(srcx, lkd, t1b, tmp / f'tables6_{nm}', feat, B=100, **kw)
                raise AssertionError(f'расхождение {nm} не поймано')
            except Stop as e:
                assert nm in str(e) and (tmp / f'tables6_{nm}' / 'STOP_R2.txt').exists(), e
        # 8. цель эталонов ВЫР-1 не совпала с целью кадра — СТОП
        e1 = pd.read_parquet(vy / 'etalon_d0_h1_test.parquet')
        e1.loc[e1.index[100], 'rv_actual'] *= 1.5
        vy2 = tmp / 'locked2'
        for f in lkd.rglob('*.parquet'):
            if not f.name.startswith('final_r2_'):
                (vy2 / f.relative_to(lkd)).parent.mkdir(parents=True, exist_ok=True)
                (vy2 / f.relative_to(lkd)).write_bytes(f.read_bytes())
        e1.to_parquet(vy2 / 'vyr' / 'etalon_d0_h1_test.parquet', index=False)
        try:
            run_r2(copy_src('R2_m4'), vy2, t1, tmp / 'tables9', feat, B=100, **{**kw, 'check_sha': False})
            raise AssertionError('расхождение цели не поймано')
        except Stop as e:
            assert 'цель эталонов' in str(e), e
    finally:
        TD2.DATA = data0
    print('selftest final_test_r2: OK (версия d0: эталоны (а), (б) и F_h(d0) — сырой ряд ВЫР-1 с s² сборки (t−1, эмбарго), ВЫР-1 с s² R1 — '
          'справочно; HAR-CJ d0 — ВЫР-2; гибрид — R2 или F_h(d0); правило закрытия биржи: test_dop2.masks = текст дирижёра, обе версии и '
          'флаг bez/pravilo/obe; сверка строк, не менявшихся в d0, с первым тестом; до замка — SHA 12 замороженных и 6 собранных (sobrano), '
          'шаблон FD, rv_actual и число строк R2; замок; цель эталонов = цель кадра; h=22 — основной панельный DM 2022–2026 и вторичные: '
          'DM 2017–2026 с оговоркой, MCS, DM по неперекрывающимся точкам каждые 22 торговые даты без NW — сверены независимым счётом):', tmp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scope', choices=['test'])
    ap.add_argument('--selftest', action='store_true')
    ap.add_argument('--src', default=str(LOCKED / 'R2/final'))
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
    src = Path(a.src if Path(a.src).is_absolute() else MAIN / a.src)
    try:
        precheck(src, LOCKED, frozen_sha())  # до чтения матрицы признаков: отказ не тратит открытие
        feat = pd.read_parquet(a.features, columns=['date', 'ticker', 'rv_m'])
        feat['date'] = pd.to_datetime(feat['date'])
        run_r2(src, LOCKED, TABLES, TABLES, feat, a.mcs_b, close=a.close_rule, first_close_tag=a.first_close_tag)
    except Stop as e:
        print('СТОП:', e, flush=True)
        sys.exit(3)


if __name__ == '__main__':
    main()
