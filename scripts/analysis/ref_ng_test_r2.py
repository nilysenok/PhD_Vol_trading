#!/usr/bin/env python3
"""R2: справочные GRU, LSTM, GARCH(1,1), GJR-GARCH на тесте второго открытия — отдельный справочный режим (решение владельца 21:40,
FINAL_DESIGN_round2: «справочные строки без изменений — отдельным PR после открытия, не в H1»). Агент A2, 26.10.2026.

python3 scripts/analysis/ref_ng_test_r2.py            # счёт, сверки, запись docs/tables/table_3_ref_ng_test_r2*.csv

Это НЕ открытие теста: final_test_r2.py --scope test не запускается, замок OPENED_R2.json не трогается, таблицы открытия
(docs/tables/*test_r2*, 87 файлов) не перезаписываются — до и после счёта сверяются их SHA-256 (выход ≠ 0 при изменении).

Что читается из test_locked/ (только чтение; SHA-256 сверяются ДО чтения данных, несовпадение — СТОП):
  ref_nn_garch/REF_{GRU,LSTM,GARCH,GJR}_h{h}_test.parquet — 12 файлов; SHA — журнал записи docs/otchety/2026-10-26_ref_nn_garch.md
    («Хэши SHA-256 (итог, 08:15)»); по ВЫР-2 (PR #154) день d видят, не перестраивались; столбцы date, ticker, rv_actual, rv_pred;
  R2/final/etalon_d0_s2emb_h{h}_test.parquet (har_raw — (а), har_corr — (б)) и final_d0_s2emb_h{h}_test.parquet (rv_pred — финал
    = F_h(d0)) — 6 файлов; SHA — models/designs/r2/r2_A_sborka.yaml (sobrano). Кадр первого теста (final3) НЕ читается.
Кадр: ключи (date, ticker, step) — файлы эталонов сборки; цель rv_actual — из справочных файлов (одинакова во всех четырёх, assert);
  фильтр rv_actual > 1e-12, сортировка (date, ticker) — как final_test_r2.frame_d0. Прогноз справочной модели ≤ 0 — пропуск (как
  tables_dlya_teksta.srez3); QLIKE модели — по строкам с прогнозом, покрытие пишется.
Сверки (расхождение — СТОП, файлы не пишутся):
  (1) n и QLIKE / RMSE / MAE строк «Гибридная», «HAR-J», «HAR-J с поправкой» по трём периодам = table_3_periody_test_r2{,_zakrytie}.csv
      открытия (строковое совпадение в формате final_test.fmt) — кадр и эталоны те же, что у открытия;
  (2) правило закрытия биржи (final_test_r2.zakrytie — текст дирижёра) = table_3_zakrytie_isklyucheno_test_r2.csv: по тикеру число
      исключённых точек, первая и последняя исключённая t;
  (3) QLIKE пул справочных моделей на 2022–2026 и 2017–2026 (обе версии правила) = dlya_teksta/istochnik_sravnenie_test.csv первого
      теста (те же ряды, ВЫР-2: не перестраивались), |Δ| < 1e-12;
  (4) DM финала против (а) и (б) тремя способами (final_test.dm) на 2022–2026 и 2017–2026 = table_3_dm5_test_r2{,_zakrytie}.csv
      открытия (n и t до 2 знаков) — код DM тот же.
Выход (новые файлы):
  table_3_ref_ng_test_r2.csv / _zakrytie.csv — h, Период, Модель, n, покрытие, QLIKE, RMSE, MAE, к (а) %, к (б) % (знак: (QLIKE_модели /
    QLIKE_эталона − 1)·100, плюс = хуже эталона); периоды — final_test_r2.PERIODY;
  table_3_ref_ng_test_r2_dm.csv / _dm_zakrytie.csv — DM против (а) и (б) тремя способами final_test.dm (основной — панельный: дневной ряд
    средней по тикерам, NW лаг h−1); d = QLIKE эталона − QLIKE модели, t > 0 — модель лучше эталона. Справочно, не H1.
"""
import hashlib, subprocess, sys
from pathlib import Path
import numpy as np, pandas as pd
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import final_test as FTST  # noqa: E402  metrics, fmt, dm — тот же код, что у открытия
import final_test_r2 as FTR  # noqa: E402  PERIODY, zakrytie, KEY (импорт; скрипт открытия не запускается)

REPO = HERE.parent.parent
LOCKED = FTR.LOCKED
TABLES = REPO / 'docs/tables'
SPEC = REPO / 'models/designs/r2/r2_A_sborka.yaml'
KEY = FTR.KEY
HS = (1, 5, 22)
OUT = 'table_3_ref_ng_test_r2'
# имя в файле -> подпись (как в табл. 3.4 первого теста, tables_dlya_teksta.SRAV)
REF = {'GRU': 'GRU', 'LSTM': 'LSTM', 'GARCH': 'GARCH(1,1) с калибровкой', 'GJR': 'GJR-GARCH с калибровкой'}
REF_SHA = {  # docs/otchety/2026-10-26_ref_nn_garch.md, «Хэши SHA-256 (итог, 08:15)»
    ('LSTM', 1): 'ae726e37ec98ad2547bc9d4f67f98dc69eff06ad6e484103d864614a9bb57e0d', ('LSTM', 5): 'f349ad30f6e0f56b7e71b427d687db9e05cbf2d4d65939c03aa5a58a14a9ec6e',
    ('LSTM', 22): '0a4e35a5da578519f0bb3d3ffbbaa03da2dd9423c56d7f6efeb5b46afbbaff3c', ('GRU', 1): '5f5443d7f4f05e21874118a8f6f3bc6a2f4a3be08442d23a9a696fab0328d2a3',
    ('GRU', 5): '02bf4ff2badcad5ef01d8b5198910f2c116371b38e00ac86c340e1172c293370', ('GRU', 22): 'a330b1183a95fb4dc7bb3c59fa76eb5307099804248dbf852ab8dfe0bf576924',
    ('GJR', 1): '714678801fd872bf8563295af10f24562d0639b0bf142ff21083ef3a6c3fe59f', ('GJR', 5): 'f2a486533ab7d48d48b242c3c62ccb005094d707bbb2b054a5fc7638efa3ae8b',
    ('GJR', 22): 'b435823e167ddfc6d16bb3b81890b3b321ece952f9b7096563c8d2c0f78588be', ('GARCH', 1): 'db4e7b93ac65227a99a8553535690cce8b7e173f05320297fa2cef52f9d56d9d',
    ('GARCH', 5): 'cb039e04ea69a5b01ee14ab3526ed400a9053b4baa89a8ec35e21b2df9be4c00', ('GARCH', 22): 'e5a8691ec20a507226b2754733174ecc71598646664a9a5d4fe5ea5972044831'}
# столбец кадра -> подпись (как в table_3_periody_test_r2*.csv открытия)
BASE = {'rv_pred': 'Гибридная', 'har_raw': 'HAR-J', 'har_corr': 'HAR-J с поправкой'}
ETAL = {'(а)': 'har_raw', '(б)': 'har_corr'}
ETAL_NAME = {'(а)': 'HAR-J как в диссертации (а) d0', '(б)': 'HAR-J с поправкой (б) d0 — эталон В2'}
VERS = {'bez': ('', 'без правила (основная)', 'test_r2.csv', 'bez_zakrytiya'),
        'pravilo': ('_zakrytie', 'с правилом закрытия биржи (рядом)', 'test_r2_zakrytie.csv', 's_zakrytiem')}
SPER = {'kontrol': '2022–2026 (основной)', 'vse': '2017–2026 (справочно)'}  # периоды istochnik_sravnenie_test.csv первого теста
NOTE = ('справочно, не H1 (FINAL_DESIGN_round2, раздел 1; решение владельца 21:40): GRU, LSTM, GARCH(1,1), GJR-GARCH — test_locked/ref_nn_garch, '
        'по ВЫР-2 (PR #154) день d видят, не перестраивались; эталоны (а), (б) и финал — сборка второго открытия (R2/final, s² t−1 с эмбарго); '
        'нового открытия нет')


class Stop(Exception):
    pass


def now():
    return subprocess.run(['date', '+%Y-%m-%d %H:%M:%S'], capture_output=True, text=True).stdout.strip()


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def opening_tables():
    """SHA-256 таблиц открытия (docs/tables/*test_r2*, кроме собственного выхода)."""
    return {p.name: sha(p) for p in sorted(TABLES.glob('*test_r2*')) if not p.name.startswith(OUT)}


def inputs():
    """Пути входов из test_locked и их SHA-256 (журнал записи и sobrano). Сверка — до чтения данных."""
    sob = yaml.safe_load(SPEC.read_text())['sobrano']
    want = {}
    for h in HS:
        for m in REF:
            want[f'ref_nn_garch/REF_{m}_h{h}_test.parquet'] = REF_SHA[(m, h)]
        for r in (f'R2/final/etalon_d0_s2emb_h{h}_test.parquet', f'R2/final/final_d0_s2emb_h{h}_test.parquet'):
            want[r] = sob[r]
    bad = [r for r, s in want.items() if sha(LOCKED / r) != s]
    if bad:
        raise Stop('SHA-256 входов не совпал: ' + ', '.join(bad))
    return want


def frame(h):
    """Кадр h: ключи эталонов сборки + финал + справочные; цель — из справочных файлов."""
    et = pd.read_parquet(LOCKED / f'R2/final/etalon_d0_s2emb_h{h}_test.parquet', columns=KEY + ['year', 'har_raw', 'har_corr'])
    fd = pd.read_parquet(LOCKED / f'R2/final/final_d0_s2emb_h{h}_test.parquet', columns=KEY + ['rv_pred'])
    for d in (et, fd):
        d['date'] = pd.to_datetime(d['date'])
    m = et.merge(fd, on=KEY, how='left', validate='1:1')
    y = None
    for nm in REF:
        r = pd.read_parquet(LOCKED / f'ref_nn_garch/REF_{nm}_h{h}_test.parquet', columns=['date', 'ticker', 'rv_actual', 'rv_pred'])
        r['date'] = pd.to_datetime(r['date'])
        if r.duplicated(['date', 'ticker']).any():
            raise Stop(f'{nm} h={h}: дубли (date, ticker)')
        r = r.rename(columns={'rv_actual': f'y_{nm}', 'rv_pred': f'ref_{nm}'})
        m = m.merge(r, on=['date', 'ticker'], how='left', validate='1:1')
        m[f'ref_{nm}'] = m[f'ref_{nm}'].where(m[f'ref_{nm}'] > 0)
        y = m[f'y_{nm}'] if y is None else y.combine_first(m[f'y_{nm}'])
    for nm in REF:
        ok = m[f'y_{nm}'].notna() & y.notna()
        if not np.allclose(m[f'y_{nm}'][ok], y[ok], rtol=1e-12, atol=0):
            raise Stop(f'h={h}: цель {nm} не совпала с целью других справочных')
    m['rv_actual'] = y
    if m[['har_raw', 'har_corr', 'rv_pred']].isna().any().any():
        raise Stop(f'h={h}: в кадре есть строки без эталона или финала')
    m = m.drop(columns=[f'y_{nm}' for nm in REF])
    m = m[m.rv_actual > 1e-12]  # как final_test_r2.frame_d0
    return m.sort_values(['date', 'ticker']).reset_index(drop=True)


def check_zakrytie(frames):
    """Правило закрытия (final_test_r2.zakrytie) = исключённые точки открытия: по (h, тикер) число, первая и последняя t."""
    X = pd.read_csv(TABLES / 'table_3_zakrytie_isklyucheno_test_r2.csv', sep=';', encoding='utf-8-sig')
    ex, bad = {}, []
    for h, df in frames.items():
        ex[h] = FTR.zakrytie(df, h)
        g = df[ex[h]].groupby('ticker').date.agg(['size', 'min', 'max'])
        x = X[(X.h == h) & (X['Тикер'] != 'ВСЕГО')].set_index('Тикер')
        if set(g.index) != set(x.index) or int(X[(X.h == h) & (X['Тикер'] == 'ВСЕГО')]['исключено точек'].iloc[0]) != int(ex[h].sum()):
            bad.append(f'h={h}: тикеры исключения не совпали')
            continue
        for tk, r in g.iterrows():
            if (r['size'], str(r['min'].date()), str(r['max'].date())) != (x.loc[tk, 'исключено точек'], x.loc[tk, 'первая исключённая t'],
                                                                           x.loc[tk, 'последняя исключённая t']):
                bad.append(f'h={h} {tk}: исключение не совпало с открытием')
    if bad:
        raise Stop('; '.join(bad))
    return ex


def metric_rows(fr, ver):
    rows = []
    for h, df in fr.items():
        for per, (a, b) in FTR.PERIODY.items():
            g0 = df[df.year.between(a, b)]
            for col, lab in {**BASE, **{f'ref_{k}': v for k, v in REF.items()}}.items():
                g = g0[g0[col].notna()]
                mt = FTST.metrics(g.rv_actual.values, g[col].values)
                qe = {e: FTST.metrics(g.rv_actual.values, g[c].values)['QLIKE'] for e, c in ETAL.items()}  # эталон на тех же строках
                rows.append(dict(h=h, Период=per, Модель=lab, n=len(g), **{'покрытие, %': round(100 * len(g) / len(g0), 1)},
                                 **{k: FTST.fmt(v, k) for k, v in mt.items()},
                                 **{f'к {e}, %': round((mt['QLIKE'] / qe[e] - 1) * 100, 2) for e in ETAL},
                                 строка='справочная (не H1)' if col.startswith('ref_') else 'открытие (контроль)',
                                 правило=VERS[ver][1], qlike_full=mt['QLIKE']))
    return pd.DataFrame(rows)


def dm_rows(fr, ver):
    rows = []
    for h, df in fr.items():
        for per, (a, b) in FTR.PERIODY.items():
            g0 = df[df.year.between(a, b)]
            for nm, lab in REF.items():
                g = g0[g0[f'ref_{nm}'].notna()].assign(rv_pred=lambda x: x[f'ref_{nm}']).sort_values(['date', 'ticker']).reset_index(drop=True)
                for e, c in ETAL.items():
                    for r in FTST.dm(g, c, h):
                        rows.append(dict(h=h, Период=per, Модель=lab, эталон=ETAL_NAME[e], **r, правило=VERS[ver][1],
                                         основной=r['способ'].startswith('основной')))
    D = pd.DataFrame(rows)
    return D.assign(t=D.t.round(2), p_two=D.p_two.map(FTST.pfmt), p_one_H1=D.p_one_H1.map(FTST.pfmt)).rename(
        columns={'p_one_H1': 'p_one (модель лучше эталона)'})


def check_opening(M, ver):
    """Сверка (1): строки открытия в формате final_test.fmt совпадают строково."""
    P = pd.read_csv(TABLES / f'table_3_periody_{VERS[ver][2]}', dtype=str)
    bad = []
    for _, r in M[M.строка == 'открытие (контроль)'].iterrows():
        x = P[(P.h == str(r.h)) & (P['Период'] == r['Период']) & (P['Модель'] == r['Модель'])]
        if len(x) != 1:
            bad.append(f'{ver} h={r.h} {r["Период"]} {r["Модель"]}: нет строки в открытии')
            continue
        for c in ('n', 'QLIKE', 'RMSE', 'MAE'):
            if str(r[c]) != x[c].iloc[0]:
                bad.append(f'{ver} h={r.h} {r["Период"]} {r["Модель"]} {c}: {r[c]} != {x[c].iloc[0]}')
    return bad


def check_dm_hybrid(fr, ver):
    """Сверка (4): DM финала тремя способами (тот же final_test.dm) = table_3_dm5_test_r2{,_zakrytie}.csv открытия, t до 2 знаков."""
    D5 = pd.read_csv(TABLES / f'table_3_dm5_{VERS[ver][2]}')
    per5 = {'2022–2026 (контрольный)': (2022, 2026), '2017–2026': (2017, 2026)}
    bad, k = [], 0
    for h, df in fr.items():
        for per, (a, b) in per5.items():
            g = df[df.year.between(a, b)].sort_values(['date', 'ticker']).reset_index(drop=True)
            for e, c in ETAL.items():
                with np.errstate(invalid='ignore', divide='ignore'):
                    mine = FTST.dm(g, c, h)
                for r in mine:
                    x = D5[(D5.h == h) & (D5['Период'] == per) & (D5['эталон'] == f'HAR-J {e}') & (D5['способ'] == r['способ'])]
                    t = round(r['t'], 2)
                    k += 1
                    if len(x) != 1 or x.n.iloc[0] != r['n'] or not ((np.isnan(t) and np.isnan(x.t.iloc[0])) or abs(t - x.t.iloc[0]) < 1e-9):
                        bad.append(f'{ver} h={h} {per} {e} {r["способ"]}: DM финала не совпал с открытием '
                                   f'(n {r["n"]} / {x.n.tolist()}, t {t} / {x.t.tolist()})')
    return bad, k


def check_first(M, ver):
    """Сверка (3): QLIKE пул справочных = первый тест (istochnik_sravnenie_test.csv), те же ряды."""
    S = pd.read_csv(TABLES / 'dlya_teksta/istochnik_sravnenie_test.csv', sep=';', decimal=',', encoding='utf-8-sig')
    bad, k = [], 0
    for tag, per in SPER.items():
        for lab in REF.values():
            for h in HS:
                a = S[(S['Период'] == tag) & (S['правило'] == VERS[ver][3]) & (S.h == h) & (S['Модель'] == lab)]
                b = M[(M['Период'] == per) & (M.h == h) & (M['Модель'] == lab)]
                if len(a) != 1 or len(b) != 1:
                    bad.append(f'{ver} {tag} h={h} {lab}: нет строки')
                    continue
                k += 1
                if int(a['строк с прогнозом'].iloc[0]) != b.n.iloc[0] or abs(a['QLIKE пул'].iloc[0] - b.qlike_full.iloc[0]) > 1e-12:
                    bad.append(f'{ver} {tag} h={h} {lab}: {b.qlike_full.iloc[0]} != {a["QLIKE пул"].iloc[0]} (n {b.n.iloc[0]} / {a["строк с прогнозом"].iloc[0]})')
    return bad, k


def main():
    t0 = now()
    before = opening_tables()
    print(f'{t0}: таблиц открытия *test_r2*: {len(before)}', flush=True)
    want = inputs()
    print(f'SHA-256 входов из test_locked: {len(want)} файлов, совпали', flush=True)
    frames = {h: frame(h) for h in HS}
    ex = check_zakrytie(frames)
    print('правило закрытия биржи = исключённые точки открытия: ' + ', '.join(f'h={h}: {int(e.sum())}' for h, e in ex.items()), flush=True)
    res, bad, nk, nd = {}, [], 0, 0
    for ver in VERS:
        fr = frames if ver == 'bez' else {h: df[~ex[h]].reset_index(drop=True) for h, df in frames.items()}
        M, D = metric_rows(fr, ver), dm_rows(fr, ver)
        bad += check_opening(M, ver)
        b3, k = check_first(M, ver)
        bad += b3; nk += k
        b4, k4 = check_dm_hybrid(fr, ver)
        bad += b4; nd += k4
        res[ver] = (M, D)
    if bad:
        raise Stop('сверки не прошли:\n' + '\n'.join(bad[:20]))
    print(f'сверка (1) со строками открытия, (3) с первым тестом ({nk} строк справочных), (4) DM финала с dm5 открытия ({nd} строк): OK', flush=True)
    for ver, (M, D) in res.items():
        suf = VERS[ver][0]
        M.drop(columns='qlike_full').assign(пометка=NOTE).to_csv(TABLES / f'{OUT}{suf}.csv', index=False)
        D.assign(пометка=NOTE + '; d = QLIKE эталона − QLIKE модели, t > 0 — модель лучше').to_csv(TABLES / f'{OUT}_dm{suf}.csv', index=False)
    after = opening_tables()
    if after != before:
        raise Stop('таблицы открытия изменились: ' + ', '.join(k for k in set(before) | set(after) if before.get(k) != after.get(k)))
    print(f'{now()}: записано {OUT}{{,_zakrytie,_dm,_dm_zakrytie}}.csv; таблицы открытия ({len(after)}) не изменились', flush=True)


if __name__ == '__main__':
    try:
        main()
    except Stop as e:
        print('СТОП:', e, flush=True)
        sys.exit(3)
