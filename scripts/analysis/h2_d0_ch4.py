#!/usr/bin/env python3
"""H2 главы 4 на d0 (решение владельца 26.10 21:50; основная точка C — C1 W504 p70 h=5, владелец ~21:55).

Пересчитываются ТОЛЬКО строки H2 (C3): C u02 h=1 (Cref) и основная точка C (C1 W504 p70 h=5),
гибрид − HAR-J (б) и гибрид − наивная. Остальной тест главы 4 не трогается: таблицы docs/tables/strat_test/,
кэш и журнал теста (data/fastbt/s9/cache_latest_final3_test.txt и т. п.) не пишутся; ch4_test_run.sh,
`--scope test` CLI не запускаются.

Код теста 10:32 импортируется, не переписывается:
  позиции A и C u02 по тикерам   — fastbt.s9._job (как fastbt.s9 cache, scope test, окна 2018–2021);
  движок МСП, C1-затворы         — fastbt.s9.Engine, S9.c_points, Engine.g_row / port;
  Δ Шарпа, p ЛВ (блок 5), p IID  — fastbt.s9.S9.run_c3 (lw_sharpe_diff, boot_delta), 10 000 повторов;
  метрики (Шарп, оборот, экспозиция) — fastbt.s9.metrics; периоды — s9.T2_CUT / T2_LABELS (2022–2026 / 2022–2024 / 2025–01.2026).
σ_pred — кодом главы 3 (scripts/analysis/sigma_pred_h1_final3_full.py): σ = √ряда, float32, сортировка (date, ticker).

Входы d0 (открытие R2 21:36, SHA — r2_A_sborka.yaml: sobrano):
  гибрид   — √rv_pred  test_locked/R2/final/final_d0_s2emb_h{1,5}_test.parquet;
  HAR-J (б) — √har_corr test_locked/R2/final/etalon_d0_s2emb_h{1,5}_test.parquet;
  наивная  — √(среднее RV дней d−4…d) из features_pit_v3.rv (прежняя: √rv_w = RV дней d−5…d−1); одна и та же для h=1 и h=5.

Стадии:
  python3 scripts/analysis/h2_d0_ch4.py proverka      # ряды d0 + проверка информационного множества (сдвиг, детекторы); метрик нет
  python3 scripts/analysis/h2_d0_ch4.py samoproverka  # старые входы ($HARJ, старая наивная): повтор чисел теста 10:32 до 1e-9
  python3 scripts/analysis/h2_d0_ch4.py schet         # новые входы d0: строки H2 — один запуск (отказ без пройденных proverka и samoproverka)
Выход — docs/tables/table_4_h2_d0*.csv («;», десятичная запятая, UTF-8 BOM, полная точность); ряды — data/fastbt/s9/h2d0/.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
MAIN = Path('/Users/nilysenok/Desktop/Старое/PhD_Vol_trading')
REGEN = MAIN / 'data' / 'regen'
R2F = REGEN / 'test_locked' / 'R2' / 'final'
S9DIR = MAIN / 'data' / 'fastbt' / 's9'
OUT = S9DIR / 'h2d0'
DET = OUT / 'detektor'
TABLES = ROOT / 'docs' / 'tables'
T_STRAT = TABLES / 'strat_test'
REPS = 10000
EPS_SAMO = 1e-9
SOBRANO = {  # models/designs/r2/r2_A_sborka.yaml: sobrano (сборка 21:35)
    'final_d0_s2emb_h1_test.parquet': '5ebcf61a1ccfd0f6c660e03431e306f8fe7719f011a66b469d8431be147c92e4',
    'final_d0_s2emb_h5_test.parquet': '4c11df7660cb174e9d169abdc67d8fd16286c21aa052351fa36b49e9b63eb887',
    'etalon_d0_s2emb_h1_test.parquet': '1dce250bc913e93a7643094261eda5695288e10f72eeac3d0ae1d7d97f8665fc',
    'etalon_d0_s2emb_h5_test.parquet': 'de559a9ed66c892842b049964003509ab338bfbac87768dd5d63bf406041b51d',
}
OLD = {  # входы теста 10:32 (fastbt.s9 SOURCES в scope test, ch4_test_run.sh)
    'hyb_h1': (REGEN / 'sigma_pred_h1_final3_full.parquet', 'sigma_pred_h1', 'h1'),
    'hyb_h5': (REGEN / 'sigma_pred_h5_final2.parquet', 'sigma_pred_h5', 'h5'),
    'harj_h1': (REGEN / 'sigma_pred_h1_harj_final2.parquet', 'sigma_pred_h1', 'h1'),
    'naive_h1': (S9DIR / 'sigma_naive_rvw_test.parquet', 'sigma_pred_h1', 'h1'),
}
TEST_CACHE = S9DIR / 'cache_682e4ce9eb.pkl'      # = cache_latest_final3_test.txt (тест 10:33)
NEW = {
    'hyb_h1': (OUT / 'sigma_hyb_d0_h1.parquet', 'sigma_pred_h1', 'h1'),
    'harj_h1': (OUT / 'sigma_harj_d0_h1.parquet', 'sigma_pred_h1', 'h1'),
    'naive_h1': (OUT / 'sigma_naive_d0.parquet', 'sigma_pred_h1', 'h1'),
    'hyb_h5': (OUT / 'sigma_hyb_d0_h5.parquet', 'sigma_pred_h5', 'h5'),
    'harj_h5': (OUT / 'sigma_harj_d0_h5.parquet', 'sigma_pred_h5', 'h5'),
    'naive_h5': (OUT / 'sigma_naive_d0.parquet', 'sigma_pred_h1', 'h5'),   # наивная от h не зависит
}
MAIN_PT = 'W504_p70'                       # основная точка C (TEST_C = C1:W504_p70 на h=5)
CUTS_TEST = ('2023-06-30', '2024-12-31', '2025-06-30')
CUTS_NAIVE = ('2018-06-29', '2020-03-02', '2021-06-30', '2017-12-31', '2019-12-31', '2020-12-31') + CUTS_TEST
SEED = 20261026


def sha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def now() -> str:
    return time.strftime('%F %T')


def to_csv(df: pd.DataFrame, name: str) -> Path:
    p = TABLES / name
    if not p.name.startswith('table_4_h2_d0'):
        raise AssertionError(f'запись только в table_4_h2_d0*: {p}')
    df.to_csv(p, sep=';', decimal=',', index=False, encoding='utf-8-sig')
    print(f'→ {p.relative_to(ROOT)} ({len(df)} строк)', flush=True)
    return p


def s9_test():
    """fastbt.s9 в режиме теста главы 4 (замок FINAL_DESIGN_strategies.md, окна 2018–2021, годы 2022–2026), σ h=1 final3_full."""
    os.environ.setdefault('NUMBA_CACHE_DIR', str(MAIN / 'data' / 'fastbt' / 'numba_cache'))
    from fastbt import s9
    s9.set_scope('test', 2018)
    s9.use_sigma_h1('final3_full')
    # A — как в тесте: σ_pred h=1 vpred = final3_full (fastbt.s9.build_cache ставит FASTBT_VPRED = SOURCES['hyb_h1'])
    os.environ['FASTBT_VPRED'] = str(s9.SIGMA_H1['final3_full'])
    return s9


# ======================= ряды σ =======================

def sigma_ch3(df, col, out_col):
    """Код главы 3 (sigma_pred_h1_final3_full.py, фаза build final3.py): σ = √ряда, float32, сортировка (date, ticker)."""
    sp = df.assign(**{out_col: np.sqrt(df[col].values).astype('float32')}).sort_values(['date', 'ticker'])
    return sp[['date', 'ticker', out_col]].reset_index(drop=True)


def rv_frame():
    f = pd.read_parquet(REGEN / 'features_pit_v3.parquet', columns=['date', 'ticker', 'rv', 'rv_w'])
    f['date'] = pd.to_datetime(f['date'])
    return f.sort_values(['ticker', 'date']).reset_index(drop=True)


def naive_from_rv(f, cut=pd.Timestamp('2026-02-03')):
    """Наивная d0: σ_d = √(среднее RV дней d−4…d) тикера (rolling 5 по строкам тикера, только прошлое и день d)."""
    m5 = f.groupby('ticker')['rv'].transform(lambda s: s.rolling(5, min_periods=5).mean())
    nv = f[['date', 'ticker']].assign(sigma_pred_h1=np.sqrt(m5.astype(float)).astype('float32'))
    nv = nv[(nv['date'] >= '2017-01-03') & (nv['date'] <= cut)].dropna()
    return nv.sort_values(['date', 'ticker']).reset_index(drop=True)


def build_series():
    OUT.mkdir(parents=True, exist_ok=True)
    for f, want in SOBRANO.items():
        got = sha(R2F / f)
        if got != want:
            raise SystemExit(f'СТОП: SHA {f} {got[:16]}… ≠ sobrano {want[:16]}…')
    info = {}
    for h in (1, 5):
        fin = pd.read_parquet(R2F / f'final_d0_s2emb_h{h}_test.parquet', columns=['date', 'ticker', 'rv_pred'])
        eta = pd.read_parquet(R2F / f'etalon_d0_s2emb_h{h}_test.parquet', columns=['date', 'ticker', 'har_corr'])
        for lab, df, col in ((f'hyb_h{h}', fin, 'rv_pred'), (f'harj_h{h}', eta, 'har_corr')):
            p, oc, _ = NEW[lab]
            s = sigma_ch3(df, col, oc)
            s.to_parquet(p, index=False)
            info[lab] = dict(file=str(p), sha256=sha(p), rows=len(s), nan=int(s[oc].isna().sum()))
    nv = naive_from_rv(rv_frame())
    nv.to_parquet(NEW['naive_h1'][0], index=False)
    info['naive'] = dict(file=str(NEW['naive_h1'][0]), sha256=sha(NEW['naive_h1'][0]), rows=len(nv))
    (OUT / 'ryady_sha256.json').write_text(json.dumps(dict(time=now(), inputs={k: v for k, v in SOBRANO.items()},
                                                          features_pit_v3=sha(REGEN / 'features_pit_v3.parquet'),
                                                          series=info), ensure_ascii=False, indent=1))
    return info


# ======================= проверка сдвига =======================

def shift_rows():
    """Сдвиг (п. 2.6 R2): r2_loyo.sdvig — Спирмен ln σ²_d с ln RV_{d+s}, s = −2…+1, по тикерам, медиана, только 2017–2021,
    решение r2_loyo.sdvig_ok с K = {0} и симметричным ε = 0,01. Рядом — профиль уровня и приращения r2_features.shift_table
    (для оконных уровней решает приращение, как в r2_loyo.reshit_F)."""
    import r2_loyo as O
    import r2_features as RF
    rv = O.rv_daily()
    rv['date'] = pd.to_datetime(rv['date'])
    eps = float(O.CFG['sdvig']['eps'])
    assert abs(eps - 0.01) < 1e-12
    f = rv_frame()
    plant = naive_from_rv(f.assign(rv=f.groupby('ticker')['rv'].shift(-1)))   # контроль: подложенная утечка d+1
    # тип ряда: model — прогноз модели, решает уровень (как r2_core.shift_check); okno — оконное среднее RV (наивная),
    # уровень окна максимален в середине окна, решает приращение (r2_O_loyo.yaml sdvig: «оконные уровни — по приращению, как у F»)
    series = {}
    for lab in ('hyb_h1', 'harj_h1', 'naive_h1', 'hyb_h5', 'harj_h5'):
        p, c, _ = NEW[lab]
        series[f'd0:{lab}'] = (pd.read_parquet(p), c, 'новый вход', 'okno' if 'naive' in lab else 'model')
    for lab in ('hyb_h1', 'harj_h1', 'naive_h1', 'hyb_h5'):
        p, c, _ = OLD[lab]
        series[f'старый:{lab}'] = (pd.read_parquet(p), c, 'контроль: вход теста 10:32', 'okno' if 'naive' in lab else 'model')
    series['контроль:naive_utechka_d+1'] = (plant, 'sigma_pred_h1', 'контроль: подложенная утечка (RV дня d+1)', 'okno')
    for lab in ('hyb_h1', 'harj_h1'):
        p, c, _ = NEW[lab]
        series[f'контроль:{lab}_utechka_d+1'] = (plant_next(pd.read_parquet(p), c), c,
                                                 'контроль: подложенная утечка (σ_d := σ_{d+1} ряда d0)', 'model')
    rows = []
    for name, (df, c, rol, tip) in series.items():
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])
        pr = df[['date', 'ticker']].assign(rv_pred=df[c].astype(float) ** 2)
        s = O.sdvig(pr, rv, K=(0,))
        F = df[df['date'] <= RF.LAST_DEV][['date', 'ticker']].assign(ln_sigma2=np.log(df.loc[df['date'] <= RF.LAST_DEV, c]
                                                                                      .astype(float) ** 2))
        prof = RF.shift_table(F, {'B': rv}, name).iloc[0]
        lev_ok = s['ok']
        dam = int(prof['d_argmax'])
        if tip == 'model':
            ok, why = lev_ok, ('по уровню' + (f'; {s["pometka"]}' if s['pometka'] else '')) if lev_ok \
                else f'уровень: {s["prichina"]}'
        elif dam > 0:
            ok, why = False, f'заглядывание (+1) по приращению (уровень — окно); уровень argmax {s["argmax"]:+d}'
        elif O.sdvig_ok(dam, (0,))[0]:
            ok, why = True, (f'по приращению (уровень — окно): argmax приращения 0; уровень argmax {s["argmax"]:+d}'
                             + ('' if lev_ok else f' ({s["prichina"]})'))
        else:
            ok, why = False, f'отстаёт (по приращению, уровень — окно): argmax приращения {dam:+d}; уровень argmax {s["argmax"]:+d}'
        rows.append(dict(ryad=name, rol=rol, tip=tip, n_tikerov=s['n_tikerov'], n_strok=s['n_strok'],
                         **{f'corr_s{k:+d}': s[f'corr_s{k:+d}'] for k in (-2, -1, 0, 1)}, argmax=s['argmax'],
                         uroven_ok=lev_ok, **{f'rho_d_s{k:+d}': prof[f'drho_{k}'] for k in (-2, -1, 0, 1)},
                         d_argmax=dam, ok=ok, reshenie=why, eps=eps, vremya=now()))
    return pd.DataFrame(rows)


# ======================= детекторы «будущее не влияет» =======================

def perturb(df, col, cut, seed):
    rng = np.random.default_rng(seed)
    d = df.copy()
    fut = (pd.to_datetime(d['date']) > pd.Timestamp(cut)).values
    d.loc[fut, col] = (d.loc[fut, col].values * rng.uniform(0.5, 2.0, int(fut.sum()))).astype(d[col].dtype)
    return d


def plant_next(df, col):
    d = df.sort_values(['ticker', 'date']).copy()
    d[col] = d.groupby('ticker')[col].shift(-1).fillna(d[col])
    return d.sort_values(['date', 'ticker']).reset_index(drop=True)


def detector_naive():
    """Детектор ряда наивной: RV после среза × U(0,5; 2) — строки ≤ среза не меняются, после — меняются;
    подложенная утечка (RV_d := RV_{d+1}) — видна на строках ≤ среза."""
    f = rv_frame()
    base = naive_from_rv(f)
    fp = f.assign(rv=f.groupby('ticker')['rv'].shift(-1).fillna(f['rv']))
    planted = naive_from_rv(fp)
    rows = []
    for i, cut in enumerate(CUTS_NAIVE):
        c = pd.Timestamp(cut)
        rng = np.random.default_rng(SEED + i)
        fut = (f['date'] > c).values
        f2 = f.copy()
        f2.loc[fut, 'rv'] = f2.loc[fut, 'rv'].values * rng.uniform(0.5, 2.0, int(fut.sum()))
        pert = naive_from_rv(f2)
        fp2 = f2.assign(rv=f2.groupby('ticker')['rv'].shift(-1).fillna(f2['rv']))
        plantp = naive_from_rv(fp2)

        def dd(a, b):
            m = a.merge(b, on=['date', 'ticker'], suffixes=('', '_p'))
            x = np.abs(np.log(m['sigma_pred_h1'].astype(float)) - np.log(m['sigma_pred_h1_p'].astype(float)))
            pre = (m['date'] <= c).values
            return float(x[pre].max()), float(x[~pre].max()), int(pre.sum())
        pre, post, n = dd(base, pert)
        ppre, _, _ = dd(planted, plantp)
        rows.append(dict(detektor='ряд наивной d0 (пересборка из RV)', ryad='naive_d0', srez=cut, strok_do=n,
                         max_dln_do=pre, max_dln_posle=post, podlog_max_dln_do=ppre,
                         ok=bool(pre == 0.0 and post > 0 and ppre > 0)))
    return rows


def structure_rows():
    """Устройство рядов открытия: ряд = сырой × F_sm, F_sm — одна константа на шаг (год), т. е. s² не несёт данных внутри года."""
    rows = []
    for h in (1, 5):
        for fn, raw, cor in ((f'final_d0_s2emb_h{h}_test.parquet', 'leg_raw', 'rv_pred'),
                             (f'etalon_d0_s2emb_h{h}_test.parquet', 'har_raw', 'har_corr')):
            d = pd.read_parquet(R2F / fn, columns=['date', 'ticker', 'step', raw, cor, 'F_sm'])
            rel = float(np.max(np.abs(d[cor] - d[raw] * d['F_sm']) / d[cor]))
            nf = d.groupby('step')['F_sm'].nunique()
            rows.append(dict(detektor='устройство ряда: ряд = сырой × F_sm', ryad=f'{fn}:{cor}', srez='—',
                             strok_do=len(d), max_dln_do=rel, max_dln_posle=np.nan,
                             podlog_max_dln_do=float(nf.max()),
                             ok=bool(rel < 1e-12 and nf.max() == 1),
                             primechanie=f'max отн. |{cor} − {raw}·F_sm| = {rel:.2e}; число F_sm на шаг ≤ {int(nf.max())}'))
    return rows


def build_cache(s9, sources: dict, workers: int = 10) -> dict:
    """Позиции A и C u02 по тикерам — fastbt.s9._job (как fastbt.s9 cache в тесте), без записи кэша теста."""
    from fastbt.data import TICKERS
    import multiprocessing as mp
    saved = dict(s9.SOURCES)
    s9.SOURCES.clear()
    s9.SOURCES.update(sources)
    try:
        srcs = {lab: (sources[lab][2], s9.load_source(lab)) for lab in sources}
    finally:
        s9.SOURCES.clear()
        s9.SOURCES.update(saved)
    jobs = [(t, {lab: (h, df[df['ticker'] == t]) for lab, (h, df) in srcs.items()}, (s9.SCOPE, s9.VAL_START))
            for t in TICKERS]
    os.environ['NUMBA_NUM_THREADS'] = '1'
    t0 = time.time()
    res = {}
    with mp.get_context('spawn').Pool(workers) as pool:
        for tk, out in pool.imap_unordered(s9._job, jobs):
            res[tk] = out
    print(f'позиции: {len(res)} тикеров, меток σ {len(sources)}, {time.time() - t0:.0f} с', flush=True)
    return res


def engine(s9, cache, sources, reps=REPS):
    s9.SOURCES.clear()
    s9.SOURCES.update(sources)
    e = s9.Engine(cache)
    return e, s9.S9(e, reps)


def main_point(S, src):
    """Основная точка C: C1 W504 p70 на σ src — S9.c_points, позиция A·затвор, портфель Engine.port (как S9.run_c)."""
    pts1, _ = S.c_points(src)
    G = dict(pts1)[MAIN_PT]
    g = S.e.g_row(G)
    return S.e.port(S.e.A_row * g), g


def detector_ch4(s9):
    """Детектор на входе главы 4: σ ряда после среза × U(0,5; 2) — позиции C u02 и затвор основной точки на барах ≤ среза
    не меняются, после — меняются; подложенная утечка σ_d := σ_{d+1} видна на барах ≤ среза."""
    DET.mkdir(parents=True, exist_ok=True)
    src = {}
    base = {}
    for lab in ('hyb_h1', 'harj_h1', 'naive_h1', 'hyb_h5', 'harj_h5'):
        p, c, h = NEW[lab]
        d = pd.read_parquet(p)
        base[lab] = (d, c, h)
        variants = {'b': d, 'pl': plant_next(d, c)}
        for i, cut in enumerate(CUTS_TEST):
            variants[f'p{i}'] = perturb(d, c, cut, SEED + i)
            variants[f'plp{i}'] = plant_next(variants[f'p{i}'], c)
        for k, v in variants.items():
            f = DET / f'{lab}__{k}.parquet'
            v.to_parquet(f, index=False)
            src[f'{lab}__{k}'] = (f, c, h)
    cache = build_cache(s9, src)
    e, S = engine(s9, cache, src, reps=10)
    rdates = e.dates[e.r['date']]
    rows = []
    for lab in base:
        gates = {k: main_point(S, f'{lab}__{k}')[1] for k in ['b', 'pl'] + [f'p{i}' for i in range(3)]
                 + [f'plp{i}' for i in range(3)]}
        for i, cut in enumerate(CUTS_TEST):
            pre = (rdates <= pd.Timestamp(cut))
            for what, getv in (('C u02 (позиции)', lambda k: S.C_rows[f'{lab}__{k}']),
                               (f'C1 {MAIN_PT} (затвор)', lambda k: gates[k])):
                b, p, pl, plp = getv('b'), getv(f'p{i}'), getv('pl'), getv(f'plp{i}')
                dpre = float(np.max(np.abs(b[pre] - p[pre])))
                dpost = float(np.max(np.abs(b[~pre] - p[~pre])))
                dpl = float(np.max(np.abs(pl[pre] - plp[pre])))
                rows.append(dict(detektor=f'вход главы 4: {what}', ryad=f'd0:{lab}', srez=cut, strok_do=int(pre.sum()),
                                 max_dln_do=dpre, max_dln_posle=dpost, podlog_max_dln_do=dpl,
                                 ok=bool(dpre == 0.0 and dpost > 0 and dpl > 0),
                                 primechanie='подлог виден' if dpl > 0 else
                                 'подлог на одном баре среза позицию не сдвинул (мощность, не утечка)'))
    return rows


# ======================= H2 =======================

def h2_ports(S, h1=('hyb_h1', 'harj_h1', 'naive_h1'), h5=('hyb_h5', 'harj_h5', 'naive_h5')):
    """{(вид, метка run_c3): (портфель,)}: Cref — C u02 h=1 (S9.C), C1:W504_p70 — основная точка на σ h=5."""
    keys = ('hyb_h1', 'harj_h1', 'naive_h1')
    cp = {}
    for k, lab in zip(keys, h1):
        if lab is not None:
            cp[('Cref', k)] = (S.C[lab], S.C_rows[lab])
    for k, lab in zip(keys, h5):
        if lab is not None:
            p, g = main_point(S, lab)
            cp[(f'C1:{MAIN_PT}', k)] = (p, g)
    return cp


def by_period(s9, S, cp, versiya):
    cut = pd.Timestamp(s9.T2_CUT)
    out, met = [], []
    for per in s9.T2_LABELS:
        def sl(p):
            if per == s9.T2_LABELS[0]:
                return p
            return p.loc[p.index < cut] if per == s9.T2_LABELS[1] else p.loc[p.index >= cut]
        cps = {k: (sl(v[0]),) for k, v in cp.items()}
        d = S.run_c3(cps)
        d.insert(0, 'period', per)
        d.insert(0, 'versiya', versiya)
        d['days'] = [len(cps[(r, 'hyb_h1')][0]) for r in d['design']]
        out.append(d)
        for (kind, lab), (p,) in cps.items():
            m = s9.metrics(p)
            met.append(dict(versiya=versiya, period=per, design=kind, sigma=lab, days=len(p),
                            **{k: m[k] for k in ('sharpe', 'sortino', 'ann_excess_pct', 'mdd_excess_pct', 'exposure_pct',
                                                 'turnover_y', 'comm_pct_y')}))
    return pd.concat(out, ignore_index=True), pd.DataFrame(met)


def design_name(d):
    return {'Cref': 'C u02 h=1 (Cref)', f'C1:{MAIN_PT}': 'C1 W504 p70 h=5 (основная точка C)'}.get(d, d)


# ======================= стадии =======================

def stage_proverka():
    s9 = s9_test()
    info = build_series()
    print(json.dumps(info, ensure_ascii=False, indent=1), flush=True)
    sh = shift_rows()
    to_csv(sh, 'table_4_h2_d0_proverka_sdvig.csv')
    det = pd.DataFrame(detector_naive() + structure_rows() + detector_ch4(s9))
    to_csv(det, 'table_4_h2_d0_proverka_detektor.csv')
    new = sh[sh['ryad'].str.startswith('d0:')]
    # контроли обязаны блокироваться: прежние HAR-J и наивная (отставание d−1), подложенные утечки d+1
    kontr = sh['ryad'].isin(['старый:harj_h1', 'старый:naive_h1']) | sh['ryad'].str.startswith('контроль:')
    ctrl_ok = bool((~sh.loc[kontr, 'ok']).all())
    ok = bool(new['ok'].all() and det['ok'].all() and ctrl_ok)
    (OUT / 'proverka.json').write_text(json.dumps(dict(time=now(), ok=ok, sdvig_ok=bool(new['ok'].all()),
                                                       detektor_ok=bool(det['ok'].all()), kontroli_blok=bool(ctrl_ok),
                                                       series={k: v['sha256'] for k, v in info.items()}),
                                                  ensure_ascii=False, indent=1))
    print(f'ПРОВЕРКА ИНФОРМАЦИОННОГО МНОЖЕСТВА: {"OK" if ok else "НЕ ПРОЙДЕНА"} '
          f'(сдвиг {bool(new["ok"].all())}, детекторы {bool(det["ok"].all())}, контроли блокируются {ctrl_ok})')
    return 0 if ok else 1


def stage_samoproverka():
    s9 = s9_test()
    for lab, (p, _, _) in OLD.items():
        print(lab, p.name, sha(p)[:16], flush=True)
    fresh = build_cache(s9, OLD)
    test = pickle.loads(TEST_CACHE.read_bytes())
    dmax = 0.0
    for t in test:
        for key in ('A', 'A0'):
            for sid, ys in test[t][key].items():
                for Y, v in ys.items():
                    dmax = max(dmax, float(np.max(np.abs(np.asarray(v) - fresh[t][key][sid][Y]))))
        for lab in OLD:
            for sid, ys in test[t]['C'][lab].items():
                for Y, v in ys.items():
                    dmax = max(dmax, float(np.max(np.abs(np.asarray(v) - fresh[t]['C'][lab][sid][Y]))))
    print(f'позиции: свежий счёт против кэша теста 10:33, max|Δ| = {dmax:.3e}', flush=True)
    tabs, mets = {}, {}
    for nm, cache in (('свежий счёт', fresh), ('кэш теста 10:33', test)):
        e, S = engine(s9, cache, OLD)
        cp = h2_ports(S, h5=('hyb_h5', None, 'naive_h1'))
        d = S.run_c3({k: v for k, v in cp.items() if k[0] == 'Cref'})
        d.insert(0, 'versiya', nm)
        tabs[nm] = d
        mets[nm] = (S, cp)
    a, b = tabs['свежий счёт'], tabs['кэш теста 10:33']
    num = [c for c in a.columns if c not in ('versiya', 'design')]
    dd = float(np.max(np.abs(a[num].values.astype(float) - b[num].values.astype(float))))
    # таблица теста (4 знака): s9_C3_tochnost_v_rezultat_final3_test.csv, строка Cref; основная точка — s9_C_osnovnaya
    c3 = pd.read_csv(T_STRAT / 's9_C3_tochnost_v_rezultat_final3_test.csv', sep=';', decimal=',', encoding='utf-8-sig')
    c3 = c3[c3['design'] == 'Cref'].iloc[0]
    dt = float(np.max(np.abs(a[num].iloc[0].values.astype(float) - c3[num].values.astype(float))))
    osn = pd.read_csv(T_STRAT / 's9_C_osnovnaya_final3_test.csv', sep=';', decimal=',', encoding='utf-8-sig').iloc[0]
    S, cp = mets['свежий счёт']
    sh5 = s9.metrics(cp[(f'C1:{MAIN_PT}', 'hyb_h1')][0])['sharpe']
    S2, cp2 = mets['кэш теста 10:33']
    sh5t = s9.metrics(cp2[(f'C1:{MAIN_PT}', 'hyb_h1')][0])['sharpe']
    ok = bool(dmax <= EPS_SAMO and dd <= EPS_SAMO and dt <= 5e-5 + 1e-12 and abs(sh5 - sh5t) <= EPS_SAMO
              and abs(sh5 - float(osn['sharpe'])) <= 5e-5 + 1e-12)
    rows = pd.concat([a, b], ignore_index=True)
    rows['sverka'] = [f'свежий − кэш: max|Δ| {dd:.3e}; свежий − таблица теста (4 знака): max|Δ| {dt:.3e}'] * len(rows)
    to_csv(rows, 'table_4_h2_d0_samoproverka.csv')
    # старые числа по периодам (тот же код) — рядом с новыми в отчёте; HAR-J h=5 в тесте не было
    cpo = {k: v for k, v in cp.items() if k[0] == 'Cref'}
    per, met = by_period(s9, S, cpo, 'до исправления выравнивания (входы теста 10:32)')
    to_csv(per.assign(design=per['design'].map(design_name)), 'table_4_h2_d0_staroe_po_periodam.csv')
    res = dict(time=now(), ok=ok, pozicii_max_abs=dmax, c3_svezhiy_vs_kesh=dd, c3_vs_tablica_4zn=dt,
               osnovnaya_h5_sharpe=sh5, osnovnaya_h5_sharpe_kesh=sh5t, osnovnaya_tablica=float(osn['sharpe']),
               cref=a[num].iloc[0].to_dict())
    (OUT / 'samoproverka.json').write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float))
    print(json.dumps(res, ensure_ascii=False, indent=1, default=float))
    print(f'САМОПРОВЕРКА: {"OK" if ok else "НЕ ПРОЙДЕНА"}')
    return 0 if ok else 1


def stage_schet():
    for f, k in (('proverka.json', 'ok'), ('samoproverka.json', 'ok')):
        j = json.loads((OUT / f).read_text())
        if not j.get(k):
            raise SystemExit(f'СТОП: {f} не пройдена — счёт не запускается')
    pv = json.loads((OUT / 'proverka.json').read_text())
    for lab in ('hyb_h1', 'harj_h1', 'hyb_h5', 'harj_h5'):
        if sha(NEW[lab][0]) != pv['series'][lab]:
            raise SystemExit(f'СТОП: ряд {lab} изменился после проверки')
    if sha(NEW['naive_h1'][0]) != pv['series']['naive']:
        raise SystemExit('СТОП: наивная изменилась после проверки')
    s9 = s9_test()
    print('H2 d0: один запуск', now(), flush=True)
    cache = build_cache(s9, NEW)
    e, S = engine(s9, cache, NEW)
    cp = h2_ports(S)
    per, met = by_period(s9, S, cp, 'd0: после исправления выравнивания эталона и наивной')
    per['design'] = per['design'].map(design_name)
    met['design'] = met['design'].map(design_name)
    per['vhody'] = ('гибрид √rv_pred final_d0_s2emb_h{1|5}; HAR-J (б) √har_corr etalon_d0_s2emb_h{1|5}; '
                    'наивная √RV(d−4…d)')
    to_csv(per, 'table_4_h2_d0.csv')
    to_csv(met, 'table_4_h2_d0_metriki.csv')
    print('готово', now())
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=['proverka', 'samoproverka', 'schet'])
    a = ap.parse_args(argv)
    return {'proverka': stage_proverka, 'samoproverka': stage_samoproverka, 'schet': stage_schet}[a.stage]()


if __name__ == '__main__':
    sys.exit(main())
