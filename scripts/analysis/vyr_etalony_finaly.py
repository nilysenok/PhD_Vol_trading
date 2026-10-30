#!/usr/bin/env python3
"""ВЫР-1. Решение владельца 26.10, 14:20: единое информационное множество для всех моделей — данные по день d включительно,
прогноз на d+1..d+h. Эталоны HAR-J (а, б) и финалы первого раунда переводятся в версию d0 (RV дня d в HAR-регрессорах) с эмбарго
по концу цели в HAR-ногах. Дефект: docs/otchety/2026-10-26_proverka_laga_harj.md.

Версия d0 и эмбарго — код агента Э (scripts/analysis/r2_nogi.py), только импортируется:
  r2_nogi.load(feat, d0=True[, plant=True]) — rv_d = RV_d, rv_w = среднее RV_{d−4..d}, rv_m = среднее RV_{d−21..d} (нулевой RV —
      последним положительным RV тикера); plant — подложенная утечка rv_d := значение дня d+1;
  r2_nogi.trmask(A, h, vy, emb=True) — обучение: годы < vy с целью, у которых окно цели [d+1, d+h] кончается до 1 янв. vy.
Шаг k: val-год vy = 2015 + k, тест — год vy + 1; обучение HAR — годы < vy (как gibrid_har и r2_nogi), шаг 10: val 2025, тест 2026.

Модели (дизайн и гиперпараметры первого раунда; меняются только HAR-ноги и HAR-признаки — на d0 с эмбарго):
  эталон (а) HAR-J как в диссертации: МНК на ln (gibrid_har), d0 + эмбарго, без поправки;
  эталон (б) — (а) × exp(s²_sm/2) (gibrid_bias.s2_est: val шага k — оценка шага k−1, на шаге 1 F = 1; тест шага k — оценка шага k);
  F_1 = D22_h1 top5/V0_XGB/B-sm/L0: бустинг d22_h1 (patch: ln HAR-X⁺₁ шага — признак har_ln), HAR-X⁺ (d22_harxp, цель 1, эмбарго уже
       в коде) пересчитан на матрице d0; признаки бустинга v3_rrv не меняются (rv дня d в них уже есть); top5 на шагах 1–6, на 7–10 —
       точка шага 6; seed s — все пять точек с seed s (правило Б1);
  F_5 = D12_h5 wrap2/V0_XGB/B-sm/L0: V9 + Д8.12 + Д7 + Huber (d24_1 FIN), нога = HuberRegressor HAR-J (l2_lib.har_ln) на rv_d/rv_w/rv_m
       версии d0 с эмбарго обучения ноги; бустинг — без эмбарго, как в первом раунде; wrap2 = (RV_xgb + RV_lgbm)/2 одного seed;
  F_22 = HARJ_corr = эталон (б) h=22 (детерминирован).
Метрики — только val шагов 2–6 (2017–2021): QLIKE года — mean(y/p − ln(y/p) − 1), rv_actual > 1e-12. Годы ≥ 2022 в метрики не входят.

Режимы (каждый — отдельный процесс: подмены l2_lib у h=1 и h=5 разные):
  etalon   — эталоны h = 1/5/22, шаги 1–10 + контроль кода (R1 = gibrid_har / final2_build.har_step10; d0 = ряды Э HARJ_emb_d0)
  hxp      — HAR-X⁺ (цель 1) на d0, шаги 1–10 -> data/regen/vyr/vyr1/HXP_lnall.parquet; контроль: R1 = data/regen/D22/HXP_lnall
  f1 / f5  — F_h(d0): val шагов 1–6 на seed 42–51; seed 42 — ещё шаги 7–10 и тест 1–10; контроль: тот же код в версии R1, seed 42 =
             сохранённые ряды первого раунда (≤ 1e-7) и QLIKE designs_rank (≤ 1e-9), иначе стоп
  f22      — F_22(d0) = эталон (б) h=22: кэш (продублирован по seed 42–51) и test_locked
  leak1 / leak5 / leakhar — детектор «будущее не влияет» (ниже)
  index    — INDEX_VYR.csv и SHA256 файлов test_locked/vyr
  tables   — сводка QLIKE до/после выравнивания
Выход:
  data/regen/vyr/etalon_d0_h{h}_val.parquet — val шагов 1–6 (har_raw = (а), har_corr = (б), F_sm)
  data/regen/test_locked/vyr/etalon_d0_h{h}_test.parquet, final_d0_h{h}_test.parquet — part = test (шаги 1–10) и val (шаги 7–10), без метрик
  data/regen/R2/cache/VYR/F_d0_h{h}_s{seed}_val.parquet (seed 42–51; h=22 — копия эталона (б)), ETA_{a,b}_d0_h{h}_sdet_val.parquet
  data/regen/R2/cache/INDEX_VYR.csv; data/regen/vyr/vyr1/ — промежуточные файлы, q_*.csv, leak_*.csv, control_*.csv
Детектор: срезы в середине года (2018-06-29, 2020-03-02, 2021-06-30) и на границе года (2017-12-31, 2019-12-31, 2020-12-31);
  после среза все числовые входы × U(0,5; 2) и цели, чей горизонт заходит за срез (r2_nogi._perturb).
  Ноги (HAR-J d0, HAR-X⁺ d0, Huber HAR-J d0): ln-прогнозы всех моделей с vy ≤ года среза (граница года: и модели следующего года)
     на строках ≤ среза совпадают точно; после среза меняются; с подложенной утечкой «RV дня d+1» изменение на строках ≤ среза видно.
  Цепочки финалов (нога -> бустинг точки rank 1 seed 42 -> s² sm): тестовые прогнозы года среза (шаг k = год − 2016; обучение < k-го
     val-года, ранняя остановка и s² — по val-годам до года среза) на строках ≤ среза совпадают точно; после среза меняются; утечка видна.
python3 scripts/analysis/vyr_etalony_finaly.py etalon|hxp|f1|f5|f22|leak1|leak5|leakhar|index|tables [--workers 3]"""
import os
for _k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_k] = '1'
DATA_S = os.environ.get('FASTBT_MAIN', '/Users/nilysenok/Desktop/Старое/PhD_Vol_trading') + '/data'  # К-4
os.environ['L2_FEAT'] = DATA_S + '/regen/features_pit_v3_rrv.parquet'  # до импорта l2_lib (матрица D22_h1); у f5 — d24_1.set_matrix
import sys, time, hashlib, argparse, tempfile, warnings, multiprocessing as mp
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import LinearRegression

warnings.filterwarnings('ignore')
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
import r2_nogi as RN
import gibrid_har as GH
import gibrid_bias as BI
import d22_harxp as P
import l2_lib as L

DATA = Path(DATA_S)
REG = DATA / 'regen'
VYR = REG / 'vyr'
WORK = VYR / 'vyr1'
LOCK = REG / 'test_locked' / 'vyr'
CACHE = REG / 'R2' / 'cache' / 'VYR'
INDEX = REG / 'R2' / 'cache' / 'INDEX_VYR.csv'
FEAT = REG / 'features_pit_v3.parquet'
RRV = REG / 'features_pit_v3_rrv.parquet'
D7H5 = REG / 'features_pit_v3_d7_h5.parquet'
SEEDS = tuple(range(42, 52))
DEV = tuple(range(1, 7))
ALL = tuple(range(1, 11))
HS = (1, 5, 22)
TOL = 1e-7
KEY = ['date', 'ticker', 'step']
COLS = ['date', 'ticker', 'year', 'step', 'rv_actual', 'rv_pred']
CFG5 = {'v9': True, 'harf': True, 'huber': True}
R1_REF = {  # QLIKE первого раунда, val 2017–2020 (среднее шагов 2–5) и 2021 — data/regen/final3/designs_rank_*.csv
    1: ('final3/designs_rank_D22_h1_h1.csv', 'top5|V0_XGB|B-sm|L0', 0.2857237110986891, 0.2291744344713429),
    5: ('final3/designs_rank_D12_h5_h5.csv', 'wrap2|V0_XGB|B-sm|L0', 0.21590568109908795, 0.1566402147552571),
    22: ('final3/designs_rank_HARJ_corr_h22.csv', 'single|V0_HAR|B-sm|L0', 0.23574941260180482, 0.1452833147573684)}


def log(*a):
    print(time.strftime('%F %T'), *a, flush=True)


def sha_file(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


CODE_SHA = sha_file(Path(__file__).resolve())


def put(df, p):
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + '.tmp')
    df.to_parquet(tmp, index=False)
    os.replace(tmp, p)


def rows_csv(p, rows, key=None):
    """Дописать строки в CSV; key — столбцы, по которым старые строки заменяются."""
    df = pd.DataFrame(rows)
    if p.exists():
        old = pd.read_csv(p)
        if key is not None:
            k = set(map(tuple, df[key].astype(str).values))
            old = old[~old[key].astype(str).apply(tuple, axis=1).isin(k)]
        df = pd.concat([old, df], ignore_index=True)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False)


def idx_row(mode, p, ryad, h, seed, df, note):
    st = sorted(int(x) for x in df.step.unique())
    rows_csv(WORK / f'index_{mode}.csv', [dict(файл=str(p.relative_to(DATA.parent)), ряд=ryad, h=h, seed=seed, шаги=f'{st[0]}-{st[-1]}',
                                               строки=len(df), sha256=sha_file(p), sha_кода=CODE_SHA, yaml='—', время=time.strftime('%F %T'),
                                               примечание=note)], key=['файл'])


def cache_write(mode, ryad, h, seed, V, note):
    V = V[COLS].sort_values(['step', 'date', 'ticker']).reset_index(drop=True)
    assert V.step.max() <= 6 and V.year.max() <= 2021, 'в кэш — только val шагов 1–6'
    p = CACHE / f'{ryad}_h{h}_s{seed}_val.parquet'
    put(V, p)
    idx_row(mode, p, ryad, h, seed, V, note)


def lock_write(mode, df, name):
    """test_locked/vyr: только запись; SHA — по своему новому файлу."""
    assert set(df.part) <= {'val', 'test'} and (df.loc[df.part == 'val', 'step'] >= 7).all()
    p = LOCK / name
    put(df.sort_values(['part', 'step', 'date', 'ticker']).reset_index(drop=True), p)
    rows_csv(WORK / f'lock_{mode}.csv', [dict(file=name, rows=len(df), steps_test=f"{df[df.part == 'test'].step.min()}-{df[df.part == 'test'].step.max()}",
                                              steps_val=f"{df[df.part == 'val'].step.min()}-{df[df.part == 'val'].step.max()}", sha256=sha_file(p),
                                              sha_koda=CODE_SHA, time=time.strftime('%F %T'))], key=['file'])
    log(f'test_locked/vyr/{name}: {len(df)} строк (метрики не считались)')


# ---------------------------------------------------------------- поправка и метрика
def sm_F(V):
    """V: val (step, rv_actual, rv_pred) -> (Fv, Ft): Fv[k] = exp(s²_sm шага k−1 / 2), Fv[1] = 1; Ft[k] = exp(s²_sm шага k / 2)."""
    v = V[V.rv_actual > 1e-12]
    E = {'c': {int(s): np.log(g.rv_actual.values + 1e-10) - np.log(g.rv_pred.values) for s, g in v.groupby('step')}}
    st = sorted(E['c'])
    has = lambda k: k >= 1 and all(j in E['c'] for j in range(max(1, k - 2), k + 1))  # детектор: val-шаги могут начинаться не с 1
    Fv = {k: 1.0 if k == 1 else float(np.exp(BI.s2_est('sm', E, 'c', k - 1) / 2)) for k in st if k == 1 or has(k - 1)}
    Ft = {k: float(np.exp(BI.s2_est('sm', E, 'c', k) / 2)) for k in st if has(k)}
    return Fv, Ft


def q_dev(V, col='rv_pred'):
    """QLIKE val 2017–2021 (шаги 2–6). Шаги > 6 отбрасываются ДО счёта."""
    d = V[V.step.between(2, 6)]
    d = d[d.rv_actual > 1e-12]
    assert d.year.max() <= 2021
    q = {int(s): RN.qlike(g.rv_actual.values, g[col].values) for s, g in d.groupby('step')}
    assert sorted(q) == [2, 3, 4, 5, 6]
    return dict(q1720=float(np.mean([q[s] for s in (2, 3, 4, 5)])), q2021=q[6], q1721=float(np.mean(list(q.values()))),
                **{f'y{2015 + s}': q[s] for s in q})


def qrow(mode, h, model, version, seed, V, col='rv_pred', note=''):
    r = dict(h=h, model=model, version=version, seed=seed, **q_dev(V, col), note=note, time=time.strftime('%F %T'))
    rows_csv(WORK / f'q_{mode}.csv', [r], key=['h', 'model', 'version', 'seed'])
    log(f"QLIKE h={h} {model} {version} s{seed}: 2017–20 {r['q1720']:.6f}, 2021 {r['q2021']:.6f}, 2017–21 {r['q1721']:.6f}")
    return r


def dmax(a, b, col='rv_pred', key=KEY):
    """max |Δ ln| по общим ключам и число строк вне пары."""
    a, b = a.copy(), b.copy()
    a['date'], b['date'] = pd.to_datetime(a.date), pd.to_datetime(b.date)
    m = a[key + [col]].merge(b[key + [col]], on=key, how='outer', suffixes=('', '_r'), indicator=True)
    bb = m._merge == 'both'
    d = float(np.abs(np.log(m.loc[bb, col]) - np.log(m.loc[bb, col + '_r'])).max()) if bb.any() else np.inf
    return d, int((~bb).sum())


def control(mode, name, a, b, tol=TOL, col='rv_pred', key=KEY):
    d, nout = dmax(a, b, col, key)
    ok = bool(d <= tol and nout == 0)
    rows_csv(WORK / f'control_{mode}.csv', [dict(check=name, max_abs_dln=d, n_vne_pary=nout, ok=ok, time=time.strftime('%F %T'))], key=['check'])
    log(f'контроль {name}: max|Δ ln| = {d:.2e}, вне пары {nout} -> {"OK" if ok else "ПРОВАЛ"}')
    if not ok:
        raise SystemExit(f'СТОП: контроль {name} не пройден')


def control_q(mode, h, V):
    f, des, m1720, q21 = R1_REF[h]
    S = pd.read_csv(REG / f)
    S = S[S.design == des].iloc[0]
    assert abs(S.mean_1720 - m1720) < 1e-15 and abs(S.q2021 - q21) < 1e-15
    q = q_dev(V)
    d = max(abs(q['q1720'] - m1720), abs(q['q2021'] - q21))
    ok = bool(d <= 1e-9)
    rows_csv(WORK / f'control_{mode}.csv', [dict(check=f'QLIKE R1 h={h} = {f} {des}', max_abs_dln=d, n_vne_pary=0, ok=ok, time=time.strftime('%F %T'))],
             key=['check'])
    log(f'контроль QLIKE R1 h={h}: {q["q1720"]:.12f} / {q["q2021"]:.12f} против {m1720:.12f} / {q21:.12f}, |Δ| {d:.1e} -> {"OK" if ok else "ПРОВАЛ"}')
    if not ok:
        raise SystemExit(f'СТОП: QLIKE R1 h={h} не повторён')


# ---------------------------------------------------------------- эталоны HAR-J (а), (б)
def harj_ln(A, harj, h, emb, steps):
    """ln HAR-J модели шага k на всех строках A (МНК на ln, как gibrid_har / r2_nogi.fam_harj; обучение — r2_nogi.trmask)."""
    t = f'rv_target_h{h}'
    H = GH.clean(np.log(A[harj].values.clip(1e-10)))
    y = np.log(A[t].values + 1e-10)
    out = {}
    for k in steps:
        tp = RN.trmask(A, h, 2015 + k, emb)
        out[k] = LinearRegression().fit(H[tp], y[tp]).predict(H)
    return out


def vt_frames(A, h, lns):
    """ln по шагам -> (val: год vy с целью; test: год vy+1 — шаги 1–9 с целью, как gibrid_har; шаг 10 — все строки 2026, как har_step10)."""
    t = f'rv_target_h{h}'
    V, T = [], []
    for k, ln in sorted(lns.items()):
        V.append(RN._val(A, h, k, ln))
        ty = 2016 + k
        sel = (A.year.values == ty) & (A[t].notna().values | (k == 10))
        T.append(pd.DataFrame({'date': pd.to_datetime(A.date.values[sel]), 'ticker': A.ticker.values[sel].astype(str), 'year': ty, 'step': k,
                               'rv_actual': A[t].values[sel], 'rv_pred': np.clip(np.exp(ln[sel]), 1e-10, None)}))
    return RN._fin(V), RN._fin(T)


def etalon_vt(h, feat=FEAT, d0=True, emb=True, plant=False, steps=ALL):
    A, harj = RN.load(feat, d0=d0, plant=plant)
    return vt_frames(A, h, harj_ln(A, harj, h, emb, steps))


def with_F(V, T):
    Fv, Ft = sm_F(V)
    V = V.assign(F_sm=V.step.map(Fv).values)
    T = T.assign(F_sm=T.step.map(Ft).values)
    return V.assign(rv_corr=V.rv_pred * V.F_sm), T.assign(rv_corr=T.rv_pred * T.F_sm)


def cmd_etalon(a):
    import final2_build as FB
    mode = 'etalon'
    for h in HS:
        # контроль R1: тот же код без d0 и эмбарго = gibrid_har.har_forecasts (шаги 1–9) и final2_build.har_step10 (шаг 10); метрик нет
        A0, harj0 = RN.load(FEAT)
        V0, T0 = vt_frames(A0, h, harj_ln(A0, harj0, h, False, ALL))
        gv, gt, _ = GH.har_forecasts(h, A0, harj0)
        v10, t10 = FB.har_step10(A0, harj0, h)
        control(mode, f'R1 h={h}: val шагов 1–10 = gibrid_har + har_step10', V0, pd.concat([gv, v10]))
        control(mode, f'R1 h={h}: тест шагов 1–10 = gibrid_har + har_step10 (только прогнозы)', T0, pd.concat([gt, t10]))
        V0c, _ = with_F(V0[V0.step <= 6], T0[T0.step <= 6])
        qrow(mode, h, 'HAR-J (а)', 'R1', 'det', V0c, 'rv_pred')
        qrow(mode, h, 'HAR-J (б)', 'R1', 'det', V0c, 'rv_corr')
        # d0 + эмбарго
        V, T = etalon_vt(h)
        control(mode, f'd0 h={h}: val шагов 1–6 = Э HARJ_emb_d0', V[V.step <= 6], pd.read_parquet(REG / 'R2/cache/E' / f'HARJ_emb_d0_h{h}_sdet_val.parquet'))
        V, T = with_F(V, T)
        Vd = V[V.step <= 6]
        qrow(mode, h, 'HAR-J (а)', 'd0', 'det', Vd, 'rv_pred')
        qrow(mode, h, 'HAR-J (б)', 'd0', 'det', Vd, 'rv_corr')
        out = lambda d: d.rename(columns={'rv_pred': 'har_raw', 'rv_corr': 'har_corr'})[['date', 'ticker', 'year', 'step', 'rv_actual', 'har_raw', 'har_corr', 'F_sm']]
        p = VYR / f'etalon_d0_h{h}_val.parquet'
        put(out(Vd), p)
        idx_row(mode, p, 'etalon_d0 (а: har_raw, б: har_corr)', h, 'det', Vd, 'эталоны HAR-J d0 + эмбарго, val шагов 1–6')
        L_ = pd.concat([out(V[V.step >= 7]).assign(part='val'), out(T).assign(part='test')], ignore_index=True)
        lock_write(mode, L_, f'etalon_d0_h{h}_test.parquet')
        for r, col, nt in (('ETA_a_d0', 'rv_pred', 'эталон (а): HAR-J d0 + эмбарго, без поправки'),
                           ('ETA_b_d0', 'rv_corr', 'эталон (б): HAR-J d0 + эмбарго × exp(s²_sm/2), val шага k — s² шага k−1')):
            cache_write(mode, r, h, 'det', Vd.assign(rv_pred=Vd[col].values), nt)


def cmd_f22(a):
    """F_22(d0) = эталон (б) h=22: кэш на seed 42–51 (копия) и test_locked."""
    mode = 'f22'
    V, T = with_F(*etalon_vt(22))
    Vd = V[V.step <= 6]
    for sd in SEEDS:
        cache_write(mode, 'F_d0', 22, sd, Vd.assign(rv_pred=Vd.rv_corr.values),
                    'F_22(d0) = эталон (б) h=22 (HARJ_corr d0 + эмбарго); детерминирован, продублирован по seed')
    qrow(mode, 22, 'F_h', 'd0', 'det', Vd, 'rv_corr', 'F_22 = эталон (б)')
    o = lambda d, part: d.assign(part=part, leg_raw=d.rv_pred, rv_pred=d.rv_corr, design='HARJ_corr d0 (single|V0_HAR|B-sm|L0)')[
        ['date', 'ticker', 'year', 'step', 'part', 'rv_actual', 'rv_pred', 'leg_raw', 'F_sm', 'design']]
    lock_write(mode, pd.concat([o(V[V.step >= 7], 'val'), o(T, 'test')], ignore_index=True), 'final_d0_h22_test.parquet')


# ---------------------------------------------------------------- HAR-X⁺ (цель 1) на d0
def hxp_ln(feat=RRV, d0=True, plant=False, steps=ALL):
    """ln HAR-X⁺₁ модели шага k на всех строках (d22_harxp.forecast_step: эмбарго и внутренняя валидация — в коде Д22)."""
    A, harj = RN.load(feat, d0=d0, plant=plant)
    A = P.add_targets(A)
    terms = P.terms_for('1')
    X = P.build_terms(A, terms)
    out = A[['date', 'ticker']].copy()
    for k in steps:
        out[f'hxp_1_{k}'] = P.forecast_step(A, '1', k, harj, terms, X)[2].values
    return out


def cmd_hxp(a):
    mode = 'hxp'
    t0 = time.time()
    r1 = hxp_ln(d0=False)
    ref = pd.read_parquet(REG / 'D22' / 'HXP_lnall.parquet')
    m = r1.merge(ref, on=['date', 'ticker'], suffixes=('', '_r1'), validate='one_to_one')
    d = max(float(np.nanmax(np.abs(m[f'hxp_1_{k}'] - m[f'hxp_1_{k}_r1']))) for k in ALL)
    ok = bool(d <= TOL and len(m) == len(ref) == len(r1))
    rows_csv(WORK / f'control_{mode}.csv', [dict(check='HXP R1 = D22/HXP_lnall (hxp_1_1..10)', max_abs_dln=d, n_vne_pary=len(ref) - len(m), ok=ok,
                                                 time=time.strftime('%F %T'))], key=['check'])
    log(f'контроль HXP R1: max|Δ ln| = {d:.2e} -> {"OK" if ok else "ПРОВАЛ"} ({time.time() - t0:.0f} с)')
    if not ok:
        raise SystemExit('СТОП: HAR-X⁺ R1 не повторён')
    d0 = hxp_ln(d0=True)
    put(d0, WORK / 'HXP_lnall.parquet')
    log(f'HXP d0 -> {WORK / "HXP_lnall.parquet"} ({time.time() - t0:.0f} с)')


# ---------------------------------------------------------------- F_1: D22_h1 (HAR-X⁺ на d0)
G = {}
_DC, _RC = {}, {}


def f1_setup(d0, feat=RRV, hxp_dir=None):
    import d22_h1 as P22
    P22.OUT = Path(hxp_dir) if hxp_dir else (WORK if d0 else REG / 'D22')  # load_base_x берёт {OUT}/HXP_lnall.parquet
    P22.patch()
    L.FEAT, L.FEAT_CFG = RRV, RRV.with_suffix('.config.json')
    A, FC = L.load_base(None if feat == RRV else feat)
    A2, cols, _, _ = L.build_features(A, FC, 1, P22.CFG)
    _DC.clear(); _RC.clear()
    G.update(A=A2, cols=cols, cfg=P22.CFG, btag=P22.TAG, h=1)
    return len(cols)


def f1_points():
    """Точки D22_h1 XGB: шаги 1–6 — top5 шага (rank 1–5), 7–10 — точка шага 6 (как tune_L2 / d22_h1 --step10)."""
    GR = pd.read_csv(REG / 'D22_h1_grid.csv').set_index(['model', 'point'])
    S = pd.read_csv(REG / 'D22_h1_selected_h1.csv')
    S = S[S.model == 'xgb'].set_index('step').point
    T = pd.read_parquet(REG / 'D22_h1_xgb_h1_top5_val.parquet', columns=['step', 'rank', 'point']).groupby(['step', 'rank']).point.first()
    out = {}
    for s in DEV:
        assert int(T[(s, 1)]) == int(S[s])
        out[s] = [(GR.loc[('xgb', int(T[(s, r)]))].to_dict(), r, int(T[(s, r)])) for r in range(1, 6)]
    p6 = int(S[6])
    assert all(int(S[s]) == p6 for s in (7, 8, 9))
    for s in (7, 8, 9, 10):
        out[s] = [(GR.loc[('xgb', p6)].to_dict(), 1, p6)]
    return out


def f1_unit(a):
    s, seed, items = a
    if s not in _DC:
        _DC.clear()
        _DC[s] = L.step_data(G['A'], G['cols'], 1, s, 'xgb', G['cfg'], G['btag'])
    D = _DC[s]
    ref = None
    if s > 1:
        if 'xgb' not in _RC:
            _RC['xgb'] = L.make_ref('xgb', G['A'], G['cols'], 1, G['cfg'], G['btag'])
        ref = _RC['xgb']
    fr = []
    for p, rank, pid in items:
        pv, pt, bi, _, _ = L.fit('xgb', p, D, G['cfg'], seed=seed, ref=ref)
        for part, pred, yr in (('val', pv, D['vy']), ('test', pt, D['ty'])):
            d = D['parts']['va' if part == 'val' else 'te']
            fr.append(pd.DataFrame({'date': pd.to_datetime(d['date'].values), 'ticker': d['ticker'].values.astype(str), 'year': yr, 'step': s,
                                    'part': part, 'rank': rank, 'point': pid, 'seed': seed, 'trees': bi, 'rv_actual': d[L.tcol(1)].values,
                                    'rv_pred': np.clip(np.exp(pred), 1e-10, None)}))
    return pd.concat(fr, ignore_index=True)


def leg_of(R, part):
    """Нога: среднее RV точек (top5 на шагах 1–6, одна точка на 7–10) по (date, ticker, step)."""
    d = R[R.part == part]
    return d.groupby(KEY, as_index=False).agg(year=('year', 'first'), rv_actual=('rv_actual', 'first'), rv_pred=('rv_pred', 'mean'))[COLS]


def run_units(fn, tasks, workers):
    t0, out = time.time(), []
    with mp.get_context('fork').Pool(workers) as pool:
        for i, r in enumerate(pool.imap_unordered(fn, tasks, chunksize=1), 1):
            out.append(r)
            if i % 5 == 0 or i == len(tasks):
                el = time.time() - t0
                log(f'  {i}/{len(tasks)} единиц, {el / 60:.1f} мин, осталось ~{el / i * (len(tasks) - i) / 60:.1f} мин')
    return pd.concat(out, ignore_index=True)


def final_outputs(mode, h, legs_v, legs_t, design):
    """legs_v: {seed: val ноги}, legs_t: тест ноги seed 42. Кэш F_d0 по seed (шаги 1–6), метрики, test_locked."""
    for sd in sorted(legs_v):
        V = legs_v[sd]
        Fv, Ft = sm_F(V)
        Fd = V[V.step <= 6]
        Fd = Fd.assign(rv_pred=Fd.rv_pred * Fd.step.map(Fv).values)
        cache_write(mode, 'F_d0', h, sd, Fd, f'F_{h}(d0) = {design}: HAR-нога/признак d0 + эмбарго, остальное — как в первом раунде')
        qrow(mode, h, 'F_h', 'd0', sd, Fd)
        rows_csv(WORK / f'factors_{mode}.csv', [dict(h=h, seed=sd, step=k, Fv=Fv[k], Ft=Ft.get(k, np.nan)) for k in Fv], key=['h', 'seed', 'step'])
        if sd == 42:
            T = legs_t
            assert sorted(V.step.unique()) == list(ALL) and sorted(T.step.unique()) == list(ALL)
            o = lambda d, F, part: d.assign(part=part, leg_raw=d.rv_pred, F_sm=d.step.map(F).values, rv_pred=d.rv_pred * d.step.map(F).values,
                                            design=f'{design} d0')[['date', 'ticker', 'year', 'step', 'part', 'rv_actual', 'rv_pred', 'leg_raw', 'F_sm', 'design']]
            lock_write(mode, pd.concat([o(V[V.step >= 7], Fv, 'val'), o(T, Ft, 'test')], ignore_index=True), f'final_d0_h{h}_test.parquet')


def cmd_f1(a):
    mode = 'f1'
    Pt = f1_points()
    # контроль: версия R1, seed 42, val шагов 1–6
    n = f1_setup(False)
    log(f'F_1 R1-контроль: признаков {n}')
    R = run_units(f1_unit, [(s, 42, Pt[s]) for s in sorted(DEV, reverse=True)], a.workers)
    sv = pd.read_parquet(REG / 'D22_h1_xgb_h1_val.parquet')
    control(mode, 'D22_h1 R1 single s42 шаги 1–6 = D22_h1_xgb_h1_val', leg_of(R[R['rank'] == 1], 'val'), sv[(sv.step <= 6) & (sv.seed == 42)])
    t5 = pd.read_parquet(REG / 'D22_h1_xgb_h1_top5_val.parquet')
    control(mode, 'D22_h1 R1 top5 s42 (5 точек) = D22_h1_xgb_h1_top5_val', R[R.part == 'val'], t5[t5.seed == 42], key=KEY + ['rank'])
    V1 = leg_of(R, 'val')
    Fv, _ = sm_F(V1)
    F1r = V1.assign(rv_pred=V1.rv_pred * V1.step.map(Fv).values)
    control_q(mode, 1, F1r)
    qrow(mode, 1, 'F_h', 'R1', 42, F1r)
    # d0
    if not (WORK / 'HXP_lnall.parquet').exists():
        raise SystemExit('нет HXP d0: сначала режим hxp')
    n = f1_setup(True)
    log(f'F_1 d0: признаков {n}')
    tasks = [(s, sd, Pt[s]) for s in sorted(ALL, reverse=True) for sd in SEEDS if s <= 6 or sd == 42]
    R = run_units(f1_unit, tasks, a.workers)
    put(R, WORK / 'f1_d0_raw.parquet')
    legs_v = {sd: leg_of(R[R.seed == sd], 'val') for sd in SEEDS}
    legs_t = leg_of(R[R.seed == 42], 'test')
    final_outputs(mode, 1, legs_v, legs_t, 'D22_h1 top5|V0_XGB|B-sm|L0')


# ---------------------------------------------------------------- F_5: D12_h5 (HAR-J Huber на d0 с эмбарго ноги)
_harj_cols0, _har_ln0 = L.harj_cols, L.har_ln
D0MAP = {'rv_d': 'd0_rv_d', 'rv_w': 'd0_rv_w', 'rv_m': 'd0_rv_m'}


def harj_cols_d0(A):
    return [D0MAP.get(c, c) for c in _harj_cols0(A)]


def har_ln_emb(tp, parts, h, hc, w=None, huber=False, y=None):
    """HAR-нога l2_lib.har_ln с эмбарго: обучение — строки окна, у которых окно цели кончается до 1 янв. vy (vy = последний год окна + 1)."""
    vy = int(tp['year'].max()) + 1
    k = tp['d0_tendy'].values < vy
    return _har_ln0(tp[k], parts, h, hc, None if w is None else w[k], huber, None if y is None else y[k])


def f5_setup(d0, feat=D7H5, plant=False):
    import d24_1 as X
    import final2_step10  # noqa: F401 — подмена l2_lib.split: шаг 10 (обучение <= 2024, val 2025, тест 2026)
    X.set_matrix(5, 'features_pit_v3.parquet', 'd7')
    A, FC = L.load_base(feat)
    A2, cols, _, _ = L.build_features(A, FC, 5, CFG5)
    if d0:
        B, _ = RN.load(feat, d0=True, plant=plant)
        m = A2[['date', 'ticker']].merge(B[['date', 'ticker', 'rv_d', 'rv_w', 'rv_m', '_tendy5']], on=['date', 'ticker'], how='left', validate='one_to_one')
        for c, c0 in D0MAP.items():
            A2[c0] = m[c].values
        A2['d0_tendy'] = m['_tendy5'].values
        assert A2[list(D0MAP.values())].notna().all().all()
        L.harj_cols, L.har_ln = harj_cols_d0, har_ln_emb
    else:
        L.harj_cols, L.har_ln = _harj_cols0, _har_ln0
    X._REF.clear()
    X.G.update(A=A2, cols=cols, cfg=CFG5, h=5)
    return X, len(cols)


def f5_points(X):
    Pt = X.points('D12_h5', 5, range(1, 10))
    for m in X.M:
        Pt[(m, 10)] = Pt[(m, 6)]
    return Pt


def f5_unit(a):
    import d24_1 as X
    m, s, seed, (p, rk, pnt) = a
    r = X._unit((m, s, [(p, seed, rk, pnt)], 'Bv2b'))[0]
    return pd.concat([r[6].assign(part='val'), r[7].assign(part='test')], ignore_index=True).assign(model=m)


def wrap2(R, part):
    d = R[R.part == part]
    x = d[d.model == 'xgb'][KEY + ['year', 'rv_actual', 'rv_pred']]
    l_ = d[d.model == 'lgbm'][KEY + ['rv_pred']]
    w = x.merge(l_, on=KEY, suffixes=('_x', '_l'), validate='one_to_one')
    assert len(w) == len(x) == len(l_)
    w['rv_pred'] = (w.rv_pred_x + w.rv_pred_l) / 2
    w['date'] = pd.to_datetime(w.date)
    return w[COLS].sort_values(['step', 'date', 'ticker']).reset_index(drop=True)


def cmd_f5(a):
    mode = 'f5'
    X, n = f5_setup(False)
    Pt = f5_points(X)
    log(f'F_5 R1-контроль: признаков {n}')
    R = run_units(f5_unit, [(m, s, 42, Pt[(m, s)][0]) for s in sorted(DEV, reverse=True) for m in X.M], a.workers)
    for m in X.M:
        sv = pd.read_parquet(REG / f'D12_h5_{m}_h5_val.parquet')
        sv = sv[(sv.step <= 6) & (sv.seed == 42)]
        control(mode, f'D12_h5 R1 {m} s42 шаги 1–6 = D12_h5_{m}_h5_val', R[(R.model == m) & (R.part == 'val')], sv)
    W = wrap2(R, 'val')
    sw = pd.read_parquet(REG / 'D12_h5_xgb_h5_wrap2_val.parquet')
    control(mode, 'D12_h5 R1 wrap2 s42 шаги 1–6 = D12_h5_xgb_h5_wrap2_val', W, sw[sw.step <= 6])
    Fv, _ = sm_F(W)
    F5r = W.assign(rv_pred=W.rv_pred * W.step.map(Fv).values)
    control_q(mode, 5, F5r)
    qrow(mode, 5, 'F_h', 'R1', 42, F5r)
    # d0
    X, n = f5_setup(True)
    log(f'F_5 d0: признаков {n}')
    tasks = [(m, s, sd, Pt[(m, s)][0]) for s in sorted(ALL, reverse=True) for sd in SEEDS for m in X.M if s <= 6 or sd == 42]
    R = run_units(f5_unit, tasks, a.workers)
    put(R, WORK / 'f5_d0_raw.parquet')
    legs_v = {sd: wrap2(R[R.seed == sd], 'val') for sd in SEEDS}
    legs_t = wrap2(R[R.seed == 42], 'test')
    final_outputs(mode, 5, legs_v, legs_t, 'D12_h5 wrap2|V0_XGB|B-sm|L0')


# ---------------------------------------------------------------- детектор «будущее не влияет»
CUTS_MID, CUTS_YE = RN.CUTS_MID, RN.CUTS_YE
CUTS = CUTS_MID + CUTS_YE


def cut_info(c):
    """-> (срез, граница года?, kmax ног: модели с vy <= года среза (граница — и следующего), шаги теста цепочки)."""
    cut = pd.Timestamp(c)
    ye = c in CUTS_YE
    k = cut.year - 2016  # шаг, чей тестовый год — год среза
    return cut, ye, cut.year - 2015 + (1 if ye else 0), ([k, k + 1] if ye else [k])


def perturbed(feat, h, td):
    rng = np.random.default_rng(20261026 + h)
    out = {}
    for c in CUTS:
        p = Path(td) / f'{feat.stem}_{c}.parquet'
        RN._perturb(feat, pd.Timestamp(c), rng, p)
        out[c] = p
    return out


def leg_frame(keys, lns, kmax):
    d = keys[['date', 'ticker']].copy()
    d['date'] = pd.to_datetime(d.date)
    for k in range(1, kmax + 1):
        d[f'ln_s{k}'] = lns[k]
    return d


def leg_cmp(b, p, cut, kmax):
    m = b.merge(p, on=['date', 'ticker'], suffixes=('', '_p'), validate='one_to_one')
    past = (m.date <= cut).values
    dp = df_ = 0.0
    for k in range(1, kmax + 1):
        d = np.abs(m[f'ln_s{k}'] - m[f'ln_s{k}_p']).values
        dp = max(dp, float(np.nanmax(d[past])))
        df_ = max(df_, float(np.nanmax(d[~past])) if (~past).any() else 0.0)
    return dp, df_


def chain_cmp(b, p, cut, steps):
    m = b.merge(p, on=KEY, suffixes=('', '_p'), validate='one_to_one')
    m = m[m.step.isin(steps)]
    d = np.abs(np.log(m.rv_pred) - np.log(m.rv_pred_p)).values
    past = (m.date <= cut).values
    return (float(d[past].max()) if past.any() else np.nan), (float(d[~past].max()) if (~past).any() else np.nan), int(past.sum())


def judge(kontur, h, c, base, pert, plant_b, plant_p, kind, steps_or_k):
    cut, ye, _, _ = cut_info(c)
    f = leg_cmp if kind == 'leg' else chain_cmp
    r0 = f(base, pert, cut, steps_or_k)
    r1 = f(plant_b, plant_p, cut, steps_or_k)
    ok = bool(r0[0] == 0.0 and r0[1] > 0 and r1[0] > 0)
    return dict(kontur=kontur, h=h, srez=c, tip='граница года' if ye else 'середина года', max_d_do_sreza=r0[0], max_d_posle_sreza=r0[1],
                podlog_max_d_do_sreza=r1[0], stroki_do_sreza=(r0[2] if kind == 'chain' else np.nan), ok=ok)


def chain_final(test, val, steps):
    """Итог цепочки на тестовых шагах steps: нога × exp(s²_sm шага k / 2) по val-остаткам шагов k−2..k."""
    _, Ft = sm_F(val)
    t = test[test.step.isin(steps)]
    return t.assign(rv_pred=t.rv_pred * t.step.map(Ft).values)


# --- h=1: нога HAR-X⁺ d0 и цепочка F_1
def _leak1_task(a):
    var, c, kmax, tsteps, feat = a
    plant = var.startswith('plant')
    hx = hxp_ln(feat, d0=True, plant=plant, steps=range(1, kmax + 1))
    with tempfile.TemporaryDirectory(dir=WORK) as td:
        hx.to_parquet(Path(td) / 'HXP_lnall.parquet', index=False)
        f1_setup(True, feat=feat, hxp_dir=td)
        Pt = f1_points()
        vs = range(max(1, min(tsteps) - 2), max(tsteps) + 1)
        R = pd.concat([f1_unit((s, 42, Pt[s][:1])) for s in vs], ignore_index=True)
    fin = chain_final(leg_of(R, 'test'), leg_of(R, 'val'), tsteps)
    lf = hx.rename(columns={f'hxp_1_{k}': f'ln_s{k}' for k in range(1, kmax + 1)})
    lf['date'] = pd.to_datetime(lf.date)
    return var, c, lf, fin


def _leak5_task(a):
    var, c, kmax, tsteps, feat = a
    plant = var.startswith('plant')
    X, _ = f5_setup(True, feat=feat, plant=plant)
    A = X.G['A']
    hc = L.harj_cols(A)
    lns = {k: L.har_ln(A[(A.year < 2015 + k) & A[L.tcol(5)].notna()], [A], 5, hc, None, True)[0] for k in range(1, kmax + 1)}
    lf = leg_frame(A, lns, kmax)
    Pt = f5_points(X)
    vs = range(max(1, min(tsteps) - 2), max(tsteps) + 1)
    R = pd.concat([f5_unit((m, s, 42, Pt[(m, s)][0])) for s in vs for m in X.M], ignore_index=True)
    fin = chain_final(wrap2(R, 'test'), wrap2(R, 'val'), tsteps)
    return var, c, lf, fin


def _leakhar_task(a):
    var, c, kmax, tsteps, feat, h = a
    plant = var.startswith('plant')
    A, harj = RN.load(feat, d0=True, plant=plant)
    lns = harj_ln(A, harj, h, True, range(1, max(kmax, max(tsteps)) + 1))
    lf = leg_frame(A, lns, kmax)
    V, T = vt_frames(A, h, lns)
    fin = chain_final(T, V, tsteps)
    return var, c, lf, fin, h


def leak_run(kontur_leg, kontur_chain, task, feat, h_list, workers, extra=()):
    rows = []
    KM = max(cut_info(c)[2] for c in CUTS)
    TS = sorted(set(sum([cut_info(c)[3] for c in CUTS], [])))
    with tempfile.TemporaryDirectory(dir=WORK) as td:
        for h in h_list:
            pf = perturbed(feat, h, td)
            jobs = [('base', None, KM, TS, feat), ('plant', None, KM, TS, feat)]
            for c in CUTS:
                _, _, km, ts = cut_info(c)
                jobs += [('pert', c, km, ts, pf[c]), ('plant_pert', c, km, ts, pf[c])]
            jobs = [j + ((h,) if extra else ()) for j in jobs]
            log(f'детектор {kontur_chain} h={h}: {len(jobs)} прогонов')
            with mp.get_context('fork').Pool(workers) as pool:
                res = pool.map(task, jobs, chunksize=1)
            R = {(r[0], r[1]): r for r in res}
            for c in CUTS:
                _, _, km, ts = cut_info(c)
                b, pb = R[('base', None)], R[('plant', None)]
                p, pp = R[('pert', c)], R[('plant_pert', c)]
                rows.append(judge(kontur_leg, h, c, b[2], p[2], pb[2], pp[2], 'leg', km))
                rows.append(judge(kontur_chain, h, c, b[3], p[3], pb[3], pp[3], 'chain', ts))
                log(f"  {c}: нога ok={rows[-2]['ok']} (Δ до {rows[-2]['max_d_do_sreza']:.1e}, после {rows[-2]['max_d_posle_sreza']:.1e}, "
                    f"подлог {rows[-2]['podlog_max_d_do_sreza']:.1e}); цепочка ok={rows[-1]['ok']} (Δ до {rows[-1]['max_d_do_sreza']:.1e}, "
                    f"после {rows[-1]['max_d_posle_sreza']:.1e}, подлог {rows[-1]['podlog_max_d_do_sreza']:.1e})")
    return rows


def cmd_leak(a, which):
    WORK.mkdir(parents=True, exist_ok=True)
    if which == 'leak1':
        rows = leak_run('нога HAR-X⁺ d0 (цель 1)', 'цепочка F_1: HAR-X⁺ d0 -> D22_h1 XGB -> s² sm', _leak1_task, RRV, [1], a.workers)
    elif which == 'leak5':
        rows = leak_run('нога HAR-J Huber d0 + эмбарго (D12_h5)', 'цепочка F_5: HAR-J Huber d0 -> D12_h5 XGB+LGBM -> wrap2 -> s² sm',
                        _leak5_task, D7H5, [5], a.workers)
    else:
        rows = leak_run('нога HAR-J d0 + эмбарго (эталон а)', 'цепочка эталона (б) = F_22 при h=22: HAR-J d0 -> s² sm', _leakhar_task, FEAT,
                        list(HS), a.workers, extra=True)
    df = pd.DataFrame(rows).assign(time=time.strftime('%F %T'))
    rows_csv(WORK / 'leak_vyr1.csv', df.to_dict('records'), key=['kontur', 'h', 'srez'])
    log(f'{which}: {int(df.ok.sum())}/{len(df)} OK')
    return bool(df.ok.all())


# ---------------------------------------------------------------- индекс, сводка
def cmd_index(a):
    parts = [pd.read_csv(p) for p in sorted(WORK.glob('index_*.csv'))]
    I = pd.concat(parts, ignore_index=True).drop_duplicates('файл', keep='last').sort_values('файл')
    I = I[I['файл'].map(lambda f: (DATA.parent / f).exists())]
    I.to_csv(INDEX, index=False)
    lk = pd.concat([pd.read_csv(p) for p in sorted(WORK.glob('lock_*.csv'))], ignore_index=True).drop_duplicates('file', keep='last')
    (LOCK / 'SHA256_vyr1.txt').write_text(''.join(f'{r.sha256}  {r.file}\n' for r in lk.sort_values('file').itertuples()))
    log(f'INDEX_VYR.csv: {len(I)} строк; SHA256_vyr1.txt: {len(lk)} файлов')


def cmd_tables(a):
    Q = pd.concat([pd.read_csv(p) for p in sorted(WORK.glob('q_*.csv'))], ignore_index=True)
    Q.to_csv(WORK / 'q_vse.csv', index=False)
    rows = []
    for h in HS:
        for model in ('HAR-J (а)', 'HAR-J (б)', 'F_h'):
            q = Q[(Q.h == h) & (Q.model == model)]
            if model == 'F_h' and h == 22:
                r1 = Q[(Q.h == 22) & (Q.model == 'HAR-J (б)') & (Q.version == 'R1')]
                d0 = Q[(Q.h == 22) & (Q.model == 'HAR-J (б)') & (Q.version == 'd0')]
            else:
                r1 = q[(q.version == 'R1') & (q.seed.astype(str).isin(['det', '42']))]
                d0 = q[(q.version == 'd0') & (q.seed.astype(str).isin(['det', '42']))]
            sd = q[(q.version == 'd0') & ~q.seed.astype(str).isin(['det'])]
            row = dict(h=h, model=model)
            for nm, d in (('R1', r1), ('d0', d0)):
                if len(d):
                    row.update({f'{nm}_1720': d.q1720.iloc[0], f'{nm}_2021': d.q2021.iloc[0], f'{nm}_1721': d.q1721.iloc[0]})
            if 'R1_1721' in row and 'd0_1721' in row:
                row['d_pct'] = 100 * (row['d0_1721'] / row['R1_1721'] - 1)
            if model == 'F_h' and len(sd) > 1:
                row.update(d0_1721_mean10=sd.q1721.mean(), d0_1721_sd10=sd.q1721.std(ddof=1), n_seed=len(sd))
            rows.append(row)
    # справочно, h=1: парно с F_1 первого раунда на тех же seed 42–51 (ряды Б1, R2/cache/B1/F1_h1_s*_val), 2017–2021
    b1 = [REG / 'R2/cache/B1' / f'F1_h1_s{sd}_val.parquet' for sd in SEEDS]
    if all(p.exists() for p in b1):
        d = np.array([q_dev(pd.read_parquet(CACHE / f'F_d0_h1_s{sd}_val.parquet'))['q1721'] - q_dev(pd.read_parquet(p))['q1721']
                      for sd, p in zip(SEEDS, b1)])
        s = d.std(ddof=1)
        rows.append(dict(h=1, model='F_h: d0 − R1 парно по 10 seed (Б1)', d_mean=d.mean(), d_sd=s, t_seed=d.mean() / (s / np.sqrt(len(d)))))
    T = pd.DataFrame(rows)
    T.to_csv(WORK / 'svodka_vyr1.csv', index=False)
    print(T.to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['etalon', 'hxp', 'f1', 'f5', 'f22', 'leak1', 'leak5', 'leakhar', 'index', 'tables'])
    ap.add_argument('--workers', type=int, default=3)
    a = ap.parse_args()
    a.workers = min(a.workers, 3)  # + главный процесс = не больше 4
    WORK.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    if a.mode.startswith('leak'):
        ok = cmd_leak(a, a.mode)
        log(f'готово за {(time.time() - t0) / 60:.1f} мин')
        sys.exit(0 if ok else 1)
    {'etalon': cmd_etalon, 'hxp': cmd_hxp, 'f1': cmd_f1, 'f5': cmd_f5, 'f22': cmd_f22, 'index': cmd_index, 'tables': cmd_tables}[a.mode](a)
    log(f'{a.mode}: готово за {(time.time() - t0) / 60:.1f} мин')


if __name__ == '__main__':
    main()
