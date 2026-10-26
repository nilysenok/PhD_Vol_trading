#!/usr/bin/env python3
"""R2, агент D: диагностика кластера 10 (протокол v3, docs/zadaniya/2026-10-26_vtoroy_raund_gibrida.md, раздел 6, строка D).

Только разработка 2017–2021. Каждая функция принимает период параметром; по умолчанию — assert «годы <= 2021» (guard_period).
Тот же код применяется к тесту после одного открытия (раздел 9) — только с флагом --posle-otkrytiya (allow_test=True);
без него test_locked/ не читается (guard_path), годы > 2021 не принимаются.

Что считается (по h и годам периода; выход — <regen>/R2/D/<profile>/*.csv + SHA256.txt):
  d_ladder.csv      лестница выигрыша: HAR-J → +s² → +HAR-X → +калибровка состояния → +бустинг → +согласование → финал;
                    QLIKE по годам и среднее T (равные веса лет), Δ к предыдущей существующей ступени и к первой, число лет с выигрышем
  d_ablation.csv    абляция: финал минус кластер, Δ = Q(без кластера)/Q(полный) − 1 (плюс — кластер помогает)
  d_shap.csv        SHAP остаточного ученика по группам признаков, всего и по терцилям RVI (вклады — pred_contrib LightGBM / XGBoost)
  d_stab.csv, d_stab_svod.csv   параметры калибровки (s², F = exp(s²/2)) и ρ_h по шагам; скачок > 50% между соседними шагами — флаг
  d_regimes.csv     QLIKE финала и эталонов по терцилям RVI, терцилям RV_m (внутри тикера), годам, 2020 / без 2020, тикерам
  d_tickers_sign.csv доля тикеров с выигрышем и знаковый тест (иллюстрация: тикеры коррелированы, p занижено)
  d_rank.csv        ранговые метрики главы 4: Спирмен σ_pred с RV (по тикерам, медиана), точность направления, автокорреляция
                    процентильного ранга σ_pred (лаг 1, 5) и средний |Δ ранга|
  d_regime_c.csv    доля дней с разным режимом C (fastbt.masks.c_masks) у пары моделей и QLIKE обеих на этих днях
  d_mz.csv, d_coverage.csv, d_rmse_mae.csv   приложение: Минсер–Зарновиц (уровни и логарифмы), охват, RMSE / MAE
  d_sdvig.csv       проверка информационного множества (п. 2.6): Спирмен ln ŷ_d с ln RV_{d+s}, s ∈ {−2, −1, 0, +1}
  d_terciles.csv    пороги терцилей RVI периода (для теста передаются пороги 2017–2021)
  d_s2_F.csv        множители Б0.1 по шагам у рядов, к которым поправку применяет этот модуль (s2 != bez)

Ядро агента S (scripts/analysis/r2_core.py, только импорт): данные d0 — RC.load_d0 + RC.check_d0 (RV_d, RV_m, RVI, цели);
ряды дизайнов R2 — путь кэша RC.cache_dir(RC.validate(design)) (D только читает, каталог не создаёт), rv_actual — из цели d0;
проверка сдвига — RC.shift_check (блоки D пишет в свой d_blok_sdvig.csv, блокирует O); эмбарго в SHAP — RC.train_mask / val_mask;
шаги — RC.assert_dev_step; «будущее не влияет» в самопроверке — RC.perturb_frame.

Ряды задаются в models/designs/r2/r2_D_diag.yaml (профиль r1 — ряды первого раунда d0: ВЫР-1, Э; профиль r2 — заполняется по
gate_E: {design: yaml, name: ряд, seeds: [...]}). Ряды с s2 != bez получают поправку Б0.1 по прошлым шагам (harx.s2_est, как r2_nogi.v0_tables): val шага k × exp(s²/2),
s² — по остаткам шагов < k; шаг 1 — F = 1. Годы до начала периода (warm_years) читаются только для s² и прогрева рангов.

Режимы:  python3 scripts/analysis/r2_diag.py [--profile r1] [--period 2017 2021] [--hs 1 5 22] [--smoke] [--selftest]
  --smoke    малый срез Мака: 3 тикера (SBER, LKOH, AFLT) × шаги 1–3 (период 2017–2018) × seed 42
  --selftest охрана периода и пути, «будущее не влияет» (Δ = 0 при порче лет после периода), подложенная утечка поймана (п. 2.6)
"""
import os
for _k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_k, '1')
import re, sys, glob, hashlib, argparse, warnings
from datetime import datetime
from pathlib import Path
import numpy as np, pandas as pd, pyarrow.parquet as pq, yaml
from scipy import stats

warnings.filterwarnings('ignore')
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (str(HERE), str(ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import harx as HX                                    # только s2_est, без изменений
import r2_core as RC                                 # ядро агента S: d0-данные, эмбарго, кэш дизайнов, shift_check, perturb_frame
from fastbt.masks import c_masks                     # режим C главы 4
from fastbt.signals.interface import TickerData

YAML = ROOT / 'models' / 'designs' / 'r2' / 'r2_D_diag.yaml'
DEV_LAST = 2021
EPS = 1e-12
TIE = 1e-10          # относительный допуск «равенства» QLIKE (среднее одинаковых seed даёт шум последнего разряда)
KEY = ['date', 'ticker']
COLS = ['date', 'ticker', 'year', 'step', 'rv_actual', 'rv_pred']


# ================================================================ охрана
def guard_period(period, allow_test=False):
    """(y0, y1); без allow_test — assert y1 <= 2021 (раздел 10: ничего за годы >= 2022)."""
    y0, y1 = int(period[0]), int(period[1])
    assert y0 <= y1, f'период {y0}–{y1}'
    if not allow_test:
        assert y1 <= DEV_LAST, f'период {y0}–{y1}: годы > {DEV_LAST} — только после открытия (--posle-otkrytiya)'
    return y0, y1


def guard_path(p, allow_test=False):
    if not allow_test:
        assert 'test_locked' not in str(p), f'{p}: test_locked/ до открытия не читается'
    return Path(p)


def guard_frame(d, y1):
    assert d.empty or int(d['year'].max()) <= y1, f'в данных год {int(d.year.max())} > {y1}'
    return d


# ================================================================ загрузка
def _read(path, y_lo, y_hi, allow_test=False):
    guard_path(path, allow_test)
    names = pq.read_schema(path).names
    if 'year' in names:
        flt = [('year', '>=', y_lo), ('year', '<=', y_hi)]
    else:
        flt = [('date', '>=', pd.Timestamp(f'{y_lo}-01-01')), ('date', '<', pd.Timestamp(f'{y_hi + 1}-01-01'))]
    d = pq.read_table(path, columns=[c for c in COLS if c in names], filters=flt).to_pandas()
    d['date'] = pd.to_datetime(d['date'])
    if 'year' not in d:
        d['year'] = d['date'].dt.year
    if 'step' not in d:
        d['step'] = d['year'] - 2015
    d['ticker'] = d['ticker'].astype(str)
    return guard_frame(d, y_hi)


def apply_s2(df, code):
    """Б0.1 по прошлым шагам: val шага k × exp(s²_{k−1}/2), шаг 1 — F = 1 (как r2_nogi.v0_tables). Возвращает (df, {шаг: F})."""
    steps = sorted(df.step.unique())
    if code in (None, 'bez'):
        return df, {int(k): 1.0 for k in steps}
    v = df[df.rv_actual > EPS]
    E = {int(st): np.log(g.rv_actual.values + 1e-10) - np.log(g.rv_pred.values) for st, g in v.groupby('step')}
    F = {int(k): (1.0 if k == 1 or (k - 1) not in E else float(np.exp(HX.s2_est(code, E, int(k) - 1) / 2))) for k in steps}
    out = df.copy()
    out['rv_pred'] = out['rv_pred'] * out['step'].map(F).astype(float)
    return out, F


def core_path(design, name, h, seed, part='val', root=None):
    """Путь ряда дизайна в кэше ядра (имя — как RC.series_path), без создания каталога: D только читает."""
    sd = 'det' if seed in (None, 'det') else int(seed)
    return RC.cache_dir(RC.validate(design), create=False, root=root) / f'{name}_h{int(h)}_s{sd}_{part}.parquet'


def attach_target(d, A, h):
    """rv_actual из d0-данных ядра (rv_target_h{h} строки d; ряды кэша R2 пишутся без цели — RC.SERIES_COLS)."""
    m = d.drop(columns=[c for c in ('rv_actual',) if c in d]).merge(
        A[['date', 'ticker', f'rv_target_h{h}']].rename(columns={f'rv_target_h{h}': 'rv_actual'}), on=KEY, how='left')
    assert m.rv_actual.notna().mean() > 0.99, f'h={h}: цель не найдена у {int(m.rv_actual.isna().sum())} строк'
    return m


def load_series(spec, h, period, regen, warm=3, allow_test=False, tickers=None, seeds=None, A=None):
    """Ряд по спецификации yaml: среднее по seed, поправка s², годы [y0 − warm, y1]. Возвращает (df, {шаг: F}).
    spec: {files: шаблон от regen} — сохранённые ряды (R1, ВЫР); или {design: yaml | dict, name: имя ряда} — кэш ядра R2
    (RC.validate → R2/cache/<sha16>/<name>_h{h}_s{seed|det}_val.parquet); нет rv_actual — берётся из A (RC.load_d0)."""
    y0, y1 = guard_period(period, allow_test)
    if 'design' in spec:
        des = spec['design']
        des = yaml.safe_load(open(ROOT / des)) if isinstance(des, str) else dict(des)
        des = dict(des, h=h)
        ss = seeds or spec.get('seeds') or ['det']
        paths = [core_path(des, spec['name'], h, s, root=spec.get('root')) for s in ss]
        files = f"{spec['name']}@{RC.design_hash(RC.validate(des))[:16]}"
    else:
        files = spec['files']
        ss = (seeds or spec.get('seeds') or [None]) if '{seed}' in files else [None]
        paths = [Path(regen) / files.format(h=h, seed=s) for s in ss]
    parts = []
    for p in paths:
        x = _read(p, y0 - warm, y1, allow_test)
        if not allow_test:
            for k in x.step.unique():
                RC.assert_dev_step(k)
        if 'rv_actual' not in x:
            assert A is not None, f'{p}: нет rv_actual — нужен A (RC.load_d0)'
            x = attach_target(x, A, h)
        parts.append(x)
    d = pd.concat(parts, ignore_index=True)
    g = d.groupby(['date', 'ticker', 'year', 'step'], sort=True)
    cnt = g.size()
    assert (cnt == len(parts)).all(), f'{files} h={h}: строки seed не совпадают'
    a = g.agg(rv_actual=('rv_actual', 'first'), rv_spread=('rv_actual', lambda x: float(np.ptp(x))), rv_pred=('rv_pred', 'mean'))
    assert (a.rv_spread.fillna(0) < 1e-12).all(), f'{files} h={h}: rv_actual seed расходятся'
    d = a.drop(columns='rv_spread').reset_index()
    d, F = apply_s2(d, spec.get('s2', 'bez'))
    if tickers:
        d = d[d.ticker.isin(tickers)]
    return d.sort_values(['date', 'ticker']).reset_index(drop=True)[COLS], F


_A = {}


def load_A(period, allow_test=False, tickers=None):
    """Данные d0 ядра: RC.load_d0 (r2_nogi.load(d0=True); rv_d, rv_w, rv_m — по день d; строки ≤ y1), RC.check_d0.
    Один раз на (y1, тикеры). После открытия max_year = y1 — только с allow_test."""
    y0, y1 = guard_period(period, allow_test)
    k = (y1, tuple(tickers) if tickers else None)
    if k not in _A:
        A, _ = RC.load_d0(tickers=tickers, max_year=y1)
        RC.check_d0(A)
        A['date'] = pd.to_datetime(A['date'])
        A['ticker'] = A['ticker'].astype(str)
        A['year'] = A['date'].dt.year
        _A[k] = guard_frame(A.sort_values(['ticker', 'date']).reset_index(drop=True), y1)
    return _A[k]


# ================================================================ метрики
def loss(y, p):
    r = np.asarray(y, float) / np.clip(np.asarray(p, float), EPS, None)
    return r - np.log(r) - 1


def qlike(y, p):
    return float(np.mean(loss(y, p))) if len(y) else np.nan


def in_period(d, period, allow_test=False):
    y0, y1 = guard_period(period, allow_test)
    return d[(d.year >= y0) & (d.year <= y1) & (d.rv_actual > EPS)]


def common(dfs, period, allow_test=False):
    """Общие строки (date, ticker) периода у нескольких рядов: {имя: df} на одном множестве строк."""
    P = {k: in_period(v, period, allow_test) for k, v in dfs.items()}
    keys = None
    for v in P.values():
        k = v[KEY + ['year']].drop_duplicates()
        keys = k if keys is None else keys.merge(k, on=KEY + ['year'])
    return {n: v.merge(keys, on=KEY + ['year']) for n, v in P.items()}


def q_years(d):
    q = {int(y): qlike(g.rv_actual.values, g.rv_pred.values) for y, g in d.groupby('year')}
    return q, (float(np.mean(list(q.values()))) if q else np.nan)


def _pct(a, b):
    return 100.0 * (a / b - 1.0) if (b and np.isfinite(a) and np.isfinite(b)) else np.nan


# ================================================================ 1. лестница
def ladder(S, rungs, h, period, allow_test=False):
    """rungs: [[ступень, ряд | None]]. Δ к предыдущей существующей ступени и к первой; T — среднее QLIKE лет."""
    have = {r: S[r] for _, r in rungs if r is not None and r in S}
    C = common(have, period, allow_test) if have else {}
    rows, prev, first = [], None, None
    for name, r in rungs:
        if r is None or r not in C:
            rows.append(dict(h=h, stupen=name, ryad=r, god='T', q=np.nan, d_prev_pct=np.nan, d_first_pct=np.nan, n=0,
                             wins_prev=np.nan, prim='нет ряда'))
            continue
        qy, qT = q_years(C[r])
        qp = q_years(C[prev])[0] if prev else None
        qf = q_years(C[first])[0] if first else None
        for y, q in qy.items():
            rows.append(dict(h=h, stupen=name, ryad=r, god=str(y), q=q, d_prev_pct=_pct(q, qp[y]) if qp else np.nan,
                             d_first_pct=_pct(q, qf[y]) if qf else np.nan, n=int((C[r].year == y).sum()), wins_prev=np.nan, prim=''))
        rows.append(dict(h=h, stupen=name, ryad=r, god='T', q=qT,
                         d_prev_pct=_pct(qT, float(np.mean(list(qp.values())))) if qp else np.nan,
                         d_first_pct=_pct(qT, float(np.mean(list(qf.values())))) if qf else np.nan, n=len(C[r]),
                         wins_prev=int(sum(qy[y] < qp[y] * (1 - TIE) for y in qy)) if qp else np.nan, prim=f'к {prev}' if prev else 'база'))
        prev = r
        first = first or r
    return pd.DataFrame(rows)


# ================================================================ 2. абляция
def ablation(S, pairs, h, period, allow_test=False):
    rows = []
    for lab, (full, abl) in pairs.items():
        if full not in S or abl not in S:
            rows.append(dict(h=h, klaster=lab, polnyy=full, bez=abl, god='T', q_poln=np.nan, q_bez=np.nan, d_pct=np.nan, prim='нет ряда'))
            continue
        C = common({full: S[full], abl: S[abl]}, period, allow_test)
        qa, qaT = q_years(C[full])
        qb, qbT = q_years(C[abl])
        for y in qa:
            rows.append(dict(h=h, klaster=lab, polnyy=full, bez=abl, god=str(y), q_poln=qa[y], q_bez=qb[y], d_pct=_pct(qb[y], qa[y]), prim=''))
        rows.append(dict(h=h, klaster=lab, polnyy=full, bez=abl, god='T', q_poln=qaT, q_bez=qbT, d_pct=_pct(qbT, qaT),
                         prim=f'лет, где кластер помогает: {sum(qb[y] > qa[y] * (1 + TIE) for y in qa)} из {len(qa)}'))
    return pd.DataFrame(rows)


# ================================================================ 3. SHAP
def shap_contrib(model, X):
    """Вклады (n × (p + 1), последний столбец — смещение): LightGBM pred_contrib, XGBoost pred_contribs."""
    mod = type(model).__module__
    if mod.startswith('lightgbm'):
        b = model.booster_ if hasattr(model, 'booster_') else model
        return np.asarray(b.predict(X, pred_contrib=True))
    if mod.startswith('xgboost'):
        import xgboost as xgb
        b = model.get_booster() if hasattr(model, 'get_booster') else model
        return np.asarray(b.predict(xgb.DMatrix(X), pred_contribs=True))
    raise TypeError(f'SHAP: неизвестная модель {mod}')


def shap_groups(contrib, cols, groups, terc=None, h=None, tag=''):
    """Средний |вклад группы| (сумма вкладов признаков группы на строке) и доля, всего и по терцилям RVI."""
    cols = list(cols)
    gmap = {g: [i for i, c in enumerate(cols) if re.match(rx, c)] for g, rx in groups.items()}
    rest = [i for i in range(len(cols)) if not any(i in v for v in gmap.values())]
    if rest:
        gmap['prochee'] = rest
    G = pd.DataFrame({g: contrib[:, ix].sum(1) if ix else 0.0 for g, ix in gmap.items()})
    rows = []
    for name, sel in [('vse', np.ones(len(G), bool))] + ([(f'RVI_T{t}', np.asarray(terc) == t) for t in (1, 2, 3)] if terc is not None else []):
        A = G[sel]
        if not len(A):
            continue
        tot = A.abs().mean().sum()
        for g in G.columns:
            rows.append(dict(h=h, model=tag, srez=name, gruppa=g, n_priznakov=len(gmap[g]), mean_abs=float(A[g].abs().mean()),
                             dolya=float(A[g].abs().mean() / tot) if tot > 0 else np.nan, mean_signed=float(A[g].mean()), n=int(sel.sum())))
    return pd.DataFrame(rows)


SHAP_RAW = (['rv_d', 'rv_w', 'rv_m', 'jump', 'close_RVI', 'rv_IMOEX', 'rsv_pos', 'rsv_neg', 'rskew', 'rkurt', 'rv_morning', 'rv_midday',
             'rv_evening']
            + [f'idx_RVI_rv_annualized_{s}' for s in ('ret1', 'ret5', 'ret22', 'vol22')]
            + [f'idx_IMOEX_rv_annualized_{s}' for s in ('ret1', 'ret5', 'ret22', 'vol22')]
            + [f'yahoo_{a}_{s}' for a in ('vix', 'brent', 'sp500') for s in ('ret1', 'ret5', 'vol22')]
            + [f'{a}_{s}' for a in ('key_rate_cbr', 'cpi_ru', 'si_fut') for s in ('lag1', 'diff22')])


def shap_smoke(cfg, h, regen, rvi_thr=None, allow_test=False):
    """Дымовой остаточный ученик: LightGBM на ln y − ln HAR-J(d0, эмбарго) (lnall Э, часть tr), 3 тикера × шаги smoke.
    Данные — RC.load_d0 (HAR-регрессоры d0), обучение — RC.train_mask (эмбарго h), val — RC.val_mask.
    Вклады — на val-годах шагов, входящих в 2017–2021. Только проверка кода: модели R2 и группы G0–G10 придут от B и F."""
    sm = cfg['shap']['smoke']
    steps = [int(k) for k in sm['steps']]
    for k in steps:
        RC.assert_dev_step(k, allow_test)
    vys = [2015 + k for k in steps]
    guard_period((min(vys), max(vys)), allow_test)
    import lightgbm as lgb
    groups = cfg['shap']['groups']
    t = f'rv_target_h{h}'
    A = load_A((min(vys), max(vys)), allow_test, sm['tickers'])
    feats = [c for c in SHAP_RAW if c in A]
    base = _read_lnall(Path(regen) / sm['base'].format(h=h), max(steps), allow_test)
    base = base[base.ticker.isin(sm['tickers'])]
    out = []
    for k in steps:
        vy = 2015 + k
        bk = base[base.step == k][KEY + ['part', 'base_ln']]
        tr = A[RC.train_mask(A, h, vy)].merge(bk[bk.part == 'tr'], on=KEY, how='inner')
        va = A[RC.val_mask(A, h, vy)].merge(bk[bk.part == 'va'], on=KEY, how='inner')
        y = np.log(tr[t].values + 1e-10) - tr.base_ln.values
        m = lgb.LGBMRegressor(random_state=int(sm['seed']), n_jobs=1, **sm['params']).fit(tr[feats], y)
        if vy < 2017:
            continue
        C = shap_contrib(m, va[feats])
        assert np.allclose(C.sum(1), m.predict(va[feats]), atol=1e-6), 'SHAP: сумма вкладов != прогноз'
        terc = None
        if rvi_thr is not None:
            terc = tercile(va['close_RVI'].values, rvi_thr)
        d = shap_groups(C, feats, groups, terc, h=h, tag=f'smoke_lgbm_shag{k}')
        d['god'] = vy
        out.append(d)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def _read_lnall(p, max_step, allow_test=False):
    guard_path(p, allow_test)
    d = pq.read_table(p, filters=[('step', '<=', int(max_step))]).to_pandas()
    d['date'] = pd.to_datetime(d['date'])
    d['ticker'] = d['ticker'].astype(str)
    assert d.empty or d.date.dt.year.max() <= 2015 + max_step
    return d


# ================================================================ 4. стабильность параметров
def calib_from_pair(Sc, Sb, h, ryad, period, allow_test=False):
    """Множитель калибровки по шагам = отношение ряда с поправкой к ряду без (внутри шага постоянен — проверяется)."""
    y0, y1 = guard_period(period, allow_test)
    m = Sc.merge(Sb, on=KEY + ['year', 'step'], suffixes=('_c', '_b'))
    m = m[(m.year >= y0) & (m.year <= y1)]
    rows = []
    for k, g in m.groupby('step'):
        r = g.rv_pred_c.values / g.rv_pred_b.values
        Fk = float(np.median(r))
        assert np.ptp(r) / Fk < 1e-6, f'{ryad} шаг {k}: множитель внутри шага не постоянен'
        rows += [dict(h=h, ryad=ryad, param='F', step=int(k), value=Fk), dict(h=h, ryad=ryad, param='s2', step=int(k), value=2 * np.log(Fk))]
    return pd.DataFrame(rows)


def rho_demo(S1, Sh, leg, fin1, h, period, allow_test=False):
    """ρ_h,k = Σ e_h·r̂₁ / Σ r̂₁² на val шага k: e_h = ln y_h − ln ŷ_h(нога), r̂₁ = ln ŷ₁(финал) − ln ŷ₁(нога). Описательно."""
    y0, y1 = guard_period(period, allow_test)
    a = S1[fin1].merge(S1[leg], on=KEY + ['year', 'step'], suffixes=('_f', '_l'))
    a['r1'] = np.log(a.rv_pred_f) - np.log(a.rv_pred_l)
    b = Sh[leg][Sh[leg].rv_actual > EPS].copy()
    b['e'] = np.log(b.rv_actual) - np.log(b.rv_pred)
    m = b.merge(a[KEY + ['r1']], on=KEY)
    m = m[(m.year >= y0) & (m.year <= y1)]
    return pd.DataFrame([dict(h=h, ryad=f'rho({leg}; r1={fin1})', param='rho', step=int(k),
                              value=float((g.e * g.r1).sum() / max((g.r1 ** 2).sum(), EPS))) for k, g in m.groupby('step')])


def stability(P, jump_max=0.5):
    """По параметру: значения по шагам, относительный скачок |Δ|/|пред.|, max, число скачков > jump_max, CV."""
    if P is None or P.empty:
        return pd.DataFrame(), pd.DataFrame()
    P = P.sort_values(['h', 'ryad', 'param', 'step']).copy()
    g = P.groupby(['h', 'ryad', 'param'])
    prev = g['value'].shift(1)
    P['skachok'] = (P['value'] - prev).abs() / prev.abs().clip(lower=EPS)
    P['flag'] = P['skachok'] > jump_max
    sv = g.agg(n_shagov=('step', 'size'), shag_min=('step', 'min'), shag_max=('step', 'max'), mean=('value', 'mean'), sd=('value', 'std'),
               max_skachok=('skachok', 'max'), n_flag=('flag', 'sum')).reset_index()
    sv['cv'] = sv['sd'] / sv['mean'].abs().clip(lower=EPS)
    sv['otkaz_c4_analog'] = sv['n_flag'] > 0
    return P, sv


def read_param_files(regen, patterns, h, period, allow_test=False):
    y0, y1 = guard_period(period, allow_test)
    out = []
    for pat in patterns or []:
        for f in sorted(glob.glob(str(Path(regen) / pat.format(h=h)))):
            guard_path(f, allow_test)
            d = pd.read_csv(f)
            if not {'step', 'param', 'value'} <= set(d.columns):
                continue
            d = d[(d.step + 2015 >= y0) & (d.step + 2015 <= y1)]
            d['h'] = h
            d['ryad'] = d['ryad'] if 'ryad' in d else Path(f).parent.name + '/' + Path(f).stem
            out.append(d[['h', 'ryad', 'param', 'step', 'value']])
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


# ================================================================ 5. режимы
def tercile(x, thr):
    x = np.asarray(x, float)
    t = np.where(x <= thr[0], 1, np.where(x <= thr[1], 2, 3))
    return np.where(np.isfinite(x), t, 0)


def regime_frame(A, period, cfg, allow_test=False, rvi_thr=None):
    """По данным d0 ядра (load_A): date, ticker, rv_d (RV дня d), rvi, rvm (rv_m ядра: среднее RV_{d−21..d}),
    rvi_t (терциль RVI по датам периода), rvm_t (терциль RV_m внутри тикера по периоду)."""
    y0, y1 = guard_period(period, allow_test)
    rc = cfg['regimes']
    F = A[(A.year >= y0) & (A.year <= y1)][['date', 'ticker', 'year', 'rv_d', rc['rvm_col'], rc['rvi_col']]].copy()
    F['rvm'] = F[rc['rvm_col']]
    F['rvi'] = F[rc['rvi_col']]
    dd = F.drop_duplicates('date')[['date', 'rvi']].dropna()
    if rvi_thr is None:
        rvi_thr = tuple(np.quantile(dd.rvi.values, [1 / 3, 2 / 3]))
    F['rvi_t'] = tercile(F.rvi.values, rvi_thr)
    F['rvm_t'] = F.groupby('ticker')['rvm'].transform(lambda s: pd.Series(tercile(s.values, np.nanquantile(s.values, [1 / 3, 2 / 3])), index=s.index))
    thr = pd.DataFrame([dict(peremennaya='RVI', period=f'{y0}-{y1}', t1=rvi_thr[0], t2=rvi_thr[1], n_dat=len(dd))])
    return F[['date', 'ticker', 'year', 'rv_d', 'rvi', 'rvm', 'rvi_t', 'rvm_t']], thr, rvi_thr


def errors_by_regime(S, final, refs, R, h, period, allow_test=False):
    """QLIKE финала и эталонов на общих строках по срезам; Δ = Q(финал)/Q(эталон) − 1 (минус — финал лучше)."""
    C = common({k: S[k] for k in [final] + refs if k in S}, period, allow_test)
    if final not in C:
        return pd.DataFrame(), pd.DataFrame()
    M = C[final][KEY + ['year', 'rv_actual']].copy()
    for k in C:
        M[f'L_{k}'] = loss(C[k].rv_actual.values, C[k].rv_pred.values)
    M = M.merge(R[KEY + ['rvi_t', 'rvm_t']], on=KEY, how='left')
    M['g2020'] = np.where(M.year == 2020, '2020', 'bez_2020')
    M['vse'] = 'vse'
    rows = []
    for dim in ('vse', 'rvi_t', 'rvm_t', 'year', 'g2020', 'ticker'):
        for val, g in M.groupby(dim):
            r = dict(h=h, srez=dim, znach=str(val), n=len(g), q_final=float(g[f'L_{final}'].mean()))
            for k in refs:
                if f'L_{k}' in g:
                    r[f'q_{k}'] = float(g[f'L_{k}'].mean())
                    r[f'd_{k}_pct'] = _pct(r['q_final'], r[f'q_{k}'])
            rows.append(r)
    E = pd.DataFrame(rows)
    sg = []
    for k in refs:
        if f'L_{k}' not in M:
            continue
        for god, G in [('T', M)] + [(str(y), g) for y, g in M.groupby('year')]:
            t = G.groupby('ticker')[[f'L_{final}', f'L_{k}']].mean()
            a, b = t[f'L_{final}'].values, t[f'L_{k}'].values
            w, l = int((a < b * (1 - TIE)).sum()), int((a > b * (1 + TIE)).sum())
            n = w + l                                      # ничьи (совпадающие ряды) в знаковый тест не входят
            p = stats.binomtest(w, n, 0.5, alternative='greater').pvalue if n else np.nan
            sg.append(dict(h=h, protiv=k, god=god, n_tikerov=len(t), n_nichya=len(t) - n, s_vyigryshem=w, dolya=w / len(t) if len(t) else np.nan, p_znak=p,
                           ogovorka='иллюстрация: тикеры коррелированы (общий рыночный фактор), p занижено'))
    return E, pd.DataFrame(sg)


# ================================================================ 6. ранговые метрики главы 4
def pct_rank(x, L):
    return pd.Series(x).rolling(L, min_periods=max(50, L // 5)).rank(pct=True).values


def rank_metrics(d, R, name, h, period, L=252, allow_test=False):
    """Спирмен σ_pred с RV (по тикерам; медиана), точность направления относительно RV дня d,
    автокорреляция процентильного ранга σ_pred (окно L, причинно, прогрев — годы до периода) лаг 1 и 5, средний |Δ ранга|."""
    y0, y1 = guard_period(period, allow_test)
    d = d.sort_values(['ticker', 'date']).copy()
    d['pr'] = d.groupby('ticker')['rv_pred'].transform(lambda s: pd.Series(pct_rank(np.sqrt(s.values), L), index=s.index))
    P = d[(d.year >= y0) & (d.year <= y1) & (d.rv_actual > EPS)].merge(R[KEY + ['rv_d']], on=KEY, how='left')
    rows = []
    for god, G in [('T', P)] + [(str(y), g) for y, g in P.groupby('year')]:
        sp = G.groupby('ticker').apply(lambda g: stats.spearmanr(g.rv_pred, g.rv_actual)[0] if len(g) > 5 else np.nan)
        ok = (G.rv_d > EPS).values
        a = np.sign(np.log(G.rv_pred.values[ok]) - np.log(G.rv_d.values[ok]))
        b = np.sign(np.log(G.rv_actual.values[ok]) - np.log(G.rv_d.values[ok]))
        nz = (a != 0) & (b != 0)
        ac = {}
        for lag in (1, 5):
            ac[lag] = G.groupby('ticker')['pr'].apply(lambda s: s.dropna().autocorr(lag) if s.notna().sum() > lag + 10 else np.nan)
        dr = G.groupby('ticker')['pr'].apply(lambda s: s.dropna().diff().abs().mean())
        rows.append(dict(h=h, model=name, god=god, n=len(G), spearman_med=float(sp.median()), spearman_mean=float(sp.mean()),
                         napravlenie=float((a[nz] == b[nz]).mean()) if nz.any() else np.nan, n_napr=int(nz.sum()),
                         ac_rank_l1=float(ac[1].median()), ac_rank_l5=float(ac[5].median()), mean_abs_drank=float(dr.median())))
    return pd.DataFrame(rows)


def _sig_hash(sig):
    h = hashlib.sha1()
    for k in sorted(sig):
        h.update(np.ascontiguousarray(np.nan_to_num(sig[k], nan=-1.0)).tobytes())
    return h.hexdigest()[:12]


def regime_c(SA, SB, tags, period, cats=('Trend', 'Contrarian', 'Range'), v41=False, allow_test=False):
    """SA, SB: {h: df} двух моделей (все h ∈ {1, 5, 22}; нет h — σ = NaN). Маски C — fastbt.masks.c_masks по σ = sqrt(rv_pred)
    на датах тикера (прогрев — годы до периода). d_t — доля конфигураций, где маски моделей различаются в день t.
    Выход: доля по конфигурациям (среднее d_t), доля дней с d_t > 0 и d_t >= 0,5; QLIKE обеих моделей на днях с d_t > 0
    (без весов и с весом d_t) и на всех днях — по каждому h."""
    y0, y1 = guard_period(period, allow_test)
    tick = sorted(set().union(*[set(v.ticker) for v in list(SA.values()) + list(SB.values())]))
    day = []
    for t in tick:
        dates = np.array(sorted(set().union(*[set(v.date[v.ticker == t]) for v in list(SA.values()) + list(SB.values())])), dtype='datetime64[ns]')
        if not len(dates):
            continue
        z = np.zeros(len(dates))
        pm = (dates >= np.datetime64(f'{y0}-01-01')) & (dates < np.datetime64(f'{y1 + 1}-01-01'))

        def sig(S):
            out = {}
            for hh in (1, 5, 22):
                a = np.full(len(dates), np.nan)
                if hh in S:
                    v = S[hh][S[hh].ticker == t]
                    ix = np.searchsorted(dates, v.date.values.astype('datetime64[ns]'))
                    a[ix] = np.sqrt(np.clip(v.rv_pred.values, EPS, None))
                out[f'h{hh}'] = a
            return out
        sa, sb = sig(SA), sig(SB)
        tda = TickerData(f'{t}|{tags[0]}|{_sig_hash(sa)}', dates, z, z, z, z, z, z, sa)
        tdb = TickerData(f'{t}|{tags[1]}|{_sig_hash(sb)}', dates, z, z, z, z, z, z, sb)
        for cat in cats:
            ra, ia, _ = c_masks(tda, cat, v41)
            rb, ib, _ = c_masks(tdb, cat, v41)
            D = (ra[ia][:, pm] != rb[ib][:, pm])
            day.append(pd.DataFrame({'ticker': t, 'date': dates[pm], 'cat': cat, 'dt': D.mean(0),
                                     'act_a': ra[ia][:, pm].mean(0), 'act_b': rb[ib][:, pm].mean(0)}))
    Dd = pd.concat(day, ignore_index=True)
    rows = []
    for cat, g in Dd.groupby('cat'):
        base = dict(para=f'{tags[0]} vs {tags[1]}', cat=cat, n_dney=len(g), dolya_konfig=float(g.dt.mean()),
                    dolya_dney_lyuboy=float((g.dt > 0).mean()), dolya_dney_bolshinstvo=float((g.dt >= 0.5).mean()),
                    aktivnost_a=float(g.act_a.mean()), aktivnost_b=float(g.act_b.mean()))
        for hh in sorted(set(SA) & set(SB)):
            C = common({'a': SA[hh], 'b': SB[hh]}, period, allow_test)
            m = C['a'][KEY + ['rv_actual', 'rv_pred']].merge(C['b'][KEY + ['rv_pred']], on=KEY, suffixes=('_a', '_b')).merge(g[KEY + ['dt']], on=KEY)
            la, lb = loss(m.rv_actual, m.rv_pred_a), loss(m.rv_actual, m.rv_pred_b)
            dif = m.dt.values > 0
            w = m.dt.values
            r = dict(base, h=hh, n_dif=int(dif.sum()), q_a_dif=float(la[dif].mean()) if dif.any() else np.nan,
                     q_b_dif=float(lb[dif].mean()) if dif.any() else np.nan,
                     q_a_w=float((la * w).sum() / w.sum()) if w.sum() > 0 else np.nan, q_b_w=float((lb * w).sum() / w.sum()) if w.sum() > 0 else np.nan,
                     q_a_vse=float(la.mean()), q_b_vse=float(lb.mean()))
            r['d_dif_pct'] = _pct(r['q_a_dif'], r['q_b_dif'])
            r['d_w_pct'] = _pct(r['q_a_w'], r['q_b_w'])
            rows.append(r)
    return pd.DataFrame(rows)


# ================================================================ 7. приложение
def _ols(x, y):
    X = np.column_stack([np.ones(len(x)), x])
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    r2 = 1 - np.sum((y - X @ b) ** 2) / np.sum((y - y.mean()) ** 2)
    return b[0], b[1], r2


def appendix(S, names, h, period, ref=None, allow_test=False):
    """MZ (y = α + β·ŷ, уровни и логарифмы; OLS без HAC — описательно), охват, RMSE/MAE (уровни RV)."""
    y0, y1 = guard_period(period, allow_test)
    names = [n for n in names if n in S]
    allk = pd.concat([S[n][(S[n].year >= y0) & (S[n].year <= y1)][KEY + ['year']] for n in names]).drop_duplicates()
    mz, cov, err = [], [], []
    C = common({n: S[n] for n in names}, period, allow_test)
    for n in names:
        d = S[n][(S[n].year >= y0) & (S[n].year <= y1)]
        for god, g in [('T', d)] + [(str(y), x) for y, x in d.groupby('year')]:
            nu = len(allk) if god == 'T' else int((allk.year == int(god)).sum())
            cov.append(dict(h=h, model=n, god=god, n=len(g), n_tikerov=g.ticker.nunique(), n_dat=g.date.nunique(),
                            dolya_rv_pol=float((g.rv_actual > EPS).mean()), dolya_pred_konech=float(np.isfinite(g.rv_pred).mean() if len(g) else np.nan),
                            dolya_ot_obedineniya=len(g) / max(nu, 1)))
        c = C[n]
        for god, g in [('T', c)] + [(str(y), x) for y, x in c.groupby('year')]:
            y, p = g.rv_actual.values, g.rv_pred.values
            a1, b1, r1 = _ols(p, y)
            a2, b2, r2 = _ols(np.log(p), np.log(y))
            mz.append(dict(h=h, model=n, god=god, n=len(g), alpha_ur=a1, beta_ur=b1, r2_ur=r1, alpha_ln=a2, beta_ln=b2, r2_ln=r2))
            e = dict(h=h, model=n, god=god, n=len(g), rmse=float(np.sqrt(np.mean((y - p) ** 2))), mae=float(np.mean(np.abs(y - p))),
                     qlike_pul=qlike(y, p))    # объединённый по строкам (в лестнице T — среднее лет)
            if ref and ref in C:
                gr = C[ref] if god == 'T' else C[ref][C[ref].year == int(god)]
                yr, pr = gr.rv_actual.values, gr.rv_pred.values
                e['rmse_k_ref'] = e['rmse'] / float(np.sqrt(np.mean((yr - pr) ** 2)))
                e['mae_k_ref'] = e['mae'] / float(np.mean(np.abs(yr - pr)))
            err.append(e)
    return pd.DataFrame(mz), pd.DataFrame(cov), pd.DataFrame(err)


# ================================================================ проверка информационного множества (п. 2.6)
def shift_check(d, A, name, h, period, block_file, allow_test=False):
    """RC.shift_check ядра на годах периода (A — load_A, строки ≤ y1: RV_{d+1} последнего дня периода не читается).
    Блок пишется в block_file D (R2/D/<профиль>/d_blok_sdvig.csv), не в R2/O/blok_sdvig.csv: блокирует ряды агент O."""
    y0, y1 = guard_period(period, allow_test)
    r = RC.shift_check(d, A, h, name, block_file=block_file, years=tuple(range(y0, y1 + 1)))
    r['blok'] = not r['ok']
    return r


# ================================================================ прогон
def load_cfg(path=YAML):
    with open(path) as f:
        return yaml.safe_load(f)


def run(cfg, profile, period, hs, out, smoke=False, allow_test=False, log=print):
    y0, y1 = guard_period(period, allow_test)
    regen = Path(cfg['regen'])
    P = cfg['profiles'][profile]
    warm = int(cfg.get('warm_years', 3))
    tick = cfg['smoke']['tickers'] if smoke else None
    seeds = cfg['smoke']['seeds'] if smoke else None
    A = load_A(period, allow_test)
    fix = cfg['regimes'].get('rvi_thr')
    R, thr, rvi_thr = regime_frame(A, period, cfg, allow_test, tuple(fix) if fix else None)
    if tick:
        R = R[R.ticker.isin(tick)]
    out.mkdir(parents=True, exist_ok=True)
    blk = out / 'd_blok_sdvig.csv'
    if blk.exists():
        blk.unlink()
    T = {k: [] for k in ('ladder', 'ablation', 'shap', 'stab', 'stab_svod', 'regimes', 'tickers_sign', 'rank', 'regime_c', 'mz',
                         'coverage', 'rmse_mae', 'sdvig')}
    SS, Fm = {}, []
    for h in hs:
        S = {}
        for n, spec in P['series'].items():
            S[n], F = load_series(spec, h, period, regen, warm, allow_test, tick, seeds, A)
            Fm += [dict(h=h, ryad=n, step=k, F=v) for k, v in F.items()]
        SS[h] = S
        log(f'h={h}: рядов {len(S)}, строк периода {", ".join(f"{n}={len(in_period(v, period, allow_test))}" for n, v in S.items())}')
        T['ladder'].append(ladder(S, P['ladder'], h, period, allow_test))
        T['ablation'].append(ablation(S, P.get('ablation') or {}, h, period, allow_test))
        ro = P['roles']
        refs = [r for r in (ro.get('etalon'), ro.get('harj')) if r]
        if ro.get('final'):
            e, s = errors_by_regime(S, ro['final'], refs, R, h, period, allow_test)
            T['regimes'].append(e)
            T['tickers_sign'].append(s)
        for n in cfg['rank']['models']:
            if n in S:
                T['rank'].append(rank_metrics(S[n], R, n, h, period, int(cfg['rank']['pct_lookback']), allow_test))
        mz, cov, err = appendix(S, list(S), h, period, ro.get('etalon'), allow_test)
        T['mz'].append(mz); T['coverage'].append(cov); T['rmse_mae'].append(err)
        AT = A[A.ticker.isin(tick)] if tick else A
        T['sdvig'].append(pd.DataFrame([shift_check(S[n], AT, n, h, period, blk, allow_test) for n in S]))
        par = [calib_from_pair(S[c], S[b], h, lab, period, allow_test) for lab, (c, b) in (P.get('calib_pairs') or {}).items()
               if c in S and b in S]
        par.append(read_param_files(regen, P.get('param_files'), h, period, allow_test))
        if P.get('rho_demo') and h != 1 and 1 in SS:
            rd = P['rho_demo']
            par.append(rho_demo(SS[1], S, rd['leg'], rd['fin_1'], h, period, allow_test))
        st, sv = stability(pd.concat([p for p in par if not p.empty], ignore_index=True) if any(not p.empty for p in par) else None,
                           float(cfg['stability']['jump_max']))
        T['stab'].append(st); T['stab_svod'].append(sv)
        try:
            T['shap'].append(shap_smoke(cfg, h, regen, rvi_thr, allow_test))
        except AssertionError:
            raise
        except Exception as ex:  # noqa: BLE001 — SHAP не блокирует остальную диагностику
            log(f'h={h}: SHAP пропущен ({type(ex).__name__}: {ex})')
    for a, b in cfg['rank']['c_pairs']:
        if all(a in SS[h] and b in SS[h] for h in hs):
            T['regime_c'].append(regime_c({h: SS[h][a] for h in hs}, {h: SS[h][b] for h in hs}, (a, b), period,
                                          tuple(cfg['rank']['c_cats']), bool(cfg['rank']['c_v41']), allow_test))
    files = []
    for k, v in T.items():
        v = [x for x in v if x is not None and not x.empty]
        if v:
            f = out / f'd_{k}.csv'
            pd.concat(v, ignore_index=True).to_csv(f, index=False, float_format='%.6g')
            files.append(f)
    if blk.exists():
        files.append(blk)
    thr.to_csv(out / 'd_terciles.csv', index=False); files.append(out / 'd_terciles.csv')
    pd.DataFrame(Fm).to_csv(out / 'd_s2_F.csv', index=False, float_format='%.8g'); files.append(out / 'd_s2_F.csv')
    with open(out / 'SHA256.txt', 'w') as f:
        f.write(f'# r2_diag.py profile={profile} period={y0}-{y1} smoke={smoke} {datetime.now():%Y-%m-%d %H:%M:%S}\n')
        for p in files:
            f.write(f'{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}\n')
    return T, SS, R


# ================================================================ самопроверка
def selftest(cfg, out, log=print):
    rows = []
    out.mkdir(parents=True, exist_ok=True)
    blk = out / 'selftest_blok_sdvig.csv'
    if blk.exists():
        blk.unlink()
    regen = Path(cfg['regen'])
    # 1. охрана периода и пути
    for per in [(2017, 2022), (2022, 2026)]:
        try:
            guard_period(per)
            ok = False
        except AssertionError:
            ok = True
        rows.append(dict(test=f'guard_period{per}', ok=ok))
    try:
        guard_path(regen / 'test_locked' / 'x.parquet')
        ok = False
    except AssertionError:
        ok = True
    rows.append(dict(test='guard_path test_locked', ok=ok))
    rows.append(dict(test='guard_period(2017,2026, allow_test) без чтения', ok=guard_period((2017, 2026), True) == (2017, 2026)))
    # 2. «будущее не влияет»: порча рядов 2019+ (×U(0,5; 2)) и данных d0 после 31.12.2018 (RC.perturb_frame) не меняет
    #    метрик периода 2017–2018 (лестница, ранговые метрики, режим C)
    P = cfg['profiles']['r1']
    per, tick = (2017, 2018), ['SBER', 'LKOH', 'AFLT']
    A = load_A((2017, 2021))
    At = A[A.ticker.isin(tick)]
    R, _, _ = regime_frame(At, per, cfg)
    Rp, _, _ = regime_frame(RC.perturb_frame(At, f'{per[1]}-12-31'), per, cfg)
    rng = np.random.default_rng(20261026)
    SS, SP = {}, {}
    for h in (1, 5, 22):
        SS[h] = {n: load_series(P['series'][n], h, (2017, 2021), regen, 3, False, tick, A=A)[0] for n in ('HAR-J', 'HAR-J+s2', 'F_d0')}
        SP[h] = {}
        for n, d in SS[h].items():
            x = d.copy()
            fut = (x.year > per[1]).values
            x.loc[fut, 'rv_pred'] *= rng.uniform(0.5, 2, fut.sum())
            x.loc[fut, 'rv_actual'] *= rng.uniform(0.5, 2, fut.sum())
            SP[h][n] = x
    lad = [['HAR-J', 'HAR-J'], ['+s2', 'HAR-J+s2'], ['final', 'F_d0']]

    def bundle(S, RR):
        a = [ladder(S[h], lad, h, per) for h in S] + [rank_metrics(S[h]['F_d0'], RR, 'F_d0', h, per) for h in S]
        a.append(regime_c({h: S[h]['F_d0'] for h in S}, {h: S[h]['HAR-J'] for h in S}, ('F_d0', 'HAR-J'), per, ('Trend', 'Range')))
        a += [errors_by_regime(S[h], 'F_d0', ['HAR-J+s2', 'HAR-J'], RR, h, per)[0] for h in S]
        return a
    BA, BB = bundle(SS, R), bundle(SP, Rp)
    dmax = 0.0
    for a, b in zip(BA, BB):
        na, nb = a.select_dtypes('number').fillna(-9).values, b.select_dtypes('number').fillna(-9).values
        dmax = max(dmax, float(np.max(np.abs(na - nb))) if na.size else 0.0)
    rows.append(dict(test='будущее не влияет: max |Δ| метрик 2017–2018 при порче 2019+', ok=dmax == 0.0, znach=dmax))
    # s² по прошлым шагам: порча 2019+ сырого ряда не меняет поправленный ряд 2017–2018
    raw = load_series(dict(P['series']['HAR-X'], s2='bez'), 5, (2017, 2021), regen, 3, A=A)[0]
    bad = raw.copy()
    fut = (bad.year > per[1]).values
    bad.loc[fut, 'rv_actual'] *= rng.uniform(0.5, 2, fut.sum())
    a, b = apply_s2(raw, 'sm')[0], apply_s2(bad, 'sm')[0]
    m = (a.year <= per[1]).values
    ds = float(np.max(np.abs(a.rv_pred.values[m] - b.rv_pred.values[m])))
    rows.append(dict(test='s² sm по прошлым шагам: порча 2019+ не меняет 2017–2018', ok=ds == 0.0, znach=ds))
    # контроль чувствительности: порча внутри периода меняет метрики
    SQ = {h: {n: d.assign(rv_pred=np.where(d.year == 2018, d.rv_pred * 1.5, d.rv_pred)) for n, d in SS[h].items()} for h in SS}
    q0 = ladder(SS[1], lad, 1, per).q.values
    q1 = ladder(SQ[1], lad, 1, per).q.values
    rows.append(dict(test='контроль: порча 2018 меняет QLIKE', ok=bool(np.nanmax(np.abs(q0 - q1)) > 0), znach=float(np.nanmax(np.abs(q0 - q1)))))
    # 3. подложенная утечка: ŷ_d := RV_{d+1} (цель h=1) — argmax = +1, ряд блокируется; честный F_d0 — argmax 0
    d = SS[1]['F_d0']
    leak = d.assign(rv_pred=np.clip(d.rv_actual.values, 1e-10, None))
    s_ok = shift_check(d, At, 'F_d0', 1, (2017, 2021), blk)
    s_lk = shift_check(leak, At, 'utechka', 1, (2017, 2021), blk)
    rows.append(dict(test='честный F_d0 h=1: argmax сдвига = 0', ok=s_ok['argmax'] == 0, znach=s_ok['argmax']))
    rows.append(dict(test='подложенная утечка поймана: argmax = +1, blok', ok=s_lk['argmax'] == 1 and s_lk['blok'], znach=s_lk['argmax']))
    # 4. ряд дизайна из кэша ядра: RC.write_series (без rv_actual) -> load_series({design, name}) = исходный ряд, цель — из A
    tmp = out / 'selftest_cache'
    des = {'design': 'D_selftest', 'h': 1}
    RC.write_series(des, 'diag_test', d.drop(columns=['rv_actual']), 1, 42, module='D', root=tmp)
    back = load_series({'design': des, 'name': 'diag_test', 'seeds': [42], 'root': str(tmp)}, 1, (2017, 2021), regen, 3, A=A)[0]
    cmp = d.merge(back, on=KEY + ['year', 'step'], suffixes=('', '_c'))
    okc = len(cmp) == len(d) == len(back) and np.allclose(cmp.rv_pred, cmp.rv_pred_c, rtol=1e-12) and \
        np.allclose(cmp.rv_actual, cmp.rv_actual_c, rtol=1e-9)
    rows.append(dict(test='кэш ядра: write_series -> load_series(design) совпадает, rv_actual из RC.load_d0', ok=bool(okc), znach=len(back)))
    # 5. SHAP: сумма вкладов = прогноз (assert внутри), группы покрывают все признаки
    sh = shap_smoke(cfg, 1, regen)
    tot = sh[sh.srez == 'vse'].groupby('model').dolya.sum()
    rows.append(dict(test='SHAP: доли групп в сумме 1', ok=bool(np.allclose(tot.values, 1.0)), znach=float(tot.mean())))
    T = pd.DataFrame(rows)
    for r in T.itertuples():
        log(f"  {'OK ' if r.ok else 'ОШИБКА'} {r.test}" + (f' ({r.znach:.3g})' if 'znach' in T and pd.notna(getattr(r, 'znach', np.nan)) else ''))
    return T


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--yaml', default=str(YAML))
    ap.add_argument('--profile', default=None)
    ap.add_argument('--period', type=int, nargs=2, default=None)
    ap.add_argument('--hs', type=int, nargs='*', default=None)
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--selftest', action='store_true')
    ap.add_argument('--posle-otkrytiya', action='store_true', help='только после «да» владельца и одного открытия (раздел 9)')
    a = ap.parse_args()
    cfg = load_cfg(a.yaml)
    allow = bool(a.posle_otkrytiya)
    prof = a.profile or cfg['profile']
    per = tuple(a.period or (cfg['smoke']['period'] if a.smoke else cfg['period']))
    guard_period(per, allow)
    hs = a.hs or cfg['hs']
    out = Path(cfg['regen']) / cfg['out'] / (prof + ('_smoke' if a.smoke else '') + ('' if per == (2017, 2021) else f'_{per[0]}_{per[1]}'))
    t0 = datetime.now()
    if a.selftest:
        out = Path(cfg['regen']) / cfg['out'] / 'selftest'
        T = selftest(cfg, out)
        T.to_csv(out / 'd_selftest.csv', index=False)
        print(f'selftest: {int(T.ok.sum())}/{len(T)} OK, {(datetime.now() - t0).seconds} с')
        sys.exit(0 if T.ok.all() else 1)
    run(cfg, prof, per, hs, out, a.smoke, allow)
    print(f'OK: {out} ({(datetime.now() - t0).seconds} с)')


if __name__ == '__main__':
    main()
