#!/usr/bin/env python3
"""R2 протокол v3, агент B (бывш. Б1 + Б2; слияние r2_roli_h1.py и r2_roli_h5_h22.py): бустинг-роли поверх ноги.
Ключи схемы models/designs/r2/schema_v1.yaml, которые реализует модуль:
  кластер 4  — boost.role (none | feature | residual | offset | residual_std), boost.resid_source (in_sample | cross_fit), boost.lib,
               boost.grid (current | resid_small_v1 — точки r2_resid_grid.py), boost.early_stop (native | qlike_total), boost.monotone,
               boost.objective, boost.clip_sigma, boost.shrink;
  кластер 12 — boost.prob (off | scale_boost | quantile5), boost.prob_dist, boost.prob_window (scale_har — у L);
  кластер 15 — boost.boost_refit (yearly | quarterly_warm | monthly);
  из кластера 1 — то, что касается обучения бустинга: data.embargo (h), data.recency_halflife, data.history_start, data.winsor, data.window.
Определения, которые схема оставляет исполнителю, — models/designs/r2/r2_B_roli.yaml (defs; ❓ — умолчания).

Интерфейс для ядра (r2_core.py агента S):  build(design: dict, ctx: dict, steps) -> DataFrame
  design — дизайн (вложенный dict по схеме; недостающие ключи — default схемы), seed — design['seed'];
  ctx:
    A          — матрица (date, ticker, year, rv, цель rv_target_h1 | rv_target_mean_h{5,22}, признаки), отсортирована по date;
    h          — горизонт;
    feat_cols  — признаки по design.features (собирает ядро / F);
    leg        — ln-прогноз ноги: DataFrame (date, ticker, step, part ∈ {tr, va}, base_ln) — формат L (lnall);
    leg_refit  — только для resid_source = cross_fit: f(step, fit_idx, pred_idx) -> ln-прогноз ноги, обученной на строках A[fit_idx]
                 (тем же оценщиком ноги), на строках A[pred_idx]. Даёт L; в дыме — встроенный МНК HAR-J (leg_refit_harj_ols).
  -> строки val шагов steps: date, ticker, year, step, rv_actual, rv_pred (уровень, без поправки s² — калибровка кластера 3 у L).
role = none -> ряд ноги (exp(base_ln) на val) — V0; бустинг не обучается.

Командная строка (Мак — только код и дым: 3 тикера × шаги 1–3 × seed 42 × h ∈ {1, 5, 22}):
  python3 scripts/analysis/r2_roli.py smoke            дым: все роли и переключатели кластеров 4, 12, 15 на трёх h
  python3 scripts/analysis/r2_roli.py control          движок = l2_lib.fit первого раунда (D12_h1_A_d812, шаги 1–3, все тикеры) ≤ 1e-7
  python3 scripts/analysis/r2_roli.py leak             детектор «будущее не влияет» (Д26): Δ = 0 до среза, подложенная утечка поймана
  python3 scripts/analysis/r2_roli.py leak_xfit        отдельный детектор cross_fit
  python3 scripts/analysis/r2_roli.py shift            проверка информационного множества (п. 2.6) на дымовых рядах
Годы >= 2022 не читаются: дым и детекторы — шаги 1–3 (val 2016–2018), контроль — шаги 1–3."""
import os, sys, time, json, hashlib, argparse
from pathlib import Path

for _k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_k, '1')
HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
ROOT = Path('/Users/nilysenok/Desktop/Старое/PhD_Vol_trading')
R = ROOT / 'data' / 'regen'
os.environ.setdefault('L2_FEAT', str(R / 'features_pit_v3.parquet'))
sys.path.insert(0, str(HERE))
import numpy as np, pandas as pd, yaml
import lightgbm as lgb, xgboost as xgb

YB = yaml.safe_load(open(REPO / 'models' / 'designs' / 'r2' / 'r2_B_roli.yaml'))
SCHEMA = yaml.safe_load(open(REPO / 'models' / 'designs' / 'r2' / 'schema_v1.yaml'))
COLS = ['date', 'ticker', 'year', 'step', 'rv_actual', 'rv_pred']
KEY = ['date', 'ticker']
PAT, NMAX, MAX_BIN = 30, 1000, 255     # как l2_lib (PAT, NMAX, MAX_BIN первого раунда)
SPACE = {'xgb': [('max_depth', 'int'), ('learning_rate', 'f'), ('subsample', 'f'), ('colsample_bytree', 'f'), ('min_child_weight', 'f'),
                 ('reg_alpha', 'f'), ('reg_lambda', 'f')],
         'lgbm': [('num_leaves', 'int'), ('learning_rate', 'f'), ('bagging_fraction', 'f'), ('feature_fraction', 'f'), ('min_data_in_leaf', 'int'),
                  ('reg_alpha', 'f'), ('reg_lambda', 'f')]}
MYKEYS = ('role', 'resid_source', 'lib', 'grid', 'early_stop', 'monotone', 'objective', 'clip_sigma', 'shrink', 'linear_tree', 'prob',
          'prob_dist', 'prob_window', 'boost_refit')
DATAKEYS = ('info_set', 'embargo', 'train_halt', 'target', 'recency_halflife', 'history_start', 'winsor', 'train_target_estimator', 'window')
ISS_FIRST = pd.Timestamp('2014-06-09')  # первая дата матрицы v3 (history_start = 2014)


def log(*a):
    print(time.strftime('%F %T'), *a, flush=True)


# ---------------------------------------------------------------- кэш наборов (r2_dataset_cache агента S; владелец 16:25, п. 4)
# Включается, только если в ctx есть '_afp' (отпечаток матрицы A, считается в родителе до fork — prepare_run) и, для cross_fit,
# '_refit_id' (оценщик ноги). Ключ = ds_key(шаг, признаки, h, цель) + SHA содержимого: строки обучения, ln ноги, метка, веса,
# init_score, бины шага 1 (reference). Детекторы (leak) портят A — у них '_afp' нет, кэш выключен. R2_B_DSCACHE=0 — выключить.
DS_ON = os.environ.get('R2_B_DSCACHE', '1') != '0'
_REFM = {}


def _ah(a):
    """SHA содержимого массива (dtype, форма, байты); None -> None."""
    if a is None:
        return None
    a = np.ascontiguousarray(a)
    return hashlib.sha256(f'{a.dtype.str}{a.shape}'.encode() + a.tobytes()).hexdigest()[:32]


def a_fingerprint(A):
    """Отпечаток матрицы A (все столбцы и строки) — в родителе до fork."""
    hv = pd.util.hash_pandas_object(A, index=False).values
    return hashlib.sha256(json.dumps([str(c) for c in A.columns]).encode() + hv.tobytes()).hexdigest()[:32]


def leg_fingerprint(leg):
    return _ah(pd.util.hash_pandas_object(leg[['date', 'ticker', 'step', 'part', 'base_ln']], index=False).values)


def _dc():
    import r2_dataset_cache as DC
    return DC


def tcol(h):
    return 'rv_target_h1' if h == 1 else f'rv_target_mean_h{h}'


def _tc(ctx):
    """Цель шага: rv_target_h{h} (ядро / r2_nogi.load кладёт туда среднее RV за [d+1, d+h]) или rv_target_mean_h{h} (матрица v3)."""
    if '_tcol' not in ctx:
        h = ctx['h']
        ctx['_tcol'] = f'rv_target_h{h}' if f'rv_target_h{h}' in ctx['A'] else tcol(h)
    return ctx['_tcol']


def qlike(y, f):
    f = np.clip(f, 1e-10, None)
    return float(np.mean(y / f - np.log(y / f) - 1))


# ---------------------------------------------------------------- дизайн: default схемы и проверка значений
def onoff(v):
    """YAML 1.1 читает off/on как False/True — возвращаем строки схемы."""
    return 'off' if v is False else ('on' if v is True else v)


def full(design):
    """Ключи boost.* и data.* дополняются default схемы; значение вне values — ошибка (стоп до счёта)."""
    d = {k: dict(v) if isinstance(v, dict) else v for k, v in design.items()}
    for blk, keys in (('boost', MYKEYS), ('data', DATAKEYS)):
        d.setdefault(blk, {})
        for k in keys:
            s = SCHEMA[blk][k]
            vals = [onoff(x) for x in s['values']]
            v = onoff(d[blk].get(k, s['default']))
            if isinstance(v, float) and v.is_integer() and any(isinstance(x, int) and not isinstance(x, bool) for x in vals):
                v = int(v)
            assert v in vals, f'{blk}.{k} = {v!r} нет в схеме {vals}'
            d[blk][k] = v
    assert d['h'] in (1, 5, 22)
    d.setdefault('seed', SCHEMA['seed']['default'])
    if d['data']['target'] != 'ln_mean' or d['data']['train_target_estimator'] != 'RV10':
        raise NotImplementedError('data.target / train_target_estimator ≠ default — не в модуле B')
    return d


# ---------------------------------------------------------------- точки (grid = current) и исключения признаков R1
_PTS, _EXC = {}, {}


def points(h, role, m, step):
    spec = YB['points'][h][role]
    k = (spec['src'], h)
    if k not in _PTS:
        S = pd.read_csv(R / f'{spec["src"]}_selected_h{h}.csv')
        _PTS[k] = {(r['model'], int(r['step'])): r for _, r in S.iterrows()}
    r = _PTS[k][(m, min(step, 6) if step > 9 else step)]
    return {n: (int(r[n]) if t == 'int' else float(r[n])) for n, t in SPACE[m]}


def excluded(h, role, m, step):
    if step == 1:
        return set()
    bt = YB['points'][h][role]['btag']
    if bt not in _EXC:
        p = R / f'{bt}_excluded_features.csv'
        _EXC[bt] = pd.read_csv(p) if p.exists() else pd.DataFrame(columns=['model', 'h', 'feature'])
    e = _EXC[bt]
    return set(e[(e.model == m) & (e.h == h)].feature)


def lhs_point(i):
    """Точка LHS-60 (r2_resid_grid.py) -> параметры LightGBM; i = 61 — текущая точка (контроль)."""
    g = pd.read_csv(REPO / YB['lhs']['file']).set_index('point')
    r = g.loc[i]
    return dict(num_leaves=int(r.num_leaves), max_depth=int(r.max_depth), min_data_in_leaf=int(r.min_data_in_leaf), learning_rate=float(r.learning_rate),
                reg_lambda=float(r.lambda_l2), reg_alpha=0.0, feature_fraction=float(r.feature_fraction), bagging_fraction=float(r.bagging_fraction))


# ---------------------------------------------------------------- окна шага
def split(ctx, d, step, before=None):
    """-> (индексы обучения, индексы val-года, vy). Маски — ядра (r2_core.train_mask / val_mask / inner_masks: эмбарго h — окно цели
    [d+1, d+h] кончается до 1 января vy). before = vy−1 — внутренняя складка (обучение < vy−1 с эмбарго, валидация — строки vy−1,
    чьё окно цели не заходит в vy)."""
    import r2_core as C
    A, h = ctx['A'], ctx['h']
    if f'rv_target_h{h}' not in A:
        A[f'rv_target_h{h}'] = A[_tc(ctx)]
    if before is None:
        vy = 2015 + step
        trm = C.train_mask(A, h, vy, emb=not ctx.get('_no_embargo'))   # _no_embargo — только контроль движка с R1
        vam = C.val_mask(A, h, vy)
    else:
        vy = before
        trm, vam, _ = C.inner_masks(A, h, before + 1)
    yr = A['year'].values
    if d['data']['window'] == 'rolling3y':
        trm = trm & (yr >= vy - 3)
    if d['data']['history_start'] == 2014:
        trm = trm & (pd.to_datetime(A['date']).values >= np.datetime64(ISS_FIRST))
    tr = np.flatnonzero(trm)
    if ctx.get('_rowmask') is not None:  # драйвер S, фаза B: --row-frac / --row-seed — подвыборка строк (дата, тикер), маска по строкам A
        tr = tr[ctx['_rowmask'][tr]]
    if ctx.get('_half'):  # скрининг (владелец 16:25): 50% строк обучения, страты тикер × год, random_state 42 (как l2_lib half)
        s = pd.DataFrame({'i': tr, 't': A['ticker'].values[tr], 'y': A['year'].values[tr]})
        tr = np.sort(s.groupby(['t', 'y'], group_keys=False).sample(frac=.5, random_state=42)['i'].values)
    return tr, np.flatnonzero(vam), vy


# ---------------------------------------------------------------- нога
def leg_ln(ctx, step, idx, part):
    """ln ноги шага на строках A[idx]: part tr — внутри выборки (формат L), va — OOS. Нет значения — NaN."""
    L = ctx['_legtab']
    if (step, part) not in L:
        return np.full(len(idx), np.nan)
    rows = ctx['A'].iloc[idx][KEY]
    return rows.merge(L[(step, part)], on=KEY, how='left')['base_ln'].values


def prep_leg(ctx):
    if '_legtab' not in ctx:
        g = ctx['leg'][['date', 'ticker', 'step', 'part', 'base_ln']].copy()
        g['date'] = g['date'].astype(ctx['A']['date'].dtype)
        ctx['_legtab'] = {(int(s), p): x[KEY + ['base_ln']] for (s, p), x in g.groupby(['step', 'part'])}


def leg_train(ctx, d, step, tr):
    """ln ноги на строках обучения: in_sample — из lnall (part tr; нет строки — HAR-J нет, ошибка); cross_fit — K = 5 блоков с эмбарго h."""
    if d['boost']['resid_source'] == 'in_sample':
        v = leg_ln(ctx, step, tr, 'tr')
        miss = np.isnan(v)
        if miss.any():  # у HEAVY нет прогноза на первую дату матрицы: строки без ноги исключаются из обучения роли
            assert pd.to_datetime(ctx['A']['date'].values[tr[miss]]).nunique() <= 1, (step, int(miss.sum()))
        return v
    return cross_fit(ctx, step, tr, ctx['h'])


def blocks(tr, A, K=5):
    """K последовательных по дате блоков строк обучения (границы — по датам, тикер целиком в блоке)."""
    dates = np.sort(A['date'].values[tr])
    cuts = [dates[int(len(dates) * k / K)] for k in range(1, K)]
    b = np.searchsorted(np.array(cuts), A['date'].values[tr], side='right')
    return b


def cross_fit(ctx, step, tr, h, K=5, embargo=None):
    """OOF-прогноз ноги на строках обучения: для блока b нога обучается на остальных блоках без h торговых дней по обе стороны блока."""
    A = ctx['A']
    emb = h if embargo is None else embargo
    b = blocks(tr, A, K)
    dts = np.sort(A['date'].unique())
    pos = {dd: i for i, dd in enumerate(dts)}
    di = np.array([pos[x] for x in A['date'].values[tr]])
    out = np.full(len(tr), np.nan)
    for k in range(K):
        inb = b == k
        lo, hi = di[inb].min(), di[inb].max()
        fit = ~inb & ((di < lo - emb) | (di > hi + emb))
        out[inb] = ctx['leg_refit'](step, tr[fit], tr[inb])
    return out


def leg_refit_harj_ols(ctx):
    """Встроенный оценщик ноги для дыма: МНК ln RV на ln регрессоров HAR-J (l2_lib.harj_cols) — как gibrid_har / l2_lib.har_ln."""
    import l2_lib as L
    A, h = ctx['A'], ctx['h']
    X = np.log(np.clip(A[ctx.get('harj') or L.harj_cols(A)].values.astype(float), 1e-10, None))
    X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
    X = np.column_stack([np.ones(len(X)), X])
    y = np.log(A[_tc(ctx)].values + 1e-10)

    def f(step, fit_idx, pred_idx):
        beta = np.linalg.lstsq(X[fit_idx], y[fit_idx], rcond=None)[0]
        return X[pred_idx] @ beta
    return f


# ---------------------------------------------------------------- признаки, веса, цель
def design_names(ctx, d, role, m, step):
    """-> (столбцы A, имена признаков модели с leg_ln)."""
    h = ctx['h']
    ex = excluded(h, role, m, step) if d['boost']['grid'] == 'current' or m == 'xgb' else set()
    if 'groups' in ctx:  # признаки по design.features: G0 (база v3 в версии d0) + группы F G1–G10
        fc = [c for g in (['G0'] + [g for g in d.get('features', ['G0']) if g != 'G0']) for c in ctx['groups'][g]]
    else:
        fc = ctx['feat_cols']
    keep = [c for c in fc if c not in ex]
    names = list(keep) + (['leg_ln'] if role == 'feature' or YB['points'][h][role]['leg_feature'] else [])
    return keep, names


def design_X(ctx, d, role, m, step, idx, legv):
    keep, names = design_names(ctx, d, role, m, step)
    X = ctx['A'].iloc[idx][keep].values.astype(np.float64)
    if names[-1:] == ['leg_ln']:
        X = np.column_stack([X, legv])
    X = np.nan_to_num(X, nan=0, posinf=0, neginf=0).astype(np.float32)
    mono = [1 if (d['boost']['monotone'] == 'on' and role == 'feature' and n in set(YB['monotone_plus'])) else 0 for n in names]
    cats = [i for i, n in enumerate(names) if n.startswith('cat_')]
    return X, names, mono, cats


def weights(ctx, d, tr):
    hl = d['data']['recency_halflife']
    if hl == 'inf':
        return None
    dt = pd.to_datetime(ctx['A']['date'].values[tr])
    age = (dt.max() - dt).days.values / 365.25
    return 0.5 ** (age / float(hl[:-1]))


def sigma_t(ctx, step, tr, va, legtr, legva, h):
    """residual_std: σ_t — СКО остатков ноги тикера за 66 строк, окна целей которых закончились к t (строки ≤ t−h), min 20; иначе СКО по обучению."""
    A = ctx['A']
    idx = np.concatenate([tr, va])
    e = np.log(A[_tc(ctx)].values[idx] + 1e-10) - np.concatenate([legtr, legva])
    df = pd.DataFrame({'i': idx, 'ticker': A['ticker'].values[idx], 'date': A['date'].values[idx], 'e': e}).sort_values(['ticker', 'date'])
    df['s'] = df.groupby('ticker')['e'].transform(lambda x: x.shift(h).rolling(66, min_periods=20).std())
    s0 = float(np.nanstd(e[:len(tr)]))
    s = df.set_index('i')['s'].reindex(idx).fillna(s0).clip(lower=0.05 * s0).values
    return s[:len(tr)], s[len(tr):]


# ---------------------------------------------------------------- обучение (параметры — как l2_lib.fit первого раунда)
def _xprm(p, obj, seed, mono, ytr, bm):
    prm = dict(tree_method='hist', max_bin=MAX_BIN, max_depth=int(p['max_depth']), eta=p['learning_rate'], subsample=p['subsample'],
               colsample_bytree=p['colsample_bytree'], min_child_weight=p['min_child_weight'], alpha=p['reg_alpha'], reg_lambda=p['reg_lambda'],
               nthread=1, seed=seed, verbosity=0)
    if obj == 'gamma':
        prm.update(objective='reg:gamma', eval_metric='gamma-nloglik')
        if not bm:
            prm['base_score'] = float(np.mean(ytr))
    elif obj == 'huber':
        prm.update(objective='reg:pseudohubererror', eval_metric='mphe')
    elif isinstance(obj, tuple):
        prm.update(objective='reg:quantileerror', quantile_alpha=obj[1], eval_metric='quantile')
    else:
        prm.update(objective='reg:squarederror', eval_metric='rmse')
    if any(mono):
        prm['monotone_constraints'] = '(' + ','.join(map(str, mono)) + ')'
    return prm


def _lprm(p, obj, seed, mono):
    prm = dict(objective='gamma' if obj == 'gamma' else ('huber' if obj == 'huber' else 'regression'),
               metric='gamma' if obj == 'gamma' else ('huber' if obj == 'huber' else 'l2'), boosting_type='gbdt', max_depth=int(p.get('max_depth', -1)),
               num_leaves=int(p['num_leaves']), learning_rate=p['learning_rate'], bagging_fraction=p['bagging_fraction'], bagging_freq=1,
               feature_fraction=p['feature_fraction'], min_data_in_leaf=int(p['min_data_in_leaf']), reg_alpha=p['reg_alpha'], reg_lambda=p['reg_lambda'],
               max_bin=MAX_BIN, verbosity=-1, seed=seed, num_threads=1, force_col_wise=True)
    if isinstance(obj, tuple):
        prm.update(objective='quantile', alpha=obj[1], metric='quantile')
    if any(mono):
        prm['monotone_constraints'] = mono
    return prm


class Fit:
    """Одна модель: m, параметры, цель. labels: (ytr, yva) в пространстве обучения; off_*: смещение (ln ноги) — для l2/huber/quantile
    метка = ln y − off, прогноз = raw + off; для gamma с офсетом — base_margin / init_score = off (как l2_lib.fit, bm = gamma и офсет)."""

    def __init__(self, m, p, obj, seed, mono, cats, has_off):
        self.m, self.p, self.obj, self.seed, self.mono, self.cats, self.bm = m, p, obj, seed, mono, cats, (obj == 'gamma' and has_off)
        self.b, self.n = None, None

    def _ft(self, X):
        return ['c' if i in self.cats else 'q' for i in range(X.shape[1])]

    def labels(self, y, off):
        if self.obj == 'gamma':
            return np.clip(y, 1e-8, None)
        return np.log(y + 1e-10) - off

    def _key(self, ck, lab, w, off):
        return _dc().ds_key(**ck, lib=self.m, lab=_ah(lab), w=_ah(w), init=_ah(off if self.bm else None), cats=list(self.cats),
                            max_bin=MAX_BIN)

    def train(self, X, y, off, w, ref, Xv=None, yv=None, offv=None, nmax=NMAX, es=True, ytr_lab=None, init=None, ck=None, stop=None):
        """ck — части ключа кэша наборов (fit_step); None — набор строится здесь (как R1). ref может быть функцией (ленивые бины шага 1).
        stop(n) -> bool — колбэк после n-го дерева (es = False): self.b — обучаемая модель; True — обучение прекращается (es_search)."""
        lab = self.labels(y, off) if ytr_lab is None else ytr_lab
        if ck is None and callable(ref):
            ref = ref()
        if self.m == 'xgb':
            kw = dict(feature_types=self._ft(X), enable_categorical=bool(self.cats))
            if ck is not None:
                dtr = _dc().xgb_qdm(self._key(ck, lab, w, off), lambda: dict(X=X, y=lab, weight=w, base_margin=off if self.bm else None),
                                    ref=ref() if callable(ref) else ref, max_bin=MAX_BIN, **kw)
            else:
                dtr = xgb.QuantileDMatrix(X, lab, weight=w, base_margin=off if self.bm else None, max_bin=MAX_BIN, ref=ref, **kw)
            prm = _xprm(self.p, self.obj, self.seed, self.mono, lab, self.bm)
            if es:
                dva = xgb.QuantileDMatrix(Xv, self.labels(yv, offv), base_margin=offv if self.bm else None, ref=dtr, max_bin=MAX_BIN, **kw)
                b = xgb.train(prm, dtr, nmax, evals=[(dva, 'val')], early_stopping_rounds=PAT, verbose_eval=False)
                self.n = int(b.best_iteration) + 1
            else:
                cbs = None
                if stop is not None:
                    fit = self

                    class _Stop(xgb.callback.TrainingCallback):
                        def after_iteration(self, model, epoch, evals_log):
                            fit.b = model
                            return bool(stop(epoch + 1))
                    cbs = [_Stop()]
                b = xgb.train(prm, dtr, nmax, verbose_eval=False, xgb_model=init, callbacks=cbs)
                self.n = int(b.num_boosted_rounds())
            self.b = b
            return self
        dprm = {'max_bin': MAX_BIN, 'feature_pre_filter': False, 'verbosity': -1}
        if ck is not None:
            dtr = _dc().lgb_dataset(self._key(ck, lab, w, off),
                                    lambda: dict(X=X, y=lab, weight=w, init_score=off if self.bm else None,
                                                 reference=ref() if callable(ref) else ref, categorical_feature=self.cats or 'auto'),
                                    params=dprm)
            if self.bm:   # init_score из save_binary не восстанавливается (контроль kesh, offset) — ставится явно
                dtr.set_init_score(off)
        else:
            dtr = lgb.Dataset(X, lab, weight=w, init_score=off if self.bm else None, reference=ref, params=dprm,
                              categorical_feature=self.cats or 'auto', free_raw_data=False)
        prm = _lprm(self.p, self.obj, self.seed, self.mono)
        if es:
            dva = lgb.Dataset(Xv, self.labels(yv, offv), init_score=offv if self.bm else None, reference=dtr,
                              categorical_feature=self.cats or 'auto', free_raw_data=False)
            b = lgb.train(prm, dtr, nmax, valid_sets=[dva], callbacks=[lgb.early_stopping(PAT, verbose=False)])
            self.n = int(b.best_iteration)
        else:
            cbs = None
            if stop is not None:
                def _cb(env):
                    self.b = env.model
                    if stop(env.iteration - env.begin_iteration + 1):
                        raise lgb.callback.EarlyStopException(env.iteration, [])
                cbs = [_cb]
            b = lgb.train(prm, dtr, nmax, init_model=init, callbacks=cbs)
            self.n = int(b.current_iteration())
        self.b = b
        return self

    def raw(self, X, off, n=None):
        """ln-прогноз (уровень ln): raw + off; n — первые n деревьев."""
        n = self.n if n is None else n
        if self.m == 'xgb':
            dm = xgb.DMatrix(X, feature_types=self._ft(X), enable_categorical=bool(self.cats), base_margin=off if self.bm else None)
            r = self.b.predict(dm, iteration_range=(0, n), output_margin=True)
            return r if self.bm else r + off
        r = self.b.predict(X, raw_score=True, num_iteration=n)
        return r + off


def ref_bins(m, X1, lab1, cats):
    """Бины окна шага 1 (как l2_lib.make_ref) — reference для шагов >= 2."""
    if m == 'xgb':
        return xgb.QuantileDMatrix(X1, lab1, max_bin=MAX_BIN, feature_types=['c' if i in cats else 'q' for i in range(X1.shape[1])],
                                   enable_categorical=bool(cats))
    return lgb.Dataset(X1, lab1, params={'max_bin': MAX_BIN, 'feature_pre_filter': False, 'verbosity': -1},
                       categorical_feature=cats or 'auto', free_raw_data=False)


def profile_q(y, p):
    """QLIKE при оптимальном множителе c* = mean(y/p) (инвариантно к постоянной поправке s²)."""
    m = y > 1e-12
    c = float(np.mean(y[m] / p[m]))
    return qlike(y[m], c * p[m])


ES_CB = os.environ.get('R2_B_ES_CB', '1') != '0'


def es_search(fit, train_call, qfun, pat):
    """Число деревьев по QLIKE итога на внутренней складке, patience pat. ES_CB (по умолчанию): обучение останавливается колбэком на
    том же n, на котором прежний цикл по готовой модели из nmax деревьев прервался бы, — деревья 1..n не зависят от того, сколько
    обучается после, поэтому n и ряды те же (контроль kesh: режим medl = прежний путь), но без лишних деревьев сверх best + pat.
    R2_B_ES_CB=0 — прежний путь."""
    st = dict(best=np.inf, bn=1, since=0)

    def step(n):
        q = qfun(n)
        if q < st['best'] - 1e-12:
            st.update(best=q, bn=n, since=0)
        else:
            st['since'] += 1
        return st['since'] >= pat
    if ES_CB:
        train_call(step)
    else:
        train_call(None)
        for n in range(1, fit.n + 1):
            if step(n):
                break
    return st['bn']


def es_qlike_total(fit, Xi, yi, offi, wi, ref, Xf, yf, offf, nmax, pat, ck=None):
    """Внутренняя складка: обучение ≤ nmax деревьев, число деревьев — по profile-QLIKE итога exp(raw + off) на складке, patience pat."""
    return es_search(fit, lambda stop: fit.train(Xi, yi, offi, wi, ref, nmax=nmax, es=False, ck=ck, stop=stop),
                     lambda n: profile_q(yf, np.exp(fit.raw(Xf, offf, n))), pat)


# ---------------------------------------------------------------- одна модель шага (одна lib)
def objective_of(d, role):
    if role == 'offset':
        return 'gamma'
    return {'l2_log': 'l2', 'huber_log': 'huber', 'gamma_level': 'gamma'}[d['boost']['objective']]


def fit_step(ctx, d, role, m, step, obj=None, label_fn=None):
    """-> dict(f_va — поправка к ln ноги на val, legva, va, n, extra): одна библиотека m, шаг step."""
    h = ctx['h']
    seed = int(d['seed'])
    tr, va, vy = split(ctx, d, step)
    legtr, legva = leg_train(ctx, d, step, tr), leg_ln(ctx, step, va, 'va')
    assert not np.isnan(legva).any(), ('нет OOS ноги на val', step)
    ok = ~np.isnan(legtr)
    tr, legtr = tr[ok], legtr[ok]
    A = ctx['A']
    ytr, yva = A[_tc(ctx)].values[tr], A[_tc(ctx)].values[va]
    obj = obj or objective_of(d, role)
    use_lhs = d['boost']['grid'] == 'resid_small_v1' and m == 'lgbm'
    gp = str(d['boost'].get('grid_point', 'none'))   # schema_v2: p01…p60, p61_tekushchaya (= текущая точка); иначе — ctx
    pt = int(gp[1:3]) if gp.startswith('p') else ctx.get('_lhs_point', 61)
    p = lhs_point(pt) if use_lhs and pt != 61 else points(h, role, m, step)
    Xtr, names, mono, cats = design_X(ctx, d, role, m, step, tr, legtr)
    Xva = design_X(ctx, d, role, m, step, va, legva)[0]
    has_off = role in ('residual', 'residual_std', 'offset') or (obj == 'gamma' and role != 'feature')
    if role in ('residual', 'offset') or (role == 'residual_std' and obj == 'gamma'):
        offtr, offva = legtr.copy(), legva.copy()
    else:
        offtr, offva = np.zeros(len(tr)), np.zeros(len(va))
    if role == 'offset':  # Д17.1: init_score = ln ноги + s²/2, s² — остатки ln ноги на обучении
        s2 = float(np.var(np.log(np.clip(ytr, 1e-10, None)) - legtr))
        offtr, offva = offtr + s2 / 2, offva + s2 / 2
    lab = None
    sgtr = sgva = None
    if role == 'residual_std' and obj != 'gamma':
        sgtr, sgva = sigma_t(ctx, step, tr, va, legtr, legva, h)
        lab = (np.log(ytr + 1e-10) - legtr) / sgtr
    elif d['data']['winsor'] == 'p1_99_by_ticker' and obj != 'gamma':
        ln = np.log(ytr + 1e-10)
        q = pd.DataFrame({'t': A['ticker'].values[tr], 'y': ln}).groupby('t')['y'].quantile([.01, .99]).unstack()
        tk = A['ticker'].values[tr]
        lab = np.clip(ln, pd.Series(tk).map(q[0.01]).values, pd.Series(tk).map(q[0.99]).values) - offtr
    w = weights(ctx, d, tr)
    # кэш наборов: части ключа, общие для наборов шага (см. DS_ON)
    rs = d['boost']['resid_source']
    ck0 = None
    if DS_ON and ctx.get('_afp') and (rs == 'in_sample' or ctx.get('_refit_id')):
        if '_legfp' not in ctx:
            ctx['_legfp'] = leg_fingerprint(ctx['leg'])
        ck0 = dict(step=step, features=names, h=h, target=f'{_tc(ctx)}|{role}|{obj}', A=ctx['_afp'], leg=ctx['_legfp'],
                   resid_source=rs, refit=ctx.get('_refit_id') if rs == 'cross_fit' else None, ref=None)
    # бины окна шага 1 (R1) — для шагов >= 2
    ref = None
    if step > 1:
        tr1, _, _ = split(dict(ctx, _half=False), d, 1)   # бины — по полному окну шага 1 (как make_ref при подборе на 50%)

        def mkref():
            l1 = leg_ln(ctx, 1, tr1, 'tr') if rs == 'in_sample' else cross_fit(ctx, 1, tr1, h)
            o1 = ~np.isnan(l1)
            X1 = design_X(ctx, d, role, m, 2, tr1[o1], l1[o1])[0]   # признаки как на шагах >= 2 (исключения шага 2)
            return ref_bins(m, X1, np.zeros(int(o1.sum())), cats)
        if ck0 is None:
            ref = mkref()
        else:
            # бины шага 1 определяются строками шага 1, ногой (и оценщиком cross_fit) и именами признаков шага 2
            ck0['ref'] = dict(rows1=_ah(tr1), names2=design_names(ctx, d, role, m, 2)[1], lib=m)
            if m == 'lgbm':
                ref = mkref        # LightGBM: бины лежат в бинарном наборе — reference строится только при промахе кэша
            else:                  # XGBoost: QuantileDMatrix требует ref всегда — память процесса по ключу бинов
                rk = json.dumps(dict(ck0['ref'], A=ck0['A'], leg=ck0['leg'], rs=rs, refit=ck0['refit'], h=h, role=role), sort_keys=True)
                if rk not in _REFM:
                    _REFM[rk] = mkref()
                ref = _REFM[rk]
    ck = None if ck0 is None else dict(ck0, rows=_ah(tr), legv=_ah(legtr), part='final')
    fit = Fit(m, p, obj if not isinstance(obj, tuple) else obj, seed, mono, cats, has_off)
    nmax = int(YB['lhs']['nmax']) if use_lhs else NMAX
    if d['boost']['early_stop'] == 'native':
        if lab is not None:
            labva = (np.log(yva + 1e-10) - legva) / sgva if sgva is not None else None
            if labva is None:
                fit.train(Xtr, ytr, offtr, w, ref, Xva, yva, offva, nmax=nmax, ytr_lab=lab, ck=ck)
            else:  # residual_std: ES на val-годе по стандартизованной цели
                fit = _train_std(fit, Xtr, lab, w, ref() if callable(ref) else ref, Xva, labva, nmax)
        else:
            fit.train(Xtr, ytr, offtr, w, ref, Xva, yva, offva, nmax=nmax, ck=ck)
    else:  # qlike_total: внутренняя складка vy−1, затем обучение на годах < vy с найденным числом деревьев
        ti, fi, _ = split(ctx, d, step, before=vy - 1)
        # нога шага t−1 (обучена на годах < vy−1): внутри выборки на обучении складки, OOS на годе vy−1; на шаге 1 (складка 2015,
        # вне метрик) шага 0 нет — нога шага 1 внутри выборки
        li = leg_train(ctx, d, step - 1, ti) if step > 1 else leg_train(ctx, d, step, ti)
        lf = leg_ln(ctx, step - 1, fi, 'va') if step > 1 else leg_ln(ctx, step, fi, 'tr')
        oki = ~np.isnan(li)
        ti, li = ti[oki], li[oki]
        Xi, Xf = design_X(ctx, d, role, m, step, ti, li)[0], design_X(ctx, d, role, m, step, fi, lf)[0]
        yi, yf = A[_tc(ctx)].values[ti], A[_tc(ctx)].values[fi]
        oi = li.copy() if np.any(offtr) else np.zeros(len(ti))
        of = lf.copy() if np.any(offtr) else np.zeros(len(fi))
        if role == 'offset':
            s2i = float(np.var(np.log(np.clip(yi, 1e-10, None)) - li))
            oi, of = oi + s2i / 2, of + s2i / 2
        nm = int(YB['lhs']['nmax']) if use_lhs else 300
        cki = None if ck0 is None else dict(ck0, rows=_ah(ti), legv=_ah(li), part='inner')
        if role == 'residual_std' and obj != 'gamma':
            sgi, sgf = sigma_t(ctx, step, ti, fi, li, lf, h)
            labi = (np.log(yi + 1e-10) - li) / sgi
            fi_fit = Fit(m, p, obj, seed, mono, cats, has_off)
            wi, zf = weights(ctx, d, ti), np.zeros(len(Xf))
            n = es_search(fi_fit, lambda stop: fi_fit.train(Xi, yi, np.zeros(len(ti)), wi, ref, nmax=nm, es=False, ytr_lab=labi, ck=cki,
                                                            stop=stop),
                          lambda k: profile_q(yf, np.exp(lf + sgf * fi_fit.raw(Xf, zf, k))), int(YB['lhs']['patience']))
        else:
            fi_fit = Fit(m, p, obj, seed, mono, cats, has_off)
            n = es_qlike_total(fi_fit, Xi, yi, oi, weights(ctx, d, ti), ref, Xf, yf, of, nm, int(YB['lhs']['patience']), ck=cki)
        fit.train(Xtr, ytr, offtr, w, ref, nmax=n, es=False, ytr_lab=lab, ck=ck)
    # прогноз val (yearly) или по подпериодам (boost_refit)
    if d['boost']['boost_refit'] == 'yearly':
        r = fit.raw(Xva, offva if sgva is None else np.zeros(len(va)))
        f = (r * sgva) if sgva is not None else (r - legva)
    else:
        f = refit_periods(ctx, d, fit, role, m, step, tr, va, Xtr, ytr, offtr, lab, w, ref, Xva, offva, legva, sgva)
    # детектор, граница года (r2_core.leak_check): прогноз итоговой модели шага на строках года vy−1 (part = all)
    allr = None
    if ctx.get('_emit_all') and d['boost']['boost_refit'] == 'yearly' and sgva is None:
        ia = np.flatnonzero(A['year'].values == vy - 1)
        la = leg_ln(ctx, step, ia, 'tr')
        okk = ~np.isnan(la)
        ia, la = ia[okk], la[okk]
        oa = (la + (offtr[0] - legtr[0] if len(tr) and np.any(offtr) else 0.0)) if np.any(offtr) else np.zeros(len(ia))
        allr = (ia, fit.raw(design_X(ctx, d, role, m, step, ia, la)[0], oa) - la, la)
    # clip и shrink поправки
    sres = float(np.std(np.log(ytr + 1e-10) - legtr))
    k_ = float(d['boost']['clip_sigma']) if d['boost']['clip_sigma'] != 'off' else None
    post = lambda x: float(d['boost']['shrink']) * (np.clip(x, -k_ * sres, k_ * sres) if k_ else x)
    f = post(f)
    if allr is not None:
        allr = (allr[0], post(allr[1]), allr[2])
    return dict(f=f, legva=legva, va=va, n=fit.n, tr=tr, legtr=legtr, Xtr=Xtr, Xva=Xva, ytr=ytr, fit=fit, ref=ref, all=allr)


def _train_std(fit, Xtr, lab, w, ref, Xva, labva, nmax):
    """ES на val-годе по стандартизованной цели (residual_std, native): метка уже готова, смещения нет."""
    z = np.zeros(len(Xtr))
    if fit.m == 'xgb':
        kw = dict(feature_types=fit._ft(Xtr), enable_categorical=bool(fit.cats))
        dtr = xgb.QuantileDMatrix(Xtr, lab, weight=w, max_bin=MAX_BIN, ref=ref, **kw)
        dva = xgb.QuantileDMatrix(Xva, labva, ref=dtr, max_bin=MAX_BIN, **kw)
        fit.b = xgb.train(_xprm(fit.p, fit.obj, fit.seed, fit.mono, lab, False), dtr, nmax, evals=[(dva, 'val')], early_stopping_rounds=PAT,
                          verbose_eval=False)
        fit.n = int(fit.b.best_iteration) + 1
    else:
        dtr = lgb.Dataset(Xtr, lab, weight=w, reference=ref, params={'max_bin': MAX_BIN, 'feature_pre_filter': False, 'verbosity': -1},
                          categorical_feature=fit.cats or 'auto', free_raw_data=False)
        dva = lgb.Dataset(Xva, labva, reference=dtr, categorical_feature=fit.cats or 'auto', free_raw_data=False)
        fit.b = lgb.train(_lprm(fit.p, fit.obj, fit.seed, fit.mono), dtr, nmax, valid_sets=[dva], callbacks=[lgb.early_stopping(PAT, verbose=False)])
        fit.n = int(fit.b.best_iteration)
    del z
    return fit


# ---------------------------------------------------------------- кластер 15: boost_refit внутри val-года
def refit_periods(ctx, d, fit, role, m, step, tr, va, Xtr, ytr, offtr, lab, w, ref, Xva, offva, legva, sgva):
    """monthly — заново на строках, окна целей которых закончились до начала месяца, n деревьев годовой модели;
    quarterly_warm — годовая модель + max(10, n/10) деревьев на тех же строках. Строки val-года до начала периода — со своим OOS-прогнозом ноги."""
    A, h = ctx['A'], ctx['h']
    dva = pd.to_datetime(A['date'].values[va])
    per = dva.to_period('M') if d['boost']['boost_refit'] == 'monthly' else dva.to_period('Q')
    out = np.full(len(va), np.nan)
    yva = A[_tc(ctx)].values[va]
    labva_all = None if lab is None else (np.log(yva + 1e-10) - legva) / sgva if sgva is not None else np.log(yva + 1e-10) - offva
    import r2_core as C
    tend_va = pd.Series(C._tend(A, h).values[va])
    for k, P in enumerate(sorted(per.unique())):
        cur = (per == P)
        start = P.start_time
        prior = np.flatnonzero((tend_va < start).fillna(False).values)   # эмбарго: окно цели кончилось до начала периода
        if k == 0 or not len(prior):
            r = fit.raw(Xva[cur], offva[cur] if sgva is None else np.zeros(int(cur.sum())))
        else:
            X2 = np.vstack([Xtr, Xva[prior]])
            y2 = np.concatenate([ytr, yva[prior]])
            o2 = np.concatenate([offtr, offva[prior]])
            l2 = None if lab is None else np.concatenate([lab, labva_all[prior]])
            w2 = None if w is None else np.concatenate([w, np.ones(len(prior))])
            g = Fit(fit.m, fit.p, fit.obj, fit.seed, fit.mono, fit.cats, fit.bm or bool(np.any(offtr)))
            g.bm = fit.bm
            if d['boost']['boost_refit'] == 'quarterly_warm' and g.bm and g.m == 'lgbm':
                raise NotImplementedError('quarterly_warm + gamma с офсетом в LightGBM: init_model затирает init_score офсета')
            if d['boost']['boost_refit'] == 'monthly':
                g.train(X2, y2, o2, w2, ref, nmax=fit.n, es=False, ytr_lab=l2)
            else:
                g.train(X2, y2, o2, w2, ref, nmax=max(10, fit.n // 10), es=False, ytr_lab=l2, init=fit.b)
            r = g.raw(Xva[cur], offva[cur] if sgva is None else np.zeros(int(cur.sum())))
        out[cur] = r * sgva[cur] if sgva is not None else r - legva[cur]
    return out


# ---------------------------------------------------------------- кластер 12: вероятностный гибрид
def prob_scale(ctx, d, role, step, base):
    """scale_boost: LightGBM масштаба на ln(e²) остатков ноги (e = ln RV − ln ноги, строки обучения в окне prob_window);
    поправка ln ŷ += ln(mean_i exp(ŝ_t z_i) / mean_i exp(s̄ z_i)), z_i = e_i/ŝ_i (empirical) или t-распределение (student).
    Только относительная (по состоянию) — общий уровень s² ставит калибровка кластера 3."""
    A, h = ctx['A'], ctx['h']
    tr, legtr, Xtr, Xva = base['tr'], base['legtr'], base['Xtr'], base['Xva']
    e = np.log(A[_tc(ctx)].values[tr] + 1e-10) - legtr
    if d['boost']['prob_window'] == '2y':
        vy = 2015 + step
        k = A['year'].values[tr] >= vy - 2
        Xs, es_ = Xtr[k], e[k]
    else:
        Xs, es_ = Xtr, e
    prm = dict(objective='regression', learning_rate=0.03, num_leaves=8, min_data_in_leaf=200, feature_fraction=0.5, bagging_fraction=0.8,
               bagging_freq=1, max_bin=MAX_BIN, verbosity=-1, seed=int(d['seed']), num_threads=1, force_col_wise=True)
    b = lgb.train(prm, lgb.Dataset(Xs, np.log(es_ ** 2 + 1e-10), params={'max_bin': MAX_BIN, 'verbosity': -1}), 200)
    s_tr = np.sqrt(np.exp(b.predict(Xs) + 1.2704))   # E[ln χ²₁] = −1,2704
    s_va = np.sqrt(np.exp(b.predict(Xva) + 1.2704))
    z = es_ / s_tr
    sbar = float(np.mean(s_tr))
    if d['boost']['prob_dist'] == 'empirical':
        zz = np.quantile(z, np.linspace(0.0005, 0.9995, 1000))
        F = lambda s: np.mean(np.exp(np.outer(s, zz)), axis=1)
    else:
        from scipy import stats
        df_, loc, sc = stats.t.fit(z)
        grid = np.linspace(-6, 6, 2401)
        pdf = stats.t.pdf(grid, df_, loc, sc)
        pdf /= pdf.sum()
        F = lambda s: np.exp(np.outer(s, grid)) @ pdf
    return np.log(F(s_va)) - float(np.log(F(np.array([sbar]))[0]))


def prob_quantile5(ctx, d, role, m, step):
    """quantile5: пять квантильных LightGBM (α = 0,1; 0,3; 0,5; 0,7; 0,9) на ln RV − ln ноги; ln ŷ = ln ноги + ln mean_k exp(q̂_k)."""
    qs = []
    base = None
    for a in (0.1, 0.3, 0.5, 0.7, 0.9):
        dd = dict(d, boost=dict(d['boost'], role='residual'))
        r = fit_step(ctx, dd, 'residual', 'lgbm', step, obj=('quantile', a))
        qs.append(r['f'])
        base = r
    f = np.log(np.mean(np.exp(np.vstack(qs)), axis=0))
    return f, base


# ---------------------------------------------------------------- дизайн целиком
def build(design, ctx, steps=range(1, 7)):
    import r2_core as C
    d = full(C.validate(design) if not ctx.get('_r1') else design)   # _r1 — контроль движка на матрице v3 (без схемных признаков)
    ctx.setdefault('h', d['h'])
    assert ctx['h'] == d['h']
    prep_leg(ctx)
    role = d['boost']['role']
    A = ctx['A']
    out = []
    for step in steps:
        if role == 'none':
            _, va, vy = split(ctx, d, step)
            ln = leg_ln(ctx, step, va, 'va')
        elif d['boost']['prob'] == 'quantile5':
            f, base = prob_quantile5(ctx, d, role, 'lgbm', step)
            va, ln = base['va'], base['legva'] + f
        else:
            libs = {'lightgbm': ['lgbm'], 'xgboost': ['xgb'], 'both_avg': ['xgb', 'lgbm']}[d['boost']['lib']]
            res = [fit_step(ctx, d, role, m, step) for m in libs]
            va, legva = res[0]['va'], res[0]['legva']
            lvl = np.mean([np.exp(legva + r['f']) for r in res], axis=0)  # both_avg — среднее уровней (как wrap2)
            ln = np.log(lvl)
            if d['boost']['prob'] == 'scale_boost':
                ln = ln + prob_scale(ctx, d, role, step, res[-1])
            if ctx.get('_emit_all') and all(r['all'] is not None for r in res) and d['boost']['prob'] == 'off':
                ia, la = res[0]['all'][0], res[0]['all'][2]
                lv = np.mean([np.exp(la + r['all'][1]) for r in res], axis=0)
                out.append(pd.DataFrame({'date': A['date'].values[ia], 'ticker': A['ticker'].values[ia], 'year': A['year'].values[ia], 'step': step,
                                         'rv_actual': A[_tc(ctx)].values[ia], 'rv_pred': np.clip(lv, 1e-10, None), 'part': 'all'}))
        out.append(pd.DataFrame({'date': A['date'].values[va], 'ticker': A['ticker'].values[va], 'year': A['year'].values[va], 'step': step,
                                 'rv_actual': A[_tc(ctx)].values[va], 'rv_pred': np.clip(np.exp(ln), 1e-10, None)}))
        if ctx.get('_emit_all'):
            out[-1]['part'] = 'va'
    o = pd.concat(out, ignore_index=True)
    assert o.rv_pred.notna().all() and (o.year <= 2021).all()
    return o


# ---------------------------------------------------------------- дым / контроль / детекторы
def load_groups(A, h):
    """G0 — признаки features_pit_v3 (config feature_cols + l2_lib.build_features) на матрице d0 ядра; G1–G10 — файл F
    (features_r2_v1.parquet, колонки групп — r2_F_features.yaml). -> (A с признаками, {группа: колонки})."""
    import l2_lib as L
    cfg = json.load(open(R / 'features_pit_v3.config.json'))
    FC = [c for c in cfg['feature_cols'] if c in A.columns]
    A2, g0, _, _ = L.build_features(A, FC, h, {})
    FY = yaml.safe_load(open(REPO / 'models' / 'designs' / 'r2' / 'r2_F_features.yaml'))
    groups = {'G0': list(g0)}
    cols = []
    for g, spec in FY['groups'].items():
        if g != 'G0':
            groups[g] = list(spec['columns'])
            cols += groups[g]
    fp = ROOT / FY['combined']['file']   # features_r2_v2 (PR #170): файл и SHA — из yaml F
    if '_fsha' not in _EXC:
        import r2_core as C
        _EXC['_fsha'] = C.sha_file(fp)
        assert _EXC['_fsha'] == FY['combined']['sha256'], f'{fp.name}: SHA не совпал с r2_F_features.yaml'
    F = pd.read_parquet(fp, columns=['date', 'ticker'] + cols)
    F['date'] = pd.to_datetime(F['date'])
    F = F[F['date'].dt.year <= 2021]
    A2 = A2.assign(date=pd.to_datetime(A2['date'])).merge(F, on=['date', 'ticker'], how='left', validate='one_to_one')
    return A2, groups


def d0_ctx(h, legname=None, tickers=None):
    """Контекст как у ядра: r2_core.load_d0 (d0, строки ≤ 2021, _tend{h}) + признаки G0–G10 + нога L (lnall)."""
    import r2_core as C
    sm = YB['smoke']
    A, harj = C.load_d0(tickers=None if tickers is False else (tickers or sm['tickers']))
    A, groups = load_groups(A, h)
    legname = legname or sm['legs']['main']
    leg = pd.read_parquet(R / 'R2' / 'cache' / 'E' / f'{legname}_h{h}_lnall.parquet')
    leg = leg[leg.ticker.isin(set(A.ticker))]
    ctx = dict(A=A, h=h, groups=groups, leg=leg, harj=harj)
    ctx['leg_refit'] = leg_refit_harj_ols(ctx)
    return ctx


def core_fn(design, seed, steps):
    """Для r2_core.run_phase: fn(design, seed, steps) -> {ряд: DataFrame}. Данные — r2_core.SHARED (кладёт вызывающий до пула):
    A (load_d0 + load_groups), B_groups, leg_provider(design, h) -> lnall ноги дизайна (L), leg_refit_provider(design, h, A) -> f (L; для cross_fit),
    B_half — скрининг на 50% строк обучения (владелец 16:25; имя ряда с суффиксом _half).
    Режим владельца 16:25: скрининг — 1 seed, lib = lightgbm, B_half = True; уровень 2 — 3 seed, LightGBM (r2_resid_grid.core_fn);
    финалисты — 10 seed, lib = both_avg. num_threads = 1 в каждой модели; воркеры — r2_core.run_pool (fork).
    ❓ (для S) бинарные Dataset / QuantileDMatrix из r2_dataset_cache.py: модуля в базе нет; до него Dataset строится в воркере."""
    import r2_core as C
    S = C.SHARED
    d = dict(C.validate(design), seed=int(seed))
    h = int(d['h'])
    legs = S['leg_provider'](d, h)
    if not isinstance(legs, dict):
        legs = {'base': legs}
    out = {}
    for tag, leg in legs.items():   # варианты ноги (сетка L внутри складки) -> ряды B / B__<тег>
        ctx = dict(A=S['A'], h=h, groups=S['B_groups'], leg=leg, _half=bool(S.get('B_half', False)), _rowmask=S.get('B_rowmask'),
                   _afp=S.get('B_afp'))
        if S.get('B_lhs_point') is not None:
            ctx['_lhs_point'] = S['B_lhs_point']
        if d['boost']['resid_source'] == 'cross_fit':
            ctx['leg_refit'] = S['leg_refit_provider'](d, h, S['A'], None if tag == 'base' else tag)
            # оценщик cross_fit (r2_nogi.fit_predict) задан разделом leg дизайна и тегом варианта — в ключ бинов шага 1
            ctx['_refit_id'] = hashlib.sha256(json.dumps(dict(leg=d.get('leg'), tag=tag, est='r2_nogi.fit_predict'), sort_keys=True,
                                                         default=str).encode()).hexdigest()[:32]
        nm = S.get('B_series', 'B') + ('' if tag == 'base' else f'__{tag}') + ('_half' if ctx['_half'] else '')
        out[nm] = build(d, ctx, steps=steps)
    return out


def leg_from_cache(design, h):
    """Ноги L в каталоге дизайна (цепочка L -> B, r2_nogi.fn_leg): leg_h{h}_lnall.parquet (вариант base) и leg__<тег>_h{h}_lnall.parquet."""
    import r2_core as C
    cd = C.cache_dir(design, create=False, root=C.SHARED.get('B_root'))
    fs = sorted(cd.glob(f'leg*_h{h}_lnall.parquet'))
    assert fs, f'нет ног L (leg*_h{h}_lnall) в {cd} — модуль L должен идти раньше B'
    out = {}
    for f in fs:
        stem = f.name[:-len(f'_h{h}_lnall.parquet')]
        out['base' if stem == 'leg' else stem.split('__', 1)[1]] = pd.read_parquet(f)
    return out


def leg_refit_L(design, h, A, tag):
    """cross_fit: нога, обученная на строках fit_idx (r2_nogi.fit_predict — тот же оценщик L), прогноз на pred_idx. M — mat_from L,
    строки M['A'] совпадают со строками A модуля (проверяется в run)."""
    import r2_core as C
    import r2_nogi as RN
    M = C.SHARED['M']
    n = len(M['A'])

    def f(step, fit_idx, pred_idx):
        tm = np.zeros(n, bool); tm[fit_idx] = True
        pm = np.zeros(n, bool); pm[pred_idx] = True
        s = RN.fit_predict(design, tm, pm, M=M, tag=tag)
        return s.reindex(np.sort(pred_idx)).values[np.argsort(np.argsort(pred_idx))]
    return f


def prepare_run(h, row_frac=None, row_seed=None, tickers=None):
    """Данные один раз в родителе до пула: load_d0 -> mat_from (L) -> признаки G0–G10 (F v3); подвыборка строк; SHARED."""
    import r2_core as C
    import r2_nogi as RN
    A0, harj = C.load_d0(tickers=tickers)
    M = RN.mat_from(A0, harj)
    A, groups = load_groups(M['A'], h)
    assert len(A) == len(M['A']) and (A['ticker'].values == M['A']['ticker'].values).all() and \
        (pd.to_datetime(A['date']).values == pd.to_datetime(M['A']['date']).values).all(), 'строки A и M[A] разошлись'
    rm = None
    if row_frac is not None and float(row_frac) < 1:
        rm = np.random.default_rng(int(row_seed)).random(len(A)) < float(row_frac)
    afp = None
    if DS_ON:   # кэш наборов: отпечаток A в родителе до fork; ключ различает данные и тикеры (дым на 3 тикерах ≠ боевой счёт)
        afp = a_fingerprint(A)
        _dc().features_r2_sha()
    C.share(A=A, M=M, harj=harj, B_groups=groups, leg_provider=leg_from_cache, leg_refit_provider=leg_refit_L, B_rowmask=rm,
            B_series='B' if rm is None else f'B_rf{int(round(float(row_frac) * 100))}', B_afp=afp)
    log(f'B prepare_run: строк {len(A)}, кэш наборов {"вкл, корень " + str(_dc().cache_root()) if afp else "выкл"}')
    return A


def cmd_run(a):
    """Контракт драйвера S (r2_S_driver.yaml, modules.B): run --designs <yaml> --h --seeds --steps --workers [--row-frac --row-seed].
    Дизайны с boost.grid = resid_small_v1 — у r2_resid_grid (BG), здесь пропускаются; role = none — не модуль B."""
    import r2_core as C
    Y = yaml.safe_load(open(a.designs))
    ds = [d for d in (Y.get('designs') or []) if str(C.get(C.validate(d), 'boost.role')) != 'none'
          and str(C.get(C.validate(d), 'boost.grid')) != 'resid_small_v1']
    log(f'B run: h={a.h}, дизайнов {len(ds)} из {len(Y.get("designs") or [])}, seed {a.seeds}, шаги {a.steps}, воркеров {a.workers}, '
        f'row_frac {a.row_frac}')
    if not ds:
        return
    prepare_run(a.h, a.row_frac, a.row_seed, tickers=a.tickers)
    T = C.run_phase('B', a.h, ds, core_fn, seeds=tuple(a.seeds), steps=tuple(a.steps), workers=a.workers, module='B', check_gate=False)
    bad = T[T.ryad == 'ОШИБКА'] if len(T) else T
    log(f'B run: рядов {len(T) - len(bad)}, ошибок {len(bad)}')
    if len(bad):
        print(bad.to_string(), flush=True)
        raise SystemExit(1)


def cmd_leak_designs(a):
    """Контракт драйвера: leak --designs <yaml> --h — детектор ядра (r2_core.leak_check) на первом дизайне модуля B из файла
    (на 3 тикерах дыма; нога — HAR-J d0 с эмбарго, пересчитывается из испорченной матрицы; early_stop = qlike_total, без row_frac)."""
    import r2_core as C
    Y = yaml.safe_load(open(a.designs))
    ds = [d for d in (Y.get('designs') or []) if str(C.get(C.validate(d), 'boost.role')) != 'none']
    if not ds:
        log('leak B: дизайнов модуля B нет — OK'); return
    d0 = C.validate(ds[0])
    h = int(a.h)
    dz = dict(design='leak', h=h, seed=42, boost=dict(role=str(d0['boost']['role']), lib=str(d0['boost']['lib']), early_stop='qlike_total'),
              features=list(d0['features']))
    sm = YB['smoke']
    A0, harj = C.load_d0(tickers=sm['tickers'])
    A0, groups = load_groups(A0, h)

    def bld(A):
        A = A.copy()
        ctx = dict(A=A, h=h, groups=groups, harj=harj, leg=lnall_harj(A, harj, h, range(1, 7)), _emit_all=True)
        ctx['leg_refit'] = leg_refit_harj_ols(ctx)
        return {'B': build(dz, ctx, steps=range(1, 7))}
    Rr = C.leak_check(bld, A0, f'B leak h={h}', out=R / 'R2' / 'B' / f'leak_driver_h{h}.csv')
    log(f'leak B h={h}: {int(Rr.ok.sum())}/{len(Rr)} OK')
    if not Rr.ok.all():
        raise SystemExit(1)


def smoke_ctx(h, legname=None, tickers=None, matrix=None):
    """Контекст первого раунда (матрица v3, rv_d/w/m со сдвигом) — только для контроля движка с R1."""
    import l2_lib as L
    sm = YB['smoke']
    A, FC = L.load_base(R / (matrix or sm['matrix']))
    if tickers is not False:
        A = A[A.ticker.isin(tickers or sm['tickers'])].reset_index(drop=True)
    A2, cols, _, _ = L.build_features(A, FC, h, {})
    A2 = A2[A2.year <= 2021].reset_index(drop=True)   # годы >= 2022 в разработке не участвуют (шаги 1–6 их не используют)
    legname = legname or sm['legs']['main']
    leg = pd.read_parquet(R / 'R2' / 'cache' / 'E' / f'{legname}_h{h}_lnall.parquet')
    leg = leg[leg.ticker.isin(set(A2.ticker))]
    ctx = dict(A=A2, h=h, feat_cols=cols, leg=leg)
    ctx['leg_refit'] = leg_refit_harj_ols(ctx)
    return ctx


SMOKE_DESIGNS = [
    ('none', {}),
    ('feature', {}), ('residual', {}), ('offset', {}), ('residual_std', {}),
    ('feature', {'lib': 'both_avg', 'monotone': 'on', 'clip_sigma': 3, 'shrink': 0.75}),
    ('residual', {'resid_source': 'cross_fit', 'lib': 'xgboost'}),
    ('residual', {'early_stop': 'qlike_total', 'objective': 'huber_log'}),
    ('residual', {'grid': 'resid_small_v1', 'early_stop': 'qlike_total'}),
    ('feature', {'objective': 'gamma_level'}),
    ('residual', {'prob': 'scale_boost', 'prob_dist': 'student', 'prob_window': '2y'}),
    ('residual', {'prob': 'scale_boost'}),
    ('residual', {'prob': 'quantile5'}),
    ('residual', {'boost_refit': 'monthly'}),
    ('feature', {'boost_refit': 'quarterly_warm'}),
    ('residual_std', {'early_stop': 'qlike_total', 'resid_source': 'cross_fit'}),
    ('feature', {'lib': 'both_avg', '_features': ['G0', 'G1']}),                    # базовый дизайн скрининга h=1 (признаки G0 + G1)
    ('residual', {'lib': 'both_avg', 'clip_sigma': 3, 'resid_source': 'cross_fit', 'monotone': 'on', '_features': ['G0', 'G1', 'G2', 'G3']}),
]


def mk(role, bk, h, seed=42, name='smoke'):
    b = {k: v for k, v in bk.items() if not k.startswith('_')}
    return dict(design=f'{name}_{role}', h=h, seed=seed, boost=dict(role=role, **b), features=bk.get('_features', ['G0']))


def cmd_smoke(a):
    sm = YB['smoke']
    rows = []
    for h in a.hs or sm['hs']:
        ctx = d0_ctx(h)
        ctx['_lhs_point'] = 7
        for role, bk in SMOKE_DESIGNS:
            dz = mk(role, bk, h, sm['seed'])
            t0 = time.time()
            o = build(dz, ctx, steps=sm['steps'])
            q = {int(s): qlike(g.rv_actual.values[g.rv_actual > 1e-12], g.rv_pred.values[g.rv_actual > 1e-12]) for s, g in o.groupby('step') if s >= 2}
            rows.append(dict(h=h, role=role, keys=json.dumps(bk, ensure_ascii=False), rows=len(o), sec=round(time.time() - t0, 1),
                             q2017=round(q.get(2, np.nan), 4), q2018=round(q.get(3, np.nan), 4)))
            log(f'дым h={h} {role} {bk}: строк {len(o)}, {time.time() - t0:.1f} с, QLIKE 2017/2018 (3 тикера, без s²) {rows[-1]["q2017"]}/{rows[-1]["q2018"]}')
    p = R / 'R2' / 'B' / 'smoke_B.csv'
    p.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(p, index=False)
    log(f'дым: {len(rows)} дизайнов -> {p}')


def cmd_control(a):
    """Движок = l2_lib.fit первого раунда: роль feature, нога HAR-J r1 (lnall Э = l2_lib.har_ln), точки B v2b, все тикеры, шаги 1–3,
    сравнение с D12_h1_A_d812_{m}_h1_val (Д8.12) — max |Δ ln| ≤ 1e-7."""
    ctx = smoke_ctx(1, legname='HARJ', tickers=False)
    ctx['_no_embargo'] = True   # R1 обучал бустинг без эмбарго
    ctx['_r1'] = True           # матрица v3 и признаки R1 (feat_cols), без проверки схемы признаков
    YB['points'][1]['feature'] = dict(src='Bv2b', btag='Bv2b', leg_feature=True)  # точки D12_h1_A_d812 (только в этом контроле)
    rows = []
    for lib, m in (('xgboost', 'xgb'), ('lightgbm', 'lgbm')):
        dz = dict(design='control', h=1, seed=42, boost=dict(role='feature', lib=lib), data={})
        o = build(dz, ctx, steps=(1, 2, 3))
        S = pd.read_parquet(R / f'D12_h1_A_d812_{m}_h1_val.parquet')
        S = S[S.step.isin((1, 2, 3)) & (S.seed == 42) & (S['rank'] == 1)]
        mm = o.merge(S[['date', 'ticker', 'step', 'rv_pred']], on=['date', 'ticker', 'step'], how='outer', suffixes=('', '_s'), indicator=True)
        b = mm[mm._merge == 'both']
        dmax = float(np.abs(np.log(b.rv_pred) - np.log(b.rv_pred_s)).max())
        ok = bool(len(b) == len(mm) and dmax <= 1e-7)
        rows.append(dict(control='D12_h1_A_d812', model=m, rows=len(mm), matched=len(b), max_abs_dln=dmax, ok=ok, time=time.strftime('%F %T')))
        log(f'контроль движка {m}: строк {len(b)}/{len(mm)}, max|Δ ln| = {dmax:.2e} {"OK" if ok else "ПРОВАЛ"}')
    p = R / 'R2' / 'B' / 'control_B.csv'
    p.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(p, index=False)
    assert all(r['ok'] for r in rows), 'контроль движка не пройден'


def corrupt(A, leg, t0, seed=7):
    rng = np.random.default_rng(seed)
    A = A.copy()
    msk = (pd.to_datetime(A['date']) >= pd.Timestamp(t0)).values
    for c in [c for c in A.columns if c not in ('date', 'ticker', 'year') and pd.api.types.is_float_dtype(A[c])]:
        v = A.loc[msk, c].values
        A.loc[msk, c] = rng.permutation(v) if 'target' in c else v * np.exp(rng.normal(size=len(v)))
    g = leg.copy()
    mg = (pd.to_datetime(g['date']) >= pd.Timestamp(t0)).values
    g.loc[mg, 'base_ln'] = g.loc[mg, 'base_ln'].values + rng.normal(size=int(mg.sum()))
    return A, g


LEAK_DESIGNS = [('feature', {}), ('residual', {'early_stop': 'qlike_total'}), ('offset', {'early_stop': 'qlike_total'}),
                ('residual', {'boost_refit': 'monthly', 'early_stop': 'qlike_total'}), ('feature', {'boost_refit': 'quarterly_warm', 'early_stop': 'qlike_total'}),
                ('residual_std', {'early_stop': 'qlike_total'}), ('residual', {'prob': 'scale_boost', 'early_stop': 'qlike_total'})]


def lnall_harj(A, harj, h, steps):
    """Нога для детектора: HAR-J d0 с эмбарго (r2_core._harj_ln) в формате L: part tr — годы < vy (внутри выборки), va — год vy."""
    import r2_core as C
    L = C._harj_ln(A, harj, h, steps)
    fr = []
    yr = A['year'].values
    for k in steps:
        vy = 2015 + k
        for part, m in (('tr', yr < vy), ('va', yr == vy)):
            fr.append(pd.DataFrame({'date': A['date'].values[m], 'ticker': A['ticker'].values[m], 'step': k, 'part': part, 'base_ln': L[k][m]}))
    return pd.concat(fr, ignore_index=True)


CORE_LEAK = [('feature', {'early_stop': 'qlike_total'}), ('residual', {'early_stop': 'qlike_total', 'lib': 'both_avg'}),
             ('offset', {'early_stop': 'qlike_total'}), ('residual', {'early_stop': 'qlike_total', 'resid_source': 'cross_fit'}),
             ('residual', {'boost_refit': 'monthly', 'early_stop': 'qlike_total'}), ('residual_std', {'early_stop': 'qlike_total'}),
             ('residual', {'prob': 'scale_boost', 'early_stop': 'qlike_total'})]


def cmd_leak(a):
    """Детектор ядра r2_core.leak_check (Д26 / ВЫР-1) на коде ролей: build(A) пересчитывает всё из A — ногу HAR-J d0 с эмбарго,
    признаки, бустинг. Середина года: Δ = 0 до среза, Δ > 0 после, подлог «вход d := d+1» пойман; 31.12 — проверка эмбарго
    (прогноз итоговой модели шага на строках года vy−1, part = all; только yearly без prob). early_stop = qlike_total: native
    (остановка по val-году, определение R1) по построению видит val-год — для него детектор leak_leg со срезом 1 января."""
    import r2_core as C
    sm = YB['smoke']
    out = []
    for h in (a.hs or (1, 22)):
        A0, harj = C.load_d0(tickers=sm['tickers'])
        A0, groups = load_groups(A0, h)
        for role, bk in CORE_LEAK:
            dz = dict(design='leak', h=h, seed=42, boost=dict(role=role, **bk))

            def bld(A, dz=dz, h=h):
                A = A.copy()
                ctx = dict(A=A, h=h, groups=groups, harj=harj, leg=lnall_harj(A, harj, h, range(1, 7)), _emit_all=True)
                ctx['leg_refit'] = leg_refit_harj_ols(ctx)
                return {'B': build(dz, ctx, steps=range(1, 7))}
            t0 = time.time()
            Rr = C.leak_check(bld, A0, f'B {role} {json.dumps(bk)}')
            Rr['h'] = h
            out.append(Rr)
            log(f'leak(core) h={h} {role} {bk}: {int(Rr.ok.sum())}/{len(Rr)} OK, {time.time() - t0:.0f} с; '
                + '; '.join(f'{r.srez}: до {r.max_dln_do:.2g}, после {r.max_dln_posle:.2g}' + (f', подлог {r.podlog_max_dln_do:.2g}' if r.tip == 'середина года' else '')
                            for r in Rr.itertuples()))
    T = pd.concat(out, ignore_index=True)
    p = R / 'R2' / 'B' / 'leak_core_B.csv'
    p.parent.mkdir(parents=True, exist_ok=True)
    T.to_csv(p, index=False)
    assert T.ok.all(), 'детектор ядра: провал'


def cmd_leak_leg(a):
    """Д26 с порчей и ноги: срез t0; всё с t0 (матрица и нога) испорчено. qlike_total: срез в середине val-года — строки val с датой < t0
    обязаны совпасть (Δ = 0); native: ранняя остановка по val-году (определение R1) — срез 1 января следующего года. Подложенная утечка —
    признак «средняя RV за [d+1, d+h]» (будущее): обязана измениться."""
    rows = []
    step = 3
    for h in (1, 22):
        ctx0 = d0_ctx(h)
        for role, bk in LEAK_DESIGNS:
            native = bk.get('early_stop', 'native') == 'native'
            t0 = '2019-01-01' if native else '2018-07-02'
            for plant in (False, True):
                ctx = dict(ctx0)
                fut = lambda X: X.groupby('ticker')['rv'].transform(lambda s: s[::-1].rolling(h, min_periods=1).mean()[::-1].shift(-1)).values
                if plant:
                    A = ctx['A'].copy()
                    A['leak_fut'] = fut(A)
                    ctx['A'] = A
                    ctx['groups'] = dict(ctx['groups'], G0=ctx['groups']['G0'] + ['leak_fut'])
                dz = dict(design='leak', h=h, seed=42, boost=dict(role=role, **bk))
                base = build(dz, dict(ctx), steps=(step,))
                A2, g2 = corrupt(ctx['A'], ctx['leg'], t0)
                if plant:  # подложенный признак пересчитывается по испорченным данным (средняя RV за [d+1, d+h])
                    A2['leak_fut'] = fut(A2)
                c2 = dict(ctx, A=A2, leg=g2)
                c2['leg_refit'] = leg_refit_harj_ols(c2)
                c2.pop('_legtab', None)
                pert = build(dz, c2, steps=(step,))
                k = pd.to_datetime(base.date) < pd.Timestamp(t0)
                mm = base[k].merge(pert, on=['date', 'ticker', 'step'], how='left', suffixes=('', '_p'))
                dd = float(np.nanmax(np.abs(np.log(mm.rv_pred) - np.log(mm.rv_pred_p)))) if mm.rv_pred_p.notna().all() else np.inf
                ok = (dd > 0) if plant else (dd == 0.0)
                rows.append(dict(h=h, role=role, keys=json.dumps(bk), cut=t0, plant=plant, rows=int(k.sum()), max_abs_dln=dd, ok=bool(ok),
                                 time=time.strftime('%F %T')))
                log(f'leak h={h} {role} {bk} срез {t0} {"ПОДЛОГ" if plant else "чисто"}: строк {int(k.sum())}, max|Δ ln| = {dd:.3g} {"OK" if ok else "ПРОВАЛ"}')
    p = R / 'R2' / 'B' / 'leak_B.csv'
    pd.DataFrame(rows).to_csv(p, index=False)
    assert all(r['ok'] for r in rows), 'детектор: провал'


def cmd_leak_xfit(a):
    """Детектор cross_fit: OOF-прогноз ноги блока b не зависит от целей блока b и полос эмбарго h вокруг него (порча целей там -> Δ = 0);
    подлог — эмбарго 0 и блок b в обучении (in-sample) -> Δ > 0."""
    rows = []
    for h in (1, 5, 22):
        ctx = d0_ctx(h)
        d = full(dict(h=h, boost=dict(role='residual', resid_source='cross_fit'), data={}))
        tr, _, _ = split(ctx, d, 3)
        base = cross_fit(ctx, 3, tr, h)
        b = blocks(tr, ctx['A'])
        A = ctx['A']
        dts = np.sort(A['date'].unique())
        pos = {x: i for i, x in enumerate(dts)}
        di = np.array([pos[x] for x in A['date'].values[tr]])
        for k in range(5):
            inb = b == k
            lo, hi = di[inb].min(), di[inb].max()
            zone = (di >= lo - h) & (di <= hi + h)
            A2 = A.copy()
            tgt = A2[_tc(ctx)].values.copy()
            rng = np.random.default_rng(k)
            tgt[tr[zone]] = tgt[tr[zone]] * np.exp(rng.normal(size=int(zone.sum())))
            A2[_tc(ctx)] = tgt
            c2 = dict(ctx, A=A2)
            c2['leg_refit'] = leg_refit_harj_ols(c2)
            pert = cross_fit(c2, 3, tr, h)
            d0 = float(np.max(np.abs(pert[inb] - base[inb])))
            c3 = dict(ctx, A=A2)
            f3 = leg_refit_harj_ols(c3)
            c3['leg_refit'] = lambda step, fit, pred, f3=f3: f3(step, np.concatenate([fit, pred]), pred)   # подлог: блок в обучении
            base3 = dict(ctx)
            fb = leg_refit_harj_ols(base3)
            base3['leg_refit'] = lambda step, fit, pred, fb=fb: fb(step, np.concatenate([fit, pred]), pred)
            d1 = float(np.max(np.abs(cross_fit(c3, 3, tr, h, embargo=0)[inb] - cross_fit(base3, 3, tr, h, embargo=0)[inb])))
            ok = (d0 == 0.0) and (d1 > 0)
            rows.append(dict(h=h, block=k, rows=int(inb.sum()), clean_max_abs=d0, planted_max_abs=d1, ok=bool(ok), time=time.strftime('%F %T')))
            log(f'leak_xfit h={h} блок {k}: строк {int(inb.sum())}, чисто Δ = {d0:.3g}, подлог Δ = {d1:.3g} {"OK" if ok else "ПРОВАЛ"}')
    p = R / 'R2' / 'B' / 'leak_xfit_B.csv'
    pd.DataFrame(rows).to_csv(p, index=False)
    assert all(r['ok'] for r in rows), 'детектор cross_fit: провал'


def shift_check(pred, A):
    """п. 2.6: Спирмен ln ŷ_d с ln RV_{d+s}, s ∈ {−2, −1, 0, +1}, по тикерам, медиана; только годы 2017–2021. -> (argmax, {s: corr})."""
    from scipy.stats import spearmanr
    a = A[['date', 'ticker', 'rv']].copy().sort_values(['ticker', 'date'])
    for s in (-2, -1, 0, 1):
        a[f'rv_{s}'] = a.groupby('ticker')['rv'].shift(-s)
    m = pred[(pred.year >= 2017) & (pred.year <= 2021)].merge(a, on=['date', 'ticker'])
    cor = {}
    for s in (-2, -1, 0, 1):
        v = []
        for _, g in m.groupby('ticker'):
            ok = (g[f'rv_{s}'] > 0) & (g.rv_pred > 0)
            if ok.sum() > 30:
                v.append(spearmanr(np.log(g.rv_pred[ok]), np.log(g[f'rv_{s}'][ok]))[0])
        cor[s] = float(np.median(v)) if v else np.nan
    return max(cor, key=lambda s: cor[s]), cor


def cmd_shift(a):
    """п. 2.6 функцией ядра r2_core.shift_check на дымовых рядах (шаги 2–3, 3 тикера); блоки — в R2/B/blok_sdvig_smoke.csv, не в файл O."""
    import r2_core as C
    rows = []
    for h in (1, 5, 22):
        ctx = d0_ctx(h)
        ctx['_lhs_point'] = 7
        for role, bk in SMOKE_DESIGNS:
            o = build(mk(role, bk, h, name='shift'), ctx, steps=(2, 3))
            r = C.shift_check(o, ctx['A'], h, f'B_{role} {json.dumps(bk)}', block_file=R / 'R2' / 'B' / 'blok_sdvig_smoke.csv')
            rows.append(dict(keys=json.dumps(bk), role=role, **r))
            log(f'сдвиг h={h} {role} {bk}: argmax {r["argmax"]}, ' + ', '.join(f'{s}: {r[f"corr_s{s:+d}"]:.3f}' for s in C.SHIFTS))
    T = pd.DataFrame(rows)
    T.to_csv(R / 'R2' / 'B' / 'shift_B.csv', index=False)
    log(f'сдвиг: {int(T.ok.sum())}/{len(T)} рядов argmax = 0')


def _smoke_leg_provider(design, h):
    """Дым: нога дизайна из кэша L (версия _emb_d0): HARJ -> HARJ_emb_d0, HARX -> HARX_inner_emb_d0."""
    import r2_core as C
    nm = {'HARJ': 'HARJ_emb_d0', 'HARX': 'HARX_inner_emb_d0'}[design['leg']['type']]
    g = pd.read_parquet(R / 'R2' / 'cache' / 'E' / f'{nm}_h{h}_lnall.parquet')
    return g[g.ticker.isin(set(C.SHARED['A'].ticker))]


def _smoke_refit_provider(design, h, A):
    return leg_refit_harj_ols(dict(A=A, h=h, harj=C_harj()))


def C_harj():
    import r2_core as C
    return C.SHARED.get('harj')


def cmd_core_smoke(a):
    """Дым через r2_core.run_phase (кэш — R2/B/smoke_cache, ворота не проверяются): режим владельца 16:25 — скрининг LightGBM на 50% строк
    (h = 5, 22 — роль residual; h = 1 — feature с группами G0 + G1), уровень 2 — точки 1 и 61 сетки (r2_resid_grid.core_fn)."""
    import r2_core as C
    import r2_resid_grid as RG
    sm = YB['smoke']
    root = R / 'R2' / 'B' / 'smoke_cache'
    rows = []
    for h in sm['hs']:
        A, harj = C.load_d0(tickers=sm['tickers'])
        A, groups = load_groups(A, h)
        C.share(A=A, harj=harj, B_groups=groups, leg_provider=_smoke_leg_provider, leg_refit_provider=_smoke_refit_provider, B_half=True,
                B_grid_points=[1, 61])
        leg = 'HARX' if h != 22 else 'HARJ'
        role = 'feature' if h == 1 else 'residual'
        scr = dict(design=f'smoke_B_h{h}', h=h, leg=dict(type=leg), boost=dict(role=role, lib='lightgbm'), features=['G0', 'G1'])
        T = C.run_phase('B', h, [scr], core_fn, seeds=(42,), steps=sm['steps'], workers=2, module='B', root=root, check_gate=False,
                        block_file=R / 'R2' / 'B' / 'blok_sdvig_smoke.csv')
        rows.append(T.assign(faza='B'))
        C.share(B_half=False)
        grd = dict(scr, design=f'smoke_D_h{h}', boost=dict(role=role, lib='lightgbm', grid='resid_small_v1', early_stop='qlike_total'))
        T = C.run_phase('D', h, [grd], RG.core_fn, seeds=(42,), steps=sm['steps'], workers=2, module='B', root=root, check_gate=False,
                        block_file=R / 'R2' / 'B' / 'blok_sdvig_smoke.csv')
        rows.append(T.assign(faza='D'))
    T = pd.concat(rows, ignore_index=True)
    T.to_csv(R / 'R2' / 'B' / 'core_smoke_B.csv', index=False)
    log('дым ядра: ' + '; '.join(f'{r.faza} h={r.h} {r.ryad}: строк {r.stroki}, сдвиг ok={r.sdvig_ok}' for r in T.itertuples()))
    assert (T.ryad != 'ОШИБКА').all(), 'ошибка задачи в run_phase'


def cmd_drv_legs(a):
    """(внутренняя) ноги L в каталоге дизайна для дыма по контракту драйвера: r2_nogi.fn_leg на 3 тикерах (кэш — R2_CACHE)."""
    import r2_core as C
    import r2_nogi as RN
    Y = yaml.safe_load(open(a.designs))
    A0, harj = C.load_d0(tickers=YB['smoke']['tickers'])
    M = RN.mat_from(A0, harj)
    C.share(M=M, A=M['A'], harj=harj, L_root=None)
    for d in Y['designs']:
        RN.fn_leg(C.validate(d), 42, list(a.steps))
    log(f'ноги L записаны: {len(Y["designs"])} дизайнов, кэш {C.CACHE_ROOT}')


def cmd_drv_smoke(a):
    """Дым по контракту драйвера S: первые дизайны B_h{5,22,1} из models/designs/r2/driver_v1 -> ноги L (fn_leg) -> `run` модуля B
    (подпроцесс, --row-frac 0.5 --row-seed 20261026) и `leak --designs`; точка LHS p01 и p61 -> `r2_resid_grid.py run`.
    3 тикера × шаги 1–3 × seed 42; кэш — R2/B/driver_smoke_cache (R2_CACHE), не боевой."""
    import subprocess
    root = R / 'R2' / 'B' / f'driver_smoke_cache_{time.strftime("%m%d_%H%M")}'   # свежий каталог: DONE прошлых дымов не пропускают счёт
    env = dict(os.environ, R2_CACHE=str(root))
    me, grid = [sys.executable, '-W', 'ignore', str(Path(__file__).resolve())], [sys.executable, '-W', 'ignore', str(HERE / 'r2_resid_grid.py')]
    tk = YB['smoke']['tickers']
    st = [str(s) for s in YB['smoke']['steps']]
    out = []
    for h in (5, 22, 1):
        src = yaml.safe_load(open(REPO / 'models' / 'designs' / 'r2' / 'driver_v1' / f'B_h{h}.yaml'))
        d = src['designs'][0] if h != 1 else src['designs'][-1]
        f = root / f'B_h{h}.yaml'
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(yaml.safe_dump(dict(designs=[d]), allow_unicode=True, sort_keys=False))
        gd = json.loads(json.dumps(d))
        gd.setdefault('boost', {}).update(grid='resid_small_v1', early_stop='qlike_total', grid_point='p01')
        if gd['boost'].get('role', 'none') == 'none':
            gd['boost']['role'] = 'residual'
        gd['design'] = f'{d.get("design")}_p01'
        g61 = json.loads(json.dumps(gd)); g61['boost']['grid_point'] = 'p61_tekushchaya'; g61['design'] = f'{d.get("design")}_p61'
        fg = root / f'D_h{h}.yaml'
        fg.write_text(yaml.safe_dump(dict(designs=[gd, g61]), allow_unicode=True, sort_keys=False))
        for cmd in (me + ['drv_legs', '--designs', str(f), '--steps'] + st,
                    me + ['drv_legs', '--designs', str(fg), '--steps'] + st):
            t0 = time.time()
            rc = subprocess.run(cmd, env=env, cwd=str(REPO)).returncode
            out.append(dict(h=h, cmd=' '.join(cmd[3:6]), rc=rc, sec=round(time.time() - t0, 1)))
            log(f'дым драйвера h={h}: {" ".join(cmd[3:5])} -> код {rc}, {time.time() - t0:.0f} с')
        if a.subprocess:
            for cmd in (me + ['run', '--designs', str(f), '--h', str(h), '--seeds', '42', '--steps'] + st + ['--workers', '2', '--row-frac', '0.5',
                                                                                                         '--row-seed', '20261026', '--tickers'] + tk,
                        grid + ['run', '--designs', str(fg), '--h', str(h), '--seeds', '42', '--steps'] + st + ['--workers', '2', '--tickers'] + tk,
                        me + ['leak', '--designs', str(f), '--h', str(h)]):
                t0 = time.time()
                rc = subprocess.run(cmd, env=env, cwd=str(REPO)).returncode
                out.append(dict(h=h, cmd=' '.join(cmd[3:6]), rc=rc, sec=round(time.time() - t0, 1)))
            continue
        # через r2_driver.Runner (боевой режим: подпроцесс по шаблону r2_S_driver.yaml modules.B / BG, детектор leak до run);
        # настройки фазы B — как в r2_driver (--row-frac 0.5 --row-seed 20261026), плюс --tickers дыма в {extra}
        import r2_driver as DRV
        os.environ['R2_CACHE'] = str(root)
        run_ = DRV.Runner(smoke=False)
        xB = f'--row-frac {DRV.CFG["B"]["row_frac"]} --row-seed {DRV.CFG["B"]["row_seed"]} --tickers ' + ' '.join(tk)
        for m, ds, ex, seeds in (('B', [d], xB, DRV.CFG['B']['seeds']), ('BG', [gd, g61], '--tickers ' + ' '.join(tk), DRV.CFG['D']['seeds'])):
            t0 = time.time()
            ok, why = run_.run(m, ds, h, seeds, [int(s) for s in st], 3, ex, tag='dymB')
            out.append(dict(h=h, cmd=f'r2_driver.Runner {m} seeds {seeds}', rc=0 if ok else 1, sec=round(time.time() - t0, 1), why=why))
            log(f'дым драйвера (Runner) h={h} {m}: {"OK" if ok else why}, {time.time() - t0:.0f} с')
    T = pd.DataFrame(out)
    T.to_csv(R / 'R2' / 'B' / 'driver_smoke_B.csv', index=False)
    assert (T.rc == 0).all(), 'дым драйвера: ошибка'


KESH_DESIGNS = [
    ('residual', {'lib': 'lightgbm'}),                                                              # native ES, валид. набор от кэша
    ('residual', {'grid': 'resid_small_v1', 'early_stop': 'qlike_total', 'lib': 'lightgbm'}),      # фаза D (точка 7)
    ('offset', {'lib': 'lightgbm', 'early_stop': 'qlike_total'}),                                   # gamma, init_score в бинарнике
    ('feature', {'lib': 'both_avg'}),                                                               # LGB + XGB (финалисты)
    ('residual', {'lib': 'xgboost', 'early_stop': 'qlike_total'}),                                  # XGB QuantileDMatrix, внутр. складка
    ('residual', {'resid_source': 'cross_fit', 'early_stop': 'qlike_total'}),                       # cross_fit: бины шага 1 лениво
    ('residual_std', {'early_stop': 'qlike_total'}),
    ('residual', {'boost_refit': 'monthly'}),                                                       # кэш + переобучение без кэша
    ('residual', {'early_stop': 'qlike_total', '_data': {'recency_halflife': '4y', 'winsor': 'p1_99_by_ticker'}}),  # веса, winsor
    ('offset', {'lib': 'both_avg'}),                                                                # gamma: init_score / base_margin
]


def cmd_kesh(a):
    """Контроль кэша наборов (r2_dataset_cache, S): ряды без кэша = с кэшем (холодный — построение, тёплый — чтение бинарника /
    массивов XGB с диска), побайтно; seed 43 на тёплом кэше = seed 43 без кэша (набор общий для seed). Замер — время одного
    обучения (шаг × библиотека) без кэша и с тёплым кэшем. Корень — отдельный каталог на /Volumes/RAMcache (не смешивается с боем)."""
    DC = _dc()
    os.environ['R2_DS_CACHE'] = str((DC.RAM if DC.RAM.is_dir() else DC.SSD) / f'r2_ds_Bkesh_{time.strftime("%H%M%S")}')
    log(f'kesh: корень {DC.cache_root()}')
    rows = []
    for h in a.hs:
        ctx0 = d0_ctx(h, tickers=False if a.all else None)
        afp = a_fingerprint(ctx0['A'])
        for i, (role, bk) in enumerate(a.designs and [KESH_DESIGNS[j] for j in a.designs] or KESH_DESIGNS):
            res, sec = {}, {}
            global ES_CB
            for mode, seed in (('medl', 42), ('bez', 42), ('holod', 42), ('teplo', 42), ('bez', 43), ('teplo', 43)):
                ES_CB = mode != 'medl'     # medl — прежний путь (R1/B до 26.10): без кэша, nmax деревьев и цикл по готовой модели
                ctx = {k: v for k, v in ctx0.items() if not k.startswith('_')}
                ctx['_lhs_point'] = 7
                if mode not in ('bez', 'medl'):
                    ctx['_afp'], ctx['_refit_id'] = afp, 'harj_ols'
                if mode == 'teplo':
                    DC._XQ.clear(); _REFM.clear()   # тёплый — из файлов кэша, не из памяти процесса
                st0 = DC.stats()
                t0 = time.time()
                dz = mk(role, bk, h, seed, 'kesh')
                if '_data' in bk:
                    dz['data'] = dict(bk['_data'])
                o = build(dz, ctx, steps=a.steps)
                sec[(mode, seed)] = time.time() - t0
                res[(mode, seed)] = o
                st = {k: v - st0[k] for k, v in DC.stats().items()}
                log(f'kesh h={h} {role} {bk} {mode} s{seed}: {sec[(mode, seed)]:.2f} с, {json.dumps(st)}')
            eq = lambda x, y: bool(np.array_equal(res[x].rv_pred.values, res[y].rv_pred.values) and res[x][KEY].equals(res[y][KEY]))
            libs = 2 if bk.get('lib') == 'both_avg' else 1
            nfit = len(a.steps) * libs * (2 if bk.get('early_stop') == 'qlike_total' else 1)
            rows.append(dict(h=h, role=role, keys=json.dumps(bk), strok=len(res[('bez', 42)]),
                             ravno_medl=eq(('medl', 42), ('bez', 42)), sek_medl=round(sec[('medl', 42)], 2),
                             ravno_holod=eq(('bez', 42), ('holod', 42)), ravno_teplo=eq(('bez', 42), ('teplo', 42)),
                             ravno_teplo_s43=eq(('bez', 43), ('teplo', 43)), raznye_seed=not eq(('bez', 42), ('bez', 43)),
                             obuch=nfit, sek_bez=round(sec[('bez', 42)], 2), sek_holod=round(sec[('holod', 42)], 2),
                             sek_teplo=round(sec[('teplo', 42)], 2), sek_obuch_bez=round(sec[('bez', 42)] / nfit, 3),
                             sek_obuch_teplo=round(sec[('teplo', 42)] / nfit, 3)))
    T = pd.DataFrame(rows)
    p = R / 'R2' / 'B' / f'kesh_B{"_all" if a.all else ""}.csv'
    p.parent.mkdir(parents=True, exist_ok=True)
    T.to_csv(p, index=False)
    with pd.option_context('display.width', 250, 'display.max_colwidth', 60):
        print(T.to_string(index=False))
    ok = bool((T.ravno_medl & T.ravno_holod & T.ravno_teplo & T.ravno_teplo_s43).all())
    log(f'kesh: {"ВСЕ РАВНЫ ПОБАЙТНО" if ok else "РАСХОЖДЕНИЕ"} -> {p}')
    if not ok:
        raise SystemExit(1)


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest='cmd', required=True)
    p = sp.add_parser('smoke'); p.add_argument('--hs', type=int, nargs='*')
    p = sp.add_parser('leak'); p.add_argument('--hs', type=int, nargs='*'); p.add_argument('--designs'); p.add_argument('--h', type=int)
    p = sp.add_parser('run'); p.add_argument('--designs', required=True); p.add_argument('--h', type=int, required=True)
    p.add_argument('--seeds', type=int, nargs='+', default=[42]); p.add_argument('--steps', type=int, nargs='+', default=[1, 2, 3, 4, 5, 6])
    p.add_argument('--workers', type=int, default=None); p.add_argument('--row-frac', dest='row_frac', type=float, default=None)
    p.add_argument('--row-seed', dest='row_seed', type=int, default=20261026); p.add_argument('--tickers', nargs='*', default=None)
    p = sp.add_parser('drv_legs'); p.add_argument('--designs', required=True); p.add_argument('--steps', type=int, nargs='+', default=[1, 2, 3])
    p = sp.add_parser('drv_smoke'); p.add_argument('--subprocess', action='store_true', help='прежний дым (прямые подпроцессы, без Runner)')
    sp.add_parser('control'); sp.add_parser('leak_leg'); sp.add_parser('leak_xfit'); sp.add_parser('shift'); sp.add_parser('core_smoke')
    p = sp.add_parser('kesh'); p.add_argument('--hs', type=int, nargs='+', default=[5, 22, 1])
    p.add_argument('--steps', type=int, nargs='+', default=[1, 2, 3]); p.add_argument('--all', action='store_true')
    p.add_argument('--designs', type=int, nargs='*', default=None)
    a = ap.parse_args()
    if a.cmd == 'leak' and a.designs:
        return cmd_leak_designs(a)
    {'smoke': cmd_smoke, 'control': cmd_control, 'leak': cmd_leak, 'leak_leg': cmd_leak_leg, 'leak_xfit': cmd_leak_xfit,
     'shift': cmd_shift, 'core_smoke': cmd_core_smoke, 'run': cmd_run, 'drv_legs': cmd_drv_legs, 'drv_smoke': cmd_drv_smoke,
     'kesh': cmd_kesh}[a.cmd](a)


if __name__ == '__main__':
    main()
