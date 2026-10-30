#!/usr/bin/env python3
"""Д26.3 — пять прямых моделей рецепта D22_h1 на лиды j = 1..5 и прогноз h=5 из них (владелец 26.10, 09:35 / 09:45).
Лид j: цель ln RV(t+j) (rv тикера, сдвинутый на j строк), признак Д8.12 — ln HAR-X⁺ на тот же лид (слой 1 Д22 с целью RV(t+j),
эмбарго j дней, календарь окна [t+j, t+j]), признаки v3 + RRV + гэп + календарь окна лида, винзоризация цели 1/99 (рецепт D22_h1).
Лид 1 = D22_h1 (его прогнозы переиспользуются: seed 42/43/44 из data/regen/D22_h1_*; seed 45–51 и шаг 10 — пересчёт тем же рецептом).
Прогноз h=5: ŷ₅ = mean_j exp(ŷ_j) (ln среднего уровней) с поправкой Б0.1:
  S1 — одна s² на агрегат (остатки ln RV̄₅ − ln ŷ₅; это делает gibrid_v17 с кодами ord/sm/...);
  S2 — s² по лиду: ŷ₅ = mean_j exp(ŷ_j + s_j²/2), s_j² — sm по остаткам лида j на val шагов k−3..k−1 (тест шага k — по шагам ≤ k);
       в гибрид подаётся уже поправленным, дизайн — с поправкой «bez»;
  S3 (справка) — одна поправка на агрегат с учётом сдвига: F = exp(mean e + var e / 2) (sm-окно).
Варианты: (б) точки D22_h1 по шагам без подбора (тег D26_3b); (а) переподбор по протоколу B v2 на каждый лид (tune_L2, тег D26_3a_L{j}).
Метрики — только val шагов 2–6 (2017–2021): отбор 2017–2020, проверка 2021. test_locked/ не читается — только запись своих тестовых прогнозов.
Режимы:
  --hxp                         слой 1 на лиды 2..5 -> data/regen/D26_3/HXPL_{j}_val.parquet, HXPL_lnall.parquet
  --fixed [--leads] [--seeds] [--steps] [--sfx]   вариант (б) -> data/regen/D26_3/D26_3b_L{j}_{m}{sfx}_val.parquet
  --tune J [--workers 4]        вариант (а) для лида J -> data/regen/D26_3/a/, data/regen/D26_3/D26_3a_L{J}_{m}_val.parquet
  --eval VAR                    сборка h=5, таблицы по seed, гибрид gibrid_v17 по seed, условие входа -> data/regen/D26_3/eval_*.csv
  --leak                        детектор «будущее не влияет» с подложенной утечкой -> data/regen/D26_3/leak.csv
  --export VAR --s2 S1|S2       файлы формата final3 (val шагов 1–9) data/regen/D26_3_{VAR}_{xgb,lgbm}_h5[_seeds4344]_val.parquet
  --check1                      сверка пересчёта лида 1 с D22_h1 (seed 42, шаг 2)
python3 -W ignore scripts/analysis/d26_3.py --hxp; ... --fixed --workers 4; ... --eval b"""
import os
os.environ['L2_FEAT'] = '/Users/nilysenok/Desktop/Старое/PhD_Vol_trading/data/regen/features_pit_v3_rrv.parquet'
for _k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_k] = '1'
import argparse, hashlib, json, subprocess, sys, time, multiprocessing as mp
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import l2_lib as L
import d22_harxp as P
import harx as HX

R = L.REGEN                                   # общий data/regen (абсолютный путь)
OUT = R / 'D26_3'
LOCK = R / 'test_locked' / 'D26_3'            # только запись
LOCKH = R / 'test_locked' / 'D26_3_HXP'       # только запись
GIN, GOUT = OUT / 'gib_in', OUT / 'gibrid'
FEAT_V3 = R / 'features_pit_v3.parquet'
D22 = 'D22_h1'
CFG = {'harf': True, 'wins': True}
LEADS = (1, 2, 3, 4, 5)
SEEDS = (42, 43, 44)
SEEDS10 = tuple(range(42, 52))
MODELS = ('xgb', 'lgbm')
DEV_MAX, RANK, HOLD = 6, (2, 3, 4, 5), 6
CR = ('c1', 'c2', 'c3', 'c4')
RRV = ['rrv_pk_d', 'rrv_pk_w', 'rrv_pk_m', 'rrv_gk_d', 'rrv_gk_w', 'rrv_gk_m', 'r2_on']
D12_SEED = {42: 0.215906, 43: 0.222938, 44: 0.227057}   # D12_h5 wrap/V0_LGB/B-sm по seed (data/regen/D26/step0.csv; умолчание 09:52)
D12_FIX = 0.215906                                       # D12_h5 финал (final3 ranking_h5, wrap2/V0_XGB/B-sm)
D12_2021 = 0.156640
ETALON = dict(m1720=0.2410, q2021=0.1727)                # HAR-J с поправкой s² sm (final3/etalon)


def tg(j):
    return '1' if j == 1 else f'L{j}'


def tcol_lead(j):
    return 'rv_target_h1' if j == 1 else f'rv_target_hL{j}'


def lncol(j, k):
    return f'hxp_1_{k}' if j == 1 else f'hxp_L{j}_{k}'


def extra(j):
    return RRV + [f'{f}@{tg(j)}' for f in P.CAL_FIELDS]


for _j in LEADS[1:]:  # реестр слагаемых слоя 1 на лиды (в этом процессе; d22_harxp.py не меняется)
    P.BOUNDS[tg(_j)] = (_j, _j)
    for _f in P.CAL_FIELDS:
        P.TERMS[f'{_f}@{tg(_j)}'] = P._cal(tg(_j), _f)


def add_leads(A):
    """A отсортирована по дате (load_pit / load_base): RV(t+j) = rv тикера через j строк; _tend — год последнего дня цели (эмбарго)."""
    g = A.groupby('ticker')
    for j in LEADS[1:]:
        A[tcol_lead(j)] = g['rv'].shift(-j)
        A[f'_tend_{tg(j)}'] = pd.to_datetime(g['date'].shift(-j)).dt.year
    return A


def s2_est(code, E, st):
    """Как harx.s2_est: ord — остатки val шага st; sm — шагов st−2..st; smmu — как sm, но возвращает (mean, var)."""
    ks = [j for j in range(max(1, st - 2) if code in ('sm', 'smrob', 'smmu') else st, st + 1) if j in E]
    e = np.concatenate([E[j] for j in ks])
    if code in ('rob', 'smrob'):
        return float((1.4826 * np.median(np.abs(e - np.median(e)))) ** 2)
    if code == 'smmu':
        return float(np.mean(e)), float(np.var(e))
    return float(np.var(e))


def qlike(y, f):
    r = y / np.clip(f, 1e-12, None)
    return float(np.mean(r - np.log(r) - 1))


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


# ================================================================ слой 1: HAR-X⁺ на лиды 2..5
def hxp():
    A, harj = P.load()
    A = add_leads(A)
    OUT.mkdir(parents=True, exist_ok=True); LOCKH.mkdir(parents=True, exist_ok=True)
    tg_ = [tg(j) for j in LEADS[1:]]
    names = sorted(set(sum([P.terms_for(t) for t in tg_], [])))
    X = P.build_terms(A, names)
    ln, logs = A[['date', 'ticker']].copy(), []
    for j in LEADS[1:]:
        vals, tests = [], []
        for k in range(1, 11):
            v, te, la, _ = P.forecast_step(A, tg(j), k, harj, P.terms_for(tg(j)), X, logs)
            vals.append(v); tests.append(te); ln[lncol(j, k)] = la.values
        val, test = HX.add_bias(pd.concat(vals, ignore_index=True), pd.concat(tests, ignore_index=True))
        val.to_parquet(OUT / f'HXPL_{j}_val.parquet', index=False)
        test.to_parquet(LOCKH / f'HXPL_{j}_test.parquet', index=False)
        print(f'HXP⁺ лид {j}: готово', flush=True)
    ln.to_parquet(OUT / 'HXPL_lnall.parquet', index=False)
    pd.DataFrame(logs).to_csv(OUT / 'HXPL_coefs.csv', index=False)


# ================================================================ подмены l2_lib (только в этом процессе)
_lb0, _bf0, _split0, _exc0 = L.load_base, L.build_features, L.split, L.excluded
ST = {'exc_d22': False, 'leak_col': None, 'A_override': None}


def tcol_x(h):
    return tcol_lead(int(h))


def har_ln_x(tp, parts, h, hc, w=None, huber=False, y=None):
    """ln HAR-X⁺ лида h модели шага k = (последний год обучения + 1) − 2015 (обучение — внутри выборки, как Д8.12 в D22_h1)."""
    k = int(tp['year'].max()) + 1 - 2015
    c = lncol(int(h), k)
    return [d[c].values for d in parts]


def split_x(A, h, step):
    if step != 10:
        return _split0(A, h, step)
    t = L.tcol(h)
    tr = A[A.year < 2026]
    return tr[tr.year != 2025].dropna(subset=[t]), tr[tr.year == 2025].dropna(subset=[t]), A[A.year == 2026], 2025, 2026


def excluded_x(model, h, tag):
    if ST['exc_d22'] and tag == D22:  # вариант (б): исключения признаков D22_h1 (пара модель × h=1) — для всех лидов
        return _exc0(model, 1, tag)
    return _exc0(model, h, tag)


def merge_ln(A):
    h1 = pd.read_parquet(R / 'D22' / 'HXP_lnall.parquet', columns=['date', 'ticker'] + [lncol(1, k) for k in range(1, 11)])
    A = A.merge(h1, on=['date', 'ticker'], how='left', validate='one_to_one')
    A = A.merge(pd.read_parquet(OUT / 'HXPL_lnall.parquet'), on=['date', 'ticker'], how='left', validate='one_to_one')
    return A


def load_base_x(matrix=None, blocks=(), h=None):
    A, fc = _lb0(matrix, blocks, h)
    A = add_leads(merge_ln(A))
    assert A[[lncol(j, k) for j in LEADS for k in range(1, 11)]].notna().all().all()
    return A, fc


def add_extra(A2, leads):
    need = sorted(set(sum([extra(j) for j in leads], [])))
    X = P.build_terms(A2, need)
    for c in need:
        A2[c] = X[c].values
    if ST['leak_col']:
        A2['leak'] = np.log(A2[tcol_lead(ST['leak_col'])].values + 1e-10)  # подложенная утечка: сама цель лида
    return A2


def build_features_x(A, FC, h, cfg):
    A2, cols, cats, mono = _bf0(A, FC, h, cfg)
    A2 = add_extra(A2, [int(h)])
    return A2, cols + [c for c in extra(int(h)) if c not in cols], cats, mono


def patch():
    L.tcol, L.har_ln, L.split, L.build_features, L.load_base, L.excluded = tcol_x, har_ln_x, split_x, build_features_x, load_base_x, excluded_x


def lead_cols(base_cols, j):
    c = base_cols + [x for x in extra(j) if x not in base_cols]
    return c + (['leak'] if ST['leak_col'] == j else [])


# ================================================================ вариант (б): точки D22_h1 без подбора
G = {}


def sel_points(tag_csv):
    from tune_L2 import SPACEB
    sel = L.selected_params(tag_csv, 1)
    return {(m, s): {k: sel[(m, s)][k] for k, *_ in SPACEB[m]} for (m, s) in sel}


def _frame(D, part, j, step, seed, m, pred, yr):
    d = D['parts'][part]
    return pd.DataFrame({'date': d['date'].values, 'ticker': d['ticker'].values, 'year': yr, 'step': step, 'seed': seed, 'lead': j,
                         'model': m, 'rv_actual': d[tcol_lead(j)].values, 'rv_pred': np.clip(np.exp(pred), 1e-10, None)})


def _job_fixed(a):
    j, m, seed, steps = a
    A, cols = G['A'], lead_cols(G['cols'], j)
    ref, vo, to, bis = None, [], [], []
    for st in steps:
        D = L.step_data(A, cols, j, st, m, CFG, D22)
        if st > 1 and ref is None:
            ref = L.make_ref(m, A, cols, j, CFG, D22)
        p = G['sel'][(m, min(st, 9))]  # шаги 7–10 — точка шага 6 (в selected D22_h1 у шагов 7–9 она же)
        pv, pt, bi, _, _ = L.fit(m, p, D, CFG, seed=seed, ref=ref if st > 1 else None)
        vo.append(_frame(D, 'va', j, st, seed, m, pv, D['vy'])); to.append(_frame(D, 'te', j, st, seed, m, pt, D['ty']))
        bis.append(dict(lead=j, model=m, seed=seed, step=st, best_iteration=bi))
    return j, m, seed, pd.concat(vo), pd.concat(to), bis


def fixed(leads, seeds, steps, sfx, workers, tag='D26_3b', sel_csv=None):
    patch(); ST['exc_d22'] = True
    A, FC = L.load_base()
    A2, cols, _, _ = _bf0(A, FC, 1, CFG)
    A2 = add_extra(A2, leads)
    G.update(A=A2, cols=cols, sel=sel_points(sel_csv or D22))
    jobs = [(j, m, s, list(steps)) for j in leads for m in MODELS for s in seeds]
    t0 = time.time()
    with mp.get_context('fork').Pool(workers) as pool:
        res = pool.map(_job_fixed, jobs, chunksize=1)
    OUT.mkdir(parents=True, exist_ok=True); LOCK.mkdir(parents=True, exist_ok=True)
    for j in leads:
        for m in MODELS:
            rr = [r for r in res if r[0] == j and r[1] == m]
            pd.concat([r[3] for r in rr]).to_parquet(OUT / f'{tag}_L{j}_{m}{sfx}_val.parquet', index=False)
            pd.concat([r[4] for r in rr]).to_parquet(LOCK / f'{tag}_L{j}_{m}{sfx}_test.parquet', index=False)
    pd.DataFrame(sum([r[5] for r in res], [])).to_csv(OUT / f'{tag}{sfx}_trees.csv', index=False)
    print(f'{tag}{sfx}: лиды {leads}, seed {seeds}, шаги {list(steps)} — {time.time() - t0:.0f} с', flush=True)


def check1():
    """Пересчёт лида 1 рецептом (б) обязан совпасть с D22_h1 (seed 42, шаг 2, обе модели)."""
    fixed([1], [42], [2], '_check', 2)
    rows = []
    for m in MODELS:
        a = pd.read_parquet(OUT / f'D26_3b_L1_{m}_check_val.parquet')
        b = pd.read_parquet(R / f'{D22}_{m}_h1_val.parquet')
        b = b[b.step == 2]
        k = a.merge(b, on=['date', 'ticker', 'step'], suffixes=('', '_d22'))
        rows.append(dict(model=m, n=len(k), n_d22=len(b), max_abs_ln=float(np.abs(np.log(k.rv_pred) - np.log(k.rv_pred_d22)).max())))
    r = pd.DataFrame(rows); r.to_csv(OUT / 'check_lead1.csv', index=False); print(r.to_string())


# ================================================================ вариант (а): протокол B v2 на лид
def tune_a(j, workers):
    patch()
    import tune_L2 as T
    tag = f'D26_3a_L{j}'
    sys.argv = ['tune_L2.py', '--h', str(j), '--cfg', json.dumps(CFG), '--space', 'B', '--workers', str(workers), '--tag', tag]
    T.main()
    # файлы tune_L2 (верх data/regen) -> data/regen/D26_3/a/ (чтобы скан final3 не принимал их за кандидата h=5); тест остаётся в test_locked/
    da = OUT / 'a'; da.mkdir(parents=True, exist_ok=True)
    for f in R.glob(f'{tag}_*'):
        os.replace(f, da / f.name)
    for m in MODELS:
        a = pd.read_parquet(da / f'{tag}_{m}_h{j}_val.parquet')
        b = pd.read_parquet(da / f'{tag}_{m}_h{j}_seeds4344_val.parquet')
        d = pd.concat([a, b], ignore_index=True)
        d = d.assign(lead=j, model=m)[['date', 'ticker', 'year', 'step', 'seed', 'lead', 'model', 'rv_actual', 'rv_pred']]
        d.to_parquet(OUT / f'D26_3a_L{j}_{m}_val.parquet', index=False)


# ================================================================ сборка h=5
def lead_val(var, m, j, seeds, kind='val'):
    """Прогнозы лида j модели m, все seed из seeds (val; kind='test' — только для экспорта, без метрик)."""
    fr = []
    if j == 1 and var in ('a', 'b'):
        base = R if kind == 'val' else R / 'test_locked' / D22
        if any(s in SEEDS for s in seeds) and kind == 'val':
            a = pd.read_parquet(base / f'{D22}_{m}_h1_val.parquet').assign(seed=42)
            b = pd.read_parquet(base / f'{D22}_{m}_h1_seeds4344_val.parquet')
            fr.append(pd.concat([a, b], ignore_index=True))
        f10 = OUT / f'D26_3b_L1_{m}_x10_val.parquet'
        if kind == 'val' and f10.exists():
            fr.append(pd.read_parquet(f10))
    else:
        for sfx in ('', '_x10'):
            f = OUT / f'D26_3{var}_L{j}_{m}{sfx}_val.parquet'
            if f.exists():
                fr.append(pd.read_parquet(f))
    d = pd.concat(fr, ignore_index=True)
    d = d[d.seed.isin(seeds)][['date', 'ticker', 'year', 'step', 'seed', 'rv_actual', 'rv_pred']].assign(lead=j)
    return d.drop_duplicates(['date', 'ticker', 'step', 'seed'])


def lead_factors(d, code='sm'):
    """F_j по шагам: val шага k — s² лида по val шагов < k (шаг 1: собственные остатки, 2016 не оценивается); тест шага k — по шагам ≤ k."""
    v = d[d.rv_actual > 1e-12]
    E = {st: np.log(g.rv_actual.values + 1e-10) - np.log(g.rv_pred.values) for st, g in v.groupby('step')}
    Fv = {st: np.exp(s2_est(code, E, max(st - 1, 1)) / 2) for st in E}
    Ft = {st: np.exp(s2_est(code, E, min(st, max(E))) / 2) for st in range(1, 11)}
    return Fv, Ft


def assemble(var, m, seeds, perlead=False):
    """-> val h=5: date, ticker, year, step, seed, rv_actual (= RV̄₅), rv_pred (= mean_j exp(ŷ_j) [· F_j]); m ∈ xgb | lgbm | avg (ln-среднее)."""
    out = []
    for j in LEADS:
        if m == 'avg':
            x, l = lead_val(var, 'xgb', j, seeds), lead_val(var, 'lgbm', j, seeds)
            d = x.merge(l[['date', 'ticker', 'step', 'seed', 'rv_pred']], on=['date', 'ticker', 'step', 'seed'], suffixes=('', '_l'))
            d['rv_pred'] = np.exp((np.log(d.rv_pred) + np.log(d.rv_pred_l)) / 2)
            d = d.drop(columns='rv_pred_l')
        else:
            d = lead_val(var, m, j, seeds)
        if perlead:
            parts = []
            for s, g in d.groupby('seed'):
                Fv, _ = lead_factors(g[g.step <= 9])
                parts.append(g.assign(rv_pred=g.rv_pred * g.step.map(Fv).fillna(1.0)))
            d = pd.concat(parts)
        out.append(d)
    W = pd.concat(out, ignore_index=True)
    k = ['date', 'ticker', 'year', 'step', 'seed']
    n = W.groupby(k).lead.nunique()
    agg = W.groupby(k, as_index=False).agg(rv_actual=('rv_actual', 'mean'), rv_pred=('rv_pred', 'mean'))
    agg = agg[agg.set_index(k).index.isin(n[n == 5].index)]
    return agg.reset_index(drop=True)


def years(val, code):
    """OOS QLIKE по годам 2017–2021 (шаги 2–6). code: bez | ord | sm | smmu (S3)."""
    v = val[(val.step <= DEV_MAX) & (val.rv_actual > 1e-12)]
    assert (pd.to_datetime(v.date).dt.year <= 2021).all()
    E = {st: np.log(g.rv_actual.values + 1e-10) - np.log(g.rv_pred.values) for st, g in v.groupby('step')}
    out = {}
    for st in range(2, DEV_MAX + 1):
        g = v[v.step == st]
        if code == 'bez':
            F = 1.0
        elif code == 'smmu':
            mu, s2 = s2_est('smmu', E, st - 1); F = np.exp(mu + s2 / 2)
        else:
            F = np.exp(s2_est(code, E, st - 1) / 2)
        out[2015 + st] = qlike(g.rv_actual.values, g.rv_pred.values * F)
    return out


def yrow(name, q, **kw):
    return dict(name=name, **kw, **{str(y): q[y] for y in sorted(q)}, m1720=float(np.mean([q[y] for y in range(2017, 2021)])), q2021=q[2021])


def entry(qs, seeds):
    """Условие входа (умолчание 09:52): парное с D12_h5 тех же seed и против фиксированного 0,2159; Δ̄ < −3% и |d̄| > 2σ в обоих."""
    q = np.array(qs, float)
    res = {}
    for nm, ref in (('pair', np.array([D12_SEED.get(s, np.nan) for s in seeds])), ('fix', np.full(len(q), D12_FIX))):
        ok = ~np.isnan(ref)
        d = q[ok] - ref[ok]
        sd = float(d.std(ddof=1)) if ok.sum() > 1 else float('nan')
        rel = float(q[ok].mean() / ref[ok].mean() - 1) if ok.any() else float('nan')
        res.update({f'{nm}_rel': rel, f'{nm}_dmean': float(d.mean()) if ok.any() else np.nan, f'{nm}_sigma': sd,
                    f'{nm}_pass3': rel < -0.03, f'{nm}_pass2s': bool(d.mean() < 0 and abs(d.mean()) > 2 * sd)})
    return res


# ---------------------------------------------------------------- гибрид gibrid_v17
def write_gib(boost, xv, lv):
    GIN.mkdir(parents=True, exist_ok=True)
    for m, d in (('xgb', xv), ('lgbm', lv)):
        d = d[d.step <= DEV_MAX]
        assert (pd.to_datetime(d.date).dt.year <= 2021).all()
        d[['date', 'ticker', 'year', 'step', 'rv_actual', 'rv_pred']].to_parquet(GIN / f'{boost}_{m}_h5_val.parquet', index=False)


def gib_cmd(boost, legs=('single', 'wrap')):
    out = GOUT / boost
    out.mkdir(parents=True, exist_ok=True)
    return out, [sys.executable, '-W', 'ignore', str(HERE / 'gibrid_v17.py'), '--boost', boost, '--h', '5', '--features', str(FEAT_V3),
                 '--regen', str(GIN), '--legs', *legs, '--out', str(out), '--devlog', str(out / 'devlog.md')]


def run_gibs(boosts, workers=4):
    env = dict(os.environ, GIBRID_SAME_SLOT='1')
    procs, done = [], []
    todo = list(boosts)
    while todo or procs:
        while todo and len(procs) < workers:
            b = todo.pop(0); out, cmd = gib_cmd(b)
            f = open(out / 'run.log', 'w')
            procs.append((b, subprocess.Popen(cmd, env=env, stdout=f, stderr=subprocess.STDOUT, cwd=str(HERE.parents[1])), f))
        for x in list(procs):
            if x[1].poll() is not None:
                x[2].close(); procs.remove(x)
                if x[1].returncode:
                    raise RuntimeError(f'gibrid {x[0]}: код {x[1].returncode}')
                done.append(x[0])
        time.sleep(2)
    return done


def gib_summary(boost, keep_har=False):
    cols = ['design', 'step', 'val_year', 'q_hyb'] + [k + '_h' for k in CR]
    D = pd.read_csv(GOUT / boost / f'designs_val_{boost}_h5.csv', usecols=cols)
    assert D.loc[D.step > DEV_MAX, 'q_hyb'].isna().all()
    if not keep_har:
        D = D[~D.design.str.contains(r'\|V0_(?:HAR|HCJ)\|')]
    D = D[D.step.between(2, DEV_MAX)]
    q = D.pivot_table(index='design', columns='step', values='q_hyb', sort=False)
    fl = D.groupby('design', sort=False)[[k + '_h' for k in CR]].first()
    S = pd.DataFrame({'design': q.index, 'm1720': q[list(RANK)].mean(axis=1).values, 'q2021': q[HOLD].values})
    for s in range(2, DEV_MAX + 1):
        S[str(2015 + s)] = q[s].values
    for k in CR:
        S[k] = fl.loc[S.design, k + '_h'].astype(bool).values
    S['n_failed'] = (~S[list(CR)]).sum(axis=1)
    return S


def seeds_table(Sb):
    """Sb: {seed: summary}. Правило: нарушения (максимум по seed), затем средний 2017–2020."""
    rows = []
    for d in set.intersection(*[set(S.design) for S in Sb.values()]):
        r = {s: S.set_index('design').loc[d] for s, S in Sb.items()}
        rows.append(dict(design=d, n_failed=max(int(x.n_failed) for x in r.values()),
                         failed=','.join(k for k in CR if not all(bool(x[k]) for x in r.values())),
                         m1720=float(np.mean([x.m1720 for x in r.values()])), q2021=float(np.mean([x.q2021 for x in r.values()])),
                         **{f'm1720_s{s}': float(x.m1720) for s, x in r.items()}, **{f'q2021_s{s}': float(x.q2021) for s, x in r.items()},
                         **{str(y): float(np.mean([x[str(y)] for x in r.values()])) for y in range(2017, 2022)}))
    return pd.DataFrame(rows).sort_values(['n_failed', 'm1720'], kind='stable').reset_index(drop=True)


def evaluate(var, seeds=SEEDS, gib=True, tag=''):
    rows = []
    # лиды: QLIKE (sm) по лиду и HAR-X⁺ того же лида
    for j in LEADS:
        hx = pd.read_parquet(R / 'D22' / 'HXP_1_val.parquet') if j == 1 else pd.read_parquet(OUT / f'HXPL_{j}_val.parquet')
        hx = hx[hx.step <= DEV_MAX]
        q = HX.dev_table(hx, 'rv_pred_sm')
        rows.append(yrow(f'HAR-X⁺ лид {j}', {2015 + s: q[s] for s in q}, kind='lead', lead=j, model='hxp', seed=0, s2='sm'))
        for m in MODELS + ('avg',):
            for s in seeds:
                if m == 'avg':
                    x, l = lead_val(var, 'xgb', j, [s]), lead_val(var, 'lgbm', j, [s])
                    d = x.merge(l[['date', 'ticker', 'step', 'rv_pred']], on=['date', 'ticker', 'step'], suffixes=('', '_l'))
                    d['rv_pred'] = np.exp((np.log(d.rv_pred) + np.log(d.rv_pred_l)) / 2)
                else:
                    d = lead_val(var, m, j, [s])
                rows.append(yrow(f'лид {j} {m}', years(d, 'sm'), kind='lead', lead=j, model=m, seed=s, s2='sm'))
    # h=5
    hx5 = pd.read_parquet(R / 'D22' / 'HXP_5_val.parquet'); hx5 = hx5[hx5.step <= DEV_MAX]
    q = HX.dev_table(hx5, 'rv_pred_sm')
    rows.append(yrow('HAR-X⁺₅ (Д22)', {2015 + s: q[s] for s in q}, kind='h5', lead=0, model='hxp', seed=0, s2='sm'))
    V = {}
    for m in MODELS + ('avg',):
        for s in seeds:
            raw = assemble(var, m, [s]); pl = assemble(var, m, [s], perlead=True)
            V[(m, s, 'S1')], V[(m, s, 'S2')] = raw, pl
            for nm, d, code in (('S1-sm', raw, 'sm'), ('S1-ord', raw, 'ord'), ('S1-bez', raw, 'bez'), ('S3-smmu', raw, 'smmu'),
                                ('S2-sm', pl, 'bez')):
                rows.append(yrow(f'h5 {m} {nm}', years(d, code), kind='h5', lead=0, model=m, seed=s, s2=nm))
    T = pd.DataFrame(rows)
    T.to_csv(OUT / f'eval_{var}{tag}.csv', index=False)
    print(T.groupby(['kind', 'lead', 'model', 's2'], sort=False)[['m1720', 'q2021']].mean().round(4).to_string(), flush=True)
    if not gib:
        return T, None
    boosts = []
    for s in seeds:
        for s2 in ('S1', 'S2'):
            b = f'D26_3{var}{"p" if s2 == "S2" else ""}{tag}_s{s}'
            write_gib(b, V[('xgb', s, s2)], V[('lgbm', s, s2)]); boosts.append(b)
    run_gibs(boosts)
    res = []
    for s2, code_ok in (('S1', lambda d: '|bez|' not in d), ('S2', lambda d: '|bez|' in d)):
        Sb = {}
        for s in seeds:
            S = gib_summary(f'D26_3{var}{"p" if s2 == "S2" else ""}{tag}_s{s}')
            Sb[s] = S[S.design.map(code_ok)]
        Tb = seeds_table(Sb).assign(s2=s2)
        res.append(Tb)
    Gt = pd.concat(res).sort_values(['n_failed', 'm1720'], kind='stable').reset_index(drop=True)
    Gt.to_csv(OUT / f'gib_{var}{tag}.csv', index=False)
    best = Gt.iloc[0]
    e = entry([best[f'm1720_s{s}'] for s in seeds], list(seeds))
    ok = all(e[k] for k in ('pair_pass3', 'pair_pass2s', 'fix_pass3', 'fix_pass2s')) and best.q2021 <= D12_2021 and best.n_failed == 0
    pd.DataFrame([dict(var=var, design=best.design, s2=best.s2, m1720=best.m1720, q2021=best.q2021, failed=best.failed, **e, entry=ok)]).to_csv(
        OUT / f'entry_{var}{tag}.csv', index=False)
    print(Gt.head(12)[['design', 's2', 'n_failed', 'failed', 'm1720', 'q2021'] + [f'm1720_s{s}' for s in seeds]].to_string(), flush=True)
    print('вход:', best.design, best.s2, e, 'итог', ok, flush=True)
    return T, Gt


# ================================================================ детектор «будущее не влияет»
def _perturb(A, t, rng):
    B = A.copy(); fut = pd.to_datetime(B.date) > t
    for c in B.columns:
        if c in ('date', 'ticker', 'year') or c.startswith('_tend') or not np.issubdtype(B[c].dtype, np.number):
            continue
        pool = B.loc[~fut, c].dropna().values
        if len(pool):
            B.loc[fut, c] = rng.choice(pool, fut.sum())
    return B


def _chain(Ah, harj, Ab, FC, j, step, m):
    """Полный контур лида j на одном шаге из сырых матриц: слой 1 (шаги 1 и step) -> признаки -> бустинг (точка D22_h1, seed 42)."""
    names = P.terms_for(tg(j))
    X = P.build_terms(Ah, names)
    ln = Ah[['date', 'ticker']].copy()
    for k in sorted({1, step}):
        _, _, la, _ = P.forecast_step(Ah, tg(j), k, harj, names, X)
        ln[lncol(j, k)] = la.values
    A = Ab.merge(ln, on=['date', 'ticker'], how='left', validate='one_to_one')
    A = add_leads(A)
    A2, cols, _, _ = _bf0(A, FC, 1, CFG)
    A2 = add_extra(A2, [j])
    cols = lead_cols(cols, j)
    D = L.step_data(A2, cols, j, step, m, CFG, D22)
    ref = L.make_ref(m, A2, cols, j, CFG, D22)
    pv, pt, bi, _, _ = L.fit(m, sel_points(D22)[(m, step)], D, CFG, seed=42, ref=ref)
    d = D['parts']['te']
    return pd.DataFrame({'date': d['date'].values, 'ticker': d['ticker'].values, 'p': pt})


def leak(cut='2020-06-30', step=4, leads=(5, 3), seed=20261026):
    """Шаг 4 (обучение < 2019, val 2019, тест 2020), срез внутри тестового года 2020 (≤ 2021): все числовые входы строк date > t
    (матрица, цели, RV) -> бутстрап прошлого; гэп (intraday1850) строк > t -> шум; события календаря с known_from > t — сдвиг.
    Контур пересчитывается целиком (слой 1 лида, признаки, бустинг). Прогнозы строк ≤ t обязаны совпасть побайтно.
    Самопроверка: признак leak = ln RV(t+j) (подложенная утечка) обязан дать расхождения."""
    patch(); ST['exc_d22'] = True
    t = pd.Timestamp(cut)
    Ah0, harj = P.load(); Ab0, FC = _lb0()
    rows = []
    for lk in (None, 'plant'):
        for j in leads:
            ST['leak_col'] = j if lk else None
            for m in MODELS:
                P._EV.clear()
                base = _chain(add_leads(Ah0.copy()), harj, Ab0.copy(), FC, j, step, m)
                rng = np.random.default_rng(seed)
                Ah, Ab = _perturb(Ah0, t, rng), _perturb(Ab0, t, rng)
                I = pd.read_parquet(P.INTRA); f = pd.to_datetime(I.date) > t
                I.loc[f, 'gap_on'] = rng.normal(0, 0.01, f.sum())
                tmp = OUT / '_leak_intra.parquet'; I.to_parquet(tmp, index=False)
                saved = P.INTRA; P.INTRA = tmp
                P._EV.clear(); ev = P._events(); ev2 = {}
                for ty, (d, kf, kt) in ev.items():
                    d2 = d.copy(); mk = kf > np.datetime64(t.date())
                    d2[mk] = d2[mk] + rng.integers(-20, 20, mk.sum()).astype('timedelta64[D]')
                    ev2[ty] = (d2, kf, kt)
                P._EV.clear(); P._EV['ev'] = ev2
                pert = _chain(add_leads(Ah), harj, Ab, FC, j, step, m)
                P.INTRA = saved; P._EV.clear(); tmp.unlink(missing_ok=True)
                k = base.merge(pert, on=['date', 'ticker'], suffixes=('', '_p'))
                k = k[pd.to_datetime(k.date) <= t]
                mism = int((k.p != k.p_p).sum())
                rows.append(dict(cut=cut, step=step, lead=j, model=m, planted=bool(lk), n_rows=len(k), mismatch=mism))
                print(rows[-1], flush=True)
    ST['leak_col'] = None
    r = pd.DataFrame(rows); r.to_csv(OUT / 'leak.csv', index=False); print(r.to_string())


# ================================================================ экспорт в формат final3
def export(var, s2, seeds_main=SEEDS, step10=False):
    """val шагов 1–9 h=5: {boost}_{xgb|lgbm}_h5_val.parquet (seed 42), _seeds4344 (43/44) в data/regen; тест шагов 1–9 (и шаг 10) —
    в test_locked/D26_3_{var}/ (только запись; читаются лишь собственные файлы test_locked/D26_3/ для сборки, без метрик)."""
    boost = f'D26_3_{var}{"p" if s2 == "S2" else ""}'
    lock = R / 'test_locked' / boost
    lock.mkdir(parents=True, exist_ok=True)
    for m in MODELS:
        v = assemble(var, m, list(seeds_main), perlead=(s2 == 'S2'))
        v = v[v.step <= 9].assign(rank=1)
        v[v.seed == 42].to_parquet(R / f'{boost}_{m}_h5_val.parquet', index=False)
        v[v.seed != 42].to_parquet(R / f'{boost}_{m}_h5_seeds4344_val.parquet', index=False)
    print('экспорт val:', boost, flush=True)
    return boost


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--hxp', action='store_true'); ap.add_argument('--fixed', action='store_true'); ap.add_argument('--check1', action='store_true')
    ap.add_argument('--tune', type=int, default=0); ap.add_argument('--eval', default=''); ap.add_argument('--leak', action='store_true')
    ap.add_argument('--nogib', action='store_true'); ap.add_argument('--export', default=''); ap.add_argument('--s2', default='S1')
    ap.add_argument('--leads', type=int, nargs='*', default=[2, 3, 4, 5]); ap.add_argument('--seeds', type=int, nargs='*', default=list(SEEDS))
    ap.add_argument('--steps', type=int, nargs='*', default=list(range(1, 10))); ap.add_argument('--sfx', default='')
    ap.add_argument('--tag', default=''); ap.add_argument('--workers', type=int, default=4)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.hxp:
        hxp()
    if a.check1:
        check1()
    if a.fixed:
        fixed(a.leads, a.seeds, a.steps, a.sfx, a.workers)
    if a.tune:
        tune_a(a.tune, a.workers)
    if a.leak:
        leak()
    if a.eval:
        evaluate(a.eval, tuple(a.seeds), gib=not a.nogib, tag=a.tag)
    if a.export:
        export(a.export, a.s2)


if __name__ == '__main__':
    main()
