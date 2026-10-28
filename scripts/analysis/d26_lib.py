"""Д26 (круг h=5 на разработке): общая часть — пути, оценка OOS (шаги 2–6, val 2017–2021), гибрид gibrid_v17 (c1–c4), правило, хэши.

Жёсткие запреты круга (умолчание дирижёра 26.10, 09:30): метрики считаются только на val шагов 1–6 (годы 2016–2021); строки шагов ≥ 7
в оценку не передаются (assert); test_locked/ не читается — только запись новых тестовых прогнозов своих дизайнов, без метрик.
"""
import hashlib, os, subprocess, sys
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
DATA = Path('/Users/nilysenok/Desktop/Старое/PhD_Vol_trading/data')
R = DATA / 'regen'
OUT = R / 'D26'
GIN = OUT / 'gib_in'        # входы гибрида (val шагов 1–6)
GOUT = OUT / 'gibrid'       # выходы gibrid_v17
LOCK = R / 'test_locked'    # только запись (D26_*), не читать
FEAT_V3 = R / 'features_pit_v3.parquet'
KEY = ['date', 'ticker', 'step']
DEV_MAX = 6
RANK_STEPS = (2, 3, 4, 5)   # 2017–2020 — отбор
HOLD = 6                    # 2021 — проверка
CR = ('c1', 'c2', 'c3', 'c4')
CODES = ('ord', 'rob', 'sm', 'smrob')
D12_REF = dict(design='wrap2|V0_XGB|B-sm|L0', m1720=0.215906, q2021=0.156640)  # final3/ranking_h5.csv (rank 3, c1–c4 выполнены)
ETALON_H5 = dict(code='sm', m1720=0.2410, q2021=0.1727)  # HAR-J с поправкой s² sm (final3/etalon.csv; определение — final_test.ETALON)


def dev_only(df):
    """Строки шагов 1–6 (val 2016–2021). Всё, что идёт в метрики, проходит через эту функцию."""
    d = df[df.step <= DEV_MAX].copy()
    assert d['step'].max() <= DEV_MAX and (pd.to_datetime(d['date']).dt.year <= 2021).all()
    return d


def qlike(y, f):
    r = y / np.clip(f, 1e-12, None)
    return float(np.mean(r - np.log(r) - 1))


def s2_est(code, E, st):
    """Как harx.s2_est / gibrid_bias.s2_est: ord — val шага st; sm/smrob — объединённые остатки шагов st−2..st."""
    e = np.concatenate([E[j] for j in range(max(1, st - 2) if code in ('sm', 'smrob') else st, st + 1) if j in E])
    if code in ('rob', 'smrob'):
        return float((1.4826 * np.median(np.abs(e - np.median(e)))) ** 2)
    return float(np.var(e))


def bias_factors(val, code, own_step1=True):
    """F шага k = exp(s²/2) по val-остаткам шагов < k (Б0.1 вне выборки, s² шага t−1). Шаг 1: own_step1 — по собственным остаткам
    (только чтобы ряд шага 1 был на том же уровне; 2016 не оценивается), иначе F = 1."""
    v = dev_only(val)
    v = v[v.rv_actual > 1e-12]
    E = {st: np.log(g.rv_actual.values + 1e-10) - np.log(g.rv_pred.values) for st, g in v.groupby('step')}
    F = {}
    for st in sorted(v.step.unique()):
        F[st] = np.exp(s2_est(code, E, st - 1) / 2) if st > 1 else (np.exp(s2_est(code, E, 1) / 2) if own_step1 else 1.0)
    return F


def years(val, code='sm'):
    """OOS QLIKE по годам 2017–2021 (шаги 2–6) с Б0.1 кода code (raw — без поправки)."""
    v = dev_only(val)
    v = v[v.rv_actual > 1e-12]
    F = bias_factors(v, code, own_step1=False) if code != 'raw' else {}
    out = {}
    for st in range(2, DEV_MAX + 1):
        g = v[v.step == st]
        out[2015 + st] = qlike(g.rv_actual.values, g.rv_pred.values * F.get(st, 1.0))
    return out


def row(name, q, **kw):
    return dict(design=name, **{str(y): q[y] for y in sorted(q)}, m1720=float(np.mean([q[y] for y in range(2017, 2021)])), y2021=q[2021], **kw)


# ---------------------------------------------------------------- гибрид gibrid_v17 (OOS, эталон с той же s²)
SFX = {'single': '', 'top5': '_top5', 'seeds3': '_seeds4344', 'wrap2': '_wrap2', 'wrap': ''}


def write_boost(boost, xgb_df, lgb_df, sfx=''):
    """Входы гибрида: {boost}_{xgb|lgbm}_h5{sfx}_val.parquet, только шаги 1–6."""
    GIN.mkdir(parents=True, exist_ok=True)
    cols = ['date', 'ticker', 'year', 'step', 'rv_actual', 'rv_pred']
    for m, d in (('xgb', xgb_df), ('lgbm', lgb_df)):
        d = dev_only(d)
        extra = [c for c in ('seed',) if c in d.columns and sfx == '_seeds4344']
        d[cols + extra].to_parquet(GIN / f'{boost}_{m}_h5{sfx}_val.parquet', index=False)


def run_gibrid(boost, legs=('single', 'wrap'), log=None):
    out = GOUT / boost
    out.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, GIBRID_SAME_SLOT='1')
    cmd = [sys.executable, '-W', 'ignore', str(HERE / 'gibrid_v17.py'), '--boost', boost, '--h', '5', '--features', str(FEAT_V3),
           '--regen', str(GIN), '--legs', *legs, '--out', str(out), '--devlog', str(out / 'devlog.md')]
    with open(log or out / 'run.log', 'w') as f:
        rc = subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT, cwd=str(HERE.parents[1])).returncode
    if rc:
        raise RuntimeError(f'gibrid {boost}: код {rc}, см. {out / "run.log"}')
    return out / f'designs_val_{boost}_h5.csv'


def summarize(path, keep_har=False):
    """Как final3.summarize (без HAR-J/HAR-CJ V0 — общие эконометрические модели): по дизайну 2017–2020, 2021, c1–c4."""
    cols = ['design', 'step', 'val_year', 'q_hyb', 'q_har', 'q_xgb_single', 'q_lgb_single'] + [k + '_h' for k in CR]
    D = pd.read_csv(path, usecols=cols)
    assert D.loc[D.step > DEV_MAX, 'q_hyb'].isna().all()
    if not keep_har:
        D = D[~D.design.str.contains(r'\|V0_(?:HAR|HCJ)\|')]
    D = D[D.step.between(2, DEV_MAX)]
    q = D.pivot_table(index='design', columns='step', values='q_hyb', sort=False)
    fl = D.groupby('design', sort=False)[[k + '_h' for k in CR]].first()
    S = pd.DataFrame({'design': q.index, 'm1720': q[list(RANK_STEPS)].mean(axis=1).values, 'q2021': q[HOLD].values})
    for s in range(2, DEV_MAX + 1):
        S[str(2015 + s)] = q[s].values
    for k in CR:
        S[k] = fl.loc[S.design, k + '_h'].astype(bool).values
    S['n_failed'] = (~S[list(CR)]).sum(axis=1)
    S['failed'] = S.apply(lambda r: ','.join(k for k in CR if not r[k]), axis=1)
    return S.sort_values(['n_failed', 'm1720'], kind='stable').reset_index(drop=True)


def gib_eval(boost, xgb_df, lgb_df, legs=('single', 'wrap')):
    write_boost(boost, xgb_df, lgb_df)
    return summarize(run_gibrid(boost, legs))


def seeds_table(S_by_seed):
    """S_by_seed: {seed: summarize-таблица}. Дизайн выбирается по правилу на средних по seed: число нарушений (максимум по seed), затем
    средний 2017–2020. Возвращает (таблица по дизайнам, лучший дизайн)."""
    rows = []
    designs = set.intersection(*[set(S.design) for S in S_by_seed.values()])
    for d in designs:
        r = {s: S.set_index('design').loc[d] for s, S in S_by_seed.items()}
        rows.append(dict(design=d, n_failed=max(int(x.n_failed) for x in r.values()),
                         failed=','.join(sorted(set(','.join(x.failed for x in r.values() if x.failed).split(',')) - {''})),
                         m1720=float(np.mean([x.m1720 for x in r.values()])), q2021=float(np.mean([x.q2021 for x in r.values()])),
                         **{f'm1720_s{s}': float(x.m1720) for s, x in r.items()}, **{f'q2021_s{s}': float(x.q2021) for s, x in r.items()},
                         **{f'{y}': float(np.mean([x[str(y)] for x in r.values()])) for y in range(2017, 2022)}))
    T = pd.DataFrame(rows).sort_values(['n_failed', 'm1720'], kind='stable').reset_index(drop=True)
    return T, T.iloc[0]


def entry(per_seed_q, per_seed_ref=None, ref=D12_REF['m1720']):
    """Условие входа по 2017–2020: Δ̄ = среднее Q(s) / ref − 1 < −3 %; d_s = Q(s) − ref(s) (ref(s) — D12_h5 того же seed, если дан, иначе
    фиксированный финал 0,2159), σ = СКО d_s (ddof = 1); буквально |d̄| > 2σ и справочно t = d̄/(σ/√n) < −2 (формула Д24.3)."""
    q = np.asarray(per_seed_q, float)
    rr = np.asarray(per_seed_ref, float) if per_seed_ref is not None else np.full(len(q), ref)
    d = q - rr
    db, sd = float(d.mean()), float(d.std(ddof=1)) if len(d) > 1 else float('nan')
    rel = float(q.mean() / (rr.mean()) - 1)
    return dict(n=len(q), mean_q=float(q.mean()), ref=float(rr.mean()), rel=rel, d_mean=db, sigma=sd, two_sigma=2 * sd,
                pass3=rel < -0.03, pass2s=(db < 0) and (abs(db) > 2 * sd), t=db / (sd / np.sqrt(len(d))) if sd > 0 else float('-inf'))


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()
