#!/usr/bin/env python3
"""Д14: стабилизация коррекции остаточного ученика V9 (h=22 основное, h=5 проверочно). Без нового обучения: по сохранённым прогнозам
бустинга и HAR-J (основа V9) восстанавливается f = ln ŷ - ln HAR и строится ŷ' = HAR * exp(alpha * clip(f, -c, c)),
f = 0, если ln RV_m(t) вне [1-й; 99-й] перцентилей обучающего окна шага (защита, prot=1).
Каждая комбинация (alpha, c, prot) применяется к КАЖДОЙ точке/seed-прогнозу (до усреднения ноги), затем — гибрид gibrid_v17.run_h
(oos, шаги 2-6, эталон — модели с той же s2) на ноге seeds3 (h=22) / wrap2 (h=5). Подтверждающий набор и тест не открываются:
для оценки в файлы пишутся шаги 1-6; на шаги 7-9 и тест параметры только применяются (--final), метрики не считаются.
  python3 -W ignore scripts/analysis/d14_run.py grid  --cand D12_h22 [--workers 2]
  python3 -W ignore scripts/analysis/d14_run.py select --cand D12_h22
  python3 -W ignore scripts/analysis/d14_run.py final --cand D12_h22 --alpha 1 --c inf --prot 0 --out D14_h22"""
import argparse, itertools, sys, time, multiprocessing as mp, argparse as _ap
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gibrid_har as GH
import gibrid_har_spec as HS
import gibrid_harcj as HC
import gibrid_designs as D
import gibrid_v17 as V
from gibrid_io import MAIN_REGEN as R

OUT = R / 'D14'
FEAT = R / 'features_pit_v3.parquet'
K = ['date', 'ticker', 'step']
MODELS = ('xgb', 'lgbm')
ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)
CS = (0.05, 0.1, 0.2, 0.3, np.inf)
CAND = {
    'D12_h22': dict(h=22, base='ols', leg='seeds3', fixed='seeds3|V0_XGB|B-smrob|L0', out='D14_h22'),
    'D12_h5': dict(h=5, base='huber', leg='wrap2', fixed='wrap2|V0_XGB|B-sm|L0', out='D14_h5'),
    'F2m_h5': dict(h=5, base='ols', leg='wrap2', fixed='wrap2|V0_LGB|B-smrob|L0', out='D14_h5F2m'),
}
G = {}


# ------------------------------------------------------------------ основа V9, диапазоны обучающего окна
def split(A, h, ty):
    t = f'rv_target_h{h}'
    tr = A[A.year < ty]
    vy = sorted(tr.year.unique())[-1]
    return tr[tr.year != vy].dropna(subset=[t]), tr[tr.year == vy].dropna(subset=[t]), A[A.year == ty], vy


def har_models(A, harj, h, base):
    """ln основы (HAR-J: МНК или HuberRegressor — как смещение V9) на val-годе и тесте шагов 1-10, диапазоны ln RV_m окна."""
    from sklearn.linear_model import LinearRegression, HuberRegressor
    t = f'rv_target_h{h}'
    X = lambda d: GH.clean(np.log(d[harj].values.clip(1e-10)))
    rows = {'val': [], 'test': []}
    rng = {}
    for step, ty in enumerate(range(2017, 2027), 1):
        tp, vi, td, vy = split(A, h, ty)
        if step <= 9:
            td = td.dropna(subset=[t])
        m = (HuberRegressor() if base == 'huber' else LinearRegression()).fit(X(tp), np.log(tp[t].values + 1e-10))
        lr = np.log(tp['rv_m'].clip(lower=1e-10).values)
        lr = lr[np.isfinite(lr)]
        rng[step] = (np.percentile(lr, 1), np.percentile(lr, 99))
        for k, d in (('val', vi), ('test', td)):
            rows[k].append(pd.DataFrame({'date': d['date'].values, 'ticker': d['ticker'].values, 'step': step,
                                         'ln_har': m.predict(X(d)), 'ln_rvm': np.log(d['rv_m'].clip(lower=1e-10).values)}))
    out = {}
    for k in rows:
        B = pd.concat(rows[k], ignore_index=True)
        B['out'] = [(not (rng[s][0] <= v <= rng[s][1])) for s, v in zip(B.step, B.ln_rvm)]
        out[k] = B
    return out, rng


def tf(df, B, al, c, prot):
    """ŷ' = exp(ln HAR + alpha * clip(f, -c, c)); prot: f = 0 вне диапазона обучающего окна."""
    m = df.merge(B[K + ['ln_har', 'out']], on=K, how='left')
    assert m.ln_har.notna().all(), ('нет основы', int(m.ln_har.isna().sum()))
    f = np.log(np.clip(m.rv_pred.values, 1e-10, None)) - m.ln_har.values
    if prot:
        f = np.where(m.out.values, 0.0, f)
    f = al * np.clip(f, -c, c)
    return df.assign(rv_pred=np.clip(np.exp(m.ln_har.values + f), 1e-10, None))


# ------------------------------------------------------------------ источники
def src(T, h, m, kind, k):
    sfx = {'single': '', 'top5': '_top5', 'seeds': '_seeds4344'}[kind]
    p = (R if k == 'val' else R / 'test_locked' / T) / f'{T}_{m}_h{h}{sfx}_{k}.parquet'
    return pd.read_parquet(p)


def wrap_recipe(T, h):
    w = pd.read_csv(R / 'final2' / f'wrap2_{T}_h{h}.csv')
    return tuple(x for x in str(w.final.iloc[0]).split('+') if x != 'none'), w.model.iloc[0]


def wrap2(S, comps, prim):
    """S: {m: single (seed 42)}; обёртка второго круга для компонентов '2' (среднее LGBM и XGB по RV) или none; только single-члены
    (как final2_wrap.build без 1a/1b)."""
    assert not (set(comps) - {'2'}), comps
    out = {m: S[m] for m in MODELS}
    if '2' in comps:
        a, b = S['lgbm'], S['xgb'][K + ['rv_pred']]
        mm = a.merge(b, on=K, suffixes=('_l', '_x'), validate='one_to_one')
        out[prim] = mm.assign(rv_pred=(mm.rv_pred_l + mm.rv_pred_x) / 2).drop(columns=['rv_pred_l', 'rv_pred_x'])
    return out


def single42(df):
    return df[df.seed == 42] if 'seed' in df else df


# ------------------------------------------------------------------ гибрид на одной комбинации
def combo_id(al, c, prot):
    return f'a{al:g}_c{c:g}_p{prot}'


def run_combo(args):
    cid, al, c, prot = args
    cfg, T, h = G['cfg'], G['T'], G['h']
    b = f'D14x_{G["T"]}_{cid}'
    d = G['tmp'] / b
    d.mkdir(parents=True, exist_ok=True)
    S1 = {m: tf(G['single'][m], G['B']['val'], al, c, prot) for m in MODELS}
    for m in MODELS:
        S1[m].drop(columns=[x for x in ('trial', 'rank') if x in S1[m]]).to_parquet(d / f'{b}_{m}_h{h}_val.parquet', index=False)
    if cfg['leg'] == 'seeds3':
        for m in MODELS:
            tf(G['seeds'][m], G['B']['val'], al, c, prot).to_parquet(d / f'{b}_{m}_h{h}_seeds4344_val.parquet', index=False)
    else:
        comps, prim = G['recipe']
        W = wrap2(S1, comps, prim)
        for m in MODELS:
            W[m].assign(wrap='+'.join(comps) or 'none').sort_values(K).to_parquet(d / f'{b}_{m}_h{h}_wrap2_val.parquet', index=False)
    a = _ap.Namespace(boost=b, legs=[cfg['leg']], blend=['rv'], dev_eval='oos', etalon='same', v7_lag='h', features=str(FEAT))
    V.LO = 2
    out = d / 'out'
    out.mkdir(exist_ok=True)
    chs = run_light(h, G['A'], G['harj'], a, d)
    rows = []
    for nm, ch in chs.items():
        x = ch[['step', 'val_year', 'q_hyb', 'q_har', 'q_xgb_single', 'q_lgb_single', 'scheme', 'w_har', 'w_xgb', 'w_lgb', 'v1_ml']].copy()
        x.insert(0, 'design', nm)
        rows.append(x)
    import shutil
    shutil.rmtree(d, ignore_errors=True)
    return cid, pd.concat(rows, ignore_index=True)


def run_light(h, A, harj, a, regen):
    """Копия цикла gibrid_v17.run_h (нога a.legs[0], смешивание rv, 5 поправок x L0/L0.25/L0.5 x step/minimax/g3 + V0 x 4) без записи файлов
    и справочных блоков; критерии/таблицы дизайнов — те же D.select / D.v0 / D.flags."""
    import gibrid_schemes as S
    from gibrid_compute import compute
    from gibrid_io import load_boost
    hv, ht, _ = GH.har_forecasts(h, A, harj)
    ht = ht.iloc[:0]
    ext = dict(hcj=HC.forecasts(h, A), state=HC.state(A))
    lag = h
    chs = {}

    def add(ch, ov, ot, name):
        ch.insert(0, 'design', name)
        ch['ratio_har'] = ch.q_hyb / ch.q_har
        chs[name] = ch
    var = a.legs[0]
    bx, bl = load_boost(a.boost, h, regen, var)
    S.MODE = 'rv'
    for code in ('ord', 'rob', 'sm', 'smrob', 'bez'):
        res = compute(hv, ht, h, bx, bl, code, lag, [], ext)
        if code == 'ord':
            etalon = [{k: r['base'][k] for k in ('q_har', 'q_xgb_single', 'q_lgb_single')} for r in res]
        elif code == 'bez':
            for r, e in zip(res, etalon):
                r['base'].update(e)
        lab = 'bez' if code == 'bez' else 'B-' + code
        for lam in V.LAMS:
            rs = D.shrink(res, lam)
            for sel in ('step', 'minimax', 'g3'):
                add(*D.select(rs, sel, h, a.dev_eval), f'{var}|{sel}|{lab}|L{lam:g}')
        for comp in ('har', 'xgb', 'lgb', 'hcj'):
            add(*D.v0(res, comp, h), f'{var}|V0_{comp.upper()}|{lab}|L0')
    return chs


def setup(cand):
    cfg = CAND[cand]
    h = cfg['h']
    A, harj = GH.load_pit(FEAT)
    A = HC.add_cj(A)
    harj_ = harj
    # кэш медленных вызовов, не зависящих от прогнозов бустинга
    cache = {}

    def memo(fn, name):
        def w(*a):
            key = (name,) + tuple(x if isinstance(x, (int, str, tuple)) else id(x) for x in a)
            if key not in cache:
                cache[key] = fn(*a)
            return cache[key]
        return w
    GH.har_forecasts = memo(GH.har_forecasts, 'hf')
    HC.forecasts = memo(HC.forecasts, 'hcf')
    HC.state = memo(HC.state, 'hcs')
    HS.run = memo(HS.run, 'hs')
    B, rng = har_models(A, harj_, h, cfg['base'])
    G.update(cfg=cfg, T=cand, h=h, A=A, harj=harj, B=B, rng=rng, tmp=OUT / 'tmp')
    G['tmp'].mkdir(parents=True, exist_ok=True)
    return cfg


def load_eval_sources(cand):
    T, h = cand, G['h']
    sing, seeds = {}, {}
    for m in MODELS:
        s = single42(src(T, h, m, 'single', 'val'))
        sing[m] = s[s.step <= 6].reset_index(drop=True)
        if G['cfg']['leg'] == 'seeds3':
            q = src(T, h, m, 'seeds', 'val')
            seeds[m] = q[q.step <= 6].reset_index(drop=True)
    G.update(single=sing, seeds=seeds)
    if G['cfg']['leg'] == 'wrap2':
        G['recipe'] = wrap_recipe(T, h)


def check_base(cand):
    """Контроль основы: ln HAR из d14 vs ln hv gibrid_har (МНК) на val; для h=22 — vs сохранённый har_ln (D12/h22/v3base_val)."""
    h = G['h']
    hv, ht, _ = GH.har_forecasts(h, G['A'], G['harj'])
    m = hv.merge(G['B']['val'][K + ['ln_har']], on=K)
    d = float(np.abs(np.log(m.rv_pred) - m.ln_har).max())
    print(f'контроль основы: макс. |ln hv(gibrid) - ln основа| = {d:.3e} (base={G["cfg"]["base"]}), строк {len(m)}', flush=True)
    s = G['single']['xgb'].merge(G['B']['val'][K + ['ln_har']], on=K)
    f = np.log(s.rv_pred) - s.ln_har
    print(f'f(xgb, single, val 1-6): медиана {f.median():.3f}, std {f.std():.3f}, 1%/99% {f.quantile(.01):.2f}/{f.quantile(.99):.2f}, |f|>0.3: {(f.abs() > .3).mean():.1%}, '
          f'доля вне диапазона окна: {s.merge(G["B"]["val"][K + ["out"]], on=K).out.mean():.1%}', flush=True)


def cmd_grid(a):
    cand = a.cand
    t0 = time.time()
    setup(cand)
    load_eval_sources(cand)
    check_base(cand)
    combos = [('a0_c0_p0', 0.0, np.inf, 0)] + [(combo_id(al, c, p), al, c, p) for al in ALPHAS[1:] for c in CS for p in (0, 1)]
    # базовая (текущая) — a1_cinf_p0: первой, чтобы прогреть кэши до fork
    combos.sort(key=lambda x: (x[0] != combo_id(1.0, np.inf, 0)))
    first = run_combo(combos[0])
    print(f'базовая комбинация посчитана за {time.time() - t0:.0f} с', flush=True)
    res = [first]
    if a.workers > 1:
        with mp.get_context('fork').Pool(a.workers) as P:
            for r in P.imap_unordered(run_combo, combos[1:]):
                res.append(r); print(f'  {r[0]} готово ({len(res)}/{len(combos)}, {time.time() - t0:.0f} с)', flush=True)
    else:
        for x in combos[1:]:
            r = run_combo(x); res.append(r); print(f'  {r[0]} готово ({len(res)}/{len(combos)}, {time.time() - t0:.0f} с)', flush=True)
    L = pd.concat([r[1].assign(combo=r[0]) for r in res], ignore_index=True)
    OUT.mkdir(parents=True, exist_ok=True)
    L.to_parquet(OUT / f'grid_raw_{cand}.parquet', index=False)
    print(f'сетка {cand}: {len(combos)} комбинаций, {time.time() - t0:.0f} с', flush=True)


# ------------------------------------------------------------------ отбор
def stats(ch):
    """ch: строки одного дизайна шагов 1-6 -> флаги на шагах 2-6 + q."""
    d = ch[ch.step >= 2].sort_values('step')
    fl = D.flags(d)
    q = d.set_index('step').q_hyb
    return dict(c1=int(fl['c1']), c2=int(fl['c2']), c3=int(fl['c3']), c4=int(fl['c4']), n_failed=4 - int(fl['c1']) - int(fl['c2']) - int(fl['c3']) - int(fl['c4']),
                q2017_2020=float(q.loc[2:5].mean()), q2021=float(q.loc[6]), q2017_2021=float(q.loc[2:6].mean()),
                worst_ratio=float(fl['worst_ratio']), worst_ratio_ml=float(fl['worst_ratio_single']))


def parse_cid(cid):
    a, c, p = cid.split('_')
    return float(a[1:]), float(c[1:]), int(p[1:])


def cmd_select(a):
    cand = a.cand
    cfg = CAND[cand]
    L = pd.read_parquet(OUT / f'grid_raw_{cand}.parquet')
    rows, per = [], {}
    for cid, g in L.groupby('combo'):
        al, c, p = parse_cid(cid)
        S = {nm: stats(x) for nm, x in g.groupby('design')}
        per[cid] = S
        # дизайн по правилу гибрида (минимум нарушений c1-c4 на 2017-2021, затем минимум q 2017-2020)
        best = min(S, key=lambda nm: (S[nm]['n_failed'], S[nm]['q2017_2020']))
        fx = S[cfg['fixed']]
        gg = g[g.design == cfg['fixed']].set_index('step')
        r = dict(combo=cid, alpha=al, c=c, prot=p, rule_design=best, **{f'rule_{k}': v for k, v in S[best].items()},
                 **{f'fix_{k}': v for k, v in fx.items()},
                 fix_q2017_2020_har=float(gg.loc[2:5, 'q_har'].mean()), fix_q2017_2020_xgbsingle=float(gg.loc[2:5, 'q_xgb_single'].mean()),
                 fix_q2017_2020_lgbsingle=float(gg.loc[2:5, 'q_lgb_single'].mean()),
                 fix_q2021_har=float(gg.loc[6, 'q_har']), fix_q2021_xgbsingle=float(gg.loc[6, 'q_xgb_single']), fix_q2021_lgbsingle=float(gg.loc[6, 'q_lgb_single']))
        rows.append(r)
    T = pd.DataFrame(rows)
    T.loc[T.combo == 'a0_c0_p0', ['combo', 'c']] = ['a0_cinf_p0', np.inf]  # alpha=0: c и защита не влияют (f = 0)
    base = T[T.combo == combo_id(1.0, np.inf, 0)].iloc[0]
    # устойчивость порядка: комбинации — по (нарушения, q 2017-2020 с допуском 0,1%, меньший alpha, меньший c, защита выкл)
    TOL = 0.001

    def order(df, nf, q):
        df = df.sort_values([nf, q]).reset_index(drop=True)
        out, rest = [], df
        while len(rest):
            top = rest[rest[nf] == rest[nf].min()]
            qm = top[q].min()
            tie = top[top[q] <= qm * (1 + TOL)].sort_values(['alpha', 'c', 'prot'])
            out.append(tie.iloc[0]); rest = rest.drop(tie.index[0])
        return pd.DataFrame(out)

    ordr = order(T, 'rule_n_failed', 'rule_q2017_2020')
    ordf = order(T, 'fix_n_failed', 'fix_q2017_2020')
    T['rank_rule'] = T.combo.map({c: i + 1 for i, c in enumerate(ordr.combo)})
    T['rank_fixed'] = T.combo.map({c: i + 1 for i, c in enumerate(ordf.combo)})
    # дубликаты alpha=0: все c/prot дают одно и то же (f = 0) — размножаем в таблице сетки
    z = T[T.alpha == 0].iloc[0]
    ext = []
    for c in CS:
        for p in (0, 1):
            if (c, p) != (np.inf, 0):
                ext.append(dict(z, combo=combo_id(0.0, c, p), c=c, prot=p, rank_rule=np.nan, rank_fixed=np.nan, dup_of='a0_cinf_p0'))
    T = pd.concat([T, pd.DataFrame(ext)], ignore_index=True)
    T['current'] = (T.combo == base.combo)
    T = T.sort_values(['alpha', 'c', 'prot']).reset_index(drop=True)
    # результаты
    OUT.mkdir(parents=True, exist_ok=True)
    T.to_csv(OUT / f'grid_h{cfg["h"]}{"" if cand != "F2m_h5" else "_F2m"}.csv', index=False, float_format='%.6g')
    print(f'текущая ({base.combo}): правило {base.rule_design} q2017-20 {base.rule_q2017_2020:.5f} q2021 {base.rule_q2021:.5f} нарушено {int(base.rule_n_failed)}; '
          f'фикс.дизайн {cfg["fixed"]}: q2017-20 {base.fix_q2017_2020:.5f}, q2021 {base.fix_q2021:.5f}, нарушено {int(base.fix_n_failed)}')
    cols = ['combo', 'rule_design', 'rule_n_failed', 'rule_q2017_2020', 'rule_q2021', 'fix_n_failed', 'fix_q2017_2020', 'fix_q2021']
    print('--- топ-8 по дизайну-правилу'); print(ordr.head(8)[cols].to_string())
    print('--- топ-8 по фиксированному дизайну'); print(ordf.head(8)[cols].to_string())
    # ходьба вперёд: комбинация шага t выбирается по q фиксированного дизайна на val-году шага t-1 (шаг 2 — по шагу 1)
    pc = L[L.design == cfg['fixed']].pivot(index='step', columns='combo', values='q_hyb')
    wf, rows2 = {}, []
    ad = {cid: parse_cid(cid) for cid in pc.columns}
    for t in range(2, 7):
        prev = pc.loc[t - 1]
        qm = prev.min()
        tie = [cid for cid in prev.index if prev[cid] <= qm * (1 + TOL)]
        wf[t] = min(tie, key=lambda cid: (ad[cid][0], ad[cid][1], ad[cid][2]))
    ch = []
    for t in range(2, 7):
        r = L[(L.design == cfg['fixed']) & (L.combo == wf[t]) & (L.step == t)].iloc[0]
        ch.append(r)
    ch = pd.DataFrame(ch)
    fl = D.flags(ch.assign(scheme=ch.scheme))
    qwf = ch.set_index('step').q_hyb
    print('--- ходьба вперёд (выбор на шаге t по val-году t-1):', {t: wf[t] for t in wf})
    print(f'ходьба вперёд, фикс. дизайн: q2017-20 {qwf.loc[2:5].mean():.5f}, q2021 {qwf.loc[6]:.5f}, c1-c4 {[int(fl[k]) for k in ("c1","c2","c3","c4")]}')
    pd.DataFrame([dict(step=t, combo=wf[t], q=float(qwf.loc[t])) for t in wf]).to_csv(OUT / f'wf_h{cfg["h"]}{"" if cand != "F2m_h5" else "_F2m"}.csv', index=False)
    top = ordr.iloc[0]
    print('ВЫБОР (дизайн-правило, 2017-2020):', top.combo, '| 2021 не хуже текущего:', bool(top.rule_q2021 <= base.rule_q2021 * (1 + 1e-12)),
          f'| фикс. дизайн q2021 {top.fix_q2021:.5f} vs текущей {base.fix_q2021:.5f}')


# ------------------------------------------------------------------ итоговые файлы
def cmd_final(a):
    cand = a.cand
    cfg = setup(cand)
    T, h = cand, cfg['h']
    al, c, prot = a.alpha, float(a.c), a.prot
    name = a.out or cfg['out']
    lock = R / 'test_locked' / name
    lock.mkdir(parents=True, exist_ok=True)
    comps, prim = wrap_recipe(T, h)
    for k in ('val', 'test'):
        S1, P = {}, {}
        for m in MODELS:
            for kind in ('single', 'top5', 'seeds'):
                d = src(T, h, m, kind, k)
                o = tf(d, G['B'][k], al, c, prot)
                sfx = {'single': '', 'top5': '_top5', 'seeds': '_seeds4344'}[kind]
                dst = (R / f'{name}_{m}_h{h}{sfx}_val.parquet') if k == 'val' else (lock / f'{name}_{m}_h{h}{sfx}_test.parquet')
                o.sort_values(K).to_parquet(dst, index=False)
                if kind == 'single':
                    S1[m] = single42(o)
        W = wrap2(S1, comps, prim)
        for m in MODELS:
            dst = (R / f'{name}_{m}_h{h}_wrap2_val.parquet') if k == 'val' else (lock / f'{name}_{m}_h{h}_wrap2_test.parquet')
            W[m].assign(wrap='+'.join(comps) or 'none').sort_values(K).to_parquet(dst, index=False)
        print(f'{k}: записано (шаги 1-9)', flush=True)
    # шаг 10 (val 2025 и тест 2026): основа и f пересчитываются так же
    s10 = {}
    for k in ('val', 'test'):
        Bk = G['B'][k]
        for m in MODELS:
            p = (R / 'final2' / 'step10' / f'{T}_{m}_h{h}_s10_val.parquet') if k == 'val' else (R / 'test_locked' / 'final2' / 'step10' / f'{T}_{m}_h{h}_s10_test.parquet')
            if not p.exists():
                continue
            d = pd.read_parquet(p)
            o = tf(d, Bk, al, c, prot)
            dst = (R / f'{name}_{m}_h{h}_s10_val.parquet') if k == 'val' else (lock / f'{name}_{m}_h{h}_s10_test.parquet')
            o.to_parquet(dst, index=False)
            s10[(k, m)] = len(o)
    print('шаг 10:', s10, flush=True)
    n = {}
    for k in ('val', 'test'):
        B = G['B'][k]
        n[k] = dict(rows=len(B), frac_outside_window=float(B.out.mean()))
    print('доля строк вне диапазона окна (по всей основе):', n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['grid', 'select', 'final'])
    ap.add_argument('--cand', required=True, choices=list(CAND))
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--alpha', type=float, default=1.0); ap.add_argument('--c', default='inf'); ap.add_argument('--prot', type=int, default=0)
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    {'grid': cmd_grid, 'select': cmd_select, 'final': cmd_final}[a.cmd](a)


if __name__ == '__main__':
    main()
