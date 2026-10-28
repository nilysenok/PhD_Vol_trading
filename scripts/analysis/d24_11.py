"""Д24.11: калибровка Минцера-Зарновица поверх поправки s² на финалистах всех h.
Финалисты: кандидаты пула data/regen/final3/kandidaty/*.yaml (кроме D24_*), D12_h1_A_d812, D12_h5; h=22 - HAR-J с поправкой (B-sm).
Прогноз кандидата - его представитель по правилу третьего круга (final3.summarize по designs_val gibrid_v17 с ногами yaml), OOS:
val шага t - параметры, схема и s² шага t-1 (select: rv_pred_oos; V0: компонента с поправкой шага t-1). Восстанавливается в процессе
теми же функциями, что gibrid_v17.build_test (compute -> shrink -> select/v0); сверка QLIKE с designs_val.
Калибровка (только ПРОШЛЫЕ шаги): для val шага s - пары (ln y, ln ŷ_oos) val-лет шагов < s (с OOS-прогнозом); для теста шага s - шагов <= s.
  OLS (основной, «MZ»): ln y = a + b ln ŷ + e;  ŷ_cal = exp(a + b ln ŷ + s²_e/2) (s²_e - дисперсия остатков истории: поправка s² сохраняется);
  QL: (a, b) = argmin QLIKE(y, exp(a) ŷ^b) на истории (справка).
Варианты: пул (одни a, b) и по тикеру (свои a_i, b_i; меньше 200 наблюдений истории - пул). Нет истории - без калибровки.
Seed-проверка: тот же дизайн с ногой single на одиночных точках seed 42/43/44 (↺ для ног top5/wrap2/seeds3, как Д18).
Прошедший (≥ 1% к своему некалиброванному по 2017-2020, лучше на каждом seed, 2021 <= текущего лучшего кругов 1-2 и <= своего) -
кандидат D24_11_<имя>: калиброванный OOS-прогноз в обоих слотах (дизайн V0/bez воспроизводит его без повторной поправки).
python3 scripts/analysis/d24_11.py [--gibrid] [--build] [--leak] [--yaml]"""
import argparse, json, os, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np, pandas as pd
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parent))
import d24b_lib as L

OUT = L.OUT / 'd24_11'
R = L.R
SFX = {'single': '', 'top5': '_top5', 'seeds3': '_seeds4344', 'wrap2': '_wrap2'}
MINN = 200


def cands():
    import yaml
    C = []
    for f in sorted((R / 'final3' / 'kandidaty').glob('*.yaml')):
        y = yaml.safe_load(f.read_text())
        if str(y['boost']).startswith('D24_'):
            continue
        regen = Path(y.get('regen') or 'data/regen'); regen = regen if regen.is_absolute() else L.ROOT / regen
        h = int(y['h'])
        legs = y.get('legs') or [lg for lg in SFX if all((regen / f"{y['boost']}_{m}_h{h}{SFX[lg]}_val.parquet").exists() for m in ('xgb', 'lgbm'))]
        td = Path(y.get('test_dir') or f"data/regen/test_locked/{y['boost']}"); td = td if td.is_absolute() else L.ROOT / td
        C.append(dict(boost=y['boost'], h=h, regen=regen, legs=legs, test_dir=td, gib=OUT / 'gibrid' / y['boost']))
    for b, h in (('D12_h1_A_d812', 1), ('D12_h5', 5)):  # финалисты второго круга (прошли подтверждающий); гибрид - как в final3 (final2/gibrid)
        C.append(dict(boost=b, h=h, regen=R, legs=['single', 'top5', 'seeds3', 'wrap2'], test_dir=R / 'test_locked' / b, gib=R / 'final2' / 'gibrid' / b))
    return C


def gibrid(C, workers=2):
    def one(c):
        if (c['gib'] / f"designs_val_{c['boost']}_h{c['h']}.csv").exists():
            return 0
        c['gib'].mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, '-W', 'ignore', str(L.HERE / 'gibrid_v17.py'), '--boost', c['boost'], '--h', str(c['h']), '--features', str(L.FEAT),
               '--regen', str(c['regen']), '--legs', *c['legs'], '--out', str(c['gib']), '--devlog', str(c['gib'] / 'devlog.md')]
        with open(c['gib'] / 'run.log', 'w') as f:
            rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=str(L.REPO)).returncode
        print(('OK ' if rc == 0 else f'ОШИБКА {rc} ') + c['boost'], flush=True)
        return rc
    with ThreadPoolExecutor(workers) as ex:
        return list(ex.map(one, C))


def representative(c):
    import final3 as F3
    D = F3.designs(dict(gib_dir=c['gib'], gboost=c['boost'], h=c['h']))
    S = F3.summarize(dict(), D)
    return S.iloc[0], D


_G = {}


def _ctx(h):
    import gibrid_har as GH, gibrid_harcj as HC
    if 'A' not in _G:
        A, harj = GH.load_pit(L.FEAT)
        _G['A'], _G['harj'] = HC.add_cj(A), harj
    if h not in _G:
        A, harj = _G['A'], _G['harj']
        hv, ht, _ = GH.har_forecasts(h, A, harj)
        _G[h] = (hv, ht, dict(hcj=HC.forecasts(h, A), state=HC.state(A)))
    return _G[h]


def rebuild(boost, h, regen, design, test_dir=None, var_override=None):
    """OOS-прогноз дизайна на val шагов 1-9 (rv_oos; шаг 1 у select - NaN) и тест шагов 1-9 (если test_dir) - как gibrid_v17.build_test."""
    import gibrid_schemes as S, gibrid_designs as D
    from gibrid_compute import compute
    from gibrid_io import load_boost
    var, sel, lab, lam, *bl = design.split('|')
    var = var_override or var
    S.MODE = bl[0] if bl else 'rv'
    code = 'bez' if lab == 'bez' else lab[2:]
    hv, ht, ext = _ctx(h)
    if test_dir is not None:
        os.environ['GIBRID_TEST_DIR'] = str(test_dir)
    bx, bb = load_boost(boost, h, regen, var, test=test_dir is not None)
    os.environ.pop('GIBRID_TEST_DIR', None)
    res = compute(hv, ht if test_dir is not None else ht.iloc[:0], h, bx, bb, code, h, [], ext)
    rs = D.shrink(res, float(lam[1:]))
    ch, ov, ot = D.v0(rs, sel[3:].lower(), h) if sel.startswith('V0_') else D.select(rs, sel, h, 'oos')
    ov = ov.copy()
    ov['rv_oos'] = ov.rv_pred if sel.startswith('V0_') else ov.rv_pred_oos
    S.MODE = 'rv'
    return ov[L.COLS[:-1] + ['rv_pred', 'rv_oos']].reset_index(drop=True), (ot[L.COLS].reset_index(drop=True) if test_dir is not None else None), ch


def harj_corr(h, code='sm'):
    hv, ht, _ = _ctx(h)
    v, t = L.correct(hv, ht, code)
    v = v.assign(rv_oos=v.rv_pred)
    return v[L.COLS[:-1] + ['rv_pred', 'rv_oos']], t[L.COLS]


# ---------------------------------------------------------------- калибровка
def fit_cal(y, p, kind):
    ly, lp = np.log(y), np.log(p)
    if kind == 'OLS':
        X = np.column_stack([np.ones(len(lp)), lp])
        b, *_ = np.linalg.lstsq(X, ly, rcond=None)
        return float(b[0]), float(b[1]), float(np.var(ly - X @ b))
    f = lambda ab: float(np.mean(y / np.exp(ab[0] + ab[1] * lp) - (np.log(y) - ab[0] - ab[1] * lp) - 1))
    r = minimize(f, np.array([0.0, 1.0]), method='Nelder-Mead', options=dict(xatol=1e-6, fatol=1e-9, maxiter=2000))
    return float(r.x[0]), float(r.x[1]), 0.0


def apply_cal(p, prm):
    a, b, s2 = prm
    return np.exp(a + b * np.log(p) + s2 / 2)


def calibrate(ov, ot, kind, per_ticker):
    """ov: val (rv_actual, rv_oos по шагам 1-9), ot: тест или None. История - только прошлые шаги."""
    v = ov.copy(); v['rv_cal'] = v.rv_oos
    hist_ok = v.rv_oos.notna() & (v.rv_actual > 1e-12)
    prm_rows = []

    def params(mask):
        H = v[mask]
        if len(H) < MINN:
            return None, {}
        pool = fit_cal(H.rv_actual.values, H.rv_oos.values, kind)
        tk = {}
        if per_ticker:
            for t, g in H.groupby('ticker'):
                tk[t] = fit_cal(g.rv_actual.values, g.rv_oos.values, kind) if len(g) >= MINN else pool
        return pool, tk

    def put(df, idx, col_in, pool, tk):
        if pool is None:
            return
        p = df.loc[idx, col_in].values
        if per_ticker:
            out = np.array([apply_cal(x, tk.get(t, pool)) for x, t in zip(p, df.loc[idx, 'ticker'].values)])
        else:
            out = apply_cal(p, pool)
        return out
    for s in range(1, 10):
        idx = v.index[v.step == s]
        pool, tk = params(hist_ok & (v.step < s))
        r = put(v, idx, 'rv_oos', pool, tk)
        if r is not None:
            v.loc[idx, 'rv_cal'] = r
        prm_rows.append(dict(step=s, part='val', a=pool[0] if pool else np.nan, b=pool[1] if pool else np.nan, s2e=pool[2] if pool else np.nan))
    t = None
    if ot is not None:
        t = ot.copy(); t['rv_cal'] = t.rv_pred
        for s in range(1, 10):
            idx = t.index[t.step == s]
            pool, tk = params(hist_ok & (v.step <= s))
            r = put(t, idx, 'rv_pred', pool, tk)
            if r is not None:
                t.loc[idx, 'rv_cal'] = r
            prm_rows.append(dict(step=s, part='test', a=pool[0] if pool else np.nan, b=pool[1] if pool else np.nan, s2e=pool[2] if pool else np.nan))
    return v, t, pd.DataFrame(prm_rows)


def yq(v, col):
    d = v[(v.step >= 2) & (v.step <= 6) & (v.rv_actual > 1e-12) & v[col].notna()]
    q = {int(s) + 2015: L.qlike(g.rv_actual.values, g[col].values) for s, g in d.groupby('step')}
    return q, float(np.mean([q[y] for y in range(2017, 2021)]))


VARS = (('OLS', False), ('OLS', True), ('QL', False), ('QL', True))


def seed_sources(c):
    """seed -> (boost, regen) одиночной точки (нога single)."""
    b, h = c['boost'], c['h']
    if (c['regen'] / f'{b}_s42_xgb_h{h}_val.parquet').exists():
        return {s: (f'{b}_{s}', c['regen']) for s in ('s42', 's43', 's44')}
    if b == 'D18x_h22':
        return {f's{n}': (f'D18xs{n}_h22', R / 'D18' / 'seeds') for n in (42, 43, 44)}
    if all((c['regen'] / f'{b}_{m}_h{h}_seeds4344_val.parquet').exists() for m in ('xgb', 'lgbm')):
        d = OUT / 'seedfiles'; d.mkdir(parents=True, exist_ok=True)
        out = {'s42': (b, c['regen'])}
        for n in (43, 44):
            for m in ('xgb', 'lgbm'):
                f = d / f'{b}_s{n}_{m}_h{h}_val.parquet'
                if not f.exists():
                    x = pd.read_parquet(c['regen'] / f'{b}_{m}_h{h}_seeds4344_val.parquet')
                    x[x.seed == n].drop(columns=[k for k in ('seed', 'rank', 'model') if k in x]).to_parquet(f, index=False)
            out[f's{n}'] = (f'{b}_s{n}', d)
        return out
    return {}


def build():
    OUT.mkdir(parents=True, exist_ok=True)
    C = cands()
    rows, prms = [], []
    store = {}
    for c in C + [dict(boost='HARJ_corr', h=22)]:
        b, h = c['boost'], c['h']
        if b == 'HARJ_corr':
            ov, ot = harj_corr(22); design = 'single|V0_HAR|B-sm|L0'; qref = None
        else:
            if not (c['gib'] / f"designs_val_{b}_h{h}.csv").exists():
                print('нет designs_val, пропуск:', b, flush=True); continue
            x, D = representative(c)
            design = x.design
            ov, ot, ch = rebuild(b, h, c['regen'], design, c['test_dir'])
            g = D[D.design == design].set_index('step').q_hyb
            mine = yq(ov, 'rv_oos')[0]
            qref = max(abs(mine[y] - g[y - 2015]) for y in range(2017, 2022))
            assert qref < 1e-9, (b, design, qref)
        q0, m0 = yq(ov, 'rv_oos')
        base = dict(boost=b, h=h, design=design, check_q=qref)
        rows.append(dict(base, var='без калибровки', **{f'q_{y}': q0[y] for y in q0}, mean_1720=m0))
        for kind, pt in VARS:
            v, t, P = calibrate(ov, ot, kind, pt)
            q, m = yq(v, 'rv_cal')
            nm = f'MZ-{kind}-' + ('тикер' if pt else 'пул')
            rows.append(dict(base, var=nm, **{f'q_{y}': q[y] for y in q}, mean_1720=m))
            prms.append(P.assign(boost=b, var=nm))
            store[(b, nm)] = (v, t)
        store[(b, 'raw')] = (ov, ot, c, design)
        print(b, design, f'{m0:.4f}', flush=True)
    Q = pd.DataFrame(rows)
    Q['gain'] = Q.mean_1720 / Q.groupby('boost').mean_1720.transform('first') - 1
    Q.to_csv(OUT / 'mz_all.csv', index=False)
    pd.concat(prms).to_csv(OUT / 'mz_params.csv', index=False)
    with pd.option_context('display.width', 250):
        print(Q.drop(columns=['check_q']).round(4).to_string(), flush=True)
    # seed-проверка для вариантов с выигрышем >= 1% по 2017-2020 и 2021 не хуже своего
    srows = []
    for _, r in Q[(Q['var'] != 'без калибровки') & (Q.gain <= -0.01)].iterrows():
        b = r.boost
        if b == 'HARJ_corr':
            continue
        ov, ot, c, design = store[(b, 'raw')]
        kind, pt = r['var'].split('-')[1], r['var'].endswith('тикер')
        for s, (sb, sreg) in seed_sources(c).items():
            sv, _, _ = rebuild(sb, c['h'], sreg, design, None, var_override='single')
            q0, m0 = yq(sv, 'rv_oos')
            v, _, _ = calibrate(sv, None, kind, pt)
            q1, m1 = yq(v, 'rv_cal')
            srows.append(dict(boost=b, var=r['var'], seed=s, uncal_1720=m0, cal_1720=m1, uncal_2021=q0[2021], cal_2021=q1[2021], better=m1 < m0))
    S = pd.DataFrame(srows)
    S.to_csv(OUT / 'mz_seeds.csv', index=False)
    if len(S):
        print(S.round(4).to_string(), flush=True)
    # итог и кандидаты
    res = []
    for _, r in Q[Q['var'] != 'без калибровки'].iterrows():
        u = Q[(Q.boost == r.boost) & (Q['var'] == 'без калибровки')].iloc[0]
        ss = S[(S.boost == r.boost) & (S['var'] == r['var'])] if len(S) else S
        ok_seed = len(ss) == 3 and bool(ss.better.all())
        res.append(dict(boost=r.boost, h=r.h, var=r['var'], design=r.design, uncal_1720=u.mean_1720, cal_1720=r.mean_1720, gain=r.gain,
                        uncal_2021=u.q_2021, cal_2021=r.q_2021, inc_2021=L.INC[r.h][2], n_seed=len(ss), seeds_better=ok_seed,
                        prohodit=bool(r.gain <= -0.01 and ok_seed and r.q_2021 <= min(L.INC[r.h][2], u.q_2021) and r.boost != 'HARJ_corr')))
    E = pd.DataFrame(res); E.to_csv(OUT / 'mz_eval.csv', index=False)
    print(E.round(4).to_string(), flush=True)
    for _, r in E[E.prohodit].iterrows():
        ov, ot, c, design = store[(r.boost, 'raw')]
        v, t = store[(r.boost, r['var'])]
        nb = f"D24_11_{r.boost}_{r['var'].split('-')[1].lower()}{'t' if r['var'].endswith('тикер') else 'p'}".replace('_h', '_k', 1) + f"_h{r.h}"
        vv = v.assign(rv_pred=v.rv_cal.where(v.rv_cal.notna(), v.rv_pred))[L.COLS]
        tt = t.assign(rv_pred=t.rv_cal)[L.COLS]
        L.save_cand(nb, r.h, vv, vv, tt, tt)
        print('кандидат', nb, flush=True)


def leak():
    """Параметры калибровки для val шага s не меняются при порче прогнозов и целей шагов >= s (история - только прошлые шаги)."""
    rng = np.random.default_rng(0)
    c = next(x for x in cands() if x['boost'] == 'D12_h5')
    x, _ = representative(c)
    ov, _, _ = rebuild(c['boost'], c['h'], c['regen'], x.design)
    res = []
    for kind, pt in VARS:
        v0, _, P0 = calibrate(ov, None, kind, pt)
        for s in range(3, 10):
            B = ov.copy(); f = B.step >= s
            B.loc[f, 'rv_oos'] *= rng.uniform(0.2, 5, f.sum()); B.loc[f, 'rv_actual'] *= rng.uniform(0.2, 5, f.sum())
            v1, _, P1 = calibrate(B, None, kind, pt)
            p0, p1 = P0[(P0.part == 'val') & (P0.step == s)].iloc[0], P1[(P1.part == 'val') & (P1.step == s)].iloc[0]
            d = float(max(abs(p0.a - p1.a), abs(p0.b - p1.b), abs(p0.s2e - p1.s2e)))
            # прогноз шага s после калибровки зависит только от своего rv_oos: на неиспорченной копии - то же, что v0
            res.append(dict(var=f'{kind}-{"тикер" if pt else "пул"}', step=s, max_dparam=d, ok=d < 1e-12))
    D = pd.DataFrame(res); D.to_csv(OUT / 'leak.csv', index=False); print(D.to_string(), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gibrid', action='store_true'); ap.add_argument('--build', action='store_true'); ap.add_argument('--leak', action='store_true')
    ap.add_argument('--workers', type=int, default=2)
    a = ap.parse_args()
    if a.gibrid:
        gibrid(cands(), a.workers)
    if a.build:
        build()
    if a.leak:
        leak()


if __name__ == '__main__':
    main()
