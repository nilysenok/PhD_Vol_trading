#!/usr/bin/env python3
"""Д25.3 — наклон срочной структуры на h=22: цель s = ln(RV̄₂₂ / RV̄₅), RV̄_h — среднее RV за [t+1, t+h] (rv_target_mean_h{5,22}).
Модели наклона: (1) HAR-J на наклон — OLS по ln-регрессорам HAR-J (rv_d, rv_w, rv_m, скачки; как gibrid_har), окно шага — годы < val-года
с эмбарго 22 торговых дня; (2) бустинг на наклон — протокол D15_h22: точки optuna-30 шага (D15_h22_selected_h22.csv, seed 42; шаги 7–9 —
точка шага 6), переобучение с seed 42/43/44, признаки features_pit_v3 (чистый V0, cfg {}), исключения Bv2b, ES на val-годе шага;
метка = ln RV̄₂₂ − ln RV̄₅ (через смещение l2_lib.fit). Прогноз бустинга наклона seed s = среднее ln-прогнозов XGB и LGBM seed s.
Прогноз h=22: ŷ₂₂ = hyb₅ · exp(ŝ + s²/2), hyb₅ — OOS-прогноз финала h=5 D12_h5 (wrap2/V0_XGB/B-sm/L0), s² — дисперсия OOS-остатков наклона
val-года шага t−1 (тест шага t — шага t). Для gibrid_v17 (c1–c4, гибрид Б) пишется сырой вариант hyb₅,raw · exp(ŝ): XGB-слот — наклон HAR-J,
LGB-слот — наклон бустинга seed s; поправку B-sm делает gibrid (одна оценка s², как у эталона).
python3 scripts/analysis/d25_3.py fit | build | gibrid | eval | leak"""
import os, sys, argparse, json, subprocess, time
from pathlib import Path
os.environ.setdefault('L2_FEAT', '/Users/nilysenok/Desktop/Старое/PhD_Vol_trading/data/regen/features_pit_v3.parquet')
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import d25_4 as L4
from d25_4 import R, OUT, LOCK, KEY, CRIT, RANK, HOLD, qlike

SEEDS = (42, 43, 44)
EMB = 22
SCR = OUT / 'd25_3'


# ---------------------------------------------------------------- HAR-J на наклон
def har_slope(A=None, harj=None, plant=False):
    """-> DataFrame KEY + year, slope_actual, slope_hat для val (шаги 1–9) и test (шаги 1–9). cut — для детектора: данные с даты cut портятся."""
    import gibrid_har as GH
    if A is None:
        A, harj = GH.load_pit(L4.FEAT)
    A = A.sort_values(['date', 'ticker']).reset_index(drop=True)
    A['slope'] = np.log(A['rv_target_mean_h22'] + 1e-10) - np.log(A['rv_target_mean_h5'] + 1e-10)
    X = lambda d: GH.clean(np.log(d[harj].values.clip(1e-10)))
    vals, tests = [], []
    for step, ty in enumerate(range(2017, 2026), 1):
        tr = A[A.year < ty]
        vy = ty - 1
        tp = tr[tr.year < (vy + 1 if plant else vy)].dropna(subset=['slope'])  # plant — только самопроверка детектора
        # эмбарго: последние EMB торговых дней перед val-годом (их цель заходит в val-год)
        dates = np.sort(tp.date.unique())
        tp = tp[tp.date < dates[-EMB]] if len(dates) > EMB else tp
        vi = tr[tr.year == vy].dropna(subset=['slope'])
        td = A[A.year == ty]
        from sklearn.linear_model import LinearRegression
        m = LinearRegression().fit(X(tp), tp.slope.values)
        for df, lst, yr in ((vi, vals, vy), (td, tests, ty)):
            lst.append(pd.DataFrame({'date': df.date.values, 'ticker': df.ticker.values, 'year': yr, 'step': step,
                                     'slope_actual': df.slope.values if lst is vals else np.nan, 'slope_hat': m.predict(X(df))}))
    return pd.concat(vals, ignore_index=True), pd.concat(tests, ignore_index=True)


# ---------------------------------------------------------------- бустинг на наклон (протокол D15_h22)
G = {}


def _init():
    import l2_lib as L
    A, FC = L.load_base(None, (), 22)
    A2, cols, _, _ = L.build_features(A, FC, 22, {})
    G.update(A=A2, cols=cols)


def _task(t):
    import l2_lib as L
    import tune_L2_optuna as T
    model, step, seed = t
    sel = pd.read_csv(R / 'D15_h22_selected_h22.csv')
    r = sel[(sel.model == model) & (sel.step == step) & (sel.seed == 42)].iloc[0]
    p = {k: (int(r[k]) if sc in ('int', 'logint') else float(r[k])) for k, _, _, sc in T.SPACEB[model]}
    D = L.step_data(G['A'], G['cols'], 22, step, model, {}, 'Bv2b')
    off5 = {k: np.log(D['parts'][k]['rv_target_mean_h5'].values + 1e-10) for k in ('tr', 'va')}
    for k in ('tr', 'va'):
        o = off5[k]
        D['off'][k] = np.where(np.isfinite(o), o, np.nanmedian(o))
    D['off']['te'] = np.zeros(len(D['parts']['te']))  # тест: прогноз наклона без чтения целей тестового года
    ref = L.make_ref(model, G['A'], G['cols'], 22, {}, 'Bv2b') if step > 1 else None
    pv, pt, bi, _, _ = L.fit(model, p, D, {}, seed=seed, ref=ref)
    sv, st = pv - D['off']['va'], pt
    va, te = D['parts']['va'], D['parts']['te']
    fv = pd.DataFrame({'date': va.date.values, 'ticker': va.ticker.values, 'year': D['vy'], 'step': step, 'model': model, 'seed': seed, 'slope_hat': sv})
    ft = pd.DataFrame({'date': te.date.values, 'ticker': te.ticker.values, 'year': D['ty'], 'step': step, 'model': model, 'seed': seed, 'slope_hat': st})
    return fv, ft, bi


def fit_boost(workers=3):
    import multiprocessing as mp
    _init()
    tasks = [(m, s, sd) for sd in SEEDS for m in ('lgbm', 'xgb') for s in range(1, 10)]
    t0 = time.time()
    with mp.get_context('fork').Pool(workers) as P:
        res = P.map(_task, tasks, chunksize=1)
    SCR.mkdir(parents=True, exist_ok=True)
    pd.concat([r[0] for r in res]).to_parquet(SCR / 'boost_slope_val.parquet', index=False)
    LOCK.joinpath('D25_3').mkdir(parents=True, exist_ok=True)
    pd.concat([r[1] for r in res]).to_parquet(LOCK / 'D25_3' / 'boost_slope_test.parquet', index=False)
    print(f'бустинг наклона: {len(tasks)} обучений, {time.time() - t0:.0f} с', flush=True)


def fit(a):
    hv, ht = har_slope()
    SCR.mkdir(parents=True, exist_ok=True)
    hv.to_parquet(SCR / 'har_slope_val.parquet', index=False)
    (LOCK / 'D25_3').mkdir(parents=True, exist_ok=True)
    ht.to_parquet(LOCK / 'D25_3' / 'har_slope_test.parquet', index=False)
    fit_boost(a.workers)


# ---------------------------------------------------------------- сборка ŷ₂₂
def hyb5():
    """OOS-прогноз финала h=5 (D12_h5 wrap2/V0_XGB/B-sm/L0): val — с поправкой шага t−1 (corr) и сырой (raw); test — финальный файл final3."""
    from gibrid_io import load_boost
    v, _, _ = L4.design_oos('D12_h5', 5, 'wrap2|V0_XGB|B-sm|L0')
    bx, _ = load_boost('D12_h5', 5, R, 'wrap2')
    raw = bx['val'][KEY + ['rv_pred']].rename(columns={'rv_pred': 'h5raw'})
    v = v[KEY + ['rv_pred']].rename(columns={'rv_pred': 'h5corr'}).merge(raw, on=KEY)
    t = pd.read_parquet(LOCK / 'final3' / 'final_h5_test.parquet', columns=KEY + ['rv_pred', 'xgb_raw']).rename(
        columns={'rv_pred': 'h5corr', 'xgb_raw': 'h5raw'})
    return v, t


def target22():
    import gibrid_har as GH
    A, harj = GH.load_pit(L4.FEAT)
    hv, ht, _ = GH.har_forecasts(22, A, harj)
    return hv[KEY + ['year', 'rv_actual']], ht[KEY + ['year']]


def s2_prev(V, col):
    """s² шага t = дисперсия OOS-остатков наклона val-года шага t−1 (для val шага t); для теста шага t — шага t. -> {шаг: (s2_val, s2_test)}."""
    e = {st: (g.slope_actual - g[col]).dropna().values for st, g in V.groupby('step')}
    out = {}
    for st in range(1, 10):
        sv = float(np.var(e[st - 1])) if st > 1 and len(e.get(st - 1, [])) else 0.0
        out[st] = (sv, float(np.var(e[st])) if len(e.get(st, [])) else sv)
    return out


def build(a):
    hv = pd.read_parquet(SCR / 'har_slope_val.parquet')
    ht = pd.read_parquet(LOCK / 'D25_3' / 'har_slope_test.parquet')
    bv = pd.read_parquet(SCR / 'boost_slope_val.parquet')
    bt = pd.read_parquet(LOCK / 'D25_3' / 'boost_slope_test.parquet')
    h5v, h5t = hyb5()
    tv, tt = target22()
    sa = hv[KEY + ['slope_actual']]
    rows = []
    for seed in SEEDS:
        bs = lambda b: b[b.seed == seed].groupby(KEY, as_index=False).slope_hat.mean()  # среднее ln XGB и LGBM seed s
        V = tv.merge(h5v, on=KEY).merge(hv[KEY + ['slope_hat']].rename(columns={'slope_hat': 's_har'}), on=KEY) \
              .merge(bs(bv).rename(columns={'slope_hat': 's_bst'}), on=KEY).merge(sa, on=KEY)
        T = tt.merge(h5t, on=KEY).merge(ht[KEY + ['slope_hat']].rename(columns={'slope_hat': 's_har'}), on=KEY) \
              .merge(bs(bt).rename(columns={'slope_hat': 's_bst'}), on=KEY)
        for nm, col in (('har', 's_har'), ('bst', 's_bst')):
            s2 = s2_prev(V, col)
            V[f'lit_{nm}'] = V.h5corr * np.exp(V[col] + V.step.map(lambda s: s2[s][0] / 2))
            T[f'lit_{nm}'] = T.h5corr * np.exp(T[col] + T.step.map(lambda s: s2[s][1] / 2))
            V[f'raw_{nm}'] = V.h5raw * np.exp(V[col])
            T[f'raw_{nm}'] = T.h5raw * np.exp(T[col])
        V.to_parquet(SCR / f'yhat22_s{seed}_val.parquet', index=False)
        T.to_parquet(LOCK / 'D25_3' / f'yhat22_s{seed}_test.parquet', index=False)
        # файлы для gibrid_v17: XGB-слот — наклон HAR-J, LGB-слот — наклон бустинга seed s (сырые)
        b = f'D25_3s{seed}'
        for slot, nm in (('xgb', 'raw_har'), ('lgbm', 'raw_bst')):
            V[KEY + ['year', 'rv_actual']].assign(rv_pred=V[nm].values).to_parquet(OUT / f'{b}_{slot}_h22_val.parquet', index=False)
            (LOCK / b).mkdir(parents=True, exist_ok=True)
            T[KEY + ['year']].assign(rv_actual=np.nan, rv_pred=T[nm].values).to_parquet(LOCK / b / f'{b}_{slot}_h22_test.parquet', index=False)
        for nm in ('lit_har', 'lit_bst', 'h5corr'):
            q = L4.year_table(V.rename(columns={nm: 'rv_pred'}) if nm != 'rv_pred' else V)
            rows.append(dict(seed=seed, variant=nm, **L4.summ(q), **{f'y{2015 + s}': q[s] for s in CRIT}))
        # наклон: QLIKE не применим; справка — R² OOS наклона на 2017–2020
        for col in ('s_har', 's_bst'):
            d = V[(V.step >= 2) & (V.step <= 5)].dropna(subset=['slope_actual'])
            rows.append(dict(seed=seed, variant=f'R2_{col}', m1720=1 - np.mean((d.slope_actual - d[col]) ** 2) / np.var(d.slope_actual)))
    S = pd.DataFrame(rows)
    S.to_csv(OUT / 'd25_3_literal.csv', index=False)
    print(S.to_string())


def base(a):
    """Та же конструкция без рычага (прямая цель ln RV̄₂₂): XGB-слот — HAR-J h=22 (сырой), LGB-слот — бустинг D15_h22 seed s (среднее ln XGB и LGBM),
    тот же протокол и точки. Только val (тест не нужен: конструкция-сравнение)."""
    import gibrid_har as GH
    A, harj = GH.load_pit(L4.FEAT)
    hv, _, _ = GH.har_forecasts(22, A, harj)
    for seed in SEEDS:
        fr = []
        for m in ('xgb', 'lgbm'):
            d = pd.read_parquet(R / f'D15_h22_{m}_h22_val.parquet') if seed == 42 else \
                pd.read_parquet(R / f'D15_h22_{m}_h22_seeds4344_val.parquet').query('seed == @seed')
            fr.append(d[KEY + ['rv_pred']].assign(lp=np.log(d.rv_pred.values)))
        bst = pd.concat(fr).groupby(KEY, as_index=False).lp.mean()
        V = hv[KEY + ['year', 'rv_actual', 'rv_pred']].rename(columns={'rv_pred': 'har'}).merge(bst, on=KEY)
        b = f'D25_3base_s{seed}'
        V[KEY + ['year', 'rv_actual']].assign(rv_pred=V.har.values).to_parquet(OUT / f'{b}_xgb_h22_val.parquet', index=False)
        V[KEY + ['year', 'rv_actual']].assign(rv_pred=np.exp(V.lp.values)).to_parquet(OUT / f'{b}_lgbm_h22_val.parquet', index=False)
    a.prefix = 'D25_3base_s'
    gibrid(a)


def gibrid(a):
    """gibrid_v17.py на трёх seed (--legs single, --dev-eval oos, --etalon same): дизайны V0_XGB (наклон HAR-J), V0_LGB (наклон бустинга), схемы (Б)."""
    procs = []
    for seed in SEEDS:
        b = f'{getattr(a, "prefix", "D25_3s")}{seed}'
        out = OUT / 'gibrid' / b
        out.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, '-W', 'ignore', str(HERE / 'gibrid_v17.py'), '--boost', b, '--h', '22', '--features', str(L4.FEAT), '--regen', str(OUT),
               '--legs', 'single', '--out', str(out), '--devlog', str(out / 'devlog.md')]
        procs.append(subprocess.Popen(cmd, stdout=open(out / 'run.log', 'w'), stderr=subprocess.STDOUT))
    for p in procs:
        p.wait()
    print('gibrid rc', [p.returncode for p in procs])


def gib_designs(b):
    p = OUT / 'gibrid' / b / f'designs_val_{b}_h22.csv'
    if not p.exists():
        p = R / 'final3' / 'gibrid' / b / f'designs_val_{b}_h22.csv'
    Dd = pd.read_csv(p, usecols=['design', 'step', 'q_hyb', 'c1_h', 'c2_h', 'c3_h', 'c4_h'])
    Dd = Dd[Dd.step.isin(CRIT)]
    q = Dd.pivot_table(index='design', columns='step', values='q_hyb')
    fl = Dd.groupby('design')[['c1_h', 'c2_h', 'c3_h', 'c4_h']].first()
    S = pd.DataFrame({'m1720': q[list(RANK)].mean(1), 'q2021': q[HOLD], 'm1721': q[list(CRIT)].mean(1)}).join(fl)
    S['n_failed'] = 4 - S[['c1_h', 'c2_h', 'c3_h', 'c4_h']].sum(1)
    return S


def export(a):
    """Кандидат D25_3_h22 в формате final3: val шагов 1–9 (data/regen/D25b), тест шагов 1–9 (test_locked/D25_3_h22, без метрик), ноги single
    (seed 42) и seeds3 (seed 42 + файл seeds4344 с seed 43/44); XGB-слот — наклон HAR-J (детерминирован), LGB-слот — наклон бустинга."""
    b = 'D25_3_h22'
    td = LOCK / b
    td.mkdir(parents=True, exist_ok=True)
    for kind, src, dst in (('val', OUT, OUT), ('test', LOCK, td)):
        for slot in ('xgb', 'lgbm'):
            fr = {s: pd.read_parquet((src / f'D25_3s{s}_{slot}_h22_{kind}.parquet') if kind == 'val' else
                                     (LOCK / f'D25_3s{s}' / f'D25_3s{s}_{slot}_h22_{kind}.parquet')) for s in SEEDS}
            fr[42].to_parquet(dst / f'{b}_{slot}_h22_{kind}.parquet', index=False)
            pd.concat([fr[s].assign(seed=s) for s in (43, 44)]).to_parquet(dst / f'{b}_{slot}_h22_seeds4344_{kind}.parquet', index=False)
    print('export OK', b)


YAML = '''boost: D25_3_h22
h: 22
semeystvo: "наклон срочной структуры: финалист h=5 × прогноз ln(RV̄22/RV̄5) (Д25.3)"
rychag: Д25.3
opisanie: "ŷ22 = hyb5 · exp(ŝ), hyb5 — OOS-прогноз финала h=5 D12_h5 (wrap2/V0_XGB, сырой), ŝ — прогноз наклона ln(RV̄22/RV̄5): XGB-слот — HAR-J на наклон (OLS ln HAR-J, окно шага с эмбарго 22 дня), LGB-слот — бустинг на наклон (точки D15_h22, среднее ln XGB и LGBM, seed 42; seeds3 — 42/43/44); поправка Б0.1 — в гибриде. Рычаг принят как самостоятельная модель (А): наклон HAR-J −1,5% к HAR-J с поправкой sm, наклон бустинга −9,6% к прямому бустингу D15_h22 (на каждом seed), 2021 ≤ 0,1404"
regen: data/regen/D25b
legs: [single, seeds3]
odinochnaya: false
chistyy: false
test_dir: data/regen/test_locked/D25_3_h22
n_variantov: 2   # модели наклона: HAR-J и бустинг (буквальная формула с s² шага t−1 — справка)
god2021: "2021 использован как отложенный: конструкция и s² выбирались по 2017-2020"
kod: scripts/analysis/d25_3.py fit|build|gibrid|base|summary|export|gibcand (детектор — d25_3.py leak)
'''


def yaml_out(a):
    import yaml
    assert yaml.safe_load(YAML)['boost'] == 'D25_3_h22'
    p = L4.KAND / 'D25_3_h22.yaml'
    p.write_text(YAML)
    print('yaml', p)


def gibcand(a):
    """gibrid_v17.py для кандидата D25_3_h22 (ноги single и seeds3, как сделает final3) — в свой каталог data/regen/D25b/gibrid/."""
    b = 'D25_3_h22'
    out = OUT / 'gibrid' / b
    out.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, GIBRID_TEST_DIR=str(LOCK / b))
    cmd = [sys.executable, '-W', 'ignore', str(HERE / 'gibrid_v17.py'), '--boost', b, '--h', '22', '--features', str(L4.FEAT), '--regen', str(OUT),
           '--legs', 'single', 'seeds3', '--out', str(out), '--devlog', str(out / 'devlog.md')]
    rc = subprocess.run(cmd, env=env, stdout=open(out / 'run.log', 'w'), stderr=subprocess.STDOUT).returncode
    print('gibrid', b, 'rc', rc)
    for f in (f'{b}_xgb_h22_test', f'{b}_lgbm_h22_test', f'{b}_lgbm_h22_seeds4344_test'):  # проверка покрытия теста (без метрик)
        d = pd.read_parquet(LOCK / b / f'{f}.parquet', columns=['step', 'rv_pred'])
        print(f, sorted(d.step.unique()), len(d), 'NaN', int(d.rv_pred.isna().sum()))
    S = gib_designs(b)
    S = S[~S.index.str.contains(r'\|V0_(?:HAR|HCJ)\|')].sort_values(['n_failed', 'm1720'], kind='stable')
    S.to_csv(OUT / 'd25_3_cand_rank.csv')
    print(S.head(8).to_string())


def evaluate(a):
    """Таблица Д25.3 по seed: (А) V0_XGB|B-sm — наклон HAR-J против HAR-J с поправкой sm (та же конструкция без рычага — прямая цель);
    (А) V0_LGB|B-sm — наклон бустинга против прямого бустинга D15_h22 того же seed; (Б) лучший по правилу гибрид (не V0) против того же
    дизайна на прямых ногах. Шаги 2–6."""
    rows = []
    for seed in SEEDS:
        Sx, Sb = gib_designs(f'D25_3s{seed}'), gib_designs(f'D25_3base_s{seed}')
        hyb = Sx[~Sx.index.str.contains(r'\|V0_')].sort_values(['n_failed', 'm1720'], kind='stable')
        best = hyb.index[0]
        for var, d, dbase in (('А: наклон HAR-J', 'single|V0_XGB|B-sm|L0', 'single|V0_HAR|B-sm|L0'),
                              ('А: наклон бустинга', 'single|V0_LGB|B-sm|L0', 'single|V0_LGB|B-sm|L0'),
                              ('Б: гибрид ' + best.replace('|', '/'), best, best)):
            r, rb = Sx.loc[d], Sb.loc[dbase]
            rows.append(dict(seed=seed, variant=var.split(' single')[0] if var.startswith('Б') else var, design=d, m1720=r.m1720, q2021=r.q2021,
                             m1721=r.m1721, base_m1720=rb.m1720, base_q2021=rb.q2021, delta=r.m1720 / rb.m1720 - 1,
                             c1=bool(r.c1_h), c2=bool(r.c2_h), c3=bool(r.c3_h), c4=bool(r.c4_h), n_failed=int(r.n_failed)))
    T = pd.DataFrame(rows)
    T['pass2021'] = T.q2021 <= L4.BEST2021[22]
    T.to_csv(OUT / 'd25_3_summary.csv', index=False)
    print(T.to_string())
    g = T.groupby('variant').agg(m1720=('m1720', 'mean'), delta=('delta', 'mean'), all_better=('delta', lambda x: bool((x < 0).all())),
                                 q2021=('q2021', 'mean'), pass2021=('pass2021', 'all'))
    g['accepted'] = (g.delta <= -0.01) & g.all_better & g.pass2021
    print(g.to_string())


def leak(a):
    """Детектор «будущее не влияет» для HAR-J наклона и s²: данные с даты среза (признаки, RV, цели строк, чей горизонт заходит за срез) портятся;
    прогноз наклона и ŷ₂₂ (литеральный) до среза не должны меняться. Подложенная утечка (без эмбарго и с обучением на val-годе) обязана сработать."""
    import gibrid_har as GH
    A, harj = GH.load_pit(L4.FEAT)
    cut = pd.Timestamp('2019-07-01')
    rng = np.random.default_rng(2)

    def spoil(A):
        B = A.copy()
        d = pd.to_datetime(B.date)
        f = d >= cut
        for c in harj + ['rv_target_mean_h5', 'rv_target_mean_h22']:
            B.loc[f, c] = B.loc[f, c] * rng.uniform(0.2, 5, f.sum())
        g = (d >= cut - pd.Timedelta(days=40)) & ~f  # цели строк перед срезом, чей горизонт заходит за срез
        for c in ('rv_target_mean_h5', 'rv_target_mean_h22'):
            B.loc[g, c] = B.loc[g, c] * rng.uniform(0.2, 5, g.sum())
        return B
    B = spoil(A)
    va, _ = har_slope(A.copy(), harj)
    vb, _ = har_slope(B.copy(), harj)
    m = va.merge(vb, on=KEY, suffixes=('', '_b'))
    pre = pd.to_datetime(m.date) < cut
    d = float(np.max(np.abs(m.slope_hat[pre] - m.slope_hat_b[pre])))
    print(f'HAR-J наклона: max |Δ ŝ| до среза = {d:.2e}', 'OK' if d < 1e-10 else 'УТЕЧКА')
    s2a, s2b = s2_prev(va, 'slope_hat'), s2_prev(vb, 'slope_hat')
    ds = max(abs(s2a[s][0] - s2b[s][0]) for s in range(1, 10) if 2015 + s <= 2019)
    print(f's² (шаги с val-годом ≤ 2019): max |Δ| = {ds:.2e}', 'OK' if ds < 1e-12 else 'УТЕЧКА')
    # подложенная утечка: модель шага обучается и на самом val-году (цели которого заходят за срез), без эмбарго
    la, _ = har_slope(A.copy(), harj, plant=True)
    lb, _ = har_slope(B.copy(), harj, plant=True)
    m2 = la.merge(lb, on=KEY, suffixes=('', '_b'))
    pre = pd.to_datetime(m2.date) < cut
    d2 = float(np.max(np.abs(m2.slope_hat[pre] - m2.slope_hat_b[pre])))
    print(f'подложенная утечка: max |Δ| = {d2:.2e}', 'сработал (OK)' if d2 > 1e-10 else 'НЕ СРАБОТАЛ')
    print('ДЕТЕКТОР Д25.3 (HAR-J наклона, s²):', 'OK' if d < 1e-10 and ds < 1e-12 and d2 > 1e-10 else 'ОШИБКА')
    # бустинг наклона: шаг 4 (val 2019, тест 2020), seed 42; данные с 2020-02-15 (за горизонтом строк val-года) портятся -> val ŝ не меняется;
    # подложенная утечка — строки тестового года в обучении
    import l2_lib as L
    _init()
    A0 = G['A']
    cutb = pd.Timestamp('2020-02-15')
    Bb = A0.copy()
    f = pd.to_datetime(Bb.date) >= cutb
    num = [c for c in G['cols'] + ['rv_target_mean_h5', 'rv_target_mean_h22'] if pd.api.types.is_numeric_dtype(Bb[c])]
    Bb.loc[f, num] = Bb.loc[f, num].values * rng.uniform(0.2, 5, (f.sum(), 1))
    okb = True
    for model in ('lgbm', 'xgb'):
        out = []
        for AA in (A0, Bb):
            G['A'] = AA
            out.append(_task((model, 4, 42))[0].slope_hat.values)
        G['A'] = A0
        db = float(np.max(np.abs(out[0] - out[1])))
        print(f'бустинг наклона {model}: max |Δ ŝ val 2019| = {db:.2e}', 'OK' if db < 1e-10 else 'УТЕЧКА')
        okb &= db < 1e-10
    orig_split = L.split
    def split_leak(A_, h, step):
        tp, vi, td, vy, ty = orig_split(A_, h, step)
        return pd.concat([tp, td.dropna(subset=[L.tcol(h), 'rv_target_mean_h5'])]), vi, td, vy, ty
    L.split = split_leak
    out = []
    for AA in (A0, Bb):
        G['A'] = AA
        out.append(_task(('lgbm', 4, 42))[0].slope_hat.values)
    L.split = orig_split
    G['A'] = A0
    db2 = float(np.max(np.abs(out[0] - out[1])))
    print(f'бустинг, подложенная утечка: max |Δ| = {db2:.2e}', 'сработал (OK)' if db2 > 1e-10 else 'НЕ СРАБОТАЛ')
    print('ДЕТЕКТОР Д25.3 (бустинг наклона):', 'OK' if okb and db2 > 1e-10 else 'ОШИБКА')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['fit', 'build', 'gibrid', 'base', 'summary', 'leak', 'export', 'gibcand', 'yaml', 'hartest'])
    ap.add_argument('--workers', type=int, default=3)
    a = ap.parse_args()
    if a.cmd == 'hartest':
        hv, _ = har_slope(); print(hv.groupby('step').apply(lambda g: np.corrcoef(g.slope_actual.fillna(0), g.slope_hat)[0, 1]))
    else:
        {'fit': fit, 'build': build, 'gibrid': gibrid, 'base': base, 'summary': evaluate, 'leak': leak, 'export': export, 'gibcand': gibcand, 'yaml': yaml_out}[a.cmd](a)
