"""Д24.4: режимный HAR-J / HAR-X. Индикатор I_t = 1{RVI_t (срез 18:50) > медианы RVI за 252 ПРОШЛЫХ торговых дня (t-252..t-1)};
до накопления 126 дней истории I = 0. Модель шага: ln y = b0 + b'x + I*(g0 + g'x) - взаимодействие всех слагаемых с индикатором,
OLS на окне шага (годы < vy, эмбарго как Д18). x - регрессоры HAR-J (rv_d, rv_w, rv_m, jump*) [RJ] или HAR-J + слагаемые HAR-X
(harx.default_terms(h): RVI, RV IMOEX, RV рынка, LHAR, PC1, HARQ, RSV±, RV66/132, ln σ̂₁ для h=5/22; пропуски - средним обучения) [RX].
Контроль без режима: та же OLS без I (J0 = HAR-J с эмбарго, X0 = HAR-X OLS).
Применения: (1) самостоятельная нога (эконометрика, таблица; не кандидат пула); (2) нога гибрида: XGB-слот = режимный HAR, LGB-слот =
бустинг-нога финалиста (d24b_lib.leg); (3) основа Д19.1 (h=5/22): ln y_h = ln HARreg_h + rho*r1, r1 = ln hyb1 - ln HAR-J_1 (как Д19),
rho - OLS без константы по OOS-истории шагов < s; (4) основа Д8.12 (h=1) - d24_4.py --d812 (бустинг с ln HARreg_1 как признаком).
python3 scripts/analysis/d24_4.py [--build] [--gibrid] [--seeds] [--eval] [--leak] [--d812 ...] [--yaml]"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import d24b_lib as L
import harx as HX

OUT = L.OUT / 'd24_4'
SEEDS = ('s42', 's43', 's44')


def regime(A):
    """I_t по датам (рыночный ряд): медиана RVI за 252 прошлых реальных торговых дня, строка-заполнитель протягивается."""
    x, v = HX._by_date(A)
    r = v['close_RVI'].astype(float)
    med = r.shift(1).rolling(252, min_periods=126).median()
    ind = ((r > med) & med.notna()).astype(float)
    s = ind.reindex(x.index, method='ffill').fillna(0.0)
    return pd.Series(s.reindex(A['date']).values, index=A.index), pd.DataFrame({'rvi': r, 'med252_past': med, 'I': ind})


def design(A, harj, h, kind):
    """Матрица регрессоров (без константы) и флаг: kind in RJ, RX, J0, X0."""
    X = L.harj_X(A, harj)
    names = list(harj)
    Z = None
    if kind in ('RX', 'X0'):
        HX.GH_HARJ = harj
        terms = HX.default_terms(h)
        Z = HX.build_terms(A, terms)
        names += terms
    I = regime(A)[0].values if kind in ('RJ', 'RX') else None
    return X, Z, I, names


def fit_pred_factory(A, h, X, Z, I):
    def fit_pred(s, vy, tr):
        if Z is not None:
            mu = Z.loc[tr].mean()
            Zf = Z.fillna(mu).fillna(0.0).values
            M = np.column_stack([X, Zf])
        else:
            M = X
        if I is not None:
            M = np.column_stack([M, I, I[:, None] * M])
        y = np.log(L.target(A, h).values[tr] + 1e-10)
        b = L.ols(M[tr], y)
        return (lambda sel: L.ols_pred(b, M[sel])), dict(coef=json.dumps([round(float(c), 5) for c in b]))
    return fit_pred


def model(A, harj, h, kind):
    X, Z, I, names = design(A, harj, h, kind)
    return L.run_steps(A, h, fit_pred_factory(A, h, X, Z, I))


# ---------------------------------------------------------------- Д19.1 на режимной основе
def r1_tables(A, harj):
    """r1 = ln hyb1 - ln HAR-J_1 (OOS; как d19_signal): val - по seed-вариантам hyb1 (main = top5, s42/s43/s44), тест - final2 (xgb_raw, har_raw)."""
    hv1, _ = L.har_ref(A, harj, 1)
    hv1 = hv1[L.K + ['rv_pred']].rename(columns={'rv_pred': 'har1'})
    out = {}
    for sd in ('main',) + SEEDS:
        v, _ = L.leg(1, sd, test=False)
        m = v.merge(hv1, on=L.K)
        out[sd] = m.assign(r1=np.log(m.rv_pred) - np.log(m.har1))[L.K + ['r1']]
    t = pd.read_parquet(L.TL / 'final2' / 'final_h1_test.parquet', columns=L.K + ['xgb_raw', 'har_raw'])
    t = t[t.step <= 9]
    out['test'] = t.assign(r1=np.log(t.xgb_raw) - np.log(t.har_raw))[L.K + ['r1']]
    return out


def d19(V, T, r1, sd):
    """ln y = ln base + rho*r1; rho - OLS без константы остатка ln y - ln base по val шагов < s (шаг 1 - свой год); тест шага s - шаги <= s."""
    v = V.merge(r1[sd], on=L.K, how='left')
    t = T.merge(r1['test'], on=L.K, how='left')
    ok = (v.rv_actual > 1e-12) & v.r1.notna()
    v['res'] = np.log(v.rv_actual.clip(1e-12)) - np.log(v.rv_pred)
    o = lambda d: float((d.r1 * d.res).sum() / (d.r1 ** 2).sum())
    pv, pt, rho = [], [], []
    for s in range(1, 10):
        rv_ = o(v[ok & ((v.step < s) | ((s == 1) & (v.step == 1)))])
        rt_ = o(v[ok & (v.step <= s)])
        a, b = v[v.step == s], t[t.step == s]
        pv.append(a[L.COLS[:-1]].assign(rv_pred=np.exp(np.log(a.rv_pred) + rv_ * a.r1.fillna(0))))
        pt.append(b[L.COLS[:-1]].assign(rv_pred=np.exp(np.log(b.rv_pred) + rt_ * b.r1.fillna(0))))
        rho.append(dict(step=s, rho_val=rv_, rho_test=rt_))
    return pd.concat(pv, ignore_index=True), pd.concat(pt, ignore_index=True), pd.DataFrame(rho)


# ---------------------------------------------------------------- сборка
def build():
    OUT.mkdir(parents=True, exist_ok=True)
    A, harj = L.load()
    ind, tab = regime(A)
    tab.to_csv(OUT / 'regime_rvi.csv')
    print('доля дней I=1 по годам:', pd.Series(ind.values, index=A.year).groupby(level=0).mean().round(2).to_dict(), flush=True)
    rows, M = [], {}
    for h in (1, 5, 22):
        hv, ht = L.har_ref(A, harj, h)
        rows += L.standalone_rows('HAR-J (эталон, канон)', h, hv, ht)
        for kind in ('J0', 'RJ', 'X0', 'RX'):
            t0 = time.time()
            V, T, info = model(A, harj, h, kind)
            M[(kind, h)] = (V, T)
            V.to_parquet(OUT / f'{kind}_h{h}_val.parquet', index=False)
            T.to_parquet(OUT / f'{kind}_h{h}_test.parquet', index=False)  # тест: только сборка (каталог D24b - не test_locked: ноги для Д22, без метрик)
            info.to_csv(OUT / f'{kind}_h{h}_coef.csv', index=False)
            rows += L.standalone_rows(kind, h, V, T)
            print(kind, h, f'{time.time() - t0:.0f} c', flush=True)
    r1 = r1_tables(A, harj)
    rho_all = []
    for h in (5, 22):
        for kind in ('RJ', 'RX', 'J0'):  # J0 - контроль: та же Д19.1 на HAR-J без режима (эмбарго), чтобы отделить вклад режима
            V, T = M[(kind, h)]
            for sd in ('main',) + SEEDS:
                pv, pt, rho = d19(V, T, r1, sd)
                M[(f'S19{kind}', h, sd)] = (pv, pt)
                rho_all.append(rho.assign(h=h, base=kind, seed=sd))
                if sd == 'main':
                    rows += L.standalone_rows(f'Д19.1 на {kind}', h, pv, pt)
    pd.concat(rho_all).to_csv(OUT / 'rho_d19.csv', index=False)
    Q = pd.DataFrame(rows)
    Q.to_csv(OUT / 'standalone.csv', index=False)
    with pd.option_context('display.width', 200):
        print(Q.round(4).to_string(), flush=True)
    # кандидаты гибрида: XGB-слот = новая нога, LGB-слот = бустинг-нога финалиста; seed-варианты - только val
    for h in (1, 5, 22):
        lv, lt = L.leg(h, 'main')
        for kind in ('RJ', 'RX'):
            V, T = M[(kind, h)]
            b = f'D24_4{kind.lower()}_h{h}'
            L.save_cand(b, h, V, L.on_frame(V, lv), T, L.on_frame(T, lt))
            for sd in SEEDS:
                L.save_cand(f'{b}_{sd}', h, V, L.on_frame(V, L.leg(h, sd, test=False)[0]))
    for h in (5, 22):
        lv, lt = L.leg(h, 'main')
        for kind in ('RJ', 'RX', 'J0'):
            b = f'D24_4s19{kind.lower()}_h{h}'
            pv, pt = M[(f'S19{kind}', h, 'main')]
            L.save_cand(b, h, pv, L.on_frame(pv, lv), pt, L.on_frame(pt, lt))
            for sd in SEEDS:
                pv_s = M[(f'S19{kind}', h, sd)][0]
                L.save_cand(f'{b}_{sd}', h, pv_s, L.on_frame(pv_s, L.leg(h, sd, test=False)[0]))
    print('кандидаты записаны', flush=True)


MAIN = [f'D24_4{k}_h{h}' for k in ('rj', 'rx') for h in (1, 5, 22)] + [f'D24_4s19{k}_h{h}' for k in ('rj', 'rx', 'j0') for h in (5, 22)]
hof = lambda b: int(b.split('_h')[1].split('_')[0])


def leak():
    """(а) индикатор на датах < cut не меняется при порче RVI с даты cut (медиана - только прошлое); контроль: медиана с текущим днём
    и будущим (центрированное окно) - меняется; (б) коэффициенты режимных моделей шага не меняются при порче всех данных с 1 янв. val-года."""
    A, harj = L.load()
    res = []
    i0 = regime(A)[0]
    for cut in ('2017-01-01', '2019-06-30', '2021-01-01'):
        B = L.perturb(A, cut, cols=['close_RVI'])
        i1 = regime(B)[0]
        pre = (A.date < pd.Timestamp(cut)).values
        res.append(dict(test='индикатор', cut=cut, max_diff_do_cut=float(np.abs(i0.values[pre] - i1.values[pre]).max()),
                        diff_posle=float(np.abs(i0.values[~pre] - i1.values[~pre]).mean())))
    for h in (1, 22):
        for kind in ('RJ', 'RX'):
            X, Z, I, _ = design(A, harj, h, kind)
            for s, ty, vy in L.STEPS[:6]:
                c0 = fit_pred_factory(A, h, X, Z, I)(s, vy, L.train_mask(A, h, vy))[1]['coef']
                B = L.perturb(A, f'{vy}-01-01', seed=s)
                Xb, Zb, Ib, _ = design(B, harj, h, kind)
                c1 = fit_pred_factory(B, h, Xb, Zb, Ib)(s, vy, L.train_mask(B, h, vy))[1]['coef']
                d = float(np.abs(np.array(json.loads(c0)) - np.array(json.loads(c1))).max())
                res.append(dict(test=f'коэф. {kind} h={h}', cut=f'{vy}-01-01', max_diff_do_cut=d, diff_posle=np.nan))
    D = pd.DataFrame(res)
    D['ok'] = D.max_diff_do_cut < 1e-9
    D.to_csv(OUT / 'leak.csv', index=False)
    print(D.to_string(), flush=True)


def ln_regime_all(h, kind='RJ'):
    """(date, ticker, hx1..hx9): ln-прогноз режимной модели шага k на всех строках (обучение - внутри выборки, как HAR-J в Д8.12)."""
    A, harj = L.load()
    X, Z, I, _ = design(A, harj, h, kind)
    fp = fit_pred_factory(A, h, X, Z, I)
    out = A[['date', 'ticker']].copy()
    allr = np.ones(len(A), bool)
    for s, ty, vy in L.STEPS:
        out[f'hx{s}'] = fp(s, vy, L.train_mask(A, h, vy))[0](allr)
    out['hx10'] = out['hx9']  # шаг 10 не строится (↺); столбец - только для проверки prepare
    return out


def d812(kind='RJ', workers=2):
    """Д8.12 (h=1) на режимной основе: точки и обёртка D16_h1_d812x / D12_h1_A_d812 (XGB top5 ЛГК B v2b, признак ln HAR), ln HAR-J
    заменён на ln режимной модели шага (d16_v9: подмена l2_lib.har_ln). XGB: ранги 1-5 шагов 1-6 + точка шагов 7-9, seed 43/44 шагов 1-9;
    LGBM: то же без seed. Выход - кандидат D24_4d812{kind}_h1 (ноги single/top5/seeds3), тест - test_locked/<кандидат>/ (без метрик)."""
    import multiprocessing as mp
    import d16_v9 as V
    cache = ln_regime_all(1, kind)
    V.HX.ln_all_steps = lambda h, terms=None, **k: cache
    V.G['control'] = False
    t0 = time.time()
    res = []
    for models, seeds in ((('xgb',), None), (('xgb',), [43, 44]), (('lgbm',), None)):
        V.CANDS['h1_d812x'] = dict(V.CANDS['h1_d812x'], models=models)
        V.prepare(['h1_d812x'], 'HARX')
        jobs = [('h1_d812x', *pt) for pt in V.points(V.CANDS['h1_d812x'], list(range(1, 10)), seeds)]
        print(f'{models} seeds={seeds}: {len(jobs)} задач', flush=True)
        with mp.get_context('fork').Pool(workers) as pool:
            for r in pool.imap_unordered(V.job, jobs):
                res.append(r)
        print(f'  готово {time.time() - t0:.0f} c', flush=True)
    b = f'D24_4d812{kind.lower()}_h1'
    d = OUT / 'd812'; d.mkdir(parents=True, exist_ok=True)
    Vv = pd.concat([r[6] for r in res], ignore_index=True); Tt = pd.concat([r[7] for r in res], ignore_index=True)
    Vv.to_parquet(d / f'{b}_all_val.parquet', index=False); Tt.to_parquet(d / f'{b}_all_test.parquet', index=False)
    pd.DataFrame([dict(model=r[1], step=r[2], seed=r[3], rank=r[4], trees=r[5]) for r in res]).to_csv(d / f'{b}_trees.csv', index=False)
    td = L.TL / b; td.mkdir(parents=True, exist_ok=True); L.CAND.mkdir(parents=True, exist_ok=True)
    cols = L.COLS
    for m in ('xgb', 'lgbm'):
        for kind_, X, dst, sfx in (('val', Vv, L.CAND, 'val'), ('test', Tt, td, 'test')):
            x = X[X.model == m]
            single = x[(x['rank'] == 1) & (x.seed == 42)]
            single[cols].to_parquet(dst / f'{b}_{m}_h1_{sfx}.parquet', index=False)
            x[(x.step <= 6) & (x.seed == 42)][cols + ['rank']].to_parquet(dst / f'{b}_{m}_h1_top5_{sfx}.parquet', index=False)
            s43 = x[x.seed.isin([43, 44])] if m == 'xgb' else single.assign(seed=43)  # у LGBM seed 43/44 нет: слот LGB seed-вариантов = seed 42 (↺)
            if m == 'lgbm':
                s43 = pd.concat([single.assign(seed=43), single.assign(seed=44)])
            s43[cols + ['seed']].to_parquet(dst / f'{b}_{m}_h1_seeds4344_{sfx}.parquet', index=False)
            if kind_ == 'val':
                for n in (42, 43, 44):
                    xs = single if n == 42 else s43[s43.seed == n]
                    xs[cols].to_parquet(L.CAND / f'{b}_s{n}_{m}_h1_val.parquet', index=False)
    print('кандидат', b, 'записан', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--d812', nargs='*')
    ap.add_argument('--build', action='store_true'); ap.add_argument('--gibrid', nargs='*'); ap.add_argument('--seeds', nargs='*')
    ap.add_argument('--eval', action='store_true'); ap.add_argument('--leak', action='store_true'); ap.add_argument('--workers', type=int, default=2)
    a = ap.parse_args()
    if a.d812 is not None:
        for kind in (a.d812 or ['RJ']):
            d812(kind, a.workers)
    if a.build:
        build()
    if a.gibrid is not None:
        L.run_gibrid([(b, hof(b)) for b in (a.gibrid or MAIN)], a.workers)
    if a.seeds is not None:
        L.run_gibrid([(f'{b}_{s}', hof(b)) for b in a.seeds for s in SEEDS], a.workers)
    if a.eval:
        rows = [r for r in (L.evaluate(b, hof(b)) for b in MAIN) if r]
        E = pd.DataFrame(rows)
        E.to_csv(OUT / 'eval.csv', index=False)
        with pd.option_context('display.width', 250, 'display.max_columns', 40):
            print(E[['boost', 'design', 'n_failed', 'failed', 'mean_1720', 'har_1720', 'gain_vs_har', 'q2021', 'v0lgb_1720', 'v0lgb_2021',
                     'seed_mean_1720', 'p_novaya_noga', 'p_1pct', 'p_seeds', 'p_2021', 'prohodit']].round(4).to_string())
    if a.leak:
        leak()


if __name__ == '__main__':
    main()
