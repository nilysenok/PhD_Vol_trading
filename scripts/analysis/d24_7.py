"""Д24.7: MIDAS-RV с бета-весами (Ghysels, Santa-Clara, Valkanov 2006), h = 22. Пул 17 тикеров, NLS на окне шага (годы < vy, эмбарго).
  L (основной, в логах):  ln y = mu + phi * sum_{k=0}^{K-1} b_k(t1, t2) * ln RV_{t-1-k}
  V (уровень под логом):  ln y = mu + phi * ln( sum_k b_k(t1, t2) * RV_{t-1-k} )
b_k = f(u_k)/sum f, f(u) = u^(t1-1) (1-u)^(t2-1), u_k = (k+1)/(K+1), K = 50 лагов дневной RV (rv_d - та же дневная RV, что в HAR-J:
rv_d строки t = RV дня t-1; лаги rv_d по тикеру). Параметры: t1, t2 > 0 (бета-функция), mu, phi (константа и наклон).
NLS - переменная проекция: при заданных (t1, t2) mu и phi - МНК; (ln t1, ln t2) - Nelder-Mead из нескольких стартов.
Нога гибрида: XGB-слот = MIDAS, LGB-слот = Д16.5 (бустинг-нога h=22, d24b_lib.leg).
python3 scripts/analysis/d24_7.py [--build] [--gibrid] [--seeds ...] [--eval] [--leak]"""
import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parent))
import d24b_lib as L

OUT = L.OUT / 'd24_7'
SEEDS = ('s42', 's43', 's44')
KLAG, H = 50, 22
STARTS = ((1.0, 5.0), (1.0, 20.0), (1.5, 3.0), (0.8, 1.5))


def lags(A):
    g = A.groupby('ticker', sort=False)['rv_d']
    M = np.column_stack([g.shift(k).values for k in range(KLAG)])
    return M


def weights(t1, t2):
    u = np.arange(1, KLAG + 1) / (KLAG + 1)
    lf = (t1 - 1) * np.log(u) + (t2 - 1) * np.log(1 - u)
    w = np.exp(lf - lf.max())
    return w / w.sum()


def regressor(M, t1, t2, kind):
    w = weights(t1, t2)
    if kind == 'L':
        return np.log(M.clip(1e-10)) @ w
    return np.log((M @ w).clip(1e-10))


def fit(M, y, tr, kind):
    Mt, yt = M[tr], y[tr]
    LMt = np.log(Mt.clip(1e-10)) if kind == 'L' else None

    def sse(eta):
        t1, t2 = np.exp(np.clip(eta, -4, 6))
        w = weights(t1, t2)
        x = LMt @ w if kind == 'L' else np.log((Mt @ w).clip(1e-10))
        X = np.column_stack([np.ones(len(x)), x])
        b, *_ = np.linalg.lstsq(X, yt, rcond=None)
        return float(np.sum((yt - X @ b) ** 2))
    best = None
    for st in STARTS:
        r = minimize(sse, np.log(st), method='Nelder-Mead', options=dict(xatol=1e-4, fatol=1e-6, maxiter=400))
        if best is None or r.fun < best.fun:
            best = r
    t1, t2 = np.exp(np.clip(best.x, -4, 6))
    x = regressor(Mt, t1, t2, kind)
    X = np.column_stack([np.ones(len(x)), x])
    b, *_ = np.linalg.lstsq(X, yt, rcond=None)
    return dict(t1=float(t1), t2=float(t2), mu=float(b[0]), phi=float(b[1]), sse=best.fun, n=int(tr.sum()))


def fit_pred_factory(A, kind, M=None):
    M = lags(A) if M is None else M
    full = np.isfinite(M).all(1)
    yl = np.log(L.target(A, H).values + 1e-10)

    def fit_pred(s, vy, tr):
        co = fit(M, yl, tr & full, kind)
        w = weights(co['t1'], co['t2'])

        def pred(sel):
            Ms = np.where(np.isfinite(M[sel]), M[sel], np.nanmean(M[sel], axis=1, keepdims=True))  # в val/тест-годах история полная; страховка
            return co['mu'] + co['phi'] * regressor(Ms, co['t1'], co['t2'], kind)
        co['w_first5'] = json.dumps([round(float(x), 4) for x in w[:5]])
        co['w_mass_0_4_5_21_22_49'] = json.dumps([round(float(w[:5].sum()), 3), round(float(w[5:22].sum()), 3), round(float(w[22:].sum()), 3)])
        return pred, co
    return fit_pred


def build():
    OUT.mkdir(parents=True, exist_ok=True)
    A, harj = L.load()
    M = lags(A)
    hv, ht = L.har_ref(A, harj, H)
    rows = L.standalone_rows('HAR-J (эталон, канон)', H, hv, ht)
    lv, lt = L.leg(H, 'main')
    for kind in ('L', 'V'):
        V, T, info = L.run_steps(A, H, fit_pred_factory(A, kind, M))
        V.to_parquet(OUT / f'MIDAS{kind}_h22_val.parquet', index=False); T.to_parquet(OUT / f'MIDAS{kind}_h22_test.parquet', index=False)
        info.to_csv(OUT / f'MIDAS{kind}_h22_coef.csv', index=False)
        print(kind, info[['step', 't1', 't2', 'mu', 'phi', 'w_mass_0_4_5_21_22_49']].round(3).to_string(), flush=True)
        rows += L.standalone_rows(f'Д24.7 MIDAS-{kind}', H, V, T)
        b = f'D24_7m{kind.lower()}_h22'
        L.save_cand(b, H, V, L.on_frame(V, lv), T, L.on_frame(T, lt))
        for sd in SEEDS:
            L.save_cand(f'{b}_{sd}', H, V, L.on_frame(V, L.leg(H, sd, test=False)[0]))
    Q = pd.DataFrame(rows); Q.to_csv(OUT / 'standalone.csv', index=False)
    print(Q.round(4).to_string(), flush=True)


MAIN = ['D24_7ml_h22', 'D24_7mv_h22']


def leak():
    A, harj = L.load()
    res = []
    for kind in ('L',):
        for s, ty, vy in L.STEPS[:6]:
            c0 = fit_pred_factory(A, kind)(s, vy, L.train_mask(A, H, vy))[1]
            B = L.perturb(A, f'{vy}-01-01', seed=s)
            c1 = fit_pred_factory(B, kind)(s, vy, L.train_mask(B, H, vy))[1]
            d = max(abs(c0[k] - c1[k]) for k in ('t1', 't2', 'mu', 'phi'))
            res.append(dict(kind=kind, step=s, max_dparam=d, ok=d < 1e-9))
    D = pd.DataFrame(res); D.to_csv(OUT / 'leak.csv', index=False); print(D.to_string(), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--build', action='store_true'); ap.add_argument('--gibrid', nargs='*'); ap.add_argument('--seeds', nargs='*')
    ap.add_argument('--eval', action='store_true'); ap.add_argument('--leak', action='store_true'); ap.add_argument('--workers', type=int, default=2)
    a = ap.parse_args()
    if a.build:
        build()
    if a.gibrid is not None:
        L.run_gibrid([(b, 22) for b in (a.gibrid or MAIN)], a.workers)
    if a.seeds is not None:
        L.run_gibrid([(f'{b}_{s}', 22) for b in a.seeds for s in SEEDS], a.workers)
    if a.eval:
        E = pd.DataFrame([r for r in (L.evaluate(b, 22) for b in MAIN) if r]); E.to_csv(OUT / 'eval.csv', index=False)
        with pd.option_context('display.width', 250, 'display.max_columns', 40):
            print(E[['boost', 'design', 'n_failed', 'failed', 'mean_1720', 'har_1720', 'gain_vs_har', 'q2021', 'v0lgb_1720', 'v0lgb_2021',
                     'seed_mean_1720', 'p_novaya_noga', 'p_1pct', 'p_seeds', 'p_2021', 'prohodit']].round(4).to_string())
    if a.leak:
        leak()


if __name__ == '__main__':
    main()
