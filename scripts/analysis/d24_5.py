"""Д24.5: частичное объединение HAR-J. Коэффициенты по тикеру i - гребень к панельным: b_i = argmin ||y_i - Z_i b||² + c*n_i*||b - b_pool||²,
Z = [1, стандартизованные ln-регрессоры HAR-J] (среднее и с.о. - по обучению шага, пул), b_pool - OLS пула на том же окне (эмбарго).
c in {0 (свой OLS тикера), 0,003 .. 10, inf (пул)}; c выбирается по валидации шага t-1: обучение на годах < vy-1 (эмбарго к vy-1),
MSE логов на годе vy-1 (= val-год шага t-1), затем переобучение на годах < vy с выбранным c. Val-год шага этих данных не видел.
Нога гибрида: XGB-слот = Д24.5, LGB-слот = бустинг-нога финалиста (d24b_lib.leg).
python3 scripts/analysis/d24_5.py [--build] [--gibrid] [--seeds ...] [--eval] [--leak]"""
import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import d24b_lib as L

OUT = L.OUT / 'd24_5'
SEEDS = ('s42', 's43', 's44')
CS = (0.0, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, np.inf)


def fit_pp(X, y, tk, tr, c):
    """-> (b_pool, {тикер: b_i}, mu, sd) по строкам tr."""
    mu, sd = X[tr].mean(0), X[tr].std(0)
    sd = np.where(sd > 1e-12, sd, 1.0)
    Z = np.column_stack([np.ones(len(X)), (X - mu) / sd])
    bp, *_ = np.linalg.lstsq(Z[tr], y[tr], rcond=None)
    B = {}
    for t in np.unique(tk[tr]):
        m = tr & (tk == t)
        if not np.isfinite(c):
            B[t] = bp; continue
        Zi, yi = Z[m], y[m]
        lam = c * m.sum()
        P = Zi.T @ Zi + lam * np.eye(Z.shape[1])
        B[t] = np.linalg.solve(P, Zi.T @ yi + lam * bp) if lam > 0 else np.linalg.lstsq(Zi, yi, rcond=None)[0]
    return bp, B, Z


def predict(Z, tk, sel, bp, B):
    out = np.empty(sel.sum())
    zs, ts = Z[sel], tk[sel]
    for t in np.unique(ts):
        m = ts == t
        out[m] = zs[m] @ B.get(t, bp)
    return out


def fit_pred_factory(A, harj, h, log=None):
    X = L.harj_X(A, harj)
    tk = A.ticker.values
    yl = np.log(L.target(A, h).values + 1e-10)

    def fit_pred(s, vy, tr):
        # выбор c: обучение < vy-1 (эмбарго к vy-1), MSE логов на годе vy-1
        tri = L.train_mask(A, h, vy - 1)
        va = ((A.year == vy - 1) & L.target(A, h).notna() & (L.target_end(A, h) < pd.Timestamp(f'{vy}-01-01'))).values  # эмбарго и для валидации
        mse = {}
        for c in CS:
            bp, B, Z = fit_pp(X, yl, tk, tri, c)
            mse[c] = float(np.mean((yl[va] - predict(Z, tk, va, bp, B)) ** 2))
        cbest = min(mse, key=mse.get)
        bp, B, Z = fit_pp(X, yl, tk, tr, cbest)
        disp = float(np.mean([np.abs(b - bp)[1:].mean() for b in B.values()]))
        co = dict(c=cbest, mse_inner=json.dumps({str(k): round(v, 5) for k, v in mse.items()}), mean_abs_dev_from_pool=disp,
                  b_pool=json.dumps([round(float(x), 4) for x in bp]))
        return (lambda sel: predict(Z, tk, sel, bp, B)), co
    return fit_pred


def build():
    OUT.mkdir(parents=True, exist_ok=True)
    A, harj = L.load()
    rows = []
    for h in (1, 5, 22):
        hv, ht = L.har_ref(A, harj, h)
        rows += L.standalone_rows('HAR-J (эталон, канон)', h, hv, ht)
        V, T, info = L.run_steps(A, h, fit_pred_factory(A, harj, h))
        V.to_parquet(OUT / f'PP_h{h}_val.parquet', index=False); T.to_parquet(OUT / f'PP_h{h}_test.parquet', index=False)
        info.to_csv(OUT / f'PP_h{h}_coef.csv', index=False)
        print(h, info[['step', 'c', 'mean_abs_dev_from_pool']].to_string(), flush=True)
        rows += L.standalone_rows('Д24.5 частичное объединение', h, V, T)
        lv, lt = L.leg(h, 'main')
        b = f'D24_5pp_h{h}'
        L.save_cand(b, h, V, L.on_frame(V, lv), T, L.on_frame(T, lt))
        for sd in SEEDS:
            L.save_cand(f'{b}_{sd}', h, V, L.on_frame(V, L.leg(h, sd, test=False)[0]))
    Q = pd.DataFrame(rows); Q.to_csv(OUT / 'standalone.csv', index=False)
    print(Q.round(4).to_string(), flush=True)


def control():
    """Контроль рычагов Д24.4/24.5/24.7/24.9 (ночное умолчание: рычаг оценивается против той же конструкции без рычага):
    XGB-слот = HAR-J с эмбарго (J0, d24_4), LGB-слот = та же бустинг-нога финалиста. Кандидат D24_ctl_h{h} - только для сравнения, в пул не подаётся."""
    for h in (1, 5, 22):
        V = pd.read_parquet(L.OUT / 'd24_4' / f'J0_h{h}_val.parquet')
        lv, _ = L.leg(h, 'main', test=False)
        L.save_cand(f'D24_ctl_h{h}', h, V, L.on_frame(V, lv))


MAIN = [f'D24_5pp_h{h}' for h in (1, 5, 22)]
hof = lambda b: int(b.split('_h')[1].split('_')[0])


def leak():
    """Коэффициенты (и выбор c) шага не меняются при порче всех данных с 1 янв. val-года (эмбарго; выбор c - по годам < vy)."""
    A, harj = L.load()
    res = []
    for h in (1, 22):
        for s, ty, vy in L.STEPS[:6]:
            c0 = fit_pred_factory(A, harj, h)(s, vy, L.train_mask(A, h, vy))[1]
            B = L.perturb(A, f'{vy}-01-01', seed=s)
            c1 = fit_pred_factory(B, harj, h)(s, vy, L.train_mask(B, h, vy))[1]
            d = float(np.abs(np.array(json.loads(c0['b_pool'])) - np.array(json.loads(c1['b_pool']))).max())
            res.append(dict(h=h, step=s, c0=c0['c'], c1=c1['c'], max_diff_b_pool=d, d_disp=abs(c0['mean_abs_dev_from_pool'] - c1['mean_abs_dev_from_pool'])))
    D = pd.DataFrame(res)
    D['ok'] = (D.max_diff_b_pool < 1e-9) & (D.c0 == D.c1) & (D.d_disp < 1e-9)
    D.to_csv(OUT / 'leak.csv', index=False); print(D.to_string(), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--build', action='store_true'); ap.add_argument('--gibrid', nargs='*'); ap.add_argument('--seeds', nargs='*')
    ap.add_argument('--eval', action='store_true'); ap.add_argument('--leak', action='store_true'); ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--ctl', action='store_true')
    a = ap.parse_args()
    if a.ctl:
        control()
        L.run_gibrid([(f'D24_ctl_h{h}', h) for h in (1, 5, 22)], a.workers)
    if a.build:
        build()
    if a.gibrid is not None:
        L.run_gibrid([(b, hof(b)) for b in (a.gibrid or MAIN)], a.workers)
    if a.seeds is not None:
        L.run_gibrid([(f'{b}_{s}', hof(b)) for b in a.seeds for s in SEEDS], a.workers)
    if a.eval:
        E = pd.DataFrame([r for r in (L.evaluate(b, hof(b)) for b in MAIN) if r]); E.to_csv(OUT / 'eval.csv', index=False)
        with pd.option_context('display.width', 250, 'display.max_columns', 40):
            print(E[['boost', 'design', 'n_failed', 'failed', 'mean_1720', 'har_1720', 'gain_vs_har', 'q2021', 'v0lgb_1720', 'v0lgb_2021',
                     'seed_mean_1720', 'p_novaya_noga', 'p_1pct', 'p_seeds', 'p_2021', 'prohodit']].round(4).to_string())
    if a.leak:
        leak()


if __name__ == '__main__':
    main()
