"""Д24.9: HAR-J на каждый день t+j, j = 1..h (прямая модель на цель ln RV_{t+j}, те же регрессоры HAR-J, OLS на окне шага, эмбарго - дата
t+j < 1 янв. vy), прогноз h - среднее дневных прогнозов в уровне RV:
  A: mean_j exp(ln-прогноз_j)                       (медианные дневные прогнозы; поправку s² на среднее делает гибрид/оценка);
  B: mean_j exp(ln-прогноз_j + s²_j/2), s²_j - дисперсия остатков обучения модели дня j (несмещённые в уровне дни).
Сравнение - прямой HAR_h (тот же контур с эмбарго, J0 из d24_4; и канон гибрида). h = 5, 22.
Нога гибрида: XGB-слот = Д24.9, LGB-слот = бустинг-нога финалиста (d24b_lib.leg).
python3 scripts/analysis/d24_9.py [--build] [--gibrid] [--seeds ...] [--eval] [--leak]"""
import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import d24b_lib as L

OUT = L.OUT / 'd24_9'
SEEDS = ('s42', 's43', 's44')


def day_models(A, harj, h, vy):
    """Коэффициенты моделей дней j = 1..h шага (обучение < vy, эмбарго по дате t+j) и s²_j обучения."""
    X = L.harj_X(A, harj)
    out = []
    for j in range(1, h + 1):
        yj = A[f'_f{j}']
        tr = ((A.year < vy) & yj.notna() & (A[f'_fd{j}'] < pd.Timestamp(f'{vy}-01-01'))).values
        yl = np.log(yj.values[tr] + 1e-10)
        b = L.ols(X[tr], yl)
        out.append((b, float(np.var(yl - L.ols_pred(b, X[tr])))))
    return X, out


def build_h(A, harj, h):
    y = L.target(A, h)
    VA, TA, VB, TB, info = [], [], [], [], []
    for s, ty, vy in L.STEPS:
        X, ms = day_models(A, harj, h, vy)
        info.append(dict(step=s, s2_j=json.dumps([round(m[1], 4) for m in ms]), b_rvd_j=json.dumps([round(float(m[0][1]), 4) for m in ms])))
        for yr, la, lb in ((vy, VA, VB), (ty, TA, TB)):
            sel = ((A.year == yr) & y.notna()).values
            P = np.column_stack([L.ols_pred(b, X[sel]) for b, _ in ms])
            s2 = np.array([m[1] for m in ms])
            la.append(L.frame(A, sel, s, yr, y, np.exp(P).mean(1)))
            lb.append(L.frame(A, sel, s, yr, y, np.exp(P + s2 / 2).mean(1)))
    c = lambda x: pd.concat(x, ignore_index=True)
    return (c(VA), c(TA)), (c(VB), c(TB)), pd.DataFrame(info)


def build():
    OUT.mkdir(parents=True, exist_ok=True)
    A, harj = L.load()
    rows = []
    for h in (5, 22):
        hv, ht = L.har_ref(A, harj, h)
        rows += L.standalone_rows('HAR-J прямой (эталон, канон)', h, hv, ht)
        J0v = pd.read_parquet(L.OUT / 'd24_4' / f'J0_h{h}_val.parquet')
        rows += L.standalone_rows('HAR-J прямой с эмбарго (J0)', h, J0v)
        (va, ta), (vb, tb), info = build_h(A, harj, h)
        info.to_csv(OUT / f'days_h{h}_coef.csv', index=False)
        for tag, v, t in (('A', va, ta), ('B', vb, tb)):
            v.to_parquet(OUT / f'D{tag}_h{h}_val.parquet', index=False); t.to_parquet(OUT / f'D{tag}_h{h}_test.parquet', index=False)
            rows += L.standalone_rows(f'Д24.9{tag} среднее HAR по дням', h, v, t, codes=('sm', 'smrob', 'bez'))
            lv, lt = L.leg(h, 'main')
            b = f'D24_9{tag.lower()}_h{h}'
            L.save_cand(b, h, v, L.on_frame(v, lv), t, L.on_frame(t, lt))
            for sd in SEEDS:
                L.save_cand(f'{b}_{sd}', h, v, L.on_frame(v, L.leg(h, sd, test=False)[0]))
        print(h, 'готово', flush=True)
    Q = pd.DataFrame(rows); Q.to_csv(OUT / 'standalone.csv', index=False)
    print(Q.round(4).to_string(), flush=True)


MAIN = [f'D24_9{t}_h{h}' for t in ('a', 'b') for h in (5, 22)]
hof = lambda b: int(b.split('_h')[1].split('_')[0])


def leak():
    A, harj = L.load()
    res = []
    for h in (5, 22):
        for s, ty, vy in L.STEPS[:6]:
            _, m0 = day_models(A, harj, h, vy)
            B = L.perturb(A, f'{vy}-01-01', seed=s)
            _, m1 = day_models(B, harj, h, vy)
            d = max(float(np.abs(a[0] - b[0]).max()) for a, b in zip(m0, m1))
            res.append(dict(h=h, step=s, max_dcoef=d, ok=d < 1e-9))
    D = pd.DataFrame(res); D.to_csv(OUT / 'leak.csv', index=False); print(D.to_string(), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--build', action='store_true'); ap.add_argument('--gibrid', nargs='*'); ap.add_argument('--seeds', nargs='*')
    ap.add_argument('--eval', action='store_true'); ap.add_argument('--leak', action='store_true'); ap.add_argument('--workers', type=int, default=2)
    a = ap.parse_args()
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
