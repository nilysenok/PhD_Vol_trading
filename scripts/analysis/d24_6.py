#!/usr/bin/env python3
"""Д24.6: HEAVY (Shephard-Sheppard 2010) и Realized GARCH (Hansen-Huang-Shek 2012, log-линейная (1,1)) на скользящей валидации.
Данные (общий data/regen): features_pit_v3.parquet - close (доходность close-to-close, лог x100) и rv (дневная RV, срез 18:50, x1e4).
Окно: шаг k = 1..10, vy = 2015+k (год оценки/val), ty = 2016+k (тест). QMLE ОДИН раз по расширяющемуся окну (годы < vy, с 2014-06-09),
дальше модель фильтрует с теми же параметрами (как garch_sv.py / ветка GARCH_sv); подбора на vy нет.
HEAVY (две Гаусс-QMLE по раздельности):
  дисперсия доходности  h_t  = w_r + a_r RM_{t-1} + b_r h_{t-1}        (r_t^2 как наблюдаемая, лог-правдоподобие sum ln h + r^2/h)
  ожидание RM           mu_t = w + a RM_{t-1} + b mu_{t-1}               (sum ln mu + RM/mu)   <- прогноз RV, итерация mu_{t+s} = w + (a+b) mu_{t+s-1}
Realized GARCH (совместная QMLE, sigma_u^2 профилируется):
  r_t = sqrt(h_t) z_t;  ln h_t = w + b ln h_{t-1} + g ln x_{t-1};  ln x_t = xi + phi ln h_t + t1 z_t + t2 (z_t^2 - 1) + u_t, u ~ N(0, su2)
  многошаговый прогноз по итерации в логах: l_1 = ln h_{t+1} (известна в t), l_{s+1} = w + g xi + (b + g phi) l_s, E ln x_{t+s} = xi + phi l_s;
  уровень RV = exp(E ln x) (поправка Б0.1 делает гибрид по правилу: четыре оценки s^2 и без поправки).
Прогноз h = среднее ожидаемых RV за [t+1, t+h] / 1e4, h in {1, 5, 22}; ноги Д18.1 / Д18.3: среднее за [6..22] и [2..22].
Страховки данных (ночное умолчание): на входе фильтра RM ограничен 1.0 (raw), |r| <= 70 (x100) - затрагивает ровно одну строку (IRAO 2015-01-20: rv=21,9, r=467%);
цели не меняются. Нули RV (4 строки, 2025) - пол 1e-7 для ln x.
Выход: data/regen/GARCH_sv{heavy,rgarch}_h{1,5,22}_val.parquet (год оценки vy, шаги 1-6) + ноги *_leg6_22/_leg2_22_val.parquet;
data/regen/test_locked/D24_6/: {model}_h{h}_test.parquet (ty, шаги 1-10), {model}_h{h}_valconfirm.parquet (vy шагов 7-10) - не открывать, метрик нет.
python3 scripts/analysis/d24_6.py fit | selftest"""
import os
for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'): os.environ[k] = '1'
import sys, time, warnings
from pathlib import Path
import numpy as np, pandas as pd
from scipy.optimize import minimize
from scipy.signal import lfilter
warnings.filterwarnings('ignore')

REG = Path('/Users/nilysenok/Desktop/Старое/PhD_Vol_trading/data/regen')
FEAT, TL = REG / 'features_pit_v3.parquet', REG / 'test_locked/D24_6'
HS, STEPS, DEV, NS = (1, 5, 22), range(1, 11), 6, 22
MODELS = ('heavy', 'rgarch')
RM_CAP, R_CAP, X_FLOOR = 1.0, 70.0, 1e-7
tcol = lambda h: 'rv_target_h1' if h == 1 else f'rv_target_mean_h{h}'


def rec(c, beta, init):  # y_t = c_t + beta y_{t-1}, y_{-1} = init
    return lfilter([1.0], [1.0, -beta], c, zi=[beta * init])[0]


# ---------------- HEAVY ----------------
def heavy_fit(dep, rm):
    """QMLE одного уравнения: dep_t | F_{t-1} ~ m_t, m_t = w + a rm_{t-1} + b m_{t-1}; sum (ln m + dep/m). dep, rm - масштабированы (~1)."""
    m0 = float(np.mean(dep))

    def nll(p):
        w, a, b = p
        m = rec(w + a * rm[:-1], b, m0)  # m[i] = значение для t=i+1
        if not np.all(m > 1e-10): return 1e10
        return float(np.sum(np.log(m) + dep[1:] / m))
    best = None
    for a0, b0 in ((0.3, 0.6), (0.1, 0.85), (0.5, 0.3)):
        x0 = [m0 * max(1e-3, 1 - a0 - b0), a0, b0]
        r = minimize(nll, x0, method='SLSQP', bounds=[(1e-8, None), (0, 1), (0, 0.9995)],
                     constraints=[{'type': 'ineq', 'fun': lambda p: 0.9995 - p[1] - p[2]}], options=dict(maxiter=300, ftol=1e-10))
        if best is None or r.fun < best.fun: best = r
    return best.x, bool(best.success), float(-best.fun), m0


def heavy_paths(w, a, b, m0, rm, ntr, mean_ref):
    """E[RM_{t+s}|F_t], s=1..NS, для каждой строки t (RM_t - последнее наблюдение)."""
    mn = rec(w + a * rm, b, m0)  # mn[t] = mu_{t+1}
    phi = a + b
    lr = w / (1 - phi) if phi < 0.9995 else mean_ref
    return lr + (mn - lr)[:, None] * phi ** np.arange(NS)[None, :], mn


# ---------------- Realized GARCH ----------------
def rg_lh(w, b, g, lx, lh0):
    # ln h_t (t>=1) = w + b ln h_{t-1} + g lx_{t-1}; ln h_0 = lh0
    out = np.empty(len(lx)); out[0] = lh0
    out[1:] = rec(w + g * lx[:-1], b, lh0)
    return out


def rg_nll(p, r, lx, lh0, full=False):
    w, b, g, xi, ph, t1, t2 = p
    lh = rg_lh(w, b, g, lx, lh0)
    z = r / np.exp(lh / 2)
    u = lx - xi - ph * lh - t1 * z - t2 * (z * z - 1)
    su2 = float(np.mean(u[1:] ** 2))
    if not np.isfinite(su2) or su2 <= 0: return 1e10
    v = 0.5 * np.sum(lh[1:] + (r[1:] / np.exp(lh[1:] / 2)) ** 2) + 0.5 * (len(r) - 1) * (np.log(su2) + 1)
    return (v, su2) if full else float(v)


def rg_fit(r, x):
    lx = np.log(np.clip(x, X_FLOOR * 1e4, None)); lh0 = float(np.log(np.mean(r ** 2)))
    best = None
    for b0, g0, f0 in ((0.5, 0.4, 1.0), (0.8, 0.15, 1.0), (0.3, 0.6, 0.9)):
        xi0 = float(np.mean(lx) - f0 * lh0 + 0.0)
        x0 = [lh0 * (1 - b0 - g0 * f0) * 0.0 + 0.0, b0, g0, xi0, f0, -0.05, 0.05]
        bnds = [(-5, 5), (0, 0.999), (0, 1.5), (-6, 6), (0.1, 2.0), (-1, 1), (-1, 1)]
        res = minimize(rg_nll, x0, args=(r, lx, lh0), method='SLSQP', bounds=bnds,
                       constraints=[{'type': 'ineq', 'fun': lambda p: 0.999 - p[1] - p[2] * p[4]}], options=dict(maxiter=400, ftol=1e-10))
        if best is None or res.fun < best.fun: best = res
    v, su2 = rg_nll(best.x, r, lx, lh0, full=True)
    return best.x, su2, bool(best.success), float(-v), lh0


def rg_paths(p, lx, lh0, mean_lh):
    w, b, g, xi, ph = p[:5]
    l1 = np.empty(len(lx)); l1[:] = w + b * rg_lh(w, b, g, lx, lh0) + g * lx  # ln h_{t+1} | F_t
    pers = min(b + g * ph, 0.999)
    c = w + g * xi
    L = np.empty((len(lx), NS)); L[:, 0] = l1
    for s in range(1, NS): L[:, s] = c + pers * L[:, s - 1]
    return np.exp(xi + ph * L)


def paths_to_f(P):
    return {1: P[:, 0], 5: P[:, :5].mean(1), 22: P.mean(1), 'leg6_22': P[:, 5:22].mean(1), 'leg2_22': P[:, 1:22].mean(1)}


def fit_forecast(rr, xx, ntr):
    """rr, xx - массивы доходностей (x100, обрезаны) и RV (x1e4, обрезаны) строк тикера (без первой строки); окно оценки - первые ntr строк.
    Возвращает {модель: (словарь прогнозов в сырой шкале x1e-4... делится позже, параметры)}."""
    out = {}
    # HEAVY
    pw, ok_rm, ll_rm, m0 = heavy_fit(xx[:ntr], xx[:ntr])
    ph_, ok_r, ll_r, h0 = heavy_fit(rr[:ntr] ** 2, xx[:ntr])
    P, mn = heavy_paths(*pw, m0, xx, ntr, float(np.mean(xx[:ntr])))
    hr = rec(ph_[0] + ph_[1] * xx, ph_[2], h0)  # h_{t+1} для справки
    out['heavy'] = (paths_to_f(P), dict(w=pw[0], a=pw[1], b=pw[2], w_r=ph_[0], a_r=ph_[1], b_r=ph_[2], persistence=pw[1] + pw[2],
                                        converged=int(ok_rm and ok_r), loglik=ll_rm + ll_r))
    # Realized GARCH
    p, su2, ok, ll, lh0 = rg_fit(rr[:ntr], xx[:ntr])
    lx = np.log(np.clip(xx, X_FLOOR * 1e4, None))
    P = rg_paths(p, lx, lh0, None)
    out['rgarch'] = (paths_to_f(P), dict(w=p[0], b=p[1], g=p[2], xi=p[3], phi=p[4], tau1=p[5], tau2=p[6], su2=su2, persistence=p[1] + p[2] * p[4],
                                         converged=int(ok), loglik=ll))
    return out


def prep(a, tk):
    g = a[a.ticker == tk].sort_values('date').reset_index(drop=True)
    g['r'] = 100 * np.log(g.close).diff()
    g = g[g.r.notna()].reset_index(drop=True)
    g['rr'] = g.r.clip(-R_CAP, R_CAP)
    g['xx'] = np.minimum(g.rv.values, RM_CAP) * 1e4
    return g


def main():
    TL.mkdir(parents=True, exist_ok=True)
    cols = ['date', 'ticker', 'close', 'rv'] + [tcol(h) for h in HS]
    a = pd.read_parquet(FEAT, columns=cols)
    a['date'] = pd.to_datetime(a.date); a['year'] = a.date.dt.year
    out = {(m, k, p): [] for m in MODELS for k in list(HS) + ['leg6_22', 'leg2_22'] for p in ('val', 'conf', 'test')}
    prm = []; t0 = time.time()
    for tk in sorted(a.ticker.unique()):
        gg = prep(a, tk)
        g_t = gg.set_index('date')
        # цель ног: среднее rv за [t+6..t+22] / [t+2..t+22] (rv не обрезан) - только для столбца rv_actual
        rv = gg.rv.values
        sub = lambda lo, hi: np.array([np.mean(rv[i + lo:i + hi + 1]) if i + hi < len(rv) else np.nan for i in range(len(rv))])
        tgt_leg = {'leg6_22': sub(6, 22), 'leg2_22': sub(2, 22)}
        for step in STEPS:
            vy, ty = 2015 + step, 2016 + step
            ntr = int((gg.year < vy).sum())
            res = fit_forecast(gg.rr.values, gg.xx.values, ntr)
            for m, (F, par) in res.items():
                prm.append(dict(ticker=tk, step=step, model=m, n_train=ntr, **par))
                for k, f in F.items():
                    t = gg[tcol(k)].values if k in HS else tgt_leg[k]
                    f = f / 1e4
                    for part, yr in (('val', vy), ('test', ty)):
                        sel = (gg.year == yr).values
                        if not (part == 'test' and step == 10): sel &= ~np.isnan(t)
                        fr = pd.DataFrame({'date': gg.date.values[sel], 'ticker': tk, 'year': yr, 'step': step, 'point': -1,
                                           'rv_actual': t[sel], 'rv_pred': np.clip(f[sel], 1e-10, None)})
                        out[(m, k, 'test' if part == 'test' else ('val' if step <= DEV else 'conf'))].append(fr)
        print(f'{time.time() - t0:.0f} с {tk}', flush=True)
    for (m, k, p), L in out.items():
        d = pd.concat(L, ignore_index=True).sort_values(['step', 'date', 'ticker']).reset_index(drop=True)
        nm = f'GARCH_sv{m}_' + (f'h{k}' if k in HS else k)
        fp = {'val': REG / f'{nm}_val.parquet', 'conf': TL / f'{nm}_valconfirm.parquet', 'test': TL / f'{nm}_test.parquet'}[p]
        d.to_parquet(fp, index=False); print(fp.name, len(d), flush=True)
    pd.DataFrame(prm).to_csv(REG / 'D24_6_params.csv', index=False)


def selftest():
    """«Будущее не влияет»: после даты отсечки cut внутри года оценки шага все входы (close, rv) умножаются на случайные множители; прогнозы на строках
    с date <= cut не меняются (параметры - по годам < vy, фильтр - только по прошлому). Контроль: прогнозы после cut меняются."""
    a = pd.read_parquet(FEAT, columns=['date', 'ticker', 'close', 'rv'] + [tcol(h) for h in HS]); a['date'] = pd.to_datetime(a.date); a['year'] = a.date.dt.year
    rs = np.random.default_rng(0); rows = []
    for tk in ('SBER', 'LKOH', 'IRAO'):
        gg = prep(a, tk)
        for step in (1, 3, 6, 9):
            vy = 2015 + step; ntr = int((gg.year < vy).sum()); cut = pd.Timestamp(f'{vy}-06-30')
            base = fit_forecast(gg.rr.values, gg.xx.values, ntr)
            fut = (gg.date > cut).values
            # пересчёт доходностей и RV из возмущённых close/rv, чтобы проверить весь путь prep -> fit
            b = a[a.ticker == tk].sort_values('date').reset_index(drop=True)
            fb = (b.date > cut).values
            b.loc[fb, 'close'] *= np.exp(rs.normal(0, 0.1, fb.sum()).cumsum()); b.loc[fb, 'rv'] *= rs.uniform(0.2, 5, fb.sum())
            g2 = b.copy(); g2['r'] = 100 * np.log(g2.close).diff(); g2 = g2[g2.r.notna()].reset_index(drop=True)
            g2['rr'] = g2.r.clip(-R_CAP, R_CAP); g2['xx'] = np.minimum(g2.rv.values, RM_CAP) * 1e4
            assert len(g2) == len(gg) and (g2.year < vy).sum() == ntr
            pert = fit_forecast(g2.rr.values, g2.xx.values, ntr)
            past = (gg.date <= cut).values
            for m in MODELS:
                for k in (1, 5, 22, 'leg6_22'):
                    f0, f1 = base[m][0][k], pert[m][0][k]
                    rows.append(dict(ticker=tk, step=step, model=m, k=k, max_diff_past=float(np.abs(f0[past] - f1[past]).max()),
                                     max_diff_future=float(np.abs(f0[fut] - f1[fut]).max())))
    R = pd.DataFrame(rows); R['ok'] = (R.max_diff_past < 1e-9) & (R.max_diff_future > 1e-6)
    R.to_csv(REG / 'D24_6_selftest.csv', index=False)
    print(R.to_string()); print('ВСЕ OK' if R.ok.all() else 'ЕСТЬ ОШИБКИ'); return bool(R.ok.all())


def d18_dev(steps_hi=DEV):
    """Д18.1 со своей ногой: XGB-слот = (5*hyb5 + 17*leg[6..22])/22, LGB-слот = Д18.3-аналог (1*hyb1 + 21*leg[2..22])/22; hyb5 = D12_h5 wrap2 XGB, hyb1 = D12_h1_A_d812 top5 XGB
    (сырые OOS-прогнозы val, как d18_struct.load_hyb). Нога - GARCH_sv{heavy,rgarch}_leg*_val.parquet (шаги 1-6). Выход: data/regen/D24_6/D24_6_d18{h,r}_{xgb,lgbm}_h22_val.parquet."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import d18_struct as S18
    from gibrid_io import load_boost
    a = pd.read_parquet(FEAT, columns=['date', 'ticker', 'rv_target_h1', 'rv_target_mean_h5', 'rv_target_mean_h22']); a['date'] = pd.to_datetime(a.date); a['year'] = a.date.dt.year
    T = S18.target_df(a, 22, 'val'); T = T[T.step <= steps_hi]
    h5 = load_boost('D12_h5', 5, REG, 'wrap2')[0]['val'][['date', 'ticker', 'step', 'rv_pred']]
    h1 = load_boost('D12_h1_A_d812', 1, REG, 'top5')[0]['val'][['date', 'ticker', 'step', 'rv_pred']]
    d = REG / 'D24_6'; d.mkdir(exist_ok=True)
    cols = ['date', 'ticker', 'year', 'step', 'rv_actual', 'rv_pred']
    for m, tag in (('heavy', 'h'), ('rgarch', 'r')):
        l6 = pd.read_parquet(REG / f'GARCH_sv{m}_leg6_22_val.parquet')[['date', 'ticker', 'step', 'rv_pred']]
        l2 = pd.read_parquet(REG / f'GARCH_sv{m}_leg2_22_val.parquet')[['date', 'ticker', 'step', 'rv_pred']]
        x = S18.combine([h5, l6], [5 / 22, 17 / 22], tgt=T)[cols]; l = S18.combine([h1, l2], [1 / 22, 21 / 22], tgt=T)[cols]
        for nm, v in (('xgb', x), ('lgbm', l)):
            v.to_parquet(d / f'D24_6_d18{tag}_{nm}_h22_val.parquet', index=False)
        print(m, len(x), len(l), flush=True)


# ---------------- оценка (только шаги 2-6; шаги 7-9 не читаются) ----------------
def dm(a, b, h):  # d = L_a - L_b по дню (среднее по тикерам), NW(h-1); t > 0 - b лучше
    qv = lambda y, p: y / np.clip(p, 1e-10, None) - np.log(y / np.clip(p, 1e-10, None)) - 1
    m = a.merge(b[['date', 'ticker', 'rv_pred']], on=['date', 'ticker'], suffixes=('_a', '_b'))
    m['d'] = qv(m.rv_actual, m.rv_pred_a) - qv(m.rv_actual, m.rv_pred_b)
    d = m.groupby('date').d.mean().values; n = len(d); u = d - d.mean(); L = h - 1
    v = u @ u / n + 2 * sum((1 - l / (L + 1)) * (u[l:] @ u[:-l]) / n for l in range(1, L + 1))
    return float(d.mean() / np.sqrt(v / n)), len(m)


def corr_pred(boost, h, code, comp, raw, steps):
    """raw (date,ticker,step,rv_actual,rv_pred) x F_oos шага (s^2 шага t-1) оценки code из bias_factors гибрида boost."""
    raw = raw[raw.step.between(*steps) & (raw.rv_actual > 1e-12)]
    if code == 'bez': return raw[['date', 'ticker', 'step', 'rv_actual', 'rv_pred']]
    bf = pd.read_csv(REG / f'{boost}_gibrid' / f'bias_factors_{boost}_h{h}.csv'); bf = bf[(bf.comp == comp) & (bf.code == code)]
    m = raw.merge(bf[['step', 'F_oos']], on='step'); return m.assign(rv_pred=m.rv_pred * m.F_oos)[['date', 'ticker', 'step', 'rv_actual', 'rv_pred']]


def ocenka():
    from scipy.special import ndtr
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import gibrid_har as GH
    A, harj = GH.load_pit(FEAT)
    rows, allv = [], []
    for m in MODELS:
        boost = f'GARCH_sv{m}'
        for h in HS:
            D = pd.read_csv(REG / f'{boost}_gibrid' / f'designs_val_{boost}_h{h}.csv'); D = D[D.step.between(2, DEV)]
            hv, _, _ = GH.har_forecasts(h, A, harj)
            # HAR-J с поправкой: оценка s^2 по q 2017-2020 среди четырёх (ord/rob/sm/smrob)
            hq = {c: D[D.design == f'single|V0_HAR|B-{c}|L0'].set_index('val_year').q_hyb for c in ('ord', 'rob', 'sm', 'smrob')}
            hc = min(hq, key=lambda c: hq[c].loc[2017:2020].mean()); ref = hq[hc]
            vs = []
            for c in ('ord', 'rob', 'sm', 'smrob', 'bez'):
                g = D[D.design == 'single|V0_XGB|' + ('bez' if c == 'bez' else 'B-' + c) + '|L0'].set_index('val_year')
                nf = [k for k in ('c1', 'c2', 'c3', 'c4') if not bool(g[k + '_h'].iloc[0])]
                vs.append(dict(h=h, model=m, s2=c, n_failed=len(nf), failed=','.join(nf), **{str(y): float(g.q_hyb[y]) for y in range(2017, 2022)},
                               q1720=float(g.q_hyb.loc[2017:2020].mean()), q1721=float(g.q_hyb.mean())))
            V = pd.DataFrame(vs); allv.append(V)
            win = V.sort_values(['n_failed', 'q1720']).iloc[0]; c = win.s2
            raw = pd.read_parquet(REG / f'{boost}_h{h}_val.parquet')[['date', 'ticker', 'step', 'rv_actual', 'rv_pred']]
            mp = corr_pred(boost, h, c, 'xgb', raw, (2, 5)); hp = corr_pred(boost, h, hc, 'har', hv, (2, 5))
            t, n = dm(hp, mp, h)  # t > 0 - модель лучше HAR-J с поправкой (2017-2020)
            mp6 = corr_pred(boost, h, c, 'xgb', raw, (6, 6)); hp6 = corr_pred(boost, h, hc, 'har', hv, (6, 6)); t6, n6 = dm(hp6, mp6, h)
            q = {y: win[str(y)] for y in range(2017, 2022)}
            rows.append(dict(h=h, model=m, s2=c, failed=win.failed or '-', n_failed=win.n_failed, **{str(y): round(v, 4) for y, v in q.items()},
                             q1720=round(win.q1720, 4), harj_s2=hc, **{f'harj_{y}': round(float(ref[y]), 4) for y in range(2017, 2022)},
                             harj_1720=round(float(ref.loc[2017:2020].mean()), 4),
                             gain1720_pct=round(100 * (1 - win.q1720 / ref.loc[2017:2020].mean()), 2), gain2021_pct=round(100 * (1 - q[2021] / ref[2021]), 2),
                             gain2020_pct=round(100 * (1 - q[2020] / ref[2020]), 2), dm_t_1720=round(t, 2), dm_p_1720=round(1 - float(ndtr(t)), 4),
                             dm_t_2021=round(t6, 2),
                             passed=bool(win.q1720 <= 0.99 * ref.loc[2017:2020].mean() and q[2021] <= ref[2021])))
    R = pd.DataFrame(rows); R.to_csv(REG / 'D24_6_ocenka_dev.csv', index=False)
    pd.concat(allv).to_csv(REG / 'D24_6_ocenka_s2_dev.csv', index=False)
    pd.set_option('display.width', 250); print(R.to_string()); return R


def ocenka_d18():
    """Представитель (n_failed, затем q 2017-2020; шаги 2-6) по всему пространству гибрида для D24_6_d18{h,r} против HAR-J с поправкой и против D18_h22 (нога HAR-J)."""
    rows = []
    base = pd.read_csv(REG / 'D18/gibrid_representatives.csv'); base = base[base.boost == 'D18_h22'].iloc[0]
    for tag in ('h', 'r'):
        b = f'D24_6_d18{tag}'
        D = pd.read_csv(REG / 'D24_6' / f'{b}_gibrid' / f'designs_val_{b}_h22.csv'); D = D[D.step.between(2, DEV)]
        hq = {c: D[D.design == f'single|V0_HAR|B-{c}|L0'].set_index('val_year').q_hyb for c in ('ord', 'rob', 'sm', 'smrob')}
        hc = min(hq, key=lambda c: hq[c].loc[2017:2020].mean()); ref = hq[hc]
        R_ = []
        for nm, g in D.groupby('design', sort=False):
            g = g.set_index('val_year'); nf = [k for k in ('c1', 'c2', 'c3', 'c4') if not bool(g[k + '_h'].iloc[0])]
            R_.append(dict(design=nm, n_failed=len(nf), failed=','.join(nf), **{str(y): float(g.q_hyb[y]) for y in range(2017, 2022)}, q1720=float(g.q_hyb.loc[2017:2020].mean())))
        R_ = pd.DataFrame(R_).sort_values(['n_failed', 'q1720'])
        w = R_[R_.design.str.startswith('single|V0_XGB|')].iloc[0]  # основная строка: структурный прогноз в слоте V0 (как одиночные модели), s2 по правилу
        z = R_[~R_.design.str.contains('V0_HAR|V0_HCJ')].iloc[0]  # справка: лучший по правилу дизайн пространства, не V0 на HAR-J/HAR-CJ
        rows.append(dict(cand=b, leg='HEAVY' if tag == 'h' else 'Realized GARCH', design=w.design, failed=w.failed or '-', **{str(y): round(w[str(y)], 4) for y in range(2017, 2022)},
                         q1720=round(w.q1720, 4), harj_s2=hc, harj_1720=round(float(ref.loc[2017:2020].mean()), 4), harj_2021=round(float(ref[2021]), 4),
                         gain1720_pct=round(100 * (1 - w.q1720 / ref.loc[2017:2020].mean()), 2), gain2021_pct=round(100 * (1 - w['2021'] / ref[2021]), 2),
                         gain2020_pct=round(100 * (1 - w['2020'] / ref[2020]), 2), vs_D18_pct=round(100 * (1 - w.q1720 / base.q1720), 2),
                         vs_D18_2021_pct=round(100 * (1 - w['2021'] / base['2021']), 2),
                         passed=bool(w.q1720 <= 0.99 * ref.loc[2017:2020].mean() and w['2021'] <= ref[2021]),
                         space_best=z.design, space_failed=z.failed or '-', space_q1720=round(z.q1720, 4), space_2020=round(z['2020'], 4), space_2021=round(z['2021'], 4)))
        R_.to_csv(REG / 'D24_6' / f'{b}_designs_rank_dev.csv', index=False)
    R = pd.DataFrame(rows); R.to_csv(REG / 'D24_6_ocenka_d18_dev.csv', index=False); pd.set_option('display.width', 250); print(R.to_string()); return R


def kandidaty():
    """Прошедшие (ocenka: >= 1% к HAR-J с поправкой по 2017-2020 и не хуже на 2021) -> файлы кандидатов третьего круга + yaml.
    Val шагов 1-9 -> data/regen/D24_6_{m}_{xgb|lgbm}_h{h}_val.parquet (метрик шагов 7-9 не считается: гибрид их скрывает); тест шагов 1-9 ->
    test_locked/D24_6_{m}/..._test.parquet; шаг 10 (val 2025 / тест 2026) -> D24_6/step10 и test_locked/D24_6_{m}/step10 (формат final2_step10). Слоты xgb и lgbm - одна модель."""
    R = pd.read_csv(REG / 'D24_6_ocenka_dev.csv'); R = R[R.passed]
    cols = ['date', 'ticker', 'year', 'step', 'rv_actual', 'rv_pred']; out = []
    for m, g in R.groupby('model'):
        b = f'D24_6_{m}'; hs = sorted(g.h.astype(int)); td, s10v, s10t = REG / 'test_locked' / b, REG / 'D24_6/step10', REG / 'test_locked' / b / 'step10'
        for d in (td, s10v, s10t): d.mkdir(parents=True, exist_ok=True)
        for h in hs:
            nm = f'GARCH_sv{m}_h{h}'
            val = pd.concat([pd.read_parquet(REG / f'{nm}_val.parquet'), pd.read_parquet(TL / f'{nm}_valconfirm.parquet')])
            tst = pd.read_parquet(TL / f'{nm}_test.parquet')
            for sl in ('xgb', 'lgbm'):
                val[val.step <= 9][cols].to_parquet(REG / f'{b}_{sl}_h{h}_val.parquet', index=False)
                tst[tst.step <= 9][cols].to_parquet(td / f'{b}_{sl}_h{h}_test.parquet', index=False)
                val[val.step == 10][cols].assign(seed=42).to_parquet(s10v / f'{b}_{sl}_h{h}_s10_val.parquet', index=False)
                tst[tst.step == 10][cols].assign(seed=42).to_parquet(s10t / f'{b}_{sl}_h{h}_s10_test.parquet', index=False)
        nice = {'heavy': ('HEAVY (Shephard-Sheppard 2010)', 'HEAVY: два уравнения (дисперсия доходности и ожидание RM) GARCH-типа, раздельная Гаусс-QMLE на расширяющемся окне шага; прогноз среднего RV за h по итерации mu_{t+s} = w + (a+b) mu_{t+s-1}; слоты xgb и lgbm - одна модель; поправка Б0.1 - по правилу гибрида'),
                'rgarch': ('Realized GARCH (Hansen-Huang-Shek 2012)', 'Realized GARCH (1,1) лог-линейный: ln h_t = w + b ln h_{t-1} + g ln x_{t-1}, ln x_t = xi + phi ln h_t + tau(z_t) + u_t, совместная QMLE на расширяющемся окне шага; прогноз среднего RV за h по итерации в логах; слоты xgb и lgbm - одна модель; поправка Б0.1 - по правилу гибрида')}[m]
        y = (f"boost: {b}\nh: {hs}\nsemeystvo: \"{nice[0]}\"\nrychag: Д24.6\nopisanie: \"{nice[1]}\"\nregen: data/regen\nlegs: [single]\n"
             f"odinochnaya: false   # ↺ ночное умолчание: явное поручение владельца 26.10 02:15 (кандидат V0) важнее решения 01:12; true - не входит в пул\n"
             f"chistyy: false\ntest_dir: data/regen/test_locked/{b}\nosnova: {b}\nstep10_dir: data/regen/D24_6/step10\nstep10_lock: data/regen/test_locked/{b}/step10\n"
             f"features_har: data/regen/features_pit_v3.parquet\nn_variantov: 2   # HEAVY и Realized GARCH, по одной спецификации; поправок четыре + без поправки\n"
             f"god2021: \"2021 использован как отложенный: оценка s2 и спецификация выбирались по 2017-2020 (шаги 2-5); 2021 смотрелся после\"\n")
        yd = REG / 'final3/kandidaty'; yd.mkdir(parents=True, exist_ok=True); (yd / f'{b}.yaml').write_text(y)
        out.append((b, hs)); print(b, hs, flush=True)
    return out


def kandidat_d18():
    """D24_6_d18r (Д18.1 с ногой Realized GARCH): val шагов 1-9 (data/regen), тест шагов 1-9 (test_locked/D24_6_d18r, ноги hyb5/hyb1 - xgb_raw файлов сборки
    второго круга, как d18_struct.load_hyb; метрики теста не считаются) + yaml. Шага 10 нет (↺: final3 берёт шаги 1-9)."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import d18_struct as S18
    a = pd.read_parquet(FEAT, columns=['date', 'ticker', 'rv_target_h1', 'rv_target_mean_h5', 'rv_target_mean_h22']); a['date'] = pd.to_datetime(a.date); a['year'] = a.date.dt.year
    T = (S18.target_df(a, 22, 'val'), S18.target_df(a, 22, 'test')); T = [t[t.step <= 9] for t in T]
    hyb = S18.load_hyb(); cols = ['date', 'ticker', 'year', 'step', 'rv_actual', 'rv_pred']
    m = 'rgarch'; b = 'D24_6_d18r'; td = REG / 'test_locked' / b; td.mkdir(parents=True, exist_ok=True)
    leg = {k: (pd.concat([pd.read_parquet(REG / f'GARCH_sv{m}_{k}_val.parquet'), pd.read_parquet(TL / f'GARCH_sv{m}_{k}_valconfirm.parquet')]),
               pd.read_parquet(TL / f'GARCH_sv{m}_{k}_test.parquet')) for k in ('leg6_22', 'leg2_22')}
    for sl, (hh, k, w) in (('xgb', (5, 'leg6_22', 5 / 22)), ('lgbm', (1, 'leg2_22', 1 / 22))):
        for i, sfx in enumerate(('val', 'test')):
            l = leg[k][i][lambda d: d.step <= 9][['date', 'ticker', 'step', 'rv_pred']]
            x = S18.combine([hyb[hh][i], l], [w, 1 - w], tgt=T[i])[cols]
            x.to_parquet((REG if sfx == 'val' else td) / f'{b}_{sl}_h22_{sfx}.parquet', index=False); print(sl, sfx, len(x), x.step.max(), flush=True)
    y = (f"boost: {b}\nh: 22\nsemeystvo: \"структурный гибрид по горизонту (Д18)\"\nrychag: Д18.1 с ногой Д24.6 (Realized GARCH)\n"
         f"opisanie: \"(5*hyb5 + 17*RG[6..22])/22 на уровне RV: hyb5 = D12_h5 wrap2 XGB (OOS, сырой), RG[6..22] = среднее ожидаемых RV за t+6..t+22 по итерации Realized GARCH (1,1), "
         f"QMLE на расширяющемся окне шага; поправка Б0.1 на итог; LGB-слот - аналог Д18.3 (1*hyb1 + 21*RG[2..22])/22\"\nregen: data/regen\nlegs: [single]\n"
         f"odinochnaya: false\nchistyy: false\ntest_dir: data/regen/test_locked/{b}\nosnova: D18\nfeatures_har: data/regen/features_pit_v3.parquet\nn_variantov: 2   # ноги HEAVY и Realized GARCH\n"
         f"god2021: \"2021 использован как отложенный: нога и s2 выбирались по 2017-2020\"\n")
    (REG / 'final3/kandidaty' / f'{b}.yaml').write_text(y)


if __name__ == '__main__':
    {'fit': main, 'selftest': selftest, 'ocenka': ocenka, 'd18': d18_dev, 'ocenka_d18': ocenka_d18, 'kand': kandidaty, 'kand_d18': kandidat_d18}[sys.argv[1] if len(sys.argv) > 1 else 'fit']()
