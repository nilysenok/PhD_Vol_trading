#!/usr/bin/env python3
"""Справочные GARCH(1,1) и GJR-GARCH(1,1) на СВ с калибровкой масштаба (ночной приоритет 2, 26.10; в FINAL не входят).
Спецификация и окно — как scripts/analysis/garch_sv.py: по тикеру, лог-доходности close x100, mean=Constant, dist=normal,
p=1, o=0|1, q=1; шаг k = 1..10: vy = 2015+k, ty = 2016+k; параметры — один раз по годам < vy; на vy и ty — фильтр с теми же
параметрами; прогноз h: среднее ожидаемых σ² за [t+1, t+h] / 1e4 (σ²_GARCH).
Калибровка (новое): на годе оценки шага t-1 (OOS: параметры шага t-1 оценены по годам < vy_{t-1}), по строкам с целью внутри года
(эмбарго), e = ln RV - ln σ²_GARCH; ln a_{t,тикер} = среднее e по тикеру (OLS по логарифмам, наклон 1);
s²_t = дисперсия e - ln a по объединению тикеров (внутритикерная); прогноз шага t на vy_t: σ² * a_t * exp(s²_t / 2).
Тест шага t (ty): a и s² по году оценки шага t (последний год до ty); вариант шага t-1 сохранён.
Справка (без выбора): cal_a (a без F) и raw (σ²_GARCH, как garch_sv.py).
Выход: data/regen/ref_nn_garch/REF_{GARCH,GJR}_h{h}_val.parquet (шаги 2-6 и шаг 1 без калибровки),
data/regen/test_locked/ref_nn_garch/REF_{m}_h{h}_valconfirm.parquet (шаги 7-10), REF_{m}_h{h}_test.parquet (ty шагов 1-10),
data/regen/ref_nn_garch/REF_garch_params.csv, REF_garch_cal_dev.csv (a и s² шагов, применяемые к шагам 2-6).
python3 scripts/analysis/ref_garch_cal.py"""
import os
for _k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'): os.environ[_k] = '1'
import time, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')

REG = Path('/Users/nilysenok/Desktop/Старое/PhD_Vol_trading/data/regen')
FEAT, OUT, TL = REG / 'features_pit_v3.parquet', REG / 'ref_nn_garch', REG / 'test_locked/ref_nn_garch'
HS, NSTEP, DEV_LAST = (1, 5, 22), 10, 6
MODELS = {'GJR': 1, 'GARCH': 0}  # имя -> o
tcol = lambda h: 'rv_target_h1' if h == 1 else f'rv_target_mean_h{h}'


def load(path=FEAT):
    a = pd.read_parquet(path, columns=['date', 'ticker', 'close'] + [tcol(h) for h in HS])
    a['date'] = pd.to_datetime(a.date)
    return a


def filt(e, om, al, ga, be):
    """Условная дисперсия GARCH/GJR — причинная рекурсия arch (backcast: EWMA 0,94 по первым 75 остаткам; на t=0 асимметричный член
    с 0,5*backcast). arch.fix() здесь не используется: его границы дисперсии (variance_bounds) считаются по всей выборке, и
    отфильтрованные σ² на ранних датах меняются в ~1e-11 от данных после t (детектор утечек, 26.10)."""
    tau = min(75, len(e))
    w = 0.94 ** np.arange(tau)
    back = float((w / w.sum()) @ (e[:tau] ** 2))
    s2 = np.empty(len(e))
    s2[0] = om + (al + 0.5 * ga + be) * back
    e2, neg = e ** 2, e < 0
    for t in range(1, len(e)):
        s2[t] = om + (al + ga * neg[t - 1]) * e2[t - 1] + be * s2[t - 1]
    return s2


def garch_ticker(g, tk, steps=range(1, NSTEP + 1), models=MODELS):
    """σ²_GARCH на годах vy и ty каждого шага; emb — цель строки vy кончается внутри vy (по строкам тикера)."""
    from arch import arch_model
    g = g.sort_values('date').reset_index(drop=True)
    yr_all = g.date.dt.year.values
    r = 100 * np.log(g.close).diff()
    ok = r.notna().values
    pos = np.flatnonzero(ok)  # индекс строки тикера (для эмбарго по строкам, как у целей)
    gg, rr, yr = g[ok].reset_index(drop=True), r[ok].values, yr_all[ok]
    rows, prm = [], []
    for step in steps:
        vy, ty = 2015 + step, 2016 + step
        ntr = int((yr < vy).sum())
        i1 = int(np.searchsorted(yr_all, ty))  # первая строка ty в исходной нумерации
        for name, o in models.items():
            res = arch_model(rr[:ntr], mean='Constant', vol='GARCH', p=1, o=o, q=1, dist='normal').fit(disp='off', show_warning=False)
            pa = res.params
            om, al, be, ga = pa['omega'], pa['alpha[1]'], pa['beta[1]'], pa.get('gamma[1]', 0.0)
            e = rr - pa['mu']
            s2 = filt(e, om, al, ga, be)
            nxt = om + (al + ga * (e < 0)) * e ** 2 + be * s2  # σ²_{t+1} по информации до t
            phi = al + ga / 2 + be
            lr = om / (1 - phi) if phi < 1 else np.mean(rr[:ntr] ** 2)
            prm.append(dict(ticker=tk, step=step, model=name, n_train=ntr, omega=om, alpha=al, gamma=ga, beta=be, mu=pa['mu'],
                            persistence=phi, converged=int(res.convergence_flag == 0)))
            for h in HS:
                f = (lr + np.outer(nxt - lr, phi ** np.arange(h))).mean(axis=1) / 1e4
                y = gg[tcol(h)].values
                for part, yy in (('val', vy), ('test', ty)):
                    m = yr == yy
                    rows.append(pd.DataFrame({'model': name, 'h': h, 'date': gg.date.values[m], 'ticker': tk, 'year': yy, 'step': step,
                                              'part': part, 'rv_actual': y[m], 'sig2': np.clip(f[m], 1e-10, None),
                                              'emb': (pos[m] + h <= i1 - 1) if part == 'val' else False}))
    return pd.concat(rows, ignore_index=True), prm


def calibrate(R, leak=False):
    """R: строки одной модели и h (step, part, date, ticker, rv_actual, sig2, emb). Добавляет a, s², rv_pred (a·F), rv_pred_a, rv_pred_raw."""
    v = R[(R.part == 'val') & R.emb & (R.rv_actual > 0)].copy()
    v['e'] = np.log(v.rv_actual) - np.log(v.sig2)
    la = v.groupby(['step', 'ticker']).e.mean().rename('la')
    v = v.join(la, on=['step', 'ticker'])
    s2 = v.assign(u=v.e - v.la).groupby('step').u.var(ddof=0).to_dict()
    la = la.to_dict()
    R = R.copy()
    src = np.where(R.part == 'val', R.step - 1, R.step)  # val шага t — оценка шага t-1; test шага t — оценка шага t
    if leak:  # только самопроверка детектора утечек: калибровка по году оценки шага t+1 (= ty шага t, будущее)
        src = R.step.values + 1
    R['la'] = [la.get((s, tk), np.nan) for s, tk in zip(src, R.ticker)]
    R['s2'] = [s2.get(s, np.nan) for s in src]
    R['la_prev'] = [la.get((s - 1, tk), np.nan) for s, tk in zip(R.step, R.ticker)]
    R['s2_prev'] = [s2.get(s - 1, np.nan) for s in R.step]
    R['rv_pred_raw'] = R.sig2
    R['rv_pred_a'] = R.sig2 * np.exp(R.la)
    R['rv_pred'] = R.rv_pred_a * np.exp(R.s2 / 2)
    return R, la, s2


def run(a, steps=range(1, NSTEP + 1), leak=False):
    """Все тикеры -> {(модель, h): откалиброванная таблица}, параметры, калибровки."""
    rows, prm = [], []
    for tk, g in a.groupby('ticker'):
        r, p = garch_ticker(g, tk, steps)
        rows.append(r); prm += p
    R = pd.concat(rows, ignore_index=True)
    out, cal = {}, []
    for (m, h), d in R.groupby(['model', 'h']):
        c, la, s2 = calibrate(d.drop(columns=['model', 'h']), leak)
        out[(m, h)] = c
        cal += [dict(model=m, h=h, step_src=s, applied_to_step=s + 1, s2=v) for s, v in s2.items() if s + 1 <= DEV_LAST]
        cal += [dict(model=m, h=h, step_src=s, applied_to_step=s + 1, ticker=tk, a=np.exp(v)) for (s, tk), v in la.items() if s + 1 <= DEV_LAST]
    return out, prm, cal


def main():
    OUT.mkdir(parents=True, exist_ok=True); TL.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    out, prm, cal = run(load())
    for (m, h), c in out.items():
        c = c.sort_values(['step', 'part', 'date', 'ticker'])
        dev = c[(c.part == 'val') & (c.step <= DEV_LAST)].dropna(subset=['rv_actual'])
        dev.drop(columns='part').to_parquet(OUT / f'REF_{m}_h{h}_val.parquet', index=False)
        c[(c.part == 'val') & (c.step > DEV_LAST)].drop(columns='part').to_parquet(TL / f'REF_{m}_h{h}_valconfirm.parquet', index=False)
        c[c.part == 'test'].drop(columns='part').to_parquet(TL / f'REF_{m}_h{h}_test.parquet', index=False)
        print(m, h, 'dev строк', len(dev), flush=True)
    pd.DataFrame(prm).to_csv(OUT / 'REF_garch_params.csv', index=False)
    pd.DataFrame(cal).to_csv(OUT / 'REF_garch_cal_dev.csv', index=False)
    print(f'{time.time() - t0:.0f} с', flush=True)


if __name__ == '__main__':
    main()
