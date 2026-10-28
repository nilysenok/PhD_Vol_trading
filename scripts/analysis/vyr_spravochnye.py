#!/usr/bin/env python3
"""ВЫР-2 (26.10): аудит информационного множества справочных моделей главы 3 и перестройка отстающих в версии d0.
Решение владельца 26.10, 14:20: единое информационное множество — данные по день d включительно, прогноз на d+1..d+h.

Модели (ряды val 2017–2021 = шаги 2–6; год оценки vy = 2015 + k):
  V0 xgb/lgbm  data/regen/ref_v0_base/REF_V0B_{m}_h{h}_val.parquet           (scripts/analysis/ref_v0_base.py)
  GRU/LSTM     data/regen/ref_nn_garch/REF_{GRU,LSTM}_h{h}_val.parquet        (ref_nn_sv.py)
  GARCH/GJR    data/regen/ref_nn_garch/REF_{GARCH,GJR}_h{h}_val.parquet       (ref_garch_cal.py)
  HEAVY/RG     data/regen/GARCH_sv{heavy,rgarch}_h{h}_val.parquet             (d24_6.py)
  HAR-CJ       пересчитывается здесь: gibrid_har.load_pit + gibrid_harcj.add_cj + gibrid_har.har_forecasts (как gibrid_v17, V0_HCJ)
  HAR-J        data/regen/D16_10_harj_h{h}_val.parquet — контроль метода (отставание известно, docs/otchety/2026-10-26_proverka_laga_harj.md)

Режимы:
  audit  — Спирмен прогноза с RV_d и RV_{d−1} (по объединению строк и среднее внутри тикера), с r²_d и r²_{d−1} (доходность
           close-to-close ×100), МНК ln-прогноза на 1, ln RV_d, ln RV_{d−1}; QLIKE по годам 2017–2021 (основной вариант модели).
           Только шаги 2–6. -> data/regen/vyr/ref_spravochnye/audit_lag.csv, qlike_dev.csv
  harcj  — HAR-CJ в четырёх версиях ('' — как в коде; _emb; _d0; _emb_d0 — основная) на шагах 1–10, те же регрессоры и OLS в логах,
           поправка Б0.1 sm (val шага k — s² шагов k−3..k−1, тест шага k — s² шагов k−2..k, как gibrid_bias), без подбора.
           d0: C_d = BV_d, C_w = среднее BV_{d−4..d}, C_m = среднее BV_{d−21..d}; J_d, J_w, J_m так же по jump (нулевой BV — последним
           положительным BV тикера, как нога _d0 в r2_nogi.py). emb: строка t обучает шаг, только если год даты строки t+h тикера < vy;
           остатки для s² — только строки, чья цель кончается до года, к которому s² применяется.
           Открыто: data/regen/vyr/ref_harcj_d0/REF_HARCJ_d0_h{h}_val.parquet (шаги 1–6). Закрыто, без метрик:
           data/regen/test_locked/vyr/ref_harcj_d0/REF_HARCJ_d0_h{h}_{valconfirm (шаги 7–10), test (ty шагов 1–10)}.parquet + SHA256.txt.
           Сверка: версия '' своим драйвером = gibrid_har.har_forecasts (шаги 1–9).
  leak   — «будущее не влияет» для HAR-CJ _emb_d0: после среза T все входы (rv, bv, jump) × U(0,5; 2) и цели, чей горизонт заходит за T;
           середина года — прогнозы val строк <= T и коэффициенты/F шагов с vy <= год T побайтно; граница года (31.12) — коэффициенты и F
           шага vy = год T + 1 побайтно (эмбарго; у _d0 без эмбарго они меняются — справка); подлог C_d := BV дня d+1 должен быть пойман.
python3 scripts/analysis/vyr_spravochnye.py audit|harcj|leak"""
import os
for _k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_k] = '1'
import sys, hashlib, argparse, warnings
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import LinearRegression

warnings.filterwarnings('ignore')
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gibrid_har as GH
import gibrid_harcj as HC

DATA = Path('/Users/nilysenok/Desktop/Старое/PhD_Vol_trading/data')
REG = DATA / 'regen'
FEAT = REG / 'features_pit_v3.parquet'
OUT_A = REG / 'vyr' / 'ref_spravochnye'
OUT_V = REG / 'vyr' / 'ref_harcj_d0'
TL = REG / 'test_locked' / 'vyr' / 'ref_harcj_d0'
HS, STEPS, DEV_LO, DEV_HI = (1, 5, 22), range(1, 11), 2, 6
CJ = HC.CJ
VERS = {'': (False, False), '_emb': (True, False), '_d0': (False, True), '_emb_d0': (True, True)}
OSN = '_emb_d0'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def qlike_rows(y, f):
    f = np.clip(f, 1e-10, None)
    return y / f - np.log(y / f) - 1


def qlike_years(d, col='rv_pred'):
    """d: date, rv_actual, col; только шаги 2–6, rv_actual > 0 -> {год: QLIKE}, среднее по годам."""
    d = d[(d.step >= DEV_LO) & (d.step <= DEV_HI) & (d.rv_actual > 0)]
    q = d.assign(q=qlike_rows(d.rv_actual.values, d[col].values)).groupby(pd.to_datetime(d.date).dt.year).q.mean()
    assert q.index.max() <= 2021
    return q.to_dict(), float(q.mean())


# ================================================================ HAR-CJ
def load_cj(d0, perturb=None, plant=False, seed=20261026):
    """Матрица HAR-CJ. perturb — срез T: входы rv/bv/jump после T и цели, чей горизонт заходит за T, × U(0,5; 2) (до построения признаков).
    plant — подлог: C_d := значение строки d+1 тикера (BV дня d+1 у d0)."""
    A, _ = GH.load_pit(FEAT)
    A['date'] = pd.to_datetime(A['date'])
    g = A.groupby('ticker', sort=False)
    for h in HS:
        A[f'_tend{h}'] = g['date'].shift(-h)
    if perturb is not None:
        T = pd.Timestamp(perturb)
        rng = np.random.default_rng(seed)
        m = (A.date > T).values
        for c in ('rv', 'bv', 'jump'):
            A.loc[m, c] = A.loc[m, c].values * rng.uniform(0.5, 2.0, m.sum())
        for h in HS:
            mt = (A[f'_tend{h}'] > T).values
            A.loc[mt, f'rv_target_h{h}'] = A.loc[mt, f'rv_target_h{h}'].values * rng.uniform(0.5, 2.0, mt.sum())
    if d0:
        A = A.copy()
        g = A.groupby('ticker', sort=False)
        bvp = A['bv'].where(A['bv'] > 0).groupby(A['ticker']).ffill().fillna(A['bv'])
        for src, p in ((bvp, 'c'), (A['jump'], 'j')):
            A[p + '_d'] = src
            for k, n in (('w', 5), ('m', 22)):
                A[p + '_' + k] = src.groupby(A['ticker']).transform(lambda v: v.rolling(n, min_periods=1).mean())
    else:
        A = HC.add_cj(A)
    if plant:
        A['c_d'] = A.groupby('ticker', sort=False)['c_d'].shift(-1).fillna(A['c_d'])
    for h in HS:
        A[f'_tendy{h}'] = A[f'_tend{h}'].dt.year.values
    A['year'] = A.date.dt.year
    return A


def fit_steps(A, h, emb, steps=STEPS):
    """OLS в логах по шагам (процедура gibrid_har.har_forecasts) -> строки val/test (сырой прогноз), коэффициенты."""
    t = f'rv_target_h{h}'
    X = GH.clean(np.log(A[CJ].values.clip(1e-10)))
    y = A[t].values
    yr, tendy = A.year.values, A[f'_tendy{h}'].values
    ok = ~np.isnan(y)
    rows, coefs = [], {}
    for k in steps:
        vy, ty = 2015 + k, 2016 + k
        m = (yr < vy) & ok
        if emb:
            m &= tendy < vy
        lr = LinearRegression().fit(X[m], np.log(y[m] + 1e-10))
        coefs[k] = np.r_[lr.intercept_, lr.coef_]
        p = np.clip(np.exp(lr.predict(X)), 1e-10, None)
        for part, yy in (('val', vy), ('test', ty)):
            sel = (yr == yy) & (ok if part == 'val' else True)
            rows.append(pd.DataFrame({'date': A.date.values[sel], 'ticker': A.ticker.values[sel], 'year': yy, 'step': k, 'part': part,
                                      'rv_actual': y[sel], 'rv_pred_raw': p[sel], 'tendy': tendy[sel], 'n_train': int(m.sum())}))
    return pd.concat(rows, ignore_index=True), coefs


def s2_pool(V, ks, before, emb):
    v = V[V.step.isin(ks) & (V.rv_actual > 0)]
    if emb:
        v = v[v.tendy < before]
    return float(np.var(np.log(v.rv_actual.values + 1e-10) - np.log(v.rv_pred_raw.values))) if len(v) else 0.0


def bias(R, emb):
    """Б0.1 sm: val шага k — s² шагов k−3..k−1 (шаг 1: F = 1), тест шага k — s² шагов k−2..k (как gibrid_bias.correct, код 'sm')."""
    V = R[R.part == 'val']
    Fv = {k: (1.0 if k == 1 else float(np.exp(s2_pool(V, range(max(1, k - 3), k), 2015 + k, emb) / 2))) for k in STEPS}
    Ft = {k: float(np.exp(s2_pool(V, range(max(1, k - 2), k + 1), 2016 + k, emb) / 2)) for k in STEPS}
    R = R.copy()
    R['F'] = np.where(R.part == 'val', R.step.map(Fv), R.step.map(Ft))
    R['rv_pred'] = R.rv_pred_raw * R.F
    return R, Fv, Ft


def run_version(v, h, perturb=None, plant=False, A=None):
    emb, d0 = VERS[v]
    A = load_cj(d0, perturb, plant) if A is None else A
    R, co = fit_steps(A, h, emb)
    R, Fv, Ft = bias(R, emb)
    return R, co, Fv, Ft


def harcj():
    OUT_A.mkdir(parents=True, exist_ok=True); OUT_V.mkdir(parents=True, exist_ok=True); TL.mkdir(parents=True, exist_ok=True)
    mats = {d0: load_cj(d0) for d0 in (False, True)}
    chk, qrows, shas = [], [], {}
    for h in HS:
        # сверка драйвера: версия '' = gibrid_har.har_forecasts (шаги 1–9)
        hv, ht, _ = GH.har_forecasts(h, mats[False], CJ)
        R0, _ = fit_steps(mats[False], h, False)
        for part, ref in (('val', hv), ('test', ht)):
            mine = R0[(R0.part == part) & (R0.step <= 9)].dropna(subset=['rv_actual'])
            j = mine.merge(ref.assign(date=pd.to_datetime(ref.date)), on=['date', 'ticker', 'step'], validate='1:1')
            chk.append(dict(h=h, part=part, rows_mine=len(mine), rows_ref=len(ref), matched=len(j),
                            max_rel_diff=float(np.max(np.abs(j.rv_pred_raw / j.rv_pred - 1)))))
        for v in VERS:
            R, co, Fv, Ft = run_version(v, h, A=mats[VERS[v][1]])
            yq, mq = qlike_years(R[R.part == 'val'])
            qrows += [dict(model='HAR-CJ', version=v or 'kod', h=h, year=y, qlike=q) for y, q in yq.items()]
            qrows.append(dict(model='HAR-CJ', version=v or 'kod', h=h, year='mean', qlike=mq))
            if v == OSN:
                keep = ['date', 'ticker', 'year', 'step', 'rv_actual', 'rv_pred_raw', 'F', 'rv_pred', 'n_train']
                R = R.sort_values(['step', 'date', 'ticker'])
                R[(R.part == 'val') & (R.step <= DEV_HI)][keep].to_parquet(OUT_V / f'REF_HARCJ_d0_h{h}_val.parquet', index=False)
                for nm, d in ((f'REF_HARCJ_d0_h{h}_valconfirm.parquet', R[(R.part == 'val') & (R.step > DEV_HI)]),
                              (f'REF_HARCJ_d0_h{h}_test.parquet', R[R.part == 'test'])):
                    d[keep].to_parquet(TL / nm, index=False)
                    shas[nm] = sha(TL / nm)
                pd.DataFrame([dict(h=h, step=k, coef=c, value=x) for k, cc in co.items() for c, x in zip(['const'] + CJ, cc)]).to_csv(
                    OUT_V / f'REF_HARCJ_d0_h{h}_coefs.csv', index=False)
                pd.DataFrame([dict(h=h, step=k, F_val=Fv[k]) for k in STEPS if k <= DEV_HI]).to_csv(OUT_V / f'REF_HARCJ_d0_h{h}_F.csv', index=False)
    C = pd.DataFrame(chk)
    C.to_csv(OUT_V / 'check_driver.csv', index=False)
    print(C.to_string())
    assert (C.matched == C.rows_ref).all() and (C.max_rel_diff < 1e-9).all(), 'драйвер не повторяет gibrid_har'
    pd.DataFrame(qrows).to_csv(OUT_V / 'qlike_dev_harcj.csv', index=False)
    (TL / 'SHA256.txt').write_text(''.join(f'{v}  {k}\n' for k, v in sorted(shas.items())))
    print('закрытые ряды (метрик нет):'); [print(' ', v, k) for k, v in sorted(shas.items())]
    Q = pd.DataFrame(qrows)
    print(Q[Q.year == 'mean'].pivot(index='version', columns='h', values='qlike').to_string())


def leak():
    OUT_V.mkdir(parents=True, exist_ok=True)
    rows = []
    cuts = (('2019-06-28', 'mid'), ('2018-12-31', 'gran'), ('2021-06-30', 'mid'))
    for h in HS:
        base = {v: run_version(v, h) for v in (OSN, '_d0')}
        for T, kind in cuts:
            Tt, Y = pd.Timestamp(T), pd.Timestamp(T).year
            for v, pl in ((OSN, False), (OSN, True), ('_d0', False)):
                R0, c0, F0, _ = base[v] if not pl else run_version(v, h, plant=True)
                R1, c1, F1, _ = run_version(v, h, perturb=T, plant=pl)
                if kind == 'mid':
                    ks = [k for k in STEPS if 2015 + k <= Y]
                    a = R0[(R0.part == 'val') & R0.step.isin(ks)].reset_index(drop=True)
                    b = R1[(R1.part == 'val') & R1.step.isin(ks)].reset_index(drop=True)
                    le, gt = (a.date <= Tt).values, (a.date > Tt).values
                    assert len(a) == len(b) and (a.date.values == b.date.values).all()
                    mism = int((a.rv_pred.values[le] != b.rv_pred.values[le]).sum())
                    moved = int((a.rv_pred.values[gt] != b.rv_pred.values[gt]).sum())
                else:
                    ks = [Y + 1 - 2015]
                    mism, moved = -1, -1
                coef_same = all(np.array_equal(c0[k], c1[k]) for k in ks)
                F_same = all(F0[k] == F1[k] for k in ks)
                rows.append(dict(h=h, T=T, kind=kind, version=v, plant=pl, steps=str(ks), coef_identical=coef_same, F_identical=F_same,
                                 mismatch_le_T=mism, changed_after_T=moved))
    L = pd.DataFrame(rows)
    L.to_csv(OUT_V / 'leak_harcj_d0.csv', index=False)
    print(L.to_string())
    osn, pl, ne = L[(L.version == OSN) & ~L.plant], L[(L.version == OSN) & L.plant], L[(L.version == '_d0') & (L.kind == 'gran')]
    mid = osn[osn.kind == 'mid']
    ok = bool(osn.coef_identical.all() and osn.F_identical.all() and (mid.mismatch_le_T == 0).all() and (mid.changed_after_T > 0).all())
    caught = bool((pl[pl.kind == 'mid'].mismatch_le_T > 0).all())
    print(f'«будущее не влияет» ({OSN}): {"PASS" if ok else "FAIL"}; подлог BV дня d+1 пойман: {"PASS" if caught else "FAIL"}; '
          f'без эмбарго (_d0) коэффициенты шага на границе года меняются: {bool((~ne.coef_identical).all())}')
    assert ok and caught


# ================================================================ аудит
def base_rv():
    a = pd.read_parquet(FEAT, columns=['date', 'ticker', 'rv', 'close'])
    a['date'] = pd.to_datetime(a.date)
    a = a.sort_values(['ticker', 'date']).reset_index(drop=True)
    g = a.groupby('ticker')
    a['rv_l1'] = g.rv.shift(1)
    a['r2'] = (100 * np.log(a.close).groupby(a.ticker).diff()) ** 2
    a['r2_l1'] = a.groupby('ticker').r2.shift(1)
    return a[['date', 'ticker', 'rv', 'rv_l1', 'r2', 'r2_l1']]


def series(h):
    """(модель, версия, ряд date/ticker/step/rv_actual/rv_pred — основной вариант модели; QLIKE: {год: q}, среднее)."""
    out = []
    for m in ('xgb', 'lgbm'):
        d = pd.read_parquet(REG / 'ref_v0_base' / f'REF_V0B_{m}_h{h}_val.parquet')
        d = d[(d.step >= 1) & (d.step <= DEV_HI)].copy()
        d['rv_pred'] = d.rv_pred * d.F_oos
        qs = [qlike_years(d[d.seed == s]) for s in (42, 43, 44)]
        yq = {y: float(np.mean([q[0][y] for q in qs])) for y in qs[0][0]}
        e = d.assign(lp=np.log(d.rv_pred)).groupby(['date', 'ticker', 'step'], as_index=False).agg(rv_actual=('rv_actual', 'first'), lp=('lp', 'mean'))
        out.append((f'V0 {m}', 'kod', e.assign(rv_pred=np.exp(e.lp)), yq, float(np.mean(list(yq.values())))))
    for m in ('GRU', 'LSTM', 'GARCH', 'GJR'):
        d = pd.read_parquet(REG / 'ref_nn_garch' / f'REF_{m}_h{h}_val.parquet')
        out.append((m, 'kod', d, *qlike_years(d)))
    for m, nm in (('heavy', 'HEAVY'), ('rgarch', 'Realized GARCH')):
        d = pd.read_parquet(REG / f'GARCH_sv{m}_h{h}_val.parquet')
        out.append((nm + ' (без Б0.1)', 'kod', d, *qlike_years(d)))
    d = pd.read_parquet(REG / f'D16_10_harj_h{h}_val.parquet')
    out.append(('HAR-J (контроль, без Б0.1)', 'kod', d, *qlike_years(d)))
    for v in ('', OSN):
        R = run_version(v, h)[0]
        d = R[R.part == 'val']
        out.append(('HAR-CJ', v or 'kod', d, *qlike_years(d)))
    return out


def audit():
    OUT_A.mkdir(parents=True, exist_ok=True)
    B = base_rv()
    arows, qrows = [], []
    for h in HS:
        for name, ver, d, yq, mq in series(h):
            d = d[(d.step >= DEV_LO) & (d.step <= DEV_HI)][['date', 'ticker', 'rv_pred']].copy()
            d['date'] = pd.to_datetime(d.date)
            j = d.merge(B, on=['date', 'ticker'], how='left', validate='1:1')
            assert pd.to_datetime(j.date).dt.year.between(2017, 2021).all()
            j = j[(j.rv > 0) & (j.rv_l1 > 0) & (j.rv_pred > 0)]
            lf = np.log(j.rv_pred.values)
            sp = lambda c: float(pd.Series(lf).corr(pd.Series(j[c].values), method='spearman'))
            spt = lambda c: float(np.mean([pd.Series(np.log(g.rv_pred.values)).corr(pd.Series(g[c].values), method='spearman') for _, g in j.groupby('ticker')]))
            X = np.column_stack([np.log(j.rv.values), np.log(j.rv_l1.values)])
            lr = LinearRegression().fit(X, lf)
            res = lf - lr.predict(X)
            Xc = np.column_stack([np.ones(len(X)), X])
            se = np.sqrt(np.diag(np.var(res, ddof=3) * np.linalg.inv(Xc.T @ Xc)))[1:]
            arows.append(dict(model=name, version=ver, h=h, n=len(j), rho_rv_d=sp('rv'), rho_rv_d1=sp('rv_l1'),
                              rho_rv_d_tk=spt('rv'), rho_rv_d1_tk=spt('rv_l1'), rho_r2_d=sp('r2'), rho_r2_d1=sp('r2_l1'),
                              b_rv_d=lr.coef_[0], b_rv_d1=lr.coef_[1], t_rv_d=lr.coef_[0] / se[0], t_rv_d1=lr.coef_[1] / se[1],
                              r2=float(1 - res.var() / lf.var())))
            qrows += [dict(model=name, version=ver, h=h, year=y, qlike=q) for y, q in yq.items()]
            qrows.append(dict(model=name, version=ver, h=h, year='mean', qlike=mq))
    A_ = pd.DataFrame(arows); Q = pd.DataFrame(qrows)
    A_.to_csv(OUT_A / 'audit_lag.csv', index=False); Q.to_csv(OUT_A / 'qlike_dev.csv', index=False)
    pd.set_option('display.width', 250)
    print(A_.round(3).to_string())
    print(Q[Q.year == 'mean'].pivot_table(index=['model', 'version'], columns='h', values='qlike').round(4).to_string())


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['audit', 'harcj', 'leak'])
    a = ap.parse_args()
    {'audit': audit, 'harcj': harcj, 'leak': leak}[a.mode]()
