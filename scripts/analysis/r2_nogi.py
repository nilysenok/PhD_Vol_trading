#!/usr/bin/env python3
"""R2, агент Э (протокол v2, docs/zadaniya/2026-10-26_vtoroy_raund_gibrida.md, строка «Э»): ноги уровня 1 без бустинга, h ∈ {1, 5, 22},
val шагов 1–6 (val-год vy = 2015 + k, обучение — годы < vy), seed нет (файлы _sdet). Код первого раунда только импортируется.
Семейства и ряды (имя ряда -> файл R2/cache/E/<ряд>_h{h}_sdet_val.parquet; rv_pred — сырой, без s²):
  HARJ                     HAR-J на features_pit_v3 (как gibrid_har.har_forecasts)
  HARX_<точка>             HAR-X = HAR-J + harx.default_terms(h, True), Фриш–Во + эластичная сеть (harx.fit_en); 46 точек:
                           HARX_ols, HARX_inner (внутренняя валидация шага, как harx.py сейчас), HARX_l<λ>_a<α> (λ ∈ harx.ALPHAS, α ∈ harx.L1S);
                           точка фиксирована на все шаги
  HARXsub_plus_<гр>        HAR-J + одна группа слагаемых, OLS
  HARXsub_minus_<гр>       HAR-X без одной группы, OLS (minus_D17 = HARX0); группы: RVI, rynok, LHAR, PC1, sig1 (h=5/22), D17
  HARXreg                  режимный HAR-X (Д24.4 RX: OLS, взаимодействия всех регрессоров с 1{RVI > медианы за 252 прошлых дня}, эмбарго)
  HEAVY                    HEAVY (Д24.6, d24_6.fit_forecast: QMLE на годах < vy, фильтр)
ln-прогнозы для ролей (Б1/Б2): R2/cache/E/lnall/<ряд>_h{h}_lnall.parquet — date, ticker, year, ln_s1..ln_s6: ln-прогноз модели шага k
  на всех строках с year <= vy_k (обучение — внутри выборки, как HAR-J в Д8.12 и HXP_lnall в Д22); строки лет > vy_k — NaN. Только для рядов
  ролей: HARJ, 8 точек HAR-X (ols, inner, λ ∈ {1e-3, 1e-2, 0,1} × α ∈ {0,5; 1}), все подмножества, HARXreg, HEAVY.
V0-ячейки: R2/cache/E/V0_F_h{h}.csv — множитель Б0.1 F(ряд, s², шаг) (val шага t — s² шага t−1, шаг 1 — F = 1; как harx.add_bias),
  s² ∈ {bez, ord, rob, sm, smrob}; R2/cache/E/V0_q_h{h}.csv — QLIKE по годам 2017–2021 (шаги 2–6) для каждого ряда × s².
Годы >= 2022 не считаются и не пишутся: шаги 7–10 не строятся. Тестовой сборки нет.
Режимы:
  zamer  — шаг 0: одна точка × один шаг по семейству, экстраполяция -> R2/cache/E/zamer_E.csv
  det    — все ряды, lnall, V0-таблицы, сверка с первым раундом (|Δ| <= 1e-7, иначе стоп) -> sverka_E.csv, INDEX_E.csv
  leak   — детектор «будущее не влияет»: всё после среза (входы и цели, чей горизонт заходит за срез) × U(0,5; 2); val-прогнозы и lnall
           моделей с vy <= года среза на строках <= среза совпадают точно (Δ = 0); подложенная утечка (rv_d дня t+1, для HEAVY — RM дня t+1)
           поймана -> leak_E.csv
python3 scripts/analysis/r2_nogi.py zamer | det [--hs 1 5 22] [--workers 2] | leak"""
import os
for _k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_k] = '1'
import sys, time, hashlib, argparse, tempfile, subprocess, warnings, multiprocessing as mp
from datetime import datetime
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import LinearRegression

warnings.filterwarnings('ignore')
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import harx as HX
import gibrid_har as GH

DATA = Path('/Users/nilysenok/Desktop/Старое/PhD_Vol_trading/data')
REG = DATA / 'regen'
CACHE = REG / 'R2' / 'cache' / 'E'
LNDIR = CACHE / 'lnall'
INDEX = REG / 'R2' / 'cache' / 'INDEX_E.csv'
YAML = HERE.parents[1] / 'models' / 'designs' / 'r2' / 'r2_E_nogi.yaml'
FEAT = REG / 'features_pit_v3.parquet'
COLS = ['date', 'ticker', 'year', 'step', 'rv_actual', 'rv_pred']
KEY = ['date', 'ticker', 'step']
HS = (1, 5, 22)
STEPS = range(1, 7)        # только разработка: val 2016–2021
TOL = 1e-7
S2 = ('bez', 'ord', 'rob', 'sm', 'smrob')
GRID = [('ols', 0.0, 1.0), ('inner', None, None)] + [(f'l{al:g}_a{l1:g}', al, l1) for l1 in HX.L1S for al in HX.ALPHAS]
ROLE_HARX = {'ols', 'inner', 'l0.001_a0.5', 'l0.001_a1', 'l0.01_a0.5', 'l0.01_a1', 'l0.1_a0.5', 'l0.1_a1'}
assert len(GRID) == 46 and ROLE_HARX <= {g[0] for g in GRID}


def groups(h):
    g = {'RVI': ['ln_rvi2'], 'rynok': ['ln_rvm_imoex', 'ln_rvm_mkt'], 'LHAR': ['lhar_1', 'lhar_5', 'lhar_22'], 'PC1': ['pc1']}
    if h != 1:
        g['sig1'] = ['ln_sig1']
    g['D17'] = list(HX.D17_TERMS[h])
    full = HX.default_terms(h, True)
    assert sorted(sum(g.values(), [])) == sorted(full), (h, g, full)
    return g


def subsets(h):
    full, g = HX.default_terms(h, True), groups(h)
    out = [(f'plus_{k}', v) for k, v in g.items()] + [(f'minus_{k}', [t for t in full if t not in v]) for k, v in g.items()]
    assert len(out) <= 12
    return out


# ================================================================ загрузка
def load(feat=FEAT, plant=False):
    """Матрица как harx.load (gibrid_har.load_pit). plant=True — подложенная утечка: rv_d дня t+1 (HAR-регрессор)."""
    HX._SIG.clear()
    A, harj = HX.load(feat)
    if plant:
        A['rv_d'] = A.groupby('ticker')['rv_d'].shift(-1).fillna(A['rv_d'])
    return A, harj


def _val(A, h, k, ln):
    t = f'rv_target_h{h}'; vy = 2015 + k
    sel = ((A.year == vy) & A[t].notna()).values
    return pd.DataFrame({'date': pd.to_datetime(A.date.values[sel]), 'ticker': A.ticker.values[sel].astype(str), 'year': vy, 'step': k,
                         'rv_actual': A[t].values[sel], 'rv_pred': np.clip(np.exp(ln[sel]), 1e-10, None)})


def _lnall(A, lns):
    """lns: {k: ln на всех строках A} -> date, ticker, year, ln_s{k} (NaN при year > vy_k)."""
    d = pd.DataFrame({'date': pd.to_datetime(A.date.values), 'ticker': A.ticker.values.astype(str), 'year': A.year.values})
    for k, ln in sorted(lns.items()):
        d[f'ln_s{k}'] = np.where(A.year.values <= 2015 + k, ln, np.nan)
    return d[d.year <= 2015 + max(lns)].sort_values(['date', 'ticker']).reset_index(drop=True)


def _fin(V):
    return pd.concat(V).sort_values(['step', 'date', 'ticker']).reset_index(drop=True)[COLS]


# ================================================================ семейства (каждое -> {ряд: (V, LN | None)})
def fam_harj(A, harj, h, steps=STEPS):
    t = f'rv_target_h{h}'; ok = A[t].notna().values
    H = GH.clean(np.log(A[harj].values.clip(1e-10)))
    V, L = [], {}
    for k in steps:
        tp = (A.year.values < 2015 + k) & ok
        ln = LinearRegression().fit(H[tp], np.log(A[t].values[tp] + 1e-10)).predict(H)
        V.append(_val(A, h, k, ln)); L[k] = ln
    return {'HARJ': (_fin(V), _lnall(A, L))}


def _harx_points(A, harj, h, terms, X, points, steps, prefix, role):
    """Точки (имя, λ, α) на наборе слагаемых terms; 'inner' — harx.forecast_step с внутренней валидацией."""
    t = f'rv_target_h{h}'; ok = A[t].notna().values
    H = GH.clean(np.log(A[harj].values.clip(1e-10)))
    Xt = X[terms]
    acc = {p[0]: ([], {}) for p in points}
    for k in steps:
        tp = (A.year.values < 2015 + k) & ok
        Z = Xt.fillna(Xt.loc[tp].mean()).fillna(0.0).values.astype('float64')
        y = np.log(A[t].values[tp] + 1e-10)
        for nm, al, l1 in points:
            if nm == 'inner':
                ln = HX.forecast_step(A, h, k, harj, terms, X, all_rows=True)[2].values
            else:
                ln = HX.fit_en(H[tp], Z[tp], y, al, l1)[0](H, Z)
            acc[nm][0].append(_val(A, h, k, ln)); acc[nm][1][k] = ln
    return {f'{prefix}{nm}': (_fin(V), _lnall(A, L) if role(nm) else None) for nm, (V, L) in acc.items()}


def fam_harx(A, harj, h, steps=STEPS, points=GRID):
    terms = HX.default_terms(h, True)
    X = HX.build_terms(A, terms)
    return _harx_points(A, harj, h, terms, X, points, steps, 'HARX_', lambda nm: nm in ROLE_HARX)


def fam_sub(A, harj, h, steps=STEPS, only=None):
    X = HX.build_terms(A, HX.default_terms(h, True))
    out = {}
    for nm, terms in subsets(h):
        if only is None or nm in only:
            out.update(_harx_points(A, harj, h, terms, X, [('ols', 0.0, 1.0)], steps, f'HARXsub_{nm}', lambda _: True))
    return {k.replace('ols', ''): v for k, v in out.items()}


def fam_reg(A, harj, h, steps=STEPS):
    """Д24.4 RX (d24_4.design / fit_pred_factory, эмбарго d24b_lib.train_mask) на той же матрице A."""
    import d24_4 as R4, d24b_lib as L4
    B = A.sort_values(['ticker', 'date']).reset_index(drop=True)
    g = B.groupby('ticker', sort=False)
    B = pd.concat([B, pd.DataFrame({f'_fd{j}': g['date'].shift(-j) for j in (1, 5, 22)}, index=B.index)], axis=1)
    HX._SIG.clear()
    X, Z, I, _ = R4.design(B, harj, h, 'RX')
    fp = R4.fit_pred_factory(B, h, X, Z, I)
    allr = np.ones(len(B), bool)
    V, L = [], {}
    for k in steps:
        vy = 2015 + k
        ln = fp(k, vy, L4.train_mask(B, h, vy))[0](allr)
        V.append(_val(B, h, k, ln)); L[k] = ln
    return {'HARXreg': (_fin(V), _lnall(B, L))}


def fam_heavy(A, harj, h, steps=STEPS, plant=False):
    """HEAVY как d24_6.main (вход — close и rv той же матрицы); plant=True — RM дня t+1."""
    import d24_6 as G
    a = A[['date', 'ticker', 'close', 'rv', f'rv_target_h{h}']].copy()
    a['date'] = pd.to_datetime(a.date); a['year'] = a.date.dt.year
    V, LN = [], []
    for tk in sorted(a.ticker.unique()):
        gg = G.prep(a, tk)
        if plant:
            gg['xx'] = gg['xx'].shift(-1).fillna(gg['xx'].iloc[-1])
        lns = {}
        for k in steps:
            vy = 2015 + k
            F, _ = G.fit_forecast(gg.rr.values, gg.xx.values, int((gg.year < vy).sum()))['heavy']
            ln = np.log(np.clip(F[h] / 1e4, 1e-10, None))
            sel = ((gg.year == vy) & gg[f'rv_target_h{h}'].notna()).values
            V.append(pd.DataFrame({'date': gg.date.values[sel], 'ticker': tk, 'year': vy, 'step': k,
                                   'rv_actual': gg[f'rv_target_h{h}'].values[sel], 'rv_pred': np.exp(ln[sel])}))
            lns[k] = ln
        LN.append(_lnall(gg, lns))
    LN = pd.concat(LN).sort_values(['date', 'ticker']).reset_index(drop=True)
    return {'HEAVY': (_fin(V), LN)}


FAMS = {'HARJ': fam_harj, 'HEAVY': fam_heavy, 'HARXreg': fam_reg, 'HARXsub': fam_sub, 'HARX': fam_harx}
SEMEYSTVO = lambda nm: 'HARXsub' if nm.startswith('HARXsub') else ('HARX' if nm.startswith('HARX_') else nm)


def build_h(h, feat=FEAT, plant=False, fams=tuple(FAMS), light=False):
    """Все ряды горизонта h. light=True (детектор) — по три точки HAR-X и два подмножества."""
    A, harj = load(feat)                                      # HEAVY: подлог внутри fam_heavy (RM дня t+1)
    Ap = load(feat, plant=True)[0] if plant else A            # остальные: rv_d дня t+1
    out = {}
    for f in fams:
        kw = {}
        if light and f == 'HARX':
            kw['points'] = [g for g in GRID if g[0] in ('ols', 'inner', 'l0.01_a0.5')]
        if light and f == 'HARXsub':
            kw['only'] = ('plus_RVI', 'minus_D17')
        if f == 'HEAVY':
            out.update(fam_heavy(A, harj, h, plant=plant))
        else:
            out.update(FAMS[f](Ap, harj, h, **kw))
    return out


# ================================================================ V0: поправка Б0.1 и QLIKE 2017–2021
def qlike(y, p):
    r = y / np.clip(p, 1e-12, None)
    return float(np.mean(r - np.log(r) - 1))


def v0_tables(series, h):
    Fr, Qr = [], []
    for nm, (V, _) in series.items():
        v = V[V.rv_actual > 1e-12]
        E = {st: np.log(g.rv_actual.values + 1e-10) - np.log(g.rv_pred.values) for st, g in v.groupby('step')}
        for code in S2:
            F = {st: (1.0 if (code == 'bez' or st == 1) else float(np.exp(HX.s2_est(code, E, st - 1) / 2))) for st in STEPS}
            Fr += [dict(ryad=nm, semeystvo=SEMEYSTVO(nm), h=h, s2=code, step=st, F=f) for st, f in F.items()]
            d = v[v.step.between(2, 6)]
            for st, g in d.groupby('step'):
                Qr.append(dict(ryad=nm, semeystvo=SEMEYSTVO(nm), h=h, s2=code, year=2015 + int(st), q=qlike(g.rv_actual.values, g.rv_pred.values * F[st])))
    return pd.DataFrame(Fr), pd.DataFrame(Qr)


# ================================================================ сверка с первым раундом (шаги 1–6)
def r1_refs(h):
    r = {f'HARX_inner': REG / f'HARX_h{h}_val.parquet', 'HARXreg': REG / 'D24b' / 'd24_4' / f'RX_h{h}_val.parquet',
         'HEAVY': REG / f'GARCH_svheavy_h{h}_val.parquet'}
    return {k: v for k, v in r.items() if v.exists()}


def _cmp(V, r, nm, h, src):
    r = r[r.step.between(1, 6)].copy(); r['date'] = pd.to_datetime(r.date)
    m = V.merge(r[KEY + ['rv_actual', 'rv_pred']], on=KEY, how='outer', suffixes=('', '_r1'), indicator=True)
    b = m._merge == 'both'
    d = float(np.abs(m.rv_pred[b] - m.rv_pred_r1[b]).max())
    da = float(np.nanmax(np.abs(m.rv_actual[b] - m.rv_actual_r1[b])))
    nout = int((~b).sum())
    return dict(ryad=nm, h=h, r1=src, n=int(b.sum()), n_nesovp_strok=nout, max_abs=d, max_abs_rv_actual=da,
                status='OK' if (nout == 0 and d <= TOL and da <= TOL) else 'РАСХОЖДЕНИЕ')


def sverka(series, F, h):
    rows = []
    for nm, f in r1_refs(h).items():
        rows.append(_cmp(series[nm][0], pd.read_parquet(f), nm, h, str(f.relative_to(REG))))
    hv, _, _ = GH.har_forecasts(h, *HX.load(FEAT))  # HAR-J: код первого раунда (gibrid_har)
    rows.append(_cmp(series['HARJ'][0], hv, 'HARJ', h, 'gibrid_har.har_forecasts'))
    # s² (Б0.1): rv_pred_{code} ряда HARX_h{h}_val первого раунда (harx.add_bias) против HARX_inner × F
    r = pd.read_parquet(REG / f'HARX_h{h}_val.parquet'); r = r[r.step.between(1, 6)]; r['date'] = pd.to_datetime(r.date)
    V = series['HARX_inner'][0]
    for code in S2[1:]:
        f = F[(F.ryad == 'HARX_inner') & (F.s2 == code)].set_index('step').F
        rr = r[KEY + ['rv_actual', f'rv_pred_{code}']].rename(columns={f'rv_pred_{code}': 'rv_pred'})
        rows.append(_cmp(V.assign(rv_pred=V.rv_pred * V.step.map(f)), rr, f'HARX_inner×F_{code}', h, f'HARX_h{h}_val.rv_pred_{code}'))
    return rows


# ================================================================ запись, индекс
def sha256(p):
    hh = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            hh.update(b)
    return hh.hexdigest()


def _git():
    try:
        return subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], cwd=HERE, capture_output=True, text=True).stdout.strip()
    except Exception:
        return ''


def write(series, h):
    CACHE.mkdir(parents=True, exist_ok=True); LNDIR.mkdir(parents=True, exist_ok=True)
    code, ys, now, git = sha256(Path(__file__))[:16], (sha256(YAML)[:16] if YAML.exists() else ''), datetime.now().strftime('%Y-%m-%d %H:%M:%S'), _git()
    rows = []
    for nm, (V, LN) in series.items():
        assert V.step.max() <= 6 and V.year.max() <= 2021
        for fp, d, kind in ((CACHE / f'{nm}_h{h}_sdet_val.parquet', V, 'val'), (LNDIR / f'{nm}_h{h}_lnall.parquet', LN, 'lnall')):
            if d is None:
                continue
            assert d.year.max() <= 2021
            d.to_parquet(fp, index=False)
            rows.append(dict(file=str(fp.relative_to(REG)), ryad=nm, semeystvo=SEMEYSTVO(nm), h=h, seed='det', vid=kind, shagi='1-6', stroki=len(d),
                             sha256=sha256(fp), sha_koda=code, git=git, yaml=ys, vremya=now))
    return rows


def write_index(rows):
    new = pd.DataFrame(rows)
    old = pd.read_csv(INDEX) if INDEX.exists() else new.iloc[:0]
    pd.concat([old[~old.file.isin(new.file)], new], ignore_index=True).sort_values('file').to_csv(INDEX, index=False)


def _job(h):
    t0 = time.time()
    S = build_h(h)
    F, Q = v0_tables(S, h)
    sv = sverka(S, F, h)
    return h, S, F, Q, sv, time.time() - t0


def det(hs=HS, workers=2):
    t0 = time.time()
    with mp.get_context('fork').Pool(min(workers, len(hs))) as pool:
        res = pool.map(_job, hs, chunksize=1)
    sv_all, idx = [], []
    for h, S, F, Q, sv, dt in res:
        sv_all += sv
        bad = [r for r in sv if r['status'] != 'OK']
        for r in sv:
            print(f"h={h} {r['ryad']}: {r['status']} max|Δ|={r['max_abs']:.1e} (строк {r['n']}, вне пары {r['n_nesovp_strok']})", flush=True)
        CACHE.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(sv_all).to_csv(CACHE / 'sverka_E.csv', index=False)
        if bad:
            raise SystemExit(f'стоп: h={h} расхождение с первым раундом: {[r["ryad"] for r in bad]}')
        idx += write(S, h)
        F.to_csv(CACHE / f'V0_F_h{h}.csv', index=False); Q.to_csv(CACHE / f'V0_q_h{h}.csv', index=False)
        hj = Q[(Q.ryad == 'HARJ') & (Q.s2 == 'sm')].set_index('year').q
        print(f'h={h}: рядов {len(S)}, {dt:.0f} с; HAR-J sm 2017–2020 {hj.loc[2017:2020].mean():.4f}, 2021 {hj.loc[2021]:.4f}', flush=True)
    write_index(idx)
    print(f'det готов: {len(idx)} файлов, {time.time() - t0:.0f} с')


# ================================================================ детектор
def _perturb(feat, cut, rng, out):
    B = pd.read_parquet(feat)
    dt = pd.to_datetime(B['date'])
    fut = (dt > cut).values
    num = [c for c in B.columns if c != 'year' and pd.api.types.is_float_dtype(B[c])]
    B.loc[fut, num] = B.loc[fut, num].values * rng.uniform(0.5, 2.0, size=(fut.sum(), len(num)))
    past = B.assign(_dt=dt).sort_values(['ticker', '_dt']).loc[lambda d: d._dt <= cut]
    for c in [c for c in num if 'target' in c]:  # цели, чей горизонт (n дней) заходит за срез
        ix = past.groupby('ticker').tail(int(c.split('_h')[-1])).index
        B.loc[ix, c] = B.loc[ix, c].values * rng.uniform(0.5, 2.0, size=len(ix))
    B.to_parquet(out, index=False)


def _dd(a, b, cut):
    """max |Δ| на строках <= cut: val-прогнозы и lnall моделей с vy <= год среза."""
    past, fut = 0.0, 0.0
    for nm in a:
        m = a[nm][0].merge(b[nm][0], on=KEY, suffixes=('', '_p'))
        d = np.abs(m.rv_pred - m.rv_pred_p).values; p = (m.date <= cut).values
        past = max(past, float(d[p].max()) if p.any() else 0.0); fut = max(fut, float(d[~p].max()) if (~p).any() else 0.0)
        if a[nm][1] is not None:
            la, lb = a[nm][1], b[nm][1]
            m = la.merge(lb, on=['date', 'ticker'], suffixes=('', '_p')); m = m[m.date <= cut]
            for k in range(1, cut.year - 2015 + 1):
                past = max(past, float(np.nanmax(np.abs(m[f'ln_s{k}'] - m[f'ln_s{k}_p']).values)) if len(m) else 0.0)
    return past, fut


def leak(hs=HS, cuts=('2018-06-29', '2020-03-02', '2021-06-30')):
    rng = np.random.default_rng(20261026); rows = []
    with tempfile.TemporaryDirectory() as td:
        tmps = {}
        for c in cuts:
            tmps[c] = Path(td) / f'{c}.parquet'; _perturb(FEAT, pd.Timestamp(c), rng, tmps[c])
        for h in hs:
            b0, b1 = build_h(h, light=True), build_h(h, plant=True, light=True)
            for c in cuts:
                cut = pd.Timestamp(c)
                p0, p1 = build_h(h, tmps[c], light=True), build_h(h, tmps[c], plant=True, light=True)
                for fam in FAMS:
                    sel = lambda D: {k: v for k, v in D.items() if SEMEYSTVO(k) == fam}
                    past0, fut0 = _dd(sel(b0), sel(p0), cut); past1, _ = _dd(sel(b1), sel(p1), cut)
                    rows.append(dict(h=h, semeystvo=fam, srez=c, max_d_do_sreza=past0, max_d_posle_sreza=fut0, podlog_max_d_do_sreza=past1,
                                     ok=bool(past0 == 0.0 and fut0 > 0 and past1 > 0)))
                    print(rows[-1], flush=True)
    R = pd.DataFrame(rows); CACHE.mkdir(parents=True, exist_ok=True); R.to_csv(CACHE / 'leak_E.csv', index=False)
    print('ВСЕ OK' if R.ok.all() else 'ЕСТЬ ОШИБКИ'); return bool(R.ok.all())


def zamer():
    A, harj = load()
    rows = []
    for f, n_units, note in (('HARJ', 3, 'h'), ('HEAVY', 3, 'h'), ('HARXreg', 3, 'h'), ('HARXsub', 3 * 12, 'h × подмножество'), ('HARX', 3 * 46, 'h × точка')):
        t = time.time()
        kw = {'points': [GRID[2]]} if f == 'HARX' else ({'only': ('plus_RVI',)} if f == 'HARXsub' else {})
        FAMS[f](A, harj, 5, steps=range(5, 6), **kw)
        s = time.time() - t
        rows.append(dict(semeystvo=f, edinica='один шаг × ' + note, sek=round(s, 3), edinic=n_units * 6, prognoz_min=round(s * n_units * 6 / 60, 2)))
        print(rows[-1], flush=True)
    Z = pd.DataFrame(rows); CACHE.mkdir(parents=True, exist_ok=True); Z.to_csv(CACHE / 'zamer_E.csv', index=False)
    print(f'итого (1 процесс): {Z.prognoz_min.sum():.1f} мин; детектор ×~8 на лёгком наборе')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['zamer', 'det', 'leak'])
    ap.add_argument('--hs', type=int, nargs='*', default=list(HS))
    ap.add_argument('--workers', type=int, default=2)
    a = ap.parse_args()
    if a.mode == 'zamer':
        zamer()
    elif a.mode == 'det':
        det(tuple(a.hs), min(a.workers, 2))
    else:
        sys.exit(0 if leak(tuple(a.hs)) else 1)


if __name__ == '__main__':
    main()
