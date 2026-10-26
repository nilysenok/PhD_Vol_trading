#!/usr/bin/env python3
"""R2, агент Б (протокол docs/zadaniya/2026-10-26_vtoroy_raund_gibrida.md): эконометрические и структурные ноги, все h.
Детерминированные ноги (seed нет, файл _sdet): HEAVY и Realized GARCH (d24_6.py: h=1/5/22 + дальние ноги [6..22], [2..22] для Д18),
HAR-X⁺ (d22_harxp.py: цели 1, 5, 22, 2_5, 6_22), HAR-X h=5 (harx.py, умолчание tag HARX), нога RVI (а) ŷ = v·k (d25_1.rvi_leg, h=1/5/22).
Код первого раунда только импортируется (final3.py, gibrid_v17.py, harx.py, d24_6.py, d22_harxp.py, d25_1.py не меняются).
Шаги k = 1..10: val-год 2015+k, тест 2016+k. Запись:
  val шагов 1-6        -> data/regen/R2/cache/B/<нога>_h{h}_s{seed|det}_val.parquet
  val шагов 7-10       -> data/regen/test_locked/R2/cache/B/<нога>_h{h}_s{seed|det}_valconf_test.parquet   (только запись)
  тест шагов 1-10      -> data/regen/test_locked/R2/cache/B/<нога>_h{h}_s{seed|det}_test.parquet          (только запись)
  колонки date, ticker, year, step, rv_actual, rv_pred (rv_pred — без поправки s²); индекс data/regen/R2/cache/INDEX_B.csv;
  test_locked/R2/cache/B/SHA256.txt. Метрик нет нигде: скрипт не считает QLIKE ни по каким годам.
Режимы:
  zamer  — шаг 0: одна нога × один seed × один шаг по каждому семейству, экстраполяция на весь объём (data/regen/R2/cache/B/zamer_B.csv)
  det    — все детерминированные ноги + сверка с рядами первого раунда (шаги 1-6, |Δ rv_pred| <= 1e-7, иначе стоп) -> sverka_B.csv
  leak   — детектор «будущее не влияет»: всё после среза (входы и цели, чей горизонт заходит за срез) портится, прогнозы на строках <= среза
           совпадают точно (Δ = 0); подложенная утечка (вход следующего дня / k без эмбарго) поймана -> leak_B.csv
python3 scripts/analysis/r2_nogi.py zamer | det | leak"""
import os
for _k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_k] = '1'
import sys, time, hashlib, argparse, tempfile, subprocess, warnings
from datetime import datetime
from pathlib import Path
import numpy as np, pandas as pd

warnings.filterwarnings('ignore')
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
DATA = Path('/Users/nilysenok/Desktop/Старое/PhD_Vol_trading/data')
REG = DATA / 'regen'
CACHE = REG / 'R2' / 'cache' / 'B'
TLC = REG / 'test_locked' / 'R2' / 'cache' / 'B'
INDEX = REG / 'R2' / 'cache' / 'INDEX_B.csv'
YAML = HERE.parents[1] / 'models' / 'designs' / 'r2' / 'r2_B_nogi.yaml'
FEAT_V3 = REG / 'features_pit_v3.parquet'
FEAT_RRV = REG / 'features_pit_v3_rrv.parquet'
COLS = ['date', 'ticker', 'year', 'step', 'rv_actual', 'rv_pred']
KEY = ['date', 'ticker', 'step']
DEV = 6           # шаги разработки 1-6 (val 2016-2021); шаги 7-10 (val 2022-2025) и тест — только в test_locked
TOL = 1e-7        # контроль совпадения с первым раундом
STEPS = range(1, 11)


# ================================================================ ноги
def _norm(d, part):
    """Колонки COLS, типы; строки без цели убираются (кроме теста шага 10: цели в 2026 не закончились)."""
    d = d[COLS].copy()
    d['date'] = pd.to_datetime(d['date'])
    d['ticker'] = d['ticker'].astype(str)
    d['year'] = d['year'].astype('int64'); d['step'] = d['step'].astype('int64')
    d['rv_actual'] = d['rv_actual'].astype('float64'); d['rv_pred'] = d['rv_pred'].astype('float64')
    keep = d.rv_actual.notna() | ((part == 'test') & (d.step == 10))
    return d[keep].sort_values(['step', 'date', 'ticker']).reset_index(drop=True)


def nogi_garch(feat=FEAT_V3, plant=False, steps=STEPS, tickers=None):
    """HEAVY и Realized GARCH как d24_6.main (QMLE один раз на годах < vy, дальше фильтр). plant=True — подложенная утечка: RM дня t+1 на входе."""
    import d24_6 as G
    a = pd.read_parquet(feat, columns=['date', 'ticker', 'close', 'rv'] + [G.tcol(h) for h in G.HS])
    a['date'] = pd.to_datetime(a.date); a['year'] = a.date.dt.year
    acc = {}
    for tk in (sorted(a.ticker.unique()) if tickers is None else tickers):
        gg = G.prep(a, tk)
        if plant:
            gg['xx'] = gg['xx'].shift(-1).fillna(gg['xx'].iloc[-1])
        rv = gg.rv.values
        sub = lambda lo, hi: np.array([np.mean(rv[i + lo:i + hi + 1]) if i + hi < len(rv) else np.nan for i in range(len(rv))])
        tgt_leg = {'leg6_22': sub(6, 22), 'leg2_22': sub(2, 22)}
        for step in steps:
            vy, ty = 2015 + step, 2016 + step
            ntr = int((gg.year < vy).sum())
            for m, (F, _) in G.fit_forecast(gg.rr.values, gg.xx.values, ntr).items():
                for k, f in F.items():
                    t = gg[G.tcol(k)].values if k in G.HS else tgt_leg[k]
                    f = f / 1e4
                    for part, yr in (('val', vy), ('test', ty)):
                        sel = (gg.year == yr).values
                        acc.setdefault((m, k, part), []).append(pd.DataFrame({'date': gg.date.values[sel], 'ticker': tk, 'year': yr, 'step': step,
                                                                               'rv_actual': t[sel], 'rv_pred': np.clip(f[sel], 1e-10, None)}))
    out = {}
    for m, M in (('heavy', 'HEAVY'), ('rgarch', 'RG')):
        for k in list(G.HS) + ['leg6_22', 'leg2_22']:
            nm, h = ((M, k) if k in G.HS else (f'{M}_l{k[3:]}', 22))
            out[(nm, h)] = (_norm(pd.concat(acc[(m, k, 'val')]), 'val'), _norm(pd.concat(acc[(m, k, 'test')]), 'test'))
    return out


HXP_MAP = {'1': ('HXP', 1), '5': ('HXP', 5), '22': ('HXP', 22), '2_5': ('HXP_2_5', 5), '6_22': ('HXP_6_22', 22)}


def nogi_hxp(feat=FEAT_RRV, plant=False, steps=STEPS, tgts=tuple(HXP_MAP)):
    """HAR-X⁺ как d22_harxp.run (forecast_step с эмбарго). plant=True — слагаемое ln RV дня t+1."""
    import harx as HX, d22_harxp as P
    HX._SIG.clear()
    A, harj = HX.load(feat); A = P.add_targets(A)
    names = sorted(set(sum([P.terms_for(tg) for tg in tgts], [])))
    X = P.build_terms(A, names)
    if plant:
        X['_plant'] = np.log(A.groupby('ticker')['rv'].shift(-1).clip(lower=1e-10)).values
    out = {}
    for tg in tgts:
        terms = P.terms_for(tg) + (['_plant'] if plant else [])
        V, T = [], []
        for k in steps:
            v, t, _, _ = P.forecast_step(A, tg, k, harj, terms, X)
            V.append(v); T.append(t)
        out[HXP_MAP[tg]] = (_norm(pd.concat(V), 'val'), _norm(pd.concat(T), 'test'))
    return out


def nogi_harx5(feat=FEAT_V3, plant=False, steps=STEPS):
    """HAR-X h=5 как harx.main по умолчанию (tag HARX: слагаемые владельца + Д17.3, без эмбарго — определение первого раунда)."""
    import harx as HX
    HX._SIG.clear()
    A, harj = HX.load(feat)
    terms = HX.default_terms(5)
    X = HX.build_terms(A, terms)
    if plant:
        X['_plant'] = np.log(A.groupby('ticker')['rv'].shift(-1).clip(lower=1e-10)).values
        terms = terms + ['_plant']
    V, T = [], []
    for k in steps:
        v, t = HX.forecast_step(A, 5, k, harj, terms, X)
        V.append(v); T.append(t)
    return {('HARX', 5): (_norm(pd.concat(V), 'val'), _norm(pd.concat(T), 'test'))}


def nogi_rvia(feat=FEAT_V3, plant=False, hs=(1, 5, 22)):
    """Нога RVI (а) как d25_1.frames: ŷ = v·k_i, k — медиана RV_h/v по 252 строкам с эмбарго h; нет k -> HAR-J шага (шаги 1-10).
    h=1 — новое применение: цель rv_target_mean_h1 (= rv_target_h1). plant=True — k без эмбарго (сдвиг 0)."""
    import harx as HX, d25_1 as D25
    A, harj = HX.load(feat)
    out = {}
    for h in hs:
        hv, ht, _ = HX.har_steps(h, A, harj)
        L = D25.rvi_leg(A, h, lag=0 if plant else h)
        fr = []
        for d in (hv, ht):
            d = d.rename(columns={'rv_pred': 'har'}).merge(L, on=['date', 'ticker'], how='left')
            fr.append(d.assign(rv_pred=d.leg.fillna(d.har)))
        out[('RVIa', h)] = (_norm(fr[0], 'val'), _norm(fr[1], 'test'))
    return out


FAMILIES = {'garch': nogi_garch, 'hxp': nogi_hxp, 'harx5': nogi_harx5, 'rvia': nogi_rvia}
FEAT_OF = {'garch': FEAT_V3, 'hxp': FEAT_RRV, 'harx5': FEAT_V3, 'rvia': FEAT_V3}

# ряды первого раунда (val шагов 1-6) для контроля совпадения
R1 = {('HEAVY', h): REG / f'GARCH_svheavy_h{h}_val.parquet' for h in (1, 5, 22)}
R1.update({('RG', h): REG / f'GARCH_svrgarch_h{h}_val.parquet' for h in (1, 5, 22)})
R1.update({(f'{M}_l{k}', 22): REG / f'GARCH_sv{m}_leg{k}_val.parquet' for m, M in (('heavy', 'HEAVY'), ('rgarch', 'RG')) for k in ('6_22', '2_22')})
R1.update({HXP_MAP[tg]: REG / 'D22' / f'HXP_{tg}_val.parquet' for tg in HXP_MAP})
R1.update({('HARX', 5): REG / 'HARX_h5_val.parquet', ('RVIa', 5): REG / 'D25' / 'D25_1a_h5_xgb_h5_val.parquet',
           ('RVIa', 22): REG / 'D25' / 'D25_1a_h22_xgb_h22_val.parquet'})


# ================================================================ запись, индекс, хэши
def sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def _git_head():
    try:
        return subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], cwd=HERE, capture_output=True, text=True).stdout.strip()
    except Exception:
        return ''


def save_leg(nm, h, V, T, seed='det'):
    """val 1-6 -> CACHE; val 7-10 и тест -> TLC (только запись). Возвращает строки индекса."""
    CACHE.mkdir(parents=True, exist_ok=True); TLC.mkdir(parents=True, exist_ok=True)
    code, ysha, now = sha256(Path(__file__)), (sha256(YAML) if YAML.exists() else ''), datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    rows = []
    for fp, d, steps in ((CACHE / f'{nm}_h{h}_s{seed}_val.parquet', V[V.step <= DEV], '1-6'),
                         (TLC / f'{nm}_h{h}_s{seed}_valconf_test.parquet', V[V.step > DEV], '7-10'),
                         (TLC / f'{nm}_h{h}_s{seed}_test.parquet', T, '1-10')):
        d.reset_index(drop=True).to_parquet(fp, index=False)
        rows.append(dict(file=str(fp.relative_to(REG)), noga=nm, h=h, seed=seed, shagi=steps, stroki=len(d), sha256=sha256(fp),
                         sha_koda=code[:16], git=_git_head(), yaml=ysha[:16], vremya=now))
    return rows


def write_index(rows):
    new = pd.DataFrame(rows)
    old = pd.read_csv(INDEX) if INDEX.exists() else pd.DataFrame(columns=new.columns)
    I = pd.concat([old[~old.file.isin(new.file)], new], ignore_index=True).sort_values('file').reset_index(drop=True)
    I.to_csv(INDEX, index=False)
    tl = I[I.file.str.startswith('test_locked/')]
    with open(TLC / 'SHA256.txt', 'w') as f:
        for _, r in tl.iterrows():
            f.write(f'{r.sha256}  {Path(r.file).name}\n')
    return I


# ================================================================ сверка с первым раундом
def sverka(nm, h, V):
    f = R1.get((nm, h))
    base = dict(noga=nm, h=h, r1=str(f.relative_to(REG)) if f else '—')
    if f is None:
        return dict(base, status='нет ряда R1 (новое применение)', n=len(V[V.step <= DEV]))
    r = pd.read_parquet(f, columns=KEY + ['rv_actual', 'rv_pred'])
    r['date'] = pd.to_datetime(r.date); r = r[r.step.between(1, DEV)]
    v = V[V.step <= DEV]
    m = v.merge(r, on=KEY, how='outer', suffixes=('', '_r1'), indicator=True)
    both = m._merge == 'both'
    d = float(np.abs(m.rv_pred[both] - m.rv_pred_r1[both]).max())
    rel = float(np.abs(m.rv_pred[both] / m.rv_pred_r1[both] - 1).max())
    da = float(np.nanmax(np.abs(m.rv_actual[both] - m.rv_actual_r1[both]))) if both.any() else np.nan
    n_out = int((~both).sum())
    ok = n_out == 0 and d <= TOL
    return dict(base, status='OK' if ok else 'РАСХОЖДЕНИЕ', n=int(both.sum()), n_nesovp_strok=n_out, max_abs=d, max_rel=rel, max_abs_rv_actual=da)


# ================================================================ режимы
def det(fams=tuple(FAMILIES)):
    t0 = time.time(); idx, sv = [], []
    for fam in fams:
        res = FAMILIES[fam](FEAT_OF[fam])
        for (nm, h), (V, T) in res.items():
            s = sverka(nm, h, V); sv.append(s)
            print(f'{nm} h={h}: {s["status"]} (max |Δ| = {s.get("max_abs", float("nan")):.2e})', flush=True)
            if s['status'] == 'РАСХОЖДЕНИЕ':
                pd.DataFrame(sv).to_csv(CACHE / 'sverka_B.csv', index=False)
                raise SystemExit(f'стоп: {nm} h={h} не совпадает с первым раундом')
            idx += save_leg(nm, h, V, T)
        print(f'{fam}: {time.time() - t0:.0f} с', flush=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    S = pd.DataFrame(sv)
    old = CACHE / 'sverka_B.csv'
    if old.exists():
        o = pd.read_csv(old); S = pd.concat([o[~(o.noga + o.h.astype(str)).isin(S.noga + S.h.astype(str))], S], ignore_index=True)
    S.to_csv(old, index=False)
    write_index(idx)
    print(f'det готов: {len(idx)} файлов, {time.time() - t0:.0f} с')


def _perturb(feat, cut, rng, out):
    """Будущее к дате cut: все числовые столбцы строк date > cut × U(0.5, 2); цели строк <= cut, чей горизонт (n = число после _h) заходит
    за cut (последние n строк тикера до cut), × U(0.5, 2). Результат — out (parquet)."""
    B = pd.read_parquet(feat)
    dt = pd.to_datetime(B['date'])
    fut = (dt > cut).values
    num = [c for c in B.columns if c not in ('year',) and pd.api.types.is_float_dtype(B[c])]
    B.loc[fut, num] = B.loc[fut, num].values * rng.uniform(0.5, 2.0, size=(fut.sum(), len(num)))
    order = B.assign(_dt=dt).sort_values(['ticker', '_dt'])
    past = order[order._dt <= cut]
    for c in [c for c in num if 'target' in c]:
        n = int(c.split('_h')[-1])
        ix = past.groupby('ticker').tail(n).index
        B.loc[ix, c] = B.loc[ix, c].values * rng.uniform(0.5, 2.0, size=len(ix))
    B.to_parquet(out, index=False)


def _stack(res):
    return pd.concat([pd.concat([V.assign(part='val'), T.assign(part='test')]).assign(noga=nm, h=h) for (nm, h), (V, T) in res.items()])


def _diff(a, b, cut):
    m = a.merge(b, on=['noga', 'h', 'part'] + KEY, suffixes=('', '_p'))
    d = np.abs(m.rv_pred - m.rv_pred_p)
    p = (m.date <= cut).values
    return float(d[p].max()), float(d[~p].max()), int(p.sum())


def leak(fams=tuple(FAMILIES), cuts=('2018-06-29', '2020-03-02', '2021-06-30')):
    rng = np.random.default_rng(20261026); rows = []
    with tempfile.TemporaryDirectory() as td:
        for fam in fams:
            f, feat = FAMILIES[fam], FEAT_OF[fam]
            b0, b1 = _stack(f(feat)), _stack(f(feat, plant=True))
            for c in cuts:
                cut = pd.Timestamp(c); tmp = Path(td) / f'{fam}_{c}.parquet'
                _perturb(feat, cut, rng, tmp)
                p0, p1 = _stack(f(tmp)), _stack(f(tmp, plant=True))
                past0, fut0, n = _diff(b0, p0, cut); past1, _, _ = _diff(b1, p1, cut)
                ok = past0 == 0.0 and fut0 > 0 and past1 > 0
                rows.append(dict(semeystvo=fam, srez=c, strok_do_sreza=n, max_d_do_sreza=past0, max_d_posle_sreza=fut0,
                                 podlog_max_d_do_sreza=past1, bez_utechki_0=past0 == 0.0, podlog_poyman=past1 > 0, ok=ok))
                print(rows[-1], flush=True)
    R = pd.DataFrame(rows); CACHE.mkdir(parents=True, exist_ok=True); R.to_csv(CACHE / 'leak_B.csv', index=False)
    print('ВСЕ OK' if R.ok.all() else 'ЕСТЬ ОШИБКИ'); return bool(R.ok.all())


def zamer():
    """Шаг 0: одна нога × один seed × один шаг, экстраполяция (объём — по таблице «Исполнители» для агента Б)."""
    import d24_6 as G, harx as HX, d22_harxp as P, d25_1 as D25
    rows = []

    def rec(fam, unit, sec, n, note):
        rows.append(dict(semeystvo=fam, edinica=unit, sek_na_edinicu=round(sec, 3), edinic=n, prognoz_min=round(sec * n / 60, 2), primechanie=note))
        print(rows[-1], flush=True)
    a = pd.read_parquet(FEAT_V3, columns=['date', 'ticker', 'close', 'rv'] + [G.tcol(h) for h in G.HS])
    a['date'] = pd.to_datetime(a.date); a['year'] = a.date.dt.year
    gg = G.prep(a, 'SBER'); ntr = int((gg.year < 2020).sum())
    t = time.time(); G.fit_forecast(gg.rr.values, gg.xx.values, ntr)
    rec('HEAVY+RG', 'тикер × шаг (обе модели, все h и дальние ноги)', time.time() - t, 17 * 10, 'seed нет')
    t = time.time(); A, harj = HX.load(FEAT_RRV); A = P.add_targets(A); X = P.build_terms(A, P.terms_for('1')); tl = time.time() - t
    t = time.time(); P.forecast_step(A, '1', 5, harj, P.terms_for('1'), X)
    rec('HAR-X⁺', 'цель × шаг', time.time() - t + tl / 50, 5 * 10, 'цели 1, 5, 22, 2_5, 6_22; seed нет')
    t = time.time(); A3, harj3 = HX.load(FEAT_V3); X3 = HX.build_terms(A3, HX.default_terms(5)); tl = time.time() - t
    t = time.time(); HX.forecast_step(A3, 5, 5, harj3, HX.default_terms(5), X3)
    rec('HAR-X h=5', 'шаг', time.time() - t + tl / 10, 10, 'seed нет')
    t = time.time(); HX.har_steps(5, A3, harj3); D25.rvi_leg(A3, 5)
    rec('RVI (а)', 'h (все шаги векторно)', time.time() - t, 3, 'h=1/5/22; seed нет')
    t = time.time(); DF, st = D25.patch_d22((5,)); st['on'] = True; I = DF.load_inputs('val'); tl = time.time() - t
    t = time.time(); DF.pipeline(5, 42, I, I)  # тест подменён val — только время, файлы не пишутся
    rec('RVI (б): объединитель Д22 + x10', 'h × seed (10 шагов)', time.time() - t + tl, 2 * 10, 'нужны L2_h1 и clf по seed (А1)')
    import d22_h1 as H1, l2_lib as L2
    t = time.time(); Ab, FC = H1.load_base_x(); Ab = P.add_targets(Ab); A2, cols, _, _ = H1.build_features_x(Ab, FC, 1, {}); tl = time.time() - t
    H1.G.update(A=A2, cols=cols, base='hxp')
    t = time.time(); H1._job_clf((5, 6, 45))
    rec('классификатор Д22 (для б, seed 45-51)', 'h × шаг × seed', time.time() - t + tl / 140, 2 * 10 * 7, '1 поток; 4 процесса -> /4')
    t = time.time(); HX.har_steps(22, A3, harj3)
    rec('структурные Р5.6, Р22.1-Р22.6, RVI (в)', 'нога × seed (OLS/арифметика по шагам)', time.time() - t, 11 * 10, 'после кэша А1/А2')
    Z = pd.DataFrame(rows); CACHE.mkdir(parents=True, exist_ok=True); Z.to_csv(CACHE / 'zamer_B.csv', index=False)
    tot = Z.prognoz_min.sum()
    print(f'итого прогноз (1 процесс): {tot:.1f} мин; с детектором и записью ×3: {3 * tot:.1f} мин')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['zamer', 'det', 'leak'])
    ap.add_argument('--fams', nargs='*', default=list(FAMILIES))
    a = ap.parse_args()
    if a.mode == 'zamer':
        zamer()
    elif a.mode == 'det':
        det(a.fams)
    else:
        sys.exit(0 if leak(a.fams) else 1)


if __name__ == '__main__':
    main()
