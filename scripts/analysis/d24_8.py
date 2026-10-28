#!/usr/bin/env python3
"""Д24.8 (26.10, ночь): дневная RV USD000UTSTOM (CETS) и ближнего фьючерса Brent (BR, FORTS) как признаки.
Источник — крейт moex_iss (scripts/analysis/d24_bars.sh): 10-мин бары -> data/regen/D24/rv_ext_daily.parquet (RV окна [09:50; 18:50) для USDRUB,
основной сессии [10:00; 18:45) для BR; склейка BR — в день d контракт с наименьшим LSTTRADE > d, RV — по барам самого контракта).
Признаки (только прошлое): на дату строки t берутся торговые дни ряда строго раньше t; RV_d — последний такой день, RV_w — среднее 5,
RV_m — среднее 22 последних (min 3 / 13). -> data/regen/rv_usdrub_brent.parquet (все календарные дни 2014-05-01 … 2026-02-27).
Стадии:
  rv      — сборка признаков + детектор «будущее не влияет» (+ самопроверка подложенной утечкой);
  harx    — HAR-X: слагаемые ln RV_d/w/m USDRUB и BR в той же эластичной сети (свой реестр harx.TERMS, harx.py не меняется), h=1/5;
  control — сверка драйвера с сохранёнными финалистами (без новых признаков, шаг 2, XGB, seed 42);
  boost   — бустинг с 6 новыми признаками: h=1 — финалист Д8.12 (D12_h1_A_d812: {"harf"}, точки ЛГК B v2b, ноги single/top5/seeds3);
            h=5 — D12_h5 (V9 + Д8.12 + Д7 + Huber, точки D12_h5_selected_h5, ноги single/seeds3 + wrap2 по правилу final2_wrap без 1a);
  gibrid  — gibrid_v17.py (oos, шаги 2-6, --features features_pit_v3) для кандидата;
  eval    — правило третьего круга (final3.designs/summarize), отложенный 2021 против текущего лучшего кругов 1-2, порог по 3 seed;
  yaml    — yaml прошедших в data/regen/final3/kandidaty/.
Подтверждающий набор (val 2022-2024, шаги 7-9) и тест не оцениваются: метрики — только шаги <= 6; тест — test_locked/<boost>/ без метрик.
python3 scripts/analysis/d24_8.py --stage rv|harx|control|boost|gibrid|eval|yaml [--h 1 5] [--workers 2]"""
import os, sys, json, time, argparse, subprocess, multiprocessing as mp
from pathlib import Path
os.environ.setdefault('L2_FEAT', '/Users/nilysenok/Desktop/Старое/PhD_Vol_trading/data/regen/features_pit_v3.parquet')
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np, pandas as pd
import l2_lib as L

R = L.REGEN
D24 = R / 'D24'
OUT = R / 'rv_usdrub_brent.parquet'
SER = (('USDRUB', 'usd'), ('BRENT', 'br'))
EXT = [f'{p}_rv_{k}' for _, p in SER for k in ('d', 'w', 'm')]
WIN = {'d': (1, 1), 'w': (5, 3), 'm': (22, 13)}
D0, D1_ = '2014-05-01', '2026-02-27'
MIN_BARS = 30  # ночное умолчание: день ряда — не меньше 30 доходностей в окне (из 53)
STALE = 10  # ночное умолчание: последний торговый день ряда старше 10 календарных дней -> NaN (в обучении -> 0, в HAR-X -> среднее)
FEAT3 = R / 'features_pit_v3.parquet'
H5CFG = {'v9': True, 'harf': True, 'huber': True}
INC = {1: ('D12_h1_B_b0.25', 'top5|V0_XGB|B-sm|L0', 0.28917618310624094, 0.2395531296561522),  # final3_dry/incumbent.csv
       5: ('F2m_h5', 'wrap2|V0_LGB|B-sm|L0', 0.2122287402429689, 0.155332001485612)}
BASE = {1: 'D12_h1_A_d812', 5: 'D12_h5'}
K = ['date', 'ticker', 'step']
TAG = 'D24_8'


def log(*a):
    print(time.strftime('%T'), *a, flush=True)


# ---------------------------------------------------------------- признаки
def ext_features(daily, dates, leak=False):
    """daily: date, ticker (USDRUB|BRENT), rv -> признаки на даты `dates` по торговым дням ряда строго раньше даты.
    leak=True — самопроверка: RV_d подменяется RV следующего торгового дня ряда (детектор обязан поймать)."""
    out = pd.DataFrame({'date': pd.DatetimeIndex(sorted(pd.unique(pd.to_datetime(dates))))})
    for tk, p in SER:
        x = daily[(daily.ticker == tk) & (daily.n_bars >= MIN_BARS)]  # неполные дни (< 30 доходностей окна) — не торговые дни ряда
        s = x.sort_values('date').set_index('date')['rv'].astype('float64')
        f = pd.DataFrame({f'{p}_rv_{k}': s.rolling(n, min_periods=mp_).mean() for k, (n, mp_) in WIN.items()})
        if leak:
            f[f'{p}_rv_d'] = s.shift(-1)
        f[f'{p}_day'] = f.index
        f = f.reset_index()
        out = pd.merge_asof(out, f, on='date', direction='backward', allow_exact_matches=False)  # строго раньше t
        stale = (out['date'] - out[f'{p}_day']).dt.days > STALE  # ряд не торгуется (USDRUB на бирже с 12.06.2024) -> NaN, не протяжка
        out.loc[stale, [f'{p}_rv_{k}' for k in WIN]] = np.nan
    return out


def load_daily():
    d = pd.read_parquet(D24 / 'rv_ext_daily.parquet')
    d['date'] = pd.to_datetime(d['date'])
    return d


def stage_rv(a):
    d = load_daily()
    dates = pd.date_range(D0, D1_, freq='D')
    F = ext_features(d, dates)
    F.to_parquet(OUT, index=False)
    log('записан', OUT, len(F), 'строк;', F[EXT].notna().mean().round(4).to_dict())
    # сводка по рядам
    rows = []
    for tk, p in SER:
        x = d[d.ticker == tk]
        rows.append(dict(ряд=tk, дней=len(x), первая=str(x.date.min().date()), последняя=str(x.date.max().date()),
                         медиана_баров=float(x.n_bars.median()), доля_дней_lt_30_баров=round(float((x.n_bars < 30).mean()), 4),
                         медиана_сигма_год=round(float(np.sqrt(x.rv.median() * 252)), 4)))
    pd.DataFrame(rows).to_csv(D24 / 'rv_ext_svodka.csv', index=False)
    print(pd.DataFrame(rows).to_string())
    # детектор «будущее не влияет»: RV дней ряда >= t -> шум (бутстрап прошлого), признаки на датах <= t должны совпасть побайтно
    rng = np.random.default_rng(20261026)
    res = []
    for t in ('2016-06-30', '2019-06-30', '2021-12-31', '2023-06-30'):
        t = pd.Timestamp(t)
        B = d.copy()
        for tk, _ in SER:
            m = (B.ticker == tk) & (B.date >= t)  # строже обычного: шум с дня t включительно (признак дня t — только дни < t)
            B.loc[m, 'rv'] = rng.choice(B.loc[(B.ticker == tk) & (B.date < t), 'rv'].values, m.sum()) * rng.uniform(0.2, 5, m.sum())
        for leak in (False, True):
            base = ext_features(d, dates, leak)
            G = ext_features(B, dates, leak)
            past = base.date <= t
            for c in EXT:
                x, y = base.loc[past, c].values, G.loc[past, c].values
                mism = int((~((x == y) | (np.isnan(x) & np.isnan(y)))).sum())
                after = int((~((base.loc[~past, c].values == G.loc[~past, c].values))).sum())
                res.append(dict(t=str(t.date()), leak=leak, feature=c, rows_le_t=int(past.sum()), mismatch_le_t=mism, changed_after_t=after))
    L_ = pd.DataFrame(res)
    L_.to_csv(D24 / 'leak_rv_ext.csv', index=False)
    ok = L_[~L_.leak].mismatch_le_t.sum() == 0
    caught = (L_[L_.leak & L_.feature.str.endswith('_rv_d')].groupby('t').mismatch_le_t.sum() > 0).all()
    log(f'детектор: расхождений до t {int(L_[~L_.leak].mismatch_le_t.sum())} -> {"PASS" if ok else "FAIL"}; '
        f'самопроверка (RV_d = RV следующего дня) поймана на всех t: {"PASS" if caught else "FAIL"}')
    assert ok and caught


def add_ext(A):
    F = pd.read_parquet(OUT)[['date'] + EXT]
    A = A.copy()
    A['date'] = pd.to_datetime(A['date'])
    n = len(A)
    A = A.merge(F, on='date', how='left', validate='many_to_one')
    assert len(A) == n
    return A


# ---------------------------------------------------------------- HAR-X
def stage_harx(a):
    import harx as HX
    A, harj = HX.load()
    A = add_ext(A).sort_values('date', kind='stable')
    A.index = range(len(A))
    names = []
    for tk, p in SER:
        for k in ('d', 'w', 'm'):
            nm = f'ln_{p}_rv_{k}'
            HX.TERMS[nm] = (lambda col: (lambda B: np.log(B[col].astype('float64').clip(lower=1e-12))))(f'{p}_rv_{k}')
            names.append(nm)
    lk = HX.leak_test(A, names)
    if HX._SIG.get('v') is None:  # leak_test возвращает в кэш None, если ln_sig1 ещё не считался
        HX._SIG.pop('v', None)
    lk.to_csv(D24 / 'leak_harx_terms.csv', index=False)
    log('детектор слагаемых HAR-X: расхождений до t', int(lk.mismatch.sum()))
    assert int(lk.mismatch.sum()) == 0
    rows = []
    for h in a.h:
        out = {}
        for lab, terms in (('HARX', HX.default_terms(h)), ('HARX+ext', HX.default_terms(h) + names)):
            lg = []
            Xt = HX.build_terms(A, terms)
            v, t = HX.harx_forecasts(h, A, harj, terms, Xt, lg)
            v, t = HX.add_bias(v, t)
            out[lab] = v
            tag = f'{TAG}_harx_{"ext" if "ext" in lab else "base"}'
            v.to_parquet(D24 / f'{tag}_h{h}_val.parquet', index=False)
            (R / 'test_locked' / f'{TAG}_harx').mkdir(parents=True, exist_ok=True)
            t.to_parquet(R / 'test_locked' / f'{TAG}_harx' / f'{tag}_h{h}_test.parquet', index=False)
            pd.DataFrame(lg).to_csv(D24 / f'{tag}_h{h}_coefs.csv', index=False)
            log(f'HAR-X h={h} {lab} готов')
        for code in ('sm', 'smrob'):
            r = dict(h=h, code=code)
            for lab, v in out.items():
                q = HX.dev_table(v, f'rv_pred_{code}')
                r.update({f'{lab}_{2015 + s}': q[s] for s in q})
                r[f'{lab}_1720'] = np.mean([q[s] for s in range(2, 6)])
            r['ratio_1720'] = r['HARX+ext_1720'] / r['HARX_1720']
            r['ratio_2021'] = r['HARX+ext_2021'] / r['HARX_2021']
            rows.append(r)
    S = pd.DataFrame(rows)
    S.to_csv(D24 / 'harx_ext_summary.csv', index=False)
    print(S.round(4).to_string())
    for h in a.h:
        c = pd.read_csv(D24 / f'{TAG}_harx_ext_h{h}_coefs.csv')
        print(f'h={h} коэффициенты новых слагаемых по шагам 1-6:\n', c[c.step <= 6][['step', 'alpha', 'l1_ratio'] + [f'g_{n}' for n in names]].round(4).to_string())


# ---------------------------------------------------------------- бустинг
def matrix(h, ext=True):
    if h == 1:
        A, FC = L.load_base(FEAT3)
        cfg = {'harf': True}
    else:
        A, FC = L.load_base(None, ('d7',), 5)
        cfg = H5CFG
    if ext:
        A = add_ext(A)
        FC = FC + EXT
    A2, cols, _, _ = L.build_features(A, FC, h, cfg)
    return A2, cols, cfg


G = {}


def job5(t):
    m, s, seed = t
    A, cols, cfg = G['A'], G['cols'], G['cfg']
    p = G['P'][(m, min(s, 6))]
    ref = L.make_ref(m, A, cols, 5, cfg, 'Bv2b') if s > 1 else None
    D = L.step_data(A, cols, 5, s, m, cfg, 'Bv2b')
    pv, pt, bi, gain, _ = L.fit(m, p, D, cfg, ref=ref, seed=seed)
    fr = []
    for k, pr in (('va', pv), ('te', pt)):
        d = D['parts'][k]
        fr.append(pd.DataFrame({'date': d.date.values, 'ticker': d.ticker.values, 'year': d.year.values, 'step': s, 'seed': seed, 'rank': 1,
                                'rv_actual': D['y'][k], 'rv_pred': np.clip(np.exp(pr), 1e-10, None)}))
    g = dict(zip(D['names'], gain)) if gain is not None else {}
    return m, s, seed, bi, fr[0], fr[1], {c: g.get(c, 0.0) for c in EXT}


def train(h, A, cols, cfg, workers, legs=('single', 'top5', 'seeds3'), steps=range(1, 10), models=('xgb', 'lgbm'), seeds=(42, 43, 44)):
    """-> список (m, s, point, seed, rank, bi, val, test)."""
    if h == 1:
        import d12_h1 as D1
        D1.G.update(A=A, grid=D1.grid(), cols={'A_d812': cols})
        pl = [t for t in D1.plan(list(legs)) if t[1] in steps and t[0] in models and t[3] in seeds]
        with mp.get_context('fork').Pool(workers) as P:
            res = P.map(D1.job, [('A_d812',) + t for t in pl], chunksize=1)
        return [r[1:9] for r in res]
    G.update(A=A, cols=cols, cfg=cfg, P=L.selected_params('D12_h5', 5))
    tasks = [(m, s, sd) for s in steps for m in models for sd in seeds]
    with mp.get_context('fork').Pool(workers) as P:
        res = P.map(job5, tasks, chunksize=1)
    gains = pd.DataFrame([dict(model=r[0], step=r[1], seed=r[2], **r[6]) for r in res])
    gains.to_csv(D24 / f'gain_ext_h5.csv', index=False)
    return [(r[0], r[1], 0, r[2], 1, r[3], r[4], r[5]) for r in res]


def write_legs(boost, h, res, lockdir):
    lockdir.mkdir(parents=True, exist_ok=True)
    for m in ('xgb', 'lgbm'):
        rr = [r for r in res if r[0] == m]
        sel = lambda f: (pd.concat([x[6] for x in rr if f(x)], ignore_index=True), pd.concat([x[7] for x in rr if f(x)], ignore_index=True))
        files = {'': sel(lambda x: x[4] == 1 and x[3] == 42), '_seeds4344': sel(lambda x: x[3] in (43, 44))}
        if h == 1:
            files['_top5'] = sel(lambda x: x[1] <= 6 and x[3] == 42)
        for sfx, (v, t) in files.items():
            v.sort_values(['step', 'date', 'ticker']).to_parquet(R / f'{boost}_{m}_h{h}{sfx}_val.parquet', index=False)
            t.sort_values(['step', 'date', 'ticker']).to_parquet(lockdir / f'{boost}_{m}_h{h}{sfx}_test.parquet', index=False)
    if h == 5:
        wrap2(boost, h, lockdir)


def wrap2(T, h, lockdir):
    """Нога wrap2 по правилу final2_wrap.py (компоненты 1b — 3 seed, 2 — среднее LGBM и XGB; база — лучшая single по ord шагов 2-6).
    ↺ компонент 1a (топ-5 точек поиска) не строится: у D12_h5 нет обучения точек ранга 2-5 с новыми признаками (у самого D12_h5 1a не взят)."""
    import final2_wrap as W
    P0 = W.build(T, h, (), 'val')
    e0 = {m: W.ev(P0[m]) for m in W.M}
    prim = min(W.M, key=lambda m: e0[m][0])
    q0 = e0[prim][0]
    ind = {c: W.ev(W.combine(W.build(T, h, (c,), 'val'), (c,), prim))[0] for c in ('1b', '2')}
    keep = [c for c in ind if ind[c] <= q0]
    while keep:
        q = W.ev(W.combine(W.build(T, h, tuple(keep), 'val'), tuple(keep), prim))[0]
        if q <= q0:
            break
        keep.remove(max(keep, key=lambda c: ind[c]))
    final = tuple(keep)
    other = [m for m in W.M if m != prim][0]
    for k in ('val', 'test'):
        P = W.build(T, h, tuple(c for c in final if c != '2'), k)
        Wd = W.combine(P, final, prim)
        slots = {prim: Wd, other: P[other]}
        d = R if k == 'val' else lockdir
        for m in W.M:
            slots[m].assign(wrap='+'.join(final) or 'none').sort_values(W.K).to_parquet(d / f'{T}_{m}_h{h}_wrap2_{k}.parquet', index=False)
    pd.DataFrame([dict(boost=T, h=h, prim=prim, base=q0, **{f'q_{c}': v for c, v in ind.items()}, final='+'.join(final) or 'none')]).to_csv(
        D24 / f'wrap2_{T}_h{h}.csv', index=False)
    log(f'{T} wrap2: база {prim} {q0:.5f}; ' + ', '.join(f'{c} {v:.5f}' for c, v in ind.items()) + f'; итог {"+".join(final) or "none"}')


def stage_control(a):
    """Драйвер без новых признаков должен повторить сохранённые файлы финалистов (шаг 2, XGB, точка ранга 1, seed 42)."""
    for h in a.h:
        A, cols, cfg = matrix(h, ext=False)
        res = train(h, A, cols, cfg, 1, legs=('single',), steps=[2], models=('xgb',), seeds=(42,))
        v = [r for r in res if r[4] == 1 and r[3] == 42][0][6]
        ref = pd.read_parquet(R / f'{BASE[h]}_xgb_h{h}_val.parquet')
        ref = ref[(ref.step == 2) & (ref.seed == 42)]
        m = v.merge(ref, on=['date', 'ticker'], suffixes=('', '_ref'))
        d = float(np.max(np.abs(np.log(m.rv_pred) - np.log(m.rv_pred_ref))))
        log(f'контроль h={h}: строк {len(v)} / {len(ref)} / общих {len(m)}, max |Δ ln| = {d:.2e}')
        open(D24 / f'control_h{h}.txt', 'w').write(f'h={h} шаг 2 XGB seed 42: строк {len(v)} / {len(ref)}, общих {len(m)}, max|dln| {d:.3e}\n')


def stage_boost(a):
    for h in a.h:
        t0 = time.time()
        A, cols, cfg = matrix(h, ext=True)
        log(f'h={h}: признаков {len(cols)} (+{len(EXT)} новых), строк {len(A)}')
        res = train(h, A, cols, cfg, a.workers)
        boost = f'{TAG}_h{h}'
        write_legs(boost, h, res, R / 'test_locked' / boost)
        pd.DataFrame([dict(model=r[0], step=r[1], point=r[2], seed=r[3], rank=r[4], trees=r[5]) for r in res]).to_csv(D24 / f'{boost}_trees.csv', index=False)
        log(f'h={h}: обучено {len(res)} моделей за {time.time() - t0:.0f} с')


# ---------------------------------------------------------------- гибрид и правило
LEGS = {1: ['single', 'top5', 'seeds3'], 5: ['single', 'seeds3', 'wrap2']}


def run_gibrid(boost, h, regen=R):
    od = D24 / 'gibrid' / boost
    od.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, '-W', 'ignore', str(HERE / 'gibrid_v17.py'), '--boost', boost, '--h', str(h), '--features', str(FEAT3), '--regen', str(regen),
           '--legs', *LEGS[h], '--out', str(od), '--devlog', str(D24 / f'gibrid_devlog_{boost}.md')]
    with open(D24 / f'gibrid_{boost}.log', 'w') as f:
        rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=str(HERE.parent.parent)).returncode
    log(f'гибрид {boost}: код {rc}')
    return rc


def stage_gibrid(a):
    for h in a.h:
        run_gibrid(f'{TAG}_h{h}', h)


def seed_q(boost, h, m='xgb'):
    """OOS QLIKE 2017-2020 (шаги 2-5) одиночной точки V0 с Б0.1 sm по каждому seed (шаги <= 6; 7-9 не читаются)."""
    a = pd.read_parquet(R / f'{boost}_{m}_h{h}_val.parquet')
    b = pd.read_parquet(R / f'{boost}_{m}_h{h}_seeds4344_val.parquet')
    out = {}
    for sd in (42, 43, 44):
        v = (a if sd == 42 else b)
        v = v[(v.seed == sd) & (v.step <= 6)] if 'seed' in v else v[v.step <= 6]
        e = L.evaluate(v, lo=2, hi=5)
        out[sd] = e['sm'][0]
    return out


def summary(boost, h, gib_dir):
    import final3 as F3
    D = F3.designs(dict(gib_dir=gib_dir, gboost=boost, h=h))
    S = F3.summarize(dict(), D)
    best = S.iloc[0]
    v0 = S[S.design.str.contains(r'\|V0_XGB\|')].iloc[0]
    return S, best, v0


def stage_eval(a, tag=TAG, label='Д24.8'):
    rows = []
    for h in a.h:
        boost = f'{tag}_h{h}'
        S, best, v0 = summary(boost, h, D24 / 'gibrid' / boost)
        S.to_csv(D24 / f'designs_rank_{boost}.csv', index=False)
        B = pd.read_csv(R / 'final3_dry' / f'designs_rank_{BASE[h]}_h{h}.csv')
        bb = B.iloc[0]
        bdes = B[B.design == best.design].iloc[0] if (B.design == best.design).any() else None
        sq, sb = seed_q(boost, h), seed_q(BASE[h], h)
        inc = INC[h]
        for kind, x in (('лучший', best), ('V0_XGB', v0)):
            r = dict(rychag=label, h=h, boost=boost, kind=kind, design=x.design, n_failed=int(x.n_failed), failed=x.failed,
                     **{f'q_{y}': x[f'q_{y}'] for y in range(2017, 2022)}, mean_1720=x.mean_1720, q2021=x.q2021,
                     base=BASE[h], base_design=bb.design, base_n_failed=int(bb.n_failed), base_1720=bb.mean_1720, base_2021=bb.q2021,
                     base_same_design_1720=(bdes.mean_1720 if bdes is not None else np.nan),
                     inc=inc[0], inc_1720=inc[2], inc_2021=inc[3], pass_2021=bool(x.q2021 <= inc[3]),
                     d_vs_base=x.mean_1720 / bb.mean_1720 - 1,
                     **{f'seed{s}': sq[s] for s in sq}, **{f'base_seed{s}': sb[s] for s in sb})
            r['seed_mean_d'] = np.mean(list(sq.values())) / np.mean(list(sb.values())) - 1
            r['seed_all_better'] = all(sq[s] < sb[s] for s in sq)
            r['porog_3seed'] = bool(r['seed_mean_d'] <= -0.01 and r['seed_all_better'])
            r['proshel'] = bool(kind == 'лучший' and r['pass_2021'] and r['n_failed'] <= r['base_n_failed'] and x.mean_1720 < bb.mean_1720 and r['porog_3seed'])
            rows.append(r)
    E = pd.DataFrame(rows)
    p = D24 / f'eval_{tag}.csv'
    E.to_csv(p, index=False)
    with pd.option_context('display.width', 250, 'display.max_columns', 40):
        print(E[['h', 'kind', 'design', 'n_failed', 'mean_1720', 'q2021', 'base_1720', 'base_2021', 'd_vs_base', 'pass_2021',
                 'seed42', 'seed43', 'seed44', 'base_seed42', 'base_seed43', 'base_seed44', 'seed_mean_d', 'porog_3seed', 'proshel']].round(4).to_string())
    return E


YAML = {1: dict(semeystvo='Д8.12: прогноз HAR-J как признак', osnova='D12_h1_A_d812',
                opisanie='Д8.12 (XGBoost/LightGBM на ln RV + ln HAR-J как признак) + RV_d/w/m USDRUB (CETS) и ближнего BR (FORTS) с лагом 1, точки ЛГК B v2b без нового подбора'),
        5: dict(semeystvo='V9 на HAR-J', osnova='D12_h5',
                opisanie='D12_h5 (V9 + Д8.12 + Д7 + Huber, точки D12_h5) + RV_d/w/m USDRUB (CETS) и ближнего BR (FORTS) с лагом 1; нога wrap2 без 1a')}


def write_yaml(boost, h, rych, info, legs, test_dir, n_var):
    y = (f"boost: {boost}\nh: {h}\nsemeystvo: \"{info['semeystvo']}\"\nrychag: {rych}\nopisanie: \"{info['opisanie']}\"\nregen: data/regen\n"
         f"legs: [{', '.join(legs)}]\nodinochnaya: false\nchistyy: false\ntest_dir: {test_dir}\nosnova: {info['osnova']}\nn_variantov: {n_var}\n")
    d = R / 'final3' / 'kandidaty'
    d.mkdir(parents=True, exist_ok=True)
    (d / f'{boost}.yaml').write_text(y)
    log('yaml ->', d / f'{boost}.yaml')


def stage_yaml(a):
    E = pd.read_csv(D24 / f'eval_{TAG}.csv')
    for h in a.h:
        e = E[(E.h == h) & (E.kind == 'лучший')].iloc[0]
        if e.proshel:
            write_yaml(f'{TAG}_h{h}', h, 'Д24.8', YAML[h], LEGS[h], f'data/regen/test_locked/{TAG}_h{h}', 1)
        else:
            log(f'h={h}: не прошёл — yaml не пишется')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', required=True, choices=['rv', 'harx', 'control', 'boost', 'gibrid', 'eval', 'score', 'yaml'])
    ap.add_argument('--h', type=int, nargs='+', default=[1, 5])
    ap.add_argument('--workers', type=int, default=2)
    a = ap.parse_args()
    D24.mkdir(parents=True, exist_ok=True)
    globals()[f'stage_{"eval" if a.stage == "score" else a.stage}'](a)


if __name__ == '__main__':
    main()
