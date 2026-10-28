#!/usr/bin/env python3
"""Д25.7 (26.10, ночь): признаки скачков J = max(RV − BV, 0) (столбец `jump` features_pit_v3) — в HAR-X (слагаемые) и в бустинг.
Признаки строки (date=d, ticker): по дням тикера строго раньше d, т. е. к дню прогноза t = d−1 включительно (как rv_d / jump_lag1):
  jx_n5     — число дней с J > 0 за последние 5 торговых дней тикера (min 3);
  jx_n22    — то же за 22 (min 13);
  jx_since  — торговых дней с последнего дня с J > 0 (0 — скачок был в день t); до первого скачка — NaN;
  jx_sum22  — сумма J за 22 дня (min 13).
Имена без подстроки 'jump' (иначе l2_lib.harj_cols подхватил бы их в HAR-J и изменил базу Д8.12).
Стадии:
  feat    — data/regen/D25/jump_feats.parquet (+ SHA-256) и детектор «будущее не влияет» с подложенной утечкой;
  harx    — HAR-X: 4 слагаемых в той же эластичной сети через harx.TERMS (harx.py не меняется), h=1/5/22, справка (HAR-X — только нога);
  control — драйвер без новых признаков повторяет сохранённые файлы базы (шаг 2, XGB, seed 42);
  boost   — h=1: D22_h1 (+4 признака, точки D22_h1, ноги single/seeds3; top5 не строится ↺);
            h=5: D12_h5 (+4 признака, точки D12_h5, ноги single/seeds3/wrap2 по правилу final2_wrap без 1a ↺);
  gibrid  — gibrid_v17.py (oos) ; eval — правило третьего круга ; yaml — прошедшим.
Подтверждающий набор (шаги 7-9) и тест не оцениваются: метрики — только шаги <= 6; тест — test_locked/D25_7_h{h}/ без метрик.
python3 scripts/analysis/d25_7.py --stage feat|harx|control|boost|gibrid|eval|yaml --h 1 [--workers 2]"""
import os, sys, time, json, hashlib, argparse, subprocess, multiprocessing as mp
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
DATA = Path('/Users/nilysenok/Desktop/Старое/PhD_Vol_trading/data')
R = DATA / 'regen'
OUTD = R / 'D25'
JF = OUTD / 'jump_feats.parquet'
FEAT3 = R / 'features_pit_v3.parquet'
JX = ['jx_n5', 'jx_n22', 'jx_since', 'jx_sum22']
TAG = 'D25_7'
BASE = {1: 'D22_h1', 5: 'D12_h5'}
# 2021 «текущего лучшего» = финал третьего круга на h (final3/final_choice.csv, designs_rank_*): D22_h1 top5 0,2292; D12_h5 wrap2 0,1566
INC = {1: ('D22_h1', 'top5|V0_XGB|B-sm|L0', 0.2857237110986891, 0.2291744344713429),
       5: ('D12_h5', 'wrap2|V0_XGB|B-sm|L0', 0.21590568109908795, 0.1566402147552571)}
LEGS = {1: ['single', 'seeds3'], 5: ['single', 'seeds3', 'wrap2']}
SEEDS = (42, 43, 44)


def log(*a):
    print(time.strftime('%T'), *a, flush=True)


# ---------------------------------------------------------------- признаки
def jump_features(B, leak=False):
    """B: date, ticker, jump -> date, ticker, JX. leak=True — самопроверка: окно сдвинуто в будущее (J следующего дня d+1 входит в признак строки d)."""
    B = B[['date', 'ticker', 'jump']].copy()
    B['date'] = pd.to_datetime(B['date'])
    B = B.sort_values(['ticker', 'date'], kind='stable').reset_index(drop=True)
    g = B.groupby('ticker', sort=False)
    j = g['jump'].shift(-1).astype('float64') if leak else g['jump'].shift(1).astype('float64')
    ind = (j > 0).astype('float64').where(j.notna())
    out = B[['date', 'ticker']].copy()
    out['jx_n5'] = ind.groupby(B['ticker']).transform(lambda s: s.rolling(5, min_periods=3).sum())
    out['jx_n22'] = ind.groupby(B['ticker']).transform(lambda s: s.rolling(22, min_periods=13).sum())
    out['jx_sum22'] = j.groupby(B['ticker']).transform(lambda s: s.rolling(22, min_periods=13).sum())
    # дней с последнего скачка: позиция строки минус позиция последнего дня с J > 0 (по строкам тикера, j уже сдвинут)
    pos = g.cumcount().astype('float64')
    last = pos.where(ind == 1).groupby(B['ticker']).ffill()
    out['jx_since'] = pos - last
    return out


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def stage_feat(a):
    B = pd.read_parquet(FEAT3, columns=['date', 'ticker', 'jump'])
    F = jump_features(B)
    OUTD.mkdir(parents=True, exist_ok=True)
    F.to_parquet(JF, index=False)
    hs = sha(JF)
    (OUTD / 'jump_feats.sha256').write_text(f'{hs}  {JF}\n')
    log('записан', JF, len(F), 'строк; SHA-256', hs)
    print(F[JX].describe().round(5).to_string())
    print('доля дней с J>0 (все строки v3):', round(float((B.jump > 0).mean()), 4))
    # детектор «будущее не влияет»: J дней >= t заменяется шумом; признаки строк с date <= t обязаны совпасть побайтно
    rng = np.random.default_rng(20261026)
    res = []
    for t in ('2016-06-30', '2019-06-30', '2021-12-31', '2023-06-30'):
        t = pd.Timestamp(t)
        Bn = B.copy()
        Bn['date'] = pd.to_datetime(Bn['date'])
        m = Bn.date >= t  # строже обычного: шум с дня t включительно (строка t использует только дни < t)
        Bn.loc[m, 'jump'] = rng.choice(Bn.loc[~m, 'jump'].dropna().values, m.sum()) * rng.choice([0.0, 1.0, 7.0], m.sum())
        for leak in (False, True):
            base = jump_features(B, leak)
            G = jump_features(Bn, leak)
            assert (base[['date', 'ticker']].values == G[['date', 'ticker']].values).all()
            past = (base.date <= t).values
            for c in JX:
                x, y = base.loc[past, c].values, G.loc[past, c].values
                mism = int((~((x == y) | (np.isnan(x) & np.isnan(y)))).sum())
                xa, ya = base.loc[~past, c].values, G.loc[~past, c].values
                after = int((~((xa == ya) | (np.isnan(xa) & np.isnan(ya)))).sum())
                res.append(dict(t=str(t.date()), leak=leak, feature=c, rows_le_t=int(past.sum()), mismatch_le_t=mism, changed_after_t=after))
    Lk = pd.DataFrame(res)
    Lk.to_csv(OUTD / 'leak_jump_feats.csv', index=False)
    ok = Lk[~Lk.leak].mismatch_le_t.sum() == 0
    caught = (Lk[Lk.leak].groupby('t').mismatch_le_t.sum() > 0).all()
    moved = (Lk[~Lk.leak].groupby('t').changed_after_t.sum() > 0).all()
    log(f'детектор: расхождений до t {int(Lk[~Lk.leak].mismatch_le_t.sum())} -> {"PASS" if ok else "FAIL"}; '
        f'шум меняет строки после t: {moved}; самопроверка (J дня d+1) поймана на всех t: {"PASS" if caught else "FAIL"}')
    assert ok and caught and moved


def add_jx(A):
    F = pd.read_parquet(JF)
    A = A.copy()
    dt = A['date'].dtype
    A['date'] = pd.to_datetime(A['date'])
    n = len(A)
    A = A.merge(F, on=['date', 'ticker'], how='left', validate='many_to_one')
    assert len(A) == n
    if dt == object:
        pass
    return A


# ---------------------------------------------------------------- HAR-X (справка: HAR-X — только нога)
HXF = {'jx_n5': lambda B: B['jx_n5'].astype('float64') / 5, 'jx_n22': lambda B: B['jx_n22'].astype('float64') / 22,
       'jx_since': lambda B: np.log1p(B['jx_since'].astype('float64')),
       'jx_sum22': lambda B: np.log(B['jx_sum22'].astype('float64').clip(lower=1e-8))}


def stage_harx(a):
    import harx as HX
    A, harj = HX.load()
    A = add_jx(A).sort_values('date', kind='stable')
    A.index = range(len(A))
    names = []
    for c, f in HXF.items():
        nm = f'hx_{c}'
        HX.TERMS[nm] = f
        names.append(nm)
    lk = HX.leak_test(A, names)
    if HX._SIG.get('v') is None:
        HX._SIG.pop('v', None)
    lk.to_csv(OUTD / 'leak_harx_jx.csv', index=False)
    log('детектор слагаемых HAR-X: расхождений до t', int(lk.mismatch.sum()))
    assert int(lk.mismatch.sum()) == 0
    rows = []
    for h in a.h:
        out = {}
        variants = (('HARX', HX.default_terms(h)), ('HARX+jx', HX.default_terms(h) + names))
        if a.harj:  # HAR-J (эконометрическая нога h=22 и эталон) + только 4 слагаемых скачков
            variants = (('HARJ0', []), ('HARJ+jx', names))
        for lab, terms in variants:
            lg = []
            Xt = HX.build_terms(A, terms)
            v, t = HX.harx_forecasts(h, A, harj, terms, Xt, lg)
            v, t = HX.add_bias(v, t)
            out[lab] = v
            tag = f'{TAG}_{"harj" if a.harj else "harx"}_{"jx" if "jx" in lab else "base"}'
            v.to_parquet(OUTD / f'{tag}_h{h}_val.parquet', index=False)
            (R / 'test_locked' / f'{TAG}_harx').mkdir(parents=True, exist_ok=True)
            t.to_parquet(R / 'test_locked' / f'{TAG}_harx' / f'{tag}_h{h}_test.parquet', index=False)
            pd.DataFrame(lg).to_csv(OUTD / f'{tag}_h{h}_coefs.csv', index=False)
            log(f'HAR-X h={h} {lab} готов')
        for code in ('sm', 'smrob'):
            r = dict(h=h, code=code)
            for lab, v in out.items():
                q = HX.dev_table(v, f'rv_pred_{code}')
                r.update({f'{lab}_{2015 + s}': q[s] for s in q})
                r[f'{lab}_1720'] = np.mean([q[s] for s in range(2, 6)])
            b0, b1 = [lab for lab, _ in variants]
            r['ratio_1720'] = r[f'{b1}_1720'] / r[f'{b0}_1720']
            r['ratio_2021'] = r[f'{b1}_2021'] / r[f'{b0}_2021']
            rows.append(r)
        pd.DataFrame(rows).to_csv(OUTD / f'{"harj" if a.harj else "harx"}_jx_summary.csv', index=False)
    S = pd.DataFrame(rows)
    print(S.round(4).to_string())


# ---------------------------------------------------------------- бустинг h=1 (D22_h1 + jx)
G = {}


def _job1(t):
    import l2_lib as L
    import d22_h1 as P
    from tune_L2 import SPACEB
    m, step, seed = t
    D = L.step_data(G['A'], G['cols'], 1, step, m, P.CFG, P.TAG)
    ref = L.make_ref(m, G['A'], G['cols'], 1, P.CFG, P.TAG) if step > 1 else None
    pv, pt, bi, gain, _ = L.fit(m, G['sel'][(m, min(step, 6))], D, P.CFG, seed=seed, ref=ref)
    fr = []
    for part, pred in (('va', pv), ('te', pt)):
        d = D['parts'][part]
        fr.append(pd.DataFrame({'date': d['date'].values, 'ticker': d['ticker'].values, 'year': d['year'].values, 'step': step, 'seed': seed,
                                'rank': 1, 'rv_actual': d[L.tcol(1)].values, 'rv_pred': np.clip(np.exp(pred), 1e-10, None)}))
    g = dict(zip(D['names'], gain)) if gain is not None else {}
    return m, step, seed, bi, fr[0], fr[1], {c: g.get(c, 0.0) for c in JX}


def matrix1(jx=True):
    import d22_h1 as P  # ставит L2_FEAT = features_pit_v3_rrv до импорта l2_lib
    import l2_lib as L
    from tune_L2 import SPACEB
    P.patch()
    A, FC = L.load_base()
    if jx:
        A = add_jx(A)
        FC = FC + JX
    A2, cols, _, _ = L.build_features(A, FC, 1, P.CFG)
    sel = L.selected_params(P.TAG, 1)
    G.update(A=A2, cols=cols, sel={(m, s): {k: sel[(m, s)][k] for k, *_ in SPACEB[m]} for m in ('xgb', 'lgbm') for s in range(1, 7)})
    return A2, cols


def train1(workers, steps=range(1, 10), models=('xgb', 'lgbm'), seeds=SEEDS):
    tasks = [(m, s, sd) for s in steps for m in models for sd in seeds]
    with mp.get_context('fork').Pool(workers) as pool:
        res = pool.map(_job1, tasks, chunksize=1)
    return res


def write1(boost, res, lockdir):
    lockdir.mkdir(parents=True, exist_ok=True)
    for m in ('xgb', 'lgbm'):
        rr = [r for r in res if r[0] == m]
        for sfx, f in (('', lambda r: r[2] == 42), ('_seeds4344', lambda r: r[2] in (43, 44))):
            v = pd.concat([r[4] for r in rr if f(r)], ignore_index=True).sort_values(['step', 'date', 'ticker'])
            t = pd.concat([r[5] for r in rr if f(r)], ignore_index=True).sort_values(['step', 'date', 'ticker'])
            v.to_parquet(R / f'{boost}_{m}_h1{sfx}_val.parquet', index=False)
            t.to_parquet(lockdir / f'{boost}_{m}_h1{sfx}_test.parquet', index=False)
    pd.DataFrame([dict(model=r[0], step=r[1], seed=r[2], trees=r[3], **r[6]) for r in res]).to_csv(OUTD / f'{boost}_trees_gain.csv', index=False)


# ---------------------------------------------------------------- бустинг h=5 (D12_h5 + jx) через драйвер Д24.8
def t5():
    import d24_8 as T
    T.D24 = OUTD
    T.EXT = JX
    T.add_ext = add_jx
    T.TAG = TAG
    return T


def stage_control(a):
    import l2_lib as L
    for h in a.h:
        if a.v0:
            import d12_h1 as D1
            A, FC = L.load_base(FEAT3)
            _, cols, _, _ = L.build_features(A, FC, 1, D1.CANDS['V0'])
            D1.G.update(A=A, grid=D1.grid(), cols={'V0': cols})
            t = [x for x in D1.plan(['single']) if x[0] == 'xgb' and x[1] == 2 and x[4] == 1][0]
            v = D1.job(('V0',) + t)[7]
        elif h == 1:
            matrix1(jx=False)
            res = train1(1, steps=[2], models=('xgb',), seeds=(42,))
            v = res[0][4]
        else:
            T = t5()
            A, cols, cfg = T.matrix(5, ext=False)
            res = T.train(5, A, cols, cfg, 1, steps=[2], models=('xgb',), seeds=(42,))
            v = res[0][6]
        ref = pd.read_parquet(R / f'{BASE[h]}_xgb_h{h}_val.parquet')
        ref = ref[(ref.step == 2) & (ref.seed == 42)] if 'seed' in ref else ref[ref.step == 2]
        v = v.assign(date=pd.to_datetime(v.date))
        ref = ref.assign(date=pd.to_datetime(ref.date))
        m = v.merge(ref, on=['date', 'ticker'], suffixes=('', '_ref'))
        d = float(np.max(np.abs(np.log(m.rv_pred) - np.log(m.rv_pred_ref))))
        log(f'контроль h={h}: строк {len(v)} / {len(ref)} / общих {len(m)}, max |Δ ln| = {d:.2e}')
        (OUTD / f'control_h{h}.txt').write_text(f'h={h} шаг 2 XGB seed 42: строк {len(v)} / {len(ref)}, общих {len(m)}, max|dln| {d:.3e}\n')


def stage_boost(a):
    for h in a.h:
        t0 = time.time()
        boost = f'{TAG}_h{h}'
        if a.v0:
            import d12_h1 as D1
            import l2_lib as L
            A, FC = L.load_base(FEAT3)
            A = add_jx(A)
            _, cols, _, _ = L.build_features(A, FC + JX, 1, D1.CANDS['V0'])
            assert all(c in cols for c in JX)
            D1.G.update(A=A, grid=D1.grid(), cols={'V0': cols})
            pl = D1.plan(['single', 'seeds3'])
            log(f'V0 h=1: признаков {len(cols)} (+{len(JX)}), задач {len(pl)}')
            with mp.get_context('fork').Pool(a.workers) as pool:
                res = pool.map(D1.job, [('V0',) + t for t in pl], chunksize=1)
            lock = R / 'test_locked' / boost
            lock.mkdir(parents=True, exist_ok=True)
            for m in ('xgb', 'lgbm'):
                rr = [r for r in res if r[1] == m]
                for sfx, f in (('', lambda x: x[5] == 1 and x[4] == 42), ('_seeds4344', lambda x: x[4] in (43, 44))):
                    pd.concat([x[7] for x in rr if f(x)], ignore_index=True).to_parquet(R / f'{boost}_{m}_h1{sfx}_val.parquet', index=False)
                    pd.concat([x[8] for x in rr if f(x)], ignore_index=True).to_parquet(lock / f'{boost}_{m}_h1{sfx}_test.parquet', index=False)
        elif h == 1:
            A, cols = matrix1(jx=True)
            log(f'h=1: признаков {len(cols)} (+{len(JX)}), строк {len(A)}')
            res = train1(a.workers)
            write1(boost, res, R / 'test_locked' / boost)
        else:
            T = t5()
            A, cols, cfg = T.matrix(5, ext=True)
            log(f'h=5: признаков {len(cols)} (+{len(JX)}), строк {len(A)}')
            res = T.train(5, A, cols, cfg, a.workers, legs=('single', 'seeds3'))
            T.write_legs(boost, 5, res, R / 'test_locked' / boost)
        log(f'h={h}: обучено {len(res)} моделей за {time.time() - t0:.0f} с')


# ---------------------------------------------------------------- h=22: нога Д16.5 (XGB single) + jx -> LGB-слот D19_1x / D19_4y
FEAT165 = R / 'features_pit_v3_d165.parquet'
D19 = R / 'D19'
C22 = {'D19_1x_h22': 'D19_1x_h22', 'D19_4y_h22': 'D19_4y_h22'}  # кандидат -> файлы XGB-слота в data/regen/D19 (без изменений)


def _job22(t):
    import l2_lib as L
    s, seed, jx = t
    cols = G['cols'] if jx else [c for c in G['cols'] if c not in JX]
    D = L.step_data(G['A'], cols, 22, s, 'xgb', {}, 'Bv2b')
    ref = L.make_ref('xgb', G['A'], cols, 22, {}, 'Bv2b') if s > 1 else None
    pv, pt, bi, gain, _ = L.fit('xgb', G['P'][s], D, {}, seed=seed, ref=ref)
    fr = []
    for part, pred in (('va', pv), ('te', pt)):
        d = D['parts'][part]
        fr.append(pd.DataFrame({'date': d['date'].values, 'ticker': d['ticker'].values, 'year': d['year'].values, 'step': s, 'seed': seed,
                                'rank': 1, 'rv_actual': d[L.tcol(22)].values, 'rv_pred': np.clip(np.exp(pred), 1e-10, None)}))
    g = dict(zip(D['names'], gain)) if gain is not None else {}
    return s, seed, bi, fr[0], fr[1], {c: g.get(c, 0.0) for c in JX}


def stage_b22(a):
    """Нога Д16.5 h=22 (XGB, точки D16_5_h22_selected_h22, матрица v3_d165, cfg {}) + 4 признака; seed 42/43/44; control — без jx, шаг 2."""
    import l2_lib as L
    from tune_L2 import SPACEB
    A, FC = L.load_base()
    A = add_jx(A)
    _, cols, _, _ = L.build_features(A, FC + JX, 22, {})
    S = pd.read_csv(R / 'D16_5_h22_selected_h22.csv')
    S = S[S.model == 'xgb'].set_index('step')
    P = {s: {k: (int(S.loc[s, k]) if sc in ('int', 'logint') else float(S.loc[s, k])) for k, _, _, sc in SPACEB['xgb']} for s in range(1, 10)}
    G.update(A=A, cols=cols, P=P)
    ctl = _job22((2, 42, False))[3]
    ref = pd.read_parquet(R / 'D16_5_h22_xgb_h22_val.parquet')
    ref = ref[ref.step == 2]
    m = ctl.assign(date=pd.to_datetime(ctl.date)).merge(ref.assign(date=pd.to_datetime(ref.date)), on=['date', 'ticker'], suffixes=('', '_r'))
    d = float(np.max(np.abs(np.log(m.rv_pred) - np.log(m.rv_pred_r))))
    log(f'контроль Д16.5 h=22 (шаг 2, seed 42, без jx): строк {len(ctl)} / {len(ref)} / общих {len(m)}, max |Δ ln| = {d:.2e}')
    (OUTD / 'control_d165_h22.txt').write_text(f'max|dln| {d:.3e}, общих {len(m)}\n')
    with mp.get_context('fork').Pool(a.workers) as pool:
        res = pool.map(_job22, [(s, sd, True) for s in range(1, 10) for sd in SEEDS], chunksize=1)
    lock = R / 'test_locked' / f'{TAG}_d165_h22'
    lock.mkdir(parents=True, exist_ok=True)
    for sfx, f in (('', lambda r: r[1] == 42), ('_seeds4344', lambda r: r[1] in (43, 44))):
        pd.concat([r[3] for r in res if f(r)], ignore_index=True).to_parquet(OUTD / f'{TAG}_d165_xgb_h22{sfx}_val.parquet', index=False)
        pd.concat([r[4] for r in res if f(r)], ignore_index=True).to_parquet(lock / f'{TAG}_d165_xgb_h22{sfx}_test.parquet', index=False)
    pd.DataFrame([dict(step=r[0], seed=r[1], trees=r[2], **r[5]) for r in res]).to_csv(OUTD / f'{TAG}_d165_h22_trees_gain.csv', index=False)
    log('Д16.5 h=22 + jx: обучено', len(res))


def stage_d19(a):
    """Сборка D25_7_D19_1x_h22 / D25_7_D19_4y_h22: XGB-слот — сохранённый файл кандидата (без изменений), LGB-слот — Д16.5 + jx (seed 42)."""
    K = ['date', 'ticker', 'step']
    c = K + ['year', 'rv_actual', 'rv_pred']
    lv = pd.read_parquet(OUTD / f'{TAG}_d165_xgb_h22_val.parquet')
    lt = pd.read_parquet(R / 'test_locked' / f'{TAG}_d165_h22' / f'{TAG}_d165_xgb_h22_test.parquet')
    for d in (lv, lt):
        d['date'] = pd.to_datetime(d['date'])
    for cand, src in C22.items():
        boost = f'{TAG}_{cand}'
        lock = R / 'test_locked' / boost
        lock.mkdir(parents=True, exist_ok=True)
        xv = pd.read_parquet(D19 / f'{src}_xgb_h22_val.parquet')
        xt = pd.read_parquet(R / 'test_locked' / 'D19_h22' / f'{src}_xgb_h22_test.parquet')
        for x, l_, dst, k in ((xv, lv, OUTD, 'val'), (xt, lt, lock, 'test')):
            x = x.assign(date=pd.to_datetime(x['date']))
            g = x[K + ['year', 'rv_actual']].merge(l_[K + ['rv_pred']], on=K, how='left')
            assert len(g) == len(x), (cand, k)
            miss = int(g.rv_pred.isna().sum())
            log(f'{boost} {k}: строк {len(g)}, без ноги Д16.5+jx {miss}')
            x[c].to_parquet(dst / f'{boost}_xgb_h22_{k}.parquet', index=False)
            g[c].to_parquet(dst / f'{boost}_lgbm_h22_{k}.parquet', index=False)


def stage_gib22(a):
    for cand in C22:
        boost = f'{TAG}_{cand}'
        od = OUTD / 'gibrid' / boost
        od.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, '-W', 'ignore', str(HERE / 'gibrid_v17.py'), '--boost', boost, '--h', '22', '--features', str(FEAT3),
               '--regen', str(OUTD), '--legs', 'single', '--out', str(od), '--devlog', str(OUTD / f'gibrid_devlog_{boost}.md')]
        env = dict(os.environ, GIBRID_SAME_SLOT='1', GIBRID_TEST_DIR=str(R / 'test_locked' / boost))
        with open(OUTD / f'gibrid_{boost}.log', 'w') as f:
            rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=str(HERE.parent.parent), env=env).returncode
        log(f'гибрид {boost}: код {rc}')


def stage_score22(a):
    """Правило третьего круга для D25_7_D19_*: представитель против той же конструкции без рычага (final3/designs_rank_<cand>_h22.csv);
    2021 — против текущего лучшего h=22 (финал: HAR-J с поправкой sm, 0,1453); порог по seed — нога Д16.5 XGB single (с jx против без), sm 2017-2020."""
    import final3 as F3
    import l2_lib as L
    rows = []
    a_ = pd.read_parquet(OUTD / f'{TAG}_d165_xgb_h22_val.parquet'); b_ = pd.read_parquet(OUTD / f'{TAG}_d165_xgb_h22_seeds4344_val.parquet')
    a0 = pd.read_parquet(R / 'D16_5_h22_xgb_h22_val.parquet'); b0 = pd.read_parquet(R / 'D16_5_h22_xgb_h22_seeds4344_val.parquet')
    sq, sb = {}, {}
    for sd in SEEDS:
        for src, dst, aa, bb in ((a_, sq, a_, b_), (a0, sb, a0, b0)):
            v = aa if sd == 42 else bb
            v = v[(v.seed == sd) & (v.step <= 6)]
            dst[sd] = L.evaluate(v, lo=2, hi=5)['sm'][0]
    for cand in C22:
        boost = f'{TAG}_{cand}'
        D = F3.designs(dict(gib_dir=OUTD / 'gibrid' / boost, gboost=boost, h=22))
        S = F3.summarize(dict(), D)
        S.to_csv(OUTD / f'designs_rank_{boost}.csv', index=False)
        Bz = pd.read_csv(R / 'final3' / f'designs_rank_{cand}_h22.csv')
        bb = Bz.iloc[0]
        x = S.iloc[0]
        r = dict(rychag='Д25.7', h=22, boost=boost, kind='лучший по правилу (Б)', design=x.design, n_failed=int(x.n_failed), failed=x.failed,
                 mean_1720=x.mean_1720, q2021=x.q2021, base=cand, base_design=bb.design, base_n_failed=int(bb.n_failed), base_1720=bb.mean_1720,
                 base_2021=bb.q2021, inc_2021=0.1452833147573684, pass_2021=bool(x.q2021 <= 0.1452833147573684), d_vs_base=x.mean_1720 / bb.mean_1720 - 1,
                 etalon_1720=0.23574941260180482, **{f'seed{s}': sq[s] for s in sq}, **{f'base_seed{s}': sb[s] for s in sb})
        r['seed_mean_d'] = np.mean(list(sq.values())) / np.mean(list(sb.values())) - 1
        r['seed_all_better'] = all(sq[s] < sb[s] for s in sq)
        r['porog_3seed'] = bool(r['seed_mean_d'] <= -0.01 and r['seed_all_better'])
        r['proshel'] = bool(r['pass_2021'] and r['n_failed'] <= r['base_n_failed'] and x.mean_1720 < bb.mean_1720 and r['porog_3seed'])
        rows.append(r)
    E = pd.DataFrame(rows)
    E.to_csv(OUTD / f'eval_{TAG}_h22.csv', index=False)
    with pd.option_context('display.width', 250, 'display.max_columns', 40):
        print(E.round(4).to_string())


# ---------------------------------------------------------------- гибрид, правило, yaml
def stage_gibrid(a):
    for h in a.h:
        boost = f'{TAG}_h{h}'
        od = OUTD / 'gibrid' / boost
        od.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, '-W', 'ignore', str(HERE / 'gibrid_v17.py'), '--boost', boost, '--h', str(h), '--features', str(FEAT3),
               '--regen', str(R), '--legs', *LEGS[h], '--out', str(od), '--devlog', str(OUTD / f'gibrid_devlog_{boost}.md')]
        env = dict(os.environ, GIBRID_SAME_SLOT='1')
        with open(OUTD / f'gibrid_{boost}.log', 'w') as f:
            rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=str(HERE.parent.parent), env=env).returncode
        log(f'гибрид {boost}: код {rc}')


def seed_q(boost, h, m='xgb'):
    """OOS QLIKE 2017-2020 (шаги 2-5) одиночной точки V0 с Б0.1 sm по каждому seed (читаются только шаги <= 6)."""
    import l2_lib as L
    a_ = pd.read_parquet(R / f'{boost}_{m}_h{h}_val.parquet')
    b_ = pd.read_parquet(R / f'{boost}_{m}_h{h}_seeds4344_val.parquet')
    out = {}
    for sd in SEEDS:
        v = a_ if sd == 42 else b_
        v = v[(v.seed == sd) & (v.step <= 6)] if 'seed' in v else v[v.step <= 6]
        if 'rank' in v:
            v = v[v['rank'] == 1]
        out[sd] = L.evaluate(v, lo=2, hi=5)['sm'][0]
    return out


def stage_eval(a):
    import final3 as F3
    rows = []
    for h in a.h:
        boost = f'{TAG}_h{h}'
        D = F3.designs(dict(gib_dir=OUTD / 'gibrid' / boost, gboost=boost, h=h))
        S = F3.summarize(dict(), D)
        S.to_csv(OUTD / f'designs_rank_{boost}.csv', index=False)
        Bz = pd.read_csv(R / 'final3' / f'designs_rank_{BASE[h]}_h{h}.csv')
        bb = Bz.iloc[0]
        best = S.iloc[0]
        sq, sb = seed_q(boost, h), seed_q(BASE[h], h)
        inc = INC[h]
        hyb = S[~S.design.str.contains(r'\|V0_')]
        kinds = [('лучший по правилу', best), ('V0_XGB (А, бустинг сам по себе)', S[S.design.str.contains(r'\|V0_XGB\|')].iloc[0])]
        if len(hyb):
            kinds.append(('лучший с ногой HAR (Б, нога гибрида)', hyb.iloc[0]))
        for kind, x in kinds:
            same = Bz[Bz.design == x.design]
            r = dict(rychag='Д25.7', h=h, boost=boost, kind=kind, design=x.design, n_failed=int(x.n_failed), failed=x.failed,
                     **{f'q_{y}': x[f'q_{y}'] for y in range(2017, 2022)}, mean_1720=x.mean_1720, q2021=x.q2021,
                     base=BASE[h], base_design=bb.design, base_n_failed=int(bb.n_failed), base_1720=bb.mean_1720, base_2021=bb.q2021,
                     base_same_design_1720=(same.iloc[0].mean_1720 if len(same) else np.nan),
                     base_same_design_2021=(same.iloc[0].q2021 if len(same) else np.nan),
                     inc_2021=inc[3], pass_2021=bool(x.q2021 <= inc[3]), d_vs_base=x.mean_1720 / bb.mean_1720 - 1,
                     **{f'seed{s}': sq[s] for s in sq}, **{f'base_seed{s}': sb[s] for s in sb})
            r['seed_mean_d'] = np.mean(list(sq.values())) / np.mean(list(sb.values())) - 1
            r['seed_all_better'] = all(sq[s] < sb[s] for s in sq)
            r['porog_3seed'] = bool(r['seed_mean_d'] <= -0.01 and r['seed_all_better'])
            r['proshel'] = bool(kind.startswith('лучший') and r['pass_2021'] and r['n_failed'] <= r['base_n_failed']
                                and x.mean_1720 < bb.mean_1720 and r['porog_3seed'])
            rows.append(r)
    E = pd.DataFrame(rows)
    p = OUTD / f'eval_{TAG}_h{"_".join(map(str, a.h))}.csv'
    E.to_csv(p, index=False)
    with pd.option_context('display.width', 250, 'display.max_columns', 40):
        print(E[['h', 'kind', 'design', 'n_failed', 'failed', 'mean_1720', 'q2021', 'base_1720', 'base_2021', 'base_same_design_1720', 'd_vs_base',
                 'pass_2021', 'seed42', 'seed43', 'seed44', 'base_seed42', 'base_seed43', 'base_seed44', 'seed_mean_d', 'porog_3seed',
                 'proshel']].round(4).to_string())


YAML = {1: dict(semeystvo='гибрид Д8.12 на HAR-X⁺ (Д22, слой 2)', osnova='D22_h1',
                opisanie='D22_h1 + признаки скачков (число дней с J>0 за 5/22, дней с последнего скачка, сумма J за 22; к дню t включительно), точки D22_h1 без нового подбора; ноги single/seeds3'),
        5: dict(semeystvo='V9 на HAR-J', osnova='D12_h5',
                opisanie='D12_h5 + признаки скачков (число дней с J>0 за 5/22, дней с последнего скачка, сумма J за 22; к дню t включительно), точки D12_h5; нога wrap2 без 1a')}


def stage_yaml(a):
    for h in a.h:
        E = pd.read_csv(OUTD / f'eval_{TAG}_h{h}.csv')
        e = E[E.kind.str.startswith('лучший')].iloc[0]
        boost = f'{TAG}_h{h}'
        if not e.proshel:
            log(f'h={h}: не прошёл — yaml не пишется')
            continue
        info = YAML[h]
        y = (f"boost: {boost}\nh: {h}\nsemeystvo: \"{info['semeystvo']}\"\nrychag: Д25.7\nopisanie: \"{info['opisanie']}\"\nregen: data/regen\n"
             f"legs: [{', '.join(LEGS[h])}]\nodinochnaya: false\nchistyy: false\ntest_dir: data/regen/test_locked/{boost}\nosnova: {info['osnova']}\nn_variantov: 1\n")
        d = R / 'final3' / 'kandidaty'
        (d / f'{boost}.yaml').write_text(y)
        log('yaml ->', d / f'{boost}.yaml')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', required=True, choices=['feat', 'harx', 'control', 'boost', 'gibrid', 'eval', 'score', 'yaml', 'b22', 'd19', 'gib22', 'score22'])
    ap.add_argument('--h', type=int, nargs='+', default=[1])
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--harj', action='store_true', help='harx: HAR-J + только слагаемые скачков')
    ap.add_argument('--v0', action='store_true', help='строка «бустинг»: чистый бустинг D12_h1_V0 (cfg {}, точки Bv2b) + 4 признака, h=1')
    a = ap.parse_args()
    if a.v0:
        global TAG
        TAG = 'D25_7_V0'
        BASE[1] = 'D12_h1_V0'
        YAML[1] = dict(semeystvo='чистый бустинг', osnova='D12_h1_V0',
                       opisanie='D12_h1_V0 (XGBoost/LightGBM на ln RV, точки ЛГК B v2b) + признаки скачков (число дней с J>0 за 5/22, дней с последнего скачка, сумма J за 22); ноги single/seeds3')
    OUTD.mkdir(parents=True, exist_ok=True)
    globals()[f'stage_{"eval" if a.stage == "score" else a.stage}'](a)


if __name__ == '__main__':
    if 'b22' in sys.argv:
        os.environ['L2_FEAT'] = str(FEAT165)  # матрица Д16.5 (v3 + регрессоры HAR-X) и её config.json
    elif '--v0' not in sys.argv and '--h' in sys.argv and sys.argv[sys.argv.index('--h') + 1] == '1' and any(s in sys.argv for s in ('control', 'boost')):
        import d22_h1  # noqa: F401  — L2_FEAT = features_pit_v3_rrv до первого импорта l2_lib
    else:
        os.environ.setdefault('L2_FEAT', str(FEAT3))
    import numpy as np, pandas as pd
    main()
else:
    import numpy as np, pandas as pd
