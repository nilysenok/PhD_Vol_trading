#!/usr/bin/env python3
"""R2, агент А1 (протокол docs/zadaniya/2026-10-26_vtoroy_raund_gibrida.md): кэш бустинга h=1 на 10 seed (42–51).
Ноги: D22_h1, D16_5_h1, D16_h1_d812x, D24_10_h1 × {xgb, lgbm} × {single, top5}; производные seeds3 / seeds10; F_1 (D22_h1 top5/V0_XGB/B-sm/L0) по seed.
Гиперпараметры не переподбираются: точки, cfg, матрицы и механизм ноги — сохранённые первого раунда (yaml models/designs/r2/r2_A1_kesh_h1.yaml).
Обучение — l2_lib.fit тем же путём, что в первом раунде (step_data, бины окна шага 1 для шагов >= 2, ES на val-годе, patience 30, nthread 1).
top5 с seed s — пять точек top5 шага (шаги 1–6), каждая с seed s; на шагах 7–10 — точка шага 6. Шаг 10 — обучение <= 2024, val 2025, тест 2026.
Выход: val шагов 1–6 -> data/regen/R2/cache/A1/<нога>_h1_s<seed>_val.parquet (date, ticker, year, step, rv_actual, rv_pred);
val шагов 7–10 и тест шагов 1–10 -> data/regen/test_locked/R2/cache/A1/<нога>_h1_s<seed>_test.parquet (+ столбец part: val | test), SHA256.txt.
Ветка test-build (сборка *_test, F_1 на тестовых шагах) читает только своё сырьё в test_locked и ничего не печатает из метрик.
Метрики — только шаги 2–6 (контроль F_1 seed 42 с designs_val первого раунда).
Команды:
  zamer  --boost B [--model xgb] [--step 5]   шаг 0: одна нога × один seed × один шаг (время; сверка с сохранённым рядом)
  leak   --boost B                            детектор «будущее не влияет» на новом коде (шаги yaml leak_steps, xgb/lgbm)
  run    --boost B [--workers N]              счёт (возобновляемый, сырьё в _raw/), сборка ног, контроль с сохранёнными рядами <= 1e-7
  derive                                      seeds3 / seeds10, F_1 по seed, INDEX_A1.csv, SHA256.txt
  chain  [--workers N] [--boosts ...]         leak + run по порядку yaml, затем derive (каждая нога — свой процесс)
python3 scripts/analysis/r2_kesh_h1.py chain --workers 3"""
import os, sys, time, json, hashlib, argparse, subprocess
from pathlib import Path

for _k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_k] = '1'
HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
ROOT = Path('/Users/nilysenok/Desktop/Старое/PhD_Vol_trading')
R = ROOT / 'data' / 'regen'
YAML = REPO / 'models' / 'designs' / 'r2' / 'r2_A1_kesh_h1.yaml'
import yaml
Y = yaml.safe_load(open(YAML))
BOOST = sys.argv[sys.argv.index('--boost') + 1] if '--boost' in sys.argv else None
if BOOST:
    os.environ['L2_FEAT'] = str(R / Y['boosts'][BOOST]['matrix'])  # до импорта l2_lib (FEAT читается при импорте)
sys.path.insert(0, str(HERE))
import numpy as np, pandas as pd

H = int(Y['h'])
M = tuple(Y['models'])
SEEDS = tuple(Y['seeds'])
STEPS = tuple(Y['steps'])
T5 = tuple(Y['top5_steps'])
CACHE = ROOT / Y['paths']['cache']
LOCK = ROOT / Y['paths']['locked']
INDEX = ROOT / Y['paths']['index']
TOL = float(Y['control_tol'])
COLS = ['date', 'ticker', 'year', 'step', 'rv_actual', 'rv_pred']
KEY = ['date', 'ticker', 'step']
STOP = CACHE / 'STOP_A1.txt'


def log(*a):
    print(time.strftime('%F %T'), *a, flush=True)


def sha_file(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


CODE_SHA = sha_file(Path(__file__).resolve())
YAML_SHA = sha_file(YAML)


def put(df, p):
    """Атомарная запись parquet."""
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + '.tmp')
    df.to_parquet(tmp, index=False)
    os.replace(tmp, p)


# ---------------------------------------------------------------- настройка ноги (только в процессе своей ноги)
G = {}


def _patch_split10(L):
    """Шаг 10 как final2_step10.py / d16_v9.py: обучение <= 2024, val 2025, тест 2026 (цель в конце ряда может отсутствовать)."""
    s0 = L.split

    def split(A, h, step):
        if step != 10:
            return s0(A, h, step)
        t = L.tcol(h)
        tr = A[A.year < 2026]
        return tr[tr.year != 2025].dropna(subset=[t]), tr[tr.year == 2025].dropna(subset=[t]), A[A.year == 2026], 2025, 2026
    L.split = split


def _pts_grid(tag):
    """tune_L2: сетка {tag}_grid.csv; top5 шага — {tag}_{m}_h1_top5_val (point); шаги 7–10 — точка шага 6 ({tag}_selected_h1)."""
    GR = pd.read_csv(R / f'{tag}_grid.csv').set_index(['model', 'point'])
    S = pd.read_csv(R / f'{tag}_selected_h{H}.csv')
    out = {}
    for m in M:
        T = pd.read_parquet(R / f'{tag}_{m}_h{H}_top5_val.parquet', columns=['step', 'rank', 'point']).groupby(['step', 'rank']).point.first()
        sp = {s: int(S[(S.model == m) & (S.step == s)].point.iloc[0]) for s in range(1, 10)}
        for s in STEPS:
            if s in T5:
                assert int(T[(s, 1)]) == sp[s]
                out[(m, s)] = [(GR.loc[(m, int(T[(s, r)]))].to_dict(), r, int(T[(s, r)])) for r in range(1, 6)]
            else:
                assert s == 10 or sp[s] == sp[6]
                out[(m, s)] = [(GR.loc[(m, sp[6])].to_dict(), 1, sp[6])]
    return out


def _pts_optuna(tag):
    """tune_L2_optuna: параметры испытаний {tag}_trials.csv; top5 — {tag}_{m}_h1_top5_val (trial); шаги 7–10 — испытание шага 6."""
    from tune_L2 import SPACEB
    TR = pd.read_csv(R / f'{tag}_trials.csv').set_index(['model', 'step', 'trial'])
    S = pd.read_csv(R / f'{tag}_selected_h{H}.csv')
    prm = lambda m, s, t: {k: (int(TR.loc[(m, s, t), k]) if sc in ('int', 'logint') else float(TR.loc[(m, s, t), k])) for k, _, _, sc in SPACEB[m]}
    out = {}
    for m in M:
        T = pd.read_parquet(R / f'{tag}_{m}_h{H}_top5_val.parquet', columns=['step', 'rank', 'trial']).groupby(['step', 'rank']).trial.first()
        sp = {s: int(S[(S.model == m) & (S.step == s)].trial.iloc[0]) for s in range(1, 10)}
        for s in STEPS:
            if s in T5:
                assert int(T[(s, 1)]) == sp[s]
                out[(m, s)] = [(prm(m, s, int(T[(s, r)])), r, int(T[(s, r)])) for r in range(1, 6)]
            else:
                assert s == 10 or sp[s] == sp[6]
                out[(m, s)] = [(prm(m, 6, sp[6]), 1, sp[6])]
    return out


def _pts_bv2b():
    """ЛГК B v2b (d12_h1.grid / plan): top5 — Bv2b_{m}_h1_top5_val (point); шаги 7–10 — точка шага 6 (Bv2b_selected_h1)."""
    import d12_h1 as D1
    grid = D1.grid()
    S = pd.read_csv(R / 'Bv2b_selected_h1.csv')
    out = {}
    for m in M:
        T = pd.read_parquet(R / f'Bv2b_{m}_h1_top5_val.parquet', columns=['step', 'rank', 'point']).groupby(['step', 'rank']).point.first()
        sp = {s: int(S[(S.model == m) & (S.step == s)].point.iloc[0]) for s in range(1, 10)}
        for s in STEPS:
            if s in T5:
                assert int(T[(s, 1)]) == sp[s]
                out[(m, s)] = [(grid[(m, int(T[(s, r)]))], r, int(T[(s, r)])) for r in range(1, 6)]
            else:
                assert s == 10 or sp[s] == sp[6]
                out[(m, s)] = [(grid[(m, sp[6])], 1, sp[6])]
    return out


def setup(boost):
    """-> G: A (матрица с признаками), cols, cfg, btag, P {(m, шаг): [(params, rank, id точки)]}, L (модуль l2_lib с подменами ноги)."""
    t0 = time.time()
    b = Y['boosts'][boost]
    cfg = dict(b['cfg'])
    if boost == 'D22_h1':
        import d22_h1 as P22  # подмены l2_lib: har_ln -> ln HAR-X⁺ шага, split шага 10, признаки слоя 2, load_base + HXP/HJ
        import l2_lib as L
        assert P22.CFG == cfg
        P22.patch()
        A, FC = L.load_base()
        A2, cols, _, _ = L.build_features(A, FC, H, cfg)
        P = _pts_grid('D22_h1')
    elif boost == 'D16_5_h1':
        import l2_lib as L
        _patch_split10(L)
        A, FC = L.load_base(None, (), H)
        A2, cols, _, _ = L.build_features(A, FC, H, cfg)
        P = _pts_optuna('D16_5_h1')
    elif boost == 'D16_h1_d812x':
        import l2_lib as L
        import d16_v9  # noqa: F401  подмены l2_lib: har_ln -> ln HAR-X шага (столбцы hx{k}), split шага 10
        import harx as HX
        A, FC = L.load_base()
        A2, cols, _, _ = L.build_features(A, FC, H, cfg)
        hx = HX.ln_all_steps(H, HX.default_terms(H, True))  # как d24_3.setup (mech harx_feat)
        A2 = A2.merge(hx, on=['date', 'ticker'], how='left', validate='one_to_one')
        assert all(f'hx{k}' in A2 for k in STEPS)
        P = _pts_bv2b()
    elif boost == 'D24_10_h1':
        import l2_lib as L
        _patch_split10(L)
        A, FC = L.load_base(R / b['matrix'])
        A2, cols, _, _ = L.build_features(A, FC, H, cfg)
        P = _pts_bv2b()
    else:
        raise SystemExit(f'неизвестная нога {boost}')
    assert Path(L.FEAT).name == b['matrix'], (L.FEAT, b['matrix'])
    G.update(A=A2, cols=cols, cfg=cfg, btag=b['btag'], P=P, L=L, boost=boost)
    log(f'{boost}: матрица {Path(L.FEAT).name}, признаков {len(cols)}, строк {len(A2)}, cfg {cfg}, btag {b["btag"]}, настройка {time.time() - t0:.0f} с')
    return G


# ---------------------------------------------------------------- единица счёта: (модель, шаг, seed) -> все ранги шага
_DC, _RC = {}, {}


def _data(m, s):
    k = (m, s)
    if k not in _DC:
        if len(_DC) >= 2:
            _DC.clear()
        _DC[k] = G['L'].step_data(G['A'], G['cols'], H, s, m, G['cfg'], G['btag'])
    return _DC[k]


def _ref(m, s):
    if s == 1:
        return None
    if m not in _RC:
        _RC[m] = G['L'].make_ref(m, G['A'], G['cols'], H, G['cfg'], G['btag'])
    return _RC[m]


def fit_unit(m, s, seed, items):
    """-> (val DataFrame, test DataFrame) со столбцами COLS + rank, pid, seed, trees, sec."""
    L = G['L']
    D = _data(m, s)
    ref = _ref(m, s)
    fv, ft = [], []
    for p, rank, pid in items:
        t0 = time.time()
        pv, pt, bi, _, _ = L.fit(m, p, D, G['cfg'], seed=seed, ref=ref)
        sec = time.time() - t0
        for part, pred, yr, acc in (('va', pv, D['vy'], fv), ('te', pt, D['ty'], ft)):
            d = D['parts'][part]
            acc.append(pd.DataFrame({'date': d['date'].values, 'ticker': d['ticker'].values, 'year': yr, 'step': s,
                                     'rv_actual': d[L.tcol(H)].values, 'rv_pred': np.clip(np.exp(pred), 1e-10, None),
                                     'rank': rank, 'pid': pid, 'seed': seed, 'trees': bi, 'sec': sec}))
    return pd.concat(fv, ignore_index=True), pd.concat(ft, ignore_index=True)


def raw_paths(boost, m, seed, s):
    n = f'{m}_s{seed}_st{s:02d}.parquet'
    return CACHE / '_raw' / boost / n, LOCK / '_raw' / boost / n


def done(boost, m, seed, s):
    pv, pt = raw_paths(boost, m, seed, s)
    return pt.exists() and (s > 6 or pv.exists())


def job(t):
    m, s, seed = t
    v, te = fit_unit(m, s, seed, G['P'][(m, s)])
    pv, pt = raw_paths(G['boost'], m, seed, s)
    if s <= 6:
        put(v, pv)
        lk = te.assign(part='test')
    else:
        lk = pd.concat([v.assign(part='val'), te.assign(part='test')], ignore_index=True)
    put(lk, pt)
    return m, s, seed, int(v.trees.iloc[0]), float(v.sec.sum()), len(v)


# ---------------------------------------------------------------- сборка ног и контроль
def leg_name(boost, m, kind):
    return f'{boost}_{m}_{kind}'


def fpath(noga, seed, part):
    sd = 'det' if seed is None else str(seed)
    return (CACHE if part == 'val' else LOCK) / f'{noga}_h{H}_s{sd}_{part}.parquet'


def _avg(df, keys):
    return df.groupby(keys, as_index=False, sort=False).agg(year=('year', 'first'), rv_actual=('rv_actual', 'first'), rv_pred=('rv_pred', 'mean'))


def assemble(boost, m, seed):
    """Сырьё (модель, seed) шагов 1–10 -> ноги single / top5: val шагов 1–6 и *_test (val 7–10 + тест 1–10; test-build, без метрик)."""
    V = pd.concat([pd.read_parquet(raw_paths(boost, m, seed, s)[0]) for s in STEPS if s <= 6], ignore_index=True)
    T = pd.concat([pd.read_parquet(raw_paths(boost, m, seed, s)[1]) for s in STEPS], ignore_index=True)
    srt = lambda d, k: d.sort_values(k).reset_index(drop=True)
    out = {}
    out[(leg_name(boost, m, 'single'), 'val')] = srt(V[V['rank'] == 1][COLS], KEY[::-1])
    out[(leg_name(boost, m, 'top5'), 'val')] = srt(_avg(V, KEY)[COLS], KEY[::-1])
    TC = COLS + ['part']
    out[(leg_name(boost, m, 'single'), 'test')] = srt(T[T['rank'] == 1][TC], ['part', 'step', 'date', 'ticker'])
    t5 = (T.part == 'test') & T.step.isin(T5)
    out[(leg_name(boost, m, 'top5'), 'test')] = srt(pd.concat([_avg(T[t5], ['part'] + KEY), T[~t5 & (T['rank'] == 1)]], ignore_index=True)[TC],
                                                    ['part', 'step', 'date', 'ticker'])
    for (noga, part), df in out.items():
        assert df.rv_pred.notna().all() and len(df.drop_duplicates(['part', *KEY] if part == 'test' else KEY)) == len(df)
        if part == 'val':
            assert sorted(df.step.unique()) == [s for s in STEPS if s <= 6]
        else:
            assert sorted(df[df.part == 'val'].step.unique()) == [s for s in STEPS if s > 6] and sorted(df[df.part == 'test'].step.unique()) == list(STEPS)
        p = fpath(noga, seed, part)
        put(df, p)
        index_add(p, noga, seed, df)
    return V


def control(boost, m, seed, V):
    """seed s повторяет сохранённый ряд первого раунда (шаги 1–6): max |Δ ln rv_pred| <= TOL, иначе STOP."""
    sv = Y['boosts'][boost]['saved']
    rows = []
    checks = []
    if seed == 42:
        checks.append(('single', sv['single'], 1))
        checks.append(('top5', sv['top5'], None))
    elif seed in (43, 44):
        checks.append(('s4344', sv['s4344'], 1))
    elif 's4551' in sv:
        checks.append(('s4551', sv['s4551'], 1))
    for kind, f, rk in checks:
        S = pd.read_parquet(R / f.format(m=m))
        S = S[S.step <= 6]
        if 'seed' in S:
            S = S[S.seed == seed]
        mine = V if rk is None else V[V['rank'] == rk]
        k = KEY + (['rank'] if rk is None else [])
        if rk is not None and 'rank' in S:
            S = S[S['rank'] == rk]
        mm = mine[k + ['rv_pred']].merge(S[k + ['rv_pred']], on=k, how='outer', suffixes=('', '_saved'), indicator=True)
        both = mm[mm._merge == 'both']
        d = float(np.abs(np.log(both.rv_pred) - np.log(both.rv_pred_saved)).max()) if len(both) else np.inf
        ok = bool(len(both) == len(mm) and d <= TOL)
        rows.append(dict(boost=boost, model=m, seed=seed, kind=kind, saved=f.format(m=m), rows=len(mm), matched=len(both), max_abs_dln=d, ok=ok,
                         time=time.strftime('%F %T')))
        log(f'контроль {boost} {m} s{seed} {kind}: строк {len(both)}/{len(mm)}, max|Δ ln| = {d:.2e} {"OK" if ok else "ПРОВАЛ"}')
    if rows:
        p = CACHE / 'control_A1.csv'
        df = pd.DataFrame(rows)
        if p.exists():
            df = pd.concat([pd.read_csv(p), df], ignore_index=True)
        df.to_csv(p, index=False)
        bad = [r for r in rows if not r['ok']]
        if bad:
            STOP.write_text(f'{time.strftime("%F %T")} контроль не пройден: {bad}\n')
            raise SystemExit(f'СТОП: контроль с сохранённым рядом не пройден ({boost} {m} s{seed})')


# ---------------------------------------------------------------- индекс и SHA256
def index_add(p, noga, seed, df):
    steps = sorted(df.step.unique()) if 'part' not in df else sorted(df[df.part == 'val'].step.unique())
    st = f'{min(steps)}-{max(steps)}' if 'part' not in df else f'val {min(steps)}-{max(steps)}; test {int(df[df.part == "test"].step.min())}-{int(df[df.part == "test"].step.max())}'
    row = dict(файл=str(p.relative_to(ROOT)), нога=noga, h=H, seed='det' if seed is None else seed, шаги=st, строки=len(df), sha256=sha_file(p),
               sha_кода=CODE_SHA, yaml=f'{YAML.relative_to(REPO)}@{YAML_SHA[:16]}', время=time.strftime('%F %T'))
    I = pd.read_csv(INDEX) if INDEX.exists() else pd.DataFrame(columns=list(row))
    I = pd.concat([I[I['файл'] != row['файл']], pd.DataFrame([row])], ignore_index=True).sort_values('файл')
    tmp = INDEX.with_name(INDEX.name + '.tmp')
    I.to_csv(tmp, index=False)
    os.replace(tmp, INDEX)


def sha_locked():
    fs = sorted(p for p in LOCK.glob('*_test.parquet'))
    tmp = LOCK / 'SHA256.txt.tmp'
    tmp.write_text(''.join(f'{sha_file(p)}  {p.name}\n' for p in fs))
    os.replace(tmp, LOCK / 'SHA256.txt')
    return len(fs)


# ---------------------------------------------------------------- детектор «будущее не влияет»
def corrupt(A, t0, seed=7):
    """Всё с даты t0 включительно: числовые столбцы × exp(N(0,1)), цели перемешаны (как d24_1.corrupt)."""
    rng = np.random.default_rng(seed)
    A = A.copy()
    msk = (pd.to_datetime(A['date']) >= pd.Timestamp(t0)).values
    num = [c for c in A.columns if c not in ('date', 'ticker', 'year') and pd.api.types.is_float_dtype(A[c])]
    for c in num:
        v = A.loc[msk, c].values
        A.loc[msk, c] = rng.permutation(v) if 'target' in c else v * np.exp(rng.normal(size=len(v)))
    return A


def cmd_leak(a):
    setup(a.boost)
    A0 = G['A']
    rows = []
    for s in Y['leak_steps']:
        ty = 2026 if s == 10 else G['L'].TEST_YEARS[s - 1]
        for m in M:
            item = G['P'][(m, s)][:1]
            _DC.clear(); _RC.clear()
            base = fit_unit(m, s, 42, item)[0]
            for t0, must in ((f'{ty}-01-01', 'равны'), (f'{ty - 1}-01-01', 'изменились')):
                G['A'] = corrupt(A0, t0)
                _DC.clear(); _RC.clear()
                v = fit_unit(m, s, 42, item)[0]
                G['A'] = A0
                d = float(np.abs(np.log(v.rv_pred.values) - np.log(base.rv_pred.values)).max()) if len(v) == len(base) else np.inf
                ok = bool(d == 0.0) if must == 'равны' else bool(d > 0)
                rows.append(dict(boost=a.boost, model=m, step=s, noise_from=t0, expect=must, max_abs_dln=d, ok=ok, time=time.strftime('%F %T')))
                log(f'leak {a.boost} {m} шаг {s}: шум с {t0} -> max|Δ ln| val = {d:.3g} (ожидается: {must}) {"OK" if ok else "ПРОВАЛ"}')
    _DC.clear(); _RC.clear()
    p = CACHE / 'leak_A1.csv'
    p.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    if p.exists():
        df = pd.concat([pd.read_csv(p), df], ignore_index=True)
    df.to_csv(p, index=False)
    if not all(r['ok'] for r in rows):
        STOP.write_text(f'{time.strftime("%F %T")} детектор: провал {a.boost}\n')
        raise SystemExit('СТОП: детектор «будущее не влияет» не пройден')


# ---------------------------------------------------------------- шаг 0: замер
def cmd_zamer(a):
    t0 = time.time()
    setup(a.boost)
    ts = time.time() - t0
    m, s = a.model, a.step
    item = G['P'][(m, s)][:1]
    t1 = time.time()
    _data(m, s); _ref(m, s)
    tp = time.time() - t1
    t2 = time.time()
    v, _ = fit_unit(m, s, 42, item)
    tf = time.time() - t2
    S = pd.read_parquet(R / Y['boosts'][a.boost]['saved']['single'].format(m=m))
    S = S[(S.step == s)]
    S = S[S.seed == 42] if 'seed' in S else S
    mm = v.merge(S[KEY + ['rv_pred']], on=KEY, suffixes=('', '_saved'))
    d = float(np.abs(np.log(mm.rv_pred) - np.log(mm.rv_pred_saved)).max())
    r = dict(boost=a.boost, model=m, step=s, seed=42, setup_sec=round(ts, 1), data_ref_sec=round(tp, 1), fit_sec=round(tf, 1), trees=int(v.trees.iloc[0]),
             rows=len(v), matched=len(mm), saved_rows=len(S), max_abs_dln=d, time=time.strftime('%F %T'))
    log('замер:', json.dumps(r, ensure_ascii=False))
    p = CACHE / 'zamer_A1.csv'
    p.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([r])
    if p.exists():
        df = pd.concat([pd.read_csv(p), df], ignore_index=True)
    df.to_csv(p, index=False)


# ---------------------------------------------------------------- счёт
def cmd_run(a):
    import multiprocessing as mp
    if STOP.exists():
        raise SystemExit(f'СТОП-файл {STOP}: {STOP.read_text()}')
    setup(a.boost)
    boost = a.boost
    seeds = tuple(a.seeds) if a.seeds else SEEDS
    tasks = [(m, s, sd) for sd in seeds for m in M for s in sorted(STEPS, reverse=True) if not done(boost, m, sd, s)]
    need = {(m, sd) for m in M for sd in seeds}
    finished = {g for g in need if all(done(boost, g[0], g[1], s) for s in STEPS)}
    log(f'{boost}: задач {len(tasks)} (из {len(M) * len(seeds) * len(STEPS)}), воркеров {a.workers}; групп готово {len(finished)}/{len(need)}')

    def close(g):
        V = assemble(boost, g[0], g[1])
        control(boost, g[0], g[1], V)
        n = sha_locked()
        log(f'{boost} {g[0]} s{g[1]}: ноги собраны (single, top5), тестовых файлов {n}')

    for g in sorted(finished, key=lambda x: (x[1], x[0])):
        if not fpath(leg_name(boost, g[0], 'top5'), g[1], 'val').exists() or not fpath(leg_name(boost, g[0], 'top5'), g[1], 'test').exists():
            close(g)
    t0 = time.time()
    n = 0
    tot_fit = 0.0
    if tasks:
        with mp.get_context('fork').Pool(a.workers) as P:
            for m, s, sd, bi, sec, nv in P.imap_unordered(job, tasks, chunksize=1):
                n += 1
                tot_fit += sec
                el = time.time() - t0
                log(f'{boost} {m} шаг {s} s{sd}: деревьев(r1) {bi}, обучение {sec:.0f} с; {n}/{len(tasks)}, прошло {el / 60:.1f} мин, '
                    f'осталось ~{el / n * (len(tasks) - n) / 60:.0f} мин')
                g = (m, sd)
                if g not in finished and all(done(boost, m, sd, x) for x in STEPS):
                    finished.add(g)
                    close(g)
    log(f'{boost}: готово, задач {n}, суммарно обучения {tot_fit / 3600:.2f} ч, стена {(time.time() - t0) / 3600:.2f} ч')


# ---------------------------------------------------------------- производные: seeds3 / seeds10 / F_1
def _mean_seeds(boost, m, seeds, part):
    fr = [pd.read_parquet(fpath(leg_name(boost, m, 'single'), sd, part)) for sd in seeds]
    keys = (['part'] if part == 'test' else []) + KEY
    d = pd.concat(fr, ignore_index=True)
    out = _avg(d, keys)
    return out[COLS + (['part'] if part == 'test' else [])].sort_values((['part'] if part == 'test' else []) + ['step', 'date', 'ticker']).reset_index(drop=True)


def f1_series(seed):
    """F_1 seed s: нога D22_h1 xgb top5 seed s × exp(s²_sm/2) (gibrid_bias: val шага t — s² по val шагов t−3..t−1, шаг 1 — 1,0;
    тест шага t — по val шагов t−2..t; final2_build для шага 10). Остатки — строки rv_actual > 1e-12.
    -> (val шагов 1–6, *_test: val 7–10 + тест 1–10). Ветка test-build: метрики не считаются."""
    spec = Y['derived']['F1']
    v = pd.read_parquet(fpath(spec['leg'], seed, 'val'))
    t = pd.read_parquet(fpath(spec['leg'], seed, 'test'))
    allv = pd.concat([v.assign(part='val'), t[t.part == 'val']], ignore_index=True)
    ok = allv[allv.rv_actual > 1e-12]
    E = {int(st): np.log(g.rv_actual.values + 1e-10) - np.log(g.rv_pred.values) for st, g in ok.groupby('step')}
    assert sorted(E) == list(STEPS) and spec['code'] == 'sm'
    s2 = lambda st: float(np.var(np.concatenate([E[j] for j in range(max(1, st - 2), st + 1)])))
    Fv = {st: (1.0 if st == 1 else float(np.exp(s2(st - 1) / 2))) for st in STEPS}
    Ft = {st: float(np.exp(s2(st) / 2)) for st in STEPS}
    vv = v.assign(rv_pred=v.rv_pred.values * v.step.map(Fv).values)
    tt = t.assign(rv_pred=t.rv_pred.values * np.where(t.part == 'val', t.step.map(Fv), t.step.map(Ft)))
    return vv[COLS], tt[COLS + ['part']]


def q_dev(df):
    """QLIKE по шагам 2–6 (разработка 2017–2021; фильтр rv_actual > 1e-12). Шаги > 6 сюда не попадают по построению."""
    import l2_lib as L
    d = df[(df.step >= 2) & (df.step <= 6) & (df.rv_actual > 1e-12)]
    return {int(s): float(L.qlike(g.rv_actual.values, g.rv_pred.values)) for s, g in d.groupby('step')}


def cmd_derive(a):
    rows = []
    for boost in Y['order']:
        for m in M:
            for kind, seeds in (('seeds3', Y['derived']['seeds3']), ('seeds10', Y['derived']['seeds10'])):
                if not all(fpath(leg_name(boost, m, 'single'), sd, p).exists() for sd in seeds for p in ('val', 'test')):
                    continue
                for part in ('val', 'test'):
                    df = _mean_seeds(boost, m, seeds, part)
                    p = fpath(leg_name(boost, m, kind), None, part)
                    put(df, p)
                    index_add(p, leg_name(boost, m, kind), None, df)
                rows.append(f'{boost} {m} {kind}')
    spec = Y['derived']['F1']
    for sd in SEEDS:
        if not (fpath(spec['leg'], sd, 'val').exists() and fpath(spec['leg'], sd, 'test').exists()):
            continue
        vv, tt = f1_series(sd)
        if sd == 42:  # контроль: QLIKE F_1 seed 42 = designs_val первого раунда (шаги 2–5 и 6)
            q = q_dev(vv)
            c = spec['control']
            m1720, q21 = float(np.mean([q[s] for s in (2, 3, 4, 5)])), q[6]
            dd = max(abs(m1720 - c['mean_1720']), abs(q21 - c['q2021']))
            log(f'контроль F_1 s42: 2017–2020 {m1720:.10f} (FD {c["mean_1720"]:.10f}), 2021 {q21:.10f} (FD {c["q2021"]:.10f}), max|Δ| {dd:.2e}')
            if dd > 1e-9:
                STOP.write_text(f'{time.strftime("%F %T")} контроль F_1 не пройден: {dd}\n')
                raise SystemExit('СТОП: F_1 seed 42 не повторяет первый раунд')
        for part, df in (('val', vv), ('test', tt)):
            p = fpath('F1', sd, part)
            put(df, p)
            index_add(p, 'F1', sd, df)
        rows.append(f'F1 s{sd}')
    n = sha_locked()
    log(f'derive: {len(rows)} рядов ({", ".join(rows)}); тестовых файлов {n}; индекс {INDEX}')


def cmd_chain(a):
    boosts = a.boosts or Y['order']
    me = [sys.executable, '-W', 'ignore', str(Path(__file__).resolve())]
    for b in boosts:
        if not a.no_leak:
            lk = CACHE / 'leak_A1.csv'
            if not (lk.exists() and ((pd.read_csv(lk).boost == b) & pd.read_csv(lk).ok).sum() >= 2 * len(M) * len(Y['leak_steps'])):
                subprocess.run(me + ['leak', '--boost', b], check=True)
        subprocess.run(me + ['run', '--boost', b, '--workers', str(a.workers)], check=True)
        subprocess.run(me + ['derive'], check=True)
    log('chain: готово')


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest='cmd', required=True)
    p = sp.add_parser('zamer'); p.add_argument('--boost', required=True); p.add_argument('--model', default='xgb'); p.add_argument('--step', type=int, default=5)
    p = sp.add_parser('leak'); p.add_argument('--boost', required=True)
    p = sp.add_parser('run'); p.add_argument('--boost', required=True); p.add_argument('--workers', type=int, default=int(Y['workers']))
    p.add_argument('--seeds', type=int, nargs='*')
    sp.add_parser('derive')
    p = sp.add_parser('chain'); p.add_argument('--workers', type=int, default=int(Y['workers'])); p.add_argument('--boosts', nargs='*')
    p.add_argument('--no-leak', action='store_true')
    a = ap.parse_args()
    CACHE.mkdir(parents=True, exist_ok=True)
    LOCK.mkdir(parents=True, exist_ok=True)
    {'zamer': cmd_zamer, 'leak': cmd_leak, 'run': cmd_run, 'derive': cmd_derive, 'chain': cmd_chain}[a.cmd](a)


if __name__ == '__main__':
    main()
