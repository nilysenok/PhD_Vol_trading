#!/usr/bin/env python3
"""Табл. 3.4, строки «бустинг» — базовый бустинг V0, как он определён в диссертации (решение владельца 26.10, 11:40). НЕ ВЫБОР ДИЗАЙНА:
без подбора и без сравнения вариантов по тесту; прогнозы открыты на тесте для отчёта, в выборе финала не участвовали.

Спецификация V0: XGBoost / LightGBM на ln RV (h=1) или ln среднего RV за [t+1, t+h] (h=5/22); признаки — базовый набор features_pit_v3
(feature_cols конфигурации матрицы; без календаря окна, ночного гэпа, RRV, слагаемых HAR-X и прогноза HAR-J/HAR-X); гиперпараметры —
протокол B v2b для одиночной модели (точка ранга 1 шага; шаги 7-9 и 10 — точка шага 6), исключения признаков B v2b; без винзоризации цели,
без linear_tree (LightGBM — обычный gbdt), без Huber; ранняя остановка на val-годе шага (patience 30); seed 42/43/44; поправка Б0.1 sm
(s² по val-остаткам своей модели и своего seed, на тесте шага k — оценка шага k, как у финала).
Источники (путь):
  (а) сохранённые прогнозы кандидатов V0: h=1 — D12_h1_V0 (кандидат V0 финального пула; l2_lib cfg {}, точки B v2b, features_pit_v3), шаги 1-9
      и шаг 10 (final2/step10); h=5/22 — Bv3 (rerun_h1_v3.py через final2_v0.sh: tune_Bv2.final, точки B v2b, features_pit_v3), шаги 1-9;
  (в) шаг 10 при h=5/22 сохранённым не был (Bv3 шага 10 нет) — обучен здесь без подбора: точка шага 6 B v2b, тот же драйвер, что у шага 10
      кандидатов (l2_lib, как final2_step10.py), seed 42/43/44. Детектор «будущее не влияет» — для этих моделей.
Режимы:
  --check   (1) D12_h1_V0 и Bv3 при h=1 (одна спецификация, разные драйверы) — расхождение val; (2) драйвер этого скрипта (l2_lib, cfg {})
            повторяет сохранённые val Bv3 при h=5/22 (шаги 2 и 9, seed 42 и 43) и val шага 10 D12_h1_V0 (seed 42) — тот же драйвер для шага 10;
  --step10  h=5/22: шаг 10 (обучение <= 2024, val 2025, прогноз 01–02.2026) -> data/regen/ref_v0_base/, test_locked/ref_v0_base/;
  --leak    h=5/22, шаг 10: шум во всех столбцах строк после 15.01.2026 до построения признаков; прогнозы val и теста до T — побайтно; самопроверка
            признаком-утечкой;
  --assemble сборка по seed: шаги 1-10, поправка Б0.1 sm -> test_locked/ref_v0_base/REF_V0B_{xgb,lgbm}_h{h}_test.parquet (+ SHA256.txt,
            istochniki_h{h}.csv — SHA-256 сохранённых входов), val -> data/regen/ref_v0_base/. Метрики теста здесь не считаются.
python3 scripts/analysis/ref_v0_base.py --h 5 --check|--step10|--leak|--assemble [--workers 3]"""
import argparse, hashlib, json, os, sys, time, multiprocessing as mp
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument('--h', type=int, required=True, choices=[1, 5, 22])
for k in ('check', 'step10', 'leak', 'assemble'):
    ap.add_argument(f'--{k}', action='store_true')
ap.add_argument('--workers', type=int, default=3)
A_ = ap.parse_args()
H = A_.h
DATA = Path('/Users/nilysenok/Desktop/Старое/PhD_Vol_trading/data')
R = DATA / 'regen'
for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[k] = '1'
os.environ['L2_FEAT'] = str(R / 'features_pit_v3.parquet')
sys.path.insert(0, str(Path(__file__).resolve().parent))
import l2_lib as L
import numpy as np, pandas as pd

_split0 = L.split


def _split10(A, h, step):  # как final2_step10._split10
    if step != 10:
        return _split0(A, h, step)
    t = L.tcol(h)
    tr = A[A.year < 2026]
    return tr[tr.year != 2025].dropna(subset=[t]), tr[tr.year == 2025].dropna(subset=[t]), A[A.year == 2026], 2025, 2026


L.split = _split10
M = ('xgb', 'lgbm')
SEEDS = (42, 43, 44)
K = ['date', 'ticker', 'step']
CFG = {}
PCOLS = ('num_leaves', 'learning_rate', 'bagging_fraction', 'feature_fraction', 'min_data_in_leaf', 'reg_alpha', 'reg_lambda',
         'max_depth', 'subsample', 'colsample_bytree', 'min_child_weight')
SRC = {1: 'D12_h1_V0', 5: 'Bv3', 22: 'Bv3'}[H]
SRC_LOCK = R / 'test_locked' / SRC
OUTV, LOCK = R / 'ref_v0_base', R / 'test_locked' / 'ref_v0_base'
G, _R = {}, {}


def log(*a):
    print(time.strftime('%T'), f'h={H}', *a, flush=True)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def point_params(m, step):
    """Точка B v2b шага (ранг 1; шаги 7-10 — точка шага 6) -> (point, params)."""
    S = pd.read_csv(R / f'Bv2b_selected_h{H}.csv')
    s = min(step, 6) if step >= 7 else step
    pt = int(S[(S.model == m) & (S.step == s)].point.iloc[0])
    g = pd.concat([pd.read_csv(R / f'Bv2b_grid_{x}.csv') for x in ('xgb40', 'xgb100', 'lgbm100')])
    r = g[(g.model == m) & (g.point == pt)].iloc[0]
    return pt, {k: r[k] for k in PCOLS}


def load(perturb_after=None, leak=False, seed=20261026):
    A, FC = L.load_base(None, (), H)
    if perturb_after is not None:
        rng = np.random.default_rng(seed)
        msk = (pd.to_datetime(A['date']) > pd.Timestamp(perturb_after)).values
        for c in A.columns:
            if c in ('date', 'ticker', 'year') or not np.issubdtype(A[c].dtype, np.number):
                continue
            src = A.loc[~msk, c].dropna().values
            if len(src):
                A.loc[msk, c] = rng.choice(src, msk.sum()) * rng.choice([0.5, 1.0, 3.0], msk.sum())
    if leak:
        A['_leak'] = A.groupby('ticker')['rv'].transform(lambda s: s[::-1].rolling(H, min_periods=1).mean()[::-1].shift(-1))
    A2, cols, _, _ = L.build_features(A, FC, H, CFG)
    if leak:
        cols = cols + ['_leak']
    return A2, cols


def job(t):
    var, m, step, seed = t
    A, cols = G[var]
    D = L.step_data(A, cols, H, step, m, CFG, 'Bv2b')
    k = (var, m)
    if step > 1 and k not in _R:
        _R[k] = L.make_ref(m, A, cols, H, CFG, 'Bv2b')
    pt, p = point_params(m, step)
    pv, pte, bi, _, _ = L.fit(m, p, D, CFG, seed=seed, ref=_R.get(k) if step > 1 else None)
    meta = dict(var=var, model=m, h=H, step=step, seed=seed, point=pt, best_iteration=bi, n_features=len(D['names']),
                har_in_features=any(n in ('har_ln', 'h1_ln') for n in D['names']), offset_zero=bool(all(np.all(v == 0) for v in D['off'].values())))
    fr = []
    for part, pred, yr in (('va', pv, D['vy']), ('te', pte, D['ty'])):
        d = D['parts'][part]
        fr.append(pd.DataFrame({'date': pd.to_datetime(d['date'].values), 'ticker': d['ticker'].values, 'year': yr, 'step': step, 'seed': seed,
                                'point': pt, 'rv_actual': d[L.tcol(H)].values, 'rv_pred': np.clip(np.exp(pred), 1e-10, None)}))
    return meta, fr[0], fr[1]


def run(tasks, workers):
    with mp.get_context('fork').Pool(workers) as P:
        return P.map(job, tasks, chunksize=1)


def saved(m, kind, k):
    """Сохранённые прогнозы источника (а): kind 'single' (seed 42) | 'seeds' (43/44) | 's10'; k — 'val' | 'test'."""
    if kind == 's10':
        f = (R / 'final2' / 'step10' if k == 'val' else R / 'test_locked' / 'final2' / 'step10') / f'{SRC}_{m}_h{H}_s10_{k}.parquet'
    else:
        sfx = '' if kind == 'single' else '_seeds4344'
        f = (R if k == 'val' else SRC_LOCK) / f'{SRC}_{m}_h{H}{sfx}_{k}.parquet'
    d = pd.read_parquet(f)
    d['date'] = pd.to_datetime(d['date'])
    return d, f


def check(workers):
    rows = []
    if H == 1:  # (1) две сохранённые реализации одной спецификации
        for m in M:
            for kind in ('single', 'seeds'):
                a, _ = saved(m, kind, 'val')
                b = pd.read_parquet(R / f'Bv3_{m}_h1{"" if kind == "single" else "_seeds4344"}_val.parquet')
                b['date'] = pd.to_datetime(b['date'])
                j = a.merge(b, on=K + ['seed'], suffixes=('', '_b'), validate='1:1')
                rows.append(dict(check='D12_h1_V0 = Bv3 (val, шаги 1-9)', model=m, step='1-9', seed=kind, rows=len(j), rows_a=len(a),
                                 max_rel_diff=float(np.max(np.abs(j.rv_pred / j.rv_pred_b - 1))), ok=True))
    G['o'] = load()
    tasks = ([('o', m, s, sd) for m in M for s in (2, 9) for sd in (42, 43)] if H != 1 else []) + ([('o', m, 10, 42) for m in M] if H == 1 else [])
    for meta, fv, _ in run(tasks, workers):
        m, s, sd = meta['model'], meta['step'], meta['seed']
        if s == 10:
            ref, f = saved(m, 's10', 'val')
        else:
            ref, f = saved(m, 'single' if sd == 42 else 'seeds', 'val')
        ref = ref[(ref.step == s) & (ref.seed == sd)]
        j = fv.merge(ref[['date', 'ticker', 'rv_pred']], on=['date', 'ticker'], suffixes=('', '_saved'), validate='1:1')
        d = float(np.max(np.abs(j.rv_pred / j.rv_pred_saved - 1)))
        rows.append(dict(check=f'драйвер (l2_lib, cfg {{}}) = сохранённые val {f.name}', model=m, step=s, seed=sd, rows=len(j), rows_a=len(ref),
                         max_rel_diff=d, best_iteration=meta['best_iteration'], n_features=meta['n_features'], ok=bool(len(j) == len(ref) and d < 1e-6)))
    Rr = pd.DataFrame(rows)
    OUTV.mkdir(parents=True, exist_ok=True)
    Rr.to_csv(OUTV / f'check_h{H}.csv', index=False)
    log('\n' + Rr.to_string())
    assert Rr.ok.all()
    log('контроль: PASS')


def step10(workers):
    assert H in (5, 22), 'шаг 10 при h=1 сохранён (D12_h1_V0, final2/step10)'
    G['o'] = load()
    res = run([('o', m, 10, sd) for m in M for sd in SEEDS], workers)
    meta = pd.DataFrame([r[0] for r in res])
    assert not meta.har_in_features.any() and meta.offset_zero.all()
    OUTV.mkdir(parents=True, exist_ok=True); LOCK.mkdir(parents=True, exist_ok=True)
    meta.to_csv(OUTV / f'V0B_h{H}_s10_trees.csv', index=False)
    for m in M:
        rr = [r for r in res if r[0]['model'] == m]
        pd.concat([r[1] for r in rr]).to_parquet(OUTV / f'V0B_{m}_h{H}_s10_val.parquet', index=False)
        pd.concat([r[2] for r in rr]).to_parquet(LOCK / f'V0B_{m}_h{H}_s10_test.parquet', index=False)
    log('шаг 10 готов: ' + ', '.join(f"{r['model']}/s{r['seed']}: {r['best_iteration']}" for r in meta.to_dict('records')))


def leak(workers):
    assert H in (5, 22)
    T = '2026-01-15'
    G['o'], G['p'] = load(), load(perturb_after=T)
    G['oL'], G['pL'] = load(leak=True), load(perturb_after=T, leak=True)
    res = {(r[0]['var'], r[0]['model'], r[0]['seed']): r for r in run([(v, m, 10, sd) for v in ('o', 'p', 'oL', 'pL') for m in M for sd in SEEDS], workers)}
    rows, tT = [], pd.Timestamp(T)
    for m in M:
        for sd in SEEDS:
            for a, b, lk in (('o', 'p', False), ('oL', 'pL', True)):
                ma, va, ta = res[(a, m, sd)]
                mb, vb, tb = res[(b, m, sd)]
                ja, jb = ta[ta.date <= tT].reset_index(drop=True), tb[tb.date <= tT].reset_index(drop=True)
                aft = ta[ta.date > tT][['date', 'ticker', 'rv_pred']].merge(tb[tb.date > tT][['date', 'ticker', 'rv_pred']], on=['date', 'ticker'])
                rows.append(dict(step=10, T=T, model=m, seed=sd, leak_injected=lk, val_identical=bool(len(va) == len(vb) and np.array_equal(va.rv_pred.values, vb.rv_pred.values)),
                                 best_it_identical=ma['best_iteration'] == mb['best_iteration'], test_rows_le_T=len(ja),
                                 mismatch_le_T=int((ja.rv_pred.values != jb.rv_pred.values).sum()) if len(ja) == len(jb) else -1,
                                 changed_after_T=int((aft.rv_pred_x != aft.rv_pred_y).sum())))
    Lk = pd.DataFrame(rows)
    OUTV.mkdir(parents=True, exist_ok=True)
    Lk.to_csv(OUTV / f'leak_h{H}.csv', index=False)
    cl, dt = Lk[~Lk.leak_injected], Lk[Lk.leak_injected]
    ok = bool(cl.val_identical.all() and cl.best_it_identical.all() and (cl.mismatch_le_T == 0).all())
    moved, caught = bool((cl.changed_after_T > 0).all()), bool((dt.mismatch_le_T > 0).all())
    log('\n' + Lk.to_string())
    log(f'детектор «будущее не влияет»: {"PASS" if ok else "FAIL"}; шум меняет прогнозы после T: {moved}; утечка поймана: {"PASS" if caught else "FAIL"}')
    assert ok and moved and caught


def factors(val):
    """{шаг k: F_k = exp(s²_k/2)}, s²_k — дисперсия ln-остатков val шагов max(1, k−2)..k (sm); как ref_v0_pure.factors / gibrid_bias.s2_est."""
    v = val[val.rv_actual > 1e-12]
    E = {st: np.log(g.rv_actual.values + 1e-10) - np.log(g.rv_pred.values) for st, g in v.groupby('step')}
    return {k: float(np.exp(np.var(np.concatenate([E[j] for j in range(max(1, k - 2), k + 1)])) / 2)) for k in sorted(E)}


def assemble():
    OUTV.mkdir(parents=True, exist_ok=True); LOCK.mkdir(parents=True, exist_ok=True)
    src, shas = [], {}
    for m in M:
        parts = {}
        for k in ('val', 'test'):
            fr = []
            for kind in ('single', 'seeds'):
                d, f = saved(m, kind, k)
                src.append(dict(file=str(f), sha256=sha(f), rows=len(d), put='а'))
                fr.append(d[d.step <= 9].assign(put='а'))
            if H == 1:
                d, f = saved(m, 's10', k)
                src.append(dict(file=str(f), sha256=sha(f), rows=len(d), put='а'))
            else:
                f = (OUTV if k == 'val' else LOCK) / f'V0B_{m}_h{H}_s10_{k}.parquet'
                d = pd.read_parquet(f)
                src.append(dict(file=str(f), sha256=sha(f), rows=len(d), put='в'))
            fr.append(d.assign(put='а' if H == 1 else 'в'))
            X = pd.concat(fr, ignore_index=True)
            X['date'] = pd.to_datetime(X['date'])
            if 'rank' in X:
                assert (X['rank'].fillna(1) == 1).all()
            X = X[['date', 'ticker', 'year', 'step', 'seed', 'rv_actual', 'rv_pred', 'put']]
            assert not X.duplicated(K + ['seed']).any() and sorted(X.seed.unique()) == list(SEEDS) and sorted(X.step.unique()) == list(range(1, 11)), (m, k)
            assert (X.groupby('seed').size().nunique() == 1), (m, k, 'разное число строк по seed')
            parts[k] = X
        V, T = parts['val'], parts['test'].rename(columns={'rv_pred': 'rv_raw'})
        Fs = {sd: factors(V[V.seed == sd]) for sd in SEEDS}
        T['F'] = [Fs[s][k] for s, k in zip(T.seed, T.step)]
        T['rv_pred'] = T.rv_raw * T.F
        V['F_oos'] = [Fs[s].get(k - 1, 1.0) for s, k in zip(V.seed, V.step)]
        V.sort_values(K + ['seed']).to_parquet(OUTV / f'REF_V0B_{m}_h{H}_val.parquet', index=False)
        f = LOCK / f'REF_V0B_{m}_h{H}_test.parquet'
        T.sort_values(K + ['seed']).to_parquet(f, index=False)
        shas[f.name] = sha(f)
        pd.DataFrame([dict(model=m, h=H, seed=sd, step=k, F=v) for sd in SEEDS for k, v in Fs[sd].items()]).to_csv(OUTV / f'REF_V0B_{m}_h{H}_factors.csv', index=False)
    pd.DataFrame(src).to_csv(OUTV / f'istochniki_h{H}.csv', index=False)
    sf = LOCK / 'SHA256.txt'
    old = dict(reversed(l.split('  ', 1)) for l in sf.read_text().splitlines() if l.strip()) if sf.exists() else {}
    old.update(shas)
    for p in sorted(LOCK.glob('V0B_*_s10_test.parquet')):
        old[p.name] = sha(p)
    sf.write_text(''.join(f'{v}  {k}\n' for k, v in sorted(old.items())))
    log('собрано (метрики теста не считались): ' + '; '.join(f'{k} {v[:16]}…' for k, v in shas.items()))


if __name__ == '__main__':
    if A_.check:
        check(A_.workers)
    if A_.step10:
        step10(A_.workers)
    if A_.leak:
        leak(A_.workers)
    if A_.assemble:
        assemble()
