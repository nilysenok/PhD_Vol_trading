#!/usr/bin/env python3
"""Справочные строки табл. 3.4 «LightGBM / XGBoost (чистый бустинг, с поправкой)» — поручение владельца 26.10, 10:55.
ЭТО НЕ ВЫБОР ДИЗАЙНА: без подбора и без сравнения вариантов по тесту; одна заранее заданная спецификация на h, seed 42.

Чистый бустинг = нога финала (те же точки гиперпараметров по шагам, те же признаки и исключения признаков, та же схема шагов, ранняя
остановка и бины), но БЕЗ прогноза HAR-J/HAR-X в признаках (harf), без офсета и без остаточной схемы (v9); цель — ln RV (h=1) или
ln среднего RV за [t+1, t+h] (h=5/22); винзоризация цели, Huber и linear_tree — как у ноги.
  h=1  — нога D22_h1 (top5): cfg ноги {harf, wins} -> чистый {wins}; признаки v3 + RRV + календарь окна h=1 + гэп (d22_h1.patch),
         исключения D22_h1; шаги 1-6 — среднее топ-5 точек шага (seed 42), шаги 7-9 и 10 — точка шага 6.
  h=5  — нога D12_h5 (single, seed 42; в финале XGB-слот = среднее V9-XGB и V9-LGBM, wrap2 «2»): cfg {v9, harf, huber} + Д7 -> {huber} + Д7;
         исключения Bv2b; точки D12_h5_selected_h5.csv (шаги 7-9 и 10 — точка шага 6).
         ↺ XGBoost h=5 (reg:pseudohubererror): цель обучения центрируется средним ln цели обучающего окна шага (константа; прогноз = константа +
         выход бустинга). Причина — механическая, выявлена на val до открытия теста: XGBoost 2.1.4 оценивает base_score для pseudo-Huber одним
         шагом Ньютона от 0, на нецентрированной ln RV (≈ −8) это ≈ −555, ранняя остановка на итерации 0 на всех 10 шагах, прогноз ≡ 1e-10
         (обрезка; деревья — ref_v0_pure/REF_V0_h5_trees_xgb_bez_centra_vyrozhden.csv). У ноги V9 цель — остаток (≈ 0), вопроса не было;
         LightGBM (boost_from_average) и XGBoost с квадратичной потерей (h=1, 22) стартуют со среднего сами — у них без изменений.
  h=22 — нога D12_h22 (single, seed 42; так стоят xgb_corr/lgb_corr файла final_h22_test: хост HAR-J — D12_h22, нога single):
         cfg {v9, linear_lambda 1} -> {linear_lambda 1}; исключения Bv2b; точки D12_h22_selected_h22.csv.
Шаги СВ: шаг k = тестовый год Y = 2016+k: обучение — годы <= Y−2, ранняя остановка (patience 30, нативная метрика) — val Y−1; шаг 10 —
обучение <= 2024, val 2025, прогноз 01–02.2026 (как у финала). Поправка Б0.1: F = exp(s²/2), s² — sm (дисперсия объединённых ln-остатков
val-лет шагов k−2..k своей модели), на тесте шага k — оценка шага k (val-годы <= Y−1), как xgb_corr/lgb_corr финала.

Режимы (по одному h на процесс: матрица признаков задаётся до импорта l2_lib):
  --repro  контроль идентичности: драйвер с cfg ноги повторяет сохранённые val-прогнозы ноги (шаги 2 и 10, seed 42, ранг 1) и поправка
           Б0.1 этого кода на сохранённых val ноги повторяет xgb_corr/xgb_raw, lgb_corr/lgb_raw финального тестового файла (по шагам);
  --train  чистый бустинг, шаги 1-10 -> data/regen/ref_v0_pure/REF_V0_{xgb,lgbm}_h{h}_{val,pts_val}.parquet,
           data/regen/test_locked/ref_v0_pure/REF_V0_{xgb,lgbm}_h{h}_{test,pts_test}.parquet (+ SHA256.txt); метрики теста НЕ считаются;
  --leak   детектор «будущее не влияет»: все столбцы строк с датой > T (T внутри тестового года шага) заменяются шумом до построения
           признаков; прогнозы val и теста на строках <= T обязаны совпасть побайтно; самопроверка — признак-утечка (среднее RV тикера за
           следующие h строк) обязан быть пойман.
python3 scripts/analysis/ref_v0_pure.py --h 1 --repro|--train|--leak [--workers 2]"""
import argparse, hashlib, json, os, sys, time, multiprocessing as mp
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument('--h', type=int, required=True, choices=[1, 5, 22])
ap.add_argument('--repro', action='store_true'); ap.add_argument('--train', action='store_true'); ap.add_argument('--leak', action='store_true')
ap.add_argument('--workers', type=int, default=2)
A_ = ap.parse_args()
H = A_.h
DATA = Path('/Users/nilysenok/Desktop/Старое/PhD_Vol_trading/data')
R = DATA / 'regen'
for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[k] = '1'
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
if H == 1:
    import d22_h1 as D22  # L2_FEAT = features_pit_v3_rrv; подмены l2_lib (признаки Д22, split шага 10) — только в этом процессе
    L = D22.L
    D22.patch()
else:
    os.environ['L2_FEAT'] = str(R / 'features_pit_v3.parquet')
    import l2_lib as L
    _split0 = L.split

    def _split10(A, h, step):  # как final2_step10._split10
        if step != 10:
            return _split0(A, h, step)
        t = L.tcol(h)
        tr = A[A.year < 2026]
        return tr[tr.year != 2025].dropna(subset=[t]), tr[tr.year == 2025].dropna(subset=[t]), A[A.year == 2026], 2025, 2026
    L.split = _split10
import numpy as np, pandas as pd
from tune_L2 import SPACEB

CONF = {
    1: dict(tag='D22_h1', btag='D22_h1', blocks=(), leg='top5', cfg_leg={'harf': True, 'wins': True}, cfg_pure={'wins': True},
            s10=('D22/step10', 'test_locked/D22_h1/step10')),
    5: dict(tag='D12_h5', btag='Bv2b', blocks=('d7',), leg='single', cfg_leg={'v9': True, 'harf': True, 'huber': True}, cfg_pure={'huber': True},
            s10=('final2/step10', 'test_locked/final2/step10'), center=('xgb',)),
    22: dict(tag='D12_h22', btag='Bv2b', blocks=(), leg='single', cfg_leg={'v9': True, 'linear_lambda': 1}, cfg_pure={'linear_lambda': 1},
             s10=('final2/step10', 'test_locked/final2/step10')),
}[H]
OUTV, LOCK = R / 'ref_v0_pure', R / 'test_locked' / 'ref_v0_pure'
FINAL = R / 'test_locked' / 'final3' / f'final_h{H}_test.parquet'
FINAL_SHA = {1: '4e56f6252a48cb47f6027a0ff86c1f197102a5e02f61123265287b119cf4e7d0', 5: '4694991372bd95d3e2a7a091c2548444d77050ca4c75cc40170f5bd78b8652fc',
             22: '91f2a48e5f4ad431ac75c85bdd3dd7a58a449e4e0c9d731fed6dbd0a4937b059'}[H]
M = ('xgb', 'lgbm')
K = ['date', 'ticker', 'step']
G, _R = {}, {}


def log(*a):
    print(time.strftime('%T'), f'h={H}', *a, flush=True)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def prm(m, row):
    return {k: row[k] for k, *_ in SPACEB[m]}


def points():
    """-> [(model, step, rank, point, params)] ноги: шаги 1-9 (h=1 — топ-5 на шагах 1-6) и шаг 10 (точка шага 6)."""
    sel = L.selected_params(CONF['tag'], H)
    out = []
    for m in M:
        pid = lambda r: int(r.get('point', r.get('trial', -1)))
        if CONF['leg'] == 'top5':
            gr = pd.read_csv(R / f"{CONF['tag']}_grid.csv").set_index(['model', 'point'])
            t5 = pd.read_parquet(R / f"{CONF['tag']}_{m}_h{H}_top5_val.parquet", columns=['step', 'point', 'rank', 'seed']).drop_duplicates()
            assert (t5.seed == 42).all() and sorted(t5.step.unique()) == [1, 2, 3, 4, 5, 6] and (t5.groupby('step').size() == 5).all()
            for r in t5.sort_values(['step', 'rank']).itertuples():
                p = prm(m, gr.loc[(m, int(r.point))].to_dict())
                if r.rank == 1:  # ранг 1 = выбранная точка шага (selected), те же гиперпараметры
                    assert int(sel[(m, r.step)]['point']) == int(r.point) and all(np.isclose(float(p[k]), float(sel[(m, r.step)][k]), rtol=0, atol=0) for k in p)
                out.append((m, int(r.step), int(r.rank), int(r.point), p))
            steps_single = (7, 8, 9)
        else:
            steps_single = (1, 2, 3, 4, 5, 6, 7, 8, 9)
        for s in steps_single:
            out.append((m, s, 1, pid(sel[(m, s)]), prm(m, sel[(m, s)])))
        for s in (7, 8, 9):  # шаги 7-9 — точка шага 6
            assert prm(m, sel[(m, s)]) == prm(m, sel[(m, 6)]), (m, s)
        out.append((m, 10, 1, pid(sel[(m, 6)]), prm(m, sel[(m, 6)])))
    return out


def load(perturb_after=None, leak=False, seed=20261026):
    """Матрица и признаки (как при подборе ноги). perturb_after — дата T: все столбцы строк с датой > T (кроме date, ticker, year) заменяются
    шумом ДО построения признаков. leak — добавить признак-утечку (среднее RV тикера за следующие H строк; только для самопроверки детектора)."""
    A, FC = L.load_base(None, CONF['blocks'], H)
    if perturb_after is not None:
        rng = np.random.default_rng(seed)
        d = pd.to_datetime(A['date'])
        m = (d > pd.Timestamp(perturb_after)).values
        for c in A.columns:
            if c in ('date', 'ticker', 'year') or not np.issubdtype(A[c].dtype, np.number):
                continue
            src = A.loc[~m, c].dropna().values
            if len(src):
                A.loc[m, c] = rng.choice(src, m.sum()) * rng.choice([0.5, 1.0, 3.0], m.sum())
    if leak:
        g = A.groupby('ticker')['rv']
        A['_leak'] = g.transform(lambda s: s[::-1].rolling(H, min_periods=1).mean()[::-1].shift(-1))
    A2, cols, _, _ = L.build_features(A, FC, H, CONF['cfg_pure'])
    if leak:
        A2['_leak'] = A['_leak'].values
        cols = cols + ['_leak']
    return A2, cols


def ref(var, kind, m):
    k = (var, kind, m)
    if k not in _R:
        _R[k] = L.make_ref(m, G[var][0], G[var][1], H, CONF[f'cfg_{kind}'], CONF['btag'])
    return _R[k]


def job(t):
    var, kind, m, step, rank, point, p, seed = t
    A, cols = G[var]
    cfg = CONF[f'cfg_{kind}']
    D = L.step_data(A, cols, H, step, m, cfg, CONF['btag'])
    base = None
    if kind == 'pure' and m in CONF.get('center', ()):  # базовый уровень = среднее ln цели обучающего окна (константа, не прогноз HAR)
        base = float(np.mean(D['ytr_ln']))
        D['off'] = {k: np.full(len(v), base) for k, v in D['off'].items()}
    t0 = time.time()
    pv, pt, bi, _, _ = L.fit(m, p, D, cfg, seed=seed, ref=ref(var, kind, m) if step > 1 else None)
    meta = dict(var=var, kind=kind, model=m, h=H, step=step, rank=rank, point=point, seed=seed, best_iteration=bi, n_features=len(D['names']),
                har_in_features=any(n in ('har_ln', 'h1_ln') for n in D['names']),
                offset_zero=bool(all(np.all(v == 0) for v in D['off'].values())) if base is None else bool(all(np.all(v == base) for v in D['off'].values())),
                const_base=base,
                sec=round(time.time() - t0, 1), **{f'p_{k}': v for k, v in p.items()})
    fr = []
    for part, pred, yr in (('va', pv, D['vy']), ('te', pt, D['ty'])):
        d = D['parts'][part]
        fr.append(pd.DataFrame({'date': pd.to_datetime(d['date'].values), 'ticker': d['ticker'].values, 'year': yr, 'step': step, 'seed': seed,
                                'rank': rank, 'point': point, 'rv_actual': d[L.tcol(H)].values, 'rv_pred': np.clip(np.exp(pred), 1e-10, None)}))
    return meta, fr[0], fr[1]


def run(tasks, workers):
    with mp.get_context('fork').Pool(workers) as P:
        return P.map(job, tasks, chunksize=1)


# ---------------------------------------------------------------- поправка Б0.1 (как gibrid_bias.s2_est / final2_build.factors)
def factors(val, code='sm'):
    """val: DataFrame[step, rv_actual, rv_pred] -> {шаг k: F_k = exp(s²_k / 2)}, s²_k — по ln-остаткам val шагов max(1, k−2)..k (sm)."""
    v = val[val.rv_actual > 1e-12]
    E = {st: np.log(g.rv_actual.values + 1e-10) - np.log(g.rv_pred.values) for st, g in v.groupby('step')}
    out = {}
    for k in sorted(E):
        e = np.concatenate([E[j] for j in range(max(1, k - 2) if code in ('sm', 'smrob') else k, k + 1)])
        out[k] = float(np.exp(np.var(e) / 2))
    return out


def leg_val(m):
    """Сохранённые val-прогнозы ноги финала по слоту (как gibrid_io.load_boost + final2_build.leg10), шаги 1-10."""
    tag = CONF['tag']
    s = pd.read_parquet(R / f'{tag}_{m}_h{H}_val.parquet')
    s = s[s.seed == 42] if 'seed' in s else s
    if CONF['leg'] == 'top5':
        t5 = pd.read_parquet(R / f'{tag}_{m}_h{H}_top5_val.parquet').groupby(K, as_index=False).agg(rv_actual=('rv_actual', 'first'), rv_pred=('rv_pred', 'mean'))
        s = pd.concat([t5, s[~s.step.isin(t5.step.unique())][K + ['rv_actual', 'rv_pred']]], ignore_index=True)
    s10 = pd.read_parquet(R / CONF['s10'][0] / f'{tag}_{m}_h{H}_s10_val.parquet')
    s10 = s10[s10.seed == 42]
    return pd.concat([s[K + ['rv_actual', 'rv_pred']], s10[K + ['rv_actual', 'rv_pred']]], ignore_index=True)


def repro(workers):
    """Контроль: (1) cfg ноги этим драйвером = сохранённые val ноги (шаги 2 и 10, seed 42, ранг 1); (2) поправка Б0.1 этого кода на val ноги =
    отношение *_corr / *_raw финального тестового файла по шагам (тест читается только как отношение двух столбцов, QLIKE не считается)."""
    A, FC = L.load_base(None, CONF['blocks'], H)
    A2, cols, _, _ = L.build_features(A, FC, H, CONF['cfg_leg'])
    G['leg'] = (A2, cols)
    P = {(m, s): (r, pt, p) for (m, s, r, pt, p) in points() if r == 1}
    tasks = [('leg', 'leg', m, s, 1, P[(m, s)][1], P[(m, s)][2], 42) for m in M for s in (2, 10)]
    res = run(tasks, workers)
    rows = []
    for meta, fv, _ in res:
        m, s = meta['model'], meta['step']
        if s == 10:
            ref_v = pd.read_parquet(R / CONF['s10'][0] / f"{CONF['tag']}_{m}_h{H}_s10_val.parquet")
        else:
            ref_v = pd.read_parquet(R / f"{CONF['tag']}_{m}_h{H}_val.parquet")
        ref_v = ref_v[(ref_v.step == s) & ((ref_v.seed == 42) if 'seed' in ref_v else True)]
        j = fv.merge(ref_v[['date', 'ticker', 'rv_pred']].assign(date=lambda d: pd.to_datetime(d.date)), on=['date', 'ticker'], suffixes=('', '_saved'), validate='1:1')
        d = float(np.max(np.abs(j.rv_pred / j.rv_pred_saved - 1)))
        rows.append(dict(check='драйвер с cfg ноги = сохранённые val ноги', model=m, step=s, rows=len(j), rows_saved=len(ref_v), max_rel_diff=d,
                         best_iteration=meta['best_iteration'], n_features=meta['n_features'], ok=bool(len(j) == len(ref_v) and d < 1e-6)))
    assert sha(FINAL) == FINAL_SHA, 'SHA-256 финального тестового файла не совпал с FINAL_DESIGN'
    T = pd.read_parquet(FINAL, columns=['step', 'xgb_corr', 'xgb_raw', 'lgb_corr', 'lgb_raw'])
    slot = {'xgb': 'xgb', 'lgbm': 'lgb'}
    for m in M:
        v = leg_val(m)
        if H == 5:  # wrap2 «2»: XGB-слот = среднее LGBM и XGB по RV до поправки
            if m == 'xgb':
                o = leg_val('lgbm').merge(v[K + ['rv_pred']], on=K, suffixes=('_l', '_x'), validate='1:1')
                v = o.assign(rv_pred=(o.rv_pred_l + o.rv_pred_x) / 2)[K + ['rv_actual', 'rv_pred']]
        F = factors(v)
        c = slot[m]
        ratio = T.assign(r=T[f'{c}_corr'] / T[f'{c}_raw']).groupby('step').r.agg(['min', 'max'])
        for st, f in F.items():
            d = float(max(abs(ratio.loc[st, 'min'] / f - 1), abs(ratio.loc[st, 'max'] / f - 1)))
            rows.append(dict(check=f'поправка Б0.1 (sm) этого кода = {c}_corr/{c}_raw финала', model=m, step=st, F=f, max_rel_diff=d, ok=bool(d < 1e-9)))
    R_ = pd.DataFrame(rows)
    OUTV.mkdir(parents=True, exist_ok=True)
    R_.to_csv(OUTV / f'repro_h{H}.csv', index=False)
    log(R_.to_string())
    assert R_.ok.all(), 'контроль идентичности не пройден'
    log('контроль идентичности: PASS')


def train(workers):
    A2, cols = load()
    G['base'] = (A2, cols)
    tasks = [('base', 'pure', m, s, r, pt, p, 42) for (m, s, r, pt, p) in points()]
    log(f'чистый бустинг: признаков {len(cols)} до исключений, задач {len(tasks)}')
    t0 = time.time()
    res = run(tasks, workers)
    meta = pd.DataFrame([r[0] for r in res])
    assert not meta.har_in_features.any() and meta.offset_zero.all(), 'в признаках прогноз HAR или офсет, отличный от нуля / константы'
    OUTV.mkdir(parents=True, exist_ok=True); LOCK.mkdir(parents=True, exist_ok=True)
    meta.to_csv(OUTV / f'REF_V0_h{H}_trees.csv', index=False)
    shas = {}
    for m in M:
        rr = [r for r in res if r[0]['model'] == m]
        pv, pt = (pd.concat([r[i] for r in rr], ignore_index=True) for i in (1, 2))
        agg = lambda d: d.groupby(K, as_index=False).agg(year=('year', 'first'), rv_actual=('rv_actual', 'first'), rv_pred=('rv_pred', 'mean'),
                                                         n_points=('rank', 'size'))
        V, T = agg(pv), agg(pt)  # h=1, шаги 1-6: среднее топ-5 по RV (как gibrid_io._avg); иначе одна точка
        F = factors(V)
        T = T.rename(columns={'rv_pred': 'rv_raw'})
        T['F'] = T.step.map(F)
        T['rv_pred'] = T.rv_raw * T.F
        T['design'] = f"чистый бустинг, нога {CONF['tag']} ({CONF['leg']}), cfg {json.dumps(CONF['cfg_pure'])}, Б0.1 sm"
        assert T.F.notna().all() and sorted(T.step.unique()) == list(range(1, 11))
        files = {OUTV / f'REF_V0_{m}_h{H}_val.parquet': V.assign(F_oos=V.step.map(lambda k: F.get(k - 1, 1.0))),
                 OUTV / f'REF_V0_{m}_h{H}_pts_val.parquet': pv,
                 LOCK / f'REF_V0_{m}_h{H}_test.parquet': T, LOCK / f'REF_V0_{m}_h{H}_pts_test.parquet': pt}
        for f, d in files.items():
            d.sort_values(K + (['rank'] if 'rank' in d else [])).to_parquet(f, index=False)
            if f.parent == LOCK:
                shas[f.name] = sha(f)
        pd.DataFrame([dict(model=m, h=H, step=k, F=v) for k, v in F.items()]).to_csv(OUTV / f'REF_V0_{m}_h{H}_factors.csv', index=False)
    old = {}
    sf = LOCK / 'SHA256.txt'
    if sf.exists():
        old = dict(reversed(l.split('  ', 1)) for l in sf.read_text().splitlines() if l.strip())
    old.update(shas)
    sf.write_text(''.join(f'{v}  {k}\n' for k, v in sorted(old.items())))
    log(f'готово за {time.time() - t0:.0f} с; тест — только запись (метрики не считались): ' + '; '.join(f'{k} {v[:16]}…' for k, v in shas.items()))


def leak(workers):
    """Детектор: шаги 2, 9, 10, оба бустинга; T — середина тестового года шага (шаг 10 — 15.01.2026)."""
    cut = {2: '2018-06-29', 9: '2025-06-30', 10: '2026-01-15'}
    P = {(m, s): (r, pt, p) for (m, s, r, pt, p) in points() if r == 1}
    rows = []
    for st, T in cut.items():
        _R.clear()
        G.clear()
        G['o'], G['p'] = load(), load(perturb_after=T)
        G['oL'], G['pL'] = load(leak=True), load(perturb_after=T, leak=True)
        tasks = [(v, 'pure', m, st, 1, P[(m, st)][1], P[(m, st)][2], 42) for v in ('o', 'p', 'oL', 'pL') for m in M]
        res = {(r[0]['var'], r[0]['model']): r for r in run(tasks, workers)}
        for m in M:
            for a, b, lk in (('o', 'p', False), ('oL', 'pL', True)):
                ma, va, ta = res[(a, m)]
                mb, vb, tb = res[(b, m)]
                tT = pd.Timestamp(T)
                same_val = len(va) == len(vb) and np.array_equal(va.rv_pred.values, vb.rv_pred.values)
                ja = ta[ta.date <= tT].reset_index(drop=True)
                jb = tb[tb.date <= tT].reset_index(drop=True)
                mism = int((ja.rv_pred.values != jb.rv_pred.values).sum()) if len(ja) == len(jb) else -1
                after = ta[ta.date > tT][['date', 'ticker', 'rv_pred']].merge(tb[tb.date > tT][['date', 'ticker', 'rv_pred']], on=['date', 'ticker'])
                rows.append(dict(step=st, T=T, model=m, leak_injected=lk, val_identical=same_val, best_it_identical=ma['best_iteration'] == mb['best_iteration'],
                                 test_rows_le_T=len(ja), mismatch_le_T=mism, changed_after_T=int((after.rv_pred_x != after.rv_pred_y).sum())))
        log(f'детектор шаг {st}: готово')
    Lk = pd.DataFrame(rows)
    OUTV.mkdir(parents=True, exist_ok=True)
    Lk.to_csv(OUTV / f'leak_h{H}.csv', index=False)
    clean, dirty = Lk[~Lk.leak_injected], Lk[Lk.leak_injected]
    ok = bool(clean.val_identical.all() and clean.best_it_identical.all() and (clean.mismatch_le_T == 0).all())
    moved = bool((clean.changed_after_T > 0).all())
    caught = bool((dirty.mismatch_le_T > 0).all())
    log(Lk.to_string())
    log(f'детектор «будущее не влияет»: {"PASS" if ok else "FAIL"}; шум меняет прогнозы после T: {moved}; утечка поймана: {"PASS" if caught else "FAIL"}')
    assert ok and moved and caught


if __name__ == '__main__':
    if A_.repro:
        repro(A_.workers)
    if A_.train:
        train(A_.workers)
    if A_.leak:
        leak(A_.workers)
