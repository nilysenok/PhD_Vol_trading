#!/usr/bin/env python3
"""R2, агент О: движок отбора второго раунда (протокол docs/zadaniya/2026-10-26_vtoroy_raund_gibrida.md, разделы «Отбор», «Критерии»,
«Итоговые ворота»; yaml models/designs/r2/r2_O_loyo.yaml).

Единица уровня (ячейка / точка LHS / дизайн) имеет внутренние варианты. Вход ядра — «таблица шагов»: строка = (unit, variant, step),
столбцы как у designs_val gibrid_v17: q_hyb, q_har, q_xgb_single, q_lgb_single, scheme, w_har, w_xgb, w_lgb, v1_ml
(+ необязательные q_har_c1 — HAR-J sm для c1, q_leg_v0 — нога единицы в роли V0 для c2, role, n_rych).
Складка Y (2017–2021 = шаги 2–6): вариант единицы — мин. средний q на T\\{Y} среди вариантов с c1–c4 на T\\{Y}; q_u(Y) — его q на Y;
ранг среди единиц h (средний при равенстве; недопущенные — хвост); R̄ — среднее по 5 складкам.
c1–c4 — gibrid_designs.flags (импорт, без изменений; GIBRID_SAME_SLOT=1 как final3.env_of). На складке флаги считаются по обучающим
годам; c4 — по непрерывным отрезкам шагов (пара соседних шагов — только если оба в обучающих годах).
Ворота: c1–c4 на всех складках; 2σ по 10 seed (Д24.3); лучше F_h на отложенном году не менее чем в 3 складках из 5.
Читаются только val шагов <= 6 (годы <= 2021, фильтр при чтении parquet и assert); метрики — только шаги 2–6.

python3 scripts/analysis/r2_loyo.py --selftest
python3 scripts/analysis/r2_loyo.py proverka [--h 1 5 22]          # механика на сохранённых рядах первого раунда (не выбор)
python3 scripts/analysis/r2_loyo.py level --level 1 [--h 1 5 22]    # после Э/Б1/Б2: R2/O/level1_h{h}.csv по spec_level1_h{h}.csv"""
import argparse, hashlib, json, os, re, subprocess, sys
from pathlib import Path
import numpy as np, pandas as pd
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
os.environ['GIBRID_SAME_SLOT'] = '1'  # одиночная модель в обоих слотах (HEAVY): слот, равный дизайну, не эталон c2/c3 (FD §3 п. 1)
import gibrid_designs as D  # noqa: E402  c1–c4 — без изменений
import gibrid_bias as BI  # noqa: E402  оценки s² (ord/rob/sm/smrob) — без изменений

REPO = HERE.parent.parent
CFG = yaml.safe_load((REPO / 'models' / 'designs' / 'r2' / 'r2_O_loyo.yaml').read_text())
R = Path(CFG['data'])
OUT = Path(CFG['out'])
T = tuple(CFG['steps_T'])
MAXSTEP = int(CFG['max_step_read'])
SEEDS = tuple(CFG['seeds'])
CR = ('c1', 'c2', 'c3', 'c4')
CODES = tuple(CFG['s2_codes'])
NEED = ['q_hyb', 'q_har', 'q_xgb_single', 'q_lgb_single', 'scheme', 'w_har', 'w_xgb', 'w_lgb', 'v1_ml']
Y = lambda s: 2015 + int(s)


def now():
    return subprocess.run(['date', '+%Y-%m-%d %H:%M:%S'], capture_output=True, text=True).stdout.strip()


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


# ---------------------------------------------------------------- чтение (только val шагов <= 6)
def read_val(path, cols=None):
    """parquet val: читаются только строки step <= 6 (фильтр pyarrow — строки шагов 7+ не загружаются); assert годы <= 2021."""
    d = pd.read_parquet(path, columns=cols, filters=[('step', '<=', MAXSTEP)])
    assert d.step.max() <= MAXSTEP, path
    if 'year' in d:
        assert d.year.max() <= 2015 + MAXSTEP, f'{path}: год > 2021'
    return d


def read_designs(path):
    """designs_val gibrid_v17 (csv): шаги 7+ должны быть без метрик (hide); оставляются шаги T."""
    cols = ['design', 'step', 'val_year'] + NEED
    d = pd.read_csv(path, usecols=cols)
    assert d.loc[d.step > MAXSTEP, 'q_hyb'].isna().all(), f'{path}: метрики шагов 7+'
    d = d[d.step.isin(T)].copy()
    assert (d.val_year == d.step + 2015).all()
    return d


# ---------------------------------------------------------------- q по шагам из ряда (адаптер уровней)
def q_steps(df, code):
    """OOS QLIKE по шагам T: s² шага t — оценка BI.s2_est(code) по ln-остаткам шагов <= t−1 (F = exp(s²/2)); bez — F = 1.
    Фильтр rv_actual > 1e-12 (как gibrid_io.merged). -> {шаг: q}."""
    d = df[df.rv_actual > 1e-12]
    assert d.step.max() <= MAXSTEP
    E = {'x': {st: np.log(g.rv_actual.values + 1e-10) - np.log(g.rv_pred.values) for st, g in d.groupby('step')}}
    out = {}
    for st in T:
        g = d[d.step == st]
        fac = 1.0 if code == 'bez' else float(np.exp(BI.s2_est(code, E, 'x', st - 1) / 2))
        y, p = g.rv_actual.values, np.clip(g.rv_pred.values * fac, 1e-10, None)
        out[st] = float(np.mean(y / p - np.log(y / p) - 1))
    return out


def factors(df, code):
    """Множители Б0.1 для тестовой сборки: тест шага k (k <= 6) — оценка по val шагов <= k; k >= 7 — заморозка на шаге 6 (протокол,
    «Тестовая часть»). -> {k: F_k}, k = 1..10."""
    d = df[df.rv_actual > 1e-12]
    E = {'x': {st: np.log(g.rv_actual.values + 1e-10) - np.log(g.rv_pred.values) for st, g in d.groupby('step')}}
    f = {k: 1.0 if code == 'bez' else float(np.exp(BI.s2_est(code, E, 'x', k) / 2)) for k in range(1, MAXSTEP + 1)}
    f.update({k: f[MAXSTEP] for k in range(MAXSTEP + 1, 11)})
    return f


def series(files, seed=None):
    """Ряд из одного или нескольких val-файлов: несколько — среднее по RV (как wrap2). seed — фильтр столбца seed, если есть."""
    parts = []
    for f in files:
        d = read_val(f)
        if seed is not None and 'seed' in d:
            d = d[d.seed == seed]
        parts.append(d[['date', 'ticker', 'year', 'step', 'rv_actual', 'rv_pred']].reset_index(drop=True))
    m = parts[0]
    for i, p in enumerate(parts[1:], 1):
        m = m.merge(p[['date', 'ticker', 'rv_pred']].rename(columns={'rv_pred': f'p{i}'}), on=['date', 'ticker'])
    if len(parts) > 1:
        m['rv_pred'] = m[['rv_pred'] + [f'p{i}' for i in range(1, len(parts))]].mean(axis=1)
    return m[['date', 'ticker', 'year', 'step', 'rv_actual', 'rv_pred']]


def qbar_T(q):
    return float(np.mean([q[s] for s in T]))


# ---------------------------------------------------------------- критерии на складке
def crit(g, steps, c1='sm', leg=True, runs=True):
    """c1–c4 варианта g (строки одного варианта) по обучающим шагам складки. Логика — D.flags без изменений:
    c1 — D.flags с q_har := q_har_c1 (HAR-J sm), если c1='sm' и столбец есть; c2 — D.flags и (если есть q_leg_v0) D.flags с нагой V0 в
    обоих ML-слотах; c3 — D.flags; c4 — D.flags по непрерывным отрезкам шагов (скачок <= 0,3), смены ML V1 — по всем шагам."""
    sub = g[g.step.isin(steps)].sort_values('step')
    assert len(sub) == len(steps), (g.variant.iloc[0] if 'variant' in g else '?', list(sub.step), steps)
    base = D.flags(sub)
    c1v = base['c1']
    if c1 == 'sm' and 'q_har_c1' in sub and sub.q_har_c1.notna().all() and not np.array_equal(sub.q_har_c1.values, sub.q_har.values):
        c1v = D.flags(sub.assign(q_har=sub.q_har_c1.values))['c1']
    c2v = base['c2']
    if leg and 'q_leg_v0' in sub and sub.q_leg_v0.notna().all():
        c2v = c2v and D.flags(sub.assign(q_xgb_single=sub.q_leg_v0.values, q_lgb_single=sub.q_leg_v0.values))['c2']
    c4v = base['c4']
    if runs:
        st = sub.step.values
        cut = np.where(np.diff(st) != 1)[0] + 1
        if len(cut):
            jump = max(D.flags(sub.iloc[s])['max_w_jump'] for s in np.split(np.arange(len(sub)), cut))
            c4v = bool(jump <= 0.3 and base['ml_switches'] <= 3)
    return dict(c1=bool(c1v), c2=bool(c2v), c3=bool(base['c3']), c4=bool(c4v))


def pick(G, order, steps, c1, sel=CR):
    """Первый по среднему q на steps вариант, у которого выполнены критерии sel (варианты перебираются по возрастанию среднего).
    -> (вариант | None, флаги c1–c4 выбранного или лучшего по q, число проверенных)."""
    first = None
    for k, v in enumerate(order):
        fl = crit(G[v], steps, c1)
        if first is None:
            first = (v, fl)
        if all(fl[x] for x in sel):
            return v, fl, k + 1
    return None, first[1], len(order)


MODES = {'vladelec_1420': dict(sel=tuple(CFG.get('vybor_kriterii', ['c3'])), pool='proshedshie'),
         'protokol_v2': dict(sel=CR, pool='vse')}


# ---------------------------------------------------------------- LOYO
def loyo(tab, F_unit, c1=None, mode=None):
    """tab — таблица шагов уровня одного h. -> (широкая таблица по единицам, длинная по складкам).
    mode 'vladelec_1420' (по умолчанию, решение 14:20): вариант в складке — по критериям vybor_kriterii (c3); вход — лучше эталона
    (q_har_c1 = HAR-J d0 sm) на отложенном году в >= etalon_folds_min складках и c3 на всех складках; ранги — среди прошедших и F_h.
    mode 'protokol_v2': c1–c4 на складке, недопущенные — хвостовой ранг среди всех единиц, вход — допуск на всех складках."""
    c1 = c1 or CFG['c1_etalon']
    mode = mode or CFG.get('rezhim', 'vladelec_1420')
    sel, pool_rule = MODES[mode]['sel'], MODES[mode]['pool']
    vmin = int(CFG.get('vhod', {}).get('etalon_folds_min', 3))
    tab = tab[tab.step.isin(T)]
    assert set(tab.step.unique()) <= set(T)
    units = list(dict.fromkeys(tab.unit))
    assert F_unit in units, F_unit
    G, var, Q = {}, {}, {}
    for (u, v), g in tab.groupby(['unit', 'variant'], sort=False):
        g = g.sort_values('step')
        if len(g) != len(T) or g.q_hyb.isna().any():
            print(f'ВНИМАНИЕ: {u} / {v}: нет q на всех шагах T — вариант пропущен', flush=True)
            continue
        G.setdefault(u, {})[v] = g
        var.setdefault(u, []).append(v)
        Q[(u, v)] = dict(zip(g.step, g.q_hyb))
    assert len(var[F_unit]) == 1, 'F_h — контрольная единица с одним вариантом'
    meta = tab.groupby('unit', sort=False).agg(role=('role', 'first') if 'role' in tab else ('unit', 'first'),
                                               n_rych=('n_rych', 'first') if 'n_rych' in tab else ('step', lambda s: 0))
    ok_units = [u for u in units if u in var]
    per = {}  # (u, Y) -> выбор в складке
    for Yst in T:
        tr = tuple(s for s in T if s != Yst)
        for u in ok_units:
            order = sorted(var[u], key=lambda v: (np.mean([Q[(u, v)][s] for s in tr]), var[u].index(v)))
            v, fl, nchk = pick(G[u], order, tr, c1, sel)
            vv = v if v is not None else order[0]
            qE = float(G[u][vv].set_index('step').q_har_c1[Yst]) if 'q_har_c1' in G[u][vv] else np.nan
            per[(u, Yst)] = dict(Y=Y(Yst), unit=u, variant=vv, dop=v is not None, q=Q[(u, vv)][Yst], q_etalon=qE,
                                 better_E=bool(Q[(u, vv)][Yst] < qE), q_train=float(np.mean([Q[(u, vv)][s] for s in tr])),
                                 n_checked=nchk, n_var=len(order), failed=','.join(k for k in CR if not fl[k]), **fl)
    dop_all = {u: all(per[(u, s)]['dop'] for s in T) for u in ok_units}
    nE = {u: sum(per[(u, s)]['better_E'] for s in T) for u in ok_units}
    vhod = {u: (dop_all[u] and nE[u] >= vmin) if mode == 'vladelec_1420' else dop_all[u] for u in ok_units}
    pool = [u for u in ok_units if vhod[u] or u == F_unit] if pool_rule == 'proshedshie' else ok_units
    long = []
    for Yst in T:
        P = [per[(u, Yst)] for u in ok_units]
        adm = [p['unit'] for p in P if p['unit'] in pool and p['dop']]
        rk = pd.Series({u: per[(u, Yst)]['q'] for u in adm}, dtype=float).rank(method='average')
        tail = (len(adm) + 1 + len(pool)) / 2  # недопущенные в складке единицы пула — средний ранг хвоста
        qF = per[(F_unit, Yst)]['q']
        for p in P:
            u = p['unit']
            p['rank'] = (float(rk[u]) if p['dop'] else tail) if u in pool else np.nan
            p['r'] = p['q'] / qF - 1
            p['better_F'] = bool(p['q'] < qF) if u != F_unit else False
            long.append(p)
    L = pd.DataFrame(long)
    rows = []
    for u in ok_units:
        g = L[L.unit == u].set_index('Y')
        order = sorted(var[u], key=lambda v: (qbar_T(Q[(u, v)]), var[u].index(v)))
        sv, flsel, _ = pick(G[u], order, T, c1, sel)
        r = dict(unit=u, role=meta.loc[u, 'role'], n_rych=meta.loc[u, 'n_rych'], n_variants=len(var[u]), is_F=u == F_unit)
        for y in g.index:
            r.update({f'var_{y}': g.loc[y, 'variant'], f'q_{y}': g.loc[y, 'q'], f'rank_{y}': g.loc[y, 'rank'],
                      f'dop_{y}': g.loc[y, 'dop'], f'fail_{y}': g.loc[y, 'failed'], f'r_{y}': g.loc[y, 'r'], f'qE_{y}': g.loc[y, 'q_etalon']})
        r.update(R_bar=g['rank'].mean() if u in pool else np.nan, mean_q=g.q.mean(), mean_r=g.r.mean(), sigma_r=g.r.std(ddof=1),
                 worst_year=int(g.r.idxmax()), worst_r=g.r.max(), dopusk=dop_all[u], n_better_E=nE[u], vhod=vhod[u],
                 c1234_all=bool(g[list(CR)].values.all()),  # справочно: c1–c4 у выбранных вариантов на всех складках
                 n_better_F=int(g.better_F.sum()), gate3=bool(g.better_F.sum() >= CFG['gate_min_folds']) if u != F_unit else False,
                 sel_all=sv if sv is not None else '', sel_all_q=qbar_T(Q[(u, sv)]) if sv is not None else np.nan,
                 sel_all_failed='' if sv is not None else ','.join(k for k in sel if not flsel[k]), rezhim=mode)
        rows.append(r)
    W = pd.DataFrame(rows)
    W['_rb'] = W.R_bar.fillna(np.inf)
    W = W.sort_values(['vhod', '_rb', 'mean_q', 'n_rych'], ascending=[False, True, True, True], kind='stable').drop(columns='_rb').reset_index(drop=True)
    W.insert(0, 'poryadok', range(1, len(W) + 1))
    return W, L


def vybor(W, roles=None, k=2):
    """Выход уровня: прошедшие условие входа единицы (без F_h) с наименьшим R̄; roles — фильтр ролей (уровень 1: признак, остаток, офсет)."""
    c = W[W.vhod & ~W.is_F]
    if roles is not None:
        c = c[c.role.isin(roles)]
    return c.head(k)


# ---------------------------------------------------------------- 2σ и ворота
def sigma2(cand, ref, seeds=SEEDS):
    """Д24.3: d_s = Q̄_T(канд, s) − Q̄_T(F_h, s); d̄ < −2σ/√n (σ — ddof = 1). Детерминированный ряд — число (одно на все seed);
    оба детерминированы — σ = 0, условие d̄ < 0."""
    val = lambda x, s: float(x[s]) if isinstance(x, dict) else float(x)
    d = np.array([val(cand, s) - val(ref, s) for s in seeds])
    n = len(d)
    dm, sg = float(d.mean()), float(d.std(ddof=1)) if n > 1 else 0.0
    thr = -2 * sg / np.sqrt(n)
    t = dm / (sg / np.sqrt(n)) if sg > 0 else (-np.inf if dm < 0 else (np.inf if dm > 0 else 0.0))
    return dict(n=n, d_mean=dm, sigma=sg, thr=thr, t=float(t), passed=bool(dm < thr), n_better=int((d < 0).sum()))


def vorota(W, unit, sig, test_ready=False):
    """Итоговые ворота кандидата; условия — раздельными колонками (дирижёр 13:24; владелец 14:20):
    u1 — условие входа (лучше эталона HAR-J d0 sm в >= 3 складках и c3 на всех; в режиме protokol_v2 — c1–c4 на всех складках);
    u2 — 2σ по 10 seed к F_h(d0); u3 — лучше F_h на отложенном году в >= 3 складках (протокол, п. 3); u4 — тестовые прогнозы собраны."""
    r = W[W.unit == unit].iloc[0]
    g = dict(unit=unit, u1_vhod=bool(r.vhod), n_better_E=int(r.n_better_E), u2_2sigma_Fd0=bool(sig['passed']) if sig else False,
             u3_luchshe_F=bool(r.gate3), n_better_F=int(r.n_better_F), u4_test=bool(test_ready),
             d_mean=sig['d_mean'] if sig else np.nan, sigma=sig['sigma'] if sig else np.nan, t=sig['t'] if sig else np.nan, R_bar=r.R_bar)
    g['proshyol'] = g['u1_vhod'] and g['u2_2sigma_Fd0'] and g['u3_luchshe_F'] and g['u4_test']
    return g


def fh_d0_seeds(h):
    """Q̄_T F_h(d0) по seed 42–51 (ряды ВЫР-1). -> {seed: Q̄_T} или None, пока файлов нет."""
    c = CFG['F_h_d0']
    fs = {s: R / c['file'].format(h=h, seed=s) for s in SEEDS}
    if not all(f.exists() for f in fs.values()):
        return None
    return {s: qbar_T(q_steps(series([f]), c['s2'])) for s, f in fs.items()}


# ---------------------------------------------------------------- адаптер: ряды -> таблица шагов (уровни 1–3)
def harcj_guard(files, what=''):
    """HAR-CJ (V0_HCJ gibrid_v17) строится по d−1 — в отборе R2 не берётся; допустим только HAR-CJ d0 с эмбарго (vyr/ref_harcj_d0, PR #154)."""
    d0 = str(R / Path(CFG['harcj_d0']).parent)
    for f in str(files).split(';'):
        p = str(f if Path(f).is_absolute() else R / f)
        if re.search(r'hcj|harcj|har_cj', Path(p).name, re.I):
            assert p.startswith(d0 + '/'), f'{what}: HAR-CJ по d−1 ({p}) — в R2 только d0 из {d0}'


def series_tab(spec, ref, h):
    """spec — DataFrame: unit, role, leg, variant, files (через ';'), s2, n_rych [, weights — csv step,w_har,w_xgb,w_lgb].
    ref — {'har': {код: {шаг: q}}, 'xgb_v0': ..., 'lgb_v0': ..., 'leg_v0:<нога>': ...}; HAR-J sm для c1 — ref['har']['sm'].
    Эталон для bez — ord (как gibrid_v17: «code == 'bez'» -> эталон ord), остальные — та же s²."""
    cache, rows = {}, []
    et = lambda code: 'ord' if code == 'bez' and CFG.get('etalon_bez', 'ord') == 'ord' else code
    for r in spec.itertuples():
        harcj_guard(r.files, f'{r.unit} / {r.variant}')
        key = (r.files, r.s2)
        if key not in cache:
            files = [x if Path(x).is_absolute() else R / x for x in str(r.files).split(';')]
            cache[key] = q_steps(series(files), r.s2)
        q = cache[key]
        e = et(r.s2)
        W = None
        if 'weights' in spec and isinstance(getattr(r, 'weights', None), str) and r.weights:
            W = pd.read_csv(r.weights).set_index('step')
        scheme = ('V0_HAR' if r.leg == 'HAR-J' else 'V0_LEG') if r.role == 'V0' else 'cell'
        lv = ref.get(f'leg_v0:{r.leg}')
        for st in T:
            rows.append(dict(unit=r.unit, variant=r.variant, step=st, role=r.role, n_rych=getattr(r, 'n_rych', 0), q_hyb=q[st],
                             q_har=ref['har'][e][st], q_har_c1=ref['har']['sm'][st], q_xgb_single=ref['xgb_v0'][e][st],
                             q_lgb_single=ref['lgb_v0'][e][st], q_leg_v0=lv[e][st] if (lv is not None and r.role != 'V0') else np.nan,
                             scheme=scheme, v1_ml='', w_har=W.loc[st, 'w_har'] if W is not None else 0.0,
                             w_xgb=W.loc[st, 'w_xgb'] if W is not None else 0.0, w_lgb=W.loc[st, 'w_lgb'] if W is not None else 0.0))
    return pd.DataFrame(rows)


def ref_from_files(files_by_key):
    """{'har': путь, 'xgb_v0': путь, ...} -> {ключ: {код: {шаг: q}}} по всем s²."""
    out = {}
    for k, f in files_by_key.items():
        harcj_guard(f, f'эталон {k}')
        s = series([f if Path(f).is_absolute() else R / f])
        out[k] = {c: q_steps(s, c) for c in CODES}
    return out


# ---------------------------------------------------------------- запись
def write_level(W, L, level, h, out=OUT, extra=None):
    out.mkdir(parents=True, exist_ok=True)
    f = out / f'level{level}_h{h}.csv'
    W.to_csv(f, index=False, float_format='%.6g')
    L.to_csv(out / f'level{level}_h{h}_folds.csv', index=False, float_format='%.6g')
    m = dict(time=now(), level=level, h=h, code_sha=sha(__file__), gibrid_designs_sha=sha(HERE / 'gibrid_designs.py'),
             yaml_sha=sha(REPO / 'models/designs/r2/r2_O_loyo.yaml'), units=int(len(W)), variants=int(W.n_variants.sum()), **(extra or {}))
    (out / f'level{level}_h{h}.json').write_text(json.dumps(m, ensure_ascii=False, indent=1, default=str))
    return f


# ---------------------------------------------------------------- проверка механики на рядах первого раунда (не выбор)
def harj_sm(d):
    """HAR-J с поправкой sm по шагам — q_har строк дизайнов с s² sm (gibrid_v17 --etalon same)."""
    return d[d.design.str.contains(r'\|B-sm\|')].groupby('step').q_har.first()


def tab_r1(h):
    pr = CFG['proverka_r1']
    rx = re.compile(pr['isklyuchit_designs'])
    parts, src = [], []
    for yf in sorted((R / pr['kandidaty']).glob('*.yaml')):
        y = yaml.safe_load(yf.read_text())
        hs = y['h'] if isinstance(y['h'], list) else [y['h']]
        if h not in hs or y.get('odinochnaya'):
            continue
        b = y['boost']
        p = next((R / gd / b / f'designs_val_{b}_h{h}.csv' for gd in pr['gibrid_dirs'] if (R / gd / b / f'designs_val_{b}_h{h}.csv').exists()), None)
        if p is None:
            print(f'h={h}: {b} — нет designs_val, пропуск', flush=True)
            continue
        d = read_designs(p)
        hs_ = harj_sm(d)
        d = d[~d.design.str.contains(rx)]
        d = d.assign(unit=b, variant=d.design, role='kandidat_r3' if 'final3_dry' not in str(p) else 'D25_posle_otkrytiya',
                     n_rych=1, q_har_c1=d.step.map(hs_).values)
        parts.append(d); src.append((b, str(p), sha(p)))
    fc = CFG['F_h'][h]
    p = R / fc['designs_val']
    d = read_designs(p)
    hs_ = harj_sm(d)
    d = d[d.design == fc['design']]
    assert len(d) == len(T), (p, fc['design'])
    parts.append(d.assign(unit=fc['name'], variant=d.design, role='F_h', n_rych=0, q_har_c1=d.step.map(hs_).values))
    src.append((fc['name'], str(p), sha(p)))
    return pd.concat(parts, ignore_index=True), src


def proverka(hs):
    pr = CFG['proverka_r1']
    out = R / pr['out']
    summ = []
    for h in hs:
        tab, src = tab_r1(h)
        # 1) сверка: crit(c1='same', все шаги T, без отрезков) = сохранённые c1_h..c4_h gibrid_v17 (шаги 2–6) — флаги импортом воспроизводятся
        full = [k + '_h' for k in CR]
        mism = 0
        stored = {}
        for b, path, _ in src:
            s = pd.read_csv(path, usecols=['design', 'step'] + full)
            s = s[s.step.isin(T)].groupby('design')[full].first()
            stored[b] = s
        for (u, v), g in tab.groupby(['unit', 'variant'], sort=False):
            fl = crit(g, T, c1='same', leg=False, runs=False)
            st = stored[u].loc[v]
            if any(bool(st[k + '_h']) != fl[k] for k in CR):
                mism += 1
        # 2) LOYO
        F = CFG['F_h'][h]['name']
        W, L = loyo(tab, F)
        write_level(W, L, 1, h, out, extra=dict(proverka_r1=True, sources=src))
        n_fold_c1 = int(sum((W[f'fail_{y}'].fillna('').str.contains('c1')).sum() for y in range(2017, 2022)))
        summ.append(dict(h=h, units=len(W), variants=int(W.n_variants.sum()), flags_checked=len(tab.groupby(['unit', 'variant'])),
                         flags_mismatch=mism, c3_vse=int(W.dopusk.sum()), vhod=int(W.vhod.sum()), gate3=int(W.gate3.sum())))
        cols = ['poryadok', 'unit', 'role', 'n_variants', 'R_bar', 'mean_q', 'mean_r', 'sigma_r', 'dopusk', 'n_better_E', 'vhod', 'c1234_all',
                'n_better_F', 'gate3', 'sel_all']
        print(f'\n=== h={h}: {len(W)} единиц, {int(W.n_variants.sum())} вариантов; сверка флагов с gibrid_v17: расхождений {mism} ===', flush=True)
        print(W[cols].to_string(index=False, float_format=lambda x: f'{x:.4f}'), flush=True)
    S = pd.DataFrame(summ)
    S.to_csv(out / 'svodka.csv', index=False)
    print('\n', S.to_string(index=False), flush=True)
    return S


def proverka_series():
    """Адаптер рядов и 2σ на сохранённых рядах (val шагов <= 6): (а) q_steps(wrap2 seed 42, sm) = q_hyb F_5 из designs_val;
    (б) мини-уровень 1 на h=5 (HAR-X / HARX0 V0 × s², Bv3 V0); (в) 2σ: LGB V0 sm и HAR-J sm против F_5 на seed 42–51."""
    pr = CFG['proverka_r1']
    out = R / pr['out']
    sd, se = pr['sigma_demo'], pr['series_demo']
    h = sd['h']
    dv = read_designs(R / se['harj_designs_val'])
    F = CFG['F_h'][h]
    qF_saved = dv[dv.design == F['design']].set_index('step').q_hyb
    w2 = series([R / sd['wrap2_s42']])
    qw = q_steps(w2, 'sm')
    da = max(abs(qw[s] - qF_saved[s]) for s in T)
    x42, l42 = series([R / sd['xgb'][0]], 42), series([R / sd['lgbm'][0]], 42)
    m = x42.merge(l42[['date', 'ticker', 'rv_pred']], on=['date', 'ticker'], suffixes=('', '_l'))
    m['rv_pred'] = (m.rv_pred + m.rv_pred_l) / 2
    mw = m.merge(w2[['date', 'ticker', 'rv_pred']], on=['date', 'ticker'], suffixes=('', '_w'))
    dw = float(np.abs(mw.rv_pred - mw.rv_pred_w).max())
    print(f'(а) wrap2 seed 42: |q_steps(sm) − designs_val F_5| max = {da:.2e}; |(xgb+lgbm)/2 − wrap2| max = {dw:.2e}', flush=True)
    # (б) мини-уровень 1 на рядах
    har = {c: dv[dv.design == f'single|V0_HAR|{"bez" if c == "bez" else "B-" + c}|L0'].set_index('step').q_hyb.to_dict() for c in CODES}
    ref = ref_from_files({'xgb_v0': se['xgb_v0'], 'lgb_v0': se['lgb_v0']})
    ref['har'] = har
    rows = []
    for leg, f in se['legs'].items():
        for c in CODES:
            rows.append(dict(unit=f'{h}|{leg}|V0', role='V0', leg=leg, variant=f'{leg}|{c}', files=f, s2=c, n_rych=1))
    for nm, f in (('XGB', se['xgb_v0']), ('LGB', se['lgb_v0'])):
        for c in CODES:
            rows.append(dict(unit=f'{h}|{nm}|V0', role='V0', leg=nm, variant=f'{nm}|{c}', files=f, s2=c, n_rych=0))
    for c in CODES:  # остаток: D12_h5 (V9) XGB / LGB / среднее — нога HAR-J
        for mname, f in (('xgb', sd['xgb'][0]), ('lgbm', sd['lgbm'][0]), ('wrap2', sd['xgb'][0] + ';' + sd['lgbm'][0])):
            rows.append(dict(unit=f'{h}|HAR-J|ostatok', role='ostatok', leg='HAR-J', variant=f'{mname}|{c}', files=f, s2=c, n_rych=1))
    spec = pd.DataFrame(rows)
    tab = series_tab(spec, ref, h)
    tab = pd.concat([tab, dv[dv.design == F['design']].assign(unit=F['name'], variant=F['design'], role='F_h', n_rych=0,
                                                              q_har_c1=pd.Series(har['sm']).reindex(T).values)], ignore_index=True)
    W, L = loyo(tab, F['name'])
    write_level(W, L, 'S', h, out, extra=dict(proverka_r1=True, series_demo=True))
    print(f'(б) мини-уровень 1 на рядах, h={h}:', flush=True)
    print(W[['poryadok', 'unit', 'role', 'n_variants', 'R_bar', 'mean_q', 'mean_r', 'dopusk', 'n_better_E', 'vhod', 'n_better_F', 'gate3', 'sel_all']].to_string(
        index=False, float_format=lambda x: f'{x:.4f}'), flush=True)
    # (в) 2σ по 10 seed
    qF, qL = {}, {}
    for s in SEEDS:
        fx = sd['xgb'][0] if s == 42 else (sd['xgb'][1] if s in (43, 44) else sd['xgb'][2])
        fl = sd['lgbm'][0] if s == 42 else (sd['lgbm'][1] if s in (43, 44) else sd['lgbm'][2])
        xs, ls = series([R / fx], s), series([R / fl], s)
        assert len(xs) and len(ls), s
        mm = xs.merge(ls[['date', 'ticker', 'rv_pred']], on=['date', 'ticker'], suffixes=('', '_l'))
        mm['rv_pred'] = (mm.rv_pred + mm.rv_pred_l) / 2
        qF[s] = qbar_T(q_steps(mm, 'sm'))
        qL[s] = qbar_T(q_steps(ls, 'sm'))
    qH = qbar_T(har['sm'])
    sL, sH = sigma2(qL, qF), sigma2(qH, qF)
    sFF = sigma2(qbar_T(har['sm']), qbar_T(har['sm']))
    print(f'(в) Q̄_T F_5 seed 42 = {qF[42]:.6f} (designs_val: {qbar_T(qF_saved.to_dict()):.6f}); F_5 по seed: '
          + ', '.join(f'{qF[s]:.4f}' for s in SEEDS), flush=True)
    for nm, sg in (('LGB V0 sm (D12_h5) vs F_5', sL), ('HAR-J sm (детерм.) vs F_5', sH), ('HAR-J vs HAR-J (оба детерм.)', sFF)):
        print(f'    {nm}: d̄ = {sg["d_mean"]:+.5f}, σ = {sg["sigma"]:.5f}, порог −2σ/√10 = {sg["thr"]:+.5f}, t = {sg["t"]:.2f}, '
              f'прошёл = {sg["passed"]}, лучше в {sg["n_better"]}/10 seed', flush=True)
    # ворота на мини-уровне (механика): HAR-X V0 (детерминированный; вариант — Sel на 5 годах, иначе лучший по среднему q) и остаток
    G = []
    for u in (f'{h}|HARX|V0', f'{h}|HAR-J|ostatok'):
        r = W[W.unit == u].iloc[0]
        tu = tab[tab.unit == u]
        v = r.sel_all or tu.groupby('variant').q_hyb.mean().idxmin()
        qv = qbar_T(tu[tu.variant == v].set_index('step').q_hyb.to_dict())
        cand = qv if 'HARX' in u else {s: (qv if s == 42 else np.nan) for s in SEEDS}
        sg = sigma2(cand, qF) if 'HARX' in u else None  # у остатка в демо есть только seed 42 — 2σ не считается
        G.append(dict(vorota(W, u, sg), variant=v, qbar_T=qv))
    print('(г) ворота (механика):', flush=True)
    print(pd.DataFrame(G).to_string(index=False, float_format=lambda x: f'{x:.4f}'), flush=True)
    res = dict(wrap2_q_maxdiff=da, wrap2_pred_maxdiff=dw, qF=qF, qL=qL, qH=qH, sigma_lgb=sL, sigma_harj=sH, sigma_det=sFF)
    (out / 'series_sigma.json').write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float))
    pd.DataFrame(G).to_csv(out / 'vorota_demo.csv', index=False)
    return res


# ---------------------------------------------------------------- самопроверка на синтетике
def selftest():
    rng = np.random.default_rng(7)

    def unit(u, nv, base, bad=None, wj=None, role='ostatok'):
        rows = []
        for v in range(nv):
            q = base + rng.normal(0, 0.003, len(T)) + 0.002 * v
            for i, st in enumerate(T):
                w = (0.0, 0.0, 0.0) if wj is None else wj[i]
                qq = q[i] * (1.5 if (bad is not None and v in bad[0] and st in bad[1]) else 1.0)
                rows.append(dict(unit=u, variant=f'{u}.v{v}', step=st, role=role, n_rych=1, q_hyb=qq, q_har=0.30, q_har_c1=0.30,
                                 q_xgb_single=0.31, q_lgb_single=0.31, scheme='cell', v1_ml='', w_har=w[0], w_xgb=w[1], w_lgb=w[2]))
        return rows
    rows = unit('F', 1, 0.29, role='F_h') + unit('A', 4, 0.25) + unit('B', 3, 0.28)
    rows += unit('C', 2, 0.26, bad=([0, 1], [3]))  # все варианты C нарушают c1/c2 в 2018 -> допущена только в складке Y=2018
    wj = [(0, 1, 0), (0, 1, 0), (0, .5, .5), (0, 0, 1), (0, 0, 1)]  # скачок 0,5 на парах 3-4 и 4-5; на [2,3] и [5,6] — 0
    rows += unit('E', 1, 0.255, wj=wj)
    tab = pd.DataFrame(rows)
    W, L = loyo(tab, 'F', mode='protokol_v2')  # тесты 1–5, 8 — режим протокола v2 (c1–c4, хвост по всем единицам)
    w = W.set_index('unit')
    # 1. C: допущена только в складке 2018 (обучающие годы без 2018); в остальных — хвостовой ранг
    lc = L[L.unit == 'C'].set_index('Y')
    assert lc.loc[2018, 'dop'] and not lc.drop(2018).dop.any(), lc[['dop', 'failed']]
    n = L.groupby('Y').unit.count().iloc[0]
    for yy in (2017, 2019, 2020, 2021):
        nad = int((~L[L.Y == yy].dop).sum())
        assert lc.loc[yy, 'rank'] == (n - nad + 1 + n) / 2
    # 2. c4 по отрезкам: E допущена только в складке 2019 (шаг 4 отложен: отрезки [2,3] и [5,6], скачков нет)
    le = L[L.unit == 'E'].set_index('Y')
    assert le.loc[2019, 'dop'] and le.loc[2019, 'c4'] and not le.drop(2019).c4.any(), le[['dop', 'c4']]
    g = tab[tab.unit == 'E']
    assert not D.flags(g[g.step.isin([2, 3, 5, 6])].sort_values('step'))['c4'], 'без отрезков flags «склеил» бы 2018 и 2020'
    # 3. нет утечки: выбор варианта в складке Y не зависит от q на Y (шум на отложенном году у всех вариантов)
    for yy in range(2017, 2022):
        t2 = tab.copy()
        m = t2.step == yy - 2015
        t2.loc[m, 'q_hyb'] *= np.exp(rng.normal(0, 0.2, m.sum()))
        for md in MODES:
            _, L1 = loyo(tab, 'F', mode=md)
            _, L2 = loyo(t2, 'F', mode=md)
            a = L1[L1.Y == yy].set_index('unit').variant
            b = L2[L2.Y == yy].set_index('unit').variant
            assert (a == b).all(), (yy, md)
    # 4. ранги: сумма рангов в складке = n(n+1)/2; R̄ — среднее рангов
    for yy, gL in L.groupby('Y'):
        assert abs(gL['rank'].sum() - len(gL) * (len(gL) + 1) / 2) < 1e-9
    assert abs(w.loc['A', 'R_bar'] - L[L.unit == 'A']['rank'].mean()) < 1e-12
    # 5. A лучше F во всех складках -> gate3; F сам себе — нет
    assert w.loc['A', 'n_better_F'] == 5 and w.loc['A', 'gate3'] and not w.loc['F', 'gate3']
    # 6. 2σ: детерминированные оба -> σ = 0, d̄ < 0; случайный шум вокруг 0 не проходит; явный выигрыш проходит
    assert sigma2(0.20, 0.21)['passed'] and not sigma2(0.21, 0.21)['passed']
    ref = {s: 0.25 + 0.001 * (s - 46) for s in SEEDS}
    assert sigma2({s: ref[s] - 0.01 + 1e-4 * ((-1) ** s) for s in SEEDS}, ref)['passed']
    assert not sigma2({s: ref[s] + 0.002 * ((-1) ** s) for s in SEEDS}, ref)['passed']
    s = sigma2({s: ref[s] - 0.001 + 0.003 * ((-1) ** s) for s in SEEDS}, ref)
    assert abs(s['thr'] - (-2 * s['sigma'] / np.sqrt(10))) < 1e-15 and not s['passed']
    # 7. чтение: шаги 7+ не загружаются (фильтр), q_steps — OOS: шаг 2 использует только остатки шага 1
    import tempfile
    tmp = Path(tempfile.mkdtemp())
    dd = pd.DataFrame([(pd.Timestamp(f'{2015 + st}-06-01') + pd.Timedelta(days=i), 'T', 2015 + st, st, 1.0, np.exp(-0.1 * st))
                       for st in range(1, 11) for i in range(5)], columns=['date', 'ticker', 'year', 'step', 'rv_actual', 'rv_pred'])
    dd.to_parquet(tmp / 'x_val.parquet')
    rd = read_val(tmp / 'x_val.parquet')
    assert rd.step.max() == 6 and rd.year.max() == 2021
    q1 = q_steps(rd, 'ord')
    dd2 = rd.copy()
    dd2.loc[dd2.step == 3, 'rv_pred'] *= 3  # правка шага 3 не меняет q шага 2
    assert q_steps(dd2, 'ord')[2] == q1[2] and q_steps(dd2, 'ord')[3] != q1[3]
    f = factors(rd, 'sm')
    assert f[7] == f[6] == f[10]
    # 8. выбор уровня 1: только роли с бустингом, допущенные, без F
    assert list(vybor(W, roles=['ostatok']).unit) == [u for u in W[W.vhod & ~W.is_F & (W.role == 'ostatok')].unit][:2]
    assert (W.vhod == W.dopusk).all()  # в режиме протокола вход = c1–c4 на всех складках
    # 9. режим владельца 14:20: вход — лучше эталона (q_har_c1 = 0,30) в >= 3 складках и c3 на всех; ранги — среди прошедших и F_h
    def fixed(u, qs):
        return [dict(unit=u, variant=f'{u}.v0', step=st, role='ostatok', n_rych=1, q_hyb=q, q_har=0.30, q_har_c1=0.30, q_xgb_single=0.31,
                     q_lgb_single=0.31, scheme='cell', v1_ml='', w_har=0.0, w_xgb=0.0, w_lgb=0.0) for st, q in zip(T, qs)]
    t9 = pd.DataFrame(rows + fixed('G', [0.28, 0.28, 0.305, 0.305, 0.305])  # c3 везде, лучше эталона только в 2 складках
                      + fixed('H', [0.25, 0.25, 0.32, 0.25, 0.25]))  # лучше эталона в 4, но c3 нарушен, когда 2018 в обучении
    W9, L9 = loyo(t9, 'F', mode='vladelec_1420')
    w9 = W9.set_index('unit')
    assert w9.loc['G', 'n_better_E'] == 2 and w9.loc['G', 'dopusk'] and not w9.loc['G', 'vhod']
    assert w9.loc['H', 'n_better_E'] == 4 and not w9.loc['H', 'dopusk'] and not w9.loc['H', 'vhod']
    assert not w9.loc['C', 'vhod']  # c3 нарушен в складках, где 2018 в обучении
    assert w9.loc['E', 'vhod'] and not w9.loc['E', 'c1234_all']  # c4 во вход не входит (справочно — нарушен)
    assert w9.loc[['A', 'B'], 'vhod'].all() and np.isnan(w9.loc['G', 'R_bar']) and np.isnan(w9.loc['H', 'R_bar'])
    pool = [u for u in w9.index if w9.loc[u, 'vhod'] or u == 'F']
    for yy, gL in L9.groupby('Y'):
        gp = gL[gL.unit.isin(pool)]
        assert gL[~gL.unit.isin(pool)]['rank'].isna().all() and abs(gp['rank'].sum() - len(gp) * (len(gp) + 1) / 2) < 1e-9
    assert set(vybor(W9, k=10).unit) == set(pool) - {'F'} and list(W9.poryadok[:len(pool)]) == list(range(1, len(pool) + 1))
    # 10. HAR-CJ: по d−1 — отказ; d0 из vyr/ref_harcj_d0 — допустим; прочие ряды не затрагиваются
    harcj_guard(CFG['harcj_d0'].format(h=5)); harcj_guard('Bv3_xgb_h5_val.parquet')
    for bad in ('ref_v0_base/HCJ_h5_val.parquet', 'HARCJ_h5_val.parquet'):
        try:
            harcj_guard(bad, 'тест')
            raise AssertionError('HAR-CJ по d−1 не пойман')
        except AssertionError as e:
            assert 'd−1' in str(e), e
    g9 = vorota(W9, 'A', sigma2({s: 0.25 for s in SEEDS}, {s: 0.29 + 1e-3 * (s % 3) for s in SEEDS}))
    assert g9['u1_vhod'] and g9['u2_2sigma_Fd0'] and g9['u3_luchshe_F'] and not g9['u4_test'] and not g9['proshyol']
    print('selftest r2_loyo: OK (протокол v2: ранги и хвост, c4 по отрезкам; режим владельца 14:20: вход «эталон >= 3 из 5 + c3», ранги среди '
          'прошедших; нет утечки отложенного года в обоих; 2σ; чтение шагов <= 6, OOS s², выбор, ворота раздельно)')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', nargs='?', choices=['proverka', 'level'])
    ap.add_argument('--selftest', action='store_true')
    ap.add_argument('--h', type=int, nargs='*', default=[1, 5, 22])
    ap.add_argument('--level', type=int, default=1)
    ap.add_argument('--no-series', action='store_true', help='proverka: без адаптера рядов и 2σ')
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if a.cmd == 'proverka':
        proverka(a.h)
        if not a.no_series:
            proverka_series()
        return
    if a.cmd == 'level':
        lc = CFG['level1']  # формат спецификации общий; уровни 2–3 — spec_level{2,3}_h{h}.csv (ряды Т / У)
        for h in a.h:
            spec = pd.read_csv(R / lc['spec'].replace('level1', f'level{a.level}').format(h=h))
            rf = pd.read_csv(R / lc['ref'].format(h=h))  # key, file
            files = dict(zip(rf.key, rf.file))
            files['har'] = CFG['etalon']['file'].format(h=h)  # владелец 14:20: эталон — HAR-J d0 с эмбарго (Э, HARJ_emb_d0), c1 — его sm
            ref = ref_from_files(files)
            tab = series_tab(spec, ref, h)
            fd = R / CFG['F_h_d0']['file'].format(h=h, seed=42)
            if fd.exists():  # F_h(d0) — финал первого раунда на ногах d0 (ВЫР-1), seed 42 — для рангов
                Fname = f'F_{h}(d0)'
                tab = pd.concat([tab, series_tab(pd.DataFrame([dict(unit=Fname, role='F_h', leg='F', variant=fd.name, files=str(fd),
                                                                     s2=CFG['F_h_d0']['s2'], n_rych=0)]), ref, h)], ignore_index=True)
            else:  # до выкладки ВЫР-1 — прежний F_h, справочно
                F = CFG['F_h'][h]
                Fname = F['name'] + ' [справочно: без выравнивания d0]'
                print(f'ВНИМАНИЕ h={h}: нет {fd} — F_h без выравнивания (справочно)', flush=True)
                dv = read_designs(R / F['designs_val'])
                fr = dv[dv.design == F['design']]
                tab = pd.concat([tab, fr.assign(unit=Fname, variant=F['design'], role='F_h', n_rych=0,
                                                q_har_c1=fr.step.map(ref['har']['sm']).values)], ignore_index=True)
            W, L = loyo(tab, Fname)
            write_level(W, L, a.level, h)
            if a.level == 1:
                vybor(W, roles=lc['boost_roles']).to_csv(OUT / f'vybor_level1_h{h}.csv', index=False)
        return
    ap.print_help()


if __name__ == '__main__':
    main()
