#!/usr/bin/env python3
"""R2, агент O: движок отбора второго раунда — протокол v3 (docs/zadaniya/2026-10-26_vtoroy_raund_gibrida.md, PR #160), пп. 2.1–2.7, 2.9, 4;
yaml models/designs/r2/r2_O_loyo.yaml. Факторный скрининг (разд. 3) — scripts/analysis/r2_factor.py.

Единица u (дизайн / ячейка / точка сетки) имеет варианты (значения-списки: сетки λ, α, κ, …). Вход ядра — «таблица шагов»: строка = (unit,
variant, step ∈ T), столбцы как у designs_val gibrid_v17: q_hyb, q_har (HAR-J той же s²), q_xgb_single, q_lgb_single, scheme, w_*, v1_ml;
+ q_har_c1 (E_h — HAR-J+s² sm d0), q_leg_v0 (нога u в роли V0, для c2), calib_param (для аналога c4), role, slozhnost.
- п. 2.1: в складке Y вариант — мин. средний q на T\\{Y} среди вариантов с c3 на T\\{Y}; q_u(Y) — его q на Y; Y во вложенный выбор не входит.
- п. 2.2: вход — (а) q_u(Y) < q_E(Y) не меньше чем в 3 складках; (б) c3 на всех складках; (в) c1, c2, c4 на всех 5 годах T по варианту,
  выбранному на T (❓ 1, умолчание). c1, c3 — против E_h; c2 — против одиночных той же s²; c4 — скачок весов ≤ 0,3 и скачок параметра
  калибровки ≤ 50%. Флаги — gibrid_designs.flags (импорт, без изменений; GIBRID_SAME_SLOT=1).
- п. 2.3: ранги q_u(Y) среди прошедших вход и F_h(d0) (средний при равенстве); R̄ = среднее по 5 складкам; рядом q/E, q/F, σ по годам.
- п. 2.4: штраф за сложность — из «неотличимых» (R̄ − R̄* ≤ 1 или средний q ≤ 1,01·q*) берётся наименьшая сложность c(u) (schema_v2 = v1 + кластеры 16, 18, 20).
- п. 2.5: финалисты (до 3 после штрафа + V0-кандидаты, прошедшие вход), 2σ к F_h(d0) по 10 seed (Д24.3), победитель.
- п. 2.6: сдвиг информационного множества — Спирмен ln ŷ_d с ln RV_{d+s}, s ∈ {−2,−1,0,+1}, медиана по тикерам; argmax ≠ 0 — блок,
  запись в R2/O/blok_sdvig.csv.
- п. 2.7: число дизайнов по уровням и ожидаемое число ложных победителей.
- п. 2.9, разд. 4: гейты R2/O/gate_{C,D,E}_h{h}.yaml с хэшами входов (gate_B — r2_factor.py).
Читаются только val шагов ≤ 6 (годы ≤ 2021): фильтр при чтении parquet и assert; дневная RV — по 2021-12-31.

python3 scripts/analysis/r2_loyo.py --selftest
python3 scripts/analysis/r2_loyo.py proverka [--h 1 5 22]       # механика на сохранённых рядах первого раунда (не выбор)
python3 scripts/analysis/r2_loyo.py level --level 1 --h 1        # фаза C: level1_h1.csv, gate_C_h1.yaml (спецификация рядов — yaml level)
python3 scripts/analysis/r2_loyo.py gate-d --h 1                 # фаза D: отсечение 60 → 30 → 15 → 8 → 1, gate_D_h1.yaml
python3 scripts/analysis/r2_loyo.py level --level 3 --h 1        # фаза E: level3, финалисты, 2σ, gate_E_h1.yaml (spec с files10)"""
import argparse, hashlib, json, os, re, subprocess, sys
from pathlib import Path
import numpy as np, pandas as pd
import yaml
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
os.environ['GIBRID_SAME_SLOT'] = '1'  # одиночная модель в обоих слотах (HEAVY): слот, равный дизайну, не эталон c2/c3 (FD §3 п. 1)
import gibrid_designs as D  # noqa: E402  c1–c4 — без изменений
import gibrid_bias as BI  # noqa: E402  оценки s² (ord/rob/sm/smrob) — без изменений

REPO = HERE.parent.parent
CFG_PATH = REPO / 'models' / 'designs' / 'r2' / 'r2_O_loyo.yaml'
CFG = yaml.safe_load(CFG_PATH.read_text())
V3 = CFG['v3']
R = Path(CFG['data'])
OUT = Path(CFG['out'])
T = tuple(CFG['steps_T'])
MAXSTEP = int(CFG['max_step_read'])
SEEDS = tuple(CFG['seeds'])
CR = ('c1', 'c2', 'c3', 'c4')
CODES = tuple(CFG['s2_codes'])
NEED = ['q_hyb', 'q_har', 'q_xgb_single', 'q_lgb_single', 'scheme', 'w_har', 'w_xgb', 'w_lgb', 'v1_ml']
Y = lambda s: 2015 + int(s)
MODES = {'v3': dict(sel=tuple(V3['vybor_kriterii']), c3_E=True, entry='v3', pool='proshedshie'),
         'protokol_v2': dict(sel=CR, c3_E=False, entry='folds', pool='vse')}


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
    d = pd.read_csv(path, usecols=['design', 'step', 'val_year'] + NEED)
    assert d.loc[d.step > MAXSTEP, 'q_hyb'].isna().all(), f'{path}: метрики шагов 7+'
    d = d[d.step.isin(T)].copy()
    assert (d.val_year == d.step + 2015).all()
    return d


# ---------------------------------------------------------------- ряды -> q по шагам
def q_steps(df, code, steps=T):
    """OOS QLIKE по шагам: s² шага t — BI.s2_est(code) по ln-остаткам шагов <= t−1 (F = exp(s²/2)); bez и шаг 1 — F = 1.
    Фильтр rv_actual > 1e-12 (как gibrid_io.merged). -> {шаг: q}."""
    d = df[df.rv_actual > 1e-12]
    assert d.step.max() <= MAXSTEP
    E = {'x': {st: np.log(g.rv_actual.values + 1e-10) - np.log(g.rv_pred.values) for st, g in d.groupby('step')}}
    out = {}
    for st in steps:
        g = d[d.step == st]
        fac = 1.0 if (code == 'bez' or st == 1) else float(np.exp(BI.s2_est(code, E, 'x', st - 1) / 2))
        y, p = g.rv_actual.values, np.clip(g.rv_pred.values * fac, 1e-10, None)
        out[st] = float(np.mean(y / p - np.log(y / p) - 1))
    return out


def factors(df, code):
    """Множители Б0.1 для тестовой сборки: тест шага k <= 6 — по val шагов <= k; k >= 7 — заморозка на шаге 6. -> {k: F_k}, k = 1..10."""
    d = df[df.rv_actual > 1e-12]
    E = {'x': {st: np.log(g.rv_actual.values + 1e-10) - np.log(g.rv_pred.values) for st, g in d.groupby('step')}}
    f = {k: 1.0 if code == 'bez' else float(np.exp(BI.s2_est(code, E, 'x', k) / 2)) for k in range(1, MAXSTEP + 1)}
    f.update({k: f[MAXSTEP] for k in range(MAXSTEP + 1, 11)})
    return f


def series(files, seed=None, col='rv_pred', agg='mean'):
    """Ряд из одного или нескольких val-файлов: несколько — агрегат по RV (mean — как wrap2 / среднее seed × lib; median — ens.agg = median,
    п. 2.1). seed — фильтр столбца seed, если он есть; col — столбец прогноза."""
    parts = []
    for f in files:
        d = read_val(f)
        if seed is not None and 'seed' in d:
            d = d[d.seed == seed]
        parts.append(d[['date', 'ticker', 'year', 'step', 'rv_actual', col]].rename(columns={col: 'rv_pred'}).reset_index(drop=True))
    m = parts[0]
    for i, p in enumerate(parts[1:], 1):
        m = m.merge(p[['date', 'ticker', 'rv_pred']].rename(columns={'rv_pred': f'p{i}'}), on=['date', 'ticker'])
    if len(parts) > 1:
        P = m[['rv_pred'] + [f'p{i}' for i in range(1, len(parts))]]
        m['rv_pred'] = P.median(axis=1) if agg == 'median' else P.mean(axis=1)
    return m[['date', 'ticker', 'year', 'step', 'rv_actual', 'rv_pred']]


def qbar_T(q):
    return float(np.mean([q[s] for s in T]))


# ---------------------------------------------------------------- критерии
def crit(g, steps, c1_E=True, c3_E=True, leg=True, runs=True):
    """c1–c4 варианта g по шагам steps. Логика — D.flags без изменений, входы:
    c1, c3 — D.flags с q_har := q_har_c1 (E_h), если c1_E / c3_E и столбец есть (иначе — HAR-J той же s², как gibrid_v17 --etalon same);
    c2 — D.flags (HAR-J той же s², ML) и, если есть q_leg_v0, D.flags с ногой V0 в обоих ML-слотах;
    c4 — D.flags по непрерывным отрезкам шагов (скачок весов <= 0,3; смены ML V1 — по всем шагам) и, если есть calib_param,
         скачок параметра калибровки между соседними шагами <= calib_jump_max (аналог c4, п. 2.2 в)."""
    sub = g[g.step.isin(steps)].sort_values('step')
    assert len(sub) == len(steps), (g.variant.iloc[0] if 'variant' in g else '?', list(sub.step), steps)
    base = D.flags(sub)
    out = dict(c1=base['c1'], c2=base['c2'], c3=base['c3'], c4=base['c4'])
    if 'q_har_c1' in sub and sub.q_har_c1.notna().all() and (c1_E or c3_E):
        fe = D.flags(sub.assign(q_har=sub.q_har_c1.values))
        if c1_E:
            out['c1'] = fe['c1']
        if c3_E:
            out['c3'] = fe['c3']
    if leg and 'q_leg_v0' in sub and sub.q_leg_v0.notna().all():
        out['c2'] = out['c2'] and D.flags(sub.assign(q_xgb_single=sub.q_leg_v0.values, q_lgb_single=sub.q_leg_v0.values))['c2']
    st = sub.step.values
    segs = np.split(np.arange(len(sub)), np.where(np.diff(st) != 1)[0] + 1) if runs else [np.arange(len(sub))]
    if runs and len(segs) > 1:
        jump = max(D.flags(sub.iloc[s])['max_w_jump'] for s in segs)
        out['c4'] = bool(jump <= 0.3 and base['ml_switches'] <= 3)
    if 'calib_param' in sub and sub.calib_param.notna().all():
        p = sub.calib_param.values.astype(float)
        jumps = [abs(p[s][i] / p[s][i - 1] - 1) for s in segs for i in range(1, len(s)) if p[s][i - 1] != 0]
        if jumps and max(jumps) > V3['calib_jump_max']:
            out['c4'] = False
    return {k: bool(v) for k, v in out.items()}


def pick(G, order, steps, sel, **kw):
    """Первый по среднему q на steps вариант с критериями sel (перебор по возрастанию среднего). -> (вариант | None, флаги, проверено)."""
    first = None
    for k, v in enumerate(order):
        fl = crit(G[v], steps, **kw)
        if first is None:
            first = (v, fl)
        if all(fl[x] for x in sel):
            return v, fl, k + 1
    return None, first[1], len(order)


# ---------------------------------------------------------------- LOYO (пп. 2.1–2.3)
def loyo(tab, F_unit, mode=None, blok=()):
    """tab — таблица шагов уровня одного h. blok — единицы, заблокированные проверкой сдвига (п. 2.6). -> (W по единицам, L по складкам)."""
    mode = mode or CFG.get('rezhim', 'v3')
    M = MODES[mode]
    kw = dict(c1_E=True, c3_E=M['c3_E'])
    tab = tab[tab.step.isin(T)]
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
    assert len(var[F_unit]) == 1, 'F_h(d0) — контрольная единица с одним вариантом'
    first = lambda c, d: (lambda s: s.iloc[0]) if c in tab else (lambda s: d)
    meta = {u: dict(role=tab.loc[tab.unit == u, 'role'].iloc[0] if 'role' in tab else '',
                    slozhnost=int(tab.loc[tab.unit == u, 'slozhnost'].iloc[0]) if 'slozhnost' in tab else 0) for u in units}
    ok = [u for u in units if u in var]
    per = {}
    for Yst in T:
        tr = tuple(s for s in T if s != Yst)
        for u in ok:
            order = sorted(var[u], key=lambda v: (np.mean([Q[(u, v)][s] for s in tr]), var[u].index(v)))
            v, fl, nchk = pick(G[u], order, tr, M['sel'], **kw)
            vv = v if v is not None else order[0]
            gq = G[u][vv].set_index('step')
            qE = float(gq.q_har_c1[Yst]) if 'q_har_c1' in gq else np.nan
            per[(u, Yst)] = dict(Y=Y(Yst), unit=u, variant=vv, dop=v is not None, q=Q[(u, vv)][Yst], q_etalon=qE,
                                 better_E=bool(Q[(u, vv)][Yst] < qE), q_train=float(np.mean([Q[(u, vv)][s] for s in tr])),
                                 n_checked=nchk, n_var=len(order), failed=','.join(k for k in CR if not fl[k]), **fl)
    dop_all = {u: all(per[(u, s)]['dop'] for s in T) for u in ok}
    nE = {u: sum(per[(u, s)]['better_E'] for s in T) for u in ok}
    selT, flT, vhod, why = {}, {}, {}, {}
    for u in ok:
        order = sorted(var[u], key=lambda v: (qbar_T(Q[(u, v)]), var[u].index(v)))
        sv, fl, _ = pick(G[u], order, T, M['sel'], **kw)
        selT[u] = sv
        flT[u] = crit(G[u][sv], T, **kw) if sv is not None else fl
        if M['entry'] == 'v3':
            a, b = nE[u] >= V3['vhod_a_min'], dop_all[u]
            c = sv is not None and all(flT[u][k] for k in V3['vhod_c'])
            vhod[u] = a and b and c and u not in blok
            why[u] = ','.join(x for x, f in (('a', a), ('b', b), ('c:' + '+'.join(k for k in V3['vhod_c'] if not flT[u][k]), c),
                                             ('blok_sdvig', u not in blok)) if not f)
        else:
            vhod[u] = dop_all[u] and u not in blok
            why[u] = '' if vhod[u] else ('blok_sdvig' if u in blok else 'c1–c4 на складках')
    pool = [u for u in ok if vhod[u] or u == F_unit] if M['pool'] == 'proshedshie' else ok
    long = []
    for Yst in T:
        P = [per[(u, Yst)] for u in ok]
        adm = [p['unit'] for p in P if p['unit'] in pool and p['dop']]
        rk = pd.Series({u: per[(u, Yst)]['q'] for u in adm}, dtype=float).rank(method='average')
        tail = (len(adm) + 1 + len(pool)) / 2  # недопущенная в складке единица пула (F_h без c3) — средний ранг хвоста
        qF = per[(F_unit, Yst)]['q']
        for p in P:
            u = p['unit']
            p['rank'] = (float(rk[u]) if p['dop'] else tail) if u in pool else np.nan
            p['r'] = p['q'] / qF - 1
            p['rE'] = p['q'] / p['q_etalon'] - 1
            p['better_F'] = bool(p['q'] < qF) if u != F_unit else False
            long.append(p)
    L = pd.DataFrame(long)
    rows = []
    for u in ok:
        g = L[L.unit == u].set_index('Y')
        r = dict(unit=u, role=meta[u]['role'], slozhnost=meta[u]['slozhnost'], n_variants=len(var[u]), is_F=u == F_unit)
        for y in g.index:
            r.update({f'var_{y}': g.loc[y, 'variant'], f'q_{y}': g.loc[y, 'q'], f'rank_{y}': g.loc[y, 'rank'], f'dop_{y}': g.loc[y, 'dop'],
                      f'fail_{y}': g.loc[y, 'failed'], f'rE_{y}': g.loc[y, 'rE'], f'rF_{y}': g.loc[y, 'r']})
        r.update(R_bar=g['rank'].mean() if u in pool else np.nan, mean_q=g.q.mean(), mean_rE=g.rE.mean(), mean_rF=g.r.mean(),
                 sigma_rF=g.r.std(ddof=1), sigma_rE=g.rE.std(ddof=1), worst_year=int(g.rE.idxmax()), worst_rE=g.rE.max(),
                 n_better_E=nE[u], c3_vse=dop_all[u], **{f'{k}_T': flT[u][k] for k in CR}, vhod=vhod[u], vhod_net=why[u],
                 blok_sdvig=u in blok, n_better_F=int(g.better_F.sum()),
                 sel_T=selT[u] if selT[u] is not None else '', sel_T_q=qbar_T(Q[(u, selT[u])]) if selT[u] is not None else np.nan,
                 rezhim=mode)
        rows.append(r)
    W = pd.DataFrame(rows)
    W['_rb'] = W.R_bar.fillna(np.inf)
    W = W.sort_values(['vhod', '_rb', 'mean_q', 'slozhnost'], ascending=[False, True, True, True], kind='stable').drop(columns='_rb')
    W = W.reset_index(drop=True)
    W.insert(0, 'poryadok', range(1, len(W) + 1))
    return W, L


# ---------------------------------------------------------------- сложность и штраф (п. 2.4)
def _merge(a, b):
    """Глубокое слияние схемы: b дополняет a (группы — по ключам; листья b — новые)."""
    out = dict(a)
    for k, v in b.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) and 'values' not in v else v
    return out


def load_schema(path):
    """Схема с `extends` (v2 = v1 + кластеры 16, 18, 20, boost.grid_point; S, PR #184): листья базы без изменений, новые — сверху."""
    raw = yaml.safe_load(Path(path).read_text())
    if raw.get('extends'):
        base = load_schema(Path(path).parent / raw['extends'])
        raw = _merge(base, {k: v for k, v in raw.items() if k != 'extends'})
    return raw


SCHEMA = load_schema(REPO / CFG['schema'])
NE_SCHITAT = ('design', 'h', 'seed', 'schema', 'ens.seeds', 'selection', 'guards')  # не ключи дизайна или фиксированы протоколом


def _specs(node, path=''):
    """Листья схемы: {путь: spec} (spec — словарь с values/default или type)."""
    out = {}
    for k, v in node.items():
        p = f'{path}.{k}' if path else k
        if isinstance(v, dict) and ('values' in v or isinstance(v.get('type'), str) or k == 'enet'):  # leg.type — ключ, а не «type: str»
            out[p] = v
        elif isinstance(v, dict):
            out.update(_specs(v, p))
    return out


SPECS = _specs({k: v for k, v in SCHEMA.items() if k != 'schema'})


def _flat(d, path=''):
    out = {}
    for k, v in d.items():
        p = f'{path}.{k}' if path else k
        if isinstance(v, dict) and p not in SPECS:
            out.update(_flat(v, p))
        else:
            out[p] = v
    return out


def _n(x):
    """Нормализация значения для сравнения со схемой: yaml.safe_load (YAML 1.1) читает off/on/yes/no схемы как False/True, а в дизайне
    они могут быть строками; числа — десятичной строкой («0.5» = 0.5)."""
    if x is True or (isinstance(x, str) and x.lower() in ('on', 'yes', 'true')):
        return 'on'
    if x is False or (isinstance(x, str) and x.lower() in ('off', 'no', 'false')):
        return 'off'
    if isinstance(x, (int, float)) and not isinstance(x, bool):
        return repr(float(x))
    try:
        return repr(float(x)) if isinstance(x, str) and x.strip() not in ('inf', '-inf', 'nan') else str(x)
    except ValueError:
        return str(x)


def validate(design):
    """Значения ключей дизайна — из списков схемы (иначе ValueError, до счёта). Списки значений — «гиперпараметры внутри складки»."""
    for p, v in _flat(design).items():
        if p in ('design', 'schema'):
            continue
        if p not in SPECS:
            raise ValueError(f'ключ {p} не из схемы v1')
        s = SPECS[p]
        if p == 'features':
            bad = [x for x in v if x not in s['values']]
        elif p == 'leg.enet':
            bad = [] if not isinstance(v, dict) else [x for kk, xs in v.items() for x in (xs if isinstance(xs, list) else [xs])
                                                       if x not in s[kk]['values']]
        elif 'values' in s:
            vals = v if isinstance(v, list) and not any(isinstance(x, list) for x in s['values']) else [v]
            bad = [x for x in vals if _n(x) not in map(_n, s['values'])]
        else:
            bad = []
        if bad:
            raise ValueError(f'{p}: значения {bad} не из схемы v1')
    return True


def otlichnye_klyuchi(design):
    """Ключи дизайна, отличные от default (то, что считает сложность): fixed, ens.seeds и сетка leg.enet по умолчанию — не в счёт.
    Сетки групп схемы v2 (k16/k18/k20: группа с метаполем default, листья без default — tau, k, q, lam, delta): список значений — выбор
    внутри складки, не в счёт; одиночное значение — один раз на группу (как leg.enet и r2_core.complexity). Ключи v2 по умолчанию off."""
    out, grids = [], []
    for p, v in _flat(design).items():
        if any(p == x or p.startswith(x + '.') for x in NE_SCHITAT) or p not in SPECS:
            continue
        s = SPECS[p]
        if s.get('fixed'):
            continue
        if p == 'features':
            if set(v) - {'G0'}:
                out.append(p)
        elif p == 'leg.enet':
            if isinstance(v, dict) and not all(isinstance(x, list) for x in v.values()):
                out.append(p)
        elif 'default' not in s and 'values' in s:  # лист сетки группы v2
            g = p.rsplit('.', 1)[0]
            if not isinstance(v, list) and g not in grids:
                grids.append(g)
        elif 'default' in s and _n(v) != _n(s['default']):
            out.append(p)
    return out + [g + '.*' for g in grids]


def slozhnost(design):
    """c(u), п. 2.4: число ключей со значением не равным default (schema_v2; ключи v2 по умолчанию off); группы признаков сверх G0 — по одной; fixed — не в счёт;
    список значений у ключа с default-сеткой (leg.enet) — не в счёт (выбор внутри складки)."""
    return sum(len(set(_flat(design)['features']) - {'G0'}) if p == 'features' else 1 for p in otlichnye_klyuchi(design))


def vybor_shtraf(W, roles=None, k=1, s=None):
    """п. 2.4, повторно k раз: u* — мин. R̄ (затем мин. средний q); неотличимые — R̄ − R̄* ≤ rank_delta или mean_q ≤ (1 + qlike_rel)·q*;
    из них — мин. сложность, затем мин. R̄, затем мин. средний q. Кандидаты — прошедшие вход, без F_h. -> W выбранных в порядке выбора."""
    s = s or V3['shtraf']
    c = W[W.vhod & ~W.is_F]
    if roles is not None:
        c = c[c.role.isin(roles)]
    out = []
    while len(c) and len(out) < k:
        us = c.sort_values(['R_bar', 'mean_q'], kind='stable').iloc[0]
        ne = c[(c.R_bar - us.R_bar <= s['rank_delta']) | (c.mean_q <= (1 + s['qlike_rel']) * us.mean_q)]
        w = ne.sort_values(['slozhnost', 'R_bar', 'mean_q'], kind='stable').iloc[0]
        out.append(w.name)
        c = c.drop(index=w.name)
    return W.loc[out].assign(shtraf_poryadok=range(1, len(out) + 1))


# ---------------------------------------------------------------- 2σ, финалисты, победитель (п. 2.5)
def sigma2(cand, ref, seeds=SEEDS):
    """Д24.3: d_s = Q̄_T(канд, s) − Q̄_T(F_h(d0), s); d̄ < −2σ/√n (σ — ddof = 1). Детерминированный ряд — число; оба — σ = 0, d̄ < 0."""
    val = lambda x, s: float(x[s]) if isinstance(x, dict) else float(x)
    d = np.array([val(cand, s) - val(ref, s) for s in seeds])
    n = len(d)
    dm, sg = float(d.mean()), float(d.std(ddof=1)) if n > 1 else 0.0
    thr = -2 * sg / np.sqrt(n)
    t = dm / (sg / np.sqrt(n)) if sg > 0 else (-np.inf if dm < 0 else (np.inf if dm > 0 else 0.0))
    return dict(n=n, d_mean=dm, sigma=sg, thr=thr, t=float(t), passed=bool(dm < thr), n_better=int((d < 0).sum()))


def finalisty(W, k=None, v0_units=()):
    """До k лучших после штрафа (без V0-ячеек) + V0-кандидаты (лучшие варианты), прошедшие вход."""
    k = k or V3['finalisty_k']
    top = vybor_shtraf(W[~W.unit.isin(v0_units)], k=k)
    v0 = W[W.unit.isin(v0_units) & W.vhod]
    return pd.concat([top.assign(tip='top'), v0.assign(tip='V0', shtraf_poryadok=np.nan)])


def pobeditel(fin, sig):
    """Первый финалист в порядке п. 2.4 (затем V0 по R̄), прошедший 2σ. sig — {unit: sigma2(...)} на 10 seed. Никто — F_h(d0)."""
    order = list(fin[fin.tip == 'top'].sort_values('shtraf_poryadok').unit) + list(fin[fin.tip == 'V0'].sort_values('R_bar').unit)
    for u in order:
        if sig.get(u, {}).get('passed'):
            return dict(pobeditel=u, V0=bool((fin.set_index('unit').loc[u, 'tip']) == 'V0'), poryadok=order)
    return dict(pobeditel='F_h(d0)', V0=False, poryadok=order)


def fh_d0_seeds(h):
    """Q̄_T F_h(d0) по seed 42–51 (ряды ВЫР-1). -> {seed: Q̄_T} или None, пока файлов нет."""
    c = CFG['F_h_d0']
    fs = {s: R / c['file'].format(h=h, seed=s) for s in SEEDS}
    if not all(f.exists() for f in fs.values()):
        return None
    return {s: qbar_T(q_steps(series([f]), c['s2'])) for s, f in fs.items()}


# ---------------------------------------------------------------- проверка информационного множества (п. 2.6)
def rv_daily(tickers=None):
    """Дневная RV тикеров (столбец rv features_pit_v3 = RV дня строки) по 2021-12-31 включительно."""
    c = CFG['sdvig']
    d = pd.read_parquet(R / c['features'], columns=['date', 'ticker', c['rv_col']],
                        filters=[('date', '<=', pd.Timestamp(c['data_max']))])
    assert d.date.max() <= pd.Timestamp(c['data_max'])
    if tickers is not None:
        d = d[d.ticker.isin(tickers)]
    return d.rename(columns={c['rv_col']: 'rv'})


def K_dizaina(design=None, features=None, leg=None, h=None):
    """Множество последних известных дней ряда (п. 2.6): по группам признаков дизайна (yaml sdvig.izvestnyy_den_grupp) и по типу ноги
    (sdvig.izvestnyy_den_nog: у RVI_anchor коэффициенты k_i — из окна 252 с эмбарго h, последний известный день цели d−h). Иначе {0}."""
    d = design or {}
    fs = features if features is not None else (d.get('features') or ['G0'])
    gr = CFG['sdvig'].get('izvestnyy_den_grupp', {})
    K = {k for g in fs for k in gr.get(g, [0])} | {0}
    lt = leg if leg is not None else (d.get('leg') or {}).get('type')
    hh = h if h is not None else d.get('h')
    for x in CFG['sdvig'].get('izvestnyy_den_nog', {}).get(lt, []):
        K.add(-int(hh) if x == '-h' else int(x))
    return tuple(sorted(K))


def sdvig_ok(argmax, K=(0,), corr=None, eps=None):
    """Правило п. 2.6 с последним известным днём и симметричным допуском ε (решение дирижёра по ❓ 5 протокола, 26.10):
    диапазон последних известных дней — [min K; max K]; c_in — максимум корреляции на сдвигах внутри диапазона, c_out — на сдвигах вне его.
    Ряд проходит, если c_out − c_in ≤ ε (при argmax вне диапазона — с пометкой «плоский профиль»); иначе блок: сдвиг вне диапазона
    справа — заглядывание, слева — отстаёт. Без corr — по argmax (вне диапазона — блок). -> (ok, причина или пометка)."""
    eps = CFG['sdvig'].get('eps', 0.0) if eps is None else eps
    if argmax is None:
        return False, 'нет пар'
    lo, hi = min(K), max(K)
    if corr is not None:
        c = {s: v for s, v in corr.items() if v == v}
        cin = [v for s, v in c.items() if lo <= s <= hi]
        cout = {s: v for s, v in c.items() if s < lo or s > hi}
        if cin and cout:
            s_out = max(cout, key=cout.get)
            ex = cout[s_out] - max(cin)
            if ex > eps:
                side = 'заглядывание (+1)' if s_out > hi else 'отстаёт (info_set < d0)'
                return False, f'{side}: сдвиг {s_out:+d} выше диапазона на {ex:.4f} > ε = {eps}'
            if argmax < lo or argmax > hi:
                return True, f'плоский профиль: сдвиг {s_out:+d} выше диапазона [{lo}; {hi}] на {ex:.4f} ≤ ε = {eps}'
            return True, ''
    if argmax > 0:
        return False, 'заглядывание (+1)'
    if argmax < min(K):
        return False, 'отстаёт (info_set < d0)'
    return True, ''


def K_str(K):
    return str(K[0]) if len(K) == 1 else f'{min(K)}..{max(K)}'


def sdvig(pred, rv, sdvigi=None, K=(0,)):
    """Спирмен ln ŷ_d с ln RV_{d+s} по тикерам (s — в торговых днях тикера), медиана по тикерам; пары с d+s > 2021-12-31 не берутся.
    K — последние известные дни входов ряда. -> {corr_s…, argmax, ok, blok, prichina, posledniy_izvestnyy_den}."""
    sdvigi = sdvigi or CFG['sdvig']['sdvigi']
    p = pred[(pd.to_datetime(pred.date).dt.year >= 2017) & (pd.to_datetime(pred.date).dt.year <= 2021)]
    res = {s: [] for s in sdvigi}
    for tk, g in p.groupby('ticker'):
        r = rv[rv.ticker == tk].sort_values('date').reset_index(drop=True)
        idx = pd.Series(r.index.values, index=pd.to_datetime(r.date))
        lr = np.log(np.clip(r.rv.values, 1e-12, None))
        pos = idx.reindex(pd.to_datetime(g.date)).values
        m = ~np.isnan(pos)
        pos, ly = pos[m].astype(int), np.log(np.clip(g.rv_pred.values[m], 1e-12, None))
        for s in sdvigi:
            j = pos + s
            ok = (j >= 0) & (j < len(r))
            if ok.sum() > 10:
                res[s].append(pd.Series(ly[ok]).corr(pd.Series(lr[j[ok]]), method='spearman'))
    med = {s: float(np.nanmedian(v)) if v else np.nan for s, v in res.items()}
    am = None if all(np.isnan(list(med.values()))) else int(max(med, key=lambda s: -np.inf if np.isnan(med[s]) else med[s]))
    ok, why = sdvig_ok(am, K, corr=med)
    return dict(n_tikerov=int(p.ticker.nunique()), n_strok=int(len(p)), **{f'corr_s{s:+d}': med[s] for s in sdvigi}, argmax=am,
                ok=ok, blok=not ok, prichina='' if ok else why, pometka=why if ok else '', posledniy_izvestnyy_den=K_str(K))


def sdvig_zapis(r, f=None):
    """Строка проверки сдвига -> blok_sdvig.csv, если ряд заблокирован или прошёл с пометкой «плоский профиль» (ok = True).
    r — словарь sdvig(...) + ryad, h, hash. Для r2_core.shift_check (S): вызывать вместо условия «if not ok»."""
    if (not r.get('ok')) or r.get('pometka'):
        blok_zapis([{k: v for k, v in r.items() if k not in ('blok', 'vremya')}], f)


# формат blok_sdvig.csv — как r2_core.shift_check агента S, плюс поле «последний известный день» (дирижёр 26.10 после PR #163)
BLOK_COLS = ['ryad', 'h', 'hash', 'n_tikerov', 'n_strok', 'corr_s-2', 'corr_s-1', 'corr_s+0', 'corr_s+1', 'argmax', 'ok', 'vremya', 'prichina',
             'posledniy_izvestnyy_den', 'pometka']


def blok_zapis(rows, f=None):
    """Строки заблокированных рядов -> R2/O/blok_sdvig.csv (дописываются под блокировкой файла; если в файле нет нового столбца —
    файл переписывается с объединённой шапкой, старые строки получают пустое поле)."""
    import fcntl
    if not rows:
        return
    f = Path(f or (R / CFG['sdvig']['out']))
    f.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows).assign(vremya=now())
    with open(f, 'a+', encoding='utf-8') as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        fh.seek(0)
        txt = fh.read()
        old = pd.read_csv(f) if txt.strip() else pd.DataFrame(columns=BLOK_COLS)
        cols = list(dict.fromkeys(BLOK_COLS + list(old.columns) + list(df.columns)))
        allr = pd.concat([old, df], ignore_index=True).reindex(columns=cols)
        fh.seek(0)
        fh.truncate()
        allr.to_csv(fh, index=False)
    print(f'БЛОК сдвига: {len(df)} рядов -> {f} (доклад дирижёру)', flush=True)


def peresmotr_blok(zapis=True):
    """Пересмотр заблокированных рядов кэша (строки blok_sdvig.csv с hash и ok = False) по действующему правилу sdvig_ok (K по метаданным
    дизайна, допуск ε): корреляции — из строки, дизайн — R2/cache/<hash>/design.yaml. Новая строка с тем же hash дописывается
    (последняя строка по (ryad, h, hash) — действующая). -> таблица пересмотра."""
    f = R / CFG['sdvig']['out']
    B = pd.read_csv(f)
    B = B[B.hash.notna() & (B.hash.astype(str) != '')].drop_duplicates(['ryad', 'h', 'hash'], keep='last')  # действующая строка
    B = B[~B.ok.astype(str).str.lower().eq('true')]
    rows = []
    for r in B.to_dict('records'):
        dz = R / 'R2' / 'cache' / str(r['hash']) / 'design.yaml'
        if not dz.exists():
            continue
        d = yaml.safe_load(dz.read_text())
        K = K_dizaina(d, h=int(r['h']))
        corr = {s: float(r[f'corr_s{s:+d}']) for s in CFG['sdvig']['sdvigi']}
        ok, why = sdvig_ok(int(r['argmax']), K, corr=corr)
        rows.append(dict(ryad=r['ryad'], h=int(r['h']), hash=r['hash'], n_tikerov=r.get('n_tikerov'), n_strok=r.get('n_strok'),
                         **{f'corr_s{s:+d}': corr[s] for s in corr}, argmax=int(r['argmax']), ok=ok, prichina='' if ok else why,
                         posledniy_izvestnyy_den=K_str(K), pometka=f'пересмотр (ε = {CFG["sdvig"]["eps"]}, K по ноге {d["leg"].get("type")})'
                         + (f'; {why}' if ok and why else '')))
    if zapis and rows:
        blok_zapis(rows, f)
    return pd.DataFrame(rows)


def reshit_F(f_dir=None, zapis=True):
    """Решение O по проверке сдвига агента F (PR #163) с последним известным днём. Случаи — группы (shift_check_groups.csv, кроме календаря)
    и столбцы со статусом F «ФЛАГ» или с лагом публикации (yaml izvestnyy_den_stolbcy); профили и argmax — как у F (по |ρ|).
    Столбец: уровень argmax = +1 — блок; уровень в [min K; 0] — ок; иначе (оконный уровень) решает приращение: +1 — блок, в [min K; 0] — ок,
    левее — блок (отстаёт). -> R2/O/sdvig_F_resheniya.csv; блок — в blok_sdvig.csv (ryad = F:<столбец|группа>)."""
    c = CFG['sdvig']
    f_dir = Path(f_dir or REPO / 'docs/otchety/r2_F')
    S = pd.read_csv(f_dir / 'shift_check.csv')
    G = pd.read_csv(f_dir / 'shift_check_groups.csv')
    kc = c.get('izvestnyy_den_stolbcy', {})
    rows, blok = [], []
    for r in S[(S.status == 'ФЛАГ') | S.col.isin(list(kc))].itertuples():
        K = (int(kc.get(r.col, 0)),)
        am, dam = int(r.argmax), int(r.d_argmax)
        if am > 0:
            v, why = 'блок', 'заглядывание (+1) по уровню'
        elif sdvig_ok(am, K)[0]:
            v, why = 'ок', 'по уровню'
        elif dam > 0:
            v, why = 'блок', 'заглядывание (+1) по приращению (уровень — окно)'
        elif sdvig_ok(dam, K)[0]:
            v, why = 'ок', 'по приращению (уровень — окно)'
        else:
            v, why = 'блок', 'отстаёт (info_set < d0)'
        rows.append(dict(obekt=f'столбец {r.col}', gruppa=r.group, posledniy_izvestnyy_den=K_str(K), argmax=am, d_argmax=dam,
                         status_F=r.status, reshenie=v, prichina=why))
        if v == 'блок':
            blok.append(dict(ryad=f'F:{r.col}', h=1, hash='', **{f'corr_s{s:+d}': S.loc[r.Index, f'rho_{s}'] for s in (-2, -1, 0, 1)},
                             argmax=am, ok=False, prichina=f'{why}; d_argmax {dam}; детектор F чистый', posledniy_izvestnyy_den=K_str(K)))
    for r in G[~G.deterministic].itertuples():
        K = tuple(c.get('izvestnyy_den_grupp', {}).get(r.group, [0])) if r.group != 'G1_bez_PC' else (0,)
        ok, why = sdvig_ok(int(r.argmax), K)
        v = 'ок' if ok else 'блок'
        rows.append(dict(obekt=f'группа {r.group}', gruppa=r.group, posledniy_izvestnyy_den=K_str(K), argmax=r.argmax, d_argmax=np.nan,
                         status_F=r.verdict, reshenie=v, prichina=why))
        if v == 'блок':
            blok.append(dict(ryad=f'F:{r.group}', h=1, hash='', **{f'corr_s{s:+d}': G.loc[r.Index, f'rho_{s}'] for s in (-2, -1, 0, 1)},
                             argmax=r.argmax, ok=False, prichina=why, posledniy_izvestnyy_den=K_str(K)))
    out = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT / 'sdvig_F_resheniya.csv', index=False)
    if zapis:
        blok_zapis(blok)
    return out


# ---------------------------------------------------------------- число дизайнов и ложные победители (п. 2.7)
def lozhnye(N, n_fin):
    """N — {уровень: (перебрано единиц, прошли вход)}; n_fin — число финалистов. -> словарь для gate/FINAL."""
    c = CFG['lozhnye']
    p0 = float(stats.t.cdf(c['t_porog'], c['t_df']))
    return dict(p0=p0, n_fin=int(n_fin), E_FW_2sigma=n_fin * p0,
                skrining_lozhno_vklyuchennyh=c['n_faktorov'] * c['skrining_alpha'],
                urovni={k: dict(perebrano=int(a), proshli_vhod=int(b), dolya_vhoda=(b / a if a else np.nan)) for k, (a, b) in N.items()},
                vhod_a_pri_nule=float(stats.binom.sf(V3['vhod_a_min'] - 1, 5, 0.5)),
                ogovorka='перебор N единиц с выбором лучшего делает LOYO-оценку победителя оптимистичной; честная оценка — только тест')


# ---------------------------------------------------------------- гейты (п. 2.9)
def write_gate(phase, h, payload, inputs=()):
    """R2/O/gate_<phase>_h<h>.yaml: решение + хэши входов, кода и yaml. Следующая фаза стартует по этому файлу."""
    OUT.mkdir(parents=True, exist_ok=True)
    g = dict(phase=phase, h=int(h), time=now(), code_sha256=sha(__file__), yaml_sha256=sha(CFG_PATH),
             gibrid_designs_sha256=sha(HERE / 'gibrid_designs.py'), inputs={str(p): sha(p) for p in inputs if Path(p).exists()}, **payload)
    f = OUT / f'gate_{phase}_h{h}.yaml'
    f.write_text(yaml.safe_dump(json.loads(json.dumps(g, default=_js)), allow_unicode=True, sort_keys=False))
    return f


def _js(x):
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return None if np.isnan(x) else float(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return str(x)


def gate_C(W, h, inputs=(), v0_units=()):
    """Протокол v3 (R̄), справочно; по v4 ворота C — r2_gates.gate_C_v4.
    Фаза C: две ячейки с ролью ≠ none (после штрафа); V0-кандидаты, прошедшие вход, — прямо в финалисты."""
    two = vybor_shtraf(W[~W.unit.isin(v0_units)], roles=CFG['level']['boost_roles'], k=2)
    v0 = W[W.unit.isin(v0_units) & W.vhod]
    N = {'C': (int((~W.is_F).sum()), int((W.vhod & ~W.is_F).sum()))}
    return write_gate('C', h, dict(yacheyki=two[['unit', 'role', 'R_bar', 'mean_q', 'slozhnost', 'sel_T']].to_dict('records'),
                                   v0_v_finalisty=v0[['unit', 'R_bar', 'mean_q', 'sel_T']].to_dict('records'),
                                   chislo=lozhnye(N, 0)['urovni'], net_yacheek=len(two) == 0), inputs)


def dizain_hashi(path):
    """Хэши дизайна для сборки (агент A, PR #165): design_sha — канонический хэш по правилу r2_core агента S (design_hash: полный yaml
    с default, канонизация шапки schema_v1), cache_dir — его первые 16 знаков, canonical_yaml_sha256 — SHA-256 полного канонического yaml
    (= design_sha), yaml_file_sha256 — SHA-256 файла дизайна как он лежит."""
    import r2_core as C
    d = yaml.safe_load(Path(path).read_text())
    can = C.canonical(d)
    ds = C.design_hash(d)
    assert ds == hashlib.sha256(can.encode('utf-8')).hexdigest()
    return dict(design_yaml=str(path), design_sha=ds, cache_dir=ds[:16], canonical_yaml_sha256=ds, yaml_file_sha256=sha(path),
                canonical_yaml=can)


LESTNICA = [  # ступень -> префиксы ключей схемы, которые на ней берутся из дизайна победителя (накопительно); остальное — default
    ('HAR-J', None), ('+s²', None),
    ('+HAR-X', ('leg',)),
    ('+калибровка', ('calib',)),
    ('+бустинг', ('boost', 'features', 'data')),
    ('+согласование', ('mh', 'ens')),
]


def _vzyat(src, prefixes, base):
    """Скопировать в base ключи src с путями, начинающимися с prefixes (вложенные словари)."""
    import copy
    out = copy.deepcopy(base)
    for k, v in src.items():
        if k in prefixes or any(p.split('.')[0] == k for p in prefixes):
            out[k] = copy.deepcopy(v)
    return out


def _zapisat_dizain(d, path):
    import r2_core as C
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(yaml.safe_dump(d, allow_unicode=True, sort_keys=False))
    ds = C.design_hash(d)
    return dict(design_yaml=str(path), design_sha=ds, cache_dir=str(R / 'R2' / 'cache' / ds[:16]),
                ryady_shablon=f'{ds[:16]}/*_h{d["h"]}_s*_val.parquet')


def lestnica_ablyacii(win, h, out_dir):
    """Лестница (кластер 10): HAR-J (эталон (а)) → +s² (эталон (б) = E_h) → +HAR-X → +калибровка → +бустинг → +согласование → финал;
    ступени 3–6 — дизайны: ключи победителя по группам LESTNICA накопительно, остальное default; финал — победитель.
    Абляции: финал минус каждый кластер — ключи этого кластера (schema_v2, поле cluster; сетка группы v2 — к «вся сетка») возвращаются к default.
    Дизайны пишутся в out_dir (<имя>.yaml), у каждого design_sha (r2_core), cache_dir, шаблон рядов."""
    import copy
    e = CFG['etalon']
    L = [dict(stupen='HAR-J', design_sha=None, ryad=dict(file=str(R / e['file'].format(h=h)), col='har_raw', s2='bez'),
              primechanie='эталон (а), замороженный ряд ВЫР-1'),
         dict(stupen='+s²', design_sha=None, ryad=dict(file=str(R / e['file'].format(h=h)), col='har_raw', s2='sm'),
              primechanie='эталон (б) = E_h (= col har_corr), замороженный ряд ВЫР-1')]
    base = {'design': '', 'h': h}
    if 'seeds' in win.get('ens', {}):  # набор seed — условие оценки, не ступень: на всех ступенях как у победителя
        base['ens'] = {'seeds': win['ens']['seeds']}
    pref, prev = (), None
    for name, p in LESTNICA[2:]:
        pref = pref + p
        d = _vzyat(win, pref, base)
        d['design'] = f'lestnica_h{h}_{name.strip("+")}'
        validate(d)
        r = _zapisat_dizain(d, Path(out_dir) / f'lestnica_{len(L) + 1}_{name.strip("+")}.yaml')
        r.update(stupen=name, bez_izmeneniy=bool(prev is not None and prev == r['design_sha']))
        prev = r['design_sha']
        L.append(r)
    fin = _zapisat_dizain(copy.deepcopy(win), Path(out_dir) / f'lestnica_{len(L) + 1}_final.yaml')
    fin.update(stupen='финал', bez_izmeneniy=fin['design_sha'] == prev,
               primechanie='' if fin['design_sha'] == prev else 'у победителя есть ключи вне групп лестницы — финал отличается от +согласование')
    L.append(fin)
    kl = {}
    for p in otlichnye_klyuchi(win):  # те же ключи, что считает сложность (п. 2.4): ens.seeds, fixed и сетка enet — не абляции
        leaf = p if not p.endswith('.*') else next(q for q in SPECS if q.startswith(p[:-1]) and 'cluster' in SPECS[q])
        kl.setdefault(int(SPECS[leaf]['cluster']), []).append(p)
    A = []
    for c in sorted(kl):
        d = copy.deepcopy(win)
        for p in kl[c]:
            a, *b = p.split('.')
            if not b:
                d.pop(a, None)
            elif b[0] == '*':  # сетка группы v2 — вернуть к «вся сетка»: убрать её листья без default
                for q in [q for q in SPECS if q.startswith(a + '.') and 'default' not in SPECS[q]]:
                    d.get(a, {}).pop(q.split('.', 1)[1], None)
                if a in d and not d[a]:
                    d.pop(a)
            else:
                d.get(a, {}).pop(b[0], None)
                if a in d and not d[a]:
                    d.pop(a)
        d['design'] = f'ablyaciya_h{h}_minus_k{c}'
        validate(d)
        r = _zapisat_dizain(d, Path(out_dir) / f'ablyaciya_minus_k{c}.yaml')
        r.update(klaster=c, klyuchi=kl[c])
        A.append(r)
    return L, A


def gate_E(W, h, sig, inputs=(), v0_units=(), N=None, dizainy=None, ryady=None):
    """Протокол v3 (R̄), справочно; по v4 ворота E — r2_gates.gate_E_v4 (порядок выбора кластера 17, мощность, itog_E).
    Фаза E: финалисты (до 3 + V0), 2σ к F_h(d0) по 10 seed (sig: {unit: sigma2}), победитель; число дизайнов и ложные победители.
    dizainy — {unit: путь к yaml дизайна}: у финалистов и победителя — design_sha (r2_core) и хэш полного yaml (сверка у агента A)."""
    dizainy = dizainy or {}
    fin = finalisty(W, v0_units=v0_units)
    pb = pobeditel(fin, sig)
    hs = {u: dizain_hashi(p) for u, p in dizainy.items() if u in set(fin.unit) and p and Path(p).exists()}
    rows = [dict(unit=r.unit, tip=r.tip, R_bar=r.R_bar, mean_q=r.mean_q, slozhnost=r.slozhnost, vhod=bool(r.vhod),
                 **{k: v for k, v in hs.get(r.unit, {}).items() if k != 'canonical_yaml'},
                 **{f's2_{k}': v for k, v in (sig.get(r.unit) or {}).items()}) for r in fin.itertuples()]
    w = pb['pobeditel']
    L, A = None, None
    if w in hs:
        pb.update(hs[w])
        pb['ryady'] = dict(cache_dir=str(R / 'R2' / 'cache' / hs[w]['cache_dir']), shablon=f'*_h{h}_s*_val.parquet',
                           files=list((ryady or {}).get(w, [])))
        L, A = lestnica_ablyacii(yaml.safe_load(Path(hs[w]['design_yaml']).read_text()), h, OUT / f'gate_E_h{h}_dizainy')
    elif w != 'F_h(d0)':
        raise ValueError(f'gate E: нет yaml дизайна победителя {w} — design_sha для сборки (агент A) не записан')
    else:
        fd = CFG['F_h_d0']['file']
        pb.update(design_sha=None, primechanie='победителя нет: строка гибрида — F_h(d0) (замороженный ряд ВЫР-1)',
                  ryady=dict(files=[str(R / fd.format(h=h, seed=s)) for s in SEEDS]))
    return write_gate('E', h, dict(finalisty=rows, pobeditel=pb, lestnica=L, ablyacii=A, primechanie_Fd0=CFG['F_h_d0'].get('otstuplenie', ''),
                                   lozhnye=lozhnye(N or {'E': (int((~W.is_F).sum()), int((W.vhod & ~W.is_F).sum()))}, len(fin))),
                      list(inputs) + [hs[u]['design_yaml'] for u in hs])


# ---------------------------------------------------------------- адаптер: ряды -> таблица шагов
def harcj_guard(files, what=''):
    """HAR-CJ (V0_HCJ gibrid_v17) строится по d−1 — в отборе R2 не берётся; допустим только HAR-CJ d0 (vyr/ref_harcj_d0, PR #154)."""
    d0 = str(R / Path(CFG['harcj_d0']).parent)
    for f in str(files).split(';'):
        p = str(f if Path(f).is_absolute() else R / f)
        if re.search(r'hcj|harcj|har_cj', Path(p).name, re.I):
            assert p.startswith(d0 + '/'), f'{what}: HAR-CJ по d−1 ({p}) — в R2 только d0 из {d0}'


def ref_from_files(files_by_key, col=None):
    """{'xgb_v0': путь, ...} -> {ключ: {код: {шаг: q}}} по всем s²."""
    out = {}
    for k, f in files_by_key.items():
        harcj_guard(f, f'эталон {k}')
        s = series([f if Path(f).is_absolute() else R / f], col=(col or {}).get(k, 'rv_pred'))
        out[k] = {c: q_steps(s, c) for c in CODES}
    return out


def etalon_ref(h):
    """E_h (эталон (б), HAR-J+s² sm d0, ВЫР-1) и HAR-J d0 той же s² для c2: {'har': {код: {шаг: q}}}; c1/c3 — 'har'['sm']."""
    e = CFG['etalon']
    return ref_from_files({'har': e['file'].format(h=h)}, col={'har': e['col']})['har']


def series_tab(spec, ref, h, files_col='files', blok_rows=None, rv=None):
    """spec — DataFrame: unit, role, leg, variant, <files_col> (';' — агрегат seed × lib), s2 [, agg, weights, calib_param, design_yaml].
    ref — {'har': …, 'xgb_v0': …, 'lgb_v0': …, 'leg_v0:<нога>': …}. Эталон c2 у bez — ord (как gibrid_v17).
    rv — дневная RV для проверки сдвига (п. 2.6): заблокированные ряды -> blok_rows, их единицы — в выходном списке blok."""
    cache, rows, blok = {}, [], set()
    et = lambda code: 'ord' if code == 'bez' and CFG.get('etalon_bez', 'ord') == 'ord' else code
    for r in spec.itertuples():
        fs = getattr(r, files_col)
        harcj_guard(fs, f'{r.unit} / {r.variant}')
        agg = getattr(r, 'agg', 'mean') if isinstance(getattr(r, 'agg', None), str) else 'mean'
        dz = getattr(r, 'design_yaml', None)
        dzd = yaml.safe_load(Path(dz).read_text()) if isinstance(dz, str) and Path(dz).exists() else None
        K = K_dizaina(dzd, leg=getattr(r, 'leg', None) if dzd is None else None, h=h)  # последние известные дни входов
        key = (fs, agg, K)
        if key not in cache:
            files = [x if Path(x).is_absolute() else R / x for x in str(fs).split(';')]
            s = series(files, agg=agg)
            sd = sdvig(s, rv, K=K) if rv is not None else None
            cache[key] = (s, sd)
            if sd and (sd['blok'] or sd.get('pometka')) and blok_rows is not None:  # блок или «плоский профиль» (ok)
                blok_rows.append(dict(ryad=f'{r.unit} / {r.variant}: {fs}', h=h, hash=sha(dz)[:16] if dzd is not None else '',
                                      **{k: v for k, v in sd.items() if k != 'blok'}))
        s, sd = cache[key]
        if sd and sd['blok']:
            blok.add(r.unit)
        q = q_steps(s, r.s2)
        e = et(r.s2)
        Wt = pd.read_csv(r.weights).set_index('step') if isinstance(getattr(r, 'weights', None), str) and r.weights else None
        Cp = pd.read_csv(r.calib_param).set_index('step') if isinstance(getattr(r, 'calib_param', None), str) and r.calib_param else None
        cx = slozhnost(dzd) if dzd is not None else int(getattr(r, 'slozhnost', 0) or 0)
        scheme = ('V0_HAR' if r.leg in ('HAR-J', 'HARJ') else 'V0_LEG') if r.role in ('V0', 'none') else 'cell'
        lv = ref.get(f'leg_v0:{r.leg}')
        for st in T:
            rows.append(dict(unit=r.unit, variant=r.variant, step=st, role=r.role, slozhnost=cx, q_hyb=q[st],
                             q_har=ref['har'][e][st], q_har_c1=ref['har']['sm'][st], q_xgb_single=ref['xgb_v0'][e][st],
                             q_lgb_single=ref['lgb_v0'][e][st], q_leg_v0=lv[e][st] if (lv is not None and r.role not in ('V0', 'none')) else np.nan,
                             scheme=scheme, v1_ml='', w_har=Wt.loc[st, 'w_har'] if Wt is not None else 0.0,
                             w_xgb=Wt.loc[st, 'w_xgb'] if Wt is not None else 0.0, w_lgb=Wt.loc[st, 'w_lgb'] if Wt is not None else 0.0,
                             calib_param=Cp.loc[st, 'param'] if Cp is not None else np.nan))
    return pd.DataFrame(rows), blok


def F_d0_tab(h, ref):
    """F_h(d0), seed 42 — контрольная единица уровня."""
    fd = R / CFG['F_h_d0']['file'].format(h=h, seed=42)
    t, _ = series_tab(pd.DataFrame([dict(unit=f'F_{h}(d0)', role='F_h', leg='F', variant=fd.name, files=str(fd), s2=CFG['F_h_d0']['s2'])]), ref, h)
    return t, f'F_{h}(d0)'


def level_inputs(L, h):
    lc = CFG['level']
    spec = pd.read_csv(R / lc['spec'].format(L=L, h=h))
    rf = pd.read_csv(R / lc['ref'].format(h=h))
    ref = ref_from_files(dict(zip(rf.key, rf.file)))
    ref['har'] = etalon_ref(h)
    files = [R / lc['spec'].format(L=L, h=h), R / lc['ref'].format(h=h), R / CFG['etalon']['file'].format(h=h)] + \
            [R / f if not Path(f).is_absolute() else Path(f) for f in rf.file]
    return spec, ref, files


def run_level(L, h, v0_units=()):
    spec, ref, files = level_inputs(L, h)
    v = CFG['level']['versii']
    bad = [f for f in spec.files for x in v['isklyuchit'] if re.search(re.escape(x) + r'(?![_a-z0-9])', str(f)) and v['noga'] not in str(f)]
    assert not bad, f'версии {v["isklyuchit"]} не берутся: {bad[:3]}'
    blok_rows = []
    rv = rv_daily()
    tab, blok = series_tab(spec, ref, h, blok_rows=blok_rows, rv=rv)
    ft, Fn = F_d0_tab(h, ref)
    W, Lf = loyo(pd.concat([tab, ft], ignore_index=True), Fn, blok=blok)
    blok_zapis(blok_rows)
    write_level(W, Lf, L, h)
    if L == 1:
        print('gate C:', gate_C(W, h, files, v0_units), flush=True)
    if L == 3 and 'files10' in spec:  # финалисты на 10 seed: вход заново, 2σ к F_h(d0)
        fin = finalisty(W, v0_units=v0_units)
        s10, _ = series_tab(spec[spec.unit.isin(fin.unit)], ref, h, files_col='files10')
        W10, _ = loyo(pd.concat([s10, ft], ignore_index=True), Fn, blok=blok)
        fd = fh_d0_seeds(h)
        sig = {}
        for u in fin.unit:
            if not W10.set_index('unit').loc[u, 'vhod']:
                continue  # вход на 10 seed не пройден
            per_seed = {}
            for s in SEEDS:
                sp = spec[spec.unit == u].copy()
                sp['files'] = sp.files10.map(lambda x: ';'.join(f for f in x.split(';') if f'_s{s}_' in f) or x)
                ts, _ = series_tab(sp, ref, h)
                Ws, _ = loyo(pd.concat([ts, ft], ignore_index=True), Fn, blok=blok)
                v = Ws.set_index('unit').loc[u, 'sel_T'] or ts.groupby('variant').q_hyb.mean().idxmin()
                per_seed[s] = qbar_T(ts[ts.variant == v].set_index('step').q_hyb.to_dict())
            sig[u] = sigma2(per_seed, fd)
        dz = {u: (R / p if not Path(p).is_absolute() else Path(p)) for u, p in
              spec.groupby('unit').design_yaml.first().dropna().items()} if 'design_yaml' in spec else {}
        ry = {u: [str(R / f if not Path(f).is_absolute() else f) for f in str(x).split(';')]
              for u, x in spec.groupby('unit').files10.first().items()}
        print('gate E:', gate_E(W10, h, sig, files, v0_units, dizainy=dz, ryady=ry), flush=True)
    return W


def gate_D(h, cells):
    """Протокол v3, справочно; по v4 отсечение — r2_halving.otsechenie, ворота D — r2_gates.gate_D_v4.
    Фаза D для каждой ячейки: 60 → 30 по val 2016 (шаг 1, вне T; мин. по вариантам), 30 → 15 → 8 по R̄ на 1 seed (files),
    8 → лучшая по R̄ на 3 seed (files3) со штрафом. spec — R2/O/spec_level2_h{h}.csv: столбец cell + столбцы уровня."""
    spec, ref, files = level_inputs(2, h)
    ft, Fn = F_d0_tab(h, ref)
    rv = rv_daily()
    res, N = [], {}
    for cell in cells:
        sp = spec[spec.cell == cell]
        s1 = {u: min(q_steps(series([R / f for f in str(r.files).split(';')]), 'bez', steps=(1,))[1] for r in g.itertuples())
              for u, g in sp.groupby('unit')}
        keep = [u for u, _ in sorted(s1.items(), key=lambda x: (x[1], x[0]))[:30]]
        trace = {'60->30': len(keep)}
        for n in (15, 8):
            t, blok = series_tab(sp[sp.unit.isin(keep)], ref, h, rv=rv, blok_rows=[])
            W, _ = loyo(pd.concat([t, ft], ignore_index=True), Fn, blok=blok)
            keep = list(W[W.vhod & ~W.is_F].sort_values(['R_bar', 'mean_q']).unit[:n]) or keep[:n]
            trace[f'->{n}'] = len(keep)
        t3, blok = series_tab(sp[sp.unit.isin(keep)], ref, h, files_col='files3', rv=rv, blok_rows=[])
        W3, _ = loyo(pd.concat([t3, ft], ignore_index=True), Fn, blok=blok)
        best = vybor_shtraf(W3, k=1)
        N[cell] = (int(sp.unit.nunique()), int((W3.vhod & ~W3.is_F).sum()))
        res.append(dict(cell=cell, tochka=best.unit.iloc[0] if len(best) else None, otsechenie=trace,
                        R_bar=float(best.R_bar.iloc[0]) if len(best) else None))
    return write_gate('D', h, dict(luchshie=res, chislo=lozhnye(N, 0)['urovni']), files)


# ---------------------------------------------------------------- запись уровня
def write_level(W, L, level, h, out=OUT, extra=None):
    out.mkdir(parents=True, exist_ok=True)
    f = out / f'level{level}_h{h}.csv'
    W.to_csv(f, index=False, float_format='%.6g')
    L.to_csv(out / f'level{level}_h{h}_folds.csv', index=False, float_format='%.6g')
    m = dict(time=now(), level=level, h=h, code_sha=sha(__file__), gibrid_designs_sha=sha(HERE / 'gibrid_designs.py'),
             yaml_sha=sha(CFG_PATH), units=int(len(W)), variants=int(W.n_variants.sum()), vhod=int(W.vhod.sum()), **(extra or {}))
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
                     slozhnost=0, q_har_c1=d.step.map(hs_).values)
        parts.append(d); src.append((b, str(p), sha(p)))
    fc = CFG['F_h'][h]
    p = R / fc['designs_val']
    d = read_designs(p)
    hs_ = harj_sm(d)
    d = d[d.design == fc['design']]
    assert len(d) == len(T), (p, fc['design'])
    parts.append(d.assign(unit=fc['name'], variant=d.design, role='F_h', slozhnost=0, q_har_c1=d.step.map(hs_).values))
    src.append((fc['name'], str(p), sha(p)))
    return pd.concat(parts, ignore_index=True), src


def proverka(hs):
    pr = CFG['proverka_r1']
    out = R / pr['out']
    summ = []
    for h in hs:
        tab, src = tab_r1(h)
        full = [k + '_h' for k in CR]
        stored = {}
        for b, path, _ in src:
            s = pd.read_csv(path, usecols=['design', 'step'] + full)
            stored[b] = s[s.step.isin(T)].groupby('design')[full].first()
        mism = sum(any(bool(stored[u].loc[v][k + '_h']) != fl[k] for k in CR)
                   for (u, v), g in tab.groupby(['unit', 'variant'], sort=False)
                   for fl in [crit(g, T, c1_E=False, c3_E=False, leg=False, runs=False)])
        W, L = loyo(tab, CFG['F_h'][h]['name'])
        write_level(W, L, 1, h, out, extra=dict(proverka_r1=True, sources=src))
        summ.append(dict(h=h, units=len(W), variants=int(W.n_variants.sum()), flags_mismatch=mism, vhod=int(W.vhod.sum())))
        cols = ['poryadok', 'unit', 'role', 'n_variants', 'R_bar', 'mean_q', 'mean_rE', 'mean_rF', 'n_better_E', 'c3_vse', 'c1_T', 'c2_T',
                'c4_T', 'vhod', 'vhod_net']
        print(f'\n=== h={h}: {len(W)} единиц, {int(W.n_variants.sum())} вариантов; сверка флагов с gibrid_v17: расхождений {mism} ===', flush=True)
        print(W[cols].to_string(index=False, float_format=lambda x: f'{x:.4f}'), flush=True)
    S = pd.DataFrame(summ)
    S.to_csv(out / 'svodka.csv', index=False)
    print('\n', S.to_string(index=False), flush=True)
    return S


# ---------------------------------------------------------------- самопроверка на синтетике
def selftest():
    global OUT
    import tempfile
    rng = np.random.default_rng(7)

    def unit(u, nv, base, bad=None, wj=None, role='residual', E=0.30, cx=1):
        rows = []
        for v in range(nv):
            q = base + rng.normal(0, 0.003, len(T)) + 0.002 * v
            for i, st in enumerate(T):
                w = (0.0, 0.0, 0.0) if wj is None else wj[i]
                qq = q[i] * (1.5 if (bad is not None and v in bad[0] and st in bad[1]) else 1.0)
                rows.append(dict(unit=u, variant=f'{u}.v{v}', step=st, role=role, slozhnost=cx, q_hyb=qq, q_har=0.30, q_har_c1=E,
                                 q_xgb_single=0.31, q_lgb_single=0.31, scheme='cell', v1_ml='', w_har=w[0], w_xgb=w[1], w_lgb=w[2]))
        return rows

    def fixed(u, qs, cx=1, **kv):
        return [dict(unit=u, variant=f'{u}.v0', step=st, role='residual', slozhnost=cx, q_hyb=q, q_har=0.30, q_har_c1=0.30,
                     q_xgb_single=0.31, q_lgb_single=0.31, scheme='cell', v1_ml='', w_har=0.0, w_xgb=0.0, w_lgb=0.0,
                     **{k: x[i] for k, x in kv.items()}) for i, (st, q) in enumerate(zip(T, qs))]
    rows = unit('F', 1, 0.29, role='F_h') + unit('A', 4, 0.25) + unit('B', 3, 0.28)
    rows += unit('C', 2, 0.26, bad=([0, 1], [3]))  # все варианты C ×1,5 в 2018
    wj = [(0, 1, 0), (0, 1, 0), (0, .5, .5), (0, 0, 1), (0, 0, 1)]  # скачок весов 0,5 на парах 3-4 и 4-5
    rows += unit('E', 1, 0.255, wj=wj)
    tab = pd.DataFrame(rows)
    # --- протокол v2 (справочно): c1–c4 на складках, хвост, c4 по отрезкам
    W, L = loyo(tab, 'F', mode='protokol_v2')
    lc = L[L.unit == 'C'].set_index('Y')
    assert lc.loc[2018, 'dop'] and not lc.drop(2018).dop.any()
    le = L[L.unit == 'E'].set_index('Y')
    assert le.loc[2019, 'dop'] and le.loc[2019, 'c4'] and not le.drop(2019).c4.any()
    g = tab[tab.unit == 'E']
    assert not D.flags(g[g.step.isin([2, 3, 5, 6])].sort_values('step'))['c4'], 'без отрезков flags «склеил» бы 2018 и 2020'
    # --- v3 (по умолчанию)
    t9 = pd.DataFrame(rows + fixed('G', [0.28, 0.28, 0.305, 0.305, 0.305])     # c3 везде, лучше E только в 2 складках -> (а) нет
                      + fixed('H', [0.25, 0.25, 0.32, 0.25, 0.25])              # лучше E в 4, c3 нарушен, когда 2018 в обучении -> (б) нет
                      + fixed('K', [0.25, 0.25, 0.25, 0.25, 0.25], calib_param=[1.0, 1.1, 2.0, 2.1, 2.2])  # скачок калибровки 82% -> (в) c4
                      + fixed('Z', [0.249, 0.251, 0.25, 0.25, 0.25], cx=0))     # простой и неотличимый от A
    W9, L9 = loyo(t9, 'F')
    w9 = W9.set_index('unit')
    assert w9.loc['G', 'n_better_E'] == 2 and w9.loc['G', 'c3_vse'] and not w9.loc['G', 'vhod'] and w9.loc['G', 'vhod_net'] == 'a'
    assert w9.loc['H', 'n_better_E'] == 4 and not w9.loc['H', 'vhod'] and 'b' in w9.loc['H', 'vhod_net']
    assert not w9.loc['K', 'c4_T'] and not w9.loc['K', 'vhod'] and 'c:c4' in w9.loc['K', 'vhod_net']
    assert not w9.loc['E', 'c4_T'] and not w9.loc['E', 'vhod']  # (в): c4 на всех 5 годах T
    assert not w9.loc['C', 'vhod'] and w9.loc[['A', 'B', 'Z'], 'vhod'].all()
    pool = [u for u in w9.index if w9.loc[u, 'vhod'] or u == 'F']
    for yy, gL in L9.groupby('Y'):
        gp = gL[gL.unit.isin(pool)]
        assert gL[~gL.unit.isin(pool)]['rank'].isna().all() and abs(gp['rank'].sum() - len(gp) * (len(gp) + 1) / 2) < 1e-9
    assert abs(w9.loc['A', 'R_bar'] - L9[L9.unit == 'A']['rank'].mean()) < 1e-12
    assert np.isclose(w9.loc['A', 'mean_rE'], (L9[L9.unit == 'A'].q / 0.30 - 1).mean())
    # нет утечки: выбор варианта в складке Y не зависит от q на Y (оба режима) — детектор (Д26-аналог для отбора)
    for yy in range(2017, 2022):
        t2 = t9.copy()
        m = t2.step == yy - 2015
        t2.loc[m, 'q_hyb'] *= np.exp(rng.normal(0, 0.2, m.sum()))
        for md in MODES:
            _, L1 = loyo(t9, 'F', mode=md)
            _, L2 = loyo(t2, 'F', mode=md)
            assert (L1[L1.Y == yy].set_index('unit').variant == L2[L2.Y == yy].set_index('unit').variant).all(), (yy, md)
    # штраф за сложность: Z (c = 0) неотличим от лучшего A (c = 1) -> выбирается Z
    vs = vybor_shtraf(W9, k=3)
    best = W9[W9.vhod & ~W9.is_F].sort_values(['R_bar', 'mean_q']).iloc[0]
    assert vs.unit.iloc[0] == 'Z' and best.unit in ('A', 'Z') and len(vs) == 3
    Wx = W9.copy()
    Wx.loc[Wx.unit == 'Z', ['R_bar', 'mean_q']] = [best.R_bar + 1.5, best.mean_q * 1.02]  # отличим -> выбирается лучший
    assert vybor_shtraf(Wx, k=1).unit.iloc[0] != 'Z'
    # сложность по схеме v1
    base1 = {'h': 1, 'leg': {'type': 'HARX', 'regressors': 'S4', 'enet': {'lambda': [0.01, 0.1, 1], 'alpha': [0, 0.5, 1]}},
             'boost': {'role': 'feature', 'lib': 'both_avg'}, 'features': ['G0', 'G1'], 'ens': {'seeds': 3}}
    assert validate(base1) and slozhnost(base1) == 5
    assert slozhnost({**base1, 'features': ['G0', 'G1', 'G2', 'G3'], 'calib': {'kappa': 'fit', 's2_est': 'sm'}}) == 8
    assert slozhnost({'h': 22, 'leg': {'type': 'HARJ'}, 'data': {'info_set': 'd', 'embargo': 'h'}}) == 0
    try:
        validate({**base1, 'calib': {'kappa': 0.7}})
        raise AssertionError('значение вне схемы не поймано')
    except ValueError:
        pass
    # схема v2 (кластеры 16, 18, 20): ключи принимаются; по умолчанию off — не в счёт; переключатель on — +1; сетка списком — 0,
    # одиночное значение сетки — +1 на группу; значение вне списка — отказ; сравнение с r2_core.complexity
    import r2_core as C
    v2a = {**base1, 'k16': {'gate': 'off'}, 'k18': {'lam_to_ref': 'off'}, 'k20': {'sign_delta': 'off'}}
    v2b = {**base1, 'k18': {'lam_to_ref': 'on', 'lam': [0.25, 0.5, 0.75, 1.0]}}
    v2c = {**base1, 'k18': {'lam_to_ref': 'on', 'lam': 0.5}, 'k16': {'gate': 'shock_fresh', 'k': [3, 5, 10], 'q': 0.95},
           'boost': {**base1['boost'], 'grid_point': 'p07'}}
    for dz, c_ in ((v2a, 5), (v2b, 6), (v2c, 10)):
        assert validate(dz) and slozhnost(dz) == c_ == C.complexity(dz), (dz, slozhnost(dz), C.complexity(dz))
    for bad in ({**base1, 'k20': {'delta': 0.15}}, {**base1, 'k16': {'gate': 'vse'}}):
        try:
            validate(bad)
            raise AssertionError('значение вне схемы v2 не поймано')
        except ValueError:
            pass
    # 2σ, финалисты, победитель
    assert sigma2(0.20, 0.21)['passed'] and not sigma2(0.21, 0.21)['passed']
    ref = {s: 0.25 + 0.001 * (s - 46) for s in SEEDS}
    assert sigma2({s: ref[s] - 0.01 + 1e-4 * ((-1) ** s) for s in SEEDS}, ref)['passed']
    assert not sigma2({s: ref[s] + 0.002 * ((-1) ** s) for s in SEEDS}, ref)['passed']
    fin = finalisty(W9, k=2, v0_units=('B',))
    assert list(fin[fin.tip == 'top'].unit) == list(vybor_shtraf(W9[W9.unit != 'B'], k=2).unit) and list(fin[fin.tip == 'V0'].unit) == ['B']
    top1 = fin[fin.tip == 'top'].unit.iloc[0]
    pb = pobeditel(fin, {top1: dict(passed=False), 'B': dict(passed=True)})
    assert pb['pobeditel'] == 'B' and pb['V0']
    assert pobeditel(fin, {})['pobeditel'] == 'F_h(d0)'
    lz = lozhnye({'C': (40, 10)}, 5)
    assert abs(lz['p0'] - 0.0383) < 5e-4 and abs(lz['E_FW_2sigma'] - 5 * lz['p0']) < 1e-12 and lz['vhod_a_pri_nule'] == 0.5
    assert abs(lz['skrining_lozhno_vklyuchennyh'] - 0.35) < 1e-12
    # сдвиг информационного множества: ряд по дню d — argmax 0; отстающий (d−1) и заглядывающий (d+1) — блок; пары после 2021 не берутся
    dts = pd.bdate_range('2016-12-01', '2021-12-31')
    def ar1(n, phi=0.6):
        x = np.zeros(n)
        e = rng.normal(0, 0.5, n)
        for i in range(1, n):
            x[i] = phi * x[i - 1] + e[i]
        return x
    rvd = pd.concat([pd.DataFrame(dict(date=dts, ticker=tk, rv=np.exp(ar1(len(dts)) - 8))) for tk in ('SBER', 'LKOH', 'AFLT')])
    def pred_from(s):
        out = []
        for tk, g in rvd.groupby('ticker'):
            g = g.sort_values('date').reset_index(drop=True)
            p = g.rv.shift(-s) * np.exp(rng.normal(0, 0.2, len(g)))  # прогноз в строке d «знает» RV дня d + s
            out.append(pd.DataFrame(dict(date=g.date, ticker=tk, rv_pred=p)).dropna())
        return pd.concat(out)
    s0, sl, sf = sdvig(pred_from(0), rvd), sdvig(pred_from(-1), rvd), sdvig(pred_from(1), rvd)
    assert s0['argmax'] == 0 and not s0['blok'] and sl['argmax'] == -1 and sl['blok'] and sf['argmax'] == 1 and sf['blok'], (s0, sl, sf)
    # последний известный день: ряд с входом, опубликованным с лагом 1 (как PC по Yahoo), — при K = {−1, 0} не блок, при K = {0} — блок;
    # заглядывание (+1) — блок при любом K
    assert K_dizaina(features=['G0', 'G1']) == (-1, 0) and K_dizaina(features=['G0', 'G3']) == (0,)
    sl1 = sdvig(pred_from(-1), rvd, K=(-1, 0))
    assert not sl1['blok'] and sl1['posledniy_izvestnyy_den'] == '-1..0' and sdvig(pred_from(1), rvd, K=(-1, 0))['blok'], sl1
    assert sdvig_ok(-2, (-1, 0)) == (False, 'отстаёт (info_set < d0)') and sdvig_ok(0, (0,))[0]
    # допуск ε (❓ 3): плоский профиль с +1 выше дня 0 на ≤ ε — проходит с пометкой; > ε — блок; RVI_anchor: K = {−h, 0}
    ok_, why_ = sdvig_ok(1, (0,), corr={-2: 0.300, -1: 0.302, 0: 0.304, 1: 0.310})
    assert ok_ and why_.startswith('плоский профиль')
    assert not sdvig_ok(1, (0,), corr={-2: 0.30, -1: 0.30, 0: 0.30, 1: 0.32})[0]
    assert not sdvig_ok(-1, (-1, 0), corr={-2: 0.30, -1: 0.30, 0: 0.30, 1: 0.39})[0]  # будущее выше диапазона на 0,09 — блок
    # симметричный ε (решение по ❓ 5): MIDAS h=22 (−1 выше дня 0 на 0,008) — проходит «плоский профиль»; прежний HAR-J d−1 (~0,09) — блок
    okm, whym = sdvig_ok(-1, (0,), corr={-2: 0.6933, -1: 0.7031, 0: 0.6949, 1: 0.5572})
    assert okm and whym.startswith('плоский профиль')
    okh, whyh = sdvig_ok(-1, (0,), corr={-2: 0.71, -1: 0.827, 0: 0.727, 1: 0.579})
    assert not okh and whyh.startswith('отстаёт')
    assert K_dizaina({'h': 22, 'leg': {'type': 'RVI_anchor'}}) == (-22, 0) and K_dizaina(leg='RVI_anchor', h=5) == (-5, 0)
    assert sdvig_ok(-2, K_dizaina(leg='RVI_anchor', h=5), corr={-2: 0.305, -1: 0.304, 0: 0.300, 1: 0.299})[0]
    # детектор утечек с допуском: плоский ряд (окно 252 дня) проходит; тот же ряд с подложенной утечкой (RV_{d+1}) — блок
    def ploskiy(leak=0.0):
        out = []
        for tk, g in rvd.groupby('ticker'):
            g = g.sort_values('date').reset_index(drop=True)
            base = np.log(g.rv).rolling(252, min_periods=60).mean()
            lp = (1 - leak) * base + leak * np.log(g.rv.shift(-1))
            out.append(pd.DataFrame(dict(date=g.date, ticker=tk, rv_pred=np.exp(lp))).dropna())
        return pd.concat(out)
    Ka = K_dizaina(leg='RVI_anchor', h=5)  # окно прошлого: профиль убывает к +1, K = {−5, 0}
    sp0, sp1 = sdvig(ploskiy(0.0), rvd, K=Ka), sdvig(ploskiy(0.3), rvd, K=Ka)
    assert sp0['ok'] and sp1['blok'] and sp1['argmax'] == 1, (sp0, sp1)
    assert sdvig(pred_from(1), rvd)['blok'] and sdvig(pred_from(1), rvd, K=(-22, 0))['blok']
    # детектор с симметричным допуском: подложенная утечка (d+1) и подложенный лаг (d−1) — оба блок при K = {0}
    lk, lg = sdvig(pred_from(1), rvd), sdvig(pred_from(-1), rvd)
    assert lk['blok'] and lk['prichina'].startswith('заглядывание') and lg['blok'] and lg['prichina'].startswith('отстаёт'), (lk, lg)
    bf = Path(tempfile.mkdtemp()) / 'blok.csv'
    pd.DataFrame([dict(ryad='core', h=1, argmax=-1, ok=False)]).to_csv(bf, index=False)  # строка ядра без нового поля
    blok_zapis([dict(ryad='O', h=1, argmax=1, ok=False, posledniy_izvestnyy_den='0')], f=bf)
    bb = pd.read_csv(bf)
    assert list(bb.ryad) == ['core', 'O'] and 'posledniy_izvestnyy_den' in bb and bb.posledniy_izvestnyy_den.isna().iloc[0]
    if (REPO / 'docs/otchety/r2_F/shift_check.csv').exists():  # решение по F — на его таблицах (запись во временный каталог)
        out0_, OUT = OUT, Path(tempfile.mkdtemp())
        try:
            rf = reshit_F(zapis=False)
        finally:
            OUT = out0_
        d = rf.set_index('obekt').reshenie
        assert d['группа G1'] == 'ок' and d['столбец g1_pc1'] == 'ок' and d['столбец g1_pc2'] == 'ок' and d['столбец g1_pc3'] == 'ок'
        tri = {'столбец g7_ln_amihud_w', 'столбец g7_ln_amihud_m', 'столбец g4_jbig_n22'}  # в F v2 (PR #170) столбцы уже убраны
        est = tri & set(d.index)
        assert (d[list(est)] == 'блок').all() and set(rf[rf.reshenie == 'блок'].obekt) == est
    # чтение: шаги 7+ не загружаются; OOS s²; заморозка множителей; median-агрегат
    tmp = Path(tempfile.mkdtemp())
    dd = pd.DataFrame([(pd.Timestamp(f'{2015 + st}-06-01') + pd.Timedelta(days=i), 'T', 2015 + st, st, 1.0, np.exp(-0.1 * st))
                       for st in range(1, 11) for i in range(5)], columns=['date', 'ticker', 'year', 'step', 'rv_actual', 'rv_pred'])
    dd.to_parquet(tmp / 'x_val.parquet')
    dd.assign(rv_pred=dd.rv_pred * 2).to_parquet(tmp / 'y_val.parquet')
    dd.assign(rv_pred=dd.rv_pred * 10).to_parquet(tmp / 'z_val.parquet')
    rd = read_val(tmp / 'x_val.parquet')
    assert rd.step.max() == 6 and rd.year.max() == 2021
    q1 = q_steps(rd, 'ord')
    dd2 = rd.copy()
    dd2.loc[dd2.step == 3, 'rv_pred'] *= 3
    assert q_steps(dd2, 'ord')[2] == q1[2] and q_steps(dd2, 'ord')[3] != q1[3]
    assert factors(rd, 'sm')[7] == factors(rd, 'sm')[6] == factors(rd, 'sm')[10]
    fs = [tmp / 'x_val.parquet', tmp / 'y_val.parquet', tmp / 'z_val.parquet']
    assert np.allclose(series(fs, agg='median').rv_pred, rd.rv_pred * 2) and np.allclose(series(fs).rv_pred, rd.rv_pred * 13 / 3)
    # гейты C и E (во временный каталог): yaml читается, решение и хэши входов на месте
    out0, OUT = OUT, tmp / 'O'
    try:
        gc = yaml.safe_load(gate_C(W9.assign(role=W9.role.replace({'residual': 'residual'})), 1, [tmp / 'x_val.parquet'], v0_units=('B',)).read_text())
        assert gc['phase'] == 'C' and len(gc['yacheyki']) == 2 and gc['v0_v_finalisty'][0]['unit'] == 'B' and gc['inputs']
        dzf = tmp / 'pobeditel.yaml'
        dzf.write_text(yaml.safe_dump(dict(design='тест', **{k: v for k, v in base1.items()}), allow_unicode=True))
        ge = yaml.safe_load(gate_E(W9, 1, {top1: sigma2(0.20, 0.21)}, [tmp / 'x_val.parquet'], v0_units=('B',),
                                   dizainy={top1: dzf}).read_text())
        import r2_core as C
        pw = ge['pobeditel']
        assert pw['pobeditel'] == top1 and abs(ge['lozhnye']['p0'] - 0.0383) < 5e-4 and ge['code_sha256']
        assert pw['design_sha'] == C.design_hash(yaml.safe_load(dzf.read_text())) and pw['cache_dir'] == pw['design_sha'][:16]
        assert pw['yaml_file_sha256'] == sha(dzf) and hashlib.sha256(pw['canonical_yaml'].encode()).hexdigest() == pw['design_sha']
        assert str(dzf) in ge['inputs']
        # лестница и абляции (для D): порядок ступеней, design_sha через r2_core, ступень = накопление групп ключей победителя
        L_ = ge['lestnica']
        assert [x['stupen'] for x in L_] == ['HAR-J', '+s²', '+HAR-X', '+калибровка', '+бустинг', '+согласование', 'финал']
        assert L_[0]['design_sha'] is None and L_[1]['ryad']['s2'] == 'sm' and L_[-1]['design_sha'] == pw['design_sha']
        assert L_[3]['bez_izmeneniy'] and not L_[4]['bez_izmeneniy'] and L_[5]['bez_izmeneniy'] and L_[-1]['bez_izmeneniy']
        hx = yaml.safe_load(Path(L_[2]['design_yaml']).read_text())
        assert hx['leg'] == base1['leg'] and 'boost' not in hx and 'features' not in hx
        ab = {a['klaster']: a for a in ge['ablyacii']}
        assert set(ab) == {2, 4, 5} and set(ab[5]['klyuchi']) == {'features'} and set(ab[2]['klyuchi']) == {'leg.type', 'leg.regressors'}
        a4 = yaml.safe_load(Path(ab[4]['design_yaml']).read_text())
        assert 'role' not in a4.get('boost', {}) and 'lib' not in a4.get('boost', {}) and a4['features'] == base1['features']
        assert all(a['design_sha'] != pw['design_sha'] for a in ge['ablyacii']) and pw['ryady']['cache_dir'].endswith(pw['cache_dir'])
        assert yaml.safe_load(gate_E(W9, 1, {}, v0_units=('B',)).read_text())['lestnica'] is None
        try:
            gate_E(W9, 1, {top1: sigma2(0.20, 0.21)}, v0_units=('B',))
            raise AssertionError('победитель без yaml дизайна не остановлен')
        except ValueError as e:
            assert 'design_sha' in str(e)
        assert yaml.safe_load(gate_E(W9, 1, {}, v0_units=('B',)).read_text())['pobeditel']['design_sha'] is None  # F_h(d0)
    finally:
        OUT = out0
    # HAR-CJ по d−1 — отказ
    harcj_guard(CFG['harcj_d0'].format(h=5))
    try:
        harcj_guard('HARCJ_h5_val.parquet', 'тест')
        raise AssertionError('HAR-CJ по d−1 не пойман')
    except AssertionError as e:
        assert 'd−1' in str(e)
    print('selftest r2_loyo: OK (v3: вложенный выбор по c3, вход (а)–(в) с c1/c2/c4 на T и калибровкой, ранги среди прошедших, '
          'штраф за сложность, сложность и валидация по schema_v2 (= r2_core.complexity), финалисты, 2σ, победитель, ложные победители, сдвиг −1/+1 пойман; '
          'нет утечки отложенного года; v2 справочно; чтение шагов ≤ 6, OOS s², median-агрегат, HAR-CJ)')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', nargs='?', choices=['proverka', 'level', 'gate-d', 'sdvig-f', 'sdvig-peresmotr'])
    ap.add_argument('--selftest', action='store_true')
    ap.add_argument('--h', type=int, nargs='*', default=[1, 5, 22])
    ap.add_argument('--level', type=int, default=1)
    ap.add_argument('--v0', nargs='*', default=[], help='единицы — V0-кандидаты (HAR-X+s², RVI-anchor+s²) на этом уровне')
    ap.add_argument('--cells', nargs='*', default=[], help='gate-d: ячейки из gate_C')
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if a.cmd == 'proverka':
        return proverka(a.h)
    if a.cmd == 'level':
        for h in a.h:
            run_level(a.level, h, tuple(a.v0))
        return
    if a.cmd == 'sdvig-peresmotr':  # пересмотр блоков кэша по правилу с K ноги и допуском ε (❓ 3)
        print(peresmotr_blok().to_string(index=False), flush=True)
        return
    if a.cmd == 'sdvig-f':  # решение O по проверке сдвига агента F (PR #163) с последним известным днём
        print(reshit_F().to_string(index=False), flush=True)
        return
    if a.cmd == 'gate-d':
        for h in a.h:
            cells = a.cells or [c['unit'] for c in yaml.safe_load((OUT / f'gate_C_h{h}.yaml').read_text())['yacheyki']]
            print(gate_D(h, cells), flush=True)
        return
    ap.print_help()


if __name__ == '__main__':
    main()
