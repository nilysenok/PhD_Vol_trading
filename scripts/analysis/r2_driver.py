#!/usr/bin/env python3
"""R2, агент S: драйвер фаз B–E (протокол v4, разделы 0, 3–7; решение владельца 16:25 — весь счёт на Маке).

Что делает:
  generate — все yaml уровней B–E сейчас, до счёта (models/designs/r2/driver_v1/): каждый дизайн проходит r2_core.validate;
             B — план 2^(14−8) IV (h = 5, 22; базы O/factor_v1 с настройками v4: 1 seed, LightGBM) и 2³ по J, K, L (h = 1);
             эконометрика (кластеры 2, 3, 6, 14, 15; роль none) на h = 1, 5, 22; C — шаблоны ячеек нога × роль (факторы B
             подставляются по gate_B); D — точки LHS-60; E — надстройки к лучшему (по одной) и параметры вне схемы (кластеры 16, 18, 20).
             Дизайнам с persist/reconcile на h ≠ 1 дописываются входы h=1 (base_1, leg_1) — они входят в хэш (r2_core.with_inputs).
  run      — конвейеры h (5 и 22 параллельно, плюс h = 1), эконометрика стартует сразу, не ждёт бустинга; фазы — по файлам ворот
             R2/O/gate_{B,C,D,E}_h{h}.yaml; ворота считает O (r2_gates.ocenka / vybor, r2_halving.otsechenie, r2_factor.effects,
             r2_loyo.sigma2 / write_gate), драйвер подаёт им ряды из кэша (r2_core.load_series); после каждых ворот —
             R2/report/gate_{X}.md для владельца. Остановка — только перед открытием теста (R2/report/STOP_pered_otkrytiem.md).
             Нет модуля другого агента или ворот — этап ждёт (опрос, poll_s) и пишет причину в состояние и отчёт.
  smoke    — дым: 3 тикера (SBER, LKOH, AFLT) × шаги 1–3 × seed 42 × h ∈ {5, 22, 1}; модули заменены дымовым (HAR-J d0 r2_core),
             ворота — функции O на двух складках (2017, 2018); всё — в R2/S/driver_smoke/.
Внутри: режим R2 r2_core (прогнозы ≥ 2022 не пишутся), проверка сдвига п. 2.6 (run_phase каждого модуля), детектор утечек модуля —
один раз на (модуль, h) перед первым боевым запуском (команда leak модуля; провал — этап блокируется).

python3 scripts/analysis/r2_driver.py generate
python3 scripts/analysis/r2_driver.py run [--h 5 22 1] [--once]
python3 scripts/analysis/r2_driver.py smoke
python3 scripts/analysis/r2_driver.py status
"""
import os

for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_k, '1')

import argparse
import copy
import itertools
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
if os.environ.get('R2_O_PATH'):  # модули O из непринятого PR (дым до слияния): каталог scripts/analysis копии ветки O
    sys.path.insert(0, os.environ['R2_O_PATH'])
import r2_core as C  # noqa: E402

CFG = yaml.safe_load((REPO / 'models' / 'designs' / 'r2' / 'r2_S_driver.yaml').read_text())
DDIR = REPO / CFG['designs_dir']
REPORT = C.REGEN / CFG['report_dir']
STATE = C.REGEN / CFG['state_dir']
FACTOR_V1 = REPO / 'models' / 'designs' / 'r2' / 'factor_v1'
LOCK = threading.Lock()
RUN_EXTRA = {}  # модуль -> добавка к {extra} (только дым --real: --tickers у B, BG)


def log(*a):
    with LOCK:
        print(time.strftime('%F %T'), *a, flush=True)


def ydump(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(json.loads(json.dumps(obj, default=str)), allow_unicode=True, sort_keys=False))
    return path


def designs_file(name):
    return DDIR / f'{name}.yaml'


def write_designs(name, designs, meta=None):
    """Каждый дизайн — через r2_core.validate (ошибка схемы = стоп до счёта); в файле — design_sha и сложность.
    Дирижёр 17:44: у бустинга (роль ≠ none) ранняя остановка — только на внутренней складке (boost.early_stop = qlike_total:
    год vy−1 с эмбарго h, ≤ 300 деревьев, patience 30); native (val-год, ≤ 1000) запрещена и в validate."""
    rows = []
    for d in designs:
        if str(C.get(C.validate(d, es_check=False), 'boost.role')) != 'none':
            d.pop('boost.early_stop', None)
            _set(d, 'boost.early_stop', 'qlike_total')
        full = C.validate(d)
        rows.append(dict(design=full.get('design', ''), design_sha=C.design_hash(d), slozhnost=C.complexity(d)))
    return ydump(dict(meta or {}, n=len(designs), designs=designs, index=rows), designs_file(name))


def read_designs(name):
    return yaml.safe_load(designs_file(name).read_text())['designs']


def zamer(m, h, tag, n_designs, n_seeds, n_steps, workers, sek, rc, bez_done):
    """Замер запуска модуля -> R2/S/driver/zamery.csv (основа оценки времени боевого счёта)."""
    f = STATE / 'zamery.csv'
    f.parent.mkdir(parents=True, exist_ok=True)
    new = not f.exists()
    with LOCK, open(f, 'a', encoding='utf-8') as fh:
        if new:
            fh.write('vremya,modul,h,tag,dizaynov,seeds,shagov,workers,tikery,sek,rc,bez_done\n')
        fh.write(f'{time.strftime("%F %T")},{m},{h},{tag},{n_designs},{n_seeds},{n_steps},{workers},'
                 f'{os.environ.get("R2_TICKERS", "17").replace(",", "+")},{sek:.1f},{rc},{bez_done}\n')


# ================================================================ generate: все yaml B–E до счёта
def _set(d, dotted, v):
    a = d
    ks = dotted.split('.')
    for k in ks[:-1]:
        a = a.setdefault(k, {})
    a[ks[-1]] = v


def h1_inputs():
    """Входы h=1 для persist / reconcile на фазе B (O, factor_v1 mh_vhod): F_1(d0) и E_1 — идентичность + SHA содержимого."""
    m = yaml.safe_load((FACTOR_V1 / 'h5.yaml').read_text())['mh_vhod']
    return {k: {kk: v for kk, v in m[k].items() if kk in ('file', 'col', 'ryad')} for k in ('base_1', 'leg_1')}


def _maybe_inputs(d):
    return C.with_inputs(d, h1_inputs()) if C.needs_h1_inputs(d) else d


def gen_B():
    """h = 5, 22: 64 прогона O (factor_v1) с настройками v4 (1 seed, LightGBM); h = 1: 2³ по J, K, L."""
    import r2_factor as RF
    out = {}
    for h in (5, 22):
        Y = yaml.safe_load((FACTOR_V1 / f'h{h}.yaml').read_text())
        ds = []
        for d in Y['dizainy']:
            d = copy.deepcopy(d)
            _set(d, 'boost.lib', CFG['B']['lib'])
            _set(d, 'ens.seeds', 1)
            _set(d, 'boost.row_frac', CFG['B']['row_frac'])  # в хэше: прогон скрининга ≠ ячейка C на всех строках
            ds.append(_maybe_inputs(d))
        out[h] = write_designs(f'B_h{h}', ds, dict(faza='B', h=h, plan='factor_plan_v1.csv (O)', nastroyki='v4: 1 seed, LightGBM, 50% строк',
                                                   row_frac=CFG['B']['row_frac'], row_seed=CFG['B']['row_seed'],
                                                   mh_vhod='base_1, leg_1 — в inputs дизайнов с persist/reconcile (входят в хэш)'))
    base1 = copy.deepcopy(RF.BAZA[1])
    base1 = {k: v for k, v in base1.items() if not k.startswith('_')}
    ds, plan = [], []
    for r, xs in enumerate(itertools.product((-1, 1), repeat=3), 1):
        x = dict(zip(CFG['B']['h1_factors'], xs[::-1]))
        d = dict(copy.deepcopy(base1), design=f'B_h1_r{r} (J/K/L = {x["J"]:+d}/{x["K"]:+d}/{x["L"]:+d})', h=1)
        _set(d, 'boost.lib', CFG['B']['lib'])
        _set(d, 'ens.seeds', 1)
        _set(d, 'boost.row_frac', CFG['B']['row_frac'])
        for f in CFG['B']['h1_factors']:
            if x[f] > 0:
                d['features'] = d['features'] + [g for g in RF.FAKTORY[f][2] if g not in d['features']]
        ds.append(d)
        plan.append(dict(run=r, **x))
    out[1] = write_designs('B_h1', ds, dict(faza='B', h=1, plan='2³ по J, K, L (v4 § 3)', tablica=plan))
    return out


def gen_econ():
    """Кластеры 2, 3, 6, 14, 15 сразу (роль none), h = 1, 5, 22."""
    E = CFG['econ']
    out = {}
    for h in (1, 5, 22):
        ds = []
        for leg in E['legs']:
            d = dict(design=f'econ_h{h}_{leg}', h=h, leg=dict(type=leg))
            if leg in ('HARX', 'HARX_regime', 'HARX_pooled'):
                d['leg']['regressors'] = list(E['harx_regressors'][h])
            if leg == 'HARX_pooled':
                d['leg']['pooled_kappa'] = [0, 0.5, 1]
            if leg == 'HAR_perlead' and h == 1:
                continue
            ds.append(d)
        for leg in E['base_legs']:
            b = dict(h=h, leg=dict(type=leg))
            if leg == 'HARX':
                b['leg']['regressors'] = list(E['harx_regressors'][h])
            mods = [('calib.s2_est', v) for v in ('smrob', 'sm_pooled', 'sm_by_ticker', 'sm_roll1y', 'sm_roll2y', 'sm_ewma097')]
            mods += [('calib.kappa', 'fit'), ('calib.s2_state', 'RVI'), ('calib.s2_state', 'RVI_volvol'), ('calib.s2_state', 'regime2'),
                     ('calib.c_qlike', 'roll_all'), ('calib.c_qlike', 'roll_2y'), ('calib.mz', 'on'), ('calib.update', 'daily')]
            mods += [('leg.panel', v) for v in ('factor', 'factor_shrink_025', 'factor_shrink_05', 'factor_shrink_state')]
            mods += [('leg.refit', 'quarterly'), ('leg.refit', 'monthly'), ('leg.rls_lambda', 0.995), ('leg.rls_lambda', 0.99),
                     ('leg.estimator', 'qmle_level')]
            if h != 1:
                mods += [('mh.scheme', v) for v in (('decomp', 'iterated', 'joint_h', 'perlead') if h == 5 else ('decomp', 'iterated', 'joint_h'))]
                mods += [('mh.persist', v) for v in ('OLS', 'qlike', 'pos_neg', 'sqrtRQ')] + [('mh.slope', 'on'), ('mh.reconcile', 'MinT')]
            for k, v in mods:
                d = copy.deepcopy(b)
                d['design'] = f'econ_h{h}_{leg}_{k}={v}'
                _set(d, k, v)
                if k == 'mh.scheme' and v == 'decomp':
                    _set(d, 'mh.decomp_k', 1 if h == 5 else 5)
                if k == 'mh.scheme' and v == 'iterated':
                    _set(d, 'mh.iterated_k', 5)
                ds.append(_maybe_inputs(d))
        out[h] = write_designs(f'econ_h{h}', ds, dict(faza='econ', h=h, klastery=[2, 3, 6, 14, 15], rol='none'))
    return out


def gen_C():
    """Шаблоны ячеек (h, нога, роль); факторы, включённые в B, подставляются по gate_B (apply_gate_B)."""
    out = {}
    for h in (1, 5, 22):
        ds = []
        for leg in CFG['C']['legs']:
            if leg == 'HAR_perlead' and h == 1:
                continue
            for role in CFG['C']['roles']:
                d = dict(design=f'C_h{h}_{leg}_{role}', h=h, leg=dict(type=leg), boost=dict(role=role, lib=CFG['C']['lib']),
                         features=['G0', 'G1'])
                if leg == 'HARX':
                    d['leg']['regressors'] = list(CFG['econ']['harx_regressors'][h])
                ds.append(d)
        for leg in CFG['C']['v0']:
            d = dict(design=f'C_h{h}_{leg}_V0', h=h, leg=dict(type=leg), boost=dict(role='none'))
            if leg == 'HARX':
                d['leg']['regressors'] = list(CFG['econ']['harx_regressors'][h])
            ds.append(d)
        out[h] = write_designs(f'C_h{h}_shablon', ds, dict(faza='C', h=h, primechanie='факторы B — по gate_B_h{h}.yaml при запуске'))
    return out


def gen_D():
    from scipy.stats import qmc
    L = CFG['D']['levels']
    names = list(L)
    U = qmc.LatinHypercube(d=len(names), seed=CFG['D']['lhs_seed']).random(CFG['D']['n'])
    pts = []
    for i, u in enumerate(U, 1):
        pts.append(dict(tochka=f'p{i:02d}', **{n: L[n][int(np.floor(u[j] * len(L[n])))] for j, n in enumerate(names)}))
    pts.append(dict(tochka='p61_tekushchaya', kontrol='текущая точка ячейки (точки R1)'))
    return ydump(dict(faza='D', lhs=dict(seed=CFG['D']['lhs_seed'], d=len(names), n=CFG['D']['n']), urovni=L,
                      max_trees=CFG['D']['max_trees'], patience=CFG['D']['patience'], seeds=CFG['D']['seeds'], tochki=pts),
                 DDIR / 'D_lhs60.yaml')


def gen_E():
    return ydump(dict(faza='E', ofat=CFG['E']['ofat'], vne_skhemy=CFG['E']['vne_skhemy'], seeds_final=CFG['E']['seeds_final'],
                      lib_final=CFG['E']['lib_final'], finalists_k=CFG['E']['finalists_k'],
                      primechanie='дизайны E = лучший дизайн gate_D + одно изменение ofat; vne_skhemy — параметры модуля G (❓ schema_v2)'),
                 DDIR / 'E_nadstroyki.yaml')


def cmd_generate(a=None):
    t0 = time.time()
    B, Ec, Cc, D, E = gen_B(), gen_econ(), gen_C(), gen_D(), gen_E()
    n = {f.name: yaml.safe_load(f.read_text()).get('n') for f in sorted(DDIR.glob('*.yaml'))}
    log(f'generate: {DDIR} — {n}; за {time.time() - t0:.1f} с')
    return 0


# ================================================================ материализация уровней по воротам
def gate_path(phase, h, smoke=False):
    return (SMOKE_O if smoke else C.O_DIR) / f'gate_{phase}_h{h}.yaml'


NEPR = {}  # (h, имя дизайна) -> [факторы B, неприменимые в цепочке модулей ячейки] — пометка в таблицах ворот


def _module_yaml(m):
    f = CFG['modules'][m].get('yaml')
    p = REPO / 'models' / 'designs' / 'r2' / f if f else None
    return yaml.safe_load(p.read_text()) if p and p.exists() else {}


def factor_unsupported(m, f):
    """Правило дирижёра: модуль объявляет неприменимые факторы в своём yaml — `neprimenimye_faktory` (буквы A–O, ключи схемы или
    группы признаков); у P (PR #181) — разделом `applicable` (features, data_supported, boost_residual_only_default, mh)."""
    import r2_factor as RF
    Y = _module_yaml(m)
    key, _, plus = RF.FAKTORY[f]
    if 'neprimenimye_faktory' in Y:
        L = set(map(str, Y['neprimenimye_faktory']))
        groups = plus if key == 'features' else []
        return f in L or key in L or key.split('.')[0] in L or any(g in L for g in groups)
    A = Y.get('applicable')
    if not A:
        return False
    if key == 'features':
        return bool(A.get('features')) and any(g not in A['features'] for g in plus)
    a, b = key.split('.')
    if a == 'data':
        return b not in (A.get('data_supported') or [])
    if a == 'boost':
        return b in (A.get('boost_residual_only_default') or [])
    if a == 'mh':
        return 'mh' in A
    return False  # calib.*, ens.* — на ряд модуля не влияют (их применяет следующий модуль цепочки)


def apply_gate_B(designs, gate, h=None):
    """Включённые в B факторы (+1) переносятся в ячейки C (ключи — r2_factor.FAKTORY). Если фактор неприменим хотя бы к одному модулю
    цепочки ячейки (объявлено в yaml модуля) — у этой ячейки он остаётся off, ячейка не выбрасывается, в таблице ворот — пометка."""
    import r2_factor as RF
    on = gate.get('vklyucheny') or []
    out = []
    for d in designs:
        d = copy.deepcopy(d)
        chain = route(d)
        nepr = []
        for f in on:
            bad = [m for m in chain if factor_unsupported(m, f)]
            if bad:
                nepr.append(f'{f} ({",".join(bad)})')
                continue
            key, _, plus = RF.FAKTORY[f]
            if key == 'features':
                d['features'] = list(d.get('features', ['G0'])) + [g for g in plus if g not in d.get('features', ['G0'])]
            else:
                _set(d, key, plus)
        if nepr:
            NEPR[(h, d.get('design'))] = nepr
        out.append(_maybe_inputs(d))
    return out


def faktory_na_E(gate):
    """Решение владельца 18:35: околопороговый фактор скрининга не выключается, а идёт на уровень E явным вариантом поверх финалиста h.
    Критерий: не включён в B, средний эффект e_bar ≤ −0,005 (порог ворот B, `porog.e_bar_max`) и полоса пересекает ноль
    (ci_low ≤ 0 ≤ ci_high). Положительный средний эффект — выключен. -> [буквы факторов]."""
    thr = float(((gate or {}).get('porog') or {}).get('e_bar_max', -0.005))
    out = []
    for f in (gate or {}).get('faktory') or []:
        if f.get('vklyuchen'):
            continue
        e, lo, hi = f.get('e_bar'), f.get('ci_low'), f.get('ci_high')
        if e is not None and lo is not None and hi is not None and e <= thr and lo <= 0 <= hi:
            out.append(f['faktor'])
    return out


def designs_faktory_E(best, letters, h=None):
    """Варианты уровня E: лучший дизайн D (финалист h) + один околопороговый фактор B (+1 уровень, r2_factor.FAKTORY). Фактор,
    неприменимый к модулю цепочки, пропускается с пометкой (NEPR). -> [дизайн]."""
    import r2_factor as RF
    out = []
    for f in letters:
        d = copy.deepcopy(best)
        key, _, plus = RF.FAKTORY[f]
        if key == 'features':
            d['features'] = list(d.get('features', ['G0'])) + [g for g in plus if g not in d.get('features', ['G0'])]
        else:
            _set(d, key, plus)
        d['design'] = f'E_{best.get("design", "best")}_faktor{f}'
        bad = [m for m in route(d) if factor_unsupported(m, f)]
        if bad:
            NEPR[(h, d['design'])] = [f'{f} ({",".join(bad)})']
            continue
        try:
            C.validate(d)
        except C.DesignError as e:
            log(f'фактор {f} поверх {best.get("design")} не принят схемой: {e}')
            continue
        if C.design_hash(d) != C.design_hash(best):  # фактор уже в лучшем — вариант совпал бы с ним
            out.append(_maybe_inputs(d))
    return out


# ---------------------------------------------------------------- уровень E по блоку владельца 20:10 (E1–E7 поверх финала R1 d0 и ноги)
def baza_R1(h):
    """Основа уровня E (владелец 20:10): финал R1 на данных по день d — прогон плана r01 (все факторы на уровне −1, D12) на ВСЕХ строках,
    LightGBM, 1 seed (финалисты потом — 10 seed, обе библиотеки), и его нога (HAR-J, роль none). -> (финал, нога)."""
    b = copy.deepcopy(yaml.safe_load((FACTOR_V1 / f'h{h}.yaml').read_text())['dizainy'][0])
    b['design'] = f'E_h{h}_F_R1_d0'
    _set(b, 'boost.lib', CFG['B']['lib'])
    _set(b, 'ens.seeds', 1)
    _set(b, 'boost.early_stop', 'qlike_total')
    noga = dict(design=f'E_h{h}_F_R1_noga', h=h, leg=copy.deepcopy(b.get('leg') or {'type': 'HARJ'}), features=['G0'])
    return b, noga


def hyb_inputs(h):
    """hyb₁ (и hyb₅ на h=22) для E1/E2: финалист своего h из этого прогона (gate_E, иначе gate_D, design_sha + ряд), иначе F_h(d0) с
    эмбарго s² (R2/cache/VYR/F_d0_s2emb_h{k}_s{seed}). leg_1 — как в factor_v1 (E_1). -> inputs."""
    inp = h1_inputs()
    for k in ((1, 5) if h == 22 else (1,)):
        src = None
        for p in ('E', 'D'):
            f = gate_path(p, k)
            if f.exists():
                g = yaml.safe_load(f.read_text()) or {}
                sha = g.get('design_sha') or (C.design_hash(g['dizayn']) if g.get('dizayn') else None)
                if sha and not (isinstance(g.get('pobeditel'), dict) and g['pobeditel'].get('pobeditel') == 'F_h(d0)'):
                    src = {'design_sha': sha, 'series': 'L3'}
                    break
        inp[f'base_{k}'] = src or {'file': f'R2/cache/VYR/F_d0_s2emb_h{k}_s{{seed}}_val.parquet'}
    return inp


def designs_E_2010(h):
    """Варианты уровня E (владелец 20:10) поверх основы baza_R1: -> [(вариант, исполнитель, дизайн | None, примечание)]."""
    b, noga = baza_R1(h)
    hy = hyb_inputs(h)
    out = [('E0 финал R1 d0', 'L>B>L3', b, ''), ('E0 нога финала', 'L>L3', noga, '')]

    def v(tag, mod, base, kv, note='', inputs=False, name=None):
        d = copy.deepcopy(base)
        for k, x in kv.items():
            _set(d, k, x)
        d['design'] = name or f'{base["design"]}_{tag.split()[0]}'
        if inputs:
            d = C.with_inputs(d, hy)
        try:
            C.validate(d)
            out.append((tag, mod, d, note))
        except C.DesignError as e:
            out.append((tag, mod, None, f'не принят схемой: {str(e).splitlines()[-1].strip()}'))
        return d

    e1 = [v('E1 разложение hyb₁', 'M', b, {'mh.scheme': 'decomp', 'mh.decomp_k': 1}, '(hyb₁ + (h−1)·HAR₂₋h)/h', True, f'E_h{h}_E1_k1')]
    if h == 22:
        e1.append(v('E1 разложение hyb₅', 'M', b, {'mh.scheme': 'decomp', 'mh.decomp_k': 5}, '(hyb₅ + 17·HAR₆₋₂₂)/22', True, f'E_h{h}_E1_k5'))
    e2 = v('E2 forward HAR', 'M', b, {'mh.persist': 'OLS'}, 'hyb₁ вместо RV_d' + ('; ❓ M: hyb₅ вместо RV_w — нет в коде' if h == 22 else ''),
           True, f'E_h{h}_E2')
    e3 = v('E3 равные веса ног', 'E', b, {'ens.leg_ens': 'equal'}, '½ HAR-J+s² + ½ HAR-X+s²; ❓ E: набор ног и вариант с RVI по ⅓ — нет в коде',
           name=f'E_h{h}_E3')
    e4 = v('E4 AveW HAR-X', 'E', dict(noga, leg=dict(type='HARX')), {'ens.window_ens': 'on'}, 'окна: расширяющееся, 5 лет, 3 года',
           name=f'E_h{h}_E4')
    v('E5 воздержание shock_fresh', 'G', b, {'k16.gate': 'shock_fresh'}, 'k ∈ {3, 5, 10} — вложенно по t; halt — только при сборке (A)',
      name=f'E_h{h}_E5')
    for i, x in enumerate(e1 + [e2, e3, e4], 1):
        v(f'E6 усадка P поверх {x["design"].split("_")[-1]}', 'P', x, {'leg.panel': 'factor_shrink_05'}, 'кросс-секционная усадка (кластер 14)',
          name=f'{x["design"]}_E6')
    v('E7 Хьюбер', 'L', b, {'leg.robust': 'Huber'}, '❓ M: «среднее прямого и итерированного HAR» — нет в коде', name=f'E_h{h}_E7')
    for f in CFG['E'].get('faktory_na_E_20_15', {}).get(h, []):
        for d in designs_faktory_E(b, [f], h):
            out.append((f'фактор {f} (ворота B, околопороговый)', ','.join(route(d)[1:-1]) or 'L', d, ''))
    for kv, mod in (({'leg.struct': 'tau_midas'}, 'M'), ({'leg.struct': 'tau_boost'}, 'M'), ({'leg.struct': 'tau_rvi'}, 'M'),
                    ({'leg.struct_split': 'bv_plus_jump'}, 'M'), ({'leg.moe': 'k2'}, 'E'), ({'leg.moe': 'k3'}, 'E')):
        k, x = next(iter(kv.items()))
        v(f'кл. 11–13 {k}={x}', mod, b, kv, 'в E, если модуль ставит DONE (дым); иначе — «не прошёл дым»', name=f'E_h{h}_{k.split(".")[-1]}_{x}')
    return out


def cmd_ryady_E(a):
    """Ряды уровня E (20:10) сразу, без ворот C/D: дизайны -> driver_v1/E_2010_h{h}.yaml (до счёта), затем пакетом через run модулей,
    h — параллельно, воркеров на h = --workers. Отчёт — R2/report/E_2010_varianty.md (что построено, исполнитель, DONE)."""
    hs = a.h or [5, 22]
    runner = Runner(False)
    import r2_dataset_cache as DS
    log(f'место на томах кэша (ГБ): {DS.proverka_mesta()}')
    plan = {h: designs_E_2010(h) for h in hs}
    for h in hs:
        ds = [x[2] for x in plan[h] if x[2] is not None]
        write_designs(f'E_2010_h{h}', ds, dict(faza='E', h=h, osnova='финал R1 d0 (r01, все строки) и нога', blok='владелец 20:10',
                                               varianty=[dict(variant=t, modul=m, design=(d or {}).get('design'), primechanie=n)
                                                         for t, m, d, n in plan[h]]))
    if a.only_generate:
        return 0
    res = {}

    def go(h):
        ds = read_designs(f'E_2010_h{h}')
        res[h] = runner.chain(ds, h, [42], CFG['steps'], a.workers, 'E', partial=True)
    th = [threading.Thread(target=go, args=(h,), name=f'E_h{h}') for h in hs]
    for t in th:
        t.start()
    for t in th:
        t.join()
    L = [f'# Уровень E — варианты блока владельца 20:10 ({time.strftime("%F %T")})', '',
         'Основа — финал R1 на данных по день d (прогон r01 плана, все строки) и его нога; seed 42; отбор — ворота E (панельная t, LOYO).', '']
    for h in hs:
        L += [f'## h = {h}', '', f'- цепочки: {res[h]}', '', '| вариант | исполнитель | дизайн | ряд L3 | примечание |', '|---|---|---|---|---|']
        for t, m, d, n in plan[h]:
            ok_ = bool(d is not None and final_series(d, h, [42])) if d is not None else False
            L.append(f'| {t} | {m} | {(d or {}).get("design", "—")} | {"есть" if ok_ else "нет"} | '
                     f'{n}{"" if ok_ or d is None else ("; не прошёл дым / модуль не поставил ряд" if "кл. 11" in t else "; ряда нет")} |')
        L.append('')
    REPORT.mkdir(parents=True, exist_ok=True)
    (REPORT / 'E_2010_varianty.md').write_text('\n'.join(L), encoding='utf-8')
    log(f'ряды E: {res} -> {REPORT / "E_2010_varianty.md"}')
    return 0


def designs_E(best, ofat):
    out = []
    for k, vals in ofat.items():
        for v in vals:
            d = copy.deepcopy(best)
            d['design'] = f'E_{best.get("design", "best")}_{k}={v}'
            _set(d, k, v)
            try:
                C.validate(d)
            except C.DesignError:
                continue
            out.append(_maybe_inputs(d))
    return out


# ================================================================ модули: маршрут, запуск, детектор
def route(design):
    """Цепочка модулей дизайна по ключам схемы (r2_S_driver.yaml chain_order)."""
    f = C.validate(design)
    g = lambda k: 'off' if C.get(f, k) is False else str(C.get(f, k))  # off/on схемы — bool (YAML 1.1)
    need = {'L'}
    if g('leg.panel') != 'off':
        need.add('P')
    if g('leg.refit') != 'yearly' or g('leg.rls_lambda') != '1' or g('leg.estimator') != 'log_ols_calib' or g('boost.boost_refit') != 'yearly':
        need.add('R')
    if g('boost.grid') == 'resid_small_v1':  # точки LHS — только BG (r2_roli.cmd_run такие дизайны пропускает, PR #185)
        need.add('BG')
    elif g('boost.role') != 'none' and g('boost.boost_refit') == 'yearly':  # boost_refit ≠ yearly — бустинг у R, итог — ряд R (PR #194)
        need.add('B')
    if any(g(k) not in ('off', 'False', 'direct', 'None') for k in ('mh.scheme', 'mh.persist', 'mh.slope', 'mh.reconcile')) \
            or g('leg.struct') != 'single':
        need.add('M')
    # дирижёр 26.10, 19:10 (находка O): L3 — ВСЕГДА, последним звеном, с calib дизайна (по умолчанию s2_est = sm, s2_embargo = on):
    # эталон ворот (ETA_b_d0_s2emb) — с поправкой s², кандидаты обязаны сравниваться так же (сырые ряды L/B — на ~15% ниже)
    need.add('L3')
    ens = f['ens']
    if any(str(ens[k]) not in ('False', 'off', 'mean', 'single', 'best_by_LOYO') for k in ens if k != 'seeds') or g('leg.moe') not in ('off', 'False'):
        need.add('E')
    if g('k16.gate') != 'off' or g('k18.lam_to_ref') != 'off' or g('k20.sign_delta') != 'off':  # schema_v2: кластеры 16, 18, 20
        need.add('G')
    return [m for m in CFG['chain_order'] if m in need]


class Runner:
    """Запуск модулей: боевой (подпроцесс по шаблону r2_S_driver.yaml) или дымовой (в процессе, HAR-J d0 r2_core)."""

    def __init__(self, smoke=False):
        self.smoke = smoke
        self.leak_done = {}

    def ready(self, m):
        f = HERE / CFG['modules'][m]['file']
        return f.exists() and "'run'" in f.read_text(encoding='utf-8', errors='ignore')

    def leak(self, m, dfile, h):
        k = (m, h)
        if k in self.leak_done:
            return self.leak_done[k]
        if self.smoke:
            ok = smoke_leak(h)
        else:
            cmd = [sys.executable, str(HERE / CFG['modules'][m]['file'])] + CFG['modules'][m]['leak'].format(designs=dfile, h=h).split()
            ok = subprocess.run(cmd, cwd=REPO).returncode == 0
        self.leak_done[k] = ok
        log(f'детектор {m} h={h}: {"OK" if ok else "ПРОВАЛ — этап блокируется"}')
        return ok

    def run(self, m, designs, h, seeds, steps, workers, extra='', tag=''):
        """-> (ok, причина). Дизайны пишутся во временный yaml этапа; модуль сам пропускает готовое (DONE_<модуль>)."""
        dfile = ydump(dict(designs=designs), STATE / ('smoke' if self.smoke else 'run') / f'{tag}_{m}_h{h}.yaml')
        if not self.smoke and not self.ready(m):
            return False, f'ждёт модуль {m} ({CFG["modules"][m]["file"]})'
        if not self.leak(m, dfile, h):
            return False, f'детектор {m} h={h} не пройден'
        if self.smoke:
            smoke_module(m, designs, h, seeds, steps, workers)
            return True, ''
        spec = CFG['modules'][m]
        if spec.get('per_design'):
            for i, d in enumerate(designs):
                f1 = ydump(d, dfile.with_name(f'{dfile.stem}_{i:03d}.yaml'))
                cmd = [sys.executable, str(HERE / spec['file'])] + spec['run'].format(design=f1, h=h).split()
                if subprocess.run(cmd, cwd=REPO).returncode != 0:
                    return False, f'модуль {m}: ошибка на дизайне {d.get("design")}'
            return True, ''
        faza = 'B' if tag in ('econ', 'B') or tag.startswith('dym') else tag[:1]  # B/эконометрика — план; C/D/E — ворота пред. фазы
        extra = f'{extra} {RUN_EXTRA.get(m, "")}'.strip()  # дым --real: --tickers для B, BG
        cmd = [sys.executable, str(HERE / spec['file'])] + spec['run'].format(
            designs=dfile, h=h, seeds=' '.join(map(str, seeds)), steps=' '.join(map(str, steps)), workers=workers, extra=extra,
            faza=faza).split()
        if m == 'L3':
            self.l3_stale(designs, h, seeds)
        if m == 'B' and '--row-frac' not in extra:
            self.snyat_podvyborku(designs, h)
        import r2_dataset_cache as DS
        try:  # дирижёр 18:40: < 2 ГБ на томе кэша — стоп до старта задач модуля (не 70 упавших задач с No space left)
            DS.proverka_mesta()
        except DS.MestaNet as e:
            log(f'модуль {m} h={h}: {e}')
            return False, f'модуль {m}: {e}'
        t0 = time.time()
        rc = subprocess.run(cmd, cwd=REPO).returncode
        net = self.done_missing(m, designs, h, seeds)
        self.last_net = net
        zamer(m, h, tag, len(designs), len(seeds), len(steps), workers, time.time() - t0, rc, len(net))
        if rc != 0 and designs and not net:  # все DONE на месте (повтор: «задач 0») — код ≠ 0 из-за пустой сводки модуля (E 18:36)
            log(f'модуль {m} h={h}: код {rc}, но все {len(designs) * len(seeds)} отметок DONE_{m} есть — этап пройден (❓ автору: код 0 при 0 задач)')
            return True, ''
        if rc != 0:
            return False, f'модуль {m}: код {rc}'
        if designs and len(net) == len(designs) * len(seeds):  # ни одной отметки DONE — модуль ничего не посчитал (P, M: код 0 и при ошибках)
            return False, f'модуль {m}: код 0, но нет ни одной отметки DONE_{m}_h{h} ({len(net)} задач)'
        if net:
            log(f'модуль {m} h={h}: без DONE {len(net)} из {len(designs) * len(seeds)} (неприменимые или ошибка задачи): {net[:3]}')
        return True, ''

    @staticmethod
    def snyat_podvyborku(designs, h):
        """B на всех строках: в каталоге дизайна не должно быть рядов скрининга на подвыборке (B_rf*, их __cal) и их DONE_B/DONE_L3 —
        до ключа boost.row_frac (19:5x) ячейка C с тем же хэшем, что прогон скрининга, брала его ряд на 50% строк. Файлы
        переименовываются в .stale_rf (не удаляются). -> [дизайн]."""
        out = []
        for d in designs:
            full = C.validate({**d, 'h': h})
            if float(C.get(full, 'boost.row_frac')) != 1:
                continue
            cd = C.cache_dir(full)
            rf = [f for f in cd.glob(f'B_rf*_h{int(h)}_*') if '.stale' not in f.name]
            if not rf:
                continue
            for f in rf + [f for f in cd.glob(f'DONE_B_h{int(h)}_*') if '.stale' not in f.name] + \
                    [f for f in cd.glob(f'DONE_L3_h{int(h)}_*') if '.stale' not in f.name]:
                os.replace(f, f.with_name(f'{f.name}.stale_rf'))
            out.append(full.get('design'))
        if out:
            log(f'B h={h}: сняты ряды подвыборки скрининга (B_rf*) у {len(out)} дизайнов на всех строках: {out[:3]}')
        return out

    @staticmethod
    def l3_stale(designs, h, seeds):
        """Перезапуск 19:10: L3 переехал в конец цепочки — у дизайнов, где раньше стоял до E/G, DONE_L3 относится к рядам другого
        звена. Отметка DONE_L3 без рядов <ряд предыдущего звена>__cal переименовывается в .stale (L3 пересчитает). -> [задача]."""
        I = _index(fresh=True)
        out = []
        for d in designs:
            full = C.validate({**d, 'h': h})
            ch = route(full)
            prev = ch[-2] if len(ch) > 1 else 'L'
            hid = C.design_hash(full)[:16]
            Ih = I[(I['hash'].astype(str) == hid) & (I['h'].astype(int) == int(h))]
            for s in seeds:
                dp = C.done_path(full, 'L3', h, s)
                if not dp.exists():
                    continue
                Is = Ih[Ih['seed'].astype(str).isin([str(s), 'det'])]
                need = {f'{r}__cal' for r in Is[Is['modul'] == prev]['ryad']}
                have = set(Is[Is['modul'] == 'L3']['ryad'])
                if need and not need <= have:
                    os.replace(dp, dp.with_name(f'{dp.name}.stale_{time.strftime("%H%M%S")}'))
                    out.append(f'{full.get("design")}/s{s}')
        if out:
            log(f'L3 h={h}: устаревших DONE_L3 (звено до L3 сменилось) — {len(out)}, пересчёт: {out[:3]}')
        return out

    @staticmethod
    def done_missing(m, designs, h, seeds):
        """Задачи (дизайн, seed) без отметки DONE_<m>_h{h}_s{seed} в кэше (корень — R2_CACHE)."""
        out = []
        for d in designs:
            full = C.validate({**d, 'h': h})
            out += [f'{full.get("design")}/s{s}' for s in seeds if not C.done_path(full, m, h, s).exists()]
        return out

    def chain(self, designs, h, seeds, steps, workers, tag, extra=None, partial=False):
        """Дизайны по цепочкам модулей: группами по модулю в порядке chain_order. partial=True (эконометрика): модуль, который не
        готов или упал, снимает только дизайны со своим звеном, остальные цепочки идут дальше (M, L3 не ждут R)."""
        by = {}
        for d in designs:
            for m in route(d):
                by.setdefault(m, []).append(d)
        snyaty, fails = set(), []
        for m in CFG['chain_order']:
            ds = [d for d in by.get(m, []) if d['design'] not in snyaty]
            if ds:
                self.last_net = []
                ok, why = self.run(m, ds, h, seeds, steps, workers, (extra or {}).get(m, ''), tag)
                if not ok:
                    if not partial:
                        return False, why
                    fails.append(why)
                    snyaty |= {d['design'] for d in ds}
                    continue
                # дирижёр 20:05: звено без DONE ни по одному seed — дизайн неприменим к модулю (R: RVI_anchor с refit/rls_lambda);
                # дальше по цепочке он не идёт (L3 не ищет ряды, которых нет), пометка — в NEPR и логе
                miss = {}
                for x in getattr(self, 'last_net', []) or []:
                    nm, _, s = x.rpartition('/s')
                    miss.setdefault(nm, set()).add(s)
                bez = {nm for nm, ss in miss.items() if len(ss) >= len(seeds)}
                for d in ds:
                    if d['design'] in bez:
                        snyaty.add(d['design'])
                        NEPR.setdefault((h, d['design']), []).append(f'неприменим к модулю {m} (нет DONE_{m})')
                if bez:
                    log(f'цепочка h={h}: {len(bez)} дизайнов без DONE_{m} — сняты с последующих звеньев: {sorted(bez)[:3]}')
        return not fails, '; '.join(fails)


_IDX = {}


def _index(root=None, fresh=False, ttl=60):
    """r2_core.read_index с памятью на ttl секунд (final_series зовётся сотни раз за ворота)."""
    k = str(root or C.CACHE_ROOT)
    t, I = _IDX.get(k, (0, None))
    if fresh or I is None or time.time() - t > ttl:
        I = C.read_index(root)
        _IDX[k] = (time.time(), I)
    return I


def final_series(design, h, seeds, root=None, steps=None):
    """Итоговый ряд дизайна: ряды последнего модуля цепочки; варианты (сетка внутри складки) — по имени ряда; seed — среднее
    (медиана при ens.agg = median). -> {вариант: кадр val}."""
    ch = route(design)
    last = ch[-1]
    assert last == 'L3', f'{design.get("design")}: итог не прошёл L3 (цепочка {ch}) — сравнение с эталоном s² невозможно'
    stand_in = SMOKE_FLAG['on'] and not SMOKE_FLAG.get('real')  # дым с заменой модулей HAR-J — ряды smoke_<модуль>
    L = C.load_series(C.design_hash(design), modul=f'smoke_{last}' if stand_in else last, seed=[str(s) for s in seeds],
                      root=root, steps=steps)
    if L.empty:
        return {}
    if not stand_in:
        assert L['ryad'].astype(str).str.endswith('__cal').all(), f'{design.get("design")}: ряды L3 без __cal'
        prev = ch[-2] if len(ch) > 1 else 'L'  # только калибровка рядов ТЕКУЩЕГО предыдущего звена (не устаревшие __cal прежней цепочки)
        I = _index(root=root)
        hid = C.design_hash(design)[:16]
        base = set(I[(I['hash'].astype(str) == hid) & (I['modul'] == prev)]['ryad'].astype(str))
        if base:
            L = L[L['ryad'].astype(str).str[:-len('__cal')].isin(base)]
        if len(ch) > 1 and ch[-2] == 'R':  # после R калибруется R_raw (R уже со своей поправкой — двойная калибровка), дирижёр 17:28
            L = L[L['ryad'].astype(str).str.startswith('R_raw')]
    agg = 'median' if str(C.get(C.validate(design), 'ens.agg')) == 'median' else 'mean'
    out = {}
    for v, g in L.groupby('ryad'):
        P = g.pivot_table(index=['date', 'ticker', 'year', 'step'], columns='seed', values='rv_pred')
        y = g.drop_duplicates(['date', 'ticker', 'step']).set_index(['date', 'ticker', 'year', 'step'])['rv_actual']
        out[v] = pd.DataFrame({'rv_pred': P.median(axis=1) if agg == 'median' else P.mean(axis=1), 'rv_actual': y}).reset_index()
    return out


# ================================================================ ворота (O): адаптеры
def etalon(h):
    """Эталон ворот — В2 с эмбарго s² (дирижёр 17:35: ETA_b_d0_s2emb, L PR #195; r2_S_driver.yaml etalon_vorot); нет файла — r2_gates.etalon_df."""
    import r2_gates as RG
    import r2_loyo as O
    f = C.REGEN / CFG['etalon_vorot']['etalon'].format(h=h) if CFG.get('etalon_vorot') else None
    if f is not None and f.exists():
        e = O.read_val(f)
        E = e[['date', 'ticker', 'year', 'step', 'rv_actual', 'rv_pred']]
    else:
        E = RG.etalon_df(h)
    if SMOKE_FLAG['on']:
        E = E[E.ticker.isin(SMOKE_TICKERS) & (E.step <= max(SMOKE_STEPS))]
    return E


def ocenka_units(units, h, seeds, root):
    """units — {имя: дизайн} -> ({имя: результат r2_gates.ocenka}, {имя: сложность}). В дыме — r2_gates.t_skladok на 2 складках."""
    import r2_gates as RG
    E = etalon(h)
    R, S = {}, {}
    for u, d in units.items():
        V = final_series(d, h, seeds, root)
        if not V:
            continue
        S[u] = C.complexity(d)
        if SMOKE_FLAG['on']:
            best = max(V, key=lambda v: RG.t_skladok(V[v], E, h, SMOKE_FOLDS)['t_mean'])
            r = RG.t_skladok(V[best], E, h, SMOKE_FOLDS)
            R[u] = dict(t_mean=r['t_mean'], vyigrysh_pct=r['vyigrysh_pct'], vhod=dict(proshyol=True, net=''), hit=np.nan,
                        n_tikerov_vyigrysh=np.nan, konc=np.nan,
                        moshchnost=dict(n_test=np.nan, d_mean=np.nan, sigma2_LR=np.nan, t_ozhid=np.nan, P_p_lt_005=np.nan, pometka='дым'))
        else:
            R[u] = RG.ocenka(V, E, h)
    return R, S


def write_gate(phase, h, payload, inputs=()):
    if SMOKE_FLAG['on']:
        g = dict(phase=phase, h=int(h), time=time.strftime('%F %T'), dym=True, **payload)
        return ydump(g, gate_path(phase, h, smoke=True))
    import r2_loyo as O
    return O.write_gate(phase, h, payload, inputs)


def gate_B(h, designs, root):
    """Фаза B: q-таблица прогонов из кэша -> r2_factor.effects (O) -> gate_B. В дыме и на h = 1 (2³) — выбор групп по t."""
    import r2_gates as RG
    units = {f'r{i + 1:02d}': d for i, d in enumerate(designs)}
    if not SMOKE_FLAG['on'] and h != 1:
        import r2_factor as RF
        rows = []
        E = etalon(h)
        for u, d in units.items():
            V = final_series(d, h, CFG['B']['seeds'], root)
            if V:
                rows.append(RG.tablica_c3(V, E, None).assign(unit=u))
        q = pd.concat(rows, ignore_index=True)
        f = C.REGEN / RF.FC['q_vhod'].format(h=h)
        f.parent.mkdir(parents=True, exist_ok=True)
        q.to_csv(f, index=False)
        RF.effects([h])
        return gate_path('B', h)
    R, S = ocenka_units(units, h, CFG['B']['seeds'], root)
    T = pd.DataFrame([dict(unit=u, t_mean=r['t_mean'], vyigrysh_pct=r['vyigrysh_pct']) for u, r in R.items()])
    if h == 1 and not SMOKE_FLAG['on']:
        best = T.sort_values('t_mean', ascending=False).unit.iloc[0]
        plan = yaml.safe_load(designs_file('B_h1').read_text())['tablica']
        x = plan[int(best[1:]) - 1]
        on = [f for f in CFG['B']['h1_factors'] if x[f] > 0]
    else:
        on = []
    return write_gate('B', h, dict(vklyucheny=on, tablica=T.to_dict('records'), N_B=len(units),
                                   pravilo='h = 5, 22 — r2_factor.effects (O); h = 1 — лучший прогон 2³ по t (❓ O)'))


def gate_C(h, designs, root):
    import r2_gates as RG
    units = {d['design']: d for d in designs}
    R, S = ocenka_units(units, h, CFG['C']['seeds'], root)
    # v4 § 4 — ворота O (PR #193): ячейки D, V0-финалисты, число дизайнов; payload — без изменений в write_gate
    return write_gate('C', h, RG.gate_C_v4(R, S, units, {u: NEPR[(h, u)] for u in units if (h, u) in NEPR}))


def _vybor_smoke(R, S):
    T = pd.DataFrame([dict(unit=u, t_mean=r['t_mean'], vyigrysh_pct=r['vyigrysh_pct'], slozhnost=S.get(u, 0), proshyol=True) for u, r in R.items()])
    T = T.sort_values(['t_mean', 'slozhnost'], ascending=[False, True])
    return T.unit.iloc[0], T


def point_design(base, tochka):
    """Дизайн точки LHS: boost.grid_point (schema_v2) — у каждой точки свой хэш и каталог кэша."""
    d = copy.deepcopy(base)
    d['design'] = f'{base.get("design")}_{tochka}'
    _set(d, 'boost.grid_point', tochka)
    return d


def gate_D(h, cells, runner, root, workers):
    """Ячейки C × LHS-60 -> r2_halving.otsechenie (O) с ряд-функцией драйвера -> 8 выживших -> r2_gates (вход, выбор)."""
    import r2_halving as RH
    import r2_gates as RG
    P = yaml.safe_load((DDIR / 'D_lhs60.yaml').read_text())['tochki']
    if SMOKE_FLAG['on']:
        P = P[:8]
        RH.stupeni = lambda: [(1, list(SMOKE_FOLDS), [42], 4), (2, list(SMOKE_FOLDS), [42], 2)]
    E = etalon(h)
    best_cells = {}
    for cname, cell in cells.items():
        base = copy.deepcopy(cell)
        _set(base, 'boost.grid', 'resid_small_v1')
        _set(base, 'boost.early_stop', 'qlike_total')

        def vybrat(V, gody):
            return list(V.values())[0] if len(V) == 1 else min(V.values(), key=lambda x: -RG.t_skladok(x, E, h, gody)['t_mean'])

        def ryad(tochka, seeds, gody, base=base):
            steps = list(range(1, max(gody) - 2015 + 1)) if not SMOKE_FLAG['on'] else list(SMOKE_STEPS)
            d = point_design(base, tochka)
            if any(runner.done_missing(m, [d], h, list(seeds)) for m in route(d)) or runner.smoke:  # готовое (предвыборка) — без запуска
                ok, why = runner.chain([d], h, list(seeds), steps, workers, f'D_{cname}_{tochka}', extra={'BG': f'--point {tochka}'})
                if not ok:
                    raise RuntimeError(why)
            return vybrat(final_series(d, h, seeds, root), gody)

        tochki = [p['tochka'] for p in P]
        if not runner.smoke:  # предвыборка пакетом (пул модуля), а не по точке: ступени 1–2 (seed 42) и запросы ступени 3
            s_all = list(range(1, max(C.DEV_YEARS) - 2015 + 1)) if not SMOKE_FLAG['on'] else list(SMOKE_STEPS)
            WD = CFG['workers'].get('D_pre', workers)
            ok, why = runner.chain([point_design(base, t) for t in tochki], h, [42], s_all, WD, f'D_{cname}_pre42')
            if not ok:
                raise RuntimeError(why)
            req = []

            def probe(t, seeds, gody, base=base):  # пробный проход по рядам s42: решения ступеней 1–2 те же, что в настоящем
                req.append((t, tuple(seeds)))
                return vybrat(final_series(point_design(base, t), h, [42], root), gody)
            RH.otsechenie(list(tochki), probe, E, h)
            nuzhno = {}
            for t, ss in req:
                if tuple(s for s in ss if s != 42):
                    nuzhno.setdefault(tuple(s for s in ss if s != 42), set()).add(t)
            for ss, ts in nuzhno.items():
                ok, why = runner.chain([point_design(base, t) for t in sorted(ts)], h, list(ss), s_all, WD, f'D_{cname}_pre')
                if not ok:
                    raise RuntimeError(why)
        zhivye, trace = RH.otsechenie(tochki, ryad, E, h)
        if not SMOKE_FLAG['on']:
            RH.zapisat(h, cname, zhivye, trace)
        else:
            trace.to_csv(SMOKE_O / f'halving_{cname}_h{h}.csv', index=False)
        best_cells[cname] = dict(zhivye=zhivye, base=base)
    units = {f'{c}_{t}': point_design(v['base'], t) for c, v in best_cells.items() for t in v['zhivye']}
    R, S = ocenka_units(units, h, CFG['D']['seeds'] if not SMOKE_FLAG['on'] else [42], root)
    zhivye = {c: [f'{c}_{t}' for t in v['zhivye']] for c, v in best_cells.items()}
    # v4 § 5 — ворота O (PR #193): 8 → 1 на ячейку, лучший уровня; число точек на входе — для числа дизайнов
    return write_gate('D', h, RG.gate_D_v4(R, S, units, zhivye, {c: len(P) for c in best_cells}))


def odin_ryad(V):
    """Итоговый ряд дизайна одним кадром: единственный вариант; при нескольких — первый по имени (сетку выбирает модуль)."""
    return V[sorted(V)[0]] if V else None


def gate_E(h, units, root, v0_units, runner=None, steps=None, workers=1):
    """v4 § 2, § 6 — ворота O (PR #193): ocenka на 10 seed (вход заново) -> 2σ r2_gates.sigma2_F парно с F_h(d0) (V0 — один ряд) ->
    r2_gates.gate_E_v4 (финалисты, победитель, мощность, itog_E по порогам владельца 16:45, нога _noga, лестница и абляции).
    Правило «остаётся нога» при P < 0,5 (дирижёр 17:10): V0 с той же ногой, что у победителя (нога + калибровка). Нет такого ряда —
    драйвер ставит «нога + калибровка» победителя в очередь (L + L3, run_phase) и только при невозможности оставляет F_h(d0)."""
    import r2_gates as RG
    seeds = CFG['E']['seeds_final'] if not SMOKE_FLAG['on'] else [42]
    R, S = ocenka_units(units, h, seeds, root)
    sig = {}
    Fe = None  # F_h(d0) с эмбарго s² (дирижёр 17:35: F_d0_s2emb); нет файлов — ряды r2_O_loyo.yaml F_h_d0
    if not SMOKE_FLAG['on'] and CFG.get('etalon_vorot'):
        import r2_loyo as O
        fs = {s: C.REGEN / CFG['etalon_vorot']['F'].format(h=h, seed=s) for s in seeds}
        Fe = {s: O.series([f]) for s, f in fs.items()} if all(f.exists() for f in fs.values()) else None
    for u in R:
        if SMOKE_FLAG['on']:  # дым: 3 тикера × 1 seed — 2σ не считается; механика: первый по t «прошёл»
            continue
        if u in v0_units:
            sig[u] = RG.sigma2_F(odin_ryad(final_series(units[u], h, [42], root)), h, seeds=seeds, F=Fe)
        else:
            sig[u] = RG.sigma2_F({s: odin_ryad(final_series(units[u], h, [s], root)) for s in seeds}, h, seeds=seeds, F=Fe)
    if SMOKE_FLAG['on'] and R:
        top = max((u for u in R if u not in v0_units), key=lambda u: R[u]['t_mean'], default=None)
        sig = {u: dict(passed=(u == top), d_mean=np.nan, sigma=np.nan, thr=np.nan, pometka='дым: 2σ не считается') for u in R}
    g = RG.gate_E_v4(R, S, units, sig, h, list(v0_units), lestnica=not SMOKE_FLAG['on'],
                     out_dir=(SMOKE_O / f'gate_E_h{h}_dizainy') if SMOKE_FLAG['on'] else None)
    pb = g['pobeditel']
    it = pb.get('itog') or {}
    noga = pb.get('noga') or {}
    if it.get('deystvie') == 'final_V0' and (not noga.get('unit') or noga.get('nuzhen_schet')):
        nd, why = noga_v_ochered(units.get(pb['pobeditel_otbora']), noga, h, runner, steps, workers, root)
        if nd is not None:
            pb.update(pobeditel=nd['design'], V0=True)
            g.update(design_sha=C.design_hash(nd), dizayn=nd)
            noga.update(unit=nd['design'], design_sha=C.design_hash(nd), nuzhen_schet=False, poschitan=True,
                        istochnik=f'нога + калибровка победителя ({">".join(route(nd))}) — посчитана драйвером; у O: {noga.get("istochnik", "")}')
            it['final'] = nd['design']
        else:
            pb['pobeditel'], it['final'] = 'F_h(d0)', 'F_h(d0)'
            noga['pometka'] = f'нога + калибровка не посчитана ({why}) — на h остаётся F_h(d0)'
        pb['noga'], pb['itog'] = noga, it
    return write_gate('E', h, g)


def noga_dizain(pob_design, noga):
    """«Нога + калибровка» победителя: дизайн из ступени лестницы O (design_yaml), иначе leg.* + calib.* победителя, роль none."""
    if noga.get('design_yaml') and Path(noga['design_yaml']).exists():
        d = yaml.safe_load(Path(noga['design_yaml']).read_text())
    else:
        d = {'h': pob_design['h'], 'leg': copy.deepcopy(pob_design.get('leg') or {}), 'calib': copy.deepcopy(pob_design.get('calib') or {})}
        for k in ('inputs',):
            if k in pob_design:
                d[k] = copy.deepcopy(pob_design[k])
    d['design'] = f'noga_{pob_design.get("design", "pob")}'
    d.setdefault('boost', {})['role'] = 'none'
    C.validate(d)
    return d


def noga_v_ochered(pob_design, noga, h, runner, steps, workers, root):
    """Очередь расчёта ноги (L + L3 по маршруту дизайна, seed 42, шаги конвейера). -> (дизайн | None, причина)."""
    if pob_design is None or runner is None:
        return None, 'нет дизайна победителя или исполнителя'
    try:
        nd = noga_dizain(pob_design, noga)
    except Exception as e:  # noqa: BLE001 — схема не приняла: F_h(d0)
        return None, f'дизайн ноги не прошёл схему: {e}'
    ok, why = runner.chain([nd], h, [42], steps, workers, 'Enoga')
    V = final_series(nd, h, [42], root) if ok else {}
    log(f'ворота E h={h}: нога {nd["design"]} ({">".join(route(nd))}) — {"посчитана" if V else why or "нет ряда"}')
    return (nd, '') if V else (None, why or 'нет итогового ряда')


# ================================================================ отчёт владельцу
def report(phase, hs, smoke=False):
    """R2/report/gate_{X}.md — таблица по h из файлов ворот (без вопросов)."""
    out = (SMOKE_REPORT if smoke else REPORT) / f'gate_{phase}.md'
    out.parent.mkdir(parents=True, exist_ok=True)
    L = [f'# Ворота {phase} — {time.strftime("%F %T")}{" (дым, 3 тикера × шаги 1–3)" if smoke else ""}', '']
    if REPORT_NOTE.get(phase):
        L += [f'**{REPORT_NOTE[phase]}**', '']
    for h in hs:
        f = gate_path(phase, h, smoke)
        if not f.exists():
            L += [f'## h = {h}', '', 'ворота ещё не решены', '']
            continue
        g = yaml.safe_load(f.read_text())
        L += [f'## h = {h}', '']
        for k, v in g.items():
            if k in ('tablica', 'dizainy', 'dizayn', 'faktory', 'inputs', 'sigma2', 'moshchnost', 'pobeditel', 'finalisty', 'lestnica',
                     'ablyacii', 'lozhnye', 'chislo'):
                continue
            L.append(f'- **{k}**: {v}')
        if phase == 'E':
            pb = g.get('pobeditel') if isinstance(g.get('pobeditel'), dict) else {}
            it, noga = pb.get('itog') or {}, pb.get('noga') or {}
            fmt = lambda x: '—' if x is None or x != x else f'{x:.2f}'
            L += ['', '**Мощность (кластер 20; пороги владельца 16:45)**: P ≥ 0,8 — «значимый»; 0,5 ≤ P < 0,8 — «не хуже, ожидание на '
                  'границе»; P < 0,5 — на h остаётся нога: V0 с той же ногой, что у победителя (нога + калибровка); нет такого ряда — '
                  'драйвер считает его (L + L3); не удалось — F_h(d0).', '',
                  '| победитель отбора | E[t] | P(p < 0,05) | категория | гипотеза | финал на открытие |', '|---|---|---|---|---|---|',
                  '| ' + ' | '.join([str(pb.get('pobeditel_otbora', '—')), fmt(it.get('t_ozhid')), fmt(it.get('P_p_lt_005')),
                                    str(it.get('pometka', '—')), str(it.get('gipoteza', '—')), str(pb.get('pobeditel', '—'))]) + ' |']
            if noga:
                L += ['', f'- **нога**: {noga.get("unit")} — {noga.get("istochnik", "")}'
                      + (f'; {noga["pometka"]}' if noga.get('pometka') else '') + ('; посчитана драйвером' if noga.get('poschitan') else '')]
            if g.get('finalisty'):
                T = pd.DataFrame(g['finalisty'])
                cols = [c for c in ('unit', 'tip', 'poryadok', 't_mean', 'vyigrysh_pct', 'vhod', 'P_moshchnost', 's2_passed') if c in T]
                L += ['', '**финалисты**', '', '| ' + ' | '.join(cols) + ' |', '|' + '---|' * len(cols)]
                L += ['| ' + ' | '.join(f'{x:.4g}' if isinstance(x, float) else str(x) for x in r) + ' |' for r in T[cols].values]
        if phase == 'B' and g.get('faktory'):
            fe = faktory_na_E(g)
            L.append(f'- **на уровень E** (владелец 18:35: e_bar ≤ {(g.get("porog") or {}).get("e_bar_max", -0.005)}, полоса через 0; '
                     f'вариант поверх финалиста h): {fe or "нет"}')
        for k in ('faktory', 'tablica'):
            if g.get(k):
                T = pd.DataFrame(g[k])
                if phase == 'B' and k == 'faktory' and 'faktor' in T:
                    T['na_uroven_E'] = T.faktor.isin(faktory_na_E(g)).map({True: 'да', False: ''})
                L += ['', f'**{k}**', '', '| ' + ' | '.join(T.columns) + ' |', '|' + '---|' * len(T.columns)]
                for _, r in T.head(40).iterrows():
                    L.append('| ' + ' | '.join(f'{x:.4g}' if isinstance(x, float) else str(x) for x in r.values) + ' |')
        if g.get('sigma2'):
            L += ['', '**2σ (Д24.3) против F_h(d0)**', '']
            for u, s in g['sigma2'].items():
                L.append(f'- {u}: d̄ = {s["d_mean"]:.4g}, σ = {s["sigma"]:.3g}, порог {s["thr"]:.3g}, прошёл {s["passed"]}')
        L.append('')
    out.write_text('\n'.join(L), encoding='utf-8')
    return out


# ================================================================ конвейер h
def save_state(h, st, smoke):
    ydump(st, (STATE / ('smoke' if smoke else 'run')) / f'state_h{h}.json')


def wait_for(cond, what, st, h, smoke, once):
    while True:
        r = cond()
        if r:
            return r
        st['zhdyot'] = what
        save_state(h, st, smoke)
        if once or smoke:
            return None
        time.sleep(CFG['poll_s'])


def pipeline(h, runner, smoke=False, once=False):
    """B -> gate_B -> C -> gate_C -> D -> gate_D -> E -> gate_E -> стоп перед открытием. Без ворот этап ждёт (опрос)."""
    st = dict(h=h, start=time.strftime('%F %T'), etapy={})
    root = SMOKE_ROOT if smoke else None
    steps = list(SMOKE_STEPS) if smoke else CFG['steps']
    W = CFG['workers']['main']

    def stage(name, fn):
        t0 = time.strftime('%F %T')
        r = fn()
        ok_ = r[0] if isinstance(r, tuple) else (r is not None and r is not False)  # chain -> (ok, причина)
        st['etapy'][name] = dict(start=t0, end=time.strftime('%F %T'), ok=bool(ok_))
        save_state(h, st, smoke)
        return r

    def chain(*a, **kw):  # нет модуля другого агента -> ждать (опрос poll_s), кроме дыма и --once
        while True:
            ok, why = runner.chain(*a, **kw)
            if ok or smoke or once or 'ждёт модуль' not in why:
                return ok, why
            st['zhdyot'] = why
            save_state(h, st, smoke)
            time.sleep(CFG['poll_s'])

    # B
    B = read_designs(f'B_h{h}')
    if smoke and SMOKE_FLAG.get('real'):
        B = B[:4]  # дым с модулями: замер на 4 прогонах плана на h
    ok, why = stage('B_schet', lambda: chain(B, h, CFG['B']['seeds'], steps, W, f'B', extra={'B': f'--row-frac {CFG["B"]["row_frac"]} --row-seed {CFG["B"]["row_seed"]}'}))
    if not ok:
        st['zhdyot'] = why
        save_state(h, st, smoke)
        return st
    if not gate_path('B', h, smoke).exists():
        stage('gate_B', lambda: gate_B(h, B, root))
    report('B', CFG['hs'] if not smoke else SMOKE_HS, smoke)
    gB = yaml.safe_load(gate_path('B', h, smoke).read_text())
    # C
    Cd = apply_gate_B(read_designs(f'C_h{h}_shablon'), gB, h)
    if smoke:
        Cd = Cd[:4] + [d for d in Cd if d['design'].endswith('_V0')]
    ok, why = stage('C_schet', lambda: chain(Cd, h, CFG['C']['seeds'], steps, W, 'C'))
    if not ok:
        st['zhdyot'] = why
        save_state(h, st, smoke)
        return st
    if not gate_path('C', h, smoke).exists():
        stage('gate_C', lambda: gate_C(h, Cd, root))
    report('C', CFG['hs'] if not smoke else SMOKE_HS, smoke)
    gC = yaml.safe_load(gate_path('C', h, smoke).read_text())
    # D
    cells = {u: gC['dizainy'][u] for u in gC['yacheyki']}
    if not gate_path('D', h, smoke).exists():
        stage('D_halving_gate_D', lambda: gate_D(h, cells, runner, root, W))
    report('D', CFG['hs'] if not smoke else SMOKE_HS, smoke)
    gD = yaml.safe_load(gate_path('D', h, smoke).read_text())
    # E
    best = gD.get('dizayn')
    if best is None:
        st['zhdyot'] = 'gate_D без лучшего дизайна — на h остаётся F_h(d0)'
        save_state(h, st, smoke)
        return st
    Ed = designs_E(best, CFG['E']['ofat'])
    if smoke:
        Ed = Ed[:4]
    # решение владельца 18:35: околопороговые факторы B (e_bar ≤ −0,005, полоса через 0) — явные варианты поверх лучшего h
    fE = faktory_na_E(gB)
    Ef = designs_faktory_E(best, fE, h)
    st['faktory_na_E'] = dict(faktory=fE, variantov=len(Ef))
    Ed = Ed + [d for d in Ef if d['design'] not in {x['design'] for x in Ed}]
    ok, why = stage('E_schet', lambda: chain(Ed, h, [42], steps, W, 'E', partial=True))  # надстройки независимы (OFAT)
    if not ok:
        st['E_schet_oshibki'] = why  # упавшие надстройки — без ряда, в ворота E не попадут (ocenka_units пропускает)
        log(f'конвейер h={h}: E — часть надстроек без ряда: {why[:300]}')
    v0 = {u: gC['dizainy'][u] for u in gC.get('v0_finalisty', [])}
    units = {d['design']: d for d in Ed} | {gD['luchshiy']: best} | v0
    if not smoke:  # финалисты: 10 seed, LightGBM + XGBoost — после выбора по 1 seed; ворота E — на 10 seed
        import r2_gates as RG
        R, S = ocenka_units(units, h, [42], root)
        _, T = RG.vybor(R, S)
        top = list(T[T.proshyol].sort_values('t_mean', ascending=False).unit.head(CFG['E']['finalists_k']))
        fin = {}
        for u in top:
            d = copy.deepcopy(units[u])
            _set(d, 'boost.lib', CFG['E']['lib_final'])
            _set(d, 'ens.seeds', 10)
            fin[u] = d
        ok, why = stage('E_finalisty_10seed', lambda: chain(list(fin.values()), h, CFG['E']['seeds_final'], steps, W, 'Efin'))
        if not ok:
            st['zhdyot'] = f'финалисты 10 seed: {why}'
            save_state(h, st, smoke)
            return st
        units = fin | v0
    stage('gate_E', lambda: gate_E(h, units, root, list(v0), runner, steps, W))
    report('E', CFG['hs'] if not smoke else SMOKE_HS, smoke)
    if not smoke:  # дирижёр 17:50: boost_refit = monthly — НЕ в финалистах и не в воротах; только seed 42, для приложения, последним
        prilozhenie_monthly(h, fin, runner, steps, W, root, st, chain, stage)
    st['zhdyot'] = 'СТОП: перед открытием теста (фаза F — агент A, «да» владельца)'
    save_state(h, st, smoke)
    return st


def prilozhenie_monthly(h, fin, runner, steps, W, root, st, chain, stage):
    """Приложение (дирижёр 17:50): вариант каждого финалиста E с boost.boost_refit = monthly, seed 42, после ворот E (последним).
    В ворота и выбор не входит; таблица — R2/report/prilozhenie_monthly_h{h}.md."""
    ml = CFG['E'].get('monthly_last')
    if not ml or not fin:
        return
    mon = {}
    for u, d in fin.items():
        dm = copy.deepcopy(d)
        _set(dm, ml['key'], ml['value'])
        dm['design'] = f'{d.get("design", u)}_{ml["key"]}={ml["value"]}'
        try:
            C.validate(dm)
            mon[u] = _maybe_inputs(dm)
        except C.DesignError as e:
            log(f'конвейер h={h}: {ml["value"]} для {u} не принят схемой: {e}')
    ok, why = stage('E_prilozhenie_monthly', lambda: chain(list(mon.values()), h, ml.get('seeds', [42]), steps, W, 'Emon', partial=True))
    rows = [f'| {u} | {C.design_hash(d)[:16]} | {"есть" if final_series(d, h, ml.get("seeds", [42]), root) else "нет"} |' for u, d in mon.items()]
    REPORT.mkdir(parents=True, exist_ok=True)
    (REPORT / f'prilozhenie_monthly_h{h}.md').write_text('\n'.join(
        [f'# Приложение: boost_refit = monthly, h = {h} ({time.strftime("%F %T")})', '',
         'Только seed 42, после ворот E; в выбор и ворота не входит (дирижёр 26.10, 17:50).', '',
         '| финалист | хэш варианта | ряд |', '|---|---|---|'] + rows + ([f'', f'ошибки: {why}'] if not ok else [])), encoding='utf-8')
    st['prilozhenie_monthly'] = dict(ok=ok, prichina=why, n=len(mon))


def econ(h, runner, smoke=False, once=False):
    """Эконометрика сразу (кластеры 2, 3, 6, 14, 15), не ждёт бустинга. Цепочки независимы (partial): нет модуля R — ждут только
    дизайны с R; в боевом режиме (не once) повтор каждые poll_s, пока ждёт модуль (готовое модули пропускают по DONE)."""
    t0 = time.strftime('%F %T')
    ds = read_designs(f'econ_h{h}')
    if smoke:
        ds = ds[:6] + [d for d in ds if 'mh.persist' in d['design']][:2] + \
            [d for d in ds if {'P', 'L3'} & set(route(d)) and 'R' not in route(d)][:2] + \
            [d for d in ds if 'R' in route(d) and 'monthly' not in d['design']][:2]
    while True:
        ok, why = runner.chain(ds, h, [42], list(SMOKE_STEPS) if smoke else CFG['steps'], CFG['workers']['econ'], 'econ', partial=True)
        if ok or smoke or once or 'ждёт модуль' not in why:
            break
        ydump(dict(h=h, start=t0, zhdyot=why), STATE / 'run' / f'econ_h{h}.json')
        time.sleep(CFG['poll_s'])
    r = dict(h=h, start=t0, end=time.strftime('%F %T'), ok=ok, prichina=why, dizaynov=len(ds))
    ydump(r, STATE / ('smoke' if smoke else 'run') / f'econ_h{h}.json')
    return r


def stop_note(states, smoke):
    f = (SMOKE_REPORT if smoke else REPORT) / 'STOP_pered_otkrytiem.md'
    L = ['# Остановка перед открытием теста', '', f'{time.strftime("%F %T")}. Драйвер дошёл до ворот E и остановился. Дальше — фаза F (агент A: '
         'сборка тестовых прогнозов финалистов, enable_test_assembly), FINAL_DESIGN_round2 и «да» владельца.', '']
    for s in states:
        L.append(f'- h = {s["h"]}: {s.get("zhdyot")}')
    f.write_text('\n'.join(L), encoding='utf-8')
    return f


REPORT_NOTE = {}  # фаза -> пометка в отчёте (дирижёр 19:10: «пересчитано после исправления калибровки, 19:xx»)


def arhiv_vorot(suffix, hs, smoke=False):
    """Перед пересчётом ворот: gate_{B,C,D,E}_h*.yaml, каталоги gate_E_h*_dizainy, R2/O/factor/*.csv, halving_*.csv и отчёты
    R2/report/*.md переименовываются с суффиксом (os.replace, не удаляются); при совпадении имени — ещё и время. -> [(было, стало)]."""
    O_ = SMOKE_O if smoke else C.O_DIR
    rep = SMOKE_REPORT if smoke else REPORT
    pats = [O_.glob(f'gate_{p}_h{h}.yaml') for p in 'BCDE' for h in hs] + [O_.glob(f'gate_E_h{h}_dizainy') for h in hs] + \
        [(O_ / 'factor').glob('*.csv'), O_.glob('halving_*.csv'), rep.glob('*.md')]
    moved = []
    for g in pats:
        for f in list(g):
            if suffix in f.name:
                continue
            t = f.with_name(f'{f.stem}{suffix}{f.suffix}')
            if t.exists():
                t = f.with_name(f'{f.stem}{suffix}_{time.strftime("%H%M%S")}{f.suffix}')
            os.replace(f, t)
            moved.append((str(f), str(t)))
    log(f'архив ворот ({suffix}): {len(moved)} файлов/каталогов')
    return moved


def cmd_run(a, smoke=False):
    if getattr(a, 'arhiv_vorot', None):
        arhiv_vorot(a.arhiv_vorot, a.h or (SMOKE_HS if smoke else CFG['hs']), smoke)
        for p in 'BC':
            REPORT_NOTE[p] = f'пересчитано после исправления калибровки (L3 последним звеном, s² дизайна), {time.strftime("%H:%M")}; ' \
                             f'прежние ворота — *{a.arhiv_vorot}'
    if not smoke:
        import r2_dataset_cache as DS
        log(f'место на томах кэша (ГБ): {DS.proverka_mesta()}')  # < 2 ГБ — MestaNet, драйвер не стартует
    if not (DDIR / 'B_h5.yaml').exists():
        cmd_generate()
    runner = Runner(smoke)
    hs = a.h or (SMOKE_HS if smoke else CFG['hs'])
    res, econ_res = {}, {}
    th = []
    for h in hs:  # эконометрика — сразу, параллельно с B
        t = threading.Thread(target=lambda h=h: econ_res.__setitem__(h, econ(h, runner, smoke, getattr(a, 'once', False))), name=f'econ_h{h}')
        t.start()
        th.append(t)
    def guarded(h):
        try:
            res[h] = pipeline(h, runner, smoke, getattr(a, 'once', False))
        except Exception as e:  # noqa: BLE001 — ошибка конвейера h не роняет другие h; причина — в состоянии и отчёте
            import traceback
            res[h] = dict(h=h, zhdyot=f'ОШИБКА: {type(e).__name__}: {e}', trace=traceback.format_exc())
            save_state(h, res[h], smoke)
            log(f'конвейер h={h}: ошибка {e}')

    for h in hs:  # конвейеры h — параллельно
        t = threading.Thread(target=guarded, args=(h,), name=f'h{h}')
        t.start()
        th.append(t)
    for t in th:
        t.join()
    f = stop_note([res[h] for h in hs if h in res], smoke)
    log(f'драйвер: {[(h, res.get(h, {}).get("zhdyot")) for h in hs]}; эконометрика {[(h, econ_res.get(h, {}).get("ok")) for h in hs]} -> {f}')
    return res, econ_res


# ================================================================ дым
SMOKE_TICKERS = ('SBER', 'LKOH', 'AFLT')
SMOKE_STEPS = (1, 2, 3)
SMOKE_FOLDS = (2017, 2018)
SMOKE_HS = [5, 22, 1]
SMOKE_DIR = C.S_DIR / 'driver_smoke'
SMOKE_ROOT = SMOKE_DIR / 'cache'
SMOKE_O = SMOKE_DIR / 'O'
SMOKE_REPORT = SMOKE_DIR / 'report'
SMOKE_FLAG = {'on': False}
_SM = {}


def _smoke_data():
    if 'A' not in _SM:
        A, harj = C.load_d0(SMOKE_TICKERS)
        _SM.update(A=A, harj=harj)
    return _SM['A'], _SM['harj']


def _smoke_fn(design, seed, steps):
    """Дымовой модуль: HAR-J d0 + эмбарго (r2_core) — механика драйвера, не модель."""
    return {f'smoke_{C.SHARED["_smoke_m"]}': C._smoke_series(C.SHARED['A'], C.SHARED['harj'], int(design['h']), steps)}


def smoke_module(m, designs, h, seeds, steps, workers):
    A, harj = _smoke_data()
    C.share(A=A, harj=harj)
    with LOCK:  # run_phase держит fn в общем SHARED: дымовые вызовы из потоков — по одному
        C.share(_smoke_m=m)  # у каждого модуля цепочки — свой ряд (иначе модули затирают файл друг друга)
        C.run_phase('S', h, designs, _smoke_fn, seeds=tuple(seeds), steps=tuple(steps), workers=1, module=f'smoke_{m}',
                    root=SMOKE_ROOT, block_file=SMOKE_DIR / 'blok_sdvig.csv', check_gate=False)
    STATE.joinpath('smoke').mkdir(parents=True, exist_ok=True)
    with open(STATE / 'smoke' / 'zapuski.csv', 'a') as f:
        f.write(f'{time.strftime("%F %T")},{m},{h},{len(designs)},{threading.current_thread().name}\n')


def smoke_leak(h):
    A, harj = _smoke_data()
    C.share(A=A, harj=harj)
    R = C.leak_check(C._smoke_build(True, h), A, f'driver_smoke_h{h}', cuts=('2017-06-30', '2017-12-31'))
    return bool(R.ok.all())


def cmd_smoke(a):
    import shutil  # noqa: F401  (не используется для удаления — каталог дыма чистится по файлам)
    t0 = time.time()
    SMOKE_FLAG['on'] = True
    for d in (SMOKE_O, SMOKE_REPORT, STATE / 'smoke'):
        d.mkdir(parents=True, exist_ok=True)
        for f in d.glob('*'):
            if f.is_file():
                f.unlink()
    for f in SMOKE_ROOT.glob('*/DONE_*'):
        f.unlink()
    rows = []
    ok = lambda name, cond, det='': rows.append(dict(proverka=name, ok=bool(cond), detal=det))
    cmd_generate()
    n = {f.stem: yaml.safe_load(f.read_text()).get('n') for f in sorted(DDIR.glob('*.yaml'))}
    ok('generate: все yaml B–E прошли схему', all(v is None or v > 0 for v in n.values()), json.dumps(n, ensure_ascii=False))
    Bh5 = yaml.safe_load(designs_file('B_h5').read_text())
    hs_ = [r['design_sha'] for r in Bh5['index']]
    ok('B h=5: 64 дизайна, хэши уникальны; persist/reconcile — с входами h=1 в хэше',
       len(set(hs_)) == 64 and any('inputs' in d for d in Bh5['designs']))
    t1 = time.time()
    res, econ_res = cmd_run(argparse.Namespace(h=None, once=False), smoke=True)
    Z = pd.read_csv(STATE / 'smoke' / 'zapuski.csv', names=['t', 'm', 'h', 'n', 'thread'])
    ok('конвейеры h = 5, 22, 1 дошли до остановки перед открытием', all('СТОП' in str(res.get(h, {}).get('zhdyot')) for h in SMOKE_HS),
       str({h: res.get(h, {}).get('zhdyot') for h in SMOKE_HS}))
    b_end = [res.get(h, {}).get('etapy', {}).get('B_schet', {}).get('end') for h in SMOKE_HS]
    ok('эконометрика стартовала сразу, параллельно с B', all(econ_res.get(h, {}).get('ok') for h in SMOKE_HS) and all(b_end) and
       min(pd.to_datetime([econ_res[h]['start'] for h in SMOKE_HS])) <= pd.to_datetime(min(b_end)),
       f'потоки: {sorted(Z.thread.unique())}')
    ok('ворота B, C, D, E на всех h; отчёты gate_{B,C,D,E}.md', all(gate_path(p, h, True).exists() for p in 'BCDE' for h in SMOKE_HS)
       and all((SMOKE_REPORT / f'gate_{p}.md').exists() for p in 'BCDE'))
    ok('D: отсечение r2_halving (O) на дымовых ступенях; трасса', all((SMOKE_O / f'halving_{c}_h{h}.csv').exists()
                                                                     for h in SMOKE_HS for c in yaml.safe_load(gate_path('C', h, True).read_text())['yacheyki']))
    # правило дирижёра: фактор B, неприменимый к модулю цепочки (yaml модуля), у ячейки остаётся off, ячейка не выбрасывается, пометка
    cell = dict(design='C_h5_panel_residual', h=5, leg=dict(type='HARJ', panel='factor'), boost=dict(role='residual', early_stop='qlike_total'),
                features=['G0', 'G1'])
    out = apply_gate_B([cell], dict(vklyucheny=['C', 'E', 'K', 'A']), 5)
    f0 = C.validate(out[0])
    ok('неприменимый фактор (P): C, E — off с пометкой, K (G5+G6) — off, A — включён; ячейка осталась',
       len(out) == 1 and str(C.get(f0, 'boost.resid_source')) == 'in_sample' and C.get(f0, 'boost.clip_sigma') is False
       and 'G5' not in f0['features'] and str(C.get(f0, 'data.recency_halflife')) == '4y' and len(NEPR.get((5, cell['design']), [])) == 3,
       str(NEPR.get((5, cell['design']))))
    ok('проверка сдвига: блоков нет (HAR-J d0)', not (SMOKE_DIR / 'blok_sdvig.csv').exists() or
       not (pd.read_csv(SMOKE_DIR / 'blok_sdvig.csv').ok == False).any())  # noqa: E712 — «плоский профиль» пишется с ok = True
    ok('режим R2: запись ≥ 2022 отклоняется', C._raises(lambda: C.check_no_future(pd.DataFrame({'date': ['2022-01-03'], 'year': [2022]}), 'driver_smoke')))
    gE = [yaml.safe_load(gate_path('E', h, True).read_text()) for h in SMOKE_HS]
    ok('ворота C/D/E — r2_gates.gate_C_v4 / gate_D_v4 / gate_E_v4 (O, PR #193): payload v4',
       all('chislo' in yaml.safe_load(gate_path(p, h, True).read_text()) for p in 'CD' for h in SMOKE_HS)
       and all('lozhnye' in g and isinstance(g.get('pobeditel'), dict) for g in gE))
    ok('ворота E: итог мощности (пороги владельца 16:45, itog_E) в gate_E и gate_E.md; P < 0,5 без V0 той же ноги — нога + калибровка '
       'посчитана драйвером (L + L3) или F_h(d0) с пометкой',
       all(g['pobeditel'].get('itog') for g in gE) and 'пороги владельца 16:45' in (SMOKE_REPORT / 'gate_E.md').read_text(encoding='utf-8')
       and all(g['pobeditel']['pobeditel'] == 'F_h(d0)' or g['pobeditel'].get('V0') or g['pobeditel']['itog'].get('deystvie') == 'na_otkrytie'
               for g in gE),
       str({h: (g['pobeditel']['pobeditel'], (g['pobeditel'].get('noga') or {}).get('istochnik')) for h, g in zip(SMOKE_HS, gE)}))
    import r2_gates as RG
    kat = {p: RG.kategoriya_moshchnosti(p)['kategoriya'] for p in (0.85, 0.8, 0.6, 0.5, 0.3)}
    ok(f'категории мощности O: {kat}', kat == {0.85: 'znachimyy', 0.8: 'znachimyy', 0.6: 'granica', 0.5: 'granica', 0.3: 'noga'})
    ok('сдвиг: симметричный ε = 0,01 через r2_loyo.sdvig_ok (MIDAS h=22: 3 тик. 0,0113 — блок, 17 тик. 0,0082 — проходит)',
       C.SDVIG_EPS == 0.01 and not _ok_midas(3) and _ok_midas(17))
    dz = C.validate(dict(design='smoke_params', h=5, leg=dict(type='HARJ')))
    for m_ in ('L', 'M'):
        C.params_path(dz, m_, root=SMOKE_ROOT).write_text(f'step,par\n1,{m_}\n', encoding='utf-8')
    Pm = C.load_params(dz, root=SMOKE_ROOT)
    ok('параметры: params_<модуль>.csv на модуль, load_params читает все (общий params.csv не используется)',
       sorted(Pm.modul.unique()) == ['L', 'M'] and len(C.load_params(dz, modul='M', root=SMOKE_ROOT)) == 1,
       f'общий params.csv ещё в коде: {params_obshchiy() or "нет"}')
    allD = [d for f in sorted(DDIR.glob('*.yaml')) for d in (yaml.safe_load(f.read_text()).get('designs') or []) if isinstance(d, dict)]
    ok(f'L3 — последнее звено у всех {len(allD)} дизайнов driver_v1 (калибровка s² дизайна, дирижёр 19:10)',
       all(route(d)[-1] == 'L3' for d in allD))
    mv = arhiv_vorot('_bez_s2', SMOKE_HS, smoke=True)
    ok(f'архив ворот перед пересчётом: {len(mv)} переименовано (_bez_s2), исходных gate_* нет',
       len(mv) >= 12 and not any(gate_path(p, h, True).exists() for p in 'BCDE' for h in SMOKE_HS)
       and all(Path(t).exists() for _, t in mv))
    import r2_dataset_cache as DS

    def _mesta_stop():
        try:
            DS.proverka_mesta(min_gb=10 ** 6)
            return False
        except DS.MestaNet:
            return True
    ok(f'место: кэш наборов по умолчанию — SSD ({DS.cache_root() if not os.environ.get("R2_DS_CACHE") else "R2_DS_CACHE"}); '
       f'< порога — MestaNet до старта задач', _mesta_stop() and bool(DS.proverka_mesta(min_gb=0))
       and (os.environ.get('R2_DS_CACHE') or str(DS.cache_root()) == str(DS.SSD_DEFAULT)))
    gB22 = dict(porog=dict(e_bar_max=-0.005), faktory=[  # таблица ворот B h=22 боевого счёта 18:18 (O): решение владельца 18:35
        dict(faktor=f, e_bar=e, ci_low=lo, ci_high=hi, vklyuchen=False) for f, e, lo, hi in (
            ('A', 0.0005, -0.015, 0.0161), ('D', -0.0073, -0.0204, 0.0059), ('E', -0.0, -0.0097, 0.0096), ('G', 0.0175, -0.0009, 0.0359),
            ('K', -0.0104, -0.0279, 0.0071), ('L', -0.0053, -0.0862, 0.0757), ('M', -0.0229, -0.0794, 0.0336),
            ('O', -0.0469, -0.1891, 0.0954))])
    best_ = dict(design='D_best', h=22, leg=dict(type='HARJ'), boost=dict(role='residual', early_stop='qlike_total'))
    vE = designs_faktory_E(best_, faktory_na_E(gB22), 22)
    ok(f'на уровень E (владелец 18:35): h=22 -> {faktory_na_E(gB22)}; вариантов поверх лучшего {len(vE)}',
       faktory_na_E(gB22) == ['D', 'K', 'L', 'M', 'O'] and len(vE) >= 4, str([d['design'] for d in vE]))
    ok('ранняя остановка: у бустинга native (год оценки) запрещена в validate; дизайны driver_v1 — qlike_total',
       C._raises(lambda: C.validate(dict(h=5, boost=dict(role='residual'))))
       and all(str(C.get(C.validate(d), 'boost.early_stop')) == 'qlike_total'
               for f in ('B_h5', 'B_h22', 'B_h1', 'C_h5_shablon') for d in read_designs(f)
               if str(C.get(C.validate(d), 'boost.role')) != 'none'))
    K = contract_check()
    ok(f'контракт подключённых модулей (опции шаблона есть в argparse модуля): {K}',
       all(v == 'OK' or v.endswith('ждёт') for v in K.values()) and all(K.get(m) == 'OK' for m in CFG['modules']))
    T = pd.DataFrame(rows)
    T.to_csv(SMOKE_DIR / 'smoke_driver.csv', index=False)
    with pd.option_context('display.max_colwidth', 120, 'display.width', 220):
        print(T.to_string(index=False))
    log(f'дым r2_driver: {"ВСЕ OK" if T.ok.all() else "ЕСТЬ ОШИБКИ"} за {time.time() - t0:.0f} с (конвейеры {time.time() - t1:.0f} с) '
        f'-> {SMOKE_DIR / "smoke_driver.csv"}')
    return 0 if T.ok.all() else 1


def cmd_smoke_real(a):
    """Дым с НАСТОЯЩИМИ модулями (подпроцессы run/leak по контракту, как в бою): R2_TICKERS = 3 тикера (модули, которые его читают:
    L, L3; P и M берут все 17), шаги 1–3, seed 42, кэш — свой корень с отметкой времени, блок сдвига — свой файл; ворота — дымовые
    (2 складки); состояние, отчёты, ворота — в R2/S/driver_smoke/real/. Модуль без файла — этап ждёт (один проход). Замеры —
    R2/S/driver/dym_real/zamery.csv (оценка боевого времени)."""
    global SMOKE_ROOT, SMOKE_O, SMOKE_REPORT, STATE
    t0 = time.time()
    base = SMOKE_DIR / 'real'
    SMOKE_ROOT = base / time.strftime('cache_%Y%m%d_%H%M%S')
    SMOKE_O, SMOKE_REPORT, STATE = base / 'O', base / 'report', STATE / 'dym_real'
    for d in (SMOKE_ROOT, SMOKE_O, SMOKE_REPORT, STATE / 'run'):
        d.mkdir(parents=True, exist_ok=True)
    for d in (SMOKE_O, SMOKE_REPORT):
        for f in d.glob('*'):
            if f.is_file():
                f.unlink()
    os.environ.update(R2_CACHE=str(SMOKE_ROOT), R2_TICKERS=','.join(SMOKE_TICKERS), R2_BLOCK_FILE=str(base / 'blok_sdvig.csv'))
    C.CACHE_ROOT = SMOKE_ROOT
    SMOKE_FLAG.update(on=True, real=True)
    RUN_EXTRA.update({m: '--tickers ' + ' '.join(SMOKE_TICKERS) for m in ('B', 'BG', 'R')})
    if not (DDIR / 'B_h5.yaml').exists():
        cmd_generate()
    runner = Runner(False)
    hs = a.h or SMOKE_HS
    res, econ_res, th = {}, {}, []
    for h in hs:
        th.append(threading.Thread(target=lambda h=h: econ_res.__setitem__(h, econ(h, runner, True)), name=f'econ_h{h}'))

        def guarded(h=h):
            try:
                res[h] = pipeline(h, runner, True, True)
            except Exception as e:  # noqa: BLE001
                res[h] = dict(h=h, zhdyot=f'ОШИБКА: {type(e).__name__}: {e}')
        th.append(threading.Thread(target=guarded, name=f'h{h}'))
    for t in th:
        t.start()
    for t in th:
        t.join()
    stop_note([res[h] for h in hs if h in res], True)
    Z = pd.read_csv(STATE / 'zamery.csv') if (STATE / 'zamery.csv').exists() else pd.DataFrame()
    print(Z.to_string(index=False) if len(Z) else 'замеров нет')
    blk = pd.read_csv(base / 'blok_sdvig.csv') if (base / 'blok_sdvig.csv').exists() else pd.DataFrame()
    if len(blk):
        blk = blk[blk.vremya >= time.strftime('%F %T', time.localtime(t0))]
    rows = [dict(h=h, econ_ok=econ_res.get(h, {}).get('ok'), econ_prichina=str(econ_res.get(h, {}).get('prichina'))[:200],
                 konveyer=str(res.get(h, {}).get('zhdyot'))[:200]) for h in hs]
    T = pd.DataFrame(rows)
    T.to_csv(base / 'smoke_real.csv', index=False)
    print(T.to_string(index=False))
    print(f'блоки сдвига (предварительные на 3 тикерах): {int((blk.ok.astype(str) == "False").sum()) if len(blk) else 0}')
    log(f'дым с модулями: {time.time() - t0:.0f} с -> {base / "smoke_real.csv"}')
    return 0 if all(r['econ_ok'] or 'ждёт модуль' in r['econ_prichina'] for r in rows) else 1


def _ok_midas(n):
    """Корреляции MIDAS_RV h=22 (1e838122…, R2/O/blok_sdvig.csv и дым L 16:54) -> решение r2_loyo.sdvig_ok при K = {0}, ε = 0,01."""
    import r2_loyo as O
    c = {3: (0.5790372348271617, 0.5949948789579242, 0.5836911193048814, 0.40939562440936667),
         17: (0.6933499441472393, 0.7031082384239836, 0.694905312380682, 0.5571923500985687)}[n]
    return O.sdvig_ok(-1, (0,), corr=dict(zip((-2, -1, 0, 1), c)), eps=C.SDVIG_EPS)[0]


def contract_check():
    """Для каждого модуля: файла нет / нет run (ждёт) / OK — все опции шаблонов run и leak объявлены в argparse модуля."""
    import re
    out = {}
    r = Runner(False)
    for m, spec in CFG['modules'].items():
        f = HERE / spec['file']
        if not f.exists():
            out[m] = 'нет файла — ждёт'
            continue
        if not r.ready(m):
            out[m] = 'нет run — ждёт'
            continue
        src = f.read_text(encoding='utf-8', errors='ignore')
        opts = set(re.findall(r'(--[a-z][a-z0-9-]*)', spec['run'] + ' ' + spec.get('leak', '')))
        miss = [o for o in sorted(opts) if f"'{o}'" not in src]
        out[m] = 'OK' if not miss else f'нет опций {miss}'
    return out


def params_obshchiy():
    """Модули, в коде которых ещё есть общий 'params.csv' (дирижёр 17:50: каждый — свой params_<модуль>.csv; r2_core.params_path)."""
    return sorted({spec['file'] for spec in CFG['modules'].values()
                   if (HERE / spec['file']).exists() and "'params.csv'" in (HERE / spec['file']).read_text(encoding='utf-8', errors='ignore')})


def cmd_status(a):
    for f in sorted((STATE / 'run').glob('*.json')):
        print(f.name, json.dumps(yaml.safe_load(f.read_text()), ensure_ascii=False)[:300])
    for p in 'BCDE':
        print(p, [h for h in CFG['hs'] if gate_path(p, h).exists()])
    return 0


def main():
    ap = argparse.ArgumentParser(description='R2 S: драйвер фаз B–E')
    sp = ap.add_subparsers(dest='cmd', required=True)
    sp.add_parser('generate')
    p = sp.add_parser('run')
    p.add_argument('--h', type=int, nargs='+')
    p.add_argument('--once', action='store_true', help='не ждать ворот/модулей (один проход)')
    p.add_argument('--arhiv-vorot', dest='arhiv_vorot', metavar='SUFFIX',
                   help='переименовать прежние ворота/эффекты/отчёты с суффиксом (напр. _bez_s2) и пересчитать ворота B–E')
    p = sp.add_parser('smoke')
    p.add_argument('--real', action='store_true', help='настоящие модули по контракту (3 тикера, шаги 1–3), замеры времени')
    p.add_argument('--h', type=int, nargs='+')
    p = sp.add_parser('ryady-E', help='ряды уровня E (владелец 20:10) сразу, без ворот C/D')
    p.add_argument('--h', type=int, nargs='+')
    p.add_argument('--workers', type=int, default=5)
    p.add_argument('--only-generate', dest='only_generate', action='store_true')
    sp.add_parser('status')
    a = ap.parse_args()
    if a.cmd == 'ryady-E':
        return cmd_ryady_E(a)
    if a.cmd == 'smoke' and a.real:
        return cmd_smoke_real(a)
    if a.cmd == 'generate':
        return cmd_generate(a)
    if a.cmd == 'run':
        cmd_run(a)
        return 0
    if a.cmd == 'smoke':
        return cmd_smoke(a)
    return cmd_status(a)


if __name__ == '__main__':
    sys.exit(main())
