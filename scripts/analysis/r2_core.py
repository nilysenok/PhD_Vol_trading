#!/usr/bin/env python3
"""R2, агент S: общая основа конвейера второго раунда (протокол v3 `docs/zadaniya/2026-10-26_vtoroy_raund_gibrida.md`, раздел 6;
схема `models/designs/r2/schema_v1.yaml`; настройки `models/designs/r2/r2_S_server.yaml`). Модули L, B, M, E, O, D только
импортируют этот файл и подключаются по ключу схемы; общий код первого раунда не меняется.

API (подробно — docs/server_r2.md, раздел «r2_core»):

  Схема и кэш
    load_schema(path=SCHEMA) -> Schema                    листья схемы: values, default, fixed, list/grid
    validate(design) -> dict                              полный дизайн: все default дописаны, значения нормализованы; DesignError со списком всех ошибок
    canonical(design) -> str                              канонический yaml (без `design`, с `schema: v1`) — вход хэша
    design_hash(design) -> str                            SHA-256 канонического yaml (64 hex)
    cache_dir(design, create=True) -> Path                R2/cache/<sha256[:16]>/, внутри design.yaml (сверка при повторе)
    complexity(design) -> int                             число ключей ≠ default (группы признаков сверх G0 — по одной; fixed, h, seed не считаются)
    grid_keys(design) / variants(design)                  ключи-списки (гиперпараметры внутри складки) и их перебор
    get(design, 'boost.role')                             значение по пути с точками
    series_path / write_series / read_series / have_series   ряды `<имя>_h{h}_s{seed|det}_{part}.parquet` в каталоге дизайна;
                                                          запись атомарная (tmp + os.replace), индекс — свой файл на процесс (один писатель)
    merge_index() / read_index() -> DataFrame             R2/cache/INDEX_R2.csv из _index/*.csv (read_index — без записи)
    load_series(design_sha, part='val', ryad=, h=, seed=, modul=, steps=) -> DataFrame
                                                          все ряды дизайна по хэшу (64 hex, префикс ≥ 16 или сам дизайн) одним
                                                          длинным DataFrame + ryad, h, seed, part, modul, hash; SHA файла сверяется с индексом

  Данные d0, эмбарго, годы
    load_d0(tickers=None, plant=False) -> (A, harj)       features_pit_v3 в версии d0 (r2_nogi.load(d0=True)), строки ≤ 2021, `_tend{h}`
    check_d0(A)                                           info_set = d: rv_d — RV дня d (или последний положительный RV)
    train_mask(A, h, vy) / val_mask(A, h, vy) / inner_masks(A, h, vy)   эмбарго h: окно цели [d+1, d+h] кончается до 1 янв. vy
    step_years(k) -> (vy, train_last); assert_dev_step(k)  шаги разработки 1–6 (val 2016–2021); ≥ 2022 — запрет

  Режим R2 (решение владельца 16:25)
    check_no_future(df)                                   отказ (PermissionError) писать прогнозы за годы > 2021 — в write_series
                                                          для любого part; отбор считает только val 2017–2021
    enable_test_assembly('A', gate_E_path)                единственный флаг записи ≥ 2022: только агент A, только при сборке
                                                          тестовых прогнозов финалистов, только при существующем gate_E

  Пул воркеров
    n_workers(requested=None) -> int                      R2_WORKERS | requested | vCPU − reserve (8 из r2_S_server.yaml), ≥ 1
    share(**data) / SHARED                                данные грузятся один раз в родителе ДО пула; воркеры (fork) видят их без копий
    run_pool(fn, tasks, workers=None) -> list[dict]       fork, imap_unordered, chunksize 1; задачи с 'cost' — дорогие первыми;
                                                          ошибка задачи не роняет остальные: {task, ok, result, error, seconds}

  Проверки (кластер 9, п. 2.6) и детектор утечек
    shift_check(V, A, h, name, design=None, block_file=None, K=None) -> dict   п. 2.6 по правилу O (импорт r2_loyo.sdvig/sdvig_ok):
                                                          решение — r2_loyo.sdvig_ok(…, eps=SDVIG_EPS = 0,01), допуск СИММЕТРИЧНЫЙ:
                                                          c_out − c_in ≤ ε вне [min K; max K] (K — r2_loyo.K_dizaina или явно);
                                                          каждый ряд -> r2_loyo.sdvig_zapis (блок и «плоский профиль» -> blok_sdvig.csv);
                                                          okonchatelno = False на неполной выборке (дым) — блок предварительный
    perturb_frame(A, cut, seed) / plant_frame(A)          шум после среза (и цели, чьё окно заходит за срез) / подложенная утечка «день d+1»
    leak_check(build, A, name, cuts=..., h=None) -> DataFrame   build(A) -> {ряд: DataFrame(date, ticker, step, rv_pred[, part])}:
                                                          Δ на строках ≤ среза = 0, после среза > 0, подлог пойман (середина года);
                                                          граница года — проверка эмбарго (ряды «на всех строках», part)

  Фазы
    check_plan()                                          SHA-256 factor_plan_v1.csv
    require_gate(phase, h) -> dict                       ворота предыдущей фазы R2/O/gate_{B,C,D}_h{h}.yaml и сверка SHA его входов
    run_phase(phase, h, designs, fn, seeds=(42,), steps=1..6, workers=None) -> DataFrame
                                                          fn(design_full, seed, steps) -> {ряд: DataFrame}: запись в кэш, shift_check
                                                          каждого ряда, отметка DONE; готовое пропускается

Запуск:
  python3 scripts/analysis/r2_core.py smoke [--workers 3]     дым: SBER, LKOH, AFLT × шаги 1–3 × seed 42 × h ∈ {1, 5, 22}
  python3 scripts/analysis/r2_core.py show <design.yaml>       полный дизайн, хэш, каталог кэша, сложность
  python3 scripts/analysis/r2_core.py index                    сборка INDEX_R2.csv
Ничего за годы ≥ 2022 не читается: load_d0 отбрасывает строки > 2021, write_series не пишет val с годами ≥ 2022.
"""
import os

for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_k, '1')

import argparse
import hashlib
import io
import itertools
import multiprocessing as mp
import re
import socket
import sys
import time
import traceback
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))

MAC_ROOT = Path('/Users/nilysenok/Desktop/Старое/PhD_Vol_trading')  # на сервере — ссылка на копию проекта (env/setup_server.sh)
REGEN = MAC_ROOT / 'data' / 'regen'
SCHEMA_PATH = REPO / 'models' / 'designs' / 'r2' / 'schema_v2.yaml'  # v2 = v1 + кластеры 16, 18, 20; хэши дизайнов v1 не меняются
CFG_PATH = REPO / 'models' / 'designs' / 'r2' / 'r2_S_server.yaml'
PLAN_PATH = REPO / 'models' / 'designs' / 'r2' / 'factor_plan_v1.csv'
PLAN_SHA = '31b2897751e5d81c776bcd11041dd2d7c2a5ba59b48d438102bbebcdcea3c9b8'
FEAT = REGEN / 'features_pit_v3.parquet'
CFG = yaml.safe_load(open(CFG_PATH)) if CFG_PATH.exists() else {}
CACHE_ROOT = Path(os.environ.get('R2_CACHE', REGEN / 'R2' / 'cache'))
O_DIR = REGEN / 'R2' / 'O'
S_DIR = REGEN / 'R2' / 'S'
HS = (1, 5, 22)
DEV_YEARS = (2017, 2018, 2019, 2020, 2021)
STEPS_DEV = range(1, 7)
MAX_YEAR = 2021
KEY = ['date', 'ticker', 'step']
SHIFTS = (-2, -1, 0, 1)
SDVIG_EPS = 0.01                 # симметричный допуск ε п. 2.6 (решение дирижёра по ❓ 5, O PR #179); сверяется с r2_O_loyo.yaml
SDVIG_N_TIKEROV = 17             # полная выборка проверки сдвига: 17 тикеров × val 2017–2021 (иначе решение предварительное)
SDVIG_GODY = (2017, 2021)
CUTS_MID = ('2018-06-29', '2020-03-02', '2021-06-30')
CUTS_YE = ('2017-12-31', '2019-12-31', '2020-12-31')
SHARED = {}


def log(*a):
    print(time.strftime('%F %T'), *a, flush=True)


def sha_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha_file(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


# ================================================================ схема
class DesignError(ValueError):
    pass


_MISSING = object()
_META = {'default', 'cluster', 'note', 'prio', 'fixed', 'values', 'type'}


def _norm(v):
    """Канон значения: числа -> десятичная строка без лишних нулей (1 == 1.0 -> '1'), bool/None/str как есть, списки и словари — поэлементно."""
    if isinstance(v, bool) or v is None or isinstance(v, str):
        return v
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        if not np.isfinite(v):
            raise DesignError(f'нечисловое значение {v!r}: в схеме бесконечность — строка inf')
        s = format(Decimal(repr(float(v))).normalize(), 'f')
        return s[:-2] if s.endswith('.0') else s
    if isinstance(v, (list, tuple)):
        return [_norm(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _norm(x) for k, x in v.items()}
    raise DesignError(f'тип значения не поддерживается: {type(v).__name__}')


_BOOL_STR = {'on': True, 'yes': True, 'true': True, 'off': False, 'no': False, 'false': False}


def _yaml_scalars(v):
    """Единое правило R2 (совпадает с r2_loyo._n агента O): схема читается yaml.safe_load (YAML 1.1), поэтому off/on/yes/no в ней —
    false/true; в дизайне те же слова в любом регистре -> bool; строка-число -> число ('0.5' = 0.5, '3' = 3); 'null' -> None;
    'inf'/'nan' остаются строками (в схеме inf — строка)."""
    if isinstance(v, str):
        s = v.strip()
        if s.lower() in _BOOL_STR:
            return _BOOL_STR[s.lower()]
        if s.lower() in ('inf', '-inf', '+inf', 'nan'):
            return v
        try:
            return float(s)  # _native: целое -> int
        except ValueError:
            pass
        return None if s in ('null', 'Null', 'NULL', '~') else v
    if isinstance(v, (list, tuple)):
        return [_yaml_scalars(x) for x in v]
    if isinstance(v, dict):
        return {k: _yaml_scalars(x) for k, x in v.items()}
    return v


def _native(v):
    """Значение для модулей: целое число — int (1.0 -> 1), прочие числа — float; списки и словари — поэлементно."""
    if isinstance(v, bool) or v is None or isinstance(v, str):
        return v
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        return int(v) if np.isfinite(v) and float(v).is_integer() else float(v)
    if isinstance(v, (list, tuple)):
        return [_native(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _native(x) for k, x in v.items()}
    raise DesignError(f'тип значения не поддерживается: {type(v).__name__}')


class Leaf:
    """Лист схемы. kind: scalar (список в дизайне = сетка внутри складки) | list (значение — список, напр. features) |
    grid (значение — список-сетка, default — вся сетка, напр. leg.enet.lambda) | const (fixed без values) | name (`design`).
    values/default — как в схеме (типы Python); nvalues/ndefault — канон для сравнения (_norm)."""

    def __init__(self, path, node, parent_default=False):
        self.path = path
        self.fixed = bool(node.get('fixed', False))
        self.note = node.get('note', '')
        if isinstance(node.get('type'), str) and 'values' not in node:
            self.kind, self.values, self.default = 'name', None, _MISSING
        elif 'values' not in node:  # fixed-константа (selection.entry и т. п.)
            self.kind = 'const'
            c = _native({k: v for k, v in node.items() if k not in ('fixed', 'note')})
            self.values, self.default = [c], c
        else:
            self.values = _native(node['values'])
            if 'default' in node:
                self.default = _native(node['default'])
                self.kind = 'list' if isinstance(self.default, list) and not self.fixed else 'scalar'
            elif parent_default:  # grid_default — сетка по умолчанию у́же values (k16.k, k18.lam: 0 — только явно, E8; хэши прежние)
                self.kind, self.default = 'grid', list(_native(node.get('grid_default', node['values'])))
                assert set(map(str, self.default)) <= set(map(str, self.values)), f'{".".join(path)}: grid_default вне values'
            elif self.fixed:
                self.kind, self.default = 'scalar', self.values[0]
            else:
                self.kind, self.default = 'scalar', _MISSING  # обязательный (h)
        self.nvalues = _norm(self.values) if self.values is not None else None
        self.ndefault = _MISSING if self.default is _MISSING else _norm(self.default)

    @property
    def dotted(self):
        return '.'.join(self.path)


class Schema:
    def __init__(self, raw, path=None):
        """`extends: <файл>` — версия поверх предыдущей: её листья берутся без изменений (since = версия базы), новые — since = эта версия;
        переопределять листья базы нельзя. base_version — версия, которая остаётся в каноне, пока новые ключи равны default."""
        self.raw, self.path = raw, path
        self.version = str(raw.get('schema', ''))
        self.leaves = {}
        self.base_version = self.version
        if raw.get('extends'):
            base = load_schema(Path(path).parent / raw['extends'])
            self.leaves.update(base.leaves)
            self.base_version = base.base_version
        self._since = self.version
        for k, node in raw.items():
            if k in ('schema', 'extends'):
                continue
            self._walk((k,), node, False)

    def _walk(self, path, node, parent_default):
        if not isinstance(node, dict):
            raise DesignError(f'схема: узел {".".join(path)} не словарь')
        is_leaf = 'values' in node or isinstance(node.get('type'), str) or (node.get('fixed') and not any(
            isinstance(v, dict) and 'values' in v for v in node.values()))
        if is_leaf:
            if path in self.leaves:
                raise DesignError(f'схема {self.version}: лист {".".join(path)} уже есть в базовой версии — переопределять нельзя')
            L = Leaf(path, node, parent_default)
            L.since = self._since
            self.leaves[path] = L
            return
        for k, v in node.items():
            if k in _META and not isinstance(v, dict):  # `leg.type` — лист, а не метаполе
                continue
            self._walk(path + (k,), v, 'default' in node)


_SCHEMA = {}


def load_schema(path=SCHEMA_PATH):
    path = Path(path)
    if path not in _SCHEMA:
        _SCHEMA[path] = Schema(yaml.safe_load(open(path)), path)
    return _SCHEMA[path]


def _flatten(d, prefix=()):
    """Вложенный или «с точками» дизайн -> {путь: значение}."""
    out = {}
    for k, v in d.items():
        p = prefix + tuple(str(k).split('.'))
        if isinstance(v, dict) and not (len(p) == 2 and p[0] == 'selection'):
            sub = _flatten(v, p)
            out.update(sub) if sub else out.__setitem__(p, v)
        else:
            out[p] = v
    return out


def _nest(flat):
    out = {}
    for p, v in flat.items():
        d = out
        for k in p[:-1]:
            d = d.setdefault(k, {})
        d[p[-1]] = v
    return out


def validate(design, schema=None, keep_scalar_grid=False, es_check=True):
    """-> полный нормализованный дизайн (вложенный dict) с `schema` и `design`; ошибки — DesignError со списком.
    Значение grid-листа (leg.enet.*) всегда список (одиночное -> [x]); keep_scalar_grid — оставить скаляр (для complexity).
    es_check — запрет ранней остановки на году оценки (роль ≠ none и early_stop = native)."""
    S = schema or load_schema()
    if isinstance(design, (str, Path)):
        design = yaml.safe_load(open(design))
    design = dict(design)
    inputs = design.pop('inputs', None)
    flat = _flatten(design)
    flat.pop(('schema',), None)
    errs, full = [], {}
    if inputs is not None:
        try:
            inputs = _check_inputs(inputs)
        except DesignError as e:
            errs.append(str(e))
    for p in flat:
        if p not in S.leaves:
            errs.append(f'неизвестный ключ {".".join(p)}')
    for p, L in S.leaves.items():
        if L.kind == 'name':
            v = flat.get(p, '')
            if not isinstance(v, str):
                errs.append(f'{L.dotted}: имя — строка')
            full[p] = v
            continue
        if p in flat:
            try:
                v = _native(_yaml_scalars(flat[p]))
                nv = _norm(v)
            except DesignError as e:
                errs.append(f'{L.dotted}: {e}')
                continue
        elif L.default is _MISSING:
            errs.append(f'{L.dotted}: обязательный ключ (значения {L.values})')
            continue
        else:
            v, nv = L.default, L.ndefault
        if L.fixed or L.kind == 'const':
            if nv != L.ndefault:
                errs.append(f'{L.dotted}: фиксирован протоколом = {L.default!r}, в дизайне {v!r}')
        elif L.kind in ('list', 'grid'):
            scalar = not isinstance(v, list)
            v, nv = ([v], [nv]) if scalar else (v, nv)
            bad = [x for x in nv if x not in L.nvalues]
            if bad or not nv or len(set(map(str, nv))) != len(nv):
                errs.append(f'{L.dotted}: {v!r} — нужен непустой список без повторов из {L.values}')
            if L.dotted == 'features' and 'G0' not in nv:
                errs.append('features: G0 обязательна')
            if scalar and keep_scalar_grid and L.kind == 'grid':
                v = v[0]
        else:
            if isinstance(v, list):  # сетка внутри складки
                if len(nv) < 2 or len(set(map(str, nv))) != len(nv) or any(x not in L.nvalues for x in nv):
                    errs.append(f'{L.dotted}: сетка {v!r} — ≥ 2 разных значения из {L.values}')
            elif nv not in L.nvalues:
                errs.append(f'{L.dotted}: {v!r} вне списка {L.values}')
        full[p] = v
    # дирижёр 26.10, 17:44 (находка B): ранняя остановка на году оценки (es_year == eval_year) запрещена — у бустинга (роль ≠ none)
    # только boost.early_stop = qlike_total: внутренняя складка vy−1 с эмбарго h (inner_masks), ≤ 300 деревьев, patience 30.
    # es_check=False — только для сверки хэшей исторических баз (factor_v1 O в check-o), они не запускаются.
    if es_check and str(full.get(('boost', 'role'), 'none')) != 'none' and str(full.get(('boost', 'early_stop'))) == 'native':
        errs.append('boost.early_stop = native: ранняя остановка на году оценки (es_year == eval_year) запрещена — нужна qlike_total '
                    '(внутренняя складка vy−1 с эмбарго h, ≤ 300 деревьев, patience 30)')
    if errs:
        raise DesignError('дизайн не прошёл схему ' + S.version + ':\n  ' + '\n  '.join(errs))
    out = _nest(full)
    out['schema'] = S.version
    if inputs:
        out['inputs'] = inputs
    return out


# ---------------------------------------------------------------- внешние входы дизайна (O, PR #172): base_1 / leg_1 для persist и reconcile
INPUT_ID = ('file', 'col', 'design_sha', 'series', 'sha256')  # поля, задающие идентичность входа (описания вроде `ryad` в хэш не идут)
H1_INPUT_NAMES = ('base_1', 'leg_1')


def _check_inputs(inputs):
    if not isinstance(inputs, dict) or not all(isinstance(v, dict) for v in inputs.values()):
        raise DesignError('inputs: словарь {имя входа: {file | design_sha + series, [col], [sha256]}}')
    out = {}
    for name, spec in inputs.items():
        if not (spec.get('file') or spec.get('design_sha')):
            raise DesignError(f'inputs.{name}: нужен file или design_sha')
        out[str(name)] = {k: spec[k] for k in spec if k in INPUT_ID or k in ('ryad', 's2', 'seed')}
    return out


def _file_sha_memo(p):
    st = p.stat()
    k = (str(p), st.st_size, st.st_mtime_ns)
    if k not in _FSHA:
        _FSHA[k] = sha_file(p)
    return _FSHA[k]


_FSHA = {}


def inputs_identity(inputs):
    """Канон входов для хэша: только поля идентичности; у входа-файла без sha256 — SHA содержимого (шаблон {seed} — все seed 42–51,
    что есть на диске, SHA их списка). Вход из кэша R2 — design_sha + series (имя ряда)."""
    out = {}
    for name, spec in sorted(inputs.items()):
        ident = {k: spec[k] for k in INPUT_ID if k in spec}
        if 'file' in spec and 'sha256' not in spec:
            f = str(spec['file'])
            fs = [f.format(seed=s) for s in range(42, 52)] if '{seed}' in f else [f]
            shas = []
            for x in fs:
                p = Path(x) if Path(x).is_absolute() else REGEN / x
                if p.exists():
                    shas.append(f'{x}:{_file_sha_memo(p)}')
            if not shas:
                raise DesignError(f'inputs.{name}: нет файла {f}')
            ident['sha256'] = sha_bytes('\n'.join(shas).encode())
        out[name] = ident
    return out


def needs_h1_inputs(design):
    """persist / reconcile (O, PR #172) и ens.ec (E, PR #187) на h ≠ 1 читают ряды h=1 (base_1, leg_1) — они обязаны входить в хэш."""
    full = validate(design, es_check=False)
    on = lambda k: str(get(full, k)) not in ('off', 'False')
    return int(full['h']) != 1 and (on('mh.persist') or on('mh.reconcile') or on('ens.ec'))


def with_inputs(design, inputs):
    """Дизайн + внешние входы (только идентичность: file/col/sha256 или design_sha/series)."""
    d = dict(validate(design, es_check=False))
    d['inputs'] = _check_inputs({k: dict(v) for k, v in inputs.items()})
    return d


def canonical(design, schema=None):
    """Канонический yaml: полный дизайн без `design`, с `schema`; числа — десятичные строки (шапка schema_v1.yaml, пп. 1–5).
    schema — путь/объект схемы (по умолчанию — текущая, v2); для сверки инвариантности хэшей v1."""
    S = schema if isinstance(schema, Schema) else load_schema(schema or SCHEMA_PATH)
    full = validate(design, S, es_check=False)  # хэш — не запуск: запрет ранней остановки на году оценки — в run_phase / validate
    full.pop('design', None)
    inputs = full.pop('inputs', None)
    # ключи версий новее базовой (v2: кластеры 16, 18, 20) в канон не пишутся, пока равны default; тогда и `schema` — базовая (v1),
    # и хэш дизайна v1 не меняется
    flat = _flatten(full)
    newer = False
    for p, L in S.leaves.items():
        if getattr(L, 'since', S.base_version) != S.base_version and p in flat:
            if _norm(flat[p]) == L.ndefault:
                del flat[p]
            else:
                newer = True
    flat[('schema',)] = S.version if newer else S.base_version
    full = _drop_empty(_nest(flat))
    if inputs:  # внешние входы (base_1, leg_1 …) — в хэш входят их идентичность и SHA содержимого
        full['inputs'] = inputs_identity(inputs)
    return yaml.safe_dump(_norm(full), sort_keys=True, allow_unicode=True, default_flow_style=False)


def _drop_empty(d):
    return {k: _drop_empty(v) if isinstance(v, dict) else v for k, v in d.items() if not (isinstance(v, dict) and not v)}


def design_hash(design, schema=None):
    return sha_bytes(canonical(design, schema).encode('utf-8'))


SCHEMA_V1_PATH = REPO / 'models' / 'designs' / 'r2' / 'schema_v1.yaml'


def check_v2_invariance(designs):
    """Хэш каждого дизайна v1 по схеме v2 = хэш по схеме v1 (новые ключи по умолчанию в канон не пишутся). -> (n, расхождения)."""
    bad = []
    for d in designs:
        a, b = design_hash(d, SCHEMA_V1_PATH), design_hash(d)
        if a != b:
            bad.append((d.get('design'), a[:16], b[:16]))
    return len(designs), bad


def get(design, dotted):
    d = design
    for k in dotted.split('.'):
        d = d[k]
    return d


def complexity(design):
    """c(u), п. 2.4 — то же правило, что r2_loyo.slozhnost агента O: число ключей со значением ≠ default; не считаются fixed,
    h, seed, ens.seeds (число seed задаёт протокол: скрининг 3, финалисты 10), design; features — число групп сверх G0;
    сетка с default «вся сетка» (leg.enet) — выбор внутри складки: в счёт идёт один раз на группу и только если задано
    одиночное значение (подсетка-список — не в счёт)."""
    S = load_schema()
    flat = _flatten(validate(design, keep_scalar_grid=True, es_check=False))
    c, grids = 0, set()
    for p, L in S.leaves.items():
        if L.kind in ('name', 'const') or L.fixed or L.dotted in ('h', 'seed', 'ens.seeds') or L.default is _MISSING:
            continue
        v = flat[p]
        if L.dotted == 'features':
            c += len(set(v) - {'G0'})
        elif L.kind == 'grid':
            if not isinstance(v, list):
                grids.add(p[:-1])
        elif _norm(v) != L.ndefault:
            c += 1
    return c + len(grids)


def grid_keys(design):
    """Ключи-скаляры, заданные списком (сетка внутри складки), и grid-листы, отличные от одиночного значения."""
    S = load_schema()
    flat = _flatten(validate(design, es_check=False))

    def group_on(p):  # сетки групп новых версий (k16/k18/k20) действуют, только если переключатель группы не default
        sib = [q for q, M in S.leaves.items() if q[:-1] == p[:-1] and M.kind == 'scalar']
        return any(_norm(flat[q]) != S.leaves[q].ndefault for q in sib)

    return {L.dotted: flat[p] for p, L in S.leaves.items()
            if isinstance(flat.get(p), list) and (L.kind == 'grid' or (L.kind == 'scalar' and not L.fixed)) and len(flat[p]) > 1
            and not (getattr(L, 'since', S.base_version) != S.base_version and L.kind == 'grid' and not group_on(p))}


def variants(design, keys=None):
    """-> [(tag, назначение {ключ: значение}, конкретный дизайн)] — декартово произведение сеток внутри складки.
    keys — только эти ключи (модуль знает, какие сетки к нему относятся: напр. leg.enet.* не действует на HARJ)."""
    full = validate(design)
    G = grid_keys(full)
    if keys is not None:
        G = {k: v for k, v in G.items() if k in set(keys)}
    keys = sorted(G)
    out = []
    for combo in itertools.product(*[G[k] for k in keys]):
        a = dict(zip(keys, combo))
        flat = _flatten(full)
        for k, v in a.items():
            flat[tuple(k.split('.'))] = v
        d = _nest(flat)
        out.append(('__'.join(f'{k}={v}' for k, v in a.items()) or 'base', a, d))
    return out


# ================================================================ кэш
def cache_dir(design, create=True, root=None):
    """R2/cache/<sha256[:16]>/ с design.yaml (канон). Повтор — сверка канона: другое содержимое = ошибка (коллизия/схема)."""
    text = canonical(design)
    h = sha_bytes(text.encode('utf-8'))
    d = Path(root or CACHE_ROOT) / h[:16]
    if create:
        d.mkdir(parents=True, exist_ok=True)
        f = d / 'design.yaml'
        if f.exists():
            if f.read_text(encoding='utf-8') != text:
                raise DesignError(f'кэш {d}: design.yaml не совпадает с каноном дизайна (коллизия префикса или смена схемы)')
        else:
            _atomic_write(f, text.encode('utf-8'))
    return d


def _atomic_write(path, data):
    path = Path(path)
    tmp = path.with_name(f'.{path.name}.{socket.gethostname()}.{os.getpid()}.tmp')
    with open(tmp, 'wb') as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def series_path(design, name, h, seed, part='val', root=None):
    sd = 'det' if seed in (None, 'det') else int(seed)
    return cache_dir(design, root=root) / f'{name}_h{int(h)}_s{sd}_{part}.parquet'


def have_series(design, name, h, seed, part='val', root=None):
    return series_path(design, name, h, seed, part, root).exists()


def read_series(design, name, h, seed, part='val', root=None, steps=None):
    p = series_path(design, name, h, seed, part, root)
    filters = [('step', '<=', max(steps))] if steps else None
    d = pd.read_parquet(p, filters=filters)
    return d[d.step.isin(list(steps))] if steps else d


SERIES_COLS = ['date', 'ticker', 'year', 'step', 'rv_pred']


def write_series(design, name, df, h, seed, part='val', module='core', root=None, note=''):
    """Атомарная запись ряда в каталог дизайна + строка индекса в файл этого процесса (один писатель на воркер)."""
    miss = [c for c in SERIES_COLS if c not in df.columns]
    if miss:
        raise ValueError(f'{name}: нет колонок {miss}')
    check_no_future(df, name)
    p = np.asarray(df['rv_pred'], dtype=float)
    if not np.isfinite(p).all() or (p <= 0).any():
        raise ValueError(f'{name}: rv_pred должен быть конечным и > 0')
    path = series_path(design, name, h, seed, part, root)
    own = path.with_name(path.name + '.owner')  # один ряд — один модуль: чужой модуль не затирает файл (цепочка L -> B -> M -> E)
    if own.exists() and own.read_text().strip() != module:
        raise PermissionError(f'{path.name}: ряд уже записан модулем {own.read_text().strip()} — модуль {module} должен писать под своим именем')
    if not own.exists():
        _atomic_write(own, module.encode())
    buf = io.BytesIO()
    df.sort_values(KEY).reset_index(drop=True).to_parquet(buf, index=False)
    b = buf.getvalue()
    _atomic_write(path, b)
    idx = Path(root or CACHE_ROOT) / '_index'
    idx.mkdir(parents=True, exist_ok=True)
    row = pd.DataFrame([dict(fayl=str(path.relative_to(Path(root or CACHE_ROOT))), hash=path.parent.name, ryad=name, h=int(h),
                             seed=seed if seed is not None else 'det', part=part, stroki=len(df),
                             shagi=','.join(map(str, sorted(pd.unique(df['step'])))), sha256=sha_bytes(b), modul=module,
                             vremya=time.strftime('%F %T'), note=note)])
    f = idx / f'{module}_{socket.gethostname().split(".")[0]}_{os.getpid()}.csv'
    row.to_csv(f, mode='a', header=not f.exists(), index=False)
    return path


def read_index(root=None):
    """Индекс кэша из файлов писателей _index/*.csv (без записи INDEX_R2.csv); последняя запись файла побеждает."""
    root = Path(root or CACHE_ROOT)
    fs = sorted((root / '_index').glob('*.csv'))
    if not fs:
        return pd.DataFrame(columns=['fayl', 'hash', 'ryad', 'h', 'seed', 'part', 'stroki', 'shagi', 'sha256', 'modul', 'vremya', 'note'])
    I = pd.concat([pd.read_csv(f, dtype={'hash': str, 'seed': str}) for f in fs], ignore_index=True)
    return I.sort_values('vremya', kind='mergesort').drop_duplicates(['fayl'], keep='last').sort_values(['hash', 'ryad']).reset_index(drop=True)


def merge_index(root=None):
    root = Path(root or CACHE_ROOT)
    I = read_index(root)
    if len(I):
        I.to_csv(root / 'INDEX_R2.csv', index=False)
    return I


_SERIES_RE = re.compile(r'^(?P<ryad>.+)_h(?P<h>\d+)_s(?P<seed>det|\d+)_(?P<part>[a-z]+)\.parquet$')


def load_series(design_sha, part='val', ryad=None, h=None, seed=None, modul=None, steps=None, root=None, verify=True):
    """Точка входа для модулей-потребителей (E, M, O, D): все ряды дизайна по его хэшу -> один длинный DataFrame
    (колонки ряда + ryad, h, seed, part, modul, hash).
      design_sha — полный SHA-256 (64 hex), префикс ≥ 16 (= имя каталога кэша) или сам дизайн (dict/yaml);
      фильтры — ryad (строка или список), h, seed (int | 'det' | список), modul (кто записал: 'B', 'M', …), steps;
      источник — индекс кэша (_index/*.csv, его пишет write_series); ряды каталога без строки индекса тоже берутся
      (modul = '?'); verify — сверка SHA-256 файла с индексом (иначе ValueError: файл переписан мимо write_series).
    Пустой результат — пустой DataFrame. Годы > 2021 в val не бывает (write_series)."""
    root = Path(root or CACHE_ROOT)
    if not isinstance(design_sha, str) or not re.fullmatch(r'[0-9a-f]{16,64}', design_sha):
        design_sha = design_hash(design_sha)
    hid = design_sha[:16]
    d = root / hid
    if not d.is_dir():
        raise FileNotFoundError(f'нет каталога кэша {d} (дизайн {design_sha[:16]}…)')
    I = read_index(root)
    I = I[I['hash'].astype(str) == hid]
    rows = {}
    for r in I.itertuples():
        rows[Path(r.fayl).name] = dict(sha256=r.sha256, modul=r.modul)
    want = lambda x, v: x is None or (str(v) in {str(y) for y in (x if isinstance(x, (list, tuple, set)) else [x])})
    out = []
    for f in sorted(d.glob('*.parquet')):
        m = _SERIES_RE.match(f.name)
        if not m or m['part'] != part:
            continue
        if not (want(ryad, m['ryad']) and want(h, m['h']) and want(seed, m['seed'])):
            continue
        meta = rows.get(f.name, dict(sha256=None, modul='?'))
        if not want(modul, meta['modul']):
            continue
        b = f.read_bytes()
        if verify and meta['sha256'] and sha_bytes(b) != meta['sha256']:
            raise ValueError(f'{f}: SHA-256 не совпадает с индексом кэша')
        x = pd.read_parquet(io.BytesIO(b))
        if steps is not None:
            x = x[x['step'].isin(list(steps))]
        out.append(x.assign(ryad=m['ryad'], h=int(m['h']), seed=m['seed'], part=part, modul=meta['modul'], hash=hid))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


# ---------------------------------------------------------------- параметры модулей (дирижёр 26.10, 17:50)
# Каждый модуль пишет СВОЙ файл params_<модуль>.csv в каталог дизайна; общий params.csv не используется (цепочка L → M → … пишет в один
# каталог — общий файл затирали друг другу; находка 17:27, M). Потребители (D) читают все params_*.csv через load_params.
PARAMS_RE = re.compile(r'^params_(?P<modul>[A-Za-z0-9]+)\.csv$')


def params_path(design, module, root=None):
    """Путь параметров модуля в каталоге дизайна: <кэш>/<hash16>/params_<модуль>.csv."""
    if not re.fullmatch(r'[A-Za-z0-9]+', str(module)):
        raise ValueError(f'имя модуля {module!r}: только буквы и цифры')
    return cache_dir(design, root=root) / f'params_{module}.csv'


def load_params(design_sha, modul=None, root=None):
    """Все params_<модуль>.csv дизайна одним кадром (колонки файла + modul, hash); modul — фильтр (строка или список).
    Общий params.csv (старый формат) не читается — он не однозначен по модулю."""
    root = Path(root or CACHE_ROOT)
    if not isinstance(design_sha, str) or not re.fullmatch(r'[0-9a-f]{16,64}', design_sha):
        design_sha = design_hash(design_sha)
    d = root / design_sha[:16]
    want = {modul} if isinstance(modul, str) else (set(modul) if modul else None)
    out = []
    for f in sorted(d.glob('params_*.csv')) if d.is_dir() else []:
        m = PARAMS_RE.match(f.name)
        if m and (want is None or m['modul'] in want):
            out.append(pd.read_csv(f).assign(modul=m['modul'], hash=design_sha[:16]))
    return pd.concat(out, ignore_index=True, sort=False) if out else pd.DataFrame()


# ================================================================ данные d0, эмбарго
def load_d0(tickers=None, plant=False, feat=FEAT, max_year=MAX_YEAR):
    """features_pit_v3 в версии d0 (код L: r2_nogi.load(d0=True)) -> (A, harj). Строки с годом > max_year отбрасываются;
    `_tend{h}` — дата последнего дня окна цели [d+1, d+h] (по полной таблице, до отбора тикеров и лет)."""
    import r2_nogi as RN
    A, harj = RN.load(feat, d0=True, plant=plant)
    A['date'] = pd.to_datetime(A['date'])
    g = A.groupby('ticker')['date']
    for h in HS:
        A[f'_tend{h}'] = g.shift(-h)
    A = A[A['date'].dt.year <= max_year]
    if tickers:
        A = A[A['ticker'].isin(list(tickers))]
    return A.reset_index(drop=True), harj


def check_d0(A):
    """info_set = d: rv_d в строке дня d — RV того же дня (нулевой RV — последним положительным RV тикера)."""
    rvp = A['rv'].where(A['rv'] > 0).groupby(A['ticker']).ffill().fillna(A['rv'])
    bad = ~np.isclose(A['rv_d'].values, rvp.values, rtol=0, atol=0)
    if bad.any():
        raise AssertionError(f'info_set: rv_d ≠ RV дня d в {int(bad.sum())} строках')
    return True


def step_years(k):
    """Шаг k -> (val-год vy = 2015 + k, последний год обучения vy − 1)."""
    return 2015 + k, 2014 + k


def assert_dev_step(k, allow_test=False):
    if not (allow_test and _TEST_ASSEMBLY['on']) and not (1 <= int(k) <= 6):
        raise AssertionError(f'шаг {k}: в разработке R2 только шаги 1–6 (val ≤ 2021); шаги 7–10 — только сборка A (enable_test_assembly)')


# ---------------------------------------------------------------- режим R2 (решение владельца 16:25): прогнозов ≥ 2022 не пишет никто,
# кроме агента A при сборке тестовых прогнозов финалистов — отдельным флагом.
_TEST_ASSEMBLY = {'on': False, 'who': None}


def enable_test_assembly(who, finalists_gate):
    """Включить запись прогнозов ≥ 2022 в этом процессе. Только агент A, только при сборке (фаза F): нужен существующий файл ворот E
    (`R2/O/gate_E_h{h}.yaml`) и who == 'A'. Отбор (фазы B–E) этим флагом не пользуется."""
    if who != 'A':
        raise PermissionError('прогнозы ≥ 2022 пишет только агент A при сборке тестовых прогнозов финалистов')
    if not Path(finalists_gate).exists():
        raise PermissionError(f'нет ворот E {finalists_gate} — сборка теста до решения ворот запрещена')
    _TEST_ASSEMBLY.update(on=True, who=who)
    log(f'РЕЖИМ СБОРКИ ТЕСТА (A): запись прогнозов ≥ 2022 разрешена в этом процессе; ворота {finalists_gate}')


def check_no_future(df, name='ряд'):
    """Отказ писать прогнозы за годы > 2021 (по дате строки и по колонке year), кроме режима сборки A."""
    if _TEST_ASSEMBLY['on']:
        return
    yrs = pd.to_datetime(df['date']).dt.year
    ymax = int(max(yrs.max(), df['year'].max() if 'year' in df else yrs.max()))
    if ymax > MAX_YEAR:
        try:  # протокол v4, разд. 1: исключение, остановка задачи, строка в R2/S/otkaz_2022.csv
            f = S_DIR / 'otkaz_2022.csv'
            S_DIR.mkdir(parents=True, exist_ok=True)
            pd.DataFrame([dict(vremya=time.strftime('%F %T'), ryad=name, god_max=ymax, stroki_2022plus=int((yrs > MAX_YEAR).sum()),
                               pid=os.getpid())]).to_csv(f, mode='a', header=not f.exists(), index=False)
        except OSError:
            pass
        raise PermissionError(f'{name}: прогнозы за {ymax} > {MAX_YEAR} — режим R2 запрещает запись ≥ 2022 (отбор — только val 2017–2021)')


def _tend(A, h):
    return pd.to_datetime(A[f'_tend{h}']) if f'_tend{h}' in A else pd.to_datetime(A.groupby('ticker')['date'].shift(-h))


def train_mask(A, h, vy, emb=True):
    """Обучение шага с val-годом vy: годы < vy с целью; эмбарго h — окно цели кончается до 1 января vy."""
    m = (A['year'].values < vy) & A[f'rv_target_h{h}'].notna().values
    if emb:
        m &= (_tend(A, h) < pd.Timestamp(f'{vy}-01-01')).fillna(False).values
    return m


def val_mask(A, h, vy):
    return (A['year'].values == vy) & A[f'rv_target_h{h}'].notna().values


def inner_masks(A, h, vy):
    """Ранняя остановка по QLIKE итога (п. 4): внутренняя складка — год vy − 1, обучение — годы < vy − 1 с эмбарго,
    внутренняя валидация — строки vy − 1, чьё окно цели не заходит в vy; итоговое переобучение — train_mask(vy)."""
    tr_in = train_mask(A, h, vy - 1)
    va_in = val_mask(A, h, vy - 1) & (_tend(A, h) < pd.Timestamp(f'{vy}-01-01')).fillna(False).values
    return tr_in, va_in, train_mask(A, h, vy)


# ================================================================ пул воркеров
def n_workers(requested=None):
    env = os.environ.get('R2_WORKERS')
    if env:
        return max(1, int(env))
    if requested:
        return max(1, int(requested))
    pool = CFG.get('pool', {})
    if sys.platform == 'darwin':  # решение владельца 16:25: весь счёт на Маке, 10 воркеров
        return max(1, min(int(pool.get('mac_workers', 10)), os.cpu_count() or 2))
    return max(1, (os.cpu_count() or 2) - int(pool.get('reserve_vcpu', 8)))


def share(**data):
    """Положить данные в SHARED до запуска пула: fork отдаёт их воркерам копией-при-записи, без pickle."""
    SHARED.update(data)


def _init_worker():
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(1)
    except Exception:  # noqa: BLE001 — threadpoolctl необязателен
        pass


def _call(a):
    fn, task = a
    t0 = time.time()
    try:
        return dict(task=task, ok=True, result=fn(task), error='', seconds=round(time.time() - t0, 2), pid=os.getpid())
    except Exception:  # noqa: BLE001 — ошибка задачи возвращается, остальные задачи идут дальше
        return dict(task=task, ok=False, result=None, error=traceback.format_exc(), seconds=round(time.time() - t0, 2), pid=os.getpid())


def run_pool(fn, tasks, workers=None, label='pool'):
    """fn(task) — функция уровня модуля; данные — из SHARED. Задачи-словари с ключом 'cost' идут по убыванию стоимости."""
    tasks = list(tasks)
    if tasks and isinstance(tasks[0], dict) and 'cost' in tasks[0]:
        tasks.sort(key=lambda t: -t['cost'])
    w = min(n_workers(workers), max(1, len(tasks)))
    log(f'{label}: задач {len(tasks)}, воркеров {w}')
    if w == 1:
        _init_worker()
        res = [_call((fn, t)) for t in tasks]
    else:
        with mp.get_context('fork').Pool(w, initializer=_init_worker) as P:
            res = list(P.imap_unordered(_call, [(fn, t) for t in tasks], chunksize=1))
    bad = [r for r in res if not r['ok']]
    log(f'{label}: готово {len(res) - len(bad)}, ошибок {len(bad)}')
    for r in bad[:3]:
        print(r['error'], file=sys.stderr)
    return res


# ================================================================ проверка сдвига (п. 2.6)
def shift_check(V, A, h, name, design=None, block_file=None, K=None):
    """П. 2.6 по правилу агента O (PR #167) — импорт r2_loyo, не копия:
      корреляции и argmax — r2_loyo.sdvig (Спирмен ln ŷ_d с ln RV_{d+s}, s ∈ {−2,−1,0,+1}, по тикерам, медиана, val 2017–2021);
      решение — r2_loyo.sdvig_ok: ок, если min K ≤ argmax ≤ 0; +1 блокируется всегда; K — последние известные дни входов ряда:
      явно (K=…) или r2_loyo.K_dizaina(design) по группам признаков (метаданные F, r2_O_loyo.yaml sdvig.izvestnyy_den_grupp); иначе {0};
      блок — строка в R2/O/blok_sdvig.csv через r2_loyo.blok_zapis (тот же формат, с posledniy_izvestnyy_den).
    A — таблица с date, ticker, rv (дневная RV дня строки), строки ≤ 2021 (load_d0).
    26.10, 17:10 (поручение дирижёра, PR #179 O): решение — явный вызов r2_loyo.sdvig_ok(argmax, K, corr, eps=SDVIG_EPS) с СИММЕТРИЧНЫМ
    допуском ε = 0,01 (значение — r2_O_loyo.yaml sdvig.eps, сверяется). Полнота выборки (17 тикеров × val 2017–2021) пишется в строку:
    блок на неполной выборке (дым: 3 тикера × 2017–2018) — предварительный (okonchatelno = False, пометка); окончательно — на полной."""
    import r2_loyo as O
    eps = float(O.CFG['sdvig']['eps'])
    if abs(eps - SDVIG_EPS) > 1e-12:
        raise AssertionError(f'r2_O_loyo.yaml sdvig.eps = {eps} ≠ {SDVIG_EPS} (симметричный допуск, решение по ❓ 5, PR #179)')
    if K is None:
        K = O.K_dizaina(design) if design is not None else (0,)
    rv = A[['date', 'ticker', 'rv']].copy()
    rv['date'] = pd.to_datetime(rv['date'])
    rv = rv[rv['date'].dt.year <= MAX_YEAR]
    s = O.sdvig(V[['date', 'ticker', 'rv_pred']], rv, K=tuple(K))
    corr = {sh: s[f'corr_s{sh:+d}'] for sh in O.CFG['sdvig']['sdvigi']}
    ok, why = O.sdvig_ok(s['argmax'], tuple(K), corr=corr, eps=SDVIG_EPS)  # решение — только функцией O (импорт, не копия)
    s.update(ok=ok, blok=not ok, prichina='' if ok else why, pometka=why if ok else '')
    gody = sorted({int(y) for y in pd.to_datetime(V['date']).dt.year if SDVIG_GODY[0] <= y <= SDVIG_GODY[1]})
    polnaya = s['n_tikerov'] >= SDVIG_N_TIKEROV and gody == list(range(SDVIG_GODY[0], SDVIG_GODY[1] + 1))
    s['okonchatelno'] = bool(polnaya)
    if not polnaya:
        tag = (f'неполная выборка ({s["n_tikerov"]} тик., val {gody[0] if gody else "—"}–{gody[-1] if gody else "—"}): '
               f'решение предварительное, окончательно — на {SDVIG_N_TIKEROV} тикерах × {SDVIG_GODY[0]}–{SDVIG_GODY[1]}')
        if not ok:
            s['pometka'] = tag
    r = dict(ryad=name, h=int(h), hash=(design_hash(design)[:16] if design is not None else ''), **s, vremya=time.strftime('%F %T'))
    # O (PR #178): каждый проверенный ряд — через r2_loyo.sdvig_zapis (пишет блоки и «плоский профиль»; допуск ε — у O)
    O.sdvig_zapis(r, block_file or (O_DIR / 'blok_sdvig.csv'))
    if not r['ok']:
        log(f'БЛОК сдвига: {name} h={h} argmax={r["argmax"]} K={r["posledniy_izvestnyy_den"]} {r["prichina"]}')
    return r


# ================================================================ детектор утечек (как Д26 / ВЫР-1, только 2017–2021)
def _num_cols(A):
    return [c for c in A.columns if c != 'year' and not c.startswith('_') and pd.api.types.is_float_dtype(A[c])]


def perturb_frame(A, cut, seed=20261026):
    """Копия A: все числовые входы с датой > cut × U(0,5; 2); цели rv_target_h{h}, чьё окно [d+1, d+h] заходит за срез, — тоже."""
    rng = np.random.default_rng(seed)
    cut = pd.Timestamp(cut)
    B = A.copy()
    dt = pd.to_datetime(B['date'])
    fut = (dt > cut).values
    num = _num_cols(B)
    B.loc[fut, num] = B.loc[fut, num].values * rng.uniform(0.5, 2.0, size=(int(fut.sum()), len(num)))
    for c in [c for c in num if c.startswith('rv_target_h')]:
        h = int(c.split('_h')[-1])
        cross = (~fut) & (_tend(B, h) > cut).fillna(True).values
        B.loc[cross, c] = B.loc[cross, c].values * rng.uniform(0.5, 2.0, size=int(cross.sum()))
    return B


def plant_frame(A):
    """Подложенная утечка: все числовые входы (не цели) строки d заменены значениями строки d+1 того же тикера."""
    B = A.sort_values(['ticker', 'date']).copy()
    cols = [c for c in _num_cols(B) if not c.startswith('rv_target')]
    nxt = B.groupby('ticker')[cols].shift(-1)
    B[cols] = nxt.fillna(B[cols])
    return B.sort_index()


def _diff(a, b, cut, kmax):
    """max |Δ ln| на строках ≤ среза (только модели шагов ≤ kmax: их обучение кончилось до среза) и после среза."""
    on = [c for c in ('date', 'ticker', 'step', 'part') if c in a.columns and c in b.columns]
    m = a[on + ['rv_pred']].merge(b[on + ['rv_pred']], on=on, suffixes=('', '_p'))
    d = np.abs(np.log(m['rv_pred'].values) - np.log(m['rv_pred_p'].values))
    p = (pd.to_datetime(m['date']) <= cut).values
    if 'step' in m:
        p &= (m['step'].values <= kmax)
    return (float(d[p].max()) if p.any() else 0.0), (float(d[~p].max()) if (~p).any() else float('nan')), int(p.sum())


def leak_check(build, A, name, cuts=CUTS_MID + CUTS_YE, seed=20261026, out=None):
    """build(A) -> {ряд: DataFrame(date, ticker, step, rv_pred[, part])}. Сравниваются строки ≤ среза у моделей шагов k ≤ kmax:
      середина года (kmax: vy ≤ год среза) — Δ = 0, после среза Δ > 0, с подложенной утечкой Δ(≤ среза) > 0 (детектор её видит);
      граница года 31.12 (kmax: vy ≤ год среза + 1) — Δ = 0: проверка эмбарго; ряд должен содержать прогнозы модели шага
      vy = год среза + 1 на строках ≤ среза (part = 'tr'/'all', «на всех строках»). Срезы ≤ 2021; строки > 2021 не читаются."""
    cuts = [pd.Timestamp(c) for c in cuts]
    if max(cuts).year > MAX_YEAR:
        raise AssertionError('срез детектора после 2021')
    base = build(A)
    planted = None
    rows = []
    for i, cut in enumerate(cuts):
        ye = cut.month == 12 and cut.day == 31
        kmax = cut.year - 2015 + (1 if ye else 0)
        Ap = perturb_frame(A, cut, seed + i)
        pert = build(Ap)
        if not ye:
            planted = planted or build(plant_frame(A))
            plantp = build(plant_frame(Ap))
        for r in base:
            past, fut, n = _diff(base[r], pert[r], cut, kmax)
            row = dict(modul=name, ryad=r, srez=str(cut.date()), tip='граница года' if ye else 'середина года', kmax=kmax, strok_do=n,
                       max_dln_do=past, max_dln_posle=fut)
            if ye:
                row['ok'] = bool(past == 0.0 and n > 0)
            else:
                p1, _, _ = _diff(planted[r], plantp[r], cut, kmax)
                row['podlog_max_dln_do'] = p1
                row['ok'] = bool(past == 0.0 and n > 0 and (fut > 0 or np.isnan(fut)) and p1 > 0)
            rows.append(row)
    R = pd.DataFrame(rows)
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        R.to_csv(out, index=False)
    return R


# ================================================================ фазы
GATE_PREV = {'C': 'B', 'D': 'C', 'E': 'D'}


def check_plan():
    got = sha_file(PLAN_PATH)
    if got != PLAN_SHA:
        raise AssertionError(f'factor_plan_v1.csv: SHA {got[:16]}… ≠ {PLAN_SHA[:16]}… (протокол v3, п. 3.2)')
    return True


def require_gate(phase, h):
    """Фаза C/D/E стартует только по файлу ворот предыдущей фазы; SHA входов из поля `inputs` ({путь: sha256}) сверяются."""
    if phase in ('A', 'B'):
        check_plan()
        return {}
    prev = GATE_PREV[phase]
    f = O_DIR / f'gate_{prev}_h{h}.yaml'
    if not f.exists():
        raise AssertionError(f'нет ворот {f} — фаза {phase} h={h} не стартует')
    G = yaml.safe_load(open(f)) or {}
    for p, want in (G.get('inputs') or {}).items():
        q = Path(p) if Path(p).is_absolute() else MAC_ROOT / p
        if sha_file(q) != want:
            raise AssertionError(f'ворота {f.name}: вход {p} изменился после решения')
    return G


def _phase_task(t):
    fn = SHARED['_phase_fn']
    d, seed, steps, h, root = t['design'], t['seed'], t['steps'], t['h'], t['root']
    for k in steps:
        assert_dev_step(k)
    out = fn(d, seed, steps)
    A = SHARED.get('A')
    rows = []
    for name, V in out.items():
        V = V[V['step'].isin(list(steps))]
        p = write_series(d, name, V, h, seed, module=t['module'], root=root)
        sc = shift_check(V, A, h, name, d, block_file=t.get('block_file')) if A is not None else {'ok': None}
        rows.append(dict(ryad=name, h=h, seed=seed, hash=p.parent.name, fayl=p.name, stroki=len(V), sdvig_ok=sc.get('ok'),
                         argmax=sc.get('argmax'), sdvig_okonchatelno=sc.get('okonchatelno'), sdvig_pometka=sc.get('pometka', '')))
    _atomic_write(done_path(d, t['module'], h, seed, root), time.strftime('%F %T').encode())
    return rows


def done_path(design, module, h, seed, root=None):
    """Отметка готовности — своя у каждого модуля (цепочка L -> B -> M -> E пишет в один каталог дизайна)."""
    return cache_dir(design, root=root) / f'DONE_{module}_h{int(h)}_s{seed}'


def run_phase(phase, h, designs, fn, seeds=(42,), steps=STEPS_DEV, workers=None, module='core', root=None, block_file=None,
              check_gate=True):
    """Фаза по списку дизайнов: ворота -> валидация и кэш -> пул (fn в SHARED) -> запись, проверка сдвига, DONE. Готовое пропускается.
    Данные (A и прочее) кладутся в SHARED вызывающим ДО run_phase (share(A=A, ...))."""
    if check_gate:
        require_gate(phase, h)
    share(_phase_fn=fn)
    tasks = []
    for d in designs:
        full = validate(d)
        if int(full['h']) != int(h):
            raise DesignError(f'дизайн {full.get("design")}: h={full["h"]} ≠ {h}')
        cd = cache_dir(full, root=root)
        for s in seeds:
            if done_path(full, module, h, s, root).exists():
                continue
            tasks.append(dict(design=full, seed=s, steps=list(steps), h=int(h), root=root, module=module, block_file=block_file,
                              cost=complexity(full) + 1))
    log(f'фаза {phase} h={h}: дизайнов {len(designs)}, задач {len(tasks)} (готовое пропущено)')
    res = run_pool(_phase_task, tasks, workers, label=f'фаза {phase} h={h}')
    rows = []
    for r in res:
        if r['ok']:
            rows += r['result']
        else:
            rows.append(dict(ryad='ОШИБКА', h=h, seed=r['task']['seed'], hash=design_hash(r['task']['design'])[:16],
                             oshibka=r['error'].strip().splitlines()[-1]))
    T = pd.DataFrame(rows)
    S_DIR.mkdir(parents=True, exist_ok=True)
    if root is None:
        T.to_csv(S_DIR / f'phase_{phase}_h{h}_{time.strftime("%Y%m%d_%H%M%S")}.csv', index=False)
    return T


# ================================================================ дым (Мак): SBER, LKOH, AFLT × шаги 1–3 × seed 42 × h ∈ {1, 5, 22}
SMOKE = CFG.get('smoke', {})
SMOKE_TICKERS = tuple(SMOKE.get('tickers', ('SBER', 'LKOH', 'AFLT')))
SMOKE_STEPS = tuple(SMOKE.get('steps', (1, 2, 3)))


def _harj_ln(A, harj, h, steps, emb=True):
    """HAR-J (МНК на ln) по шагам: {k: ln-прогноз на всех строках A}; как r2_nogi.fam_harj, эмбарго через train_mask."""
    from sklearn.linear_model import LinearRegression
    import gibrid_har as GH
    H = GH.clean(np.log(A[harj].values.clip(1e-10)))
    y = np.log(A[f'rv_target_h{h}'].values + 1e-10)
    out = {}
    for k in steps:
        tp = train_mask(A, h, step_years(k)[0], emb=emb)
        out[k] = LinearRegression().fit(H[tp], y[tp]).predict(H)
    return out


def smoke_fn(design, seed, steps):
    """Модуль дыма для run_phase: нога HAR-J d0 + эмбарго (leg.type = HARJ, boost.role = none). Детерминирован (seed не влияет)."""
    return {'HARJ_d0': _smoke_series(SHARED['A'], SHARED['harj'], int(design['h']), steps)}


def _smoke_series(A, harj, h, steps, emb=True):
    L = _harj_ln(A, harj, h, steps, emb=emb)
    V = []
    for k in steps:
        vy = step_years(k)[0]
        m = val_mask(A, h, vy)
        V.append(pd.DataFrame({'date': A['date'].values[m], 'ticker': A['ticker'].values[m].astype(str), 'year': vy, 'step': k,
                               'rv_actual': A[f'rv_target_h{h}'].values[m], 'rv_pred': np.clip(np.exp(L[k][m]), 1e-10, None)}))
    return pd.concat(V, ignore_index=True)


def _smoke_pool_task(t):
    if t['h'] < 0:
        raise RuntimeError('намеренная ошибка задачи (проверка изоляции)')
    return float(np.log(_smoke_series(SHARED['A'], SHARED['harj'], t['h'], SMOKE_STEPS).rv_pred).sum())


def _smoke_build(emb, h):
    """Для детектора: прогнозы моделей шагов дыма «на всех строках» годов ≤ vy (part = all) — нужно для проверки эмбарго."""
    def build(A):
        harj = SHARED['harj']
        L = _harj_ln(A, harj, h, SMOKE_STEPS, emb=emb)
        fr = []
        for k, ln in L.items():
            vy = step_years(k)[0]
            m = (A['year'].values <= vy)
            fr.append(pd.DataFrame({'date': A['date'].values[m], 'ticker': A['ticker'].values[m], 'step': k, 'part': 'all',
                                    'rv_pred': np.exp(ln[m])}))
        return {f'HARJ_d0{"" if emb else "_noemb"}': pd.concat(fr, ignore_index=True)}
    return build


def _schema_selftest():
    rows = []
    ok = lambda name, cond: rows.append(dict(proverka=name, ok=bool(cond)))
    base = {'design': 'smoke', 'h': 5}
    full = validate(base)
    ok('default дописаны (boost.role=none, features=[G0])', get(full, 'boost.role') == 'none' and full['features'] == ['G0'])
    ok('имя не входит в хэш', design_hash(base) == design_hash({'design': 'другое', 'h': 5}))
    ok('явные default = пропущенные', design_hash(base) == design_hash({'h': 5, 'seed': 42, 'boost': {'role': 'none'}}))
    ok('1 == 1.0 в хэше', design_hash({'h': 5, 'boost': {'shrink': 1}}) == design_hash({'h': 5, 'boost': {'shrink': 1.0}}))
    ok('ключи с точками = вложенные', design_hash({'h': 5, 'boost.role': 'residual'}) == design_hash({'h': 5, 'boost': {'role': 'residual'}}))
    ok('разные значения — разный хэш', design_hash(base) != design_hash({'h': 5, 'boost': {'role': 'residual'}}))
    for bad, why in (({'h': 7}, 'h вне списка'), ({'h': 5, 'boost': {'role': 'x'}}, 'значение вне списка'),
                     ({'h': 5, 'foo': 1}, 'неизвестный ключ'), ({'h': 5, 'data': {'embargo': 0}}, 'fixed'),
                     ({'h': 5, 'features': ['G1']}, 'без G0'), ({}, 'нет h'),
                     ({'h': 5, 'selection': {'sigma': {'mult': 3}}}, 'fixed-константа')):
        try:
            validate(bad)
            ok(f'ошибка схемы ловится: {why}', False)
        except DesignError:
            ok(f'ошибка схемы ловится: {why}', True)
    d = {'h': 1, 'boost': {'role': 'feature', 'monotone': 'on', 'early_stop': 'qlike_total'}, 'features': ['G0', 'G2', 'G3'],
         'calib': {'kappa': [0.5, 'fit']}}
    ok('сложность = 6 (role, monotone, early_stop, G2, G3, kappa-сетка)', complexity(d) == 6)
    ok('ранняя остановка на году оценки (роль ≠ none, early_stop = native) — DesignError; хэш считается',
       _raises(lambda: validate({'h': 5, 'boost': {'role': 'residual'}})) and len(design_hash({'h': 5, 'boost': {'role': 'residual'}})) == 64)
    ok('сетка внутри складки: calib.kappa (+ enet 3 × 3 по умолчанию)', 'calib.kappa' in grid_keys(d) and len(variants(d)) == 18
       and len(variants(d, keys=['calib.kappa'])) == 2)
    ok('сетка enet по умолчанию — вся (3 × 3)', len(variants({'h': 5})) == 9)
    ok('план: SHA factor_plan_v1.csv', check_plan())
    # режим R2: прогнозы ≥ 2022 не пишутся (любой part), флаг — только A и только при воротах E
    fut = pd.DataFrame({'date': pd.to_datetime(['2021-12-30', '2022-01-03']), 'ticker': 'SBER', 'year': [2021, 2022], 'step': 6,
                        'rv_pred': [1e-4, 1e-4]})
    for part in ('val', 'test'):
        try:
            write_series({'h': 5}, 'zapret', fut, 5, 42, part=part, root=S_DIR / 'smoke' / 'cache_zapret')
            ok(f'режим R2: запись ≥ 2022 (part={part}) запрещена', False)
        except PermissionError:
            ok(f'режим R2: запись ≥ 2022 (part={part}) запрещена', True)
    for who, gate in (('B', __file__), ('A', S_DIR / 'нет_ворот.yaml')):
        try:
            enable_test_assembly(who, gate)
            ok(f'флаг сборки: отказ ({who}, {Path(gate).name})', False)
        except PermissionError:
            ok(f'флаг сборки: отказ ({who}, {Path(gate).name})', True)
    ok('шаг 7 в разработке запрещён', _raises(lambda: assert_dev_step(7)))
    # входы h=1 (O, PR #172): base_1 / leg_1 входят в хэш (идентичность + SHA содержимого), описания — нет
    dp = {'h': 5, 'mh': {'persist': 'qlike'}}
    i1 = {'base_1': {'file': 'R2/cache/VYR/F_d0_h1_s{seed}_val.parquet', 'ryad': 'F_1(d0)'},
          'leg_1': {'file': 'vyr/etalon_d0_h1_val.parquet', 'col': 'har_corr'}}
    i2 = {'base_1': dict(i1['base_1'], ryad='другое описание'), 'leg_1': i1['leg_1']}
    i3 = {'base_1': {'design_sha': '0' * 64, 'series': 'B_feature_h1'}, 'leg_1': i1['leg_1']}
    hs_ = [design_hash(x) for x in (dp, with_inputs(dp, i1), with_inputs(dp, i2), with_inputs(dp, i3))]
    ok('входы h=1 в хэше: без входов ≠ с F_1(d0) ≠ с ячейкой C; описание входа хэш не меняет',
       needs_h1_inputs(dp) and hs_[0] != hs_[1] and hs_[1] == hs_[2] and hs_[1] != hs_[3])
    return rows


def _raises(fn):
    try:
        fn()
        return False
    except (AssertionError, PermissionError, ValueError):
        return True


def cmd_smoke(a):
    t0 = time.time()
    out_dir = S_DIR / 'smoke'
    root = out_dir / 'cache'
    blk = out_dir / 'blok_sdvig.csv'
    if blk.exists():
        blk.unlink()
    rows = [dict(blok='схема', **r) for r in _schema_selftest()]
    if FACTOR_DIR.exists():
        ok_o, rows_o, mism = check_o(verbose=False)
        rows.append(dict(blok='схема', proverka=f'дизайны O factor_v1 (3 × 64): проходят схему, хэши уникальны, сложность = r2_loyo.slozhnost; '
                                                 f'нормализация = r2_loyo._n (расхождений {len(mism)})', ok=bool(ok_o)))
    log(f'схема: {sum(r["ok"] for r in rows)}/{len(rows)}')
    A, harj = load_d0(SMOKE_TICKERS)
    check_d0(A)
    rows.append(dict(blok='данные', proverka=f'd0: rv_d = RV дня d; тикеры {sorted(A.ticker.unique())}; max год {A.year.max()}',
                     ok=bool(set(A.ticker.unique()) == set(SMOKE_TICKERS) and A.year.max() <= MAX_YEAR)))
    tr, va, full_tr = inner_masks(A, 5, 2018)
    tend5 = _tend(A, 5)
    rows.append(dict(blok='данные', proverka='эмбарго h=5: окно цели обучения кончается до 1 янв. vy; внутренняя складка — тоже',
                     ok=bool((tend5[full_tr] < '2018-01-01').all() and (tend5[tr] < '2017-01-01').all() and (tend5[va] < '2018-01-01').all()
                             and not (train_mask(A, 5, 2018) & (A.year.values >= 2018)).any())))
    share(A=A, harj=harj)
    import r2_nogi as RN  # положительный контроль проверки сдвига: та же нога на регрессорах дня d−1 (версия R1)
    A1, harj1 = RN.load(FEAT, d0=False)
    A1['date'] = pd.to_datetime(A1['date'])
    A1 = A1[(A1.year <= MAX_YEAR) & A1.ticker.isin(SMOKE_TICKERS)].reset_index(drop=True)
    designs = {h: {'design': f'smoke_HARJ_d0_h{h}', 'h': h, 'leg': {'type': 'HARJ'}, 'boost': {'role': 'none'}} for h in HS}
    for h in HS:
        for f in cache_dir(designs[h], root=root).glob('DONE_*'):
            f.unlink()  # дым всегда пересчитывает
        T = run_phase('A', h, [designs[h]], smoke_fn, seeds=(42,), steps=SMOKE_STEPS, workers=a.workers, module='S_smoke',
                      root=root, block_file=blk, check_gate=False)
        V = read_series(designs[h], 'HARJ_d0', h, 42, root=root)
        rows.append(dict(blok='фаза', proverka=f'h={h}: ряд в кэше {cache_dir(designs[h], root=root).name}, шаги '
                                                f'{sorted(map(int, V.step.unique()))}, тикеров {V.ticker.nunique()}, сдвиг argmax={T.argmax.iloc[0]}',
                         ok=bool(len(T) == 1 and T.sdvig_ok.iloc[0] and sorted(V.step.unique()) == list(SMOKE_STEPS))))
        V1 = _smoke_series(A1, harj1, h, SMOKE_STEPS)  # обязан блокироваться (argmax = −1)
        sc1 = shift_check(V1, A, h, f'kontrol_R1_dminus1_h{h}', block_file=blk)
        rows.append(dict(blok='сдвиг', proverka=f'h={h}: ряд на d−1 при K={{0}} блокируется (argmax={sc1["argmax"]}; '
                                                f'corr −1/0 = {sc1["corr_s-1"]:.4f}/{sc1["corr_s+0"]:.4f})',
                         ok=bool(sc1['argmax'] == -1 and not sc1['ok'])))
        # правило O (PR #167): тот же ряд при K = {−1, 0} (лаг публикации, как G1) проходит; заглядывание +1 блокируется при любом K
        import r2_loyo as O
        Kg1 = O.K_dizaina(validate({'h': h, 'features': ['G0', 'G1']}))
        sc2 = shift_check(V1, A, h, f'kontrol_R1_dminus1_K_G1_h{h}', block_file=blk, K=Kg1)
        D_ = A[['date', 'ticker', 'rv']].sort_values(['ticker', 'date'])
        nxt = D_.assign(rv_pred=D_.groupby('ticker')['rv'].shift(-1).clip(lower=1e-10)).dropna(subset=['rv_pred'])
        Vp = V[['date', 'ticker', 'step']].merge(nxt[['date', 'ticker', 'rv_pred']], on=['date', 'ticker'])
        sc3 = shift_check(Vp, A, h, f'kontrol_zaglyadyvanie_h{h}', block_file=blk, K=Kg1)
        rows.append(dict(blok='сдвиг', proverka=f'h={h}: K(G0+G1) = {Kg1}: ряд на d−1 проходит (argmax={sc2["argmax"]}); '
                                                f'ряд = RV_{{d+1}} блокируется (argmax={sc3["argmax"]}, {sc3["prichina"]})',
                         ok=bool(tuple(Kg1) == (-1, 0) and sc2['ok'] and sc3['argmax'] == 1 and not sc3['ok'])))
        # load_series: по хэшу дизайна те же ряды, что read_series; фильтр по модулю
        L1 = load_series(design_hash(designs[h]), root=root, modul='S_smoke')
        L0 = load_series(design_hash(designs[h])[:16], root=root, modul='чужой')
        rows.append(dict(blok='кэш', proverka=f'h={h}: load_series(design_sha) = read_series ({len(L1)} строк, ряды {sorted(L1.ryad.unique())}); '
                                              f'фильтр modul', ok=bool(len(L1) == len(V) and L0.empty and np.allclose(
                             L1.sort_values(KEY).rv_pred.values, V.sort_values(KEY).rv_pred.values, rtol=0, atol=0))))
        # детектор: «будущее не влияет» и эмбарго; отрицательный контроль — без эмбарго граница года обязана «протечь»
        cuts = ('2017-06-30', '2017-12-31') if a.fast else ('2017-06-30', '2018-06-29', '2017-12-31')
        R = leak_check(_smoke_build(True, h), A, f'smoke_h{h}', cuts=cuts, out=out_dir / f'leak_h{h}.csv')
        rows.append(dict(blok='утечки', proverka=f'h={h}: HAR-J d0 + эмбарго: Δ до среза = {R.max_dln_do.max():.1e}, после > 0, '
                                                  f'подлог пойман (min Δ = {R.podlog_max_dln_do.min():.1e})', ok=bool(R.ok.all())))
        Rn = leak_check(_smoke_build(False, h), A, f'smoke_noemb_h{h}', cuts=('2017-12-31',))
        rows.append(dict(blok='утечки', proverka=f'h={h}: без эмбарго граница года ловится (Δ до среза = {Rn.max_dln_do.max():.2e})',
                         ok=bool(Rn.max_dln_do.max() > 0)))
    # пул: fork, данные из SHARED без копирования, ошибка одной задачи не роняет остальные, результат = последовательному
    res = run_pool(_smoke_pool_task, [dict(h=h, cost=h) for h in HS] + [dict(h=-1, cost=0)], workers=a.workers, label='пул дыма')
    good = {r['task']['h']: r['result'] for r in res if r['ok']}
    seq = {h: float(np.log(_smoke_series(A, harj, h, SMOKE_STEPS).rv_pred).sum()) for h in HS}
    rows.append(dict(blok='пул', proverka=f'fork × {min(n_workers(a.workers), 4)}: 3 задачи = последовательно, задача с ошибкой изолирована, '
                                           f'pid воркеров {len({r["pid"] for r in res})}',
                     ok=bool(all(good.get(h) == seq[h] for h in HS) and sum(not r['ok'] for r in res) == 1)))
    I = merge_index(root)
    rows.append(dict(blok='кэш', proverka=f'индекс: {len(I)} строк, хэши {sorted(I.hash.unique()) if len(I) else []}', ok=bool(len(I) == len(HS))))
    T = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    T.to_csv(out_dir / 'smoke_core.csv', index=False)
    with pd.option_context('display.max_colwidth', 120, 'display.width', 200):
        print(T.to_string(index=False))
    log(f'дым r2_core: {"ВСЕ OK" if T.ok.all() else "ЕСТЬ ОШИБКИ"} за {time.time() - t0:.0f} с -> {out_dir / "smoke_core.csv"}')
    return 0 if T.ok.all() else 1


FACTOR_DIR = REPO / 'models' / 'designs' / 'r2' / 'factor_v1'


def check_o(verbose=True):
    """Сверка с агентом O: (1) все дизайны factor_v1/h{h}.yaml проходят validate; (2) complexity = r2_loyo.slozhnost = meta.slozhnost;
    (3) правило нормализации совпадает с r2_loyo._n: для каждого листа и каждого значения (из схемы, из дизайнов O и пробных строк
    on/ON/True/yes/off/'0.5'/'3'/3.0) принадлежность списку схемы одинакова. -> (ok, строки отчёта)."""
    import r2_loyo as O
    S = load_schema()
    rows, ok = [], True
    probes = ['on', 'ON', 'On', 'True', 'yes', 'off', 'OFF', 'False', 'no', '0.5', '3', 3.0, 1.0, '1', 'inf', 'x']
    for h in HS:
        f = FACTOR_DIR / f'h{h}.yaml'
        Y = yaml.safe_load(open(f))
        D, M = Y['dizainy'], Y['meta']
        hashes, bad_v, bad_c = [], [], []
        for d, m in zip(D, M):
            try:
                hashes.append(design_hash(d))
            except DesignError as e:
                bad_v.append((d.get('design'), str(e)))
                continue
            c, co = complexity(d), O.slozhnost(d)
            if not (c == co == m['slozhnost']):
                bad_c.append((d.get('design'), c, co, m['slozhnost']))
        r = dict(h=h, dizainov=len(D), proshli_skhemu=len(D) - len(bad_v), unikalnyh_heshey=len(set(hashes)),
                 slozhnost_sovpala=len(D) - len(bad_v) - len(bad_c), slozhnost_min=min(m['slozhnost'] for m in M),
                 slozhnost_max=max(m['slozhnost'] for m in M))
        r['ok'] = not bad_v and not bad_c
        ok &= r['ok']
        rows.append(r)
        for x in (bad_v + bad_c)[:5]:
            print('  расхождение:', x)
    mism = []
    for p, L in S.leaves.items():
        if L.nvalues is None or L.kind == 'const':
            continue
        spec = O.SPECS.get(L.dotted)
        if spec is None or 'values' not in spec:
            continue
        cand = list(L.values) + probes
        for h in HS:
            for d in yaml.safe_load(open(FACTOR_DIR / f'h{h}.yaml'))['dizainy']:
                v = _flatten(d).get(p)
                if v is not None and not isinstance(v, list):
                    cand.append(v)
        for x in cand:
            if isinstance(x, (list, dict)):
                continue
            mine = _norm(_native(_yaml_scalars(x))) in L.nvalues
            theirs = O._n(x) in list(map(O._n, spec['values']))
            if mine != theirs:
                mism.append((L.dotted, x, mine, theirs))
    rows.append(dict(h='все', proverka='нормализация = r2_loyo._n', rasxozhdeniy=len(mism), ok=not mism))
    ok &= not mism
    # schema_v2: хэши дизайнов v1 не меняются (factor_v1 3 × 64 и driver_v1 без ключей v2)
    allD = [d for h in HS for d in yaml.safe_load(open(FACTOR_DIR / f'h{h}.yaml'))['dizainy']]
    ddir = REPO / 'models' / 'designs' / 'r2' / 'driver_v1'
    for f in sorted(ddir.glob('*.yaml')) if ddir.exists() else []:
        Y = yaml.safe_load(open(f))
        allD += [d for d in (Y.get('designs') or []) if not any(k in str(d) for k in ('grid_point', 'k16', 'k18', 'k20', 'row_frac'))]
    n, bad = check_v2_invariance(allD)
    rows.append(dict(h='все', proverka=f'schema_v2: хэш v1 = хэш v2 у {n} дизайнов (factor_v1 + driver_v1)', rasxozhdeniy=len(bad), ok=not bad))
    ok &= not bad
    for x in bad[:5]:
        print('  хэш v1 ≠ v2:', x)
    if verbose:
        print(pd.DataFrame(rows).to_string(index=False))
        for x in mism[:10]:
            print('  нормализация расходится:', x)
    return ok, rows, mism


def cmd_check_o(a):
    ok, _, _ = check_o()
    log('сверка с O: ' + ('OK' if ok else 'ЕСТЬ РАСХОЖДЕНИЯ'))
    return 0 if ok else 1


def cmd_show(a):
    full = validate(a.design)
    print(yaml.safe_dump(full, sort_keys=True, allow_unicode=True, default_flow_style=False))
    print('hash', design_hash(full))
    print('cache', cache_dir(full, create=False))
    print('complexity', complexity(full), 'grid', grid_keys(full))
    return 0


def main():
    ap = argparse.ArgumentParser(description='R2 S: основа конвейера (схема, кэш, пул, d0, эмбарго, сдвиг, утечки, фазы)')
    sp = ap.add_subparsers(dest='cmd', required=True)
    p = sp.add_parser('smoke')
    p.add_argument('--workers', type=int, default=3)
    p.add_argument('--fast', action='store_true', help='один срез середины года вместо двух')
    p = sp.add_parser('show')
    p.add_argument('design')
    sp.add_parser('index')
    sp.add_parser('check-o')
    a = ap.parse_args()
    if a.cmd == 'check-o':
        return cmd_check_o(a)
    if a.cmd == 'smoke':
        return cmd_smoke(a)
    if a.cmd == 'show':
        return cmd_show(a)
    I = merge_index()
    print(f'INDEX_R2.csv: {len(I)} строк')
    return 0


if __name__ == '__main__':
    sys.exit(main())
