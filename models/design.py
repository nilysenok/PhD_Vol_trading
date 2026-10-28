"""Дизайн из B_selected.csv (tune_B.py) или selected_*.csv (models tune): гиперпараметры по (модель, h, шаг), схема гибрида —
по правилу 25.10 (gibrid_v17). Точки B берутся из сетки tune_B.make_grid (исполняется через ast), а не из CSV: CSV pandas
обрезает float до 15-16 знаков, и lightgbm (double) даёт другие прогнозы (до 1e-5 по RV) — ворота 1e-9 не проходятся."""
import ast, glob
from pathlib import Path
import numpy as np, pandas as pd
import yaml
from . import config as C

HEAD = {'B': '# протокол B (tune_B.py): ES по l2 на ln(rv), patience 50, max_bin по умолчанию библиотеки — ворота совпадения с B_*',
        'default': '# конвейер по умолчанию: ES по QLIKE (patience 30), max_bin 255 (63 не прошёл проверку), force_col_wise, float32',
        'b_v2': '# протокол B v2 (tune_Bv2.py, финальное обучение на всех строках, точки из Bv2_selected_h*.csv) — ворота совпадения с Bv2_*'}


def grid_B(script=None):
    """= GRID скрипта B (tune_B.py / tune_Bv2.py: ЛГК, seed 42) без импорта (импорт грузит данные)."""
    from types import SimpleNamespace
    ns = {'np': np, 'pd': pd, 'PRE': SimpleNamespace(hs='1,5,22', max_bin=63, tag='B', features=None, target='point')}  # ключи скриптов B
    for n in ast.parse(Path(script or C.TUNE_B).read_text()).body:
        names = {x.id for t in getattr(n, 'targets', []) for x in ast.walk(t) if isinstance(x, ast.Name)}
        if names & {'N_PTS', 'SPACE'} or (isinstance(n, ast.FunctionDef) and n.name in ('lhs', 'grid', 'make_grid')):
            exec(compile(ast.Module([n], []), 'tune_B', 'exec'), ns)
    g = ns['make_grid']()
    return g if g.index.names == ['model', 'point'] else g.set_index(['model', 'point'])


def params_from_row(r, model, grid=None):
    out = {}
    src = grid.loc[(model, int(r.point))] if grid is not None else None
    for k, lo, hi, sc in C.space_B()[model]:
        v = src[k] if src is not None else getattr(r, k)
        out[k] = int(round(v)) if sc in ('int', 'logint') else float(v)
    return out


def parse_target(s):
    return {int(a): b for a, b in (x.split('=') for x in s.split(','))} if s else None


def from_selected(csv, name, train='B', max_bin=None, features=None, target=None, grid_script=None, levers=None, horizons=None, note=None, exclude=None):
    files = sorted(glob.glob(str(C.data_path(csv))))
    sel = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    if 'seed' in sel.columns:  # Bv2_selected_h*: основная строка — seed 42, rank 1
        sel = sel[(sel.seed == 42) & (sel['rank'] == 1)]
    grid = grid_B(grid_script) if 'point' in sel.columns else None
    models = {}
    for (m, h), g in sel.groupby(['model', 'h']):
        models.setdefault(m, {})[f'h{int(h)}'] = {int(r.step): params_from_row(r, m, grid) for r in g.itertuples()}
    tr = dict({'B': C.B_TRAIN, 'b_v2': C.B_V2_TRAIN}.get(train, C.DEFAULT_TRAIN))
    if max_bin is not None:
        tr['max_bin'] = None if str(max_bin) == 'none' else int(max_bin)
    if exclude:
        tr['exclude_features'] = exclude
    d = {'name': name, 'seed': 42, 'horizons': [int(x) for x in horizons.split(',')] if horizons else list(C.HS), 'source': str(csv).replace(str(C.MAIN) + '/', ''), 'train': tr}
    if features or target:
        d['data'] = {k: v for k, v in (('features', features), ('target', parse_target(target))) if v}
    d['hybrid'] = {'variants': ['single'], 'select': 'правило 25.10 (мин. нарушенных c1-c4, затем мин. средний val-QLIKE 2016-2021)',
                   'v7_lag': 'h', 'ref': True}
    if levers:
        d['levers'] = {x: True for x in levers.split(',')}
    d['models'] = models
    p = C.ROOT / 'models/designs' / f'{name}.yaml'
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(HEAD[train] + '\n' + (f'# {note}\n' if note else '') + yaml.safe_dump(d, allow_unicode=True, sort_keys=False, width=200))
    return p
