"""Дизайн из B_selected.csv (tune_B.py) или selected_*.csv (models tune): гиперпараметры по (модель, h, шаг), схема гибрида —
по правилу 25.10 (gibrid_v17). Точки B берутся из сетки tune_B.make_grid (исполняется через ast), а не из CSV: CSV pandas
обрезает float до 15-16 знаков, и lightgbm (double) даёт другие прогнозы (до 1e-5 по RV) — ворота 1e-9 не проходятся."""
import ast
import numpy as np, pandas as pd
import yaml
from . import config as C

HEAD = {'B': '# протокол B (tune_B.py): ES по l2 на ln(rv), patience 50, max_bin по умолчанию библиотеки — ворота совпадения с B_*',
        'default': '# конвейер по умолчанию: ES по QLIKE (patience 30), max_bin 63'}


def grid_B():
    """= tune_B.GRID (ЛГК 100, seed 42) без импорта tune_B (импорт грузит данные)."""
    ns = {'np': np, 'pd': pd}
    for n in ast.parse(C.TUNE_B.read_text()).body:
        names = {x.id for t in getattr(n, 'targets', []) for x in ast.walk(t) if isinstance(x, ast.Name)}
        if names & {'N_PTS', 'SPACE'} or (isinstance(n, ast.FunctionDef) and n.name in ('lhs', 'make_grid')):
            exec(compile(ast.Module([n], []), 'tune_B', 'exec'), ns)
    return ns['make_grid']().set_index(['model', 'point'])


def params_from_row(r, model, grid=None):
    out = {}
    src = grid.loc[(model, int(r.point))] if grid is not None else None
    for k, lo, hi, sc in C.space_B()[model]:
        v = src[k] if src is not None else getattr(r, k)
        out[k] = int(round(v)) if sc in ('int', 'logint') else float(v)
    return out


def parse_target(s):
    return {int(a): b for a, b in (x.split('=') for x in s.split(','))} if s else None


def from_selected(csv, name, train='B', max_bin=None, features=None, target=None):
    sel = pd.read_csv(csv)
    grid = grid_B() if 'point' in sel.columns else None
    models = {}
    for (m, h), g in sel.groupby(['model', 'h']):
        models.setdefault(m, {})[f'h{int(h)}'] = {int(r.step): params_from_row(r, m, grid) for r in g.itertuples()}
    tr = dict(C.B_TRAIN if train == 'B' else C.DEFAULT_TRAIN)
    if max_bin is not None:
        tr['max_bin'] = None if str(max_bin) == 'none' else int(max_bin)
    d = {'name': name, 'seed': 42, 'horizons': list(C.HS), 'source': str(csv).replace(str(C.ROOT) + '/', ''), 'train': tr}
    if features or target:
        d['data'] = {k: v for k, v in (('features', features), ('target', parse_target(target))) if v}
    d['hybrid'] = {'variants': ['single'], 'select': 'правило 25.10 (мин. нарушенных c1-c4, затем мин. средний val-QLIKE 2016-2021)',
                   'v7_lag': 'h', 'ref': True}
    d['models'] = models
    p = C.ROOT / 'models/designs' / f'{name}.yaml'
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(HEAD[train] + '\n' + yaml.safe_dump(d, allow_unicode=True, sort_keys=False, width=200))
    return p
