"""Константы, пути, переиспользуемые функции существующего кода и загрузка yaml дизайна.
qlike / clean_features / qlike_losses / dm_test берутся из walk_forward_full.py через ast (как в tune_B.py: скрипт исполняется
на уровне модуля), пространство поиска — SPACE из tune_B.py (ast.literal_eval), гибрид и HAR-J — модули scripts/analysis/gibrid_*."""
import ast, hashlib, os, sys
from pathlib import Path
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
MV = ROOT / 'moex_volatility'
ANALYSIS = ROOT / 'scripts/analysis'
# правило владельца 25.10 18:20: данные, кэши и артефакты — только в общем data/ основного репо (копии агентов — только код)
MAIN = Path(os.environ.get('MODELS_MAIN', '/Users/nilysenok/Desktop/Старое/PhD_Vol_trading'))
DATA_ROOT = MAIN / 'moex_volatility/data/prepared'
CACHE = MAIN / 'data/models_cache'
RUN = MAIN / 'data/run'
FINAL = ROOT / 'docs/otchety/FINAL_DESIGN.md'
TABLES = ROOT / 'docs/tables'
HS, MODELS = (1, 5, 22), ('xgb', 'lgbm')
YEARS = tuple(range(2017, 2026))  # тестовые годы шагов 1-9 (walk_forward_full.py, annual)
DEV = 6  # разработочный набор: шаги 1-6 (val 2016-2021); подтверждающий: 7-9 (val 2022-2024) — только --confirm/--finalists
WFF = MV / 'scripts/walk_forward_full.py'
TUNE_B = MV / 'scripts/tune_B.py'
# протокол B (tune_B.py): ES по l2 на ln(rv), patience 50, max_bin по умолчанию библиотеки (lgb 255, xgb 256)
B_TRAIN = {'n_estimators': 2000, 'es_metric': 'l2', 'patience': 50, 'max_bin': None, 'force_col_wise': False, 'float32': False}
# протокол B v2 (tune_Bv2.py, финальное обучение): models/v2.py
B_V2_TRAIN = {'protocol': 'b_v2', 'n_estimators': 1000, 'es_metric': 'l2', 'patience': 30, 'max_bin': 63, 'force_col_wise': True, 'float32': True,
              'exclude_features': 'data/regen/Bv2_excluded_features.csv'}
# конвейер по умолчанию (задание 25.10): ES по QLIKE, patience 30; max_bin 255 — проверка 63 на шагах 1-6 не прошла (хуже на 1,5% в среднем,
# до 7,5% у lgbm h=22; docs/otchety/2026-10-25_models_pipeline.md); ускорения 25.10 18:25: force_col_wise, float32
DEFAULT_TRAIN = {'n_estimators': 2000, 'es_metric': 'qlike', 'patience': 30, 'max_bin': 255, 'force_col_wise': True, 'float32': True}

if str(ANALYSIS) not in sys.path:
    sys.path.insert(0, str(ANALYSIS))


def _from_wff(names):
    import scipy.stats  # noqa: F401  (нужен dm_test)
    ns = {'np': np, 'scipy': sys.modules['scipy']}
    for n in ast.parse(WFF.read_text()).body:
        if isinstance(n, ast.FunctionDef) and n.name in names:
            exec(compile(ast.Module([n], []), 'wff', 'exec'), ns)
    return [ns[k] for k in names]


qlike, clean_features, qlike_losses, dm_test = _from_wff(('qlike', 'clean_features', 'qlike_losses', 'dm_test'))


def space_B():
    """SPACE из tune_B.py без импорта (импорт tune_B загружает данные): {модель: [(имя, lo, hi, шкала)]}."""
    for n in ast.parse(TUNE_B.read_text()).body:
        if isinstance(n, ast.Assign) and getattr(n.targets[0], 'id', '') == 'SPACE':
            return ast.literal_eval(n.value)
    raise RuntimeError('SPACE не найден в tune_B.py')


def data_path(p):
    """Относительные пути данных в дизайне — от основного репо (MAIN), абсолютные — как есть."""
    return Path(p) if os.path.isabs(str(p)) else MAIN / p


def blob_hash(path):
    """= `git hash-object <path>` (sha1 от 'blob <len>\\0' + содержимое), без вызова git на каждый ключ."""
    b = Path(path).read_bytes()
    return hashlib.sha1(b'blob %d\0' % len(b) + b).hexdigest()


def code_hash(*names):
    """Версии файлов стадии: имена модулей models/*.py и/или абсолютные пути."""
    out = []
    for n in names:
        p = Path(n) if os.path.isabs(str(n)) else ROOT / 'models' / n
        out.append(blob_hash(p))
    return out


def steps(confirm=False, smoke=False):
    if smoke:
        return [1]
    return list(range(1, len(YEARS) + 1)) if confirm else list(range(1, DEV + 1))


def load_design(path):
    d = yaml.safe_load(Path(path).read_text())
    d.setdefault('name', Path(path).stem)
    d.setdefault('seed', 42)
    t = dict(DEFAULT_TRAIN)
    t.update(d.get('train') or {})
    d['train'] = t
    d.setdefault('hybrid', {'v7_lag': 'h'})
    return d


def step_params(design, model, h, step):
    """Гиперпараметры (модель, h, шаг) из дизайна: models.<m>.h<h>.<шаг> или models.<m>.all."""
    m = design['models'][model]
    if 'all' in m:
        return dict(m['all'])
    return dict(m[f'h{h}'][step])
