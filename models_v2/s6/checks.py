"""Стадии check (A13, A16) и audit (A14); проверка состава INVENTORY (A1) — inventory_problems."""
from __future__ import annotations

import ast
import json
import re
import subprocess

import pandas as pd

from models_v2.s6 import paths6

REQUIRED_ID = ("harj_b", "fx_b2_g132_b", "b2_f4", "c1_opt", "c2_h3_roll_f4", "c3_f3calm_rv", "c3_f3calm_g4", "e3_roll")
REQUIRED_COLS = ("id", "model_class", "library", "objective", "target", "features", "hyperparameters", "window", "retrain_freq", "early_stopping", "seed",
                 "forecast_path", "forecast_sha256", "code_path", "code_sha256", "trial_id", "status", "eval_selected", "qlike_recomputed")
REQUIRED_FAMILY = ("OBJECTIVE_EXPERIMENTS", "HAR_BOOSTING_RESULTS", "REGIME_MODEL_RESULTS", "FEATURE_ABLATIONS", "WINDOW_RESEARCH", "C_ALL", "ENSEMBLE_RESULTS",
                   "forecasts_pb", "fix_forecasts_a", "fix_forecasts_c")
# ↺ 06.11: семейства Б в forecasts_pb — b1…b5, b7; b6_ в сохранённых прогнозах и коде пилота нет (в GATES_A написано b1…b7)
REQUIRED_PREFIX = ("fx_",) + tuple("b%d_" % i for i in (1, 2, 3, 4, 5, 7))
LIMITS = {"xgboost": (45, 55), "lightgbm": (45, 55)}
_FORBIDDEN = re.compile("|".join(["_te" + "st\\.", "test" + "_locked", "\\." + "env"]))


def inventory_problems(df: pd.DataFrame) -> list:
    """A1: состав INVENTORY (все обязательные id, колонки, семейства, префиксы; нет дублей)."""
    bad = ["нет колонки " + c for c in REQUIRED_COLS if c not in df.columns]
    if bad:
        return bad
    ids = set(df["id"])
    bad += ["нет id " + i for i in REQUIRED_ID if i not in ids]
    bad += ["дубль id: %s" % ", ".join(sorted(df.loc[df["id"].duplicated(), "id"].astype(str)))] if df["id"].duplicated().any() else []
    fam = set(df["family"].astype(str).str.split(";").str[0])
    bad += ["нет семейства " + f for f in REQUIRED_FAMILY if f not in fam]
    bad += ["нет строк с префиксом " + p for p in REQUIRED_PREFIX if not any(str(i).startswith(p) for i in ids)]
    for lib in ("xgb", "lgbm"):
        if int(df["family"].astype(str).str.contains("этап2_top3_" + lib).sum()) != 3:
            bad.append("лучших 3 строки этапа 2 для " + lib)
    return bad


def a13() -> list:
    bad = []
    for alg, (o, h) in LIMITS.items():
        d = paths6.REP / alg
        if len([x for x in (d / "OPTUNA_TRIALS.csv").read_text().splitlines() if x.strip()]) != 1:
            bad.append(alg + ": OPTUNA_TRIALS не пуст")
        ck = json.loads((d / "CHECKPOINT.json").read_text())
        if any(ck["used"].values()) or (ck["limits"]["obj_feat"], ck["limits"]["hp"]) != (o, h):
            bad.append(alg + ": CHECKPOINT")
    fs = [p for p in (paths6.ROOT / "models_v2").rglob("*") if p.is_file() and "__pycache__" not in p.parts and "s6" not in p.relative_to(paths6.ROOT / "models_v2").parts[:1]]
    lit = [p for p in fs if p.stat().st_mtime > paths6.ref_time()]
    outside = [str(p) for p in fs if p.stat().st_mtime > paths6.ref_time(paths6.FIX_COMMIT)]
    print("A13: файлов models_v2 вне s6 новее коммита GATES_A (буквально): %d; новее коммита правок D %s: %d %s" % (len(lit), paths6.FIX_COMMIT, len(outside), outside[:5]))
    return bad + (["models_v2 вне s6 изменён"] if outside else [])


def a16() -> list:
    hits = []
    for p in sorted((paths6.ROOT / "models_v2/s6").glob("*.py")):
        for i, ln in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            if _FORBIDDEN.search(ln):
                hits.append("%s:%d" % (p.name, i))
    print("A16: запрещённые шаблоны в models_v2/s6: %d %s" % (len(hits), hits))
    return hits


def audit_files(paths) -> tuple:
    """-> (файлы > 200 строк, функции > 40 строк без пометки «длинная» в docstring)."""
    big, longf = [], []
    for p in paths:
        text = p.read_text(encoding="utf-8")
        if len(text.splitlines()) > 200:
            big.append(p.name)
        for n in ast.walk(ast.parse(text)):
            if isinstance(n, ast.FunctionDef) and n.end_lineno - n.lineno + 1 > 40 and "длинная" not in (ast.get_docstring(n) or ""):
                longf.append("%s::%s" % (p.name, n.name))
    return big, longf


def a14() -> list:
    paths = sorted((paths6.ROOT / "models_v2/s6").glob("*.py")) + sorted((paths6.ROOT / "tests").glob("test_v2_xl_*.py"))
    big, longf = audit_files(paths)
    n = len(subprocess.run(["pgrep", "-f", "run_a.py"], capture_output=True, text=True).stdout.split())
    print("A14: файлов > 200 строк: %d %s; функций > 40 строк: %d %s; процессов run_a: %d (<= 6)" % (len(big), big, len(longf), longf, n))
    return big + longf + (["процессов > 6"] if n > 6 else [])


def stage(a) -> int:
    if a.stage == "audit":
        return int(bool(a14()))
    inv_bad = inventory_problems(paths6.read_csv(paths6.REP / "INVENTORY.csv"))
    print("A1 состав INVENTORY:", inv_bad or "OK")
    bad = a13() + a16()
    print("A13/A16 нарушений:", len(bad), bad)
    return int(bool(bad))
