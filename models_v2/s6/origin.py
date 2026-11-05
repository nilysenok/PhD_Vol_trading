"""R1: происхождение алгоритма строки — по коду (ast), не по названию. Функция обучения -> набор вызовов бустеров/OLS."""
from __future__ import annotations

import ast
import textwrap
from pathlib import Path

from models_v2.s6.paths6 import ROOT

BOOST_CALLS = {("xgb", "train"): "XGBoost", ("lgb", "train"): "LightGBM", ("xgb", "XGBRegressor"): "XGBoost", ("lgb", "LGBMRegressor"): "LightGBM"}
OLS_CALLS = {"lstsq", "solve", "pinv"}


def scan(src: str) -> list:
    """[(метка, строка вызова)] по исходнику: xgb.train / lgb.train / lstsq (OLS)."""
    out = []
    for n in ast.walk(ast.parse(src)):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute):
            a, v = n.func.attr, n.func.value
            if isinstance(v, ast.Name) and (v.id, a) in BOOST_CALLS:
                out.append((BOOST_CALLS[(v.id, a)], n.lineno))
            elif a in OLS_CALLS:
                out.append(("OLS", n.lineno))
    return sorted(out, key=lambda t: t[1])


def func_src(path, name: str) -> tuple:
    """(исходник функции, номер первой строки) по файлу и имени (в т. ч. метода)."""
    text = Path(path).read_text(encoding="utf-8")
    lines = text.splitlines()
    for n in ast.walk(ast.parse(text)):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name:
            body = textwrap.dedent("\n".join(lines[n.lineno - 1:n.end_lineno]))
            return "\n" * (n.lineno - 1) + body, n.lineno
    raise KeyError("%s::%s" % (path, name))


def classify_src(src: str) -> str:
    """Класс по набору вызовов: XGBoost / LightGBM / OLS / без бустера и OLS; оба бустера в одном исходнике — «XGBoost+LightGBM»."""
    kinds = {k for k, _ in scan(src)}
    b = sorted(kinds - {"OLS"})
    if b:
        return "+".join(b)
    return "OLS" if "OLS" in kinds else "без бустера и OLS"


def classify_chain(chain: list) -> dict:
    """chain — [(путь от корня, функция)]. -> класс, ссылки файл:строка вызова, перечень функций."""
    kinds, refs = set(), []
    for rel, fn in chain:
        src, _ = func_src(ROOT / rel, fn)
        for k, ln in scan(src):
            kinds.add(k)
            refs.append("%s:%d (%s в %s)" % (rel, ln, k, fn))
    b = sorted(kinds - {"OLS"})
    cls = "+".join(b) if b else ("OLS" if "OLS" in kinds else "без бустера и OLS")
    return {"class": cls, "refs": "; ".join(refs), "chain": "; ".join("%s::%s" % c for c in chain)}


def spec_name_uses_lib() -> bool:
    """Название строки пилота собрано из поля lib (Spec.name), а fitter выбирается по тому же полю в boost.fit — имя согласовано с кодом."""
    src, _ = func_src(ROOT / "models_v2/s2/boost.py", "name")
    ok_name = "self.lib" in src
    fsrc, _ = func_src(ROOT / "models_v2/s2/boost.py", "fit")
    return ok_name and 'spec.lib == "xgb"' in fsrc and "_fit_xgb" in fsrc and "_fit_lgbm" in fsrc


S2 = "models_v2/s2/boost.py"
CHAINS = {
    "har": [("models_v2/har.py", "ols")],
    "naive": [],
    "p2_xgb": [(S2, "_fit_xgb")], "p2_lgbm": [(S2, "_fit_lgbm")],
    "b3": [("models_v2/s3/gen.py", "fit_linear")],
    "fix": [("models_v2/s4/fit.py", "fit_i1")],
    "b1_obj_xgb": [("models_v2/s5/b1/fitobj.py", "_xgb")], "b1_obj_lgb": [("models_v2/s5/b1/fitobj.py", "_lgb")],
    "b2_xgb": [("models_v2/s5/c/fitx.py", "fit_i1x")],
    "d_lgb": [("models_v2/s5/d/lgbfit.py", "fit_lgbx")],
    "s6_lgb": [("models_v2/s6/engine6.py", "fit_var")],
    "fx_xgb_ref": [("models_v2/s5/c/fitx.py", "fit_i1x")],
}
