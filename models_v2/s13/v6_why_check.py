"""V6 лист Z3: проверки `--check` (ТОЛЬКО ЧТЕНИЕ): G65 (диагностики из сохранённых прогнозов, без обучателей), G66 (PIT режимов), G67 (тождества), G68 (документ из таблиц).
Пересчёт в памяти сверяется с диском; независимо — с таблицами V6 FIN (`FIN_ALL`, `FIN_TESTS`: те же Δ и Δ в % K1)."""
from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd

from models_v2.s13 import v6_0_io as IO, v6_why_a as A, v6_why_data as DT, v6_why_docheck as DC, v6_why_doc as W0, v6_why_stage as ST

TOL = 1e-9
ID_TOL = 1e-12
TRAINERS = ("lightgbm", "xgboost", "catboost", "sklearn", "v6_1_fit")


def disk_vs_memory(r: dict) -> list:
    bad = []
    for n in DT.TABLES:
        if not (DT.WHY / (n + ".csv")).exists():
            bad.append("G65: нет таблицы %s" % n)
            continue
        a, b = DT.read_csv(n), r["t"][n].reset_index(drop=True)
        if list(a.columns) != list(b.columns) or len(a) != len(b):
            bad.append("G65: %s — столбцы/строки не совпали" % n)
            continue
        for c in a.columns:
            ok = np.allclose(a[c].to_numpy(float), b[c].to_numpy(float), rtol=0, atol=TOL, equal_nan=True) if pd.api.types.is_numeric_dtype(b[c]) else (a[c].astype(str) == b[c].astype(str)).all()
            if not ok:
                bad.append("G65: %s, столбец %s: диск ≠ пересчёт" % (n, c))
    return bad


def identities(t: dict, n_dates: int) -> list:
    """G67: сумма вкладов = общей Δ (годы, топ + остальные, группы), допуск 1e-12; группы и годы покрывают все даты."""
    bad, y = [], t["WHY_YEARS"]
    tot = y[y.year == "total"]
    bad += ["G67: годы %s к %s: Δ %.3e ≠ сумма вкладов %.3e" % (r.hybrid, r.base, r.delta, r.contrib) for r in tot.itertuples() if abs(r.delta - r.contrib) > ID_TOL]
    bad += ["G67: годы %s к %s: охвачено %d дат из %d" % (r.hybrid, r.base, r.n, n_dates) for r in tot.itertuples() if r.n != n_dates]
    z = t["WHY_TAILS"]
    bad += ["G67: топ-дни %s к %s (%s): Δ ≠ топ + остальные" % (r.hybrid, r.base, r.frac) for r in z.itertuples() if abs(r.delta - r.contrib_top - r.contrib_rest) > ID_TOL]
    g = t["WHY_REGIME"]
    tg = g[g.group == "total"]
    bad += ["G67: группы %s к %s (%s): Δ %.3e ≠ сумма вкладов %.3e" % (r.hybrid, r.base, r.kind, r.delta, r.contrib) for r in tg.itertuples() if abs(r.delta - r.contrib) > ID_TOL]
    bad += ["G67: группы %s к %s (%s): охвачено %d дат из %d" % (r.hybrid, r.base, r.kind, r.n, n_dates) for r in tg.itertuples() if r.n != n_dates]
    return bad


def pit(d: dict) -> list:
    """G66: группа даты d не зависит от K1 после d (усечение ряда на любой точке не меняет метки до неё)."""
    f, full, bad = d["f"]["K1"], A.tercile_groups(d["f"]["K1"]), []
    for k in (300, 700, 1200, 1900):
        if not (A.tercile_groups(f[:k]) == full[:k]).all():
            bad.append("G66: метки терцилей на первых %d датах зависят от данных после них" % k)
    if not (full[: A.MIN_HIST] == "no_group").all() or (full[A.MIN_HIST:] == "no_group").any():
        bad.append("G66: первые %d дат должны быть без группы, остальные — с группой" % A.MIN_HIST)
    return bad


def vs_fin(t: dict) -> list:
    """Независимо: Δ и Δ в % K1 вся история = FIN_TESTS (config против base, window all), знак и величина."""
    ft = IO.read_csv("FIN_TESTS.csv")
    ft = ft[(ft["section"] == "delta_dm") & (ft["window"] == "all")]
    bad, y = [], t["WHY_YEARS"]
    for h, b in DT.PAIRS:
        a = y[(y.hybrid == h) & (y.base == b) & (y.year == "total")].iloc[0]
        e = ft[(ft["config"] == h) & (ft["base"] == b)].iloc[0]
        if abs(a.delta_pct - e.delta_pct) > 1e-6:
            bad.append("G65: Δ%% %s к %s: %.6f ≠ FIN_TESTS %.6f" % (h, b, a.delta_pct, e.delta_pct))
    return bad


def no_trainers() -> list:
    """G65: в модулях v6_why_* и run_v6_why нет импорта обучателей (AST)."""
    bad = []
    for p in sorted(Path(__file__).parent.glob("v6_why_*.py")) + [Path(__file__).parent / "run_v6_why.py"]:
        bad += imports_trainer(p.read_text(encoding="utf-8"), p.name)
    return bad


def imports_trainer(src: str, name: str) -> list:
    out = []
    for n in ast.walk(ast.parse(src)):
        mods = [a.name for a in n.names] if isinstance(n, ast.Import) else ([n.module or ""] + [a.name for a in n.names]) if isinstance(n, ast.ImportFrom) else []
        out += ["G65: %s импортирует обучатель %s" % (name, m) for m in mods if any(x in m.split(".")[-1].lower() or x == m.split(".")[0].lower() for x in TRAINERS)]
    return out


def registry() -> list:
    if not DT.CP_FILE.exists():
        return ["G65: нет CHECKPOINT_WHY.json"]
    e, sh = json.loads(DT.CP_FILE.read_text()), ST.own_digest()
    bad = ["G65: CHECKPOINT_WHY: хэш %s не совпал" % n for n in DT.TABLES if e["tables_sha"].get(n) != sh[n + ".csv"]]
    bad += ["G65: CHECKPOINT_WHY: хэш документа не совпал"] if e["doc_sha"] != sh[DT.DOC.name] else []
    bad += ["G65: CHECKPOINT_WHY: израсходовано конфигураций ≠ 0"] if e["configs_spent"] != 0 else []
    return bad + (["G65: CHECKPOINT_WHY: хэш кода/входов устарел"] if (e["code_hash_v6_why"], e["inputs_hash"]) != (DT.code_hash(), IO.inputs_hash()) else [])


def all_bad(r: dict) -> list:
    t = W0.load_tables() if all((DT.WHY / (n + ".csv")).exists() for n in DT.TABLES) else r["t"]
    return disk_vs_memory(r) + identities(r["t"], len(r["d"]["dates"])) + pit(r["d"]) + vs_fin(r["t"]) + no_trainers() + registry() + DC.check_doc(t) + DC.check_static(t)
