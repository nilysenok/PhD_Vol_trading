"""INVENTORY.csv: реестр конфигураций Ж-197…Ж-204 (п. 1 задания). Числа — пересчёт metrics.qlike из сохранённых прогнозов / пересборка смесей."""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_v2 import metrics
from models_v2.s6 import inv_meta, origin, paths6

PILOT = ("forecasts_p1", "forecasts_p2", "forecasts_p3", "forecasts_pb", "fix_forecasts_a", "fix_forecasts_c")
COLS = ("id", "model_class", "library", "objective", "target", "features", "hyperparameters", "window", "retrain_freq", "early_stopping", "seed",
        "forecast_path", "forecast_sha256", "code_path", "code_sha256", "trial_id", "status", "eval_selected", "qlike_recomputed",
        "family", "origin_refs", "qlike_table", "qlike_diff", "verified", "note")
_SHA: dict = {}


def sha(path) -> str:
    p = str(path)
    if p not in _SHA:
        _SHA[p] = paths6.sha_file(path)
    return _SHA[p]


def row(rid, m, cls, fpath, sigma, rv, **kw) -> dict:
    q = float(metrics.qlike(rv, sigma).mean()) if sigma is not None else np.nan
    chain = cls["chain"].split("; ")[0].split("::")[0] if cls["chain"] else ""
    qt = kw.pop("qlike_table", np.nan)
    fsha = kw.pop("fsha", None)
    fsha = fsha if fsha is not None else (sha(paths6.ROOT / fpath) if fpath else "")
    ok = sigma is not None and len(sigma) == paths6.N_GRID
    r = {"id": rid, "model_class": kw.pop("model_class", cls["class"]), "library": cls["library"], "objective": m["objective"], "target": m["target"],
         "features": m["features"], "hyperparameters": m["hp"], "window": m["window"], "retrain_freq": m["retrain"], "early_stopping": m["es"],
         "seed": m["seed"], "forecast_path": fpath, "forecast_sha256": fsha,
         "code_path": chain, "code_sha256": sha(paths6.ROOT / chain) if chain else "", "trial_id": kw.pop("trial_id", ""),
         "status": "DEVELOPMENT / HISTORICAL WALK-FORWARD", "eval_selected": kw.pop("eval_selected", "нет"), "qlike_recomputed": q,
         "family": kw.pop("family", ""), "origin_refs": cls["refs"], "qlike_table": qt, "qlike_diff": abs(q - qt) if ok and qt == qt else np.nan,
         "verified": "подтверждено по коду (ast)" if cls["refs"] else "не подтверждено: нет вызова обучения в разобранной цепочке (смесь/наивный/не разобрано)",
         "note": kw.pop("note", "")}
    if not ok:
        r["verified"] = "не подтверждено: прогноз для пересчёта не сохранён"
    return r


def pilot_rows() -> list:
    out = []
    for src in PILOT:
        path = "data/regen/v2_sber_h1/%s.parquet" % src
        df = pd.read_parquet(paths6.ROOT / path)
        for name, g in df.groupby("model", sort=False):
            g = g.sort_values("d")
            m = inv_meta.pilot(name, src)
            cls = inv_meta.klass(m)
            kw = {"family": src}
            if src == "forecasts_p3":
                kw["model_class"] = "смесь/постобработка"
            if m["chain_key"] in ("p2_xgb", "p2_lgbm", "b3"):
                kw["note"] = "библиотека — по полю Spec.lib (имя согласовано с кодом: %s)" % origin.spec_name_uses_lib()
            out.append(row(name, m, cls, path, g["sigma2"].to_numpy(float), g["rv"].to_numpy(float), **kw))
    return out


def stage2_top3(rows: list) -> None:
    """Лучшие 3 строки этапа 2 на библиотеку (QLIKE `_b`, ежедневное): метка в family."""
    df = pd.DataFrame(rows)
    p2 = df[(df.family == "forecasts_p2") & df.id.str.endswith("_b")]
    for lib in ("xgb", "lgbm"):
        top = p2[p2.id.str.startswith(lib + "_")].nsmallest(3, "qlike_recomputed")["id"]
        for r in rows:
            if r["id"] in set(top):
                r["family"] += ";этап2_top3_" + lib


def old_table_rows(L: dict) -> list:
    """B1/B2: прогнозы fc_*.parquet в deep_v1; значения таблиц — qlike_all."""
    out, seen = [], set()
    for sheet, tab, key, pre in (("B1", "OBJECTIVE_EXPERIMENTS", "row_id", "b1/fc_%s"), ("B1", "HAR_BOOSTING_RESULTS", "row_id", "b1/fc_%s"),
                                 ("B1", "REGIME_MODEL_RESULTS", "row_id", "b1/fc_%s"), ("B2", "FEATURE_ABLATIONS", "model", "b2/fc_%s"),
                                 ("B2", "WINDOW_RESEARCH", "model", "b2/fc_%s")):
        t = paths6.read_csv(paths6.OLD_REP / (tab + ".csv"))
        for r in t.itertuples():
            rid = getattr(r, key)
            if rid in seen or rid.startswith(("c1_", "c2_", "c3_")):
                continue
            seen.add(rid)
            fn = {"BASE": "b2/fc_BASE"}.get(rid, pre % rid)
            f = paths6.OLD_DATA[1] / (fn + ".parquet")
            fc = pd.read_parquet(f).sort_values("d") if f.exists() else None
            m = inv_meta.table_row(rid, sheet)
            cls = inv_meta.klass(m)
            if rid.startswith(("har_", "reg_")):
                cls["class"] = "смесь/остаток к HAR-J(б) или режим; fitter — " + cls["class"]
            out.append(row(rid, m, cls, paths6.rel(f) if f.exists() else "", None if fc is None else fc["sigma2"].to_numpy(float),
                           None if fc is None else fc["rv"].to_numpy(float), family=tab, trial_id=str(getattr(r, "trial_id", "") or ""),
                           qlike_table=float(r.qlike_all), note="" if f.exists() else "файл прогноза не найден: " + paths6.rel(f)))
    return out


def build(L: dict) -> pd.DataFrame:
    from models_v2.s6 import inv_cd
    rows = pilot_rows()
    stage2_top3(rows)
    rows += old_table_rows(L) + inv_cd.rows()
    df = pd.DataFrame(rows)[list(COLS)]
    if df["id"].duplicated().any():
        raise ValueError("дубли id: %s" % df["id"][df["id"].duplicated()].tolist())
    return df


def write(df: pd.DataFrame) -> None:
    paths6.mkdirs()
    paths6.write_csv(df, paths6.REP / "INVENTORY.csv", "инвентаризация; qlike_recomputed = metrics.qlike на 2271 датах (target_date) из сохранённых/пересобранных прогнозов")
    paths6.write_csv(df, paths6.OUT / "INVENTORY.csv", "копия для проверки")
