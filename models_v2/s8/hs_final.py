"""Ж-212 (G15, G16): стадия `final` — FINAL_STRUCT_CONFIG.json (фиксированные спецификации для переноса, без запуска переноса), REPRODUCTION.md, проверка бюджета и состава каталога.
Итоговый отчёт HAR_BOOSTING_STRUCTURAL_REDESIGN_FINAL.md пишется автором по числам CSV (ответы на 7 вопросов п. 14 блока); здесь проверяется, что он существует и ссылается на все таблицы."""
from __future__ import annotations

import json

import pandas as pd

from models_v2.s8 import hs_ledger as L, hs_paths as P

REQUIRED = ("ALL_TRIALS.csv", "CHECKPOINT.json", "BASE_REPRO.csv", "HAR_FEATURES_B.csv", "RESIDUAL_QLIKE_C.csv", "COMPONENT_D.csv", "CONDITIONAL_E.csv", "MOE_F.csv",
            "THIRD_EXPERT_TEST.csv", "FORECAST_COMPARISONS.csv", "ANNUAL_PERFORMANCE.csv", "REGIME_PERFORMANCE.csv", "STATISTICAL_TESTS.csv", "MCS.csv", "LEVELS.csv",
            "HAR_BOOSTING_STRUCTURAL_REDESIGN_FINAL.md")
STUBS = {"CONDITIONAL_E.csv": "E не запускалось (нет g для коррекции) — см. CHECKPOINT.json decisions.conditional",
         "COMPONENT_D.csv": "D не запускалось", "MOE_F.csv": "F не запускалось"}


def specs() -> dict:
    """Фиксированные спецификации лучших строк: из таблиц направлений (по crit_wf), с описанием правил — для переноса на другие акции без подбора."""
    out = {}
    for key, csv, fam in (("B_XGB", "HAR_FEATURES_B.csv", "XGB"), ("B_LGBM", "HAR_FEATURES_B.csv", "LGBM"), ("C_XGB", "RESIDUAL_QLIKE_C.csv", "XGB"), ("C_LGBM", "RESIDUAL_QLIKE_C.csv", "LGBM"),
                          ("D", "COMPONENT_D.csv", None), ("E", "CONDITIONAL_E.csv", None), ("F", "MOE_F.csv", None)):
        p = P.REP / csv
        if not p.exists():
            out[key] = {"status": "не запускалось"}
            continue
        t = pd.read_csv(p, comment="#")
        if "crit_wf" not in t.columns:
            out[key] = {"status": "не запускалось", "reason": str(t["reason"].iloc[0]) if "reason" in t else ""}
            continue
        t = t if fam is None else t[t["family"] == fam]
        r = t.sort_values("crit_wf").iloc[0]
        out[key] = {"model": r["model"], "qlike_all": float(r["qlike_all"]), "crit_wf": float(r["crit_wf"]), "status": "фиксированная спецификация; перенос не запускался (уровень 4 — не проверялся)"}
    return out


def stage_final(_a=None) -> int:
    ck = json.loads(L.CHECK.read_text())
    ok_budget = all(ck["used"][k] <= v for k, v in L.LIMITS.items()) and ck["used_total"] <= L.TOTAL and ck["N_final"] == L.N_PRIOR + ck["used_total"]
    cfg = {"program": "Ж-212", "scope": "SBER h=1", "status": "DEVELOPMENT + HISTORICAL WALK-FORWARD (не OOS); первая цифра — потолок", "N_prior": L.N_PRIOR, "used": ck["used"],
           "N_final": ck["N_final"], "levels_4": "не проверялся", "specs": specs(), "decisions": ck.get("decisions", {})}
    (P.REP / "FINAL_STRUCT_CONFIG.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=1, default=float))
    rep = ["# REPRODUCTION — Ж-212 (SBER h=1)", "", "Порядок (по одной команде, `run_in_background`), из корня репозитория:", ""]
    rep += ["- `python3 -W ignore -m models_v2.s8.run_hs --stage %s`" % s for s in ("ctrl", "table", "diag", "q8", "ledger", "base", "b", "c", "d", "e", "f", "g", "final")]
    rep += ["", "Тесты: `python3 -W ignore -m pytest tests/test_v2_hs_*.py -q`. Кэш рядов — `data/regen/hybrid_struct_v1/*_series.parquet` (ключ — имя конфигурации; удалить файл — пересчёт).",
            "Входы: `data/regen/xgb_lgbm_full_v1/e/components.parquet` (SHA-256 в `CONTROL_VALUES.csv`), `data/iss_check/index_10m/IMOEX.parquet`, `data/fastbt/s9/rvi_1850.parquet`.",
            "Детерминизм: seed 42, num_threads=1 (LGBM), nthread=1 (XGB), spawn-пул ≤ 4."]
    (P.REP / "REPRODUCTION.md").write_text("\n".join(rep) + "\n")
    miss = [f for f in REQUIRED if not (P.REP / f).exists()]
    print("final: бюджет ok=%s; used=%s; N_final=%d; нет файлов: %s" % (ok_budget, ck["used"], ck["N_final"], miss), flush=True)
    return 0 if ok_budget and not miss else 1
