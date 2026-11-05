"""C: строки таблиц (evalrows.row + разбивка потолок/чисто), правило «перспективно», Холм по семейству 40, реестр и контрольная точка."""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from models_v2 import metrics
from models_v2.s3 import dm
from models_v2.s5 import common, registry
from models_v2.s5.b1 import evalrows

TRIALS_C = common.REP / "OPTUNA_TRIALS_C.csv"
TRIALS_ALL = common.REP / "OPTUNA_TRIALS.csv"
CEIL = "потолок (кандидаты видели 2017–2021)"
LIM_C = {"feature_engineering": 25, "har_boosting": 15, "regime_aware": 10}
FAMILY = 90        # B1 28 + B2 22 + C 40: все уникальные trials программы на момент сравнения
FAMILY_C = 40      # справочно: только trials C


def qlike_of(fc) -> np.ndarray:
    return metrics.qlike(fc["rv"].to_numpy(float), fc["sigma2"].to_numpy(float))


def split_cols(fc: pd.DataFrame, f4: pd.DataFrame, opt_based: bool) -> dict:
    """2017–2021 (потолок для строк на c1_opt) и 2022–2026 (чисто) отдельно, их разность к b2_f4 (без Optuna)."""
    y = fc["target_date"].dt.year.to_numpy()
    q, q4 = qlike_of(fc), qlike_of(f4)
    a, b = y <= 2021, y >= 2022
    out = {"qlike_2022_2026": float(q[b].mean()), "d_f4_2017_2021": float((q - q4)[a].mean()), "d_f4_2022_2026": float((q - q4)[b].mean()),
           "mark_2017_2021": CEIL if opt_based else ""}
    return out


def q_int_b2(kind: str, ctrl: pd.DataFrame, crit_loyo_b2: float) -> float:
    if kind == "loyo":
        return crit_loyo_b2
    fc = pd.DataFrame({"rv": ctrl["rv"], "sigma2": ctrl["s2_b2"], "target_date": ctrl["target_date"]})
    return evalrows.q_wf(qlike_of(fc), fc["target_date"])


def build_row(rid: str, fc: pd.DataFrame, ctrl: pd.DataFrame, f4: pd.DataFrame, kind: str, q_int: float, crit_b2: float, opt_based: bool,
              direction: str, trial_id: str, extra: dict | None = None) -> dict:
    if len(fc) != common.N_GRID or not (fc["d"].to_numpy("datetime64[ns]") == ctrl["d"].to_numpy("datetime64[ns]")).all():
        raise ValueError("сетка строки %s не равна контролю" % rid)
    if not ((fc["sigma2"] > 0).all() and np.isfinite(fc["sigma2"]).all() and (fc["rv"] > 0).all()):
        raise ValueError("σ̂² или RV не положительны: %s" % rid)
    r = evalrows.row(fc, ctrl, extra)
    ref = q_int_b2(kind, ctrl, crit_b2)
    q4 = float(qlike_of(f4).mean())
    r.update(split_cols(fc, f4, opt_based))
    r.update({"row_id": rid, "trial_id": trial_id, "direction": direction, "q_int_kind": kind, "q_int": q_int, "q_int_b2": ref,
              "rel_gain": (ref - q_int) / ref, "perspective": evalrows.perspective(q_int, ref, r["d_harj_no_240222"]),
              "n_floor": 0, "qlike_f4_ref": q4})
    r["q_loyo"], r["q_wf"] = (q_int, r["q_wf"]) if kind == "loyo" else (np.nan, q_int)
    return r


def holm_cols(df: pd.DataFrame, m: int = FAMILY, tag: str = "") -> pd.DataFrame:
    """Холм с N = m (недостающие до m строки — p = 1: trials без дневного прогона), отдельно к HAR-J(б) и Б-2.
    Основной N = все уникальные trials программы (90 после C, ↺ дирижёра); N = 40 (trials C) — только справочно (tag='_ref40')."""
    df = df.copy()
    for src, dst in (("p_harj", "p_holm_harj" + tag), ("p_b2", "p_holm_b2" + tag)):
        p = np.concatenate([df[src].to_numpy(float), np.ones(max(0, m - len(df)))])
        df[dst] = dm.holm(p)[:len(df)]
    return df


def top3(df: pd.DataFrame) -> pd.DataFrame:
    """Лучшие 3 среди perspective по относительному выигрышу внутреннего критерия; ничья — меньший d_harj_no_240222. Описание не используется для отбора."""
    p = df[df["perspective"] == True]  # noqa: E712
    return p.sort_values(["rel_gain", "d_harj_no_240222"], ascending=[False, True]).head(3)


def trial_row(tid: str, direction: str, cfg: dict, inner: str, desc: str, secs: float, seed: int = 42, repeat: str = "") -> dict:
    return {"trial_id": tid, "волна": "C", "направление": direction, "конфигурация": json.dumps(cfg, ensure_ascii=False, sort_keys=True),
            "метрика_внутр_валидации": inner, "метрики_описания": desc, "время": time.strftime("%Y-%m-%d %H:%M:%S") if not secs else "%.1f" % secs,
            "seed": seed, "повторный_seed": repeat}


def check_budget(rows: list, used_before: dict) -> None:
    """Стоп при превышении лимита волны C или лимита п. 26 (GATES_C §1)."""
    cnt = {}
    for r in rows:
        cnt[r["направление"]] = cnt.get(r["направление"], 0) + 1
    for k, v in cnt.items():
        if v > LIM_C[k] or used_before.get(k, 0) + v > registry.LIMITS[k]:
            raise SystemExit("стоп: бюджет trials %s превышен (%d)" % (k, v))


def merge_registry(c_rows: list) -> dict:
    """OPTUNA_TRIALS.csv = B1 + B2 (без изменений значений) + C; trial_id уникальны; возвращает счётчики по направлениям."""
    cols = list(registry.TRIAL_COLS)
    b1, b2 = (pd.read_csv(common.REP / n, dtype=str).fillna("") for n in ("OPTUNA_TRIALS_B1.csv", "OPTUNA_TRIALS_B2.csv"))
    used_b = pd.concat([b1, b2])["направление"].value_counts().to_dict()
    check_budget(c_rows, used_b)
    c = pd.DataFrame(c_rows).astype(str).fillna("")
    allr = pd.concat([b1[cols], b2[cols], c[cols]], ignore_index=True)
    if allr["trial_id"].duplicated().any():
        raise SystemExit("стоп: дубль trial_id в реестре")
    c.to_csv(TRIALS_C, index=False)
    allr.to_csv(TRIALS_ALL, index=False)
    return allr["направление"].value_counts().to_dict()


def write_checkpoint(counts: dict) -> None:
    p = common.REP / "CHECKPOINT.json"
    ck = json.loads(p.read_text())
    for k in registry.LIMITS:
        ck["trials_used"][k] = int(counts.get(k, 0))
    ck["waves"].update({"B": "done", "C": "done"})
    assert sum(ck["trials_used"].values()) <= ck["trials_limit_total"]
    p.write_text(json.dumps(ck, ensure_ascii=False, indent=1))


def append_table(path, df: pd.DataFrame, key: str) -> int:
    """Дописывает строки df в существующий csv (шапка-комментарий сохраняется), колонки — как в файле; строки с уже имеющимся key пропускаются."""
    old = pd.read_csv(path, comment="#")
    new = df[~df[key].isin(old[key])].reindex(columns=old.columns)
    if len(new):
        new.to_csv(path, mode="a", header=False, index=False)
    return len(new)
