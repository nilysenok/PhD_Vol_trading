"""Ж-211 (HF10/HF11): тесты A/B/C (DM + блочный бутстрап), срезы по периодам и уровни успеха 1/2 для финалистов. Чистые функции над рядами σ̂²."""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_v2.s5.b1 import evalrows
from models_v2.s5.d import stats_d
from models_v2.s6.e import comp
from models_v2.s7 import hf_cmp as C, hf_data as D

TESTS = (("A", "HAR", "harj"), ("B", "XJ", "xj"), ("C", "LF", "lf"))
SLICES = {"all": None, "no_240222": "no_240222", "P1_2017_2021": (2017, 2021), "P2_2022": (2022, 2022), "P3_2023_2025": (2023, 2025), "P4_2026": (2026, 2026),
          "early_2017_2018": (2017, 2018), "late_2019_2026": (2019, 2026), "G_2019_2021": (2019, 2021)}


CEILING = "потолок: выбор по crit_wf на всей истории"
POSTSEL = "постселекционно: финалист выбран по crit_wf на всей истории, p не поправлены на этот выбор (HF10)"


def ceiling_note(model: str) -> str:
    """Пометка финалиста: выбран по min crit_wf на всей истории (потолок); H3/H4 — ещё и набор признаков по EX1. Опоры (HAR-J, XGBoost, LightGBM) — пусто."""
    if ":" not in model:
        return ""
    return CEILING + (" + признаки по EX1" if model.split(":")[0][:2] in ("H3", "H4") else "")


def mask_of(df, key):
    if key == "all":
        return np.ones(len(df), bool)
    if key == "no_240222":
        return evalrows.masks(df["target_date"])["no_240222"]
    return D.pmask(df, *SLICES[key])


def tests_rows(df: pd.DataFrame, name: str, s2, n_final: int) -> list:
    """Строки STATISTICAL_TESTS: модель против HAR-J (A), XJ (B), LF (C) по срезам. delta = QLIKE(M) − QLIKE(опоры), минус — модель лучше; p двусторонний (Ньюи–Уэст, лаг 5)."""
    rv, s2, rows = df["rv"].to_numpy(float), np.asarray(s2, float), []
    for t, ref, _k in TESTS:
        base = comp.col(df, ref)
        dl = D.qs(df, s2) - D.qs(df, base)
        for key in SLICES:
            m = mask_of(df, key)
            r = evalrows.vs(rv, base, s2, m)
            row = {"model": name, "test": t, "ref": ref, "period": key, "n": int(m.sum()), "delta": r["d"], "dm": r["dm"], "p_raw": r["p"], "N": n_final}
            if key in ("all", "late_2019_2026"):
                b = stats_d.boot_delta(dl[m], stats_d.block_idx(int(m.sum())))
                row.update({"boot_lo95": b["lo95"], "boot_hi95": b["hi95"], "boot_p": b["p_boot"]})
            rows.append(row)
    return rows


def add_holm(t: pd.DataFrame, n_final: int) -> pd.DataFrame:
    """Холм с N по полному реестру: семейство первичных = (модель × тест A/B/C) на всей истории; прочие срезы — только сырые."""
    t = t.copy()
    t["p_holm"] = np.nan
    ix = t.index[t["period"] == "all"]
    t.loc[ix, "p_holm"] = stats_d.holm_n(t.loc[ix, "p_raw"].fillna(1.0).to_numpy(), n_final)
    return t


def pick(t: pd.DataFrame, model: str, test: str, period: str) -> pd.Series:
    return t[(t["model"] == model) & (t["test"] == test) & (t["period"] == period)].iloc[0]


def levels(t: pd.DataFrame, model: str, crit: dict, qall: dict, star: str) -> dict:
    """HF11. crit/qall — crit_wf и QLIKE по всей истории: ключи HAR, model, star; (д) — против бустинга, не равного S*. Возврат — флаги по пунктам и итог 'L2' / 'L1' / 'L1 только сырой' / 'нет'."""
    A = lambda p: pick(t, model, "A", p)
    B = lambda p: pick(t, model, "B", p)
    a = qall[model] < qall["HAR"] and crit[model] < crit["HAR"]
    b = sum(A(p)["delta"] < 0 for p in ("P1_2017_2021", "P2_2022", "P3_2023_2025")) >= 2 and A("no_240222")["delta"] < 0
    c = A("all")["p_raw"] < 0.05 and A("all")["boot_hi95"] < 0
    d = A("all")["p_holm"] < 0.10
    l1 = bool(a and b and c and d)
    sk = lambda p: pick(t, model, "B" if star == "XJ" else "C", p)
    l2 = {"b": bool(crit[model] < crit[star] - 0.0005 and sk("all")["delta"] < 0),
          "c": bool(sk("all")["p_raw"] < 0.05 and sk("all")["boot_hi95"] < 0 and sk("all")["p_holm"] < 0.10),
          "d": bool(sum(sk(p)["delta"] < 0 for p in ("G_2019_2021", "P2_2022", "P3_2023_2025")) >= 2 and sk("late_2019_2026")["delta"] < 0 and sk("late_2019_2026")["p_raw"] < 0.10),
          "e": bool(pick(t, model, "C" if star == "XJ" else "B", "all")["delta"] < 0)}
    out = {"L1_a": bool(a), "L1_b": bool(b), "L1_c": bool(c), "L1_d": bool(d), "L1": l1, **{"L2_" + k: v for k, v in l2.items()}}
    out["L2"] = bool(l1 and all(l2.values()))
    out["verdict"] = "L2" if out["L2"] else ("L1" if l1 else ("L1 только сырой" if (a and b and c) else "нет"))
    return out
