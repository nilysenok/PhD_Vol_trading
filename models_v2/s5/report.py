"""README волны A: автоматические блоки (между маркерами) строятся только из CSV/JSON; проза пишется вручную вокруг."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from models_v2.s5 import common

BEGIN, END = "<!-- A:auto -->", "<!-- /A:auto -->"


def md(df: pd.DataFrame, nd: int = 4) -> str:
    cols = list(df.columns)
    def f(c, v):
        if c.startswith("n") and isinstance(v, (float, np.floating)) and np.isfinite(v) and float(v).is_integer():
            return str(int(v))
        return ("%.*f" % (nd, v)) if isinstance(v, (float, np.floating)) else str(v)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    lines += ["| " + " | ".join(f(c, v) for c, v in zip(cols, r)) + " |" for r in df.itertuples(index=False)]
    return "\n".join(lines)


def _rd(name: str) -> pd.DataFrame:
    return common.read_comment_csv(common.REP / name)


def auto_block() -> str:
    man = json.loads((common.REP / "BASELINE_MANIFEST.json").read_text())
    q = man["qlike_recomputed"]
    ed = _rd("ERROR_DIAGNOSTICS.csv")
    per = ed[ed.scope == "period"]
    b, h = per[per.model == "b2"].set_index("label"), per[per.model == "harj"].set_index("label")
    t_per = pd.DataFrame({"период": b.index, "n": b["n"].to_numpy(), "QLIKE Б-2": b["qlike"].to_numpy(), "QLIKE HAR-J(б)": h["qlike"].to_numpy(),
                          "Δ": b["delta_qlike_b2_minus_harj"].to_numpy(), "доля Б-2 лучше": b["win_share_b2"].to_numpy(),
                          "bias ln Б-2": b["bias_ln"].to_numpy(), "bias ln HAR-J": h["bias_ln"].to_numpy()})
    an = _rd("ANNUAL_MODEL_COMPARISON.csv")
    t_an = an[["year", "n", "qlike_b2", "qlike_harj", "delta_qlike_b2_minus_harj", "win_share_b2", "bias_ln_b2", "bias_ln_harj", "mean_r_b2", "mean_r_harj"]]
    hy = _rd("HYPOTHESES_A.csv")[["id", "diagnostic", "number", "verdict"]]
    gp = _rd("RV_GROUPS_PIT.csv")
    t_g = gp.pivot(index="group", columns="model", values=["n", "qlike", "bias_ln", "under_share"])
    t_g.columns = ["%s_%s" % c for c in t_g.columns]
    t_g = t_g.reset_index()
    lc = _rd("LOSS_CHECK.csv")
    t_l = lc[lc.slice.isin(["p2017_2021", "all_after_252_closed", "all"])][["model", "variant", "slice", "n", "qlike", "mean_r", "mean_c"]]
    rg = _rd("REGIME_DIAGNOSTICS.csv")
    rg = rg[rg.kind.isin(["quintile_w", "stress_g4", "window"])].pivot(index=["kind", "label"], columns="model", values=["n", "qlike", "mean_r"])
    rg.columns = ["%s_%s" % c for c in rg.columns]
    t_r = rg.reset_index().drop(columns="n_harj").rename(columns={"n_b2": "n"})
    t_s = _rd("SENSITIVITY_A.csv")
    t_f = _rd("RV_GROUPS_FACT.csv")
    t_f = t_f.pivot(index="group", columns="model", values=["n", "qlike", "mean_r", "under_share"])
    t_f.columns = ["%s_%s" % c for c in t_f.columns]
    t_f = t_f.reset_index().drop(columns="n_harj")
    parts = ["**Контроли (пересчёт из parquet).** QLIKE Б-2 = %.12f; HAR-J(б) = %.12f; n = %d; воспроизведение на 3 датах: %s." % (
        q["b2"], q["harj"], q["n"], man["reproduced"]),
        "**QLIKE по периодам (по дате цели).**", md(t_per), "**По годам.**", md(t_an), "**Гипотезы (а)–(е).**", md(hy),
        "**Режимы по прошлому (квинтили RV_w, стресс Г4).**", md(t_r), "**Чувствительность к выборке (диагностика, не результат).**", md(t_s),
        "**Группы по факту RV цели (объяснительно, не PIT).**", md(t_f),
        "**Группы по RV, известной на d (PIT).**", md(t_g), "**Сдвиг масштаба exp(c), PIT (диагностика, не trial).**", md(t_l, 4)]
    return "\n\n".join(parts)


def write_readme() -> None:
    p = common.REP / "README.md"
    body = BEGIN + "\n" + auto_block() + "\n" + END
    if p.exists() and BEGIN in p.read_text():
        s = p.read_text()
        a, z = s.index(BEGIN), s.index(END) + len(END)
        p.write_text(s[:a] + body + s[z:])
    else:
        p.write_text("# SBER h=1, глубокая доработка бустинга: волна A\n\n" + body + "\n")


def check_readme() -> bool:
    s = (common.REP / "README.md").read_text()
    a, z = s.index(BEGIN), s.index(END) + len(END)
    return s[a:z] == BEGIN + "\n" + auto_block() + "\n" + END
