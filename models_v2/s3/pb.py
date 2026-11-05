"""Б-1…Б-5 (Ж-198): дизайны, реестр строк, сборка прогнозов (а)/(б). Б-7 — post.b7_*."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from models_v2 import har, metrics
from models_v2.s2 import boost
from models_v2.s2 import features as ft
from models_v2.s3 import gen, resid, scale

LIBS = ("xgb", "lgbm")
C_GRID = (0.5, 0.75, 1.0)
CSTR = {0.5: "c050", 0.75: "c075", 1.0: "c100"}


@dataclass(frozen=True)
class Row:
    name: str
    design: str
    lib: str
    mono: bool
    c: float = 1.0
    bias: bool = True


def build_designs(F: ft.Fp) -> tuple:
    """-> ({имя: Design}, {имя: ln-смещение offset[d]}). lb — PIT ln HAR-J(б)."""
    p = har.Panel(F.dates, F.seg, F.rv, F.X[:, :4])
    lb = resid.harj_ab(p)[1]
    seg, base_cols = F.seg, tuple(range(ft.N_BASE))
    zero = np.zeros(len(F.rv))
    D, off = {"b1": gen.Design(F.X[:, :ft.N_BASE], seg, resid.target_minus(F, lb), base_cols)}, {"b1": lb}
    for kind in ("m", "252"):
        S = scale.scale(F, kind)
        D["s" + kind] = gen.Design(scale.normalize(F.X, S), seg, resid.target_minus(F, np.log(S)), base_cols)
        off["s" + kind] = np.log(S)
    y = resid.target_minus(F, zero)
    D["b3n"] = gen.Design(F.X[:, :ft.N_BASE], seg, y, base_cols, linear=True)
    D["b3h"] = gen.Design(F.X, seg, y, tuple(range(ft.N_BASE + 1)), need=(0, 1, 2, 3, ft.I_HAR), linear=True)
    D["b5"] = gen.Design(F.X[:, [0, 1, 2, 3, ft.I_HAR]], seg, y, (0, 1, 2, 3, 4), need=(0, 1, 2, 3, 4))
    for k in ("b3n", "b3h", "b5"):
        off[k] = zero
    return D, off


def rows() -> list:
    out = []
    for lib in LIBS:
        out += [Row("b1_%s_%s" % (lib, CSTR[c]), "b1", lib, False, c) for c in C_GRID]
        out.append(Row("b1_%s_c100_nobias" % lib, "b1", lib, False, 1.0, False))
        for kind in ("m", "252"):
            out.append(Row("b2_%s_%s" % (kind, lib), "s" + kind, lib, False))
            out.append(Row("b4_%s_%s" % (kind, lib), "s" + kind, lib, True))
        out += [Row("b5_%s_%s" % (lib, m), "b5", lib, m == "mono") for m in ("free", "mono")]
    out += [Row("b3_nohar_lgbm", "b3n", "lgbm", False), Row("b3_har_lgbm", "b3h", "lgbm", False)]
    return out


def fit_keys(rs: list) -> list:
    """Уникальные подгонки (design, lib, mono): c и bias подгонку не меняют."""
    return sorted({(r.design, r.lib, r.mono) for r in rs})


def spec_of(lib: str, mono: bool) -> boost.Spec:
    return boost.Spec(lib, "mse", False, mono)


def frame(F: ft.Fp, r: Row, off: np.ndarray, res: list, kind: str) -> pd.DataFrame:
    """res: [(d_idx, результат run_design)] -> long-таблица прогнозов строки r; kind 'a' | 'b'."""
    i = np.concatenate([np.asarray(d) for d, _ in res])
    f = np.concatenate([x["f"] for _, x in res])
    s2 = np.concatenate([np.full(len(d), x["s2_es"]) for d, x in res])
    o = np.argsort(i)
    i, f, s2 = i[o], f[o], s2[o]
    bias = (s2 / 2) if (kind == "b" and r.bias) else 0.0
    return pd.DataFrame({"d": F.dates[i], "target_date": F.dates[i + 1], "model": "%s_%s" % (r.name, kind),
                         "sigma2": np.exp(off[i] + r.c * f + bias), "rv": F.rv[i + 1]})


def extra_slices(fc: pd.DataFrame) -> pd.DataFrame:
    """Срез «без 2022» по дате цели (metrics.slice_masks даёт y2022 и остальное)."""
    out = []
    for model, f in fc.groupby("model", sort=False):
        m = (f["target_date"].dt.year != 2022).to_numpy()
        out.append({"model": model, "slice": "no2022", **metrics.summary(f["rv"][m], f["sigma2"][m]), "note": ""})
    return pd.DataFrame(out)


STRONG = ("mar_jun_2020", "y2023", "y2024", "y2025")


def verdict(mt: pd.DataFrame, ref: pd.Series, best2: pd.Series) -> pd.DataFrame:
    """Критерий GB (до счёта): closes2022 = QLIKE y2022 <= harj_b; strong = на STRONG-срезах <= min(harj_b, лучшая строка
    (б) этапа 2); weak = <= harj_b. ref, best2 — QLIKE по срезам. Допуск 0."""
    q = mt.pivot(index="model", columns="slice", values="qlike")
    lim = np.minimum(ref[list(STRONG)], best2[list(STRONG)])
    res = pd.DataFrame({"closes2022": q["y2022"] <= ref["y2022"],
                        "strong_no_loss": (q[list(STRONG)] <= lim).all(axis=1),
                        "weak_no_loss": (q[list(STRONG)] <= ref[list(STRONG)]).all(axis=1)})
    res["closes_and_strong"] = res["closes2022"] & res["strong_no_loss"]
    res["closes_and_weak"] = res["closes2022"] & res["weak_no_loss"]
    return res.reset_index()
