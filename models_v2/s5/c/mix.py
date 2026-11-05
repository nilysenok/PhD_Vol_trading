"""C2/C3: смеси H3 и переключатели над готовыми прогнозами. Всё — только по закрытым прогнозам (target_date <= d) и по режиму на d."""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_v2.s5 import common, pit
from models_v2.s5.b1 import designs, rules

H3_KINDS = ("w05", "exp", "roll", "reg")
BOOST = ("f4", "opt")
C2_H3 = tuple("c2_h3_%s_%s" % (k, b) for b in BOOST for k in H3_KINDS)
C3 = ("c3_r4_f4", "c3_r4_opt", "c3_str_f4", "c3_f3calm_g4", "c3_f3calm_rv", "c3_f3calm_har")
NEEDS = {r: (r.rsplit("_", 1)[1],) for r in C2_H3}
NEEDS.update({"c3_r4_f4": ("f4",), "c3_r4_opt": ("opt",), "c3_str_f4": ("f4",), "c3_f3calm_g4": ("f4", "top2"),
              "c3_f3calm_rv": ("f4", "top2"), "c3_f3calm_har": ("top2",)})


def times(ctrl: pd.DataFrame) -> tuple:
    """(d, td): td — target_date (прогноз закрыт при target_date <= d); мутация td = d берёт незакрытые прогнозы."""
    return ctrl["d"].to_numpy("datetime64[ns]"), ctrl["target_date"].to_numpy("datetime64[ns]")


def context(F, grid, ctrl: pd.DataFrame) -> dict:
    """Режимы и входы правил на сетке (всё по прошлому). ctrl: d, target_date, rv, s2_b2, s2_harj."""
    rg = designs.regime_cols(F)
    st = pd.read_parquet(common.OUT / "stress_p3.parquet").set_index("d")["stress"].reindex(ctrl["d"]).to_numpy(bool)
    d, td = times(ctrl)
    return {"rv": ctrl["rv"].to_numpy(float), "h": ctrl["s2_harj"].to_numpy(float), "d": d, "td": td, "qw": rg["qw"][grid], "grp": (rg["qw"][grid] >= 4).astype(float),
            "ty": ctrl["target_date"].dt.year.to_numpy(), "st_g4": st, "st_rv": np.nan_to_num(rg["stress"][grid], nan=0.0) > 0.5}


def h3(kind: str, c: dict, boost: np.ndarray, extra_closed: int = 0) -> np.ndarray:
    if kind == "w05":
        w = np.full(len(boost), 0.5)
    else:
        w = rules.h3_weights(kind, c["rv"], c["h"], boost, c["d"], c["td"], c["grp"] if kind == "reg" else None, extra_closed)
    return rules.mix(w, c["h"], boost)


def r4(c: dict, alt: np.ndarray, include_year: bool = False) -> np.ndarray:
    th = rules.theta_by_year(c["qw"], c["ty"], c["h"], alt, c["rv"], include_year=include_year)
    return rules.switch(c["qw"], c["ty"], th, c["h"], alt)


def regime_rv(rv_w: np.ndarray, include_current: bool = False) -> np.ndarray:
    """Стресс_RV: RV_w(d) строго выше квантиля 2/3 по 252 предыдущим (include_current=True — мутация № 22)."""
    if not include_current:
        q = pit.q_prior_nan(rv_w, 2 / 3)
    else:
        q = np.full(len(rv_w), np.nan)
        for k in range(pit.WIN, len(rv_w)):
            q[k] = np.nanquantile(rv_w[k - pit.WIN + 1:k + 1], 2 / 3)
    return np.where(np.isfinite(q) & np.isfinite(rv_w), rv_w > q, False)


def switch(stress: np.ndarray, in_stress: np.ndarray, calm: np.ndarray) -> np.ndarray:
    return np.where(stress, in_stress, calm)


def build(row: str, c: dict, b: dict) -> np.ndarray:
    """σ̂² строки row по входам c (контекст) и прогнозам бустеров b: {'f4','opt','top2'}."""
    if row in C2_H3:
        return h3(row.split("_")[2], c, b[row.rsplit("_", 1)[1]])
    if row in ("c3_r4_f4", "c3_r4_opt"):
        return r4(c, b[row.rsplit("_", 1)[1]])
    if row == "c3_str_f4":
        return switch(c["st_g4"], b["f4"], c["h"])
    if row == "c3_f3calm_g4":
        return switch(c["st_g4"], b["f4"], b["top2"])
    if row == "c3_f3calm_rv":
        return switch(c["st_rv"], b["f4"], b["top2"])
    if row == "c3_f3calm_har":
        return switch(c["st_g4"], c["h"], b["top2"])
    raise KeyError(row)
