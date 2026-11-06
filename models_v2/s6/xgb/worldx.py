"""Мир потока XGB: Б-2 (F, сетка, форма m, HP I2) + признаки F1–F6 (принятые группы Ж-204) — ТОЛЬКО ЧТЕНИЕ прежних каталогов; Spec строк B1."""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd

from models_v2 import prov
from models_v2.s5 import common, pit
from models_v2.s5.b1 import designs
from models_v2.s5.b2 import engine, world as b2w
from models_v2.s6 import paths6

B2_DIR = common.DEEP / "b2"
FOLDS_B2 = common.DEEP / "b1" / "loyo_ctrl_b2.parquet"


def feats_key() -> str:
    return hashlib.sha256((common.sha_file(prov.BARS) + common.sha_file(common.IMOEX) + b2w.code_sha()).encode()).hexdigest()[:16]


def load() -> dict:
    """F, grid, dz0 (Б-2), i2, feats (F1–F6, нормированы), acc (принятые группы), eps, ctrl. Ключ признаков не совпал — стоп (не пересчитываем в чужой каталог)."""
    w = paths6.world()
    if (B2_DIR / "feats_b2.key").read_text() != feats_key():
        raise SystemExit("стоп: ключ feats_b2 не совпал с кодом/входами; пересчёт в прежний каталог запрещён")
    w["feats"] = pd.read_parquet(B2_DIR / "feats_b2.parquet")
    w["acc"] = json.loads((B2_DIR / "accepted.json").read_text())
    w["eps"] = pit.eps_floor(w["F"].rv)
    w["ctrl"] = common.controls()
    return w


def cols_of(acc: dict, groups) -> tuple:
    return tuple(c for g in groups for c in acc[g])


def dz_of(w: dict, c: dict):
    return engine.make_dz(w["dz0"], w["feats"], engine.Cfg(c["name"], add=cols_of(w["acc"], c["groups"]), drop=c["drop"]))


def spec_of(w: dict, c: dict) -> designs.Spec:
    """l2: ln-цель, σ̂² = exp(off + f + s²/2); gamma / qlike: цель RV/S, σ̂² = exp(off + f); веса c['gw'] — L5 (RV_w^(−γ), строка t)."""
    dz = dz_of(w, c)
    if c["loss"] == "l2":
        return designs.Spec(c["name"], "l2", dz.X, dz.y, "ln", dz.off, gw=c["gw"])
    return designs.Spec(c["name"], c["loss"], dz.X, np.exp(dz.y), "lnmu", dz.off, gw=c["gw"])


def sigma_frame(w: dict, spec, raw: pd.DataFrame) -> pd.DataFrame:
    """d, target_date, rv, sigma2 по сетке из сырого потока (row, f, s2)."""
    s2, _, _ = designs.sigma2(spec, raw["row"].to_numpy(), raw["f"].to_numpy(), raw["s2"].to_numpy(), w["eps"])
    i = raw["row"].to_numpy()
    F = w["F"]
    return pd.DataFrame({"d": F.dates[i], "target_date": F.dates[i + 1], "rv": F.rv[i + 1], "sigma2": s2})
