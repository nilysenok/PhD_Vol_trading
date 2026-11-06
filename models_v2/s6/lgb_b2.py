"""LightGBM-аналог Б-2 на всей сетке (наследованные HP I2 из сетки 132), seed-контроли 43–46, S1/S2 и контроль воспроизводимости на 3 датах.
Это контроли, не trials бюджета. Основной прогноз считается кодом `s5/d/engine_d` с lib='lgbm' (как d_lgbm_f4)."""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from models_v2.s4 import fit
from models_v2.s5.c import engine_c
from models_v2.s5.d import engine_d
from models_v2.s6 import engine6, paths6

LGB = paths6.OUT / "lightgbm"
SEEDS = (42, 43, 44, 45, 46)


def mk_world(L: dict) -> engine_c.W:
    return engine_c.W(L["F"], L["grid"], {"b2": L["dz0"]})


def fc_path(tag: str):
    return LGB / ("fc_lgbm_b2%s.parquet" % tag)


def _save(fc: pd.DataFrame, st: dict, tag: str) -> None:
    fc.to_parquet(fc_path(tag))
    (LGB / ("st_lgbm_b2%s.json" % tag)).write_text(json.dumps(st, default=float))


def run_main(p, w, L: dict, seed: int, force: bool = False) -> pd.DataFrame:
    tag = "" if seed == 42 else "_s%d" % seed
    if fc_path(tag).exists() and not force:
        return pd.read_parquet(fc_path(tag))
    fc, st = engine_d.daily(p, w, "b2", L["i2"], seed, "lgbm")
    st.update(seed=seed, lib="lightgbm", variant="base")
    _save(fc, st, tag)
    return fc


def run_variant(p, w, L: dict, variant: str, force: bool = False) -> pd.DataFrame:
    tag = "_" + variant
    if fc_path(tag).exists() and not force:
        return pd.read_parquet(fc_path(tag))
    t0 = time.time()
    df = engine6.run(p, w, L["i2"], L["grid"], 42, variant)
    fc = df[["d", "target_date", "sigma2", "rv"]]
    _save(fc, {"trees": float(df["k"].mean()), "secs": time.time() - t0, "variant": variant, "seed": 42, "lib": "lightgbm"}, tag)
    return fc


def control(L: dict) -> dict:
    """A9: Б-2 (xgb, engine_d.day_fit) на 3 датах побитно к fix_forecasts_c; LightGBM — два запуска побитно равны."""
    F, dz = L["F"], L["dz0"]
    fx = pd.read_parquet(paths6.OLD_DATA[0] / "fix_forecasts_c.parquet")
    fx = fx[fx.model == "fx_b2_g132_b"].set_index("d")["sigma2"]
    out = {}
    for ds in paths6.CONTROL_DATES:
        d = int(np.flatnonzero(F.dates == np.datetime64(ds))[0])
        hp = L["i2"][int(F.dates[d + 1].astype("datetime64[Y]").astype(int)) + 1970]
        sig = {}
        for lib in ("xgb", "lgbm", "lgbm"):
            m = engine_d.day_fit(F, dz, d, hp, fit.SEED, lib)
            sig.setdefault(lib, []).append(float(np.exp(dz.off[d] + m["predict"](dz.X[[d]])[0] + m["s2_es"] / 2)))
        ref = float(fx.loc[pd.Timestamp(ds)])
        out[ds] = {"xgb_sigma2": sig["xgb"][0], "fx_b2_sigma2": ref, "xgb_bitwise": sig["xgb"][0] == ref,
                   "lgbm_run1": sig["lgbm"][0], "lgbm_run2": sig["lgbm"][1], "lgbm_bitwise": sig["lgbm"][0] == sig["lgbm"][1]}
    paths6.dump_json(out, LGB / "CONTROL_A9.json")
    return out


def run_all(L: dict, control_only: bool = False) -> None:
    paths6.mkdirs()
    if control_only:
        r = control(L)
        bad = [k for k, v in r.items() if not (v["xgb_bitwise"] and v["lgbm_bitwise"])]
        print("A9:", "OK" if not bad else "ПРОВАЛ %s" % bad, json.dumps(r, default=float))
        return
    w = mk_world(L)
    with engine_d.pool(w, engine6.PROCS) as p:
        for s in SEEDS:
            fc = run_main(p, w, L, s)
            print("seed", s, "QLIKE", float(np.mean(fc["rv"] / fc["sigma2"] - np.log(fc["rv"] / fc["sigma2"]) - 1)), flush=True)
    with engine6.pool(w) as p2:
        for v in ("S1", "S2"):
            fc = run_variant(p2, w, L, v)
            print(v, len(fc), flush=True)
