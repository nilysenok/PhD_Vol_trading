"""S-D G1, G2: формула S-RCK по сохранённым θ и пересчёт слоёв над C_R. Только чтение сохранённых рядов; обучения нет.
Запуск из корня: PYTHONPATH=. python3 -W ignore <sp>/s_d/recon.py"""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_v2.s16 import x1_w3_core as C
from models_v2.s18 import cm_feat as CF
from models_v2.s18 import cm_layers as L
from models_v2.s18 import cm_model as CM

R = C.ROOT / "data/regen/harj_struct"
TH = ("gam_c", "gam_p", "th_c0", "th_p0", "th_c1", "th_p1")


def arrays():
    x, gap = L.inputs()
    V = CM.build_data(x, CF.load_feat()).view(np.arange(len(x)))
    g = lambda t: t.numpy()
    return x, gap, dict(c8=g(V.c8), xc=g(V.xc), xp=g(V.xp), zc=g(V.zc), zp=g(V.zp), sc=g(V.sc), sp=g(V.sp))


def mult(a, th):
    ec = th[:, 0] * a["xc"] + th[:, 2] + th[:, 4] * a["zc"]
    ep = th[:, 1] * a["xp"] + th[:, 3] + th[:, 5] * a["zp"]
    return 1.0 + a["sc"] * np.expm1(ec) + a["sp"] * np.expm1(ep)


def rel(a, b):
    return float(np.max(np.abs(np.asarray(a, float) / np.asarray(b, float) - 1.0)))


def main():
    x, gap, a = arrays()
    w5 = pd.read_parquet(C.ROOT / "data/regen/harj_x1/W5/w5g.parquet")
    for f in ("R/rck", "R/rckb", "R/rckc", "CM/ck"):
        df = pd.read_parquet(R / (f + ".parquet"))
        assert (pd.to_datetime(df["d"]).to_numpy() == x["d"].to_numpy()).all()
        th = df[list(TH)].to_numpy(float)
        Cr = a["c8"] * mult(a, th)
        print("G1 %s: max отн. |cfg8·M(θ) − C| = %.2e" % (f, rel(Cr, df["C"])))
        lay = L.layers(x, df["C"].to_numpy(float), gap)
        x1c = "X1_R" if f.startswith("R/") else "X1_V"
        print("G2 %s: F %.2e, X1 %.2e; дней π ≠ x1.pi: %d из %d; α₀ ≠: %d, α₁ ≠: %d; месяцев β ≠ w5g.β: %d из %d" % (
            f, rel(lay["F"], df["F"]), rel(lay["X1_V"], df[x1c]), int((lay["pi"].to_numpy() != x["pi"].to_numpy()).sum()), len(x),
            int((lay["a0"].to_numpy() != x["a0"].to_numpy()).sum()), int((lay["a1"].to_numpy() != x["a1"].to_numpy()).sum()),
            int(lay.assign(b=lay["beta_gap"].to_numpy() != w5["beta"].to_numpy()).dropna(subset=["refit_date"]).groupby("refit_date")["b"].any().sum()),
            int(lay["refit_date"].dropna().nunique())))


if __name__ == "__main__":
    main()
