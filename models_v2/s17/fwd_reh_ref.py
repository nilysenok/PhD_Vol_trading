"""Л5: опорные данные репетиции (архив) и запись строки сверки `rec` (протокол HARJ_HYBRID_SBER_FWD, раздел 7).

Архив только читается: таблицы `har_SBER`, `k23_SBER`, `x1`, `zb`, `zb0`, `cfg8` V7, артефакты `tcn_w5_2026_s*`; мир K2/K3, склад и состояния e_d/s_d архива и
полный ход cfg8 (λ по seeds блоков) пересчитываются замороженным кодом один раз. Новых данных не читает (cut <= 2026-02-27).
"""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
import torch  # noqa: F401  (первым: общий OpenMP, класс № 152)

from models_v2.s14 import v7_1b_sta as STA, v7_1m_feat as FE, v7_3_cfg as C8
from models_v2.s17 import fwd_cfg8 as F8, fwd_k23 as K, fwd_store as FS
from models_v2.s17.fwd_src import ARCHIVE_END, ROOT, Src

PAN = ROOT / "data/regen/harj_panel_hybrid"
EXPECT_REFIT = {(2025, 12): "2025-12-01", (2026, 1): "2026-01-05", (2026, 2): "2026-02-02"}   # раздел 7 протокола
TOL = 1e-12                                                                                    # объявленный допуск torch / xgboost / lightgbm
COLS = ["d", "t_plus", "node", "got", "ref", "bitwise", "relmax", "tol", "ok", "mode", "detail"]


def _s(v) -> str:
    a = np.asarray(v)
    if a.dtype.kind == "M":
        a = a.astype("datetime64[D]").astype(str)
    if a.size > 40:
        return "n=%d sha=%s" % (a.size, hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16])
    return repr(a.item()) if a.size == 1 else json.dumps(a.tolist())


def rec(d, tp, node: str, got, ref, tol: float, mode: str, detail: str = "") -> dict:
    """Строка (день × узел): побитное равенство, макс. относительное расхождение, ok при rel <= tol (tol = 0 — только побитно)."""
    g, r = np.asarray(got), np.asarray(ref)
    if g.dtype.kind in "fiub" and r.dtype.kind in "fiub" and g.shape == r.shape:
        g, r = g.astype(float), r.astype(float)
        nn = np.isnan(g) | np.isnan(r)
        same_nan = bool(np.array_equal(np.isnan(g), np.isnan(r)))
        bit = same_nan and bool(np.array_equal(g[~nn], r[~nn]))
        rel = float(np.max(np.abs(g[~nn] - r[~nn]) / np.maximum(np.abs(r[~nn]), 1e-300))) if (~nn).any() else 0.0
        ok = same_nan and rel <= tol
    else:
        bit = ok = bool(g.shape == r.shape and np.array_equal(g, r))
        rel = float("nan")
    return dict(d=pd.Timestamp(d).date(), t_plus=pd.Timestamp(tp).date(), node=node, got=_s(got), ref=_s(ref), bitwise=bit, relmax=rel, tol=tol, ok=bool(ok), mode=mode, detail=detail)


class Ref:
    """Архивные опоры: таблицы по датам, мир K2/K3, склад, состояния, λ cfg8 по блокам, z артефактов."""

    def __init__(self, src: Src):
        ix = lambda p, c: pd.read_parquet(p).set_index(c, drop=False)  # noqa: E731
        self.har, self.k23 = ix(PAN / "har_SBER.parquet", "d"), ix(PAN / "k23_SBER.parquet", "date")
        self.days = pd.read_parquet(PAN / "daily_SBER.parquet").index
        self.x1 = ix(ROOT / "data/regen/harj_deep/E1/x1.parquet", "d")
        self.zb, self.zb0 = (ix(ROOT / f"data/regen/harj_x1/Z/{n}.parquet", "d") for n in ("zb", "zb0"))
        self.cfg8 = pd.read_parquet(ROOT / "data/regen/harj_hybrid_v7/3/cfg8.parquet")
        self.world = K.build_world_at(ARCHIVE_END, src)
        store = FS.build_store(ARCHIVE_END, src)
        self.states = FS.build_states(ARCHIVE_END, src, store).set_index("d", drop=False)
        self.z = {s: STA.load(STA.directory() / STA.name("tcn", 2026, s)) for s in F8.SEEDS}
        inp = F8.inputs_at(ARCHIVE_END, src, store=store, xw=FS.build_xw5(store))
        self.tab = inp["tab"]
        _, per = C8.run_cfg(inp["tab"], FE.build_Z(inp["tab"]).to_numpy(float), lambda s: inp["zmap"][s], F8.SEEDS)
        self.fits = {pd.Timestamp(f["d0"]): [] for f in per[0]["fits"]}
        for p in per:
            for f in p["fits"]:
                self.fits[pd.Timestamp(f["d0"])].append(list(f["lam"]))

    def lam(self, d0) -> np.ndarray:
        """(seeds, 2): (λ_g, λ_LB) блока d0 — архивный полный ход."""
        return np.array(self.fits[pd.Timestamp(d0)], float)

    def z_at(self, d) -> np.ndarray:
        """(5, 4): z дня d из артефактов `tcn_w5_2026_s*` (поле z, дни — ns)."""
        out = []
        for s in F8.SEEDS:
            days = pd.DatetimeIndex(self.z[s]["days"].astype("datetime64[ns]"))
            out.append(self.z[s]["z"][days.get_loc(pd.Timestamp(d))])
        return np.array(out, float)

    def dz0_row(self, d) -> np.ndarray:
        return self.world["dz0"].X[self.world["F"].dates.astype("datetime64[ns]").searchsorted(np.datetime64(pd.Timestamp(d), "ns"))]
