"""Л4: ZB и ZB0 (поправка X1 по экстремальным парам e_d = 1: F = X1·exp(e_d·(b + a·s_d)); ZB0 — только b) SBER на склейке «архив + новые» (протокол, раздел 11).

Только замороженный `x1_z_walk.walk` (помесячная переоценка в первый день месяца: пул пар t⁺ ≤ D, только e = 1, ≥ 30 пар, λ ∈ {0; 0,001; 0,01; 0,1}, валидация — последние 60
экстремальных пар, L-BFGS-B) и признаки e_d, s_d Л3. Вход — ряд X1 (`fwd_x1`) и состояния склада; новая логика — только сборка D = (u, s, e, Z) как в `x1_z_stage.compute`.
Z = (e, x, s, v) строится для полноты D (ZB, ZB0 его не читают). Строка d = cut без цели (rv NaN) допустима последней: в пул не входит.
Запись: `python3 -m models_v2.s17.fwd_zb --write [--check]` — только архивные строки (x1.parquet Л4) в data/fwd/models/zb.parquet и zb0.parquet.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

import models_v2.s16  # noqa: F401  (torch первым)
from models_v2.s14 import v7_0_seq as SQ
from models_v2.s16 import x1_z_feat as FE, x1_z_walk as W
from models_v2.s17 import fwd_store as FS, fwd_x1 as X1
from models_v2.s17.fwd_src import ARCHIVE_END, ROOT, Src

OUT_DIR = ROOT / "data/fwd/models"
KINDS = {"ZB": "zb.parquet", "ZB0": "zb0.parquet"}


def r2_store(store: dict) -> dict:
    """Склад для `x_v`: days, r2 (из float32-склада, как `x1_ya_feat.load_store`), present."""
    a = FS.store_arrays(store)
    return dict(days=pd.DatetimeIndex(store["days"]), r2=np.asarray(a["raw"][:, :, list(SQ.CH).index("r2")], float), present=a["present"])


def inputs_d(x1: pd.DataFrame, states: pd.DataFrame, store: dict) -> dict:
    """D = (u = rv/X1, s = s_d, e = e_d, Z) на датах ряда X1."""
    d = pd.DatetimeIndex(x1["d"])
    st = states.set_index("d").reindex(d)
    e, s = st["e"].to_numpy(int), st["s"].to_numpy(float)
    x, v, _ = FE.x_v(r2_store(store), d)
    return dict(u=x1["rv"].to_numpy(float) / x1["F"].to_numpy(float), s=s, e=e, Z=FE.raw_z(e, x, s, v))


def build_zb(cut, src: Src, x1: pd.DataFrame, states: pd.DataFrame | None = None, store: dict | None = None, kind: str = "ZB") -> pd.DataFrame:
    """Прогноз ZB (kind = "ZB") или ZB0 на строках ряда X1 (последняя — d = cut): колонки `x1_z_walk.OUT_COLS`."""
    if kind not in KINDS:
        raise ValueError(f"fwd_zb: только ZB и ZB0, не {kind}")
    store = FS.build_store(cut, src) if store is None else store
    states = FS.build_states(cut, src, store) if states is None else states
    return W.walk(x1[["d", "t_plus", "rv", "F"]], inputs_d(x1, states, store), kind)


def state_of(zb: pd.DataFrame) -> dict:
    """Параметры на последней строке: λ, b, a, число экстремальных пар пула."""
    r = zb.iloc[-1]
    return dict(lam=float(r["lam"]), b=float(r["b"]), a=float(r["a"]), n_ext=int(r["n_ext"]), active=int(r["active"]))


def main(argv: list) -> int:
    src = Src()
    store = FS.build_store(ARCHIVE_END, src)
    states, x1 = FS.build_states(ARCHIVE_END, src, store), pd.read_parquet(X1.OUT)
    for kind, name in KINDS.items():
        zb = build_zb(ARCHIVE_END, src, x1, states, store, kind)
        print(f"{kind} archive rows={len(zb)} d {zb['d'].iloc[0].date()}..{zb['d'].iloc[-1].date()}")
        if "--write" in argv:
            OUT_DIR.mkdir(parents=True, exist_ok=True)
            tmp = Path(str(OUT_DIR / name) + ".tmp")
            zb.to_parquet(tmp, index=False)
            tmp.replace(OUT_DIR / name)
        if "--check" in argv:
            pd.testing.assert_frame_equal(pd.read_parquet(OUT_DIR / name), zb)
            pd.testing.assert_frame_equal(zb, pd.read_parquet(ROOT / "data/regen/harj_x1/Z" / name))
            print(f"{name} == пересчёт == архив Z: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
