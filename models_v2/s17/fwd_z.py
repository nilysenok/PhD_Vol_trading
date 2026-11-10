"""Л3: z энкодера TCN-w5 2026 (5 seeds) на склейке «архив + новые» — ТОЛЬКО из артефакта `stageA/tcn_w5_2026_s*.npz` (протокол, раздел 4).

Веса и PCA (`pca_mu/sd/comps/ssd`) берутся из артефакта как есть: энкодер не обучается, PCA не пересчитывается (период < 2027).
`content_hash` артефакта сверяется с CHECKPOINT V7 (`1b.stageA.hashes`); несовпадение — ошибка. z дня d — из окна 5 сессий <= d (`v7_1e_rep.encode`
и `transform_pca`, как в `z_pit`, но без `fit_pca_past`). Строки — те же дни, что у артефакта: `v7_1b_sta.rows_to_year_end(pack, 2026)`.
Новые дни (cut > 2026-02-27) требуют замка (бары новых файлов, `fwd_src`). Запись: `python3 -m models_v2.s17.fwd_z --write [--check]` — только архивные строки.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from models_v2.s14 import v7_1b_sta as STA, v7_1e_net as NET, v7_1e_rep as REP
from models_v2.s17 import fwd_store as FS
from models_v2.s17.fwd_src import ARCHIVE_END, ROOT, Src

KIND, YEAR, SEEDS = "tcn", 2026, (0, 1, 2, 3, 4)
CHECKPOINT = ROOT / "docs/otchety/harj_hybrid_v7/CHECKPOINT.json"
OUT = ROOT / "data/fwd/models"


def frozen_hash(seed: int) -> str:
    cp = json.loads(CHECKPOINT.read_text(encoding="utf-8"))
    return cp["1b"]["stageA"]["hashes"][STA.name(KIND, YEAR, seed)]


def load_artifact(seed: int) -> dict:
    """Артефакт (kind tcn, 2026, seed) с проверкой content_hash по CHECKPOINT V7."""
    art = STA.load(STA.directory() / STA.name(KIND, YEAR, seed))
    got = STA.content_hash(art)
    if got != frozen_hash(seed):
        raise ValueError(f"{STA.name(KIND, YEAR, seed)}: content_hash {got[:16]} ≠ CHECKPOINT {frozen_hash(seed)[:16]}")
    return art


def model_of(art: dict):
    m = NET.build(KIND)
    m.load_state_dict({k[3:]: torch.as_tensor(v) for k, v in art.items() if k.startswith("w__")})
    return m.eval()


def pca_of(art: dict) -> dict:
    return {k: np.asarray(art["pca_" + k], float) for k in ("mu", "sd", "comps", "ssd")}


def build_z(cut, src: Src, seed: int, store: dict | None = None, xw: dict | None = None) -> pd.DataFrame:
    """z (4 компоненты) дней склада <= cut: столбцы d, z0…z3. Энкодер и PCA — из артефакта (не обучаются)."""
    store = FS.build_store(cut, src) if store is None else store
    xw = FS.build_xw5(store) if xw is None else xw
    pack = dict(days=xw["days"], norm_ok=xw["norm_ok"], mask=xw["mask"])
    rows = STA.rows_to_year_end(pack, YEAR)
    art = load_artifact(seed)
    z = REP.transform_pca(pca_of(art), REP.encode(model_of(art), xw["X"][rows], xw["mask"][rows]))
    out = pd.DataFrame(z, columns=[f"z{i}" for i in range(z.shape[1])])
    out.insert(0, "d", pd.DatetimeIndex(xw["days"][rows]))
    return out


def main(argv: list) -> int:
    src = Src()
    store = FS.build_store(ARCHIVE_END, src)
    xw = FS.build_xw5(store)
    for s in SEEDS:
        z = build_z(ARCHIVE_END, src, s, store, xw)
        path = OUT / f"z_tcn2026_s{s}.npz"
        arrs = dict(z=z[[c for c in z.columns if c != "d"]].to_numpy(), days=z["d"].to_numpy("datetime64[ns]"), content_hash=np.array(frozen_hash(s)))
        print(f"seed {s}: {len(z)} строк {z['d'].iloc[0].date()}..{z['d'].iloc[-1].date()}")
        if "--write" in argv:
            OUT.mkdir(parents=True, exist_ok=True)
            tmp = Path(str(path.with_name("tmp_" + path.name)))
            np.savez_compressed(tmp, **arrs)
            tmp.replace(path)
        if "--check" in argv:
            old = np.load(path)
            assert all(np.array_equal(old[k], v) for k, v in arrs.items()), path.name
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
