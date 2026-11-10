"""Л4: X1 (E1 = D1: F = M^(1−π)·C^π, M = √(K2·K3), C = cfg8, π по состоянию e_d) SBER на склейке «архив + новые» — тонкий загрузчик (протокол, раздел 11).

Только замороженные `de_e1_state.run` (π и η — ежедневно по закрытым парам t⁺ ≤ d, α — помесячно по последним 126 закрытым парам состояния) и e_d из Л3
(`fwd_store.build_states`, `de_e1_state.states`). Входы — ряды на общих датах cfg8: d, t⁺, rv, C = F cfg8; M — из таблицы `fwd_cfg8.make_tab`.
Новая логика отсутствует: `build_x1(cfg8, tab, states)` = `de_e1_stage.compute` без чтения файлов. Строка d = cut без цели (rv NaN) допустима последней: в пары не входит.
Запись: `python3 -m models_v2.s17.fwd_x1 --write [--check]` — только архивные строки (cfg8.parquet Л4) в data/fwd/models/x1.parquet.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from models_v2.s15 import de_e1_state as S
from models_v2.s17 import fwd_cfg8 as F8, fwd_store as FS
from models_v2.s17.fwd_src import ARCHIVE_END, ROOT, Src

OUT = ROOT / "data/fwd/models/x1.parquet"
COLS = ["d", "t_plus", "rv", "M", "C", "s", "pi", "F", "pi0", "pi1", "a0", "a1"]


def run_x1(cfg8: pd.DataFrame, tab: pd.DataFrame, states: pd.DataFrame) -> tuple:
    """-> (x1 DataFrame, результат `S.run`). cfg8: d, t_plus, rv, F; tab: K2, K3 на тех же датах; states: d, e (e_d) на днях склада."""
    idx = pd.DatetimeIndex(cfg8["d"])
    k2, k3 = tab["K2"].reindex(idx).to_numpy(float), tab["K3"].reindex(idx).to_numpy(float)
    M, C, tp = np.sqrt(k2 * k3), cfg8["F"].to_numpy(float), pd.DatetimeIndex(cfg8["t_plus"])
    rv = cfg8["rv"].to_numpy(float)
    e = states.set_index("d")["e"].reindex(idx)
    if e.isna().any() or not np.isfinite(M).all():
        raise ValueError("X1: нет e_d или K2/K3 на датах cfg8")
    s = e.to_numpy(int)
    r = S.run(rv, M, C, s, idx.to_numpy(), tp.to_numpy())
    x1 = pd.DataFrame(dict(d=idx, t_plus=tp, rv=rv, M=M, C=C, s=s, pi=r["pi"], F=r["F"], pi0=r["pis"][:, 0], pi1=r["pis"][:, 1], a0=r["a0"], a1=r["a1"]))[COLS]
    if not (np.isfinite(r["F"]).all() and (r["F"] > 0).all()):
        raise ValueError("X1: F не положителен / не конечен")
    return x1, r


def build_x1(cut, src: Src, cfg8: pd.DataFrame, tab: pd.DataFrame, states: pd.DataFrame | None = None) -> pd.DataFrame:
    """X1 на строках cfg8 (последняя — d = cut). states по умолчанию — `fwd_store.build_states(cut, src)`."""
    return run_x1(cfg8, tab, FS.build_states(cut, src) if states is None else states)[0]


def state_of(r: dict) -> dict:
    """Состояние на последней строке: π₀, π₁ (после пар, закрытых к d), η, α₀, α₁."""
    return dict(pi0=float(r["pis"][-1, 0]), pi1=float(r["pis"][-1, 1]), eta=float(r["eta"][-1]), a0=float(r["a0"][-1]), a1=float(r["a1"][-1]))


def main(argv: list) -> int:
    src = Src()
    cfg8 = pd.read_parquet(F8.OUT)
    x1 = build_x1(ARCHIVE_END, src, cfg8, F8.tab_at(ARCHIVE_END, src))
    print(f"x1 archive rows={len(x1)} d {x1['d'].iloc[0].date()}..{x1['d'].iloc[-1].date()}")
    if "--write" in argv:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        tmp = Path(str(OUT) + ".tmp")
        x1.to_parquet(tmp, index=False)
        tmp.replace(OUT)
    if "--check" in argv:
        pd.testing.assert_frame_equal(pd.read_parquet(OUT), x1)
        pd.testing.assert_frame_equal(x1, pd.read_parquet(ROOT / "data/regen/harj_deep/E1/x1.parquet"))
        print("x1.parquet == пересчёт == архив E1: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
