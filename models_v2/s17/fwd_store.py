"""Л3: склад M10 и окна X_w5 на склейке «архив + новые», состояния e_d и s_d (протокол HARJ_HYBRID_SBER_FWD, раздел 11).

Только импорт замороженных функций: склад — `v7_0_seq.build_store` (float32 при записи, как `run_v7.write_stage`), окна — `run_v7.sequences`,
e_d — `de_e1_state.states` (порог — 99-й перцентиль |r| баров строго до d), s_d — `x1_ya_feat.features`. Ничего не обучает.
Дни — канонические (`daily.build_daily`, правило IMOEX через `src.index_days`); бары — `src.bars(·, cut)`: begin <= cut 18:40.
До замка новые бары цен не отдают (SealedError): без замка считаются только архивные строки (cut <= 2026-02-27).
e_d и s_d считаются по float32-складу, как у архивных загрузчиков (`de0_data.load_store`, `x1_ya_feat.load_store`).
Запись: `python3 -m models_v2.s17.fwd_store --write [--check]` — ТОЛЬКО архивные строки (cut = 2026-02-27) в data/fwd/models/.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from models_v2 import daily as D
from models_v2.s14 import run_v7 as RV, v7_0_seq as SQ
from models_v2.s15 import de_e1_state as E1
from models_v2.s16 import x1_ya_feat as YF
from models_v2.s17 import fwd_cal as FC
from models_v2.s17.fwd_src import ARCHIVE_END, ROOT, Src

OUT = ROOT / "data/fwd/models"
W = 5


def build_store(cut, src: Src) -> dict:
    """Склад (days, raw (D, 53, 11) float64 с NaN, present, cn, cs, css) по дням SBER <= cut."""
    bars = src.bars("SBER", cut)
    days = D.build_daily(bars, index_days=src.index_days(cut))[0].index
    return SQ.build_store(bars, days)


def store_arrays(store: dict) -> dict:
    """Содержимое `store_raw.npz`: raw float32, present, days, channels."""
    return dict(raw=store["raw"].astype(np.float32), present=store["present"], days=store["days"].to_numpy("datetime64[ns]"), channels=np.array(SQ.CH))


def build_xw5(store: dict) -> dict:
    """Окна 5 сессий: X (float32), mask, day_ok, norm_ok, days, channels — как `X_w5.npz`."""
    return RV.sequences(store, W)


def _view(store: dict) -> tuple:
    """r (D, 53) и r2 (D, 53) из float32-склада — выражения архивных загрузчиков `de0_data.load_store` и `x1_ya_feat.load_store`."""
    a, ch = store_arrays(store), list(SQ.CH)
    r = np.where(a["present"], a["raw"][:, :, ch.index("r")], np.nan).astype(float)
    r2 = np.asarray(a["raw"][:, :, ch.index("r2")], float)
    return r, r2


def build_states(cut, src: Src, store: dict | None = None) -> pd.DataFrame:
    """Строка на каждый день склада: d, t_plus (из календаря по begin; последний день — NaT), e (e_d), s (s_d), n, nlast, zero, rv_d, delta."""
    store = build_store(cut, src) if store is None else store
    days = pd.DatetimeIndex(store["days"])
    r, r2 = _view(store)
    e = E1.states(dict(days=days, r=r), days)
    cal = FC.build_cal(cut, src).set_index("date")["t_plus"].reindex(days)
    f = YF.features(dict(days=days, r2=r2, present=store["present"]), days, pd.DatetimeIndex(cal.to_numpy()))
    return pd.DataFrame({"d": days.as_unit("ns"), "t_plus": pd.DatetimeIndex(cal.to_numpy()).as_unit("ns"), "e": e, "s": f["s"], "n": f["n"], "nlast": f["nlast"], "zero": f["zero"], "rv_d": f["rv_d"], "delta": f["delta"]})


def same(a, b) -> bool:
    """Побитное равенство массивов (NaN по месту равны)."""
    a, b = np.asarray(a), np.asarray(b)
    return a.dtype == b.dtype and np.array_equal(a, b, equal_nan=a.dtype.kind == "f")


def _write(path: Path, fn) -> None:
    tmp = path.with_name("tmp_" + path.name)
    fn(tmp)
    tmp.replace(path)


def archive_outputs(src: Src | None = None) -> tuple:
    """-> (store, store_arrays, X_w5, state_es) по cut = 2026-02-27."""
    src = src or Src()
    store = build_store(ARCHIVE_END, src)
    return store, store_arrays(store), build_xw5(store), build_states(ARCHIVE_END, src, store)


def main(argv: list) -> int:
    store, arr, xw, st = archive_outputs()
    print(f"архив: дней {len(store['days'])} {store['days'][0].date()}..{store['days'][-1].date()}; X_w5 {xw['X'].shape}; e_d=1: {int(st['e'].sum())}")
    if "--write" in argv:
        OUT.mkdir(parents=True, exist_ok=True)
        _write(OUT / "store.npz", lambda p: np.savez_compressed(p, **arr))
        _write(OUT / "X_w5.npz", lambda p: np.savez_compressed(p, **xw))
        _write(OUT / "state_es.parquet", lambda p: st.to_parquet(p))
    if "--check" in argv:
        a, x = np.load(OUT / "store.npz"), np.load(OUT / "X_w5.npz")
        assert all(same(a[k], arr[k]) for k in arr), "store.npz"
        assert all(same(x[k], xw[k]) for k in xw), "X_w5.npz"
        pd.testing.assert_frame_equal(pd.read_parquet(OUT / "state_es.parquet"), st)
        print("store.npz, X_w5.npz, state_es.parquet == пересчёт: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
