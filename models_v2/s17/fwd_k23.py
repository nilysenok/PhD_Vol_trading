"""Л2: K2 (XGBoost `x2_best_j`) и K3 (LightGBM `C0_inh_F4`) SBER на склейке «архив + новые» (протокол HARJ_HYBRID_SBER_FWD, раздел 11).

`build_k23(cut, src)` — прогноз σ̂² на строку d = cut (или на `dates`) по барам с begin <= cut 18:40: SBER, 16 бумаг `prov4.PAPERS`, IMOEX.
Всё — замороженные функции архива (`s10/hp_k23_world.build_world`, `hp_k23_calc.forecast_frame` → `c_calc.block_job`, `job.run_days`; seed 42, дневная
переобучка, ранняя остановка И1); новых алгоритмов нет. И2 по годам читается из `k23_meta_SBER.json` и НЕ пересчитывается (период < 01.01.2027).
Две обвязки (архивный код не правится): (1) `index_rule` — на время построения мира подставляет в `daily._IDAYS` дни сессий IMOEX склейки
(решение 8: путь `IMOEX_PATH` архивный; `build_daily("auto")` внутри замороженных функций иначе видел бы только архив), с восстановлением в `finally`;
(2) `pad_world` — строка-заглушка t⁺ (дата, X/y/rv = NaN) после последней строки: у обрезанного на d мира цели строки d ещё нет, а `rows.year_of`
берёт год HP по дате цели. t⁺ — следующий канонический день из календаря Л1 (только begin; календарь биржи, не цены). Заглушка в обучение не входит.
Новые строки (t⁺ > 2026-02-27) требуют замка: `Src.bars` до него бросает SealedError. `--write` пишет только архивные строки (t⁺ <= 2026-02-27).
"""
from __future__ import annotations

import json
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd

from models_v2 import daily as D, har
from models_v2.s2 import features as ft
from models_v2.s4 import design
from models_v2.s10 import hp_k23_calc as KC, hp_k23_world as KW
from models_v2.s17 import fwd_cal
from models_v2.s17.fwd_src import ARCHIVE_END, LAST_SESSION, ROOT, Src, limit_of

META = ROOT / "data/regen/harj_panel_hybrid/k23_meta_SBER.json"
OUT = ROOT / "data/fwd/models/k23.parquet"
TICKER = "SBER"


def load_i2() -> dict:
    """И2 по годам 2017…2026 из метаданных архива (не пересчитывается)."""
    return KW.i2_load(json.loads(META.read_text())["i2"])


@contextmanager
def index_rule(idays):
    """На время построения мира `daily.build_daily("auto")` видит дни сессий IMOEX склейки; прежнее значение восстанавливается всегда."""
    old = D._IDAYS
    D._IDAYS = set(idays)
    try:
        yield
    finally:
        D._IDAYS = old


def build_world_at(cut, src: Src) -> dict:
    """Мир SBER (F, grid, dz0 = Б-2 + рыночный блок 17 бумаг + IMOEX, feats F1–F6, eps) по барам <= cut 18:40; без заглушки."""
    with index_rule(src.index_days(cut)):
        return KW.build_world(src.bars(TICKER, cut), src.bars("IMOEX", cut), KW.pool_papers(lambda t: src.bars(t, cut)))


def next_day(last: pd.Timestamp, src: Src):
    """Следующий канонический день после `last` по календарю Л1 (только begin); None, если не определён (последний день склейки)."""
    cal = fwd_cal.build_cal(LAST_SESSION, src)
    tp = cal.loc[cal["date"] == last, "t_plus"]
    return tp.iloc[0] if len(tp) and pd.notna(tp.iloc[0]) else None


def _nan_row(a: np.ndarray) -> np.ndarray:
    return np.vstack([a, np.full((1, a.shape[1]), np.nan)])


def pad_world(w: dict, t_next: pd.Timestamp) -> dict:
    """Добавляет строку-заглушку (дата t⁺, NaN) ко всем массивам мира и пересчитывает сетку тем же `har.eligible`."""
    F = w["F"]
    F2 = ft.Fp(np.append(F.dates, t_next.to_datetime64()), np.append(F.seg, F.seg[-1]), np.append(F.rv, np.nan), _nan_row(F.X))
    dz = lambda z: design.Dz(_nan_row(z.X), np.append(z.y, np.nan), np.append(z.off, np.nan), z.mono)  # noqa: E731
    fe = w["feats"]
    feats = pd.concat([fe, pd.DataFrame(np.nan, index=pd.DatetimeIndex([t_next], name=fe.index.name), columns=fe.columns)])
    grid = har.eligible(har.Panel(F2.dates, F2.seg, F2.rv, F2.X[:, :4]))
    return {**w, "F": F2, "grid": np.asarray(grid, int), "dz0": dz(w["dz0"]), "lvl": dz(w["lvl"]), "feats": feats, "eps": np.append(w["eps"], np.nan)}


def build_k23(cut, src: Src, dates=None, procs: int = 2, t_next=None) -> pd.DataFrame:
    """Строки (date, t_plus, rv, sigma2_k2, sigma2_k3, hp_year, hp_k2, hp_k3). dates: None — только d = cut; "all" — вся сетка; иначе список дат d.
    Для строки d = последнего дня склейки rv = NaN (цель неизвестна), t_plus — из календаря."""
    w = build_world_at(cut, src)
    nxt = t_next if t_next is not None else next_day(pd.Timestamp(w["F"].dates[-1]), src)
    if nxt is not None:
        w = pad_world(w, pd.Timestamp(nxt))
    i2 = load_i2()
    ds = None if isinstance(dates, str) else pd.DatetimeIndex([cut] if dates is None else dates)
    return KC.forecast_frame(w, i2, KC.select_rows(w, min(i2), ds), procs)


def availability(cut, src: Src) -> dict:
    """Для каждого из 18 инструментов — максимальный begin бара основной сессии, вошедшего в признаки дня cut (должен быть <= cut 18:40); NaT — баров нет."""
    out = {}
    for t in KW.POOL + ["IMOEX"]:
        b = D.main_session(src.begins(t, cut))["begin"]
        out[t] = b.max() if len(b) else pd.NaT
    return out


def versions() -> dict:
    import lightgbm
    import xgboost
    return {"python": sys.version.split()[0], "numpy": np.__version__, "pandas": pd.__version__, "xgboost": xgboost.__version__,
            "lightgbm": lightgbm.__version__, "threads_xgb": 1, "threads_lgbm": 1, "procs": 2, "limit_of_example": str(limit_of("2026-02-26"))}


def main(argv: list) -> int:
    src = Src()
    fr = build_k23(ARCHIVE_END, src, dates="all")
    fr = fr[fr["t_plus"] <= ARCHIVE_END].reset_index(drop=True)
    print(f"k23 archive rows={len(fr)} d {fr['date'].min().date()}..{fr['date'].max().date()} t+max={fr['t_plus'].max().date()}\n{versions()}")
    if "--write" in argv:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        tmp = Path(str(OUT) + ".tmp")
        fr.to_parquet(tmp)
        tmp.replace(OUT)
    if "--check" in argv:
        pd.testing.assert_frame_equal(pd.read_parquet(OUT), fr)
        print("k23.parquet == пересчёт: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
