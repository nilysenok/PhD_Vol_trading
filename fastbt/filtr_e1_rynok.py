"""Ось `filtr`: фильтры по рынку и единая точка входа `razresheno` (контракт etap1 §2.1; Л-4).

`razresheno(imya, t, inp, sd, dates)` → bool «вход разрешён» по дням тикера или None («не считана»).
Ряды тикера — `inp.sk._td[t][0]`, σ — `inp.sigma["rv5|rv22|rv66"][t]`, события — `inp.sobytiya`,
IMOEX — `rezhim_e1.imoex_daily()` (кэш на `inp`). Только данные ≤ close d; нет истории → запрет.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

from fastbt import e1_sektor, filtr_e1 as f, rezhim_e1

ZNACHENIYA = ("net", "cbr", "div", "post_shok", "szhatie", "rv_perevorot", "obyom", "rel_sila_imoex", "sektor_trend")
BEZ_SEKTORA = ("AFLT", "LSRG")
SEKTOR_PT = {"sektor_trend": "po_trendu", "shirina": "net", "ma_n": 200}   # центр итерации 4


def _imoex(inp, dates) -> tuple[np.ndarray, np.ndarray]:
    """(R_d, Z_d) на датах тикера: R = ln-доходность close IMOEX, Z = R_d / sig_mkt_{d−1}; нет даты → NaN."""
    im = getattr(inp, "_filtr_imoex", None)
    if im is None:
        im = rezhim_e1.imoex_daily()
        try:
            inp._filtr_imoex = im
        except AttributeError:
            pass
    s = im.set_index(pd.DatetimeIndex(im["date"]))
    r = pd.Series(f.ret(s["close"].to_numpy()), index=s.index)
    sg = pd.Series(f.prev(s["sig_mkt"].to_numpy()), index=s.index)
    ix = pd.DatetimeIndex(dates)
    R, g = r.reindex(ix).to_numpy(), sg.reindex(ix).to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        return R, R / g


def rel_series(r: np.ndarray, R: np.ndarray, sig22: np.ndarray) -> np.ndarray:
    """Σ_{5 дней}(r − R) / (σ̂_{d−1}·√5)."""
    x = pd.Series(np.asarray(r, float) - np.asarray(R, float)).rolling(5, min_periods=5).sum().to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        return x / (f.prev(sig22) * np.sqrt(5))


def rel_sila_imoex(rel: np.ndarray, sd: np.ndarray) -> np.ndarray:
    """sd · rel ≥ 1 (по силе)."""
    with np.errstate(invalid="ignore"):
        return np.asarray(sd, float) * rel >= 1


def _sektor_ctx(inp, dates):
    ctx = getattr(inp, "_filtr_sektor_ctx", None)
    if ctx is None:
        idx = e1_sektor.load_indexes(sorted(set(pd.read_csv(e1_sektor.SEKTORY_CSV)["indeks"])))
        close = {k: pd.Series(v[0].close, index=pd.DatetimeIndex(v[0].dates)) for k, v in inp.sk._td.items()}
        ctx = e1_sektor.Ctx(idx, close, e1_sektor.sektor_map(e1_sektor.SEKTORY_CSV, idx))
        try:
            inp._filtr_sektor_ctx = ctx
        except AttributeError:
            pass
    return ctx


def _td_local(inp, t: str, dates: np.ndarray) -> SimpleNamespace:
    """Ряды тикера на локальной сетке `dates` (σ, sd, сигнал живут на ней: с inp.meta[t]['s0'], до конца окна).
    `sk._td[t]` — полный ряд (с прогрева, до конца разработки): режем по s0 и сверяем даты, а не обрезаем по длине."""
    td, n, s0 = inp.sk._td[t][0], len(dates), int(inp.meta[t]["s0"])
    if not np.array_equal(np.asarray(td.dates[s0:s0 + n], "datetime64[D]"), np.asarray(dates, "datetime64[D]")):
        raise ValueError(f"{t}: даты sk._td[s0:s0+n] не совпали с локальной сеткой dates — отказ")
    return SimpleNamespace(**{k: np.asarray(getattr(td, k))[s0:s0 + n] for k in ("close", "high", "low", "volume")})


def razresheno(imya: str, t: str, inp, sd: np.ndarray, dates: np.ndarray) -> np.ndarray | None:
    if imya not in ZNACHENIYA:
        raise ValueError(f"filtr: {imya!r} вне {ZNACHENIYA}")
    n = len(dates)
    sd = np.asarray(sd, float)
    if imya == "net":
        return np.ones(n, bool)
    if imya in ("cbr", "div"):
        ev = inp.sobytiya.get(imya)
        return None if ev is None else ~np.asarray(ev[t], bool)
    td = _td_local(inp, t, dates)
    s22 = inp.sigma["rv22"][t]
    r = f.ret(td.close)
    z = f.z_series(r, s22)
    if imya == "post_shok":
        return f.post_shok(z, _imoex(inp, dates)[1], sd)
    if imya == "szhatie":
        return f.szhatie(f.rg_series(td.high, td.low, td.close))
    if imya == "rv_perevorot":
        rv5 = inp.sigma["rv5"][t]
        return f.rv_perevorot(f.u_series(rv5, inp.sigma["rv66"][t]), rv5)
    if imya == "obyom":
        return f.obyom(f.v_series(td.volume), z)
    if imya == "rel_sila_imoex":
        return rel_sila_imoex(rel_series(r, _imoex(inp, dates)[0], s22), sd)
    if t in BEZ_SEKTORA:
        return np.ones(n, bool)
    lo, sr = e1_sektor.allow(SEKTOR_PT, t, dates, _sektor_ctx(inp, dates))
    return np.where(sd > 0, lo, np.where(sd < 0, sr, False))
