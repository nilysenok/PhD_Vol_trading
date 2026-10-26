"""23. Кэши по хэшу: изменение входа меняет ключ (FA2b).

- Сигналы (fastbt.sigprov.get_signals → fastbt.cache): ключ = код пакета + td.digest() (даты и
  OHLCV). Правка любого бара, даты или объёма меняет ключ; σ во вход сигналов не входит.
- Кэш С9 (fastbt.s9.cache_key: позиции A и C основного дизайна): ключ обязан меняться при правке
  σ-источника, файла баров (A и C строятся по OHLCV) и data/corporate_actions.csv (дизайн
  включает корпоративные события).
- Кэш масок C / затворов D (fastbt.masks._cached): маски строятся из td.sigma — ключ обязан
  различать σ (до исправления FA2b-Н2 — td.digest() без σ; s9 чистил кэш вручную).
"""
import dataclasses
import hashlib

import numpy as np
import pandas as pd
import pytest

from synth_fa2b import bars
from fastbt import cache as fcache
from fastbt import data as fd
from fastbt import masks, s9, sigprov
from fastbt.indicators import compute_base
from fastbt.signals.interface import TickerData


@pytest.fixture
def keys(monkeypatch):
    seen = []
    monkeypatch.setattr(fcache, "load", lambda k: seen.append(k))      # не читаем общий кэш
    monkeypatch.setattr(fcache, "save", lambda k, a: None)             # и не пишем в него
    return seen


def _key(seen, td):
    sigprov.get_signals("S1", td, compute_base(td.close, td.high, td.low, td.open, td.volume))
    return seen[-1]


def _edit(td, field, i, f=1.001):
    a = np.asarray(getattr(td, field)).copy()
    if field == "dates":
        a[i] = a[i] + np.timedelta64(1, "h")
    else:
        a[i] *= f
    return dataclasses.replace(td, **{field: a})


def test_signal_key_tracks_bars(keys):
    td = bars(n=260)
    k0 = _key(keys, td)
    assert _key(keys, dataclasses.replace(td)) == k0                   # тот же вход — тот же ключ
    for field in ("dates", "open", "high", "low", "close", "volume"):
        assert _key(keys, _edit(td, field, 250)) != k0, field
    sg = {h: v * 1.5 for h, v in td.sigma.items()}
    assert _key(keys, dataclasses.replace(td, sigma=sg)) == k0         # σ — не вход сигналов


def _s9_env(tmp_path, monkeypatch):
    src = {}
    for lab, col in (("hyb_h1", "sigma_pred_h1"), ("hyb_h5", "sigma_pred_h5")):
        p = tmp_path / f"{lab}.parquet"
        pd.DataFrame({"date": pd.bdate_range("2019-01-01", periods=5), "ticker": "ZZZ",
                      col: np.arange(5, dtype=float)}).to_parquet(p)
        src[lab] = (p, col, lab[-2:])
    monkeypatch.setattr(s9, "SOURCES", src)
    ohlcv = tmp_path / "ohlcv.parquet"
    pd.DataFrame({"date": pd.bdate_range("2019-01-01", periods=5), "ticker": "ZZZ",
                  "close": np.arange(5.0)}).to_parquet(ohlcv)
    corp = tmp_path / "corp.csv"
    corp.write_text("ticker,date,factor,status\nZZZ,2019-01-03,0.5,applied\n")
    monkeypatch.setattr(fd, "OHLCV", ohlcv)
    monkeypatch.setattr(fd, "CORP", corp)
    return src, ohlcv, corp


def test_s9_key_tracks_sigma(tmp_path, monkeypatch):
    src, _, _ = _s9_env(tmp_path, monkeypatch)
    k0 = s9.cache_key()
    assert s9.cache_key() == k0
    p, col, _ = src["hyb_h5"]
    d = pd.read_parquet(p)
    d.loc[4, col] = 99.0
    d.to_parquet(p)
    assert s9.cache_key() != k0


def test_s9_key_tracks_bars_and_corp(tmp_path, monkeypatch):
    _, ohlcv, corp = _s9_env(tmp_path, monkeypatch)
    k0 = s9.cache_key()
    d = pd.read_parquet(ohlcv)
    d.loc[4, "close"] = 99.0
    d.to_parquet(ohlcv)
    k1 = s9.cache_key()
    assert k1 != k0                                                    # правка бара
    corp.write_text("ticker,date,factor,status\nZZZ,2019-01-03,0.25,applied\n")
    assert s9.cache_key() != k1                                        # правка корп. события


def test_mask_cache_tracks_sigma(monkeypatch):
    from fastbt import design as dsg
    dsg.apply(())
    td = bars(n=420)
    td2 = dataclasses.replace(td, sigma={h: v[::-1].copy() for h, v in td.sigma.items()})
    masks._CACHE.clear()
    try:
        masks.c_masks(td, "Trend", False)
        masks.d_gates(td, True)
        got_c = masks.c_masks(td2, "Trend", False)[0]
        got_d = masks.d_gates(td2, True)[0]
        ref_c = masks._c_masks(td2, "Trend", False)[0]
        ref_d = masks._d_gates(td2, True)[0]
    finally:
        masks._CACHE.clear()
    assert np.array_equal(got_c, ref_c)
    assert np.array_equal(got_d, ref_d)


# ---------- мутации ----------

def _key_no_sigma():
    from fastbt.cache import code_hash
    h = hashlib.sha1(code_hash().encode())
    h.update(s9.MAIN_YAML.read_bytes())
    return h.hexdigest()[:10]


def _key_no_bars():
    """Ключ до исправления FA2b-Н1: код, main.yaml, σ — без баров и корп. событий."""
    from pathlib import Path
    h = hashlib.sha1(_key_no_sigma().encode())
    for lab, (p, _, _) in s9.SOURCES.items():
        h.update(lab.encode() + hashlib.sha1(Path(p).read_bytes()).digest())
    return h.hexdigest()[:10]


MUTATIONS = {
    "sig_key_no_bars": lambda mp: mp.setattr(
        TickerData, "digest", lambda self: hashlib.sha1(self.ticker.encode()).hexdigest()),
    "s9_key_no_sigma": lambda mp: mp.setattr(s9, "cache_key", _key_no_sigma),
    "s9_key_no_bars": lambda mp: mp.setattr(s9, "cache_key", _key_no_bars),
    "mask_key_no_sigma": lambda mp: mp.setattr(masks, "_sigma_digest", lambda td: ""),
}
