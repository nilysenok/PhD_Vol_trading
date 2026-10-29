"""R-г, период В2 (Ж-113): 2015–2019, прогрев с 06.2014 — год прогрева не попадает в метрики (синтетика, без данных)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

R = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(R / "scripts"))
sys.path.insert(0, str(R / "scripts" / "revers"))
import diag_timeframes as dt  # noqa: E402
import r1_g  # noqa: E402


def _ryad():
    dates = pd.bdate_range("2014-06-09", "2020-06-01")
    c = 100 + np.cumsum(np.random.default_rng(0).normal(size=len(dates)))
    f = pd.DataFrame({"date": dates, "open": c, "high": c + 1, "low": c - 1, "close": c, "volume": 1000})
    f["day_idx"] = np.arange(len(dates))
    return f, c


def test_granicy_v2_i_cfg():
    assert dt.PERIODS["v2"] == ("2015-01-01", "2020-01-01", "2014-06-01")
    assert dt.PERIODS["train4"] == ("2016-01-01", "2020-01-01", "2015-01-01")      # train4 не тронут
    assert r1_g.CFG["v2"][:2] == (100, 5)                                           # 5 лет в sig_v_god


def test_v2_signaly_tolko_2015_2019_a_god_progreva_ne_v_metrikah():
    f, c = _ryad()
    a, b = (pd.Timestamp(x) for x in dt.PERIODS["v2"][:2])
    d = dt.one(f, c, "kontrtrend", "kazhdyj_bar", np.random.default_rng(1), a, b, 3)[0]
    assert len(d) > 0 and d.min() >= np.datetime64("2015-01-01") and d.max() < np.datetime64("2020-01-01")
    # мутация: окно оценки начато с прогрева (2014-06) — год прогрева попадает в метрики, тест обязан это увидеть
    dm = dt.one(f, c, "kontrtrend", "kazhdyj_bar", np.random.default_rng(1), pd.Timestamp(dt.PERIODS["v2"][2]), b, 3)[0]
    assert dm.min() < np.datetime64("2015-01-01")


def test_worker_v2_beryot_okno_iz_periods(monkeypatch):
    f, c = _ryad()
    monkeypatch.setattr(dt, "frames", lambda ticker, per="train4": ({k: f for k in dt.TFS}, c))
    res = dt.worker((0, "X", "v2", 2))
    d = res[("den", "kontrtrend", "den_1850")][0]
    assert len(d) > 0 and d.min() >= np.datetime64("2015-01-01") and d.max() < np.datetime64("2020-01-01")


def test_vyhod_v2_v_progon_e1():
    assert r1_g.OUT_DIR == {"v2": ("progon_e1", "timeframes_v2.csv")}      # train4/test1 пишутся как раньше, без шапки
