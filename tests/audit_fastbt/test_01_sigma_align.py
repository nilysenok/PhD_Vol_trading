"""1. Выравнивание σ с барами: решение на закрытии d, позиция d→d+1, σ не знает d+1.

Бар d получает последнюю строку σ с датой ≤ d (s9.align, data.align_sigma); доходность бара
log_ret[d] = log(close[d+1]/close[d]) — позиция d зарабатывает d→d+1.
"""
import numpy as np
import pandas as pd
import pytest

from conftest import SIGMA_H1, read_dev
from fastbt import data as fd
from fastbt import s9

BARS = pd.to_datetime(["2021-03-01", "2021-03-02", "2021-03-03", "2021-03-04", "2021-03-05",
                       "2021-03-09"]).values.astype("datetime64[ns]")
SRC = pd.DataFrame({"date": pd.to_datetime(["2021-03-01", "2021-03-03", "2021-03-06",
                                            "2021-03-09", "2021-03-10"]),
                    "sigma": [1.0, 3.0, 6.0, 9.0, 10.0]})
# вручную: 03-02 → строка 03-01; 03-04, 03-05 → 03-03; 03-09 → 03-09 (суббота 03-06 перекрыта);
# строка 03-10 (будущее) не используется нигде
EXPECTED = [1.0, 1.0, 3.0, 3.0, 3.0, 9.0]


def _shift_plus1(f):
    def g(*a, **k):
        out = f(*a, **k)
        sh = lambda x: np.r_[x[1:], np.nan]
        return {h: sh(v) for h, v in out.items()} if isinstance(out, dict) else sh(out)
    return g


def _lr_backward(f):
    def g(*a, **k):
        td = f(*a, **k)
        lr = np.zeros(td.n)
        lr[1:] = np.log(td.close[1:] / td.close[:-1])
        td.log_ret[:] = lr
        return td
    return g


MUTATIONS = {
    "sigma_plus1": lambda mp: (mp.setattr(s9, "align", _shift_plus1(s9.align)),
                               mp.setattr(fd, "align_sigma", _shift_plus1(fd.align_sigma))),
    "lr_backward": lambda mp: mp.setattr(fd, "load_ticker", _lr_backward(fd.load_ticker)),
}


def test_s9_align_by_hand():
    assert np.allclose(s9.align(BARS, SRC), EXPECTED)


def test_data_align_sigma_by_hand(monkeypatch):
    vp = SRC.rename(columns={"sigma": "sigma_h1"}).assign(ticker="ZZZ")
    monkeypatch.setattr(fd, "vpred_df", lambda: vp)
    out = fd.align_sigma(BARS, "ZZZ")
    assert np.allclose(out["h1"], EXPECTED)
    assert np.isnan(out["h5"]).all()            # горизонта нет в файле — NaN, не чужой ряд


def test_forward_return_convention(monkeypatch):
    c = np.array([100.0, 101.0, 99.0, 103.0])
    d = pd.DataFrame({"ticker": "ZZZ", "date": pd.to_datetime(BARS[:4]), "open": c, "high": c,
                      "low": c, "close": c, "volume": 1.0})
    monkeypatch.setattr(fd, "daily_df", lambda: d)
    monkeypatch.setattr(fd, "vpred_df", lambda: SRC.rename(columns={"sigma": "sigma_h1"})
                        .assign(ticker="ZZZ"))
    fd.load_ticker.cache_clear()
    td = fd.load_ticker("ZZZ")
    fd.load_ticker.cache_clear()
    assert np.allclose(td.log_ret, [np.log(101 / 100), np.log(99 / 101), np.log(103 / 99), 0.0])
    # σ бара d — строка с датой ≤ d: позиция d (вход по close d) не знает σ дня d+1
    assert np.allclose(td.sigma["h1"], EXPECTED[:4])


def test_dev_align_never_uses_future_rows():
    """Данные ≤ 2021: SBER, σ_pred h1 финала; используемая строка σ всегда с датой ≤ даты бара."""
    bars = read_dev(fd.OHLCV, ["ticker", "date"])
    dates = np.sort(bars.loc[bars["ticker"] == "SBER", "date"].values)
    sg = read_dev(SIGMA_H1)
    sg = sg[sg["ticker"] == "SBER"].sort_values("date").rename(columns={"sigma_pred_h1": "sigma"})
    sg = sg.assign(sigma=np.arange(len(sg), dtype=float))        # σ = номер строки
    got = s9.align(dates, sg)
    ok = ~np.isnan(got)
    assert ok.sum() > 1000
    used = sg["date"].values[got[ok].astype(int)]
    assert (used <= dates[ok]).all()
    ref = pd.merge_asof(pd.DataFrame({"date": pd.to_datetime(dates)}), sg, on="date")["sigma"].values
    assert np.allclose(got[ok], ref[ok])
