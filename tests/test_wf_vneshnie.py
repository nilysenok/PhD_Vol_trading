"""wf-L6: граница wf во внешних рядах, ось usdrub (склейка по доходностям USD→CNY), Brent с пустым контрактом."""
import numpy as np
import pandas as pd
import pytest

from fastbt import period_v2, vneshnie_1850 as v, vneshnie_usdrub as vu

D = pd.Timestamp


def _ryady():
    ix = pd.bdate_range("2024-06-05", "2024-06-18")
    usd = pd.Series(90.0 * np.exp(np.cumsum(np.full(len(ix), 0.01))), index=ix)
    cny = pd.Series(12.0 * np.exp(np.cumsum(np.full(len(ix), 0.002))), index=ix)   # другой уровень и другой темп
    return usd[usd.index <= "2024-06-11"], cny


def test_sklejka_dohodnosti_12_iz_cny_11_iz_usd():
    usd, cny = _ryady()
    dx = vu.dx_usdrub(usd, cny)
    assert dx[D("2024-06-11")] == pytest.approx(0.01)     # из USD
    assert dx[D("2024-06-12")] == pytest.approx(0.002)    # из CNY: ln(CNY_12/CNY_11), не через стык уровней
    assert dx[D("2024-06-17")] == pytest.approx(0.002)
    assert dx.index.is_monotonic_increasing and not dx.index.duplicated().any()


def test_cherez_styk_urovnei_net_dohodnosti():
    usd, cny = _ryady()
    dx = vu.dx_usdrub(usd, cny)
    skachok = abs(np.log(cny[D("2024-06-12")] / usd[D("2024-06-11")]))
    assert skachok > 1.5 and dx.abs().max() < 0.011       # прыжок уровней ln(12/90) в ряд не попал


def test_mutatsiya_sklejka_po_urovnyam_lovitsya(monkeypatch):
    usd, cny = _ryady()
    monkeypatch.setattr(vu, "SPLICE_BY_LEVEL", True)
    dx = vu.dx_usdrub(usd, cny)
    assert dx[D("2024-06-12")] == pytest.approx(np.log(cny[D("2024-06-12")] / usd[D("2024-06-11")]))
    assert abs(dx[D("2024-06-12")] - 0.002) > 1           # тест выше (0.002) на этой мутации падает


def test_do_obryva_pobaytno_prezhnee_train4():
    usd, _ = _ryady()
    dx = vu.dx_usdrub(usd, usd.iloc[0:0])
    pd.testing.assert_series_equal(dx, v.change("usdrub", usd), check_names=False)
    assert v.granitsa() == v.granitsa("train4") == v.granitsa("v2") == D("2022-01-01") == v.END
    assert v.PATHS["usdrub"].endswith("currency_10m/USD000UTSTOM.parquet")


def test_uroven_tolko_usd():
    usd, _ = _ryady()
    long = pd.concat([usd, pd.Series([12.0], index=[D("2024-06-12")])])
    assert vu.uroven_usd(long).index.max() == vu.USD_LAST


def _bary(sdvig_dnei, vol):
    ts = [pd.Timestamp(d) + pd.Timedelta(hours=10, minutes=10 * k) for d in pd.bdate_range("2024-06-05", "2024-06-18") for k in range(12)]
    c = 90.0 * np.exp(np.cumsum(np.tile([vol, -vol], len(ts) // 2)))
    return pd.DataFrame({"timestamp": ts, "close": c, "open": c, "high": c, "low": c, "volume": 1})


def test_faktor_rv_iz_svoego_ryada():
    usd = _bary(0, 0.001)
    usd = usd[usd["timestamp"] <= "2024-06-11 23:59"]
    cny = _bary(0, 0.01)
    dx, sg = vu.faktor_usdrub(usd, cny)
    rv = vu.rv_usdrub(usd, cny)
    assert rv[D("2024-06-11")] < 1e-4 < rv[D("2024-06-12")]          # до — rv USD, после — rv CNY
    assert D("2024-06-12") in dx.index and len(sg) == len(rv)


def test_net_cny_posle_obryva_znacheniy_net():
    usd = _bary(0, 0.001)
    usd = usd[usd["timestamp"] <= "2024-06-11 23:59"]
    dx, _ = vu.faktor_usdrub(usd, None)
    assert dx.index.max() <= vu.USD_LAST                              # без ffill: после обрыва строк нет


# --- граница wf и отметка снятого замка ---------------------------------------------------
def test_granitsa_wf_s_otmetkoi(tmp_path):
    otm = tmp_path / "wf_zamok_snyat.json"
    otm.write_text("{}")
    assert v.granitsa("wf", otmetka=otm) == D("2026-03-01")


def test_granitsa_wf_bez_otmetki_otkaz(tmp_path):
    with pytest.raises(period_v2.ZamokVladeltsa):
        v.granitsa("wf", otmetka=tmp_path / "net.json")


def test_mutatsiya_granitsa_bez_otmetki_lovitsya(tmp_path, monkeypatch):
    monkeypatch.setattr(period_v2, "proverit_chtenie", lambda *a, **k: None)
    assert v.granitsa("wf", otmetka=tmp_path / "net.json") == D("2026-03-01")   # отказа нет -> тест выше падает


def test_read_bars_wf_chitaet_2026_do_28_02(tmp_path):
    ts = pd.to_datetime(["2025-12-30 18:40", "2026-02-27 18:40", "2026-03-02 10:00"])
    p = tmp_path / "x.parquet"
    pd.DataFrame({"begin": ts.strftime("%Y-%m-%d %H:%M:%S"), "close": [1.0, 2.0, 3.0]}).to_parquet(p)
    otm = tmp_path / "o.json"
    otm.write_text("{}")
    b = v.read_bars(p, v.granitsa("wf", otmetka=otm))
    assert list(b["close"]) == [1.0, 2.0]
    assert len(v.read_bars(p)) == 0 or v.read_bars(p)["timestamp"].max() < D("2022-01-01")


# --- Brent: пустой контракт (BRF8_2018.parquet) -----------------------------------------------
def test_brent_pustoy_kontrakt_net_strok_i_net_ffill():
    t = pd.DataFrame({"secid": ["A", "B", "C"], "lsttrade": pd.to_datetime(["2020-03-20", "2020-04-20", "2020-05-20"])})
    t["roll"] = [v.roll_date(x) for x in t["lsttrade"]]
    ix = pd.bdate_range("2020-02-03", "2020-05-20")
    mk = lambda lo, hi: pd.Series(50 * np.exp(0.001 * np.arange(len(ix))), index=ix)[lo:hi]    # noqa: E731
    out = v.splice_brent({"A": mk("2020-02-03", "2020-03-12"), "C": mk("2020-04-08", "2020-05-12")}, t)  # B пуст
    d = pd.DatetimeIndex(out["date"])
    gap = (d > D("2020-03-12")) & (d < D("2020-04-08"))
    assert not gap.any()                                            # ни строки, ни подстановки в дыре
    first_c = out[out["secid"] == "C"].iloc[0]
    assert first_c["ret"] == pytest.approx(0.001)                   # доходность внутри C, не через дыру и не через смену контракта
    assert out.loc[out["secid"] == "A", "ret"].iloc[-1] == pytest.approx(0.001)
