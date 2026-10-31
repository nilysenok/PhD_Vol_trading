"""Источники v7 (V7-8b): sobrat на синтетике (2018, без сети и файлов). sha ловит подмену 10-минуток / панели / фона; корпсобытия до
агрегации; множитель дивиденда в 10-минутках; окно чтения строго < конца; мутация «sha без 10-минуток» падает на тесте подмены."""
import numpy as np
import pandas as pd
import pytest

from fastbt import eval_v7_ist as E
from fastbt.signals.v7_kross import Panel

TK = ["AAA", "BBB", "CCC"]
KARTA = pd.DataFrame({"ticker": TK, "indeks": ["MOEXFN", "MOEXFN", "MOEXOG"]})
DNI = pd.bdate_range("2018-01-09", periods=40)


def _bary(k: float, dni=DNI) -> pd.DataFrame:
    ts = [d + pd.Timedelta(minutes=m) for d in dni for m in range(600, 1130, 10)]
    n = len(ts)
    c = k * (1 + 0.01 * np.sin(np.arange(n) / 7.0 + k))
    return pd.DataFrame({"timestamp": ts, "open": c, "high": c * 1.001, "low": c * 0.999, "close": c, "volume": 100})


def _raw(dni=DNI):
    return {t: _bary(10.0 * (i + 1), dni) for i, t in enumerate(TK)}


def _im(dni=DNI):
    return pd.Series(1000 + np.arange(len(dni), dtype=float), index=dni)


def _korp0(b, t):
    return b


def _mn1(t, dts, close):
    return np.ones(len(close))


def _sobrat(raw=None, im=None, korp=_korp0, mn=_mn1, konec="2019-01-01"):
    return E.sobrat(raw or _raw(), _im() if im is None else im, KARTA, TK, "2018-01-01", konec, korp=korp, mnozh=mn)


def test_sostav_istochnikov():
    s = _sobrat()
    assert set(s.b10) == set(TK) and s.panel.c.shape == (40, 3) and list(s.panel.c.columns) == TK
    assert set(s.fon) == {"sektor", "imoex"} and len(s.sha) == 64
    assert s.ist7().b10 is s.b10 and s.ist7().panel is s.panel
    assert set(s.b10["AAA"]["slot"]) == set(range(53))          # prep_10m: слоты [10:00; 18:50)


def test_sha_stabilen_i_menyaetsya_ot_10minutok():
    assert _sobrat().sha == _sobrat().sha
    raw = _raw()
    raw["BBB"].loc[100, ["open", "high", "low", "close"]] *= 1.5            # слот внутри дня, close дня не меняется
    assert _sobrat(raw).sha != _sobrat().sha


def test_sha_menyaetsya_ot_paneli_imoex_i_fona():
    im = _im()
    im.iloc[10] += 1.0
    assert _sobrat(im=im).sha != _sobrat().sha


def test_sha_metki_sektora_v_hashe():
    s = _sobrat()
    p2 = Panel(**{**s.panel.__dict__, "metki": np.where(s.panel.metki == "MOEXOG", "MOEXTN", s.panel.metki)})
    assert E.sha_istochnikov(s.b10, p2, s.fon) != s.sha


def _sha_bez_10minutok(b10, panel, fon):
    return E.sha_istochnikov({}, panel, fon)


def test_mutaciya_sha_bez_10minutok_lovitsya():
    """Обязательный свойство: подмена 10-минуток меняет sha. Мутант (b10 не в хэше) этого не выполняет."""
    a = _sobrat()
    raw = _raw()
    raw["CCC"].loc[5, "volume"] = 999
    b = _sobrat(raw)
    assert E.sha_istochnikov(a.b10, a.panel, a.fon) != E.sha_istochnikov(b.b10, b.panel, b.fon)
    assert _sha_bez_10minutok(a.b10, a.panel, a.fon) == _sha_bez_10minutok(b.b10, b.panel, b.fon)      # мутант не отличает


def test_korpsobytie_do_agregacii():
    """Сплит на дату 2018-02-01: все бары ДО даты × 2 — и в 10-минутках, и в дневном close панели."""
    def korp(b, t):
        m = (b["timestamp"] < pd.Timestamp("2018-02-01")).to_numpy()
        b = b.copy()
        b.loc[m, ["open", "high", "low", "close"]] *= 2.0
        return b
    s0, s1 = _sobrat(), _sobrat(korp=korp)
    d = DNI[0]
    assert s1.panel.c.loc[d, "AAA"] == pytest.approx(2 * s0.panel.c.loc[d, "AAA"])
    x0 = s0.b10["AAA"].query("date == @d and slot == 0")["open"].iloc[0]
    assert s1.b10["AAA"].query("date == @d and slot == 0")["open"].iloc[0] == pytest.approx(2 * x0)


def test_mnozhitel_dividenda_v_10minutkah_i_dnevke():
    """Множитель дня 0.9 на первую половину: 10-минутки и close панели — одним множителем (сигнальный ряд согласован)."""
    def mn(t, dts, close):
        return np.where(pd.DatetimeIndex(dts) < pd.Timestamp("2018-02-01"), 0.9, 1.0)
    s0, s1 = _sobrat(), _sobrat(mn=mn)
    d = DNI[3]
    assert s1.panel.c.loc[d, "BBB"] == pytest.approx(0.9 * s0.panel.c.loc[d, "BBB"])
    a0 = s0.b10["BBB"].query("date == @d and slot == 10")["close"].iloc[0]
    assert s1.b10["BBB"].query("date == @d and slot == 10")["close"].iloc[0] == pytest.approx(0.9 * a0)
    assert s1.sha != s0.sha


def test_okno_chteniya_strogo_do_konca():
    """Бары на дату конца и позже не читаются (граница < конец): с хвостом 2019 результат тот же, что без него."""
    dni2 = pd.bdate_range("2018-01-09", periods=300)
    s_hvost, s_net = _sobrat(raw=_raw(dni2), im=_im(dni2), konec="2018-06-01"), None
    n = len(DNI)
    s_net = E.sobrat({t: b[b["timestamp"] < "2018-06-01"] for t, b in _raw(dni2).items()}, _im(dni2), KARTA, TK, "2018-01-01",
                     "2018-06-01", korp=_korp0, mnozh=_mn1)
    assert s_hvost.sha == s_net.sha and s_hvost.panel.daty.max() < np.datetime64("2018-06-01") and n < 300


def test_fon_otnositelno_bez_nan_i_flag():
    s = _sobrat()
    fon, flag = s.fon["sektor"]
    assert fon.shape == s.panel.c.shape and np.isfinite(fon.to_numpy()[1:]).all() and flag.shape == fon.shape
    assert np.allclose(s.fon["imoex"][0].iloc[5].to_numpy(), s.fon["imoex"][0].iloc[5, 0])     # IMOEX одинаков для всех тикеров


def test_zagruzit_period_ne_polnaya_otkaz():
    with pytest.raises(NotImplementedError):
        E.zagruzit("train4", TK)


def test_zagruzit_polnaya_bez_otmetki_zamka_otkaz(monkeypatch, tmp_path):
    """wf/polnaya без отметки снятого замка — отказ до чтения (period_v2.proverit_chtenie)."""
    from fastbt import period_v2
    monkeypatch.setattr(period_v2, "OTMETKA", tmp_path / "net.json")
    with pytest.raises(Exception):
        E.zagruzit(E.POLNAYA, TK)
