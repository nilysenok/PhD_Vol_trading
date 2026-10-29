"""A_strogaya на wf (К-wf3, §11 черновика предрегистрации v2): годы метрик 2017…2026 (10 окон), train4 не изменился.
Синтетика, без данных. Мутация: годы wf = годы train4 (паспорт wf теряет 6 окон) — check обязан упасть."""
import numpy as np
import pytest

from fastbt import a_strogaya as AS


def _r(start, end, seed=0):
    dates = np.arange(start, end, dtype="datetime64[D]")
    rng = np.random.default_rng(seed)
    yr = dates.astype("datetime64[Y]").astype(int) + 1970
    return dict(x=rng.normal(0, 0.01, len(dates)), baza=np.full(len(dates), 0.5), dates=dates,
                expo=rng.random(len(dates)), cnt=np.full(len(dates), 4.0), god=yr)


def check_wf_desyat_okon(gody_fn):
    assert gody_fn("wf") == tuple(range(2017, 2027)) and len(gody_fn("wf")) == 10


def test_wf_desyat_okon():
    check_wf_desyat_okon(AS.gody_perioda)


def test_mut_wf_gody_kak_train4():
    with pytest.raises(AssertionError):
        check_wf_desyat_okon(lambda p: AS.gody_perioda("train4"))


def check_pasport_wf(monkeypatch, gody_fn):
    monkeypatch.setattr(AS, "PER", "wf")
    monkeypatch.setattr(AS, "gody_perioda", gody_fn)
    rows = AS.pasport("A", "strogaya", _r("2017-01-02", "2026-02-28"))
    yrs = [r["god"] for r in rows if r["plata_zaem"] == 0.0]
    assert yrs == [*range(2017, 2027), "все"]
    assert len([r for r in rows if r["plata_zaem"] == AS.PLATA]) == 11


def test_pasport_wf_10_okon_i_vse(monkeypatch):
    check_pasport_wf(monkeypatch, AS.gody_perioda)


def test_mut_pasport_wf_s_godami_train4(monkeypatch):
    with pytest.raises(AssertionError):
        check_pasport_wf(monkeypatch, lambda p: AS.GODY)


def test_train4_umolchanie_prezhnee(monkeypatch):
    assert AS.PER == "train4" and AS.GODY == (2016, 2017, 2018, 2019)
    assert AS.gody_perioda("train4") == AS.GODY
    r = _r("2016-01-04", "2020-01-01", seed=1)
    assert AS.pasport("A", "strogaya", r) == AS.pasport("A", "strogaya", r, gody=AS.GODY)
    assert [x["god"] for x in AS.pasport("A", "strogaya", r)][:5] == [2016, 2017, 2018, 2019, "все"]
