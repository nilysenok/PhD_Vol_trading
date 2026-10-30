"""К-wf2b: охрана `Pul` по периоду. wf с датами ≥ 2020 — только при отметке снятого замка (`period_v2`), синтетика.
Мутация «охрана wf пропускает без отметки» ловится test_wf_bez_otmetki_otkaz. python3 -m pytest tests/test_kalibr_wf2b.py -q
"""
import numpy as np
import pytest

from fastbt import etap1_eval as EV
from fastbt import kalibr_v5 as K
from fastbt import period_v2 as PV

_D = np.arange("2014-08-26", "2026-03-01", dtype="datetime64[D]")
DATY = _D[np.is_busday(_D)]


def _pul(daty=DATY, period="wf"):
    return K.Pul("S1", daty, {}, [{}], np.zeros((1, 1, daty.size)), period=period)


def _otmetka(tmp_path, monkeypatch, est=True):
    otm = tmp_path / "wf_zamok_snyat.json"
    if est:
        otm.write_text('{"preregistraciya_v2": "docs/preregistraciya_v2.md", "sha256": "%s", "data_snyatiya": "2026-10-30"}' % ("a" * 64))
    monkeypatch.setattr(PV, "OTMETKA", otm)       # только временный каталог (отметка репозитория тестами не используется)


def test_wf_bez_otmetki_otkaz(tmp_path, monkeypatch):
    _otmetka(tmp_path, monkeypatch, est=False)
    with pytest.raises(PV.ZamokVladeltsa):
        _pul()


def test_wf_s_otmetkoy_pul_stroitsya(tmp_path, monkeypatch):
    _otmetka(tmp_path, monkeypatch)
    assert _pul().daty.max() == np.datetime64("2026-02-27")


def test_wf_s_otmetkoy_pozzhe_konca_perioda_otkaz(tmp_path, monkeypatch):
    _otmetka(tmp_path, monkeypatch)
    with pytest.raises(ValueError):
        _pul(np.append(DATY, np.datetime64("2026-03-02")))


@pytest.mark.parametrize("period", ["train4", "v2"])
def test_prochie_periody_otkaz_kak_ranshe(period, tmp_path, monkeypatch):
    _otmetka(tmp_path, monkeypatch)               # даже с отметкой — ValueError
    with pytest.raises(ValueError):
        _pul(period=period)


def test_wf_do_2020_bez_otmetki_ok(tmp_path, monkeypatch):
    _otmetka(tmp_path, monkeypatch, est=False)
    assert _pul(DATY[DATY < np.datetime64("2020-01-01")]).daty.size > 0


def test_otmetka_v_repozitorii_tolko_godnaya():
    assert not PV.OTMETKA.exists() or PV.otmetka_godna(PV.OTMETKA)    # ставит владелец; пустой/битый файл — не отметка


def test_periods_wf_v_etap1_eval():
    assert EV.PERIODS["wf"] == (2015, 2026) and EV.years_of("wf") == tuple(range(2015, 2027))
    assert EV.PERIODS["v2"] == (2015, 2019)
