"""Д1 для wf: синтетика (testy.md п. 4). Честный ряд проходит, ряд с утечкой +1 день (σ_d := RV_{d+1}) блокируется."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import proverka_sdviga_wf as P  # noqa: E402
from fastbt import period_v2  # noqa: E402


def panel(n=600, m=5, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2018-01-01", periods=n)
    y = pd.DataFrame(rng.normal(size=(n, m)), index=idx, columns=list("ABCDE"))
    return y, rng


def test_chestnyj_ryad_prohodit():
    y, rng = panel()
    x = y + 0.3 * rng.normal(size=y.shape)
    ok, _ = P.verdikt(P.profil(x, y))
    assert ok and ok == P.verdikt(P.profil(x, y, diff=True))[0]


@pytest.mark.parametrize("sdvig", [1, -1, 2])
def test_utechka_blokiruetsya(sdvig):
    y, rng = panel()
    x = y.shift(-sdvig) + 0.3 * rng.normal(size=y.shape)      # x_d знает y_{d+sdvig}
    prof = P.profil(x, y)
    ok, pr = P.verdikt(prof)
    assert not ok, pr


def test_utechka_po_prirashcheniyam_dlya_okonnogo():
    y, rng = panel()
    okno = y.rolling(10).mean().shift(-1) + 0.1 * rng.normal(size=y.shape)    # оконный ряд с будущим днём
    assert not P.verdikt(P.profil(okno, y.rolling(10).mean(), diff=True))[0]


def test_ploskij_trebuet_vtoroj_proverki():
    prof = {-2: 0.101, -1: 0.103, 0: 0.102, 1: 0.100}           # размах 0,003 < ε = 0,01
    assert P.verdikt(prof, None)[0] is False                     # плоский без второй проверки — не проходит
    assert P.verdikt(prof, False)[0] is False
    assert P.verdikt(prof, True)[0] is True


def test_pustoj_ryad_ne_prohodit():
    assert not P.verdikt({})[0]
    assert not P.verdikt({k: float("nan") for k in P.KS})[0]


def test_stavka_smena_ranshe_zasedaniya_lovitsya():
    z = np.array(["2018-02-09", "2018-03-23"], dtype="datetime64[ns]")
    chest = pd.DataFrame({"data_vstupleniya": pd.to_datetime(["2017-12-18", "2018-02-12", "2018-03-26"])})
    utechka = pd.DataFrame({"data_vstupleniya": pd.to_datetime(["2017-12-18", "2018-02-05", "2018-03-26"])})   # вступила до заседания
    assert P.stavka_ok(chest, z) and not P.stavka_ok(utechka, z)


def test_kod_vyhoda_1_pri_utechke_i_zamok(monkeypatch, tmp_path):
    ok = pd.DataFrame([P.stroka("a", "t", "c", {-2: 0.1, -1: 0.3, 0: 0.9, 1: 0.3})])
    bad = pd.DataFrame([P.stroka("a", "t", "c", {-2: 0.1, -1: 0.3, 0: 0.4, 1: 0.9})])
    monkeypatch.setattr(P, "OUT", tmp_path / "t.csv")
    monkeypatch.setattr(P, "shapka", lambda *a: "# s\n")
    for t, kod in ((ok, 0), (bad, 1)):
        monkeypatch.setattr(P, "sobrat", lambda m, d, t=t: t)
        assert P.main(["--do", "2021-12-31", "--main", str(tmp_path)]) == kod
    monkeypatch.setattr(period_v2, "OTMETKA", tmp_path / "net_otmetki.json")
    with pytest.raises(period_v2.ZamokVladeltsa):
        P.main(["--do", "2026-02-28", "--main", str(tmp_path)])
