"""R-а, C/D: покрытие RVI для D — колонка, шапка, примечание при покрытии < 1 (синтетика, без данных проекта)."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts" / "analysis"), str(ROOT / "scripts" / "revers")]
import r1_a_cd as R  # noqa: E402
from test_r1_a_cd import FakeEng, FakeS, _res  # noqa: E402


def _res_pk(pk, k=40):
    res = _res()
    res["D"]["pokrytie_rvi"], res["D"]["bez_rvi_dnej"] = pk, k
    return res


def test_pokrytie_kolonka_i_primechanie_pri_09():
    df = R.tablica(_res_pk(0.9))
    d = df[df.portfel.str.startswith("D:")]
    assert (d.pokrytie_rvi == 0.9).all()
    assert d.primechanie.str.contains("на 40 днях без ранга RVI затвор открыт (D = A)", regex=False).all()
    assert df[~df.portfel.str.startswith("D:")].pokrytie_rvi.isna().all()


def test_polnoe_pokrytie_bez_primechaniya():
    df = R.tablica(_res_pk(1.0, 0))
    d = df[df.portfel.str.startswith("D:")]
    assert (d.pokrytie_rvi == 1.0).all()
    assert not d.primechanie.fillna("").str.contains("затвор открыт").any()


def test_mutaciya_primechanie_ne_pishetsya_lovitsya(monkeypatch):
    monkeypatch.setattr(R, "primechanie_rvi", lambda pk, k: "")
    d = R.tablica(_res_pk(0.9))
    assert not d.primechanie.fillna("").str.contains("затвор открыт").any()   # при мутации примечания нет,
    # а тест выше (pri_09) на ней падает: проверяем, что именно подмена его роняет
    assert R.primechanie_rvi(0.9, 40) == ""


def test_schitat_pokrytie_chitaetsya_iz_dvizhka():
    eng = FakeEng()
    eng._rvi[0, :40] = np.nan
    assert abs(R.pokrytie_rvi(eng) - 0.9) < 1e-12
    assert R.primechanie_rvi(0.9, 40).startswith("на 40 днях")


def test_shapka_pokrytie_pechataetsya(monkeypatch, tmp_path):
    def fake(per):
        r = {k: dict(prichina="синтетика") for k in "ACD"}
        r["D"]["pokrytie_rvi"] = 0.9
        return r
    monkeypatch.setattr(R, "schitat", fake)
    monkeypatch.setattr(R, "shapka", lambda per: "# s\n")
    monkeypatch.setattr(R.s9, "set_period", lambda per: None)
    R.main(["--period", "test1", "--out", str(tmp_path)])
    assert "# покрытие RVI для D: 0.9000\n" in (tmp_path / "a_cd_test1.csv").read_text(encoding="utf-8")
