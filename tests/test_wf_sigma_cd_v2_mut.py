"""Тестировщик #820: непойманная мутация «ДСВД = любые сб/вс ≥ 2025 без учёта календаря» (рабочая суббота 2025-11-01 торговая).
Остальные мутации (float64, шаг 9 из v1, конец 2026-02-04) ловятся tests/test_wf_sigma_cd_v2.py (проверено в копии).
Сеть не нужна; календарь — docs/issledovaniya/2026-10-30_kalendar_moex_2025_2026.csv (официальный, #754).
python3 -m pytest tests/test_wf_sigma_cd_v2_mut.py -q"""
import importlib.util
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("t820", ROOT / "tests" / "test_wf_sigma_cd_v2.py")
T = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(T)
V2, W = T.V2, T.W
SAT = pd.Timestamp("2025-11-01")
CAL = ROOT / "docs" / "issledovaniya" / "2026-10-30_kalendar_moex_2025_2026.csv"


def _bez_kalendarya(v1, kal):
    """Мутация: ДСВД = все сб/вс ≥ 2025 в v1."""
    d = pd.to_datetime(v1["date"])
    return set(d[(d >= V2.WF0) & (d.dt.dayofweek >= 5)])


@pytest.fixture
def s_subbotoy(monkeypatch):
    monkeypatch.setattr(T, "DAYS", T.DAYS.append(pd.DatetimeIndex([SAT])).sort_values())


def test_rabochaya_subbota_est_v_v2(tmp_path, s_subbotoy):
    out = tmp_path / "out"
    V2.run(T.make_root(tmp_path), out)
    for h in W.HORIZONS:
        s = pd.read_parquet(out / f"sigma_hyb_h{h}_dev.parquet")
        assert (s["date"] == SAT).sum() == len(T.TICKERS)
        assert not s["date"].isin(T.DSVD).any()


def test_mutaciya_subbota_kak_dsvd_lovitsya(tmp_path, s_subbotoy, monkeypatch):
    root = T.make_root(tmp_path)
    monkeypatch.setattr(V2, "dni_dsvd", _bez_kalendarya)
    with pytest.raises(SystemExit, match="ДСВД"):
        V2.run(root, None)


def test_dsvd_ravno_ofitsialnomu_kalendaryu():
    c = pd.read_csv(CAL, parse_dates=["date"])
    c = c[c["date"] <= "2026-02-03"]
    torg = c[c["is_traded"] == 1]                                    # v1 = торговые даты, в т.ч. выходные сессии
    v1 = pd.DataFrame({"date": torg["date"]})
    kal = pd.DataFrame({"date": c.loc[c["osnovnaya_sessiya"] == 1, "date"]})
    dsvd = V2.dni_dsvd(v1, kal)
    ofic = set(c.loc[(c["is_traded"] == 1) & (c["osnovnaya_sessiya"] == 0), "date"])
    assert dsvd == ofic and len(dsvd) == 70 and SAT not in dsvd
    assert not dsvd & set(c.loc[c["reason"] == "H", "date"])         # праздничные будни — не ДСВД
    assert len(_bez_kalendarya(v1, kal)) == 71                       # мутация добавляет рабочую субботу
