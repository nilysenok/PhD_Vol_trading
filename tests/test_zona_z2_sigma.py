"""R-P / Часть 1: σ̂ зоны Z2 — склейка dev ≤ 2021 + final_h*__t1. Синтетика во временном каталоге; реальные файлы не читаются.
python3 -m pytest tests/test_zona_z2_sigma.py -q"""
import numpy as np
import pandas as pd
import pytest

from fastbt import data, sigma_z2, zona


@pytest.fixture(autouse=True)
def _restore():
    yield
    zona.set_zone("dev")


def _dev():
    d = pd.to_datetime(["2021-12-29", "2021-12-30", "2022-01-04"])          # последняя строка dev — за границей, обрезается
    return pd.DataFrame({"date": d, "ticker": "SBER", "sigma_h1": [0.1, 0.2, 9.0], "sigma_h5": 0.3, "sigma_h22": 0.4}).astype(
        {"sigma_h1": "float32"})


def _z2(kat, dates=("2022-01-04", "2022-01-05"), rows=None):
    for h in sigma_z2.HORIZONS:
        r = rows or [(d, "SBER", 0.04 * h) for d in dates]
        pd.DataFrame(r, columns=["date", "ticker", "rv_pred"]).assign(step=1).to_parquet(kat / f"final_h{h}__t1.parquet")


def test_sklejka_dev_i_z2(tmp_path):
    _z2(tmp_path)
    s = sigma_z2.sklei(_dev(), tmp_path)
    assert [str(x.date()) for x in s["date"]] == ["2021-12-29", "2021-12-30", "2022-01-04", "2022-01-05"]
    assert not s.duplicated(["date", "ticker"]).any() and s["date"].max() < zona.HARD
    assert s["sigma_pred_h1"].dtype == np.float32
    assert s["sigma_pred_h1"].iloc[2] == np.float32(np.sqrt(0.04))             # σ = √rv_pred
    assert s["sigma_pred_h1"].iloc[1] == np.float32(0.2) and 9.0 not in s["sigma_pred_h1"].to_numpy()   # dev ≥ 2022 отрезан


def test_otkaz_data_2025(tmp_path):
    _z2(tmp_path, dates=("2024-12-31", "2025-01-02"))
    with pytest.raises(PermissionError):
        sigma_z2.sklei(_dev(), tmp_path)


def test_otkaz_dubl_data_tiker(tmp_path):
    _z2(tmp_path, rows=[("2022-01-04", "SBER", 0.04), ("2022-01-04", "SBER", 0.05)])
    with pytest.raises(ValueError, match="дубли"):
        sigma_z2.sklei(_dev(), tmp_path)


def test_otkaz_peresecheniya_s_dev(tmp_path):
    _z2(tmp_path, dates=("2021-12-30", "2022-01-04"))
    with pytest.raises(ValueError, match="пересечение"):
        sigma_z2.sklei(_dev(), tmp_path)


def test_otkaz_otricatelnyj_rv(tmp_path):
    _z2(tmp_path, rows=[("2022-01-04", "SBER", -0.01)])
    with pytest.raises(ValueError, match="отрицательный"):
        sigma_z2.sklei(_dev(), tmp_path)


def test_vpred_df_z2_cherez_peremennuyu(tmp_path, monkeypatch):
    """data.vpred_df в зоне test1 читает FASTBT_Z2_FINAL3; align_sigma отдаёт σ̂ Z2 на 2022 и прогрев dev на 2021."""
    kat = tmp_path / "z"
    kat.mkdir()
    _z2(kat)
    dev = tmp_path / "dev.parquet"
    _dev().to_parquet(dev)
    monkeypatch.setattr(data, "VPRED", dev)
    monkeypatch.setenv("FASTBT_Z2_FINAL3", str(kat))
    data.vpred_df.cache_clear()
    with zona.zone_ctx("test1"):
        a = data.align_sigma(np.array(pd.to_datetime(["2021-12-30", "2022-01-05"]), dtype="datetime64[ns]"), "SBER")
        assert a["h1"][0] == pytest.approx(0.2) and a["h1"][1] == pytest.approx(0.2)    # √0.04
        assert a["h5"][1] == pytest.approx(np.sqrt(0.2))
    data.vpred_df.cache_clear()
