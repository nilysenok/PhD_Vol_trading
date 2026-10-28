"""R-P / Часть 1: σ̂ зоны Z2 — склейка dev (ch4_dev, sigma_pred_h*) ≤ 2021 + R2 final_d0_s2emb_h*__t1 (rv_pred → √).
Синтетика во временном каталоге; реальные файлы не читаются.  python3 -m pytest tests/test_zona_z2_sigma.py -q"""
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
    return pd.DataFrame({"date": d, "ticker": "SBER", "sigma_pred_h1": np.float32([0.1, 0.2, 9.0]),
                         "sigma_pred_h5": np.float32(0.3)})


def _dev_kat(kat, dev=None):
    dev = _dev() if dev is None else dev
    for h in (1, 5):
        dev[["date", "ticker", f"sigma_pred_h{h}"]].to_parquet(sigma_z2.dev_path(h, kat))
    return kat


def _z2(kat, dates=("2022-01-04", "2022-01-05"), rows=None, design="d0_s2emb"):
    for h in sigma_z2.HORIZONS:
        r = rows or [(d, "SBER", 0.04 * h) for d in dates]
        d = pd.DataFrame(r, columns=["date", "ticker", "rv_pred"]).assign(step=1)
        if design is not None:
            d["design"] = design
        d.to_parquet(sigma_z2.z2_path(h, kat))


def test_sklejka_dev_i_z2(tmp_path):
    _z2(tmp_path)
    s = sigma_z2.sklei(_dev(), tmp_path)
    assert [str(x.date()) for x in s["date"]] == ["2021-12-29", "2021-12-30", "2022-01-04", "2022-01-05"]
    assert not s.duplicated(["date", "ticker"]).any() and s["date"].max() < zona.HARD
    assert s["sigma_pred_h1"].dtype == np.float32
    assert s["sigma_pred_h1"].iloc[2] == np.float32(np.sqrt(0.04))             # σ = √rv_pred
    assert s["sigma_pred_h1"].iloc[1] == np.float32(0.2) and 9.0 not in s["sigma_pred_h1"].to_numpy()   # dev ≥ 2022 отрезан


def test_preobrazovanie_dev_i_z2_odinakovo():
    """Семейство едино: dev (ch4_dev) построен как sigma_ch3 = √ряда float32; Z2 — тот же _sigma_ch3 (scripts/analysis/h2_d0_ch4.py)."""
    rv = pd.Series([0.0, 0.01, 0.04, 1.7e-5])
    ref = np.sqrt(rv.values).astype("float32")                     # тело sigma_ch3
    assert np.array_equal(sigma_z2._sigma_ch3(rv), ref) and ref.dtype == np.float32


def test_stykovka_nepreryvna(tmp_path):
    """Одинаковые σ̂ на стыке (dev 30.12.2021 = Z2 04.01.2022) → склеенный ряд без скачка и дыр."""
    dev = pd.DataFrame({"date": pd.to_datetime(["2021-12-29", "2021-12-30"]), "ticker": "SBER",
                        "sigma_pred_h1": np.float32(0.2), "sigma_pred_h5": np.float32(0.2)})
    _z2(tmp_path, rows=[("2022-01-04", "SBER", 0.04), ("2022-01-05", "SBER", 0.04)])     # √0.04 = 0.2
    s = sigma_z2.sklei(dev, tmp_path)
    assert s["sigma_pred_h1"].notna().all() and s["sigma_pred_h5"].notna().all()
    assert np.allclose(s["sigma_pred_h1"], 0.2) and np.allclose(s["sigma_pred_h5"], 0.2)


def test_otkaz_drugogo_semejstva_design(tmp_path):
    _z2(tmp_path)
    p = sigma_z2.z2_path(5, tmp_path)
    pd.read_parquet(p).assign(design="final3").to_parquet(p)                       # h5 из другого семейства
    with pytest.raises(ValueError, match="разные design"):
        sigma_z2.sklei(_dev(), tmp_path)


def test_otkaz_neskolko_design_v_fajle(tmp_path):
    _z2(tmp_path, rows=[("2022-01-04", "SBER", 0.04), ("2022-01-05", "SBER", 0.04)])
    p = sigma_z2.z2_path(1, tmp_path)
    d = pd.read_parquet(p)
    d["design"] = ["a", "b"]
    d.to_parquet(p)
    with pytest.raises(ValueError, match="не одно семейство"):
        sigma_z2.sklei(_dev(), tmp_path)


def test_otkaz_ozhidaemyj_design(tmp_path, monkeypatch):
    _z2(tmp_path, design="final3")
    monkeypatch.setenv("FASTBT_Z2_DESIGN", "d0_s2emb")
    with pytest.raises(ValueError, match="ожидаемого"):
        sigma_z2.sklei(_dev(), tmp_path)


def test_otkaz_net_fajla_final3(tmp_path):
    """Файлы старого семейства (final_h*__t1) не читаются: каталог только с ними — отказ."""
    for h in (1, 5):
        pd.DataFrame({"date": pd.to_datetime(["2022-01-04"]), "ticker": "SBER", "rv_pred": 0.04}
                     ).to_parquet(tmp_path / f"final_h{h}__t1.parquet")
    with pytest.raises(FileNotFoundError):
        sigma_z2.sklei(_dev(), tmp_path)


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
    """data.vpred_df в зоне test1 читает FASTBT_DEV_SIGMA + FASTBT_Z2_R2FINAL; align_sigma: Z2 на 2022, прогрев dev на 2021."""
    dk, zk = tmp_path / "d", tmp_path / "z"
    dk.mkdir()
    zk.mkdir()
    _dev_kat(dk)
    _z2(zk)
    monkeypatch.setenv("FASTBT_DEV_SIGMA", str(dk))
    monkeypatch.setenv("FASTBT_Z2_R2FINAL", str(zk))
    data.vpred_df.cache_clear()
    with zona.zone_ctx("test1"):
        a = data.align_sigma(np.array(pd.to_datetime(["2021-12-30", "2022-01-05"]), dtype="datetime64[ns]"), "SBER")
        assert a["h1"][0] == pytest.approx(0.2) and a["h1"][1] == pytest.approx(0.2)    # √0.04
        assert a["h5"][1] == pytest.approx(np.sqrt(0.2))
        assert np.isnan(a["h22"]).all()                                                # в семействе R2 горизонта 22 нет
    data.vpred_df.cache_clear()
