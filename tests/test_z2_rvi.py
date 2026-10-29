"""RVI зоны Z2 для D6lag1 на period="test1": склейка dev ≤ 2021 + Z2 (FASTBT_Z2_RVI), отказ ≥ 2025, лаг 1, ключ кэша.
Синтетика во временном каталоге. Мутации: RVI_d вместо RVI_{d−1}; чтение без отсечки 2025; подмена Z2 при том же ключе.
python3 -m pytest tests/test_z2_rvi.py -q"""
import inspect

import numpy as np
import pandas as pd
import pytest

from fastbt import s9, zona

DEV = pd.to_datetime(["2021-12-29", "2021-12-30"])
Z2 = pd.to_datetime(["2022-01-04", "2022-01-05", "2024-12-30"])
BARS = np.array(list(DEV) + list(Z2), dtype="datetime64[ns]")


def _ryad(dates, vals, p):
    pd.DataFrame({"date": dates, "close_RVI": vals}).to_parquet(p)


@pytest.fixture
def rvi(tmp_path, monkeypatch):
    dev, z2 = tmp_path / "dev.parquet", tmp_path / "z2.parquet"
    # в dev-файле заведомо чужая строка 2022 (999): на test1 она не должна попасть в ряд
    _ryad(list(DEV) + [Z2[0]], [20.0, 21.0, 999.0], dev)
    _ryad(Z2, [30.0, 31.0, 32.0], z2)
    monkeypatch.setattr(s9, "RVI", dev)
    monkeypatch.setenv("FASTBT_Z2_RVI", str(z2))
    old = s9.PERIOD, s9.YEARS, s9.VAL_START
    s9.set_period("test1")
    yield z2
    s9._restore_period(*old)


def _engine():
    e = object.__new__(s9.Engine)
    e.tickers, e.cache, e.nmax, e.n_t = ["SBER"], {"SBER": {"dates": BARS}}, len(BARS), np.array([len(BARS)])
    return e


def test_sklejka_dev_do_2021_plus_z2(rvi):
    r = _engine().rvi(0)[0]
    assert r == pytest.approx([20.0, 21.0, 30.0, 31.0, 32.0])           # 999 из dev ≥ 2022 отброшено


def test_lag1_beret_rvi_d_minus_1(rvi):
    r = _engine().rvi(1)[0]
    assert np.isnan(r[0]) and r[1:] == pytest.approx([20.0, 21.0, 30.0, 31.0])    # RVI_{d−1}: на баре d — вчерашнее


def test_d6lag1_zadan_lagom_1():
    src = inspect.getsource(s9.S9.d_points)
    assert '(1, "D6lag1")' in src and '(0, "D6")' in src


def test_mutaciya_rvi_d_vmesto_d_minus_1_lovitsya(rvi, monkeypatch):
    real = s9.Engine.rvi
    monkeypatch.setattr(s9.Engine, "rvi", lambda self, lag=0: real(self, 0))      # лаг потерян: RVI_d
    with pytest.raises(AssertionError):
        test_lag1_beret_rvi_d_minus_1(None)


def test_otkaz_2025(rvi, monkeypatch):
    _ryad(list(Z2) + [pd.Timestamp("2025-01-02")], [30.0, 31.0, 32.0, 33.0], rvi)
    with pytest.raises(PermissionError):
        s9.rvi_frame()
    monkeypatch.setattr(zona, "check", lambda *a, **k: None)                      # мутация: проверки зоны нет
    monkeypatch.setattr(s9, "_cut", lambda: pd.Timestamp("2100-01-01"))
    with pytest.raises(AssertionError):                                           # ≥ 2025 дошло бы до счёта
        assert s9.rvi_frame()["date"].max() < pd.Timestamp("2025-01-01")


def test_dev_i_train4_pobaytno_prezhnie(rvi, monkeypatch):
    for per in ("train", "train4"):
        s9.set_period(per)
        monkeypatch.setenv("FASTBT_Z2_RVI", "/net/takogo/faila")                  # Z2 в dev не читается
        pd.testing.assert_frame_equal(s9.rvi_frame(), pd.read_parquet(s9.RVI).assign(date=lambda d: pd.to_datetime(d["date"])))
    s9.set_period("test1")


def _klyuch(monkeypatch, tmp_path):
    f = tmp_path / "x.bin"
    f.write_bytes(b"x")
    monkeypatch.setattr(zona, "path", lambda kind, p=None: f)
    monkeypatch.setattr(s9, "SOURCES", {"hyb_h1": (f, "sigma_pred_h1", "h1")})
    monkeypatch.setattr(s9, "MAIN_YAML", f)
    return s9.cache_key()


def test_klyuch_menyaetsya_pri_podmene_rvi_z2(rvi, monkeypatch, tmp_path):
    k0 = _klyuch(monkeypatch, tmp_path)
    assert _klyuch(monkeypatch, tmp_path) == k0
    _ryad(Z2, [30.0, 31.0, 32.001], rvi)
    assert _klyuch(monkeypatch, tmp_path) != k0


def test_skript_gate_razrezhennye_dni():
    """z2_rvi: расхождение с dev на полном дне — отказ (мутация: значение подменено)."""
    import importlib.util
    from pathlib import Path
    sp = importlib.util.spec_from_file_location("z2_rvi", Path(__file__).resolve().parents[1] / "scripts/dannye/z2_rvi.py")
    z = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(z)
    new = pd.DataFrame({"date": DEV, "close_RVI": [20.0, 21.0]})
    nb = pd.Series([53, 3], index=DEV)
    dev = pd.DataFrame({"date": DEV, "close_RVI": [20.0, 25.0]})              # день 2: разреженный, расхождение допустимо
    r = z.gate_dev(new, nb, dev)
    assert r["full_checked"] == 1 and r["sparse_differ"] == 1
    with pytest.raises(AssertionError):
        z.gate_dev(new, pd.Series([53, 53], index=DEV), dev)                  # тот же день полный — расхождение недопустимо
    with pytest.raises(ValueError):
        z.assert_cut(pd.to_datetime(["2025-01-02"]), "RVI")
