"""wfП-7 (Д-f): проверка сдвига оси usdrub в wf идёт по ряду счёта (USD→CNY, #795), хвост после 11.06.2024 проверяется."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import proverka_sdviga_wf as P  # noqa: E402
from fastbt import vneshnie_1850 as v  # noqa: E402

DNI = pd.bdate_range("2024-03-01", "2024-12-31")
STYK = pd.Timestamp("2024-06-11")


def _bary(dni, uroven):
    ts = [d + pd.Timedelta(hours=10, minutes=10 * k) for d in dni for k in range(12)]
    c = np.repeat(np.asarray(uroven, float), 12)
    return pd.DataFrame({"timestamp": ts, "close": c, "open": c, "high": c, "low": c, "volume": 1})


def _mir(monkeypatch, tmp_path, imoex_r, bez_cny=False):
    """USD (до стыка) и CNY (после) с доходностью = imoex_r + шум; подмена чтения баров, файлы-заглушки на месте."""
    rng = np.random.default_rng(1)
    r = (imoex_r + 0.3 * rng.normal(size=len(DNI)) * imoex_r.std()).to_numpy()
    lvl = 90 * np.exp(np.cumsum(r))
    usd = _bary(DNI[DNI <= STYK], lvl[DNI <= STYK])
    cny = _bary(DNI[DNI > STYK], lvl[DNI > STYK])
    for n in ("usdrub", "cnyrub"):
        p = tmp_path / v.PATHS[n]
        p.parent.mkdir(parents=True, exist_ok=True)
        if not (bez_cny and n == "cnyrub"):
            p.write_bytes(b"")
    monkeypatch.setattr(v, "read_bars", lambda p, end=None: usd if "USD000" in str(p) else cny)


def _imoex():
    return pd.Series(np.random.default_rng(0).normal(0, 0.01, len(DNI)), index=DNI)


def test_hvost_posle_styka_proveryaetsya_i_chestnyj_prohodit(monkeypatch, tmp_path):
    y = _imoex()
    _mir(monkeypatch, tmp_path, y)
    dx = P.dx_usdrub_schet(tmp_path, pd.Timestamp("2024-12-31"))
    assert dx[dx.index > STYK + pd.Timedelta(days=1)].notna().sum() > 100
    assert P.verdikt(P.profil(dx.to_frame("x"), y.to_frame("x")))[0]


def test_mutaciya_usdrub_bez_cny_lovitsya(monkeypatch, tmp_path):
    _mir(monkeypatch, tmp_path, _imoex(), bez_cny=True)
    with pytest.raises(FileNotFoundError, match="CNYRUB_TOM"):                 # было: молча пустой хвост; после #821 (Д4) — отказ
        P.dx_usdrub_schet(tmp_path, pd.Timestamp("2024-12-31"))


def test_mutaciya_sklejka_so_sdvigom_plus_den_blokiruetsya(monkeypatch, tmp_path):
    y = _imoex()
    _mir(monkeypatch, tmp_path, y)
    dx = P.dx_usdrub_schet(tmp_path, pd.Timestamp("2024-12-31"))
    utechka = dx.shift(-1)                                                    # x_d знает доходность d+1
    assert not P.verdikt(P.profil(utechka.to_frame("x"), y.to_frame("x")))[0]
