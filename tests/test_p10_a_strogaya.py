"""П-10 (проверяющий): PR #474 a-strogaya. Независимые проверки, код автора не менялся.
1. Ж-086 п. 6: строка rynok с кодом прохода 1 (d4711aa5, считалась по close d) — справочная. Сейчас флаг глобальный -> xfail strict (блокер).
2. Chitatel не смешивает код: разные kod в каталоге -> отказ; обход только явным sverka_kod=True.
3. Паспорт A_strogaya: mSR train4 пересчитан независимо (scipy) по ryad_A.parquet (2016–2019)."""
import numpy as np
import pandas as pd
import pytest
from scipy import stats

from fastbt import ledger_v2 as L
from fastbt import progon_osnovnaya as P
from fastbt.a_strogaya import OUT

KOD1 = "d4711aa5" + "0" * 32          # код проходов 1 и 2 (счёт по close d)
KOD_NOV = "f" * 40


def _row(kod, vhod):
    pt = {"t_vyhod": "open2", "vhod": vhod}
    return {"klyuch": {"pt": pt}, "kod": kod, "istochnik": "sobol1"}


@pytest.mark.xfail(strict=True, reason="блокер P-10: флаг RYNOK_ISPOLNENIE глобальный, строка прохода 1 (rynok по close d) ложно основная")
def test_rynok_s_kodom_prohoda1_spravochnaya():
    assert P.osnovnaya_stroki(_row(KOD1, "rynok")) is False


def test_rynok_s_novym_kodom_i_limitka_so_starym_kodom_osnovnye():
    assert P.osnovnaya_stroki(_row(KOD_NOV, "rynok")) is True      # посчитана новым кодом
    assert P.osnovnaya_stroki(_row(KOD1, "0.5")) is True           # лимитный вход PR не менял


def _row_led(kod):
    pt = {"i": 0, "t_vyhod": "open2", "vhod": "rynok"}
    kl = L.klyuch("S5", pt, "train4", 0.0004, None)
    yrs = {str(y): 0.5 for y in (2016, 2017, 2018, 2019)}
    kvs = {f"{g}Q{q}": 1.0 for g in (2016, 2017, 2018, 2019) for q in range(1, 5)}
    return {"v": "2.0", "format": "etap1-ledger", "strategiya": "S5", "klyuch": kl, "klyuch_sha": L.klyuch_sha(kl, kod, "V1"),
            "kod": kod, "vhody": "V1", "kod_slit": True, "period": "train4", "komissiya": 0.0004, "fdr": None, "status": "ok",
            "prichina": None, "istochnik": "sobol1", "sloy": "fast", "by_year": yrs, "sdelok_y": {k: 10.0 for k in yrs},
            "msr_by_year": dict(yrs), "by_quarter": kvs, "mediana_kv": 1.5, "dolya_pol_kv": 0.5,
            "by_quarter_zaem": {"5": kvs, "10": kvs}, "flagi": {"vyrozhdena": False, "kray": [], "short_shag1_kray": False}, "t_sek": 0.1}


def test_chitatel_otvergaet_smeshenie_kodov(tmp_path):
    man = {"T": 4, "kod": KOD1, "vhody": "V1", "kod_slit": True}
    pi = L.Pisatel(tmp_path, "S5", 1, man)
    pi.add(_row_led(KOD_NOV), np.zeros(4), np.zeros((17, 4)))     # строка нового кода в каталоге прохода 1
    pi.close()
    with pytest.raises(ValueError, match="разные kod"):
        L.Chitatel(tmp_path, "S5")
    assert len(L.Chitatel(tmp_path, "S5", sverka_kod=True).rows()) == 1     # обход — только явный флаг (так читали разборы 1 и 2)


def _msr_nezavisimo(x):
    """mSR Фавра–Галеано по формуле, scipy-моменты (смещённые): mean·252 / mVaR95."""
    z, m, sd = -1.645, x.mean(), x.std(ddof=1)
    s, k = stats.skew(x), stats.kurtosis(x)                         # смещённые, избыточный эксцесс
    zcf = z + (z**2 - 1) * s / 6 + (z**3 - 3 * z) * k / 24 - (2 * z**3 - 5 * z) * s**2 / 36
    return m * 252 / (-(m + zcf * sd) * np.sqrt(252))


@pytest.mark.parametrize("kol,ozhid", [("excess", -1.8152268080315095), ("excess_zaem5", -1.9704644685608703)])
def test_pasport_msr_A_strogaya(kol, ozhid):
    df = pd.read_parquet(OUT / "ryad_A.parquet")
    assert df["date"].dt.year.between(2016, 2019).all() and df["date"].is_unique and df[kol].notna().all()
    assert _msr_nezavisimo(df[kol].to_numpy()) == pytest.approx(ozhid, abs=1e-9)
    assert round(ozhid, 2) in (-1.82, -1.97)
