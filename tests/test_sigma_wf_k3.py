"""К-3: D13 f_intraday без ДСВД (scripts/dannye/wf_intraday_bez_dsvd.py). Синтетика без сети и без данных.

Календарь: пт 28.02.2025, вс 02.03.2025 (ДСВД), пн 03.03.2025; рабочая суббота 27.04.2024; рабочая суббота 01.11.2025; 03 и 04.02.2026.
Мутации: «ДСВД-бары оставлены», «граница 2025 → 2024» (выходные с 2024 режутся), «конец 2026-02-04».
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

R = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(R / "scripts/dannye"), str(R / "scripts/analysis")]
import wf_intraday_bez_dsvd as K  # noqa: E402

DSVD = {pd.Timestamp("2025-03-02")}
DNI = ["2024-04-26", "2024-04-27", "2025-02-28", "2025-03-02", "2025-03-03", "2025-11-01", "2026-02-03", "2026-02-04"]


def bary():
    rows = []
    for i, d in enumerate(DNI):
        cl = 100.0 + 10 * i  # цена скачет от дня ко дню: гэп от воскресенья отличается от гэпа от пятницы
        for j in range(54):
            t = pd.Timestamp(d) + pd.Timedelta(hours=9, minutes=50 + 10 * j)
            c = cl + 0.01 * ((j * 7) % 5)
            rows.append((t, c - 0.005, c))
    return pd.DataFrame(rows, columns=["begin", "open", "close"])


def gap(den, pred):
    """ln(open первого бара окна дня / close последнего бара предыдущего дня) — прямо по синтетике."""
    b = bary().assign(d=lambda x: x["begin"].dt.strftime("%Y-%m-%d"))
    return float(np.log(b[b.d == den]["open"].iloc[0] / b[b.d == pred]["close"].iloc[-1]))


def sobrat(**kw):
    return K.sobrat({"SBER": bary()}, DSVD, **kw).set_index("date")


def proverit(r):
    """Инварианты выхода; на мутациях обязаны падать."""
    d = [x.strftime("%Y-%m-%d") for x in r.index]
    assert "2025-03-02" not in d  # G2: ДСВД нет
    assert "2025-11-01" in d and "2024-04-27" in d  # рабочие субботы остаются
    assert max(d) <= "2026-02-03"
    assert r.loc["2025-03-03", "gap_on"] == pytest.approx(gap("2025-03-03", "2025-02-28"), abs=1e-12)  # G3: пн от пятницы


def test_ish_proshodit():
    proverit(sobrat())


def test_ponedelnik_ot_pyatnitsy_a_ne_voskresenya():
    r = sobrat()
    assert abs(r.loc["2025-03-03", "gap_on"] - gap("2025-03-03", "2025-03-02")) > 0.01


def test_mutaciya_dsvd_bary_ostavleny_lovitsya():
    ostavit = lambda c, ds, k: c[pd.to_datetime(c["begin"]).dt.normalize() <= k]
    with pytest.raises(AssertionError):
        proverit(sobrat(filtr=ostavit))


def test_mutaciya_granica_2025_na_2024_lovitsya():
    def rezhet(c, ds, k):
        d = pd.to_datetime(c["begin"]).dt.normalize()
        return c[(d <= k) & ~((d >= "2024-01-01") & (d.dt.dayofweek >= 5))]  # выходные режутся с 2024, суббота 01.11.2025 тоже
    with pytest.raises(AssertionError):
        proverit(sobrat(filtr=rezhet))


def test_mutaciya_konec_0204_lovitsya():
    assert K.KONEC == pd.Timestamp("2026-02-03")
    with pytest.raises(AssertionError):
        proverit(sobrat(konec=pd.Timestamp("2026-02-04")))


def test_sdvig_argmax_nol_i_mutaciya_sdviga_lovitsya():
    rng = np.random.default_rng(1)
    dates = pd.bdate_range("2020-01-01", periods=300)
    rv = pd.DataFrame({"date": dates, "ticker": "X", "rv": np.exp(rng.normal(size=300))})
    new = rv.rename(columns={"rv": "rv_half2"}).assign(rv_half2=lambda d: d.rv_half2 * np.exp(rng.normal(0, 0.1, 300)))
    s = K.sdvig(new, rv)
    assert max(s, key=s.get) == 0
    sdv = new.assign(rv_half2=new.rv_half2.shift(-1))  # утечка: признак из следующего дня
    s = K.sdvig(sdv, rv)
    assert max(s, key=s.get) != 0
