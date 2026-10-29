"""Синтетика для scripts/analysis/rassl_a_s9_yakor.py: сравнение ловит подложенное расхождение и находит первый день."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "analysis"))
import rassl_a_s9_yakor as R  # noqa: E402
from fastbt import a_strogaya as AS  # noqa: E402


def _mat(delta: float = 0.0):
    idx = pd.bdate_range("2022-01-03", periods=5)
    m = pd.DataFrame(np.full((5, 2), 0.001), index=idx, columns=["A", "B"])
    m.iloc[3, 1] += delta
    return m


def test_pervyj_nahodit_pervyj_den():
    s = pd.Series([0, 0, 0.5, 0, 0.1], index=pd.bdate_range("2022-01-03", periods=5))
    assert "первый 2022-01-05" in R.pervyj(s) and "дней 2" in R.pervyj(s)


def test_sravnit_lovit_podlozhennoe(capsys):
    a = {"pairs": {s: {"net": _mat()} for s in AS.SIDS}}
    b = {s: {"net": _mat(0.01 if s == "S3" else 0.0)} for s in AS.SIDS}
    R.sravnit(a, b)
    out = capsys.readouterr().out
    assert out.count("дней с расхождением: 1 из 5; первый: 2022-01-06") == 1
    assert out.count("дней с расхождением: 0 из 5") == 5


def test_vhody_pos_s9_i_ceny(capsys, monkeypatch):
    import types
    import rassl_a_vhody as V
    from fastbt import s9
    dates = pd.bdate_range("2022-01-03", periods=6).values
    monkeypatch.setattr(s9, "YEARS", (2022,))
    eng = types.SimpleNamespace(cache={"X": {"dates": dates, "A": {"S3": {2022: np.array([0, 1, 1, 0, -1, 0.0])}}}},
                                _slice=lambda t, Y: (0, 6))
    assert V.pos_s9(eng, "X", "S3").tolist() == [0, 1, 1, 0, -1, 0]
    bar = lambda c: types.SimpleNamespace(pusto=False, c=np.array([c]))        # noqa: E731
    inp = types.SimpleNamespace(meta={"X": {"dates": dates}}, close={"X": np.arange(6.0) + 10},
                                bars={"X": [bar(10), bar(11), bar(12.5), bar(13), bar(14), bar(15)]})
    monkeypatch.setattr(V.data, "daily_df", lambda: pd.DataFrame({"ticker": "X", "date": pd.DatetimeIndex(dates),
                                                                  "close": np.arange(6.0) + 10}))
    V.ceny(inp, "X")
    out = capsys.readouterr().out
    assert "inp.close − daily: max|Δ|=0" in out and "посл. 5-мин. close − daily: max|Δ|=0.5, дней с Δ 1" in out


def test_komponenty_razdelyaet_stavku(capsys):
    idx = pd.bdate_range("2022-01-03", periods=3)
    mk = lambda ex: pd.DataFrame(dict(r=[0.0] * 3, gross=[0.0] * 3, exposure=[1.0] * 3, excess=ex), index=idx)  # noqa: E731
    a = {"port": mk([0.0, 0.0, 0.0]), "cnt": pd.Series([2.0] * 3, index=idx), "tickers": []}
    b = {s: {"port": mk([0.0, 0.001, 0.0]), "cnt_pairs": pd.Series([2.0] * 3, index=idx), "div": pd.DataFrame()} for s in AS.SIDS}
    R.komponenty(a, b)
    out = capsys.readouterr().out
    assert "ставка/плечо excess−r: дней 1, первый 2022-01-04" in out
    assert "gross: дней 0" in out
