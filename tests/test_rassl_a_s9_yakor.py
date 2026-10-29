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


def test_komponenty_razdelyaet_stavku(capsys):
    idx = pd.bdate_range("2022-01-03", periods=3)
    mk = lambda ex: pd.DataFrame(dict(r=[0.0] * 3, gross=[0.0] * 3, exposure=[1.0] * 3, excess=ex), index=idx)  # noqa: E731
    a = {"port": mk([0.0, 0.0, 0.0]), "cnt": pd.Series([2.0] * 3, index=idx), "tickers": []}
    b = {s: {"port": mk([0.0, 0.001, 0.0]), "cnt_pairs": pd.Series([2.0] * 3, index=idx), "div": pd.DataFrame()} for s in AS.SIDS}
    R.komponenty(a, b)
    out = capsys.readouterr().out
    assert "ставка/плечо excess−r: дней 1, первый 2022-01-04" in out
    assert "gross: дней 0" in out
