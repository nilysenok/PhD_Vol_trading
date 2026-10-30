"""К-2, мутации тестировщика PR #801: RVI Д7 со сдвигом +1 день и конец ряда по умолчанию (--konec). Синтетика, без сети."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "scripts/dannye"), str(REPO / "scripts/analysis")]
import d7_block as D7  # noqa: E402
import wf_rrv_d7_bez_dsvd as K2  # noqa: E402


def _matritsa(n=420):
    rng = np.random.default_rng(7)
    d = pd.bdate_range("2023-01-02", periods=n)
    f = pd.DataFrame({"date": d, "ticker": "A"})
    f["close_IMOEX"] = 3000 + rng.normal(size=n).cumsum()
    f["rv_IMOEX"] = 0.2 + 0.01 * rng.random(n)
    f["imoex_low_bars"] = 0
    f["close_RVI"] = 20 + rng.normal(size=n).cumsum() * 0.1
    for c in D7.GLOBAL + ["vix"]:
        f[c] = 100 + rng.normal(size=n).cumsum() * 0.5 + 20
    return f


def test_d7_rvi_dnya_d_ravno_rvi_dnya_d_a_ne_zavtra():
    base = _matritsa()
    out = K2.sobrat_d7(base, 5, REPO / "data/calendar")
    assert np.array_equal(out["d7_rvi"].to_numpy()[1:], base["close_RVI"].to_numpy()[1:])  # k=0 точно
    assert not np.array_equal(out["d7_rvi"].to_numpy()[1:-1], base["close_RVI"].to_numpy()[2:])  # не d+1


def test_mutatsiya_rvi_plus_den_lovitsya(monkeypatch):
    orig = D7._rvi

    def sdvig(v):
        r = orig(v)
        r["d7_rvi"] = v["close_RVI"].shift(-1)  # подложенная утечка: RVI завтрашнего дня
        return r

    monkeypatch.setattr(D7, "_rvi", sdvig)
    base = _matritsa()
    out = K2.sobrat_d7(base, 5, REPO / "data/calendar")
    assert not np.array_equal(out["d7_rvi"].to_numpy()[1:-1], base["close_RVI"].to_numpy()[1:-1])


def test_konec_po_umolchaniyu_20260203(monkeypatch, tmp_path):
    # матрица с датой 2026-02-04 (позже конца) обязана отвергаться до любого расчёта; умолчание 2026-02-04 это бы пропустило
    base = pd.DataFrame({"date": pd.to_datetime(["2026-02-03", "2026-02-04"]), "ticker": "A"})
    monkeypatch.setattr(K2.pd, "read_parquet", lambda *a, **k: base)
    monkeypatch.setattr(K2.K1, "daty_dsvd", lambda: set())
    monkeypatch.setattr(sys, "argv", ["wf_rrv_d7_bez_dsvd.py", "--out", str(tmp_path)])
    with pytest.raises(AssertionError, match="2026-02-04"):
        K2.main()
