"""R-а, C/D: строки TEST_C и D, период в воркере, отказ при дате ≥ 2025 — на синтетике и моках (без данных проекта и сети)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts" / "analysis"), str(ROOT / "scripts" / "revers")]
import r1_a_cd as R  # noqa: E402
from fastbt import s9, zona  # noqa: E402


@pytest.fixture
def period_restore():
    old = s9.PERIOD, s9.YEARS, s9.VAL_START
    yield
    s9._restore_period(*old)


class FakeEng:
    """Движок s9 в миниатюре: 1 тикер, 400 строк; port даёт excess/exposure; позиции A = 1."""
    def __init__(self, rvi_ok=True):
        n = 400
        self.A_row = np.ones(n)
        self.r = dict(tick=np.zeros(n, int), bar=np.arange(n))
        self.n_t = np.array([n])
        self.calls = []
        self._rvi = np.full((1, n), 0.2 if rvi_ok else np.nan)
        self.idx = pd.bdate_range("2022-01-03", periods=n)

    def rvi(self, lag=0):
        self.calls.append(("rvi", lag))
        return self._rvi

    def g_row(self, G):
        return np.asarray(G, float)

    def port(self, pos, comm=None, cap=1.0):
        self.calls.append(("port", comm))
        x = np.tile([0.002, -0.001], len(pos) // 2) * np.asarray(pos)
        return pd.DataFrame(dict(excess=x, exposure=np.abs(pos)), index=self.idx)

    def calib_closed(self, G):
        return 0.3

    def select(self, stack, kind, screen):
        return stack[0], {0: 1}


class FakeS:
    def __init__(self):
        n = 400
        self.c = {"W504_p70": np.where(np.arange(n) % 3 == 0, 0.0, 1.0), "W504_p80": np.ones(n)}
        self.d = np.where(np.arange(n) % 5 == 0, 0.0, 1.0)

    def c_points(self, src):
        assert src == "hyb_h5"
        return list(self.c.items()), []

    def d_points(self, src):
        assert src == "hyb_h1"
        return {"D6lag1": [("W252_p70", self.d, "mask")], "D1": [("x", np.ones(400), "mask")]}


def _res(eng=None):
    eng = eng or FakeEng()
    out = R.portfeli(eng, FakeS(), 0.0004)
    return {k: (v if isinstance(v, dict) else {"x": v["excess"].to_numpy(), "delo": v["exposure"].to_numpy() > 0,
                                               "dates": v.index.to_numpy().astype("datetime64[D]")}) for k, v in out.items()}


def test_stroki_C_i_D_est_i_konstanty_zamorozheny():
    assert (R.C_POINT, R.C_SRC) == ("W504_p70", "hyb_h5") and s9.TEST_C == ("C1:W504_p70", "hyb_h5")
    assert (R.D_FAM, R.D_SRC) == ("D6lag1", "hyb_h1")
    df = R.tablica(_res())
    for p in ("TEST_C", "D:"):
        sub = df[df.portfel.str.startswith(p)]
        assert {"(а) полные", "(б) без 10 худших дней", "(б) без 10 лучших дней"} <= set(sub.stroka)
        assert sub.stroka.str.startswith("по году 2022").any() and sub.stroka.str.startswith("по году 2023").any()
        assert sub[sub.stroka.str.contains("стоп|размер")].primechanie.str.startswith("неприменимо").all()
    full = df[(df.portfel.str.startswith("TEST_C")) & (df.stroka == "(а) полные")].iloc[0]
    hud = df[(df.portfel.str.startswith("TEST_C")) & (df.stroka == "(б) без 10 худших дней")].iloc[0]
    assert hud.sharpe > full.sharpe and full.dni == 400


def test_C_beret_tochku_W504_p70_a_D_semyu_D6lag1():
    eng = FakeEng()
    res = R.portfeli(eng, FakeS(), 0.0004)
    n = 400
    assert np.isclose(res["C"]["excess"].to_numpy(), np.tile([0.002, -0.001], n // 2) * np.where(np.arange(n) % 3 == 0, 0, 1)).all()
    assert np.isclose(res["D"]["excess"].to_numpy(), np.tile([0.002, -0.001], n // 2) * np.where(np.arange(n) % 5 == 0, 0, 1)).all()
    assert ("rvi", 1) in eng.calls and all(c == 0.0004 for k, c in eng.calls if k == "port")      # комиссия 0,04 %, лаг RVI = 1


def test_D_bez_RVI_neprimenimo_a_ne_tihoe_A():
    res = R.portfeli(FakeEng(rvi_ok=False), FakeS(), 0.0004)
    assert "prichina" in res["D"] and "RVI" in res["D"]["prichina"]
    df = R.tablica(_res(FakeEng(rvi_ok=False)))
    assert df[df.portfel.str.startswith("D:")].primechanie.str.startswith("неприменимо").all()


def test_period_v_vorokere(period_restore):
    s9.set_period("train")
    with pytest.raises(RuntimeError, match="не передан в initializer"):
        R.schitat("test1")                               # мутация «период не передан в воркер» (initializer не вызван)
    R.ustanovit_period("test1")                           # то, что делает Pool(initializer=…)
    assert s9.PERIOD == "test1" and s9.YEARS == (2022, 2023, 2024) and zona.name() == "test1"


def test_main_peredaet_period_v_initializer(monkeypatch, tmp_path, period_restore):
    seen = {}

    class P:
        def __init__(self, n, initializer, initargs):
            seen["init"], seen["args"] = initializer, initargs

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def apply(self, fn, args):
            return {"A": dict(prichina="синтетика"), "C": dict(prichina="синтетика"), "D": dict(prichina="синтетика")}
    monkeypatch.setattr(R, "Pool", P)
    monkeypatch.setattr(R, "shapka", lambda per: f"# синтетика {per}\n")
    R.main(["--period", "test1", "--out", str(tmp_path)])
    assert seen["init"] is R.ustanovit_period and seen["args"] == ("test1",)
    assert (tmp_path / "a_cd_test1.csv").read_text(encoding="utf-8").startswith("# синтетика test1")


def test_otkaz_pri_date_2025():
    res = _res()
    res["C"]["dates"] = res["C"]["dates"] + np.timedelta64(1100, "D")     # ряд уехал в 2025
    with pytest.raises(PermissionError):
        R.tablica(res)


def test_train4_otkaz():
    with pytest.raises(SystemExit, match="нет σ-источников"):
        R.main(["--period", "train4"])
