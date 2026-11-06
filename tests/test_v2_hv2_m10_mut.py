"""Ж-214, M10: мутации через боевой путь `hv2_m10.build` (замена тавтологичных проверок в test_v2_hv2_m10.py).
(а) RQ с днём d+1: подмена параметра сдвига боевого построителя; проверка «порча баров после d не меняет строку d».
(б) C := BV: подмена `decomposition` боевого построителя; проверка тождества C + J = RV через `identities`."""
import numpy as np
import pandas as pd

from models_v2.s9 import hv2_m10 as M, hv2_synth as Y

DAYS = Y.bdays("2016-04-11", 12)


def _bars():
    return Y.make([(d, dict(last_min=1120, alt_amp=0.001 * (1 + i % 3))) for i, d in enumerate(DAYS)], seed=7)


def _corrupt_after(bars, day):
    bad = bars.copy()
    after = bad["begin"].dt.normalize() > pd.Timestamp(day)
    bad.loc[after, "close"] *= 1.3
    return bad


def _rq_sees_future(bars, **kw) -> bool:
    """Меняется ли RQ дня d при порче баров после d (через реальный build)."""
    d = pd.Timestamp(DAYS[5])
    a, b = M.build(bars, **kw), M.build(_corrupt_after(bars, d), **kw)
    return bool(a.loc[d, "rq"] != b.loc[d, "rq"]) or bool(a.loc[d, "rv_se"] != b.loc[d, "rv_se"])


def test_rq_boevoj_put_ne_vidit_budushchee():
    assert not _rq_sees_future(_bars())


def test_mutaciya_rq_d_plus_1_lovitsya_cherez_boevoj_build():
    bars = _bars()
    assert _rq_sees_future(bars, rq_lag=1)                      # мутант: строка d читает RQ дня d+1 -> тест обязан его заметить
    d = pd.Timestamp(DAYS[5])
    ok, bad = M.build(bars), M.build(bars, rq_lag=1)
    assert bad.loc[d, "rq"] == ok.loc[pd.Timestamp(DAYS[6]), "rq"] != ok.loc[d, "rq"]


def test_tozhdestvo_c_plus_j_na_boevom_postroitele():
    t = M.build(_bars())
    assert (t["bv"] > t["rv"]).any()
    assert M.identities(t)["c_plus_j"] <= 1e-12


def test_mutaciya_c_ravno_bv_lovitsya_tozhdestvom(monkeypatch):
    orig = M.decomposition

    def mutant(d):
        out = orig(d)
        out["c"] = d["bv"]                                      # C := BV (двойной счёт: BV + J != RV при BV > RV)
        return out

    monkeypatch.setattr(M, "decomposition", mutant)
    t = M.build(_bars())
    assert M.identities(t)["c_plus_j"] > 1e-6
    assert not np.allclose(t["c"] + t["j"], t["rv"], rtol=1e-6)
