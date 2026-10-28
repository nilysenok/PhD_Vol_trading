"""A_strogaya, ряды (продолжение tests/test_a_strogaya.py): портфель A из рядов пар, последний день окна, проверка на train4 против якоря.
Мутации: сделка последнего дня окна в портфеле; A не равен среднему по парам (отказ при разном числе пар).
Данные (S3, train4) — пропуск без 1-минуток (FASTBT_BARS1M)."""
import numpy as np
import pytest

from fastbt import a_strogaya as AS
from fastbt.etap1_eval_v2 import EvaluatorV2


def _res(rng, T=30, n=17):
    out = {}
    for k, s in enumerate(AS.SIDS):
        tik = rng.normal(0, 0.01, (n, T))
        tik[k % n, :3] = np.nan                                     # у пары нет строки на дату — исключается из среднего
        out[s] = dict(tik=tik, baza=np.full(T, 0.001 * (k + 1)), dates=np.arange("2016-01-04", T, dtype="datetime64[D]"),
                      expo=np.zeros(2), cnt=np.zeros(2), god=np.zeros(2))
    return out


def test_portfel_A_ravnyj_ves_po_param():
    res = _res(np.random.default_rng(3))
    A = AS.portfel_A(res)
    allp = np.concatenate([r["tik"] for r in res.values()])
    assert A["x"] == pytest.approx(np.nanmean(allp, axis=0), abs=1e-14)
    assert A["baza"] == pytest.approx(np.mean([r["baza"] for r in res.values()], axis=0))


def test_portfel_A_otkaz_pri_raznom_chisle_par():
    res = _res(np.random.default_rng(4))
    res["S2"]["tik"][5, 10] = np.nan
    with pytest.raises(ValueError, match="число пар"):
        AS.portfel_A(res)


def test_posledniy_den_okna_ne_vhoditsya(monkeypatch):
    pos = np.array([0, 1, 1, 0, 0, 0, 1, 0.0])                      # индексы 1–2 и 6; len − 2 = 6: сделка, стартовавшая в последний день окна
    monkeypatch.setattr(EvaluatorV2, "_sig", lambda *a: pos.copy())
    ev = AS.AStrogaya.__new__(AS.AStrogaya)
    assert list(ev._sig(None, None, 0, True, 0, 7, "obe")) == [0, 1, 1, 0, 0, 0, 0, 0]


def test_mut_posledniy_den_vhoditsya(monkeypatch):
    monkeypatch.setattr(AS, "gate_trades", lambda pos, allow: pos)
    with pytest.raises(AssertionError):
        test_posledniy_den_okna_ne_vhoditsya(monkeypatch)


@pytest.fixture(scope="module")
def s3():
    from fastbt import bars5m
    if not bars5m.SRC.exists():
        pytest.skip("нет 1-минуток D25 (FASTBT_BARS1M)")
    anc, st = AS.evaluatory("S3")
    return anc, st, AS.schitat(anc, False), AS.schitat(st, True)


def test_dannye_yakor_ne_izmenilsya_a_strogaya_otlichaetsya(s3):
    anc, st, ra, rs = s3
    from fastbt.etap1_eval import Evaluator
    sp = {k: v for k, v in anc.base.items() if k in anc.sk.sig_names}
    ref = Evaluator.excess(anc, sp, AS.PER)[0]["excess"]
    ref = ref[ref.index.year.isin(AS.GODY)].to_numpy()
    assert np.abs(ra["x"] - ref).max() <= 1e-12                          # якорь через V2 = якорь v1 (эталон кода не тронут)
    assert not EvaluatorV2.STROGAYA and AS.AStrogaya.STROGAYA
    assert 0 < np.abs(rs["x"] - ra["x"]).max() < 0.05 and np.isfinite(rs["x"]).all()
    assert (rs["expo"] >= 0).all() and rs["cnt"].sum() > 0
