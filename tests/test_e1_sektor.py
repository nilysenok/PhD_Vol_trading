"""e1_sektor: секторный тренд и ширина (Ж-051, итерация 4). Синтетика; каждый инвариант — с мутацией, которая его роняет."""
import numpy as np
import pandas as pd
import pytest

from fastbt import e1_sektor as es
from fastbt.etap1_data import gate_trades

D = pd.bdate_range("2018-01-01", periods=40)


def _ctx(idx=None, close=None, smap=None):
    return es.Ctx(idx or {}, close or {}, smap if smap is not None else {})


def _zigzag():
    """Рост 20 дней, затем падение: тренд меняет знак внутри ряда."""
    v = np.r_[np.arange(1.0, 21.0), np.arange(19.0, -1.0, -1.0)]
    return pd.Series(v + 10, index=D)


def _pt(**k):
    return dict(sektor_trend="net", shirina="net", ma_n=5, porog_shiriny=0.5) | k


def test_net_net_golden_pozicii_pobaytno():
    pos = np.array([0, 1, 1, 0, -1, -1, 0, 1.0])
    lo, sr = es.allow(_pt(), "SBER", D[:8], None)              # без данных вообще
    assert lo.all() and sr.all() and lo.dtype == bool
    assert gate_trades(pos, lo).tobytes() == pos.tobytes()
    assert gate_trades(pos, sr).tobytes() == pos.tobytes()


def test_trend_po_trendu_i_protiv_storony():
    ctx = _ctx({"MOEXOG": _zigzag()}, smap={"LKOH": "MOEXOG"})
    lo, sr = es.allow(_pt(sektor_trend="po_trendu"), "LKOH", D, ctx)
    assert not lo[:4].any() and not sr[:4].any()               # нет n дней — вход запрещён
    assert lo[4:20].all() and not sr[4:20].any()               # рост: лонг можно, шорт нельзя
    assert sr[25:].all() and not lo[25:].any()                 # падение: наоборот
    lp, sp = es.allow(_pt(sektor_trend="protiv"), "LKOH", D, ctx)
    assert (lp == sr).all() and (sp == lo).all()


@pytest.mark.parametrize("axis,lag", [("MA_LAG", -1), ("CLOSE_LAG", -1)])
def test_trend_bez_budushchego_i_mutatsiya(monkeypatch, axis, lag):
    s = _zigzag()
    pt = _pt(sektor_trend="po_trendu")

    def leak():
        full = es.allow(pt, "LKOH", D, _ctx({"MOEXOG": s}, smap={"LKOH": "MOEXOG"}))
        bad = []
        for d in range(5, 38):
            cut = es.allow(pt, "LKOH", D[:d + 1], _ctx({"MOEXOG": s.iloc[:d + 1]}, smap={"LKOH": "MOEXOG"}))
            bad.append(bool(full[0][d] != cut[0][d] or full[1][d] != cut[1][d]))
        return any(bad)
    assert not leak()
    monkeypatch.setattr(es, axis, lag)                         # окно/закрытие d+1
    assert leak()


def _bars(evening: float):
    ts, cl = [], []
    for k, d in enumerate(D):
        base = 100.0 + k                                       # день за днём растёт: close 18:40 > MA
        for hm, c in (("10:00", base), ("18:40", base), ("23:40", evening)):
            ts.append(pd.Timestamp(f"{d.date()} {hm}"))
            cl.append(c)
    return pd.DataFrame({"timestamp": ts, "open": cl, "high": cl, "low": cl, "close": cl, "volume": 1})


def test_ryad_indeksa_do_1850_i_mutatsiya_vechera(monkeypatch):
    b = _bars(evening=1.0)                                     # вечерний бар 23:40 по 1.0
    s = es.idx_daily(b, "MOEXOG")
    assert (s.to_numpy() == 100.0 + np.arange(len(D))).all()
    pt = _pt(sektor_trend="po_trendu")
    ok = es.allow(pt, "LKOH", D, _ctx({"MOEXOG": s}, smap={"LKOH": "MOEXOG"}))
    assert ok[0][5:].all() and not ok[1].any()
    monkeypatch.setattr(es, "CUT_MIN", 24 * 60)                # вечер входит в день: close = 1.0
    bad = es.idx_daily(b, "MOEXOG")
    assert (bad.to_numpy() == 1.0).all()
    lo, _ = es.allow(pt, "LKOH", D, _ctx({"MOEXOG": bad}, smap={"LKOH": "MOEXOG"}))
    assert not lo[5:].all()                                    # тест видит подмену


def test_tiker_bez_indeksa_sektor_ne_deystvuet(monkeypatch):
    ctx = _ctx({"MOEXOG": _zigzag()}, smap={"LKOH": "MOEXOG"})
    for t in ("AFLT", "LSRG"):
        lo, sr = es.allow(_pt(sektor_trend="po_trendu"), t, D, ctx)
        assert lo.all() and sr.all()
    assert es.bez_indeksa(ctx.smap, ["LKOH", "AFLT", "LSRG"]) == ["AFLT", "LSRG"]
    monkeypatch.setattr(es, "BEZ_INDEKSA_VPUSKAT", False)
    lo, sr = es.allow(_pt(sektor_trend="po_trendu"), "AFLT", D, ctx)
    assert not lo.any() and not sr.any()                       # мутация: сектор без индекса запрещает всё


def _closes():
    up = pd.Series(np.arange(1.0, 41.0), index=D)
    dn = pd.Series(np.arange(40.0, 0.0, -1), index=D)
    short = pd.Series(np.arange(1.0, 11.0), index=D[30:])      # 10 дней истории: при n=5 есть, при n=20 вне знаменателя
    return {"A": up, "B": up, "C": dn, "D": short}


def test_shirina_znachenie_znamenatel_i_storony():
    ctx = _ctx(close=_closes())
    b = ctx.breadth(5)
    assert b.iloc[10] == pytest.approx(2 / 3)                  # D ещё нет, A,B выше MA, C ниже
    assert b.iloc[39] == pytest.approx(3 / 4)                  # D вошёл: 3 из 4
    assert ctx.breadth(20).iloc[39] == pytest.approx(2 / 3)    # у D < 20 дней — вне знаменателя
    assert np.isnan(b.iloc[2])                                 # нет n дней ни у кого
    lo, sr = es.allow(_pt(shirina="po_trendu", porog_shiriny=0.6), "SBER", D, ctx)
    assert lo[10] and not sr[10] and not lo[2] and not sr[2]
    lp, sp = es.allow(_pt(shirina="protiv", porog_shiriny=0.6), "SBER", D, ctx)
    assert sp[10] and not lp[10]
    lo4, sr4 = es.allow(_pt(shirina="po_trendu", porog_shiriny=0.4), "SBER", D, ctx)
    assert lo4[10] and sr4[10] == (2 / 3 <= 0.6)               # пороги 0,4 / 0,6 различаются


def test_shirina_bez_budushchego_i_mutatsiya(monkeypatch):
    cl = _closes()
    cl["C"] = _zigzag()                                        # ширина меняется во времени
    pt = _pt(shirina="po_trendu")

    def leak():
        full = es.allow(pt, "SBER", D, _ctx(close=cl))
        for d in range(6, 38):
            part = {k: v[v.index <= D[d]] for k, v in cl.items()}
            cut = es.allow(pt, "SBER", D[:d + 1], _ctx(close=part))
            if full[0][d] != cut[0][d] or full[1][d] != cut[1][d]:
                return True
        return False
    assert not leak()
    monkeypatch.setattr(es, "SHIRINA_LAG", -1)
    assert leak()


def test_oba_filtra_i_parametry():
    ctx = _ctx({"MOEXOG": _zigzag()}, _closes(), {"LKOH": "MOEXOG"})
    both = es.allow(_pt(sektor_trend="po_trendu", shirina="po_trendu", porog_shiriny=0.6), "LKOH", D, ctx)
    s = es.allow(_pt(sektor_trend="po_trendu"), "LKOH", D, ctx)
    w = es.allow(_pt(shirina="po_trendu", porog_shiriny=0.6), "LKOH", D, ctx)
    assert (both[0] == (s[0] & w[0])).all() and (both[1] == (s[1] & w[1])).all()
    with pytest.raises(ValueError):
        es.allow(_pt(shirina="x"), "LKOH", D, ctx)


def test_zapret_dannykh_2022():
    with pytest.raises(PermissionError):
        es.load_indexes(["MOEXOG"], end=pd.Timestamp("2022-06-01"))
