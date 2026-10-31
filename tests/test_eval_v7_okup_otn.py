"""V7-3 × V7-1a: оси otnositelno и k_okup вместе (S1, синтетика): обе оси снимаются с точки до v6, ключ кэша отличается от каждой оси по отдельности."""
import pandas as pd

from fastbt.eval_v6 import EvaluatorV6
from fastbt.eval_v7 import EvaluatorV7, Ist7
from fastbt.eval_v7_otn import OtnIst


def _klyuch(monkeypatch, pt):
    vid = []
    monkeypatch.setattr(EvaluatorV6, "evaluate_v2", lambda self, p, period, vozm=None: vid.append((dict(p), self._fkey, self.otn, self._okup_k)) or "ok")
    e = object.__new__(EvaluatorV7)
    imoex = pd.Series([1.0, 2.0], index=pd.to_datetime(["2018-01-02", "2018-01-03"]))
    e.sid, e.proxy, e.ist7, e._fk = "S1", "S1", Ist7(otn=OtnIst(imoex, {})), ()
    assert e.evaluate_v2(pt, "train4") == "ok"                       # оценка не падает
    return vid[0]


def test_obe_osi_vmeste_klyuch_otlichaetsya_ot_kazhdoj(monkeypatch):
    oba = _klyuch(monkeypatch, {"a": 1, "otnositelno": "sektor", "k_okup": 3})
    otn, okup, net = (_klyuch(monkeypatch, p) for p in ({"a": 1, "otnositelno": "sektor"}, {"a": 1, "k_okup": 3}, {"a": 1}))
    assert oba[0] == {"a": 1} and oba[2:] == ("sektor", 3)           # обе оси сняты с точки, режимы заданы
    assert len({repr(x[1]) for x in (oba, otn, okup, net)}) == 4
    assert net[1] == () and set(oba[1]) == set(otn[1]) | set(okup[1])  # ключ содержит обе добавки
