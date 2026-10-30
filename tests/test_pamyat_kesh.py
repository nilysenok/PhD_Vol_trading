"""Кэш оценщика (_sigs, _pos) не копится между точками прогона (рост RSS рабочего Соболя: 1,8–15,6 МБ/точку, расследование 30.10).
Идёт реальный EvaluatorV2.evaluate_v2 (общий для v4/v5/v6) с заглушкой `evaluate`. Мутация «кэш не сбрасывается» (удалить цикл
очистки в evaluate_v2) = красный тест: на входе k-й точки в кэше k записей. Поведение ledger не меняется (кэш — мемоизация), сверено побайтно."""
import numpy as np

from fastbt.etap1_eval_v2 import EvaluatorV2, NeSchitana


class Zaglushka(EvaluatorV2):
    """Как EvaluatorV5: кладёт сигналы и позиции в словари по ключу точки; NeSchitana — ранний выход evaluate_v2 после очистки."""
    def __init__(self):
        self._sigs, self._pos, self.vhod, self.comm, self.fdr = {}, {}, [], 0.0004, "net"

    def evaluate(self, pt, period):
        self.vhod.append((len(self._sigs), len(self._pos)))
        self._sigs[tuple(pt.items())] = {"T": np.zeros(1000)}
        self._pos[("T", tuple(pt.items()))] = np.zeros(1000)
        raise NeSchitana("заглушка")


def test_kesh_ne_kopitsya_mezhdu_tochkami():
    ev = Zaglushka()
    for i in range(50):
        assert ev.evaluate_v2({"a": i}, "train4")[0]["status"] == "ne_schitana"
    assert max(max(x) for x in ev.vhod) == 0, f"кэш растёт: на входе точек {ev.vhod[:4]}…{ev.vhod[-1]}"
    assert len(ev._sigs) == 1 and len(ev._pos) == 1             # последняя точка остаётся (диагностика читает после вызова)


def test_kesh_v6_cherez_tot_zhe_put():
    from fastbt.eval_v6 import EvaluatorV6
    assert EvaluatorV6.evaluate_v2 is not EvaluatorV2.evaluate_v2
    import inspect
    assert "EvaluatorV2.evaluate_v2(self" in inspect.getsource(EvaluatorV6.evaluate_v2)    # v6 идёт через общий метод с очисткой
