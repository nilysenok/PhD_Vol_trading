"""Якорь этапа 1 против s9 после правки дивидендов (Ж-057 п. 1): режим «якорь» (exec_e + div) = «A только Sk» (lr с дивидендом),
2018 и 2019, допуск 1e-12 — и при D×0,87, и при полном D. Мутация: etap1 не передаёт дивиденды в exec_e → якорь расходится.
Нужны 1-минутки (FASTBT_BARS1M) и данные s9; нет данных — пропуск. python3 -m pytest tests/test_dividendy_yakor.py -q"""
import pytest

from fastbt import bars5m, dividendy

pytestmark = pytest.mark.skipif(not bars5m.SRC.exists(), reason="нет 1-минуток D25 (FASTBT_BARS1M)")
SIDS = ["S2", "S4"]            # у S4 больше всего позиций через отсечку (#362), S2 — со смешанными знаками
TOL = 1e-12


def _diff(sid, comm=0.0004):
    from fastbt import etap1
    from fastbt.etap1_eval import Evaluator
    inp = etap1.build_inputs(sid)
    return etap1.anchor_s9_diff(sid, Evaluator(sid, inp, etap1.base_point(sid), comm), comm)


@pytest.mark.parametrize("nalog", [0.87, 1.0])
@pytest.mark.parametrize("sid", SIDS)
def test_yakor_etap1_raven_s9_s_dividendami(sid, nalog, monkeypatch):
    monkeypatch.setattr(dividendy, "DIV_NALOG", nalog)
    d = _diff(sid)
    assert max(d.values()) <= TOL, d


def test_mutatsiya_etap1_bez_dividendov_lovitsya(monkeypatch):
    """Подложить: run_e1 получает div=None (etap1 «забыл» дивиденды) — якорь против s9 обязан разойтись."""
    from fastbt import etap1_eval, etap1_run

    def bad(*a, **kw):
        kw["div"] = None
        return etap1_run.run_e1(*a, **kw)
    monkeypatch.setattr(etap1_eval, "run_e1", bad)
    assert max(_diff("S4").values()) > TOL
