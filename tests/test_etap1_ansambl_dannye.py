"""Ж-051 итерация 2 на данных 2017–2019: golden «ансамбль из одной точки = точка» побайтно, сверка с s9, мутации.
Нужны 1-минутки (FASTBT_BARS1M) и данные этапа 1; нет данных — пропуск."""
import numpy as np
import pytest

from fastbt import bars5m

pytestmark = pytest.mark.skipif(not bars5m.SRC.exists(), reason="нет 1-минуток D25 (FASTBT_BARS1M)")
PT = dict(k_s=2.0, R=2.0, tau=5, proverka="bars5m", vhod=0.5, sigma="ewma094", razmer=20)


@pytest.fixture(scope="module")
def evr():
    from fastbt import etap1
    from fastbt.etap1_ansambl_eval import AnsamblEvaluator
    return AnsamblEvaluator("S2", etap1.build_inputs("S2"), etap1.base_point("S2"))


def _same(a, b):
    return all(a["excess_d"][y].tobytes() == b["excess_d"][y].tobytes() if hasattr(a["excess_d"][y], "tobytes")
               else np.asarray(a["excess_d"][y]).tobytes() == np.asarray(b["excess_d"][y]).tobytes() for y in a["excess_d"])


@pytest.mark.parametrize("mode", ["srednee", "golos"])
def test_golden_ansambl_iz_odnoy_tochki(evr, mode):
    from fastbt.etap1_eval import YAKOR
    ref = evr.evaluate({**YAKOR, **PT}, "train")
    got = evr.evaluate_ens({**YAKOR, **PT}, [{**YAKOR, **PT}], mode)
    assert _same(ref, got) and ref["sdelok_y"] == got["sdelok_y"] and ref["msr_by_year"] == got["msr_by_year"]


def test_ansambl_iz_dvukh_tochek_otlichaetsya_i_mutatsiya_golos(evr):
    from fastbt.etap1_eval import YAKOR
    c = {**YAKOR, **PT}
    nb = dict(c)
    for k in evr.sk.sig_names:
        v = evr.full(c)[k]
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            nb[k] = int(round(v * 1.5)) if isinstance(v, int) else v * 1.5
            break
    ref = evr.evaluate(c, "train")
    s = evr.evaluate_ens(c, [c, nb, {**nb}], "srednee")
    g = evr.evaluate_ens(c, [c, nb, {**nb}], "golos")
    assert not _same(ref, s)                      # сигнал соседа меняет результат
    assert not _same(s, g)                        # srednee ≠ golos: режим подменить нельзя незаметно


def _spy_weights(monkeypatch):
    from fastbt import etap1_ansambl_eval as E
    W, real = [], E.run_e1

    def spy(*a, **kw):
        W.append(np.asarray(kw["w"], float))
        return real(*a, **kw)
    monkeypatch.setattr(E, "run_e1", spy)
    return W


def _two(evr):
    from fastbt.etap1_eval import YAKOR
    c = {**YAKOR, **PT, "razmer": "ravnyy"}
    nb = dict(c)
    for k in evr.sk.sig_names:
        v = evr.full(c)[k]
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            nb[k] = int(round(v * 1.5)) if isinstance(v, int) else v * 1.5
            break
    return c, nb


def test_ves_m_dohodit_do_dvizhka(evr, monkeypatch):
    """Вес сделки в движке = размер центра × |среднее позиций|: при расхождении точек есть дробные веса."""
    W = _spy_weights(monkeypatch)
    c, nb = _two(evr)
    evr.evaluate_ens(c, [c, nb], "srednee")
    w = np.concatenate(W)
    assert ((w > 0) & (w < 1)).any() and set(np.unique(w[(w == 0) | (w == 1)])) <= {0.0, 1.0}
    assert np.allclose(w[(w > 0) & (w < 1)], 0.5)                     # две точки: |m| = 1/2 там, где одна из них не в позиции


def test_mutatsiya_ves_m_poteryan_lovitsya(evr, monkeypatch):
    from fastbt import etap1_ansambl_eval as E
    c, nb = _two(evr)
    ok = evr.evaluate_ens(c, [c, nb], "srednee")
    real = E.combine
    monkeypatch.setattr(E, "combine", lambda P, mode: (lambda d, w: (d, (w > 0).astype(float)))(*real(P, mode)))   # вес |m| → 1
    W = _spy_weights(monkeypatch)
    bad = evr.evaluate_ens(c, [c, nb], "srednee")
    w = np.concatenate(W)
    assert not ((w > 0) & (w < 1)).any()                              # первая проверка теста веса упала бы
    assert not _same(ok, bad)                                         # и результат движка изменился


def test_sverka_s9_tochki_vosproizvodyatsya(evr):
    from fastbt.etap1_ansambl_eval import sverka_s9
    from fastbt.etap1_eval import YAKOR
    r = sverka_s9(evr, [{**YAKOR, **PT}])
    assert r["rashozhdeniy"] == [] and r["sovpalo"] + r["net_etalona"] == 1
