"""Проверка тестировщика #726 (сверх тестов автора): точность правил по одному (ii), порядок «до canon_v5», покрытие каждого правила.
Оракул `otkazy` — из test_canon_nesovm_orakul (та же сборка точки, что в EvaluatorV6). Сеть и данные не нужны."""
import functools
import warnings
from unittest import mock

import numpy as np
import pytest
from scipy.stats import qmc
from test_canon_nesovm_orakul import N, SIDS, otkazy

from fastbt import canon_v6 as c6, canon_v6_nesovm as nsv

GRUPPY = ({1, 8}, {4, 5, 6})      # отказ в одной проверке оценщика/движка маскирует остальные причины той же проверки (первый raise)


@functools.cache
def tochki(sid):
    """[(pt, real, c0, trig, otk)]: pt — точка Соболя; real — canon_v6 с правилами; c0 — без правил; trig — номера сработавших правил на c0; otk — отказы оракула."""
    kub = c6._kub(sid)
    ox, s = c6.osi(kub), kub["sobol"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        u = qmc.Sobol(len(ox), scramble=s["scramble"], seed=s["seed"]).random(2048)[:N]
    out = []
    for row in u:
        pt = {n: g[c6._indeks(x, len(g), sp.get("p_vykl"))] for (n, g, sp), x in zip(ox, row)}
        real = c6.canon_v6(sid, pt, kub)
        with mock.patch.object(nsv, "PRAVILA", ()):
            c0 = c6.canon_v6(sid, pt, kub)
        if c0 is None:
            assert real is None
            continue
        out.append((pt, real, c0, {i + 1 for i, f in enumerate(nsv.PRAVILA) if f(sid, c0)}, otkazy(sid, c0)))
    return out


@pytest.mark.parametrize("sid", SIDS)
def test_real_ravno_pravilam_na_kanone_bez_pravil(sid):
    """Порядок «правило до canon_v5»: итог canon_v6 = None ⇔ правила срабатывают на каноне c0; иначе итог = c0 (правила не меняют точку)."""
    for _, real, c0, trig, _ in tochki(sid):
        assert (real is None) == bool(trig)
        assert real is None or real == c0


@pytest.mark.parametrize("sid", SIDS)
def test_poryadok_do_canon_v5_na_sluchajnyh_tochkah(sid):
    """То же, что выше, на 20000 равномерных точках (не Соболь): правила до canon_v5 и на каноне c0 дают один итог (сворачивание v5 ось не меняет)."""
    kub = c6._kub(sid)
    ox = c6.osi(kub)
    u = np.random.default_rng(726).random((20000, len(ox)))
    for row in u:
        pt = {n: g[c6._indeks(x, len(g), sp.get("p_vykl"))] for (n, g, sp), x in zip(ox, row)}
        real = c6.canon_v6(sid, pt, kub)
        with mock.patch.object(nsv, "PRAVILA", ()):
            c0 = c6.canon_v6(sid, pt, kub)
        assert (real is None) == (c0 is None or nsv.nesovm(sid, c0))


@pytest.mark.parametrize("sid", SIDS)
def test_vybroshennaya_tochka_poluchila_by_otkaz_po_prichine(sid):
    """(ii) Точка с единственным сработавшим правилом k получила бы отказ именно по k; при нескольких — недостающая причина замаскирована в своей проверке."""
    for _, _, _, trig, otk in tochki(sid):
        if not trig:
            assert not otk                                           # (i) не выброшена → отказа нет
            continue
        assert otk - trig <= ({1, 8} if trig & {1, 8} else set()), (trig, otk)   # отказ без правила — только связка 1 ↔ 8
        for k in trig - otk:                                         # правило сработало, а причина k не видна → её маскирует другая из той же проверки
            assert any(k in g and otk & g for g in GRUPPY), (k, trig, otk)
        if len(trig) == 1:
            (k,) = trig
            assert otk == {k} or (k in (1, 8) and otk and otk <= {1, 8}), (k, otk)


def test_edinichnye_srabatyvaniya_v_vyborke():
    """Правила 6 и 8 в выборке Соболя (2000 точек × 6 кубов) одни не срабатывают (6 — 7 точек, всегда с 5; 8 — только с 1 или 7): их точность — адресными точками ниже."""
    odin = {k: 0 for k in range(1, len(nsv.PRAVILA) + 1)}
    for sid in SIDS:
        for _, _, _, trig, _ in tochki(sid):
            if len(trig) == 1:
                odin[next(iter(trig))] += 1
    assert [k for k, v in odin.items() if not v] == [6, 8], odin


_KE = dict(tejk="rodnoy", k_e=0.5, vyhod_lim="da", t_vyhod="close")
ADRES = [   # (куб, оси, правила, отказы оракула): точка базы + оси; ожидание — ровно одно правило и отказ ровно по нему
    ("S5", dict(stop_nach="piv:next", tejk="pol_puti"), {6}, {6}),
    ("S2", dict(**_KE, ma_tip="sma", tau=5), {8}, {8}),
    ("S6", dict(**_KE, tau=5), {8}, {8}),
    ("S2", dict(**_KE, ma_tip="sma", tau="net"), set(), set()),                  # без τ rodnoy идёт по close — совместимо
    ("S6", dict(**{**_KE, "vyhod_lim": "net"}, tau=5), set(), set()),
    ("S1", dict(k_e=0.5, vyhod_lim="da", t_vyhod="1500"), {1}, {1}),
]


@pytest.mark.parametrize("sid,kw,pravila,otkaz", ADRES)
def test_adresnaya_tochnost_pravil_6_i_8(sid, kw, pravila, otkaz):
    from test_canon_nesovm import baza
    with mock.patch.object(nsv, "PRAVILA", ()):
        c0 = c6.canon_v6(sid, baza(sid, **kw))
    assert c0 is not None
    assert {i + 1 for i, f in enumerate(nsv.PRAVILA) if f(sid, c0)} == pravila
    otk = otkazy(sid, c0)
    assert otk == otkaz or (pravila == {1} and otk <= {1, 8} and otk)
    assert (c6.canon_v6(sid, baza(sid, **kw)) is None) == bool(pravila)
