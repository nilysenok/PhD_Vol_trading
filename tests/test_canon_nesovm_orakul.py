"""Свойство: каноническая точка (не None) не получает в оценщике v6 отказ по 7 причинам; выброшенная — получила бы (правила не шире отказов).
Оракул без данных: та же сборка точки, что в EvaluatorV6.evaluate_v2 (razdelit → razobrat → razbor → primenit), затем те же проверки оценщика и движка."""
import functools
import warnings
from dataclasses import replace
from types import SimpleNamespace
from unittest import mock

import pytest
from scipy.stats import qmc

from fastbt import canon_v6 as c6, canon_v6_nesovm as nsv, eval_v6 as ev6, eval_v6_ispoln as isp6, eval_v6_osi as osi6, eval_v6_piv as piv6, \
    eval_v6_trend as trend6
from fastbt.canon_v5 import NET, SIGNAL, TREND
from fastbt.eval_v5 import rodnoy_ok
from fastbt.eval_v5_osi import razbor
from fastbt.etap1_eval_v2 import EvaluatorV2, NeSchitana
from fastbt.exec_e import _check_params
from fastbt.exec_e_parab import parab_check

N = 2000
SIDS = ("S1", "S2", "S3", "S4", "S5", "S6")
STUB = SimpleNamespace(inp=SimpleNamespace(tick={"X": 0.01}), comm=0.0004)


class _Vse:
    def __contains__(self, k):
        return True


def _tag(tags, f, msgs, exc):
    try:
        f()
    except exc as e:
        tags |= {t for m, t in msgs.items() if m in str(e)}


def otkazy_pravila():
    """Правила-отказы оценщика: все, кроме bb_predel (структурный ноль, не отказ). Отбор по тождеству функции, не по индексу."""
    return tuple(p for p in nsv.PRAVILA if p is not nsv.bb_predel)


def otkazy(sid, c):
    """Номера правил 1–7, по которым оценщик/движок отказал бы канонической точке c (первый отказ в каждой проверке)."""
    pt5, v6 = ev6.razdelit(sid, c)
    pt5, piv = piv6.razobrat(trend6.vernut(sid in TREND, pt5, v6))
    r = razbor(sid, pt5, SIGNAL[sid], None)
    if "vhod_tip" in v6 and sid in ("S3", "S4"):
        r.nov["vhod_tip"] = v6["vhod_tip"]
    osi6.primenit(r, v6, _Vse())
    isp6.primenit(r, v6)
    piv6.primenit(r, piv, v6)
    trend6.primenit(sid in TREND, r, v6)
    p = replace(EvaluatorV2._params(STUB, r.pt4, "X", 5.0, False, r.nov, {}), **r.ext)
    t = set()
    _tag(t, lambda: isp6.proverit(r, v6), {"vyhod_lim и t_vyhod": 1}, NeSchitana)
    _tag(t, lambda: _check_params(p), {"t_vyhod требуют": 8}, ValueError)
    _tag(t, lambda: parab_check(p), {"несовместим": 2}, ValueError)
    _tag(t, lambda: piv6.proverit(r, piv, sid), {"M = SMA": 4, "единственный стоп": 5, "канал tejk_M": 6}, NeSchitana)
    if r.nov["vhod_tip"] == "stop_zayavka" and "a_rang" in r.ext:                    # eval_v5.EvaluatorV5._rows
        t.add(3)
    if r.nov["tejk_signal"] != NET and (v6.get("norm") == "s252" or not rodnoy_ok(sid, r.sp)):   # eval_v6._rows / eval_v5._rows
        t.add(7)
    return t


@functools.cache
def vyborka(sid):
    """[(c0, otkazy(c0))]: каноны 2000 точек Соболя куба sid БЕЗ 7 правил (c0 ≠ None), оракул считается один раз."""
    kub = c6._kub(sid)
    ox, s = c6.osi(kub), kub["sobol"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        u = qmc.Sobol(len(ox), scramble=s["scramble"], seed=s["seed"]).random(2048)[:N]
    out = []
    with mock.patch.object(nsv, "PRAVILA", ()):
        for row in u:
            pt = {n: g[c6._indeks(x, len(g), sp.get("p_vykl"))] for (n, g, sp), x in zip(ox, row)}
            c0 = c6.canon_v6(sid, pt, kub)
            if c0 is not None:
                out.append((c0, otkazy(sid, c0)))
    return out


def rashozhdeniya(sid):
    """Точки, где правила и оракул расходятся: [(c0, правила, отказы)]. Пусто — каноны не получают отказ и не шире отказа."""
    bad = []
    for c0, otk in vyborka(sid):
        trig = {i + 1 for i, f in enumerate(otkazy_pravila()) if f(sid, c0)}     # 1–8 — отказы оценщика; 9 (bb_predel) — не отказ, свой тест
        if trig & {1, 8}:
            trig |= {1, 8}                  # движок (exec_e._check_params) одним текстом отказывает и по правилу 1, и по правилу 8
        if not otk <= trig or bool(trig) != bool(otk):
            bad.append((c0, trig, otk))
    return bad


@pytest.mark.parametrize("sid", SIDS)
def test_kanon_ne_poluchaet_otkaz_po_semi_prichinam(sid):
    assert rashozhdeniya(sid) == []


def test_vyborka_nepustaya_po_kazhdoj_prichine():
    vstrecheno = set().union(*(o for sid in SIDS for _, o in vyborka(sid)))
    assert vstrecheno >= {1, 2, 3, 4, 5, 6, 7}         # без этого свойство пустое: каждая причина встречается в выборке оракула


def test_mutaciya_pravilo_snyato_lovitsya_svojstvom():
    for i in range(len(otkazy_pravila())):              # правила-отказы; bb_predel (не отказ) — test_canon_bb_predel.py
        with mock.patch.object(nsv, "PRAVILA", tuple(p for p in nsv.PRAVILA if p is not otkazy_pravila()[i])):
            assert any(rashozhdeniya(s) for s in SIDS), i


def test_mutaciya_vyhod_lim_rasshiren_lovitsya_svojstvom():
    mut = lambda sid, q: q.get("vyhod_lim", NET) == "da"                  # noqa: E731  (выбрасывает и при t_vyhod = close)
    with mock.patch.object(nsv, "PRAVILA", (mut,) + nsv.PRAVILA[1:]):
        assert any(rashozhdeniya(s) for s in ("S1", "S2", "S5", "S6"))
