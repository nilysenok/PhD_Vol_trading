"""Ист-v6, малая проверка на реальных данных ≤ 2021 (train4, 3 тикера, 1 процесс, S1): каждая ось v6 с источником и первые
20 точек Морриса считаются без отказа из-за источников. Нет файлов данных (FASTBT_MAIN) — пропуск. Сеть не нужна; Соболь/большой счёт не запускается."""
import pytest

from fastbt import canon_v6, data, etap1, istochniki_v6 as iv, morris_v6 as mv
from fastbt.eval_v6 import EvaluatorV6, KONTEKST, SOB, VN
from fastbt.etap1_eval import years_of

pytestmark = pytest.mark.skipif(not data.OHLCV.exists(), reason="нет data/fastbt/daily (FASTBT_MAIN)")
TIK, SID, PER = ["SBER", "LKOH", "TATN"], "S1", "train4"
ОСИ = ("rel_signal",) + KONTEKST + SOB + VN
ЛОЖЬ_ИСТОЧНИК = ("IstochnikiV6", "нет Dannye", "нет sig", "нет rel", "не подключ", "ещё не")


@pytest.fixture(scope="module")
def sreda():
    from fastbt import sweep_sk
    old, sweep_sk.load_ticker = sweep_sk.load_ticker, etap1.load_ticker_nf
    try:
        ev = sweep_sk.SkEvaluator(SID, TIK)
    finally:
        sweep_sk.load_ticker = old
    inp = etap1._build(SID, ev, etap1.END_TRAIN, PER)
    evr = EvaluatorV6(SID, inp, etap1.base_point(SID), 0.0004, fdr="t0d", ist=iv.zagruzit(PER, TIK))
    kub = canon_v6.load_kub(SID)
    return evr, kub, canon_v6.osi(kub)


def _schitat(evr, pt):
    """Точка с дивидендами (sob_tip ∋ div): при div_verified=True (есть свод Д8, #717) считается без отказа; при
    div_verified=False (свода нет, ❓5) отказывается ПО ЗАМЫСЛУ: ValueError «не сверены»."""
    try:
        r, ex_p, _ = evr.evaluate_v2(pt, PER)
    except ValueError as e:
        assert not evr.ist.div_verified and "div" in str(pt.get("sob_tip", "")) and "дивиденды не сверены" in str(e), (e, pt)
        return {"status": "div_otkaz"}, None
    assert evr.ist.div_verified or "div" not in str(pt.get("sob_tip", "")) or r["status"] == "ne_schitana", r
    return r, ex_p


def _ok(r):
    p = str(r.get("prichina", ""))
    assert r.get("status", "ok") in ("ok", "ne_schitana", "div_otkaz") and not any(s in p for s in ЛОЖЬ_ИСТОЧНИК), r
    return r.get("status", "ok")


def test_kazhdaya_os_s_istochnikom_schitaetsya(sreda):
    evr, kub, ox = sreda
    baza = {n: g[0] for n, g, _ in ox}
    posch = []
    for n, g, _ in ox:
        if n not in ОСИ:
            continue
        for v in g[1:3]:
            pt = canon_v6.canon_v6(SID, {**baza, n: v}, kub)
            if pt is not None:
                r, _ = _schitat(evr, pt)
                posch.append((n, v, _ok(r)))
    assert {n for n, _, s in posch if s == "ok"} >= {"rel_signal", "rezhim", "sob_tip", "vn_uslovie"}, posch


def test_20_tochek_morrisa_s1_vse_poschitany(sreda):
    evr, kub, ox = sreda
    ind, _ = mv.traektorii([len(g) for _, g, _ in ox], 1, int(kub["sobol"]["morris_seed"]), 1)
    pts, seen = [], set()
    for row in ind[0]:
        pt = canon_v6.canon_v6(SID, {n: g[i] for (n, g, _), i in zip(ox, row)}, kub)
        if pt is not None and repr(sorted(pt.items(), key=str)) not in seen:
            seen.add(repr(sorted(pt.items(), key=str)))
            pts.append(pt)
    pts = pts[:20]
    assert len(pts) == 20
    assert all(_ok(_schitat(evr, pt)[0]) for pt in pts)


def _tochka_div(sreda):
    evr, kub, ox = sreda
    baza = {n: g[0] for n, g, _ in ox}
    sob = next(g for n, g, _ in ox if n == "sob_tip")
    for v in sob:
        if "div" in str(v):
            pt = canon_v6.canon_v6(SID, {**baza, "sob_tip": v}, kub)
            if pt is not None:
                return evr, pt
    raise AssertionError(("нет точки с div", sob))


def test_div_schitaetsya_pri_svode(sreda):
    evr, pt = _tochka_div(sreda)
    assert evr.ist.div_verified is True
    r, _ = _schitat(evr, pt)
    assert _ok(r) == "ok", r


def test_div_otkaz_bez_svoda(sreda):
    import copy
    import dataclasses
    evr, pt = _tochka_div(sreda)
    ev2 = copy.copy(evr)
    # sha другой, как и без свода (div = полная таблица): кэши точек общие у копии и оригинала, ключ их разводит
    ev2.ist = dataclasses.replace(evr.ist, div_verified=False, sha="bez_svoda")
    assert _schitat(ev2, pt)[0] == {"status": "div_otkaz"}
