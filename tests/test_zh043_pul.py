"""Ж-043: пул плато включает достроенных соседей (дефект пула Ж-042, PR #340) и ничья по σ. Синтетика, данных не читает."""
import math
import sys

sys.path[:0] = [".", "scripts"]
import zh043_pul as P                                   # noqa: E402

KEY = lambda p: p["x"]                                  # noqa: E731


def _ev(f, aktiv=lambda x: True):
    return lambda p, rol: dict(p=p, obj=f(p["x"]), aktivnost_ok=aktiv(p["x"]), rol=rol)


def _pool(f, seeds):
    ev = _ev(f)
    return {s: ev({"x": s}, "faktor") for s in seeds}, ev, lambda p: [{"x": p["x"] + d} for d in (-1, 1) if 0 <= p["x"] + d <= 9]


def test_luchshaya_tochka_sosed_vybiraetsya():
    f = lambda x: -abs(x - 5)                           # пик x=5; в посеве его нет, он сосед лидера x=4
    rows, ev, nb = _pool(f, [1, 2, 3, 4])
    top = P.build_pool(rows, ev, nb, KEY)
    assert top[0]["p"] == {"x": 5} and top[0]["rol"] == "sosed"


def test_mutatsiya_staryy_pul_tolko_faktor_oat_teryaet_soseda():
    f = lambda x: -abs(x - 5)
    rows, ev, nb = _pool(f, [1, 2, 3, 4])
    P.build_pool(rows, ev, nb, KEY)
    staryy = sorted((r for r in rows.values() if r["rol"] in ("faktor", "oat")), key=lambda r: -r["obj"])
    assert staryy[0]["p"] == {"x": 4}                   # прежний пул Ж-042 выбрал бы x=4: тест выше его ловит


def test_sosedi_novyh_liderov_dostraivayutsya_do_stabilizatsii():
    f = lambda x: x                                     # цель растёт вправо: лидер каждый круг сдвигается
    rows, ev, nb = _pool(f, [0, 1])
    top = P.build_pool(rows, ev, nb, KEY, n=1, max_rep=2)
    assert top[0]["p"] == {"x": 4}                      # круг + 2 повтора: 1 -> 2 -> 3 -> 4; дальше лидеров не ищем
    assert "nb" in rows[3] and top[0]["zakryt"] == 1 and top[0]["stab"] == 0   # закрывающий проход достроил соседей x=4; лидер 5 появился после
    assert top[0]["kray"] == 1


def test_nedopustimye_ne_vhodyat_v_pul():
    f = lambda x: math.nan if x == 5 else -abs(x - 5)   # mSR не определён -> точка недопустима
    ev = _ev(f, aktiv=lambda x: x != 4)
    rows = {s: ev({"x": s}, "oat") for s in (3, 4, 5)}
    top = P.build_pool(rows, ev, lambda p: [], KEY)
    assert [r["p"]["x"] for r in top] == [3]            # x=5 — nan, x=4 — мало сделок


def test_neraznye_tochki_odin_rezultat_ne_zapolnyayut_top_i_ne_sosedi():
    f = lambda x: 1.0 if x in (4, 5) else 0.0           # x=4 и x=5 различаются только параметром, который не действует
    ev0 = _ev(f)
    ev = lambda p, rol: dict(ev0(p, rol), sig=(f(p["x"]),))
    rows = {4: ev({"x": 4}, "oat"), 5: ev({"x": 5}, "oat")}
    top = P.build_pool(rows, ev, lambda p: [{"x": p["x"] + d} for d in (-1, 1)], KEY)
    assert len([r for r in top if r["obj"] == 1.0]) == 1          # дубль результата — одна точка
    r = next(t for t in top if t["obj"] == 1.0)
    assert 5 not in r["nb"] and 4 not in r["nb"]                  # сосед с тем же результатом не сосед


def test_plato_i_chislo_sosedey_s_polozhitelnoy_celyu():
    rows, ev, nb = _pool(lambda x: x - 4.5, [4])
    top = P.build_pool(rows, ev, nb, KEY, max_rep=0)
    r = next(t for t in top if t["p"] == {"x": 4})
    assert r["sosedey"] == 2 and r["sosedey_pos"] == 1 and r["kray"] == 1
