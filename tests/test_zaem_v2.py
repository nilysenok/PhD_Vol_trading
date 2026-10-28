"""Ж-089: плата за заём шорта — только за перенос через ночь (fastbt/zaem_v2.py). Синтетика, движок exec_e по 5-мин барам.
Три случая владельца + мутации (исходник функции правится текстом и исполняется: каждая мутация роняет check_vse).
Шарды: порядок и номер как у Pisatel; pisat() не меняет port побайтно."""
import hashlib
import inspect
import json

import numpy as np
import pytest

from fastbt import etap1_run, ledger_v2, zaem_v2
from fastbt.exec_e_types import Day, Params

DATES = np.array(["2018-03-01", "2018-03-02", "2018-03-05", "2018-03-06"], dtype="datetime64[D]")   # чт, пт, пн, вт


def bars(*rows):
    a = np.array(rows, float).reshape(-1, 4)
    return Day(a[:, 0], a[:, 1], a[:, 2], a[:, 3], False)


def flat(x=100.0, n=3):
    return bars(*[(x, x, x, x)] * n)


def rows_of(days, sig, **kw):
    p = Params(comm=0.0, k=0.5, **kw)
    n = len(days)
    return etap1_run.run_e1("T", days, np.array(sig, float), np.full(n, 0.02), p, win=(0, n - 1))


def short_intraday():
    """Шорт: лимит 101 исполнен на дне входа (1), стоп 103.02 пробит 5-мин баром того же дня → exit_day == entry_day."""
    d1 = bars((100, 100, 100, 100), (100, 101.5, 100, 101), (101, 104, 101, 103))
    return rows_of([flat(), d1, flat(), flat()], [1, 0, 0, 0], side=-1, b=1.0)


def hold_over(side):
    """Вход в пятницу (день 1), выход по сигналу на закрытии понедельника (день 2)."""
    fill = bars((100, 100, 100, 100), (100, 101.5, 98.5, 100), (100, 100, 100, 100))
    return rows_of([flat(), fill, flat(), flat()], [1, 1, 0, 0], side=side)


def check_vse(baza, stavka_znak=1.0):
    n = len(DATES) - 1
    w = np.ones(len(DATES))
    r = short_intraday()
    t = r.trades[0]
    assert t.side == -1 and t.entry_day == t.exit_day == 1, "синтетика: шорт должен закрыться в день входа"
    assert not baza(r.trades, DATES, w, 0, n).any(), "внутридневной шорт: плата 0"
    r = hold_over(-1)
    t = r.trades[0]
    assert (t.side, t.entry_day, t.exit_day) == (-1, 1, 2), "синтетика: шорт пятница → понедельник"
    b = baza(r.trades, DATES, w, 0, n)
    assert b[1] == pytest.approx(3 / 365, abs=1e-15) and b.sum() == pytest.approx(3 / 365, abs=1e-15), "пт → пн: 3 дня"
    assert not np.delete(b, 1).any()
    w2 = np.full(len(DATES), 0.5)
    assert baza(r.trades, DATES, w2, 0, n).sum() == pytest.approx(0.5 * 3 / 365, abs=1e-15), "плата ∝ весу"
    rl = hold_over(1)
    assert rl.trades[0].side == 1 and rl.trades[0].exit_day == 2
    assert not baza(rl.trades, DATES, w, 0, n).any(), "лонг: плата 0 всегда"


def test_tri_sluchaya_vladeltsa():
    check_vse(zaem_v2.baza_tikera)


def test_okno_obrezaet_progrev_i_summa_tochna():
    r = hold_over(-1)
    w = np.ones(len(DATES))
    assert not zaem_v2.baza_tikera(r.trades, DATES, w, 2, 3).any(), "ночь пт → пн лежит до окна [2; 3)"
    assert zaem_v2.baza_tikera(r.trades, DATES, w, 1, 3).sum() == pytest.approx(3 / 365, abs=1e-15)


def test_derzhanie_do_posledneho_dnya_okna():
    r = rows_of([flat(), bars((100, 100, 100, 100), (100, 101.5, 98.5, 100), (100, 100, 100, 100)), flat(), flat()],
                [1, 1, 1, 0], side=-1)
    assert r.trades[0].exit_day in (None, 3)
    s = zaem_v2.baza_tikera(r.trades, DATES, np.ones(len(DATES)), 0, 3).sum()
    assert s == pytest.approx(4 / 365, abs=1e-15), "пт 02.03 → вт 06.03: 4 календарных дня (3 + 1)"


def mutant(imya, staroe, novoe):
    src = inspect.getsource(getattr(zaem_v2, imya))
    assert staroe in src, f"мутация не применима: {staroe!r}"
    ns = dict(vars(zaem_v2))
    exec(src.replace(staroe, novoe), ns)                                                    # noqa: S102 — мутация исходника
    return ns[imya]


MUT = {
    "торговые дни вместо календарных": ('np.diff(np.asarray(dates).astype("datetime64[D]")).astype(np.int64)',
                                        "np.ones(len(dates) - 1, np.int64)"),
    "плата за внутридневной шорт": ("tr.exit_day, b)", "tr.exit_day + 1, b)"),
    "плата за лонг": ("if tr.side >= 0:", "if False:"),
    "знак платы": ("out[lo - a:hi - a] += float", "out[lo - a:hi - a] -= float"),
}


@pytest.mark.parametrize("nazv", list(MUT))
def test_mutatsiya_lovitsya(nazv):
    f = mutant("baza_tikera", *MUT[nazv])
    with pytest.raises(AssertionError):
        check_vse(f)


def check_znak(f):
    p = np.array([0.001, 0.002])
    assert (f(p, np.array([0.0, 0.01]), 0.05) <= p).all() and f(p, np.array([0.0, 0.01]), 0.05)[1] < p[1]
    assert np.array_equal(f(p, np.array([0.0, 0.01]), 0.0), p)


def test_excess_s_platoy_znak():
    check_znak(zaem_v2.excess_s_platoy)
    with pytest.raises(AssertionError):
        check_znak(mutant("excess_s_platoy", "np.asarray(port, float) - ", "np.asarray(port, float) + "))


def test_ocenshchik_daet_ryad_na_dannyh_do_2020():
    """S1, train4: ряд float64 [T] ≥ 0, нан нет; у точки без шортов (storona=long) ряд нулевой; excess не затронут."""
    from test_etap1_eval_v2 import PER, make, rand_points
    ev = make("S1")
    pt = rand_points("S1", 1, 20261028)[0]
    row, x, _ = ev.evaluate_v2(pt, PER)
    z = ev.zaem
    assert row["status"] == "ok" and z.dtype == np.float64 and z.shape == x.shape
    assert np.isfinite(z).all() and (z >= 0).all() and z.sum() > 0, "обе стороны: шорты платят за ночи"
    assert (zaem_v2.excess_s_platoy(x, z, 0.05) <= x).all()
    row2, x2, _ = ev.evaluate_v2({**pt, "storona": "long"}, PER)
    assert not ev.zaem.any(), "только лонг → плата 0"
    assert row2["status"] == "ok"


# --- шарды рядом с ledger ---
def make_pisatel(tmp_path, shard=3, T=5):
    man = dict(T=T, kod="k", vhody="v", period="train4")
    return ledger_v2.Pisatel(tmp_path, "S1", 0, man, shard=shard)


def feed(p, n, T=5, zaem=True):
    for i in range(n):
        row = {"status": "ok", "klyuch_sha": f"sha{i}"}
        port = np.full(T, float(i))
        zaem_v2.pisat(p, row, port, np.full((17, T), np.nan), np.full(T, 100.0 + i) if zaem else None)


def test_shardy_zaem_v_nogu_s_port(tmp_path):
    p = make_pisatel(tmp_path)
    feed(p, 7)
    zaem_v2.zakryt(p)
    p.close()
    for i in range(7):
        ref = {"potok": "p00", "shard": i // 3, "stroka": i % 3}
        assert (zaem_v2.zaem_ryad(tmp_path, "S1", ref) == 100.0 + i).all()
        row = json.loads((p.pd / "rows.jsonl").read_text().splitlines()[i])
        assert (zaem_v2.zaem_ryad(tmp_path, "S1", row) == 100.0 + i).all()
    assert sorted(f.name for f in (p.pd / "zaem").iterdir()) == sorted(f.name for f in (p.pd / "port").iterdir())
    sums = (p.d / zaem_v2.SUMS).read_text().splitlines()
    assert len(sums) == 3 and all(
        s.split("  ")[0] == hashlib.sha256((tmp_path / "S1" / s.split("  ")[1]).read_bytes()).hexdigest() for s in sums)


def test_port_pobaytno_ne_menyaetsya(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    pa, pb = make_pisatel(a), make_pisatel(b)
    feed(pa, 7)
    pa.close()
    for i in range(7):
        pb.add({"status": "ok", "klyuch_sha": f"sha{i}"}, np.full(5, float(i)), np.full((17, 5), np.nan))
    pb.close()
    for sub in ("port", "tik"):
        for f in (pb.pd / sub).iterdir():
            assert f.read_bytes() == (pa.pd / sub / f.name).read_bytes()


def test_ne_ok_i_otsutstvie_ryada_dayut_nan(tmp_path):
    p = make_pisatel(tmp_path)
    row = {"status": "oshibka", "klyuch_sha": "x"}
    zaem_v2.pisat(p, row, np.zeros(5), np.zeros((17, 5)), None)
    feed(p, 1)
    zaem_v2.zakryt(p)
    p.close()
    z = np.load(p.pd / "zaem" / "00000.npy")
    assert np.isnan(z[0]).all() and (z[1] == 100.0).all()


def test_obryv_udalyaet_zaem_bez_port(tmp_path):
    p = make_pisatel(tmp_path)
    feed(p, 3)                                                                          # шард 0 закрыт
    zaem_v2.pisat(p, {"status": "ok", "klyuch_sha": "z"}, np.zeros(5), np.zeros((17, 5)), np.zeros(5))
    zaem_v2.zakryt(p)                                                                   # шард 1 zaem есть, port ещё нет
    assert (p.pd / "zaem" / "00001.npy").exists() and not (p.pd / "port" / "00001.npy").exists()
    q = make_pisatel(tmp_path)
    zaem_v2.ZaemPisatel(q)
    assert [f.name for f in (q.pd / "zaem").iterdir()] == ["00000.npy"]
