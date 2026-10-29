"""П5b (Ж-111, Ж-125 п. 3): внутридневное исполнение tf ∈ {10m…2h} поверх автомата v2 — fastbt/tf_ispolnenie_v5.py.
Только синтетика с известным ответом, без сети и данных.

Мутации: исполнение на баре сигнала (Д2; mutations.md № 1), касание лимита засчитано (№ 2), тейк раньше стопа (№ 3),
гэп исполнен по уровню (№ 4), нет комиссии выхода в конце сессии (№ 5), перенос через ночь.
"""
import numpy as np
import pandas as pd
import pytest

from fastbt import bars_agg as ba
from fastbt import tf_ispolnenie_v5 as X
from fastbt import tf_os_v5 as T
from fastbt.exec_e_types import Params, Trade

COMM = 0.0004
P0 = Params(k=1.0, b=2.0, a=2.0, vyhod_lim=True, stop_touch=True, comm=COMM)   # σ = 0,01: лимит 99, стоп 97,02, тейк 100,98
SG = 0.01


def bars(rows):
    """rows: [(o, h, l, c)] → массивы o, h, l, c и σ."""
    a = np.array(rows, float)
    return a[:, 0], a[:, 1], a[:, 2], a[:, 3], np.full(len(a), SG)


def run(rows, sost, p=P0):
    o, h, l, c, sg = bars(rows)
    return X.sessiya("SYN", o, h, l, c, np.array(sost), sg, p)


FLAT = (100.0, 100.2, 99.8, 100.0)
# бар 0: сигнал на close 100 → лимит 99 на бар 1; бар 1 пробивает лимит до 98,9, закрывается 99,5
VHOD = [FLAT, (100.0, 100.5, 98.9, 99.5)]


def test_limit_ispolnen_probojem_i_tejk_po_urovnyu():
    x = run(VHOD + [(99.5, 101.5, 99.2, 100.5), FLAT, FLAT], [1, 1, 0, 0, 0])
    t = x.trades[0]
    assert (t.entry_day, t.entry_px, t.exit_day, t.reason) == (1, 99.0, 2, "take")
    assert t.exit_px == pytest.approx(99.0 * 1.02)
    assert x.gross == pytest.approx(np.log(99.0 * 1.02 / 99.0))
    assert x.cost == pytest.approx(2 * COMM)


def test_kasanie_limita_ne_ispolnyaet_i_myutant_tick0_lovitsya():
    rows = [FLAT, (100.0, 100.5, 99.0, 99.5)] + [FLAT] * 3        # low = L ровно
    assert run(rows, [1, 1, 0, 0, 0]).trades == []
    x = run(rows, [1, 1, 0, 0, 0], Params(**{**P0.__dict__, "tick": 0.0}))   # мутант: касание = исполнение
    assert len(x.trades) == 1                                      # тест различает: мутант даёт сделку


def test_stop_i_tejk_v_odnom_bare_stop_pervyj_i_myutant_lovitsya(monkeypatch):
    rows = VHOD + [(99.5, 101.5, 96.5, 98.0), FLAT, FLAT]
    t = run(rows, [1, 1, 0, 0, 0]).trades[0]
    assert (t.reason, t.exit_day) == ("stop", 2) and t.exit_px == pytest.approx(99.0 * 0.98)
    import fastbt.exec_e_run as R
    monkeypatch.setattr(R, "stop_hit", lambda *a: (False, 0.0))  # мутант: стоп не проверяется до тейка
    assert run(rows, [1, 1, 0, 0, 0]).trades[0].reason == "take"


def test_gep_cherez_stop_ispolnen_po_open_a_ne_po_urovnyu():
    t = run(VHOD + [(95.0, 96.0, 94.0, 95.5), FLAT, FLAT], [1, 1, 0, 0, 0]).trades[0]
    assert t.reason == "stop" and t.exit_px == 95.0                # open < S = 97,02


def test_zakrytie_v_konce_sessii_platit_vyhod_i_myutant_bez_vyhoda_lovitsya(monkeypatch):
    rows = VHOD + [(99.5, 100.0, 99.2, 99.9), (99.9, 100.1, 99.5, 100.0), (100.0, 100.3, 99.7, 100.2)]
    x = run(rows, [1, 1, 1, 1, 1])
    t = x.trades[0]
    assert (t.reason, t.exit_day, t.exit_px) == ("sessiya", 4, 100.2)
    assert x.gross == pytest.approx(np.log(100.2 / 99.0)) and x.cost == pytest.approx(2 * COMM)
    from fastbt.exec_e_run import Run
    orig = Run._exit

    def bez_vyhoda(s, i, j, b, px, reason):
        orig(s, i, j, b, px, reason)
        if reason == "vremya":
            s.cost[i] -= s.p.comm
    monkeypatch.setattr(Run, "_exit", bez_vyhoda)
    assert run(rows, [1, 1, 1, 1, 1]).cost == pytest.approx(COMM)  # мутант платит одну сторону — assert выше его ловит


def test_vyhod_po_signalu_zayavkoj_na_sleduyushchem_bare():
    rows = VHOD + [(99.5, 99.9, 99.3, 99.8), (99.8, 100.9, 99.6, 100.5), FLAT]
    t = run(rows, [1, 1, 0, 0, 0]).trades[0]                      # сигнал ушёл на close бара 2, лимит 99,8·1,01 = 100,798
    assert (t.reason, t.exit_day) == ("lim", 3) and t.exit_bar == 0 and t.exit_px == pytest.approx(99.8 * 1.01)


def test_posledniy_bar_resheniya_ne_ispolnyaetsya_i_pervyj_bar_bez_pozicii():
    x = run([FLAT, FLAT, FLAT], [0, 0, 1])                         # сигнал на последнем баре — завтра, не сегодня
    assert x.trades == [] and x.n_orders == 0
    x = run([FLAT, (100.0, 100.5, 98.0, 99.0), FLAT], [1, 0, 0])   # решение на баре 0 исполнено на баре 1, не на баре 0
    assert x.trades[0].entry_day == 1 and x.trades[0].d_sig == 0


def test_perenos_cherez_noch_zapreshchen_i_myutant_lovitsya():
    d1 = VHOD + [FLAT, FLAT]                                       # держим до конца дня 1 (close 100)
    d2 = [(101.0, 101.5, 100.5, 101.2)] * 3
    a = run(d1, [1, 1, 1, 1]).gross + run(d2, [0, 0, 0]).gross
    assert a == pytest.approx(np.log(100.0 / 99.0))
    sklejeno = run(d1 + d2, [1, 1, 1, 1, 1, 1, 1])                 # мутант: сессии склеены, позиция живёт через ночь
    assert sklejeno.gross != pytest.approx(a)


def test_d2_zapreshcheny_vhod_po_close_i_vyhod_po_close_bara_signala():
    with pytest.raises(ValueError, match="Д2"):
        run(VHOD, [1, 1], Params(**{**P0.__dict__, "k": None}))
    with pytest.raises(ValueError, match="Д2"):
        run(VHOD, [1, 1], Params(**{**P0.__dict__, "vyhod_lim": False}))
    with pytest.raises(ValueError, match="Д2"):                    # мутант: сделка, вошедшая на баре сигнала
        X.proverka_d2([Trade("SYN", d_sig=3, entry_day=3, entry_bar=0, entry_px=100.0)])
    X.proverka_d2([Trade("SYN", d_sig=3, entry_day=4, entry_bar=0, entry_px=100.0)])


def test_sigma_bez_budushchego_i_bez_nochnyh_razryvov():
    n = 40
    rng = np.random.default_rng(1)
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.003, n)))
    c[20:] *= 1.5                                                  # ночной разрыв +50 %
    den = np.repeat([0, 1], 20)

    class D:
        close = c
    s = X.sigma_bar(D, den)
    assert np.nanmax(s) < 0.02                                     # разрыв не попал в σ
    d2 = D()
    d2.close = c.copy()
    d2.close[25:] *= 3.0
    assert np.array_equal(s[:25], X.sigma_bar(d2, den)[:25], equal_nan=True)   # σ(t) не зависит от баров > t


def make_td(days=40, seed=2):
    rng = np.random.default_rng(seed)
    n = days * ba.SLOTS
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.003, n)))
    o = np.r_[100.0, c[:-1]]
    h, l = np.maximum(o, c) * 1.001, np.minimum(o, c) * 0.999
    d = pd.bdate_range("2018-01-01", periods=days)
    b10 = pd.DataFrame(dict(date=np.repeat(d, ba.SLOTS), slot=np.tile(np.arange(ba.SLOTS), days),
                            open=o, high=h, low=l, close=c, volume=np.full(n, 1e3)))
    return T.ticker_data(T.bary(b10, "1h"), "SYN")


def test_progon_stroka_izderzhek_i_tf_den_ne_zdes():
    td = make_td()
    grid = [dict(okno=10, podtv="none"), dict(okno=5, podtv="none")]
    rows = X.progon("S3", "1h", {"SYN": td}, grid, "2018-01-01", "2018-03-01")
    assert len(rows) == 2 and all(r["tf"] == "1h" for r in rows)
    for r in rows:
        assert r["netto_pct_god"] == pytest.approx(r["brutto_pct_god"] - r["izderzhki_pct_god"])
        assert r["izderzhki_pct_god"] >= 0 and r["zayavok"] >= 0
        if r["sdelok"]:                                            # каждая сделка платит две стороны
            assert r["izderzhki_pct_god"] > 0
    with pytest.raises(ValueError):
        X.progon("S3", "den", {"SYN": td}, grid, "2018-01-01", "2018-03-01")
