"""П-6 (проверяющий), Л-18: плата за заём шорта (`fastbt/zaem_v2.py`, Ж-088 п. 5, Ж-089). Независимо от tests/test_zaem_v2.py:
сделки собираются руками (Trade), ответ считается в тесте от календаря, а не функцией автора.
Три случая владельца: шорт в один день = 0; пятница -> понедельник = 3 дня; лонг = 0. Плюс: многосуточное удержание, вес,
вход лимитом на d+1, сумма по портфелю на данных <= 2019 при razmer != ravnyy (независимый пересчёт от календарных дат).
Мутации: торговые дни; внутридневной шорт платит; лонг платит; знак; плата с d+1 входа; стоимость не по весу -> test_mutacii_*.
python3 -m pytest tests/proverka_progon/test_p6_zaem.py -q --tb=short (FASTBT_BARS1M=…/data/regen/D25/bars1m)"""
import numpy as np
import pytest

from fastbt import zaem_v2
from fastbt.exec_e_types import Trade

# чт 1.03, пт 2.03, пн 5.03, вт 6.03, ср 7.03
DATES = np.array(["2018-03-01", "2018-03-02", "2018-03-05", "2018-03-06", "2018-03-07"], dtype="datetime64[D]")
N = len(DATES)


def tr(side, entry, exit_, d_sig=None):
    return Trade("T", entry if d_sig is None else d_sig, entry, 0, 100.0, exit_day=exit_, exit_bar=1, exit_px=100.0,
                 reason="signal" if exit_ is not None else None, side=side)


def baza(trades, w=None, a=0, b=N):
    return zaem_v2.baza_tikera(trades, DATES[:b + 1], np.ones(N) if w is None else np.asarray(w, float), a, b)


def test_shorty_v_odin_den_nol():
    assert baza([tr(-1, 1, 1)]).sum() == 0.0 and baza([tr(-1, 2, 2)]).sum() == 0.0


def test_pyatnitsa_ponedelnik_tri_dnya():
    z = baza([tr(-1, 1, 2)])
    assert z.sum() == pytest.approx(3 / 365)
    assert z[1] == pytest.approx(3 / 365) and np.count_nonzero(z) == 1          # ночь после входа (строка 1), не d+1


def test_lonng_nol():
    assert baza([tr(1, 1, 2), tr(1, 0, 4)]).sum() == 0.0


def test_neskolko_nochey_kalendar():                                           # пт -> вт: 3 + 1 = 4 календарных (не 2 торговых)
    assert baza([tr(-1, 1, 3)]).sum() == pytest.approx(4 / 365)


def test_ves_sdelki_po_dnyu_signala():
    z = baza([tr(-1, 1, 2, d_sig=0)], w=[0.5, 9.0, 9.0, 9.0, 9.0])              # вес берётся по d_sig, не по дню входа
    assert z.sum() == pytest.approx(0.5 * 3 / 365)


def test_otkrytaya_do_konca_okna():
    assert baza([tr(-1, 1, None)], b=4).sum() == pytest.approx((3 + 1 + 1) / 365)   # ночи 1->2, 2->3, 3->4 до дня b = 4


def test_summa_neskolkih_sdelok_i_znak():
    z = baza([tr(-1, 0, 1), tr(-1, 1, 2), tr(1, 0, 3)])
    assert z.sum() == pytest.approx((1 + 3) / 365) and (z >= 0).all()           # плата положительная; excess − r·baza уменьшается
    assert zaem_v2.excess_s_platoy(np.zeros(N), z, 0.05).sum() < 0


# --- мутации: правим исходник baza_tikera, функция обязана упасть на check_vse ---
def check_vse():
    test_shorty_v_odin_den_nol()
    test_pyatnitsa_ponedelnik_tri_dnya()
    test_lonng_nol()
    test_neskolko_nochey_kalendar()
    test_ves_sdelki_po_dnyu_signala()
    test_summa_neskolkih_sdelok_i_znak()


MUT = {
    "torgovye_dni": ("gap[lo:hi] / DNEI", "1.0 / DNEI"),
    "vnutridnevnoj_platit": ("if hi > lo:", "hi = max(hi, lo + 1)\n        if hi > lo:"),
    "long_platit": ("if tr.side >= 0:", "if False:"),
    "znak": ("+= float(w[tr.d_sig]) * gap[lo:hi] / DNEI", "-= float(w[tr.d_sig]) * gap[lo:hi] / DNEI"),
    "s_d_plus_1": ("lo = max(tr.entry_day, a)", "lo = max(tr.entry_day + 1, a)"),
    "ne_po_vesu": ("float(w[tr.d_sig]) * gap", "1.0 * gap"),
    "ves_po_dnyu_vhoda": ("float(w[tr.d_sig]) * gap", "float(w[tr.entry_day]) * gap"),
}


@pytest.mark.parametrize("nazv", MUT)
def test_mutacii_lovyatsya(nazv, monkeypatch):
    import inspect
    import textwrap
    src = textwrap.dedent(inspect.getsource(zaem_v2.baza_tikera))
    old, new = MUT[nazv]
    assert src.count(old) == 1, nazv
    ns = {"np": np, "DNEI": zaem_v2.DNEI}
    exec(src.replace(old, new), ns)
    monkeypatch.setattr(zaem_v2, "baza_tikera", ns["baza_tikera"])
    with pytest.raises(AssertionError):
        check_vse()


# --- реальные данные <= 2019: портфельная сумма при razmer != ravnyy, независимый пересчёт ---
@pytest.fixture(scope="module")
def ev_s1():
    from fastbt import bars5m, cache, etap1
    from fastbt.etap1_eval_v2 import EvaluatorV2
    if not bars5m.SRC.exists() and (cache.MAIN / "data/regen/D25/bars1m").exists():
        bars5m.SRC = cache.MAIN / "data/regen/D25/bars1m"
    return EvaluatorV2("S1", etap1.build_inputs("S1", period="train4"), etap1.base_point("S1"), 0.0004)


@pytest.mark.parametrize("razmer", ["ravnyy", 20])
def test_dannye_summa_bazy_ravna_summe_po_sdelkam(ev_s1, razmer):
    from fastbt import rezhim_e1
    pt = {"vhod": 0.5, "proverka": "bars5m", "razmer": razmer, "storona": "obe"}
    row = ev_s1.evaluate_v2(pt, "train4")[0]
    assert row["status"] == "ok"
    df, inv, eng, cnt, rs = ev_s1._last
    ozh, got = 0.0, 0.0
    for ti, r in enumerate(rs):
        t = ev_s1.sk.tickers[ti]
        m = ev_s1.inp.meta[t]
        dates = m["dates"]
        w = np.ones(len(dates))
        if razmer != "ravnyy":
            w = rezhim_e1.razmer_ves(ev_s1.inp.sigma["rv22"][t][:len(dates)], "obratnyj", razmer / 100.0)
        a = int(np.searchsorted(dates, np.datetime64("2016-01-01")))
        for x in r.trades:                                                       # только сделки целиком внутри окна
            if x.side < 0 and x.exit_day is not None and x.entry_day >= a and x.exit_day > x.entry_day:
                dni = int((dates[x.exit_day].astype("datetime64[D]") - dates[x.entry_day].astype("datetime64[D]")).astype(int))
                ozh += float(w[x.d_sig]) * dni / 365.0                           # в весах сделок
    zr = ev_s1.zaem                                                              # по датам портфеля, уже поделено на cnt
    assert zr is not None and np.isfinite(zr).all() and (zr >= 0).all() and zr.sum() > 0
    # Σ zr·cnt == Σ по сделкам; сделки через границу окна (прогрев/хвост) в ожидании не учтены — допуск 3 %
    keep = np.isin(df.index.year, [2016, 2017, 2018, 2019])
    got = float((zr * eng.cnt[keep]).sum())
    assert got == pytest.approx(ozh, rel=0.03)
    # единицы веса те же, что у прежней оценки (шорт-экспозиция × календарные дни): новая плата не больше старой — старая берёт и долю дня
    sh = np.bincount(inv, np.concatenate([r.short for r in rs]), len(eng.dates)) / eng.cnt
    staraya = float((sh * eng.cal / 365.0)[keep].sum())
    assert 0.3 * staraya < float(zr.sum()) <= staraya * (1 + 1e-9)
