"""T-CE (тестировщик, ДО КОДА): C/D/E поверх A_strogaya, строгое исполнение. Ж-088 пп. 2, 4; Ж-089; Ж-086.
Источник контракта: docs/otchety/2026-10-28_plan_ce_train4.md (ветка origin/arh-ce-prereg), разделы 3-5, листы CE-0…CE-9.
Тесты красные, пока нет fastbt/strogoe.py и др.; skip/xfail нет. Данные только ≤ 2019.

Допущения (в плане сигнатуры нет, взяты по смыслу; кодер сверяет, расхождение — вопрос, а не правка теста):
 A1 strogoe.port_strogiy(eng, pos_row, comm, zaem=0.05, nogi=(n_row, dn_row)) -> DataFrame с колонками _finish + zaem, noch;
    zaem[t] = s·max(0, −pos[t−1])·cal[t]/365 (деление на cnt, как в Engine); noch[t] = |pos[t−1]|; стороны |pos[t]−pos[t−1]|;
    последняя строка окна: pos[T] не исполняется, ночная нога = pos[T−1], комиссия выхода |pos[T−1]|.
 A2 strogoe.patched(eng, nogi, zaem=0.05): Engine.port(pos) → port_strogiy; port(pos, wgt=...) → ValueError.
 A3 strogoe.ISPOLNENIE, strogoe.V == "1.0"; strogoe.ispolnenie_tochki(semya, tochka: dict) -> dict (поле `ispolnenie` строки).
 A4 ledger_ce.osnovnaya_ce(ispolnenie) -> bool; без ispolnenie — ValueError; Chitatel_ce().razobrat(row: dict) -> row, иначе ValueError.
 A5 sem_e_strogiy.cena_vhoda(days, d, tochka) и cena_vyhoda(days, x, tochka, prichina) — цена входа/выхода E (days — exec_e.Day).
 A6 perescet_ce_aprim.perescet(zamorozhennye: list, aprim, otbor) -> список точек; otbor — вызываемый, вызывать нельзя.
 A7 «внутридневной шорт»: в дневной строке шорт, держимый в ночь, = pos[t−1]; решение pos[t] = −1 в день t платы в строке t не даёт.
Правка 28.10 (#484): время бара в load_bars5m лежит в `ts`, `date` — только день (ошибка теста, автор кода был прав); тест на данных
требует FASTBT_BARS1M=<data/regen/D25/bars1m>. test_konec_okna усилен: ненулевые ноги последней строки и три знака pos[T], иначе
мутация «конец окна по close» не ловилась.
Мутации (каждая роняет тест из скобок): М-CE1 весь lr на pos[t] (test_znanie_otvet, test_close_v_osnovnoy_*); М-CE2 patched не подменяет
port (test_patched_*); М-CE3 E вход по close d (test_e_vhod_*); М-CE4 τ по close x (test_e_vyhod_*); М-CE5 к2 старым ядром (test_k2_*);
М-CE6 конец окна по close (test_konec_okna_*); М-CE7 плата по барам (test_zaem_*); ночь у новой позиции (test_znanie_otvet);
σ̂ d+1 в затворе C (test_zatvor_c_*); пересчёт с новым отбором (test_perescet_*); версия ≠ 1.0 молча (test_chitatel_*)."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from fastbt import s9
from fastbt.s9 import Engine

C = 0.0004
SW = Path(__file__).resolve().parents[2] / "fastbt" / "designs" / "sweep"
DATES = pd.bdate_range("2019-01-02", periods=8)       # ср 02.01 … чт 10.01; пт 04.01 → пн 07.01: 3 календарных дня
CL = np.array([100, 102, 101, 104, 103, 105, 107, 106.0])


def eng(n=8, rf=0.0):
    e = Engine.__new__(Engine)
    e.r = {"date": np.arange(n), "tick": np.zeros(n, int), "bar": np.arange(n)}
    e.first = np.r_[True, np.zeros(n - 1, bool)]
    e.dates = DATES[:n]
    e.cnt, e.kr = np.ones(n), np.full(n, rf)
    e.rf, e.cal = np.full(n, rf / 365.0), np.ones(n)
    e.cal[:] = np.r_[np.diff(e.dates.values).astype("timedelta64[D]").astype(float), 1.0]
    e.lr_row = np.r_[np.log(CL[1:n] / CL[:n - 1]), 0.0]
    return e


def nogi_true(n=8):
    """n[t] = ln(o10[t+1]/c[t]), dn[t] = ln(c[t+1]/o10[t+1]); o10[t+1] = c[t]·(1 + 0,01·(−1)^t) — open ≠ close."""
    o = CL[:n - 1] * (1 + 0.01 * (-1.0) ** np.arange(n - 1))
    return np.r_[np.log(o / CL[:n - 1]), 0.0], np.r_[np.log(CL[1:n] / o), 0.0]


def expect_g(pos, nr, dr):
    prev = np.r_[0.0, pos[:-1]]
    g = prev * nr + pos * dr
    sides = np.abs(pos - prev)
    g[-1], sides[-1] = prev[-1] * nr[-1], abs(prev[-1])          # граница окна: pos[T] не исполняется (A1)
    return g, sides


def test_znanie_otvet_dve_nogi():                                # М-CE1, ночь у новой позиции
    from fastbt import strogoe
    e, (nr, dr) = eng(), nogi_true()
    pos = np.array([0, 1, 1, 0, -1, -1, 0, 0.0])
    p = strogoe.port_strogiy(e, pos, C, zaem=0.0, nogi=(nr, dr))
    g, sides = expect_g(pos, nr, dr)
    assert np.allclose(p["gross"].values, g, atol=1e-15)
    assert np.allclose(p["r"].values, g - sides * C, atol=1e-15)
    assert not np.allclose(p["gross"].values, pos * e.lr_row), "close-путь: строка равна pos[t]·lr[t]"


def test_toghdestvo_o10_ravno_c():                               # якорь s9 (CE-1): o10 ≡ c → Engine.port побайтно
    from fastbt import strogoe
    e = eng()
    pos = np.array([0, 1, 1, 0, 1, 1, 0, 0.0])
    nr, dr = np.zeros(8), e.lr_row.copy()
    a, b = e.port(pos, C), strogoe.port_strogiy(e, pos, C, zaem=0.0, nogi=(nr, dr))
    for col in ("r", "gross", "turnover", "total"):
        assert np.array_equal(a[col].values, b[col].values), col


def test_konec_okna_zakryvaet_po_o10():                          # М-CE6
    from fastbt import strogoe
    e, (nr, dr) = eng(), nogi_true()
    nr, dr = nr.copy(), dr.copy()
    nr[-1], dr[-1] = 0.004, 0.007                                 # ненулевые ноги последней строки (иначе мутация не видна)
    for last in (1.0, 0.0, -1.0):                                 # решение close T любого знака не исполняется
        pos = np.array([0, 0, 0, 0, 0, 0, 1, last])
        p = strogoe.port_strogiy(e, pos, C, zaem=0.0, nogi=(nr, dr))
        assert p["gross"].values[-1] == pytest.approx(nr[-1]), "ночь прежней позиции; решение close T не исполняется"
        assert p["turnover"].values[-1] == pytest.approx(1.0), "комиссия выхода |pos[T−1]| на последней строке"
        assert p["r"].values[-1] == pytest.approx(nr[-1] - C)


def test_zaem_tri_sluchaya_zh089():                              # М-CE7; пт→пн = 3 дня, шорт 1 день = 1, лонг = 0
    from fastbt import strogoe
    e, (nr, dr) = eng(), nogi_true()
    pos = np.array([0, 0, -1, 0, 0, 0, 0, 0.0])                  # решение шорт на close пт (строка 2), выход утром следующего дня
    z = strogoe.port_strogiy(e, pos, C, zaem=0.05, nogi=(nr, dr))["zaem"].values
    assert z[2] == 0.0, "A7: в день решения шорт на 18:50 не держится — плата 0 (внутридневной шорт = 0)"
    assert z[3] == pytest.approx(0.05 * 1 * e.cal[3] / 365) and e.cal[3] == 1.0, "пн→вт: 1 календарный день"
    pos = np.array([0, 1, 1, 0, 0, 0, 0, 0.0])
    assert not strogoe.port_strogiy(e, pos, C, zaem=0.05, nogi=(nr, dr))["zaem"].any(), "лонг платит 0"


def test_zaem_pyatnitsa_ponedelnik_3_dnya():
    from fastbt import strogoe
    e, (nr, dr) = eng(), nogi_true()
    assert e.cal[2] == 3.0                                       # пятница 04.01 → понедельник 07.01
    pos = np.array([0, -1, 0, 0, 0, 0, 0, 0.0])                  # pos[1] — чт; ночь чт→пт: cal[1] = 1, а не пт: строка 2 держит pos[1]
    z = strogoe.port_strogiy(e, pos, C, zaem=0.05, nogi=(nr, dr))["zaem"].values
    assert z[2] == pytest.approx(0.05 * 3 / 365), "шорт, принятый чт и удержанный через пт→пн: строка пт платит 3 дня"
    assert z[1] == 0.0 and z[3] == 0.0


def test_noch_zanyatost():
    from fastbt import strogoe
    e, (nr, dr) = eng(), nogi_true()
    pos = np.array([0, 1, -1, 0, 0, 1, 0, 0.0])
    p = strogoe.port_strogiy(e, pos, C, zaem=0.05, nogi=(nr, dr))
    assert list(p["noch"].values) == list(np.abs(np.r_[0.0, pos[:-1]]))


def test_patched_podmenyaet_port_i_otkazyvaet_na_wgt():          # М-CE2
    from fastbt import strogoe
    e, nog = eng(), nogi_true()
    pos = np.array([0, 1, 1, 0, 0, 0, 0, 0.0])
    with strogoe.patched(e, nog, zaem=0.05):
        inside = e.port(pos, C)
        with pytest.raises(ValueError):
            e.port(pos, C, wgt=np.ones(8))
    assert not np.allclose(inside["gross"].values, pos * e.lr_row), "под patched port не считает по close"
    assert np.allclose(inside["gross"].values, strogoe.port_strogiy(e, pos, C, 0.05, nog)["gross"].values)
    assert np.allclose(e.port(pos, C)["gross"].values, pos * e.lr_row), "вне patched вернулся старый port"


def test_ispolnenie_versiya_i_formy():
    from fastbt import strogoe
    assert strogoe.V == "1.0" and strogoe.ISPOLNENIE["v"] == "1.0"
    assert strogoe.ISPOLNENIE["vhod"] == "open1000_d1" and strogoe.ISPOLNENIE["vyhod"] == "open1000_x1"


def test_close_v_osnovnoy_stroke_krasnyy():                      # Ж-088 п. 4, статический: все точки сеток × комиссии
    from fastbt import ledger_ce, strogoe
    pts = []
    c = yaml.safe_load((SW / "C_setka_e3.yaml").read_text(encoding="utf-8"))
    e = yaml.safe_load((SW / "E_setka_e2.yaml").read_text(encoding="utf-8"))
    for sem in ("C", "D"):
        pts += [(sem, dict(family=f"{sem}{i}", W=w, rung=r)) for i in (1, 2) for w in (126, 252, 504) for r in c["lestnitsa"]]
    for f in e["semyi"]["E1"]["osi"]["f"]:
        pts += [("E", dict(family="E1", f=f, q=q, k=k)) for q in (0.0, 0.5) for k in (0.25, 1.0)]
    pts += [("E", dict(family=f, q=0.0)) for f in ("E0", "E2")]
    assert len(pts) > 60
    for sem, pt in pts:
        isp = strogoe.ispolnenie_tochki(sem, pt)
        main = ledger_ce.osnovnaya_ce(isp)
        assert main == ("close" not in str(isp.values())), (sem, pt, isp)
        if pt.get("f") == "zakrytie_d1":
            assert main is False, "E1 f=zakrytie_d1 — справочная строка"
        elif sem in "CD" or pt["family"] in ("E0", "E2"):
            assert main is True and "close" not in str(isp), (sem, pt)


def test_osnovnaya_zavisit_ot_parametrov():
    from fastbt import ledger_ce
    ok = {"vhod": "open1000_d1", "vyhod": "open1000_x1", "v": "1.0"}
    assert ledger_ce.osnovnaya_ce(ok) is True
    for k, v in (("vyhod", "close_x"), ("vhod", "close_d")):
        assert ledger_ce.osnovnaya_ce(dict(ok, **{k: v})) is False
    with pytest.raises(ValueError):
        ledger_ce.osnovnaya_ce({})


def test_chitatel_otkazyvaet():                                  # close в основной; версия ≠ 1.0; нет ispolnenie
    from fastbt import ledger_ce
    row = {"v": "1.0", "semya": "C", "osnovnaya": True, "period": "train_ce",
           "klyuch": {"ispolnenie": {"vhod": "open1000_d1", "vyhod": "open1000_x1", "v": "1.0"}}}
    r = ledger_ce.Chitatel_ce()
    assert r.razobrat(row)
    bad = [dict(row, klyuch={"ispolnenie": {"vhod": "open1000_d1", "vyhod": "close_x", "v": "1.0"}}),
           dict(row, klyuch={"ispolnenie": dict(row["klyuch"]["ispolnenie"], v="1.1")}),
           dict(row, klyuch={}), dict(row, v="2.0")]
    for b in bad:
        with pytest.raises(ValueError):
            r.razobrat(b)


def test_zatvor_c_ne_vidit_sigmu_d_plus_1():                     # σ̂ дня d+1 в затворе C
    rng = np.random.default_rng(3)
    S = rng.uniform(0.01, 0.03, (1, 400))
    spec = {"min_periods": {126: 50}, "semi": {"C1": {"signal": "sigma", "okna": [126], "filtr": [20]}}}
    base = s9.c_dop_gates(S, None, [400], spec)[0][1][0][1]
    for d in (200, 300):
        S2 = S.copy()
        S2[0, d + 1:] *= 5.0
        G = s9.c_dop_gates(S2, None, [400], spec)[0][1][0][1]
        assert np.array_equal(G[:, :, :d + 1], base[:, :, :d + 1]), "G[d] зависит от σ̂ после d"
    S3 = S.copy()
    S3[0, 250] = 1.0                                              # σ̂ на закрытии d (прогноз на d+1) разрешён; меняется только G[d…]
    G3 = s9.c_dop_gates(S3, None, [400], spec)[0][1][0][1]
    assert np.array_equal(G3[:, :, :250], base[:, :, :250])


def test_e_vhod_po_open_1000_d_plus_1():                         # М-CE3: E0/E2 вход rynok по open 10:00 d+1, не по close d
    from fastbt import sem_e_strogiy
    from fastbt.exec_e import Day
    def day(p):
        t = np.arange(590, 590 + 5 * 12, 5)
        c = p + 0.01 * np.arange(12)
        return Day(c - 0.5, c, c - 0.5, c, False, t=t)
    days = [day(100.0), day(103.0)]
    for fam in ("E0", "E2"):
        px = sem_e_strogiy.cena_vhoda(days, 0, dict(family=fam, q=0.0))
        assert px == pytest.approx(days[1].o[2]), (fam, "бар 10:00 = индекс 2 при старте 09:50")
        assert px != pytest.approx(days[0].c[-1])


def test_e_vyhod_tau_po_close_dnya_x():                          # М-CE4; К-wf5b, Ж-137 п. 6: τ — close дня истечения
    from fastbt import sem_e_strogiy
    from fastbt.exec_e import Day
    t = np.arange(590, 590 + 5 * 12, 5)
    mk = lambda p: Day(p + 0.0 * t - 0.5, p + 0.01 * np.arange(12), p + 0.0 * t - 0.5, p + 0.01 * np.arange(12), False, t=t)
    days = [mk(100.0), mk(101.0), mk(102.0)]
    px = sem_e_strogiy.cena_vyhoda(days, 1, dict(family="E2", tau=1), "tau")
    assert px == pytest.approx(days[1].c[-1]) and px != pytest.approx(days[2].o[2])


def test_k2_strogiy_lag0_raven_iskhodnoy():                      # М-CE5: к2 строгим ядром; лаг 0 = исходная точка (mutations № 9)
    from fastbt import fast_kontrol_strogiy as fk
    e, nog = eng(), nogi_true()
    pos = np.array([0, 1, 1, 0, -1, -1, 0, 0.0])
    g = np.ones(8)
    assert hasattr(fk, "shift_strogiy"), "ядро сдвига с двумя ногами (n, dn)"
    a = fk.shift_strogiy(e, pos, g, 0, C, nog)
    from fastbt import strogoe
    b = strogoe.port_strogiy(e, pos * g, C, nogi=nog)
    assert np.allclose(a["excess"].values, b["excess"].values, atol=1e-12)
    c = fk.shift_strogiy(e, pos, np.r_[0.0, 1, 1, 1, 0, 1, 1, 1], 1, C, nog)
    assert not np.allclose(c["gross"].values, a["gross"].values), "сдвиг g на 1 день меняет ряд"


def test_perescet_ne_vyzyvaet_otbor():                           # пересчёт поверх A′ берёт только замороженные точки
    from fastbt import perescet_ce_aprim as pc
    frozen = [dict(family="C1", W=126, label="W126_p20"), dict(family="D2", W=252, label="W252_p10_20")]
    def otbor(*a, **k):
        raise AssertionError("отбор вызван при пересчёте")
    out = pc.perescet(frozen, aprim=None, otbor=otbor)
    assert [dict((k, o[k]) for k in ("family", "W", "label")) for o in out] == frozen
    other = [dict(frozen[0], W=252)] + frozen[1:]
    assert [o["W"] for o in pc.perescet(other, aprim=None, otbor=otbor)] != [o["W"] for o in out]


def test_dannye_do_2020_o10_otlichaetsya_ot_close():             # реальные бары ≤ 2019 (SBER, неделя 03–07.06.2019)
    from fastbt import strogoe
    from fastbt.bars5m import load_bars5m
    from fastbt.exec_e_vremya import bar_at
    from fastbt.exec_e import Day
    b = load_bars5m("SBER", "2019-06-03", "2019-06-07", "s_predotkrytiem")
    b = b[~b["pusto"]]
    assert pd.to_datetime(b["date"]).max() < pd.Timestamp("2020-01-01")
    days = []
    for _, g in b.groupby(pd.to_datetime(b["date"]).dt.normalize()):
        ts = pd.to_datetime(g["ts"])                              # время бара — в `ts`; `date` — только день (ошибка теста, 28.10)
        t = (ts.dt.hour * 60 + ts.dt.minute).to_numpy()
        days.append(Day(g["open"].to_numpy(), g["high"].to_numpy(), g["low"].to_numpy(), g["close"].to_numpy(), False, t=t))
    assert len(days) == 5
    c = np.array([d.c[-1] for d in days])
    o = np.array([d.o[bar_at(d, 600)] for d in days])
    assert np.abs(o[1:] / c[:-1] - 1).max() > 0 and not np.allclose(o[1:], c[:-1])
    e = eng(5)
    e.lr_row = np.r_[np.log(c[1:] / c[:-1]), 0.0]
    nr, dr = np.r_[np.log(o[1:] / c[:-1]), 0.0], np.r_[np.log(c[1:] / o[1:]), 0.0]
    pos = np.array([0, 1, 1, 0, 0.0])
    p = strogoe.port_strogiy(e, pos, C, zaem=0.0, nogi=(nr, dr))
    g, _ = expect_g(pos, nr, dr)
    assert np.allclose(p["gross"].values, g, atol=1e-15)
    assert not np.allclose(p["gross"].values, pos * e.lr_row)
