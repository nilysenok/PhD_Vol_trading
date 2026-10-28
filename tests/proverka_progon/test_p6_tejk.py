"""П-6 (проверяющий), лист Л-13: ось `tejk_signal` (Ж-084 п. 1, толкование [Ж-084] п. 1). Пишется ДО кода, красные до Л-13.
Источники: docs/resheniya/resheniya.md Ж-084, tolkovaniya.md [Ж-084], docs/otchety/2026-10-28_progon_etap1.md §1.1–1.3.

ДОПУЩЕНИЯ (в контракте сигнатур нет; поправить здесь, если Л-13 назвал иначе):
 А1. `fastbt.tejk_signal.uroven_M(sid, sig_params, dannye_do_d)`: dannye_do_d — dict ndarray [ticker, d]
     ключи close/high/low/volume + "dates" (datetime64[D], [d]); M — ndarray [ticker, d], столбец d использует только столбцы <= d.
     Параметры сигнала — имена из kub_S*.yaml (ma_window, bb_window, pivot_period, vwap_window ...).
 А2. `fastbt.tejk_signal.uroven_tejka(tejk, sd, p_vh, M) -> float | None`: уровень T без округления по тику; None — тейка нет.
 А3. Исполнение: `Params.tejk: str = "net"` (net|k_srednemu|pol_puti); `run(..., tejk_M=ndarray[len(days)])` — M дня сигнала d_sig;
     `Result.n_bez_tejka` — число сделок, где M не в сторону прибыли от цены входа.
 А4. `EvaluatorV2` берёт ось из `pt["tejk_signal"]` (как остальные NOVYE); `canon_v2(sid, pt)` схлопывает R и k_x.
 А5. P_vh = фактическая цена входа сделки (Trade.entry_px), не уровень лимита.
 А6. Куб: ключ `tejk_signal` где угодно в YAML (рекурсивно), setka = [net, k_srednemu, pol_puti].
Для S5/S6 формулы P и VWAP_w в контракте не расписаны — проверяется только «не смотрит вперёд» (мутация 1), для S1/S2 — ещё значение SMA.
Мутации (каждая обязана ронять): 1 -> test_M_*; 2 -> test_take_prohod_na_tik; 3 -> test_stop_pervym; 4 -> test_pol_puti_short*, test_uroven_tejka_znaki;
5 -> test_M_za_cenoj_net_tejka, test_uroven_tejka_znaki; 6 -> test_canon_R_skhlopnut; 7 -> test_canon_kx_skhlopnut; 8 -> test_tejk_ne_v_bare_vkhoda;
9 -> test_perenos_cherez_noch; 10 -> test_gep_cherez_T_po_open; 11 -> test_os_tolko_v_S1_S2_S5_S6.
Дыры по k_x: (а) test_kx_dohodit_do_Params_a, (б) test_kx_ne_yakor*, (в) test_kx_ne_v_bare_vkhoda, (г) test_kx_perenos_cherez_noch.
Данные <= 2019: FASTBT_BARS1M=/Users/nilysenok/Desktop/Старое/PhD_Vol_trading/data/regen/D25/bars1m
Запуск: python3 -m pytest tests/proverka_progon/test_p6_tejk.py -q --tb=short"""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from fastbt import bars5m, cache
from fastbt.exec_e import Params, run
from fastbt.exec_e_types import Day

_MAIN_BARS = cache.MAIN / "data" / "regen" / "D25" / "bars1m"
if not bars5m.SRC.exists() and _MAIN_BARS.exists():
    bars5m.SRC = _MAIN_BARS

TK = 0.01
KUB = Path("fastbt/designs/progon")
OSI = ["net", "k_srednemu", "pol_puti"]


def D(rows):
    a = np.array(rows, float).reshape(-1, 4)
    return Day(a[:, 0], a[:, 1], a[:, 2], a[:, 3])


def F(x=100.0, n=1):
    return [(x, x, x, x)] * n


def go(days, M=100.0, side=1, tejk="k_srednemu", **kw):
    """Все дни сигнал 1 (при side=-1 — шорт), σ̂ = 0,02, M одинаков на всех днях сигнала."""
    n = len(days)
    kw.setdefault("comm", 0.0)
    p = Params(side=side, tick=TK, check="bary", tejk=tejk, **kw)
    return run("T", days, np.ones(n), np.full(n, 0.02), p, tejk_M=np.full(n, float(M)))


def tr0(r):
    return r.trades[0]


# --- чистая функция уровня (А2) ---
def test_uroven_tejka_znaki():
    from fastbt.tejk_signal import uroven_tejka as u
    assert u("k_srednemu", 1, 100.0, 104.0) == pytest.approx(104.0)
    assert u("k_srednemu", -1, 100.0, 96.0) == pytest.approx(96.0)
    assert u("pol_puti", 1, 100.0, 104.0) == pytest.approx(102.0)
    assert u("pol_puti", -1, 100.0, 96.0) == pytest.approx(98.0)             # мутация 4: зеркало шорта (102 — ошибка)
    for t in ("k_srednemu", "pol_puti"):                                      # мутация 5: M «за» ценой входа
        assert u(t, 1, 100.0, 98.0) is None
        assert u(t, -1, 100.0, 102.0) is None


# --- исполнение: рынок на close d, вход 100 ---
def test_take_prohod_na_tik():                                                # мутация 2
    assert tr0(go([D(F()), D([(100, 104.0, 100, 100)]), D(F())], M=104.0)).reason is None   # касание — нет выхода
    t = tr0(go([D(F()), D([(100, 104.01, 100, 100)]), D(F())], M=104.0))                     # +1 тик — выход по T
    assert t.reason == "take" and t.exit_px == pytest.approx(104.0)


def test_stop_pervym():                                                       # мутация 3
    t = tr0(go([D(F()), D([(100, 105, 97, 100)]), D(F())], M=104.0, b=1.0, stop_touch=True))   # S = 98, T = 104 в одном баре
    assert t.reason == "stop" and t.exit_px == pytest.approx(98.0)


def test_pol_puti_long():
    t = tr0(go([D(F()), D([(100, 101.99, 100, 100), (100, 102.01, 100, 100)]), D(F())], M=104.0, tejk="pol_puti"))
    assert t.reason == "take" and t.exit_bar == 1 and t.exit_px == pytest.approx(102.0)


def test_pol_puti_short():                                                    # мутация 4
    t = tr0(go([D(F()), D([(100, 100, 98.01, 100), (100, 100, 97.99, 100)]), D(F())], M=96.0, side=-1, tejk="pol_puti"))
    assert t.side == -1 and t.reason == "take" and t.exit_bar == 1 and t.exit_px == pytest.approx(98.0)


def test_pol_puti_short_gep():
    t = tr0(go([D(F()), D([(97, 97.5, 96.5, 97)]), D(F())], M=96.0, side=-1, tejk="pol_puti"))      # open 97 < T = 98
    assert t.reason == "take" and t.exit_px == pytest.approx(97.0)


# --- лимитный вход: L = 99 (k = 0,5, σ̂ = 0,02) ---
LIM = dict(k=0.5, f="propusk")


def test_tejk_ne_v_bare_vkhoda():                                             # мутация 8
    days = [D(F()), D([(100, 102, 98.9, 100), (100, 101.5, 100, 100)]), D(F())]      # бар 0: вход 99 и high 102 > T = 101
    t = tr0(go(days, M=101.0, **LIM))
    assert t.entry_px == pytest.approx(99.0) and t.entry_bar == 0
    assert t.reason == "take" and t.exit_bar == 1 and t.exit_px == pytest.approx(101.0)


def test_perenos_cherez_noch():                                               # мутация 9
    days = [D(F()), D([(100, 100, 98.9, 100)]), D([(100, 101.5, 100, 100)]), D(F())]
    t = tr0(go(days, M=101.0, **LIM))
    assert t.exit_day == 2 and t.reason == "take" and t.exit_px == pytest.approx(101.0)


def test_gep_cherez_T_po_open():                                              # мутация 10
    days = [D(F()), D([(100, 100, 98.9, 100)]), D([(103, 104, 102.5, 103)]), D(F())]
    t = tr0(go(days, M=101.0, **LIM))
    assert t.exit_day == 2 and t.reason == "take" and t.exit_px == pytest.approx(103.0)


def test_pol_puti_ot_ceny_ispolneniya():                                      # А5: P_vh = факт входа 98 (гэп), не L = 99
    days = [D(F()), D([(98, 98.2, 97.9, 98), (98, 99.51, 98, 99)]), D(F())]       # T = 98 + 0,5·(101 − 98) = 99,5
    t = tr0(go(days, M=101.0, tejk="pol_puti", **LIM))
    assert t.entry_px == pytest.approx(98.0)
    assert t.reason == "take" and t.exit_bar == 1 and t.exit_px == pytest.approx(99.5)


def test_M_za_cenoj_net_tejka():                                              # мутация 5: лимитка исполнилась за уровнем
    days = [D(F()), D([(100, 100, 98.9, 100), (100, 110, 100, 100)]), D(F())]
    r = go(days, M=98.5, **LIM)                                               # вход 99 > M: тейка нет
    assert tr0(r).reason != "take" and r.n_bez_tejka == 1
    assert go(days, M=101.0, **LIM).n_bez_tejka == 0


def test_tejk_net_pobaytno_prezhnee():
    days = [D(F()), D([(100, 105, 98.9, 100)]), D([(100, 110, 100, 100)]), D(F())]
    a, b = go(days, M=101.0, tejk="net", **LIM), run("T", days, np.ones(4), np.full(4, 0.02), Params(side=1, tick=TK, check="bary", comm=0.0, **LIM))
    assert a.trades == b.trades
    for f in ("gross", "cost", "pos", "free"):
        assert np.array_equal(getattr(a, f), getattr(b, f))


# --- M: только данные <= d (А1) ---
def _dannye(n=60, nt=2, seed=3):
    r = np.random.default_rng(seed)
    c = 100 + np.cumsum(r.normal(0, 1, (nt, n)), axis=1)
    return {"close": c, "high": c + r.uniform(0.1, 1, (nt, n)), "low": c - r.uniform(0.1, 1, (nt, n)),
            "volume": r.uniform(1e3, 2e3, (nt, n)), "dates": np.datetime64("2018-01-01") + np.arange(n) * np.timedelta64(1, "D")}


def _poison(dn, d):
    out = {k: v.copy() for k, v in dn.items()}
    for k in ("close", "high", "low", "volume"):
        out[k][:, d + 1:] *= 7.0
    return out


SIG = {"S1": {"ma_window": 5, "z_entry": 2.0, "z_exit": 0.5}, "S2": {"bb_window": 5, "bb_std": 2.0},
       "S5": {"pivot_period": "weekly", "pivot_type": "classic", "filter_set": 3}, "S6": {"vwap_window": 5, "dev_mult": 1.5, "filter_set": 2, "dir_bias": "OFF"}}


@pytest.mark.parametrize("sid,w", [("S1", "ma_window"), ("S2", "bb_window")])
def test_M_sma_znachenie(sid, w):
    from fastbt.tejk_signal import uroven_M
    dn = _dannye()
    M = uroven_M(sid, SIG[sid], dn)
    for d in (4, 20, 59):
        assert np.allclose(M[:, d], dn["close"][:, d - 4:d + 1].mean(axis=1))      # SMA(w) включает close d, не d+1
    assert np.isnan(M[:, :4]).all()


@pytest.mark.parametrize("sid", ["S1", "S2", "S5", "S6"])
def test_M_ne_smotrit_vpered(sid):                                                # мутация 1
    from fastbt.tejk_signal import uroven_M
    dn, d = _dannye(), 30
    a, b = uroven_M(sid, SIG[sid], dn), uroven_M(sid, SIG[sid], _poison(dn, d))
    np.testing.assert_array_equal(a[:, :d + 1], b[:, :d + 1])
    cut = {k: v[..., :d + 1] for k, v in dn.items()}                              # данные ровно до d: тот же M[:, d]
    np.testing.assert_array_equal(a[:, d], uroven_M(sid, SIG[sid], cut)[:, d])


# --- куб (мутация 11) ---
def _find(o, key="tejk_signal"):
    if isinstance(o, dict):
        return [o[key]] if key in o else [x for v in o.values() for x in _find(v, key)]
    return [x for v in o for x in _find(v, key)] if isinstance(o, list) else []


@pytest.mark.parametrize("sid", ["S1", "S2", "S5", "S6"])
def test_os_v_kube(sid):
    f = _find(yaml.safe_load((KUB / f"kub_{sid}.yaml").read_text(encoding="utf-8")))
    assert len(f) == 1 and list(f[0]["setka"]) == OSI


@pytest.mark.parametrize("sid", ["S3", "S4"])
def test_os_tolko_v_S1_S2_S5_S6(sid):                                             # у S3/S4 оси нет совсем
    assert "tejk_signal" not in (KUB / f"kub_{sid}.yaml").read_text(encoding="utf-8")
    from fastbt.ledger_v2 import canon_v2
    assert "tejk_signal" not in canon_v2(sid, {"k_s": 2.0, "R": 2.0, "k_x": "net", "vhod": 0.5})


# --- canon_v2 (мутации 6, 7) ---
def _canon(**kw):
    from fastbt.ledger_v2 import canon_v2
    return canon_v2("S1", {"k_s": "net", "R": "net", "k_x": "net", "k_tr": "net", "bezubytok": "net", "vhod": 0.5, **kw})


@pytest.mark.parametrize("t", ["k_srednemu", "pol_puti"])
def test_canon_R_skhlopnut(t):
    assert _canon(k_s=2.0, R=3.0, tejk_signal=t)["R"] == "net"


@pytest.mark.parametrize("t", ["k_srednemu", "pol_puti"])
def test_canon_kx_skhlopnut(t):
    q = _canon(R=3.0, k_x=1.5, tejk_signal=t)                                     # k_s = net: прежнее правило k_x не схлопывает
    assert q["k_x"] == "net" and q["R"] == "net"


def test_canon_net_ne_trogaet():
    assert _canon(k_s=2.0, R=3.0, tejk_signal="net")["R"] == 3.0
    assert _canon(k_x=1.5, tejk_signal="net")["k_x"] == 1.5


def test_canon_klyuch_sha_odin():
    from fastbt.ledger_v2 import canon_v2, klyuch, klyuch_sha
    base = {"k_s": 2.0, "k_x": "net", "vhod": 0.5, "tejk_signal": "pol_puti"}
    s = lambda R: klyuch_sha(klyuch("S1", {**base, "R": R}, "train4", 0.0004, None), "k", "v")   # noqa: E731
    assert s(1.0) == s(4.0)                                                       # неразличимые точки — один ключ (в N один раз)
    base["tejk_signal"] = "net"
    assert s(1.0) != s(4.0)


# --- дыры по k_x (а)–(г) ---
def _ev():
    from fastbt.etap1_eval_v2 import EvaluatorV2
    ev = object.__new__(EvaluatorV2)
    ev.inp, ev.comm = SimpleNamespace(tick={"T": TK}), 0.0004
    return ev


def _nov(**kw):
    from fastbt.etap1_eval_v2 import NOVYE
    return {**NOVYE, **kw}


def _anchor_flag(ev, pt, nov):
    """Какой флаг anchor `EvaluatorV2.excess` отдаёт в _rows."""
    from fastbt.etap1_eval import YAKOR

    class Got(Exception):
        pass

    def rows(p, period, anchor):
        raise Got(anchor)
    ev._ctx, ev.full, ev._rows = dict(nov=nov, vozm={}, default=False), (lambda p: {**YAKOR, **p}), rows
    with pytest.raises(Got) as e:
        ev.excess(pt, "train4")
    return e.value.args[0]


def test_kx_dohodit_do_Params_a():                                               # (а): прямой вызов
    from fastbt.etap1_eval import YAKOR
    p = _ev()._params({**YAKOR}, "T", 5.0, False, _nov(k_x=1.0), {})
    assert p.a == 1.0 and p.b is None


def test_kx_ne_yakor():                                                          # (б) + (а) по цепочке excess -> _params
    from fastbt.etap1_eval import YAKOR
    ev, nov = _ev(), _nov(k_x=1.0)
    anchor = _anchor_flag(ev, {}, nov)
    assert anchor is False                                                       # k_x ≠ net не якорь: иначе ось теряется
    assert ev._params({**YAKOR}, "T", 5.0, anchor, nov, {}).a == 1.0             # тейк реально считается


def test_kx_net_ostayotsya_yakorem():                                            # охранный: якорь не сломан
    assert _anchor_flag(_ev(), {}, _nov()) is True


def test_kx_ne_v_bare_vkhoda():                                                  # (в): a = 1 -> T = 99·1,02 = 100,98
    days = [D(F()), D([(100, 102, 98.9, 100), (100, 101.5, 100, 100)]), D(F())]
    n = len(days)
    t = tr0(run("T", days, np.ones(n), np.full(n, 0.02), Params(k=0.5, a=1.0, tick=TK, check="bary", comm=0.0)))
    assert t.entry_bar == 0 and t.reason == "take" and t.exit_bar == 1 and t.exit_px == pytest.approx(100.98)


def test_kx_perenos_cherez_noch():                                               # (г)
    days = [D(F()), D([(100, 100, 98.9, 100)]), D([(103, 104, 102.5, 103)]), D(F())]
    n = len(days)
    t = tr0(run("T", days, np.ones(n), np.full(n, 0.02), Params(k=0.5, a=1.0, tick=TK, check="bary", comm=0.0)))
    assert t.exit_day == 2 and t.reason == "take" and t.exit_px == pytest.approx(103.0)


# --- данные <= 2019 (одна точка на ось): ось действует на реальной ленте ---
@pytest.fixture(scope="module")
def ev_s1():
    from fastbt import etap1
    from fastbt.etap1_eval_v2 import EvaluatorV2
    return EvaluatorV2("S1", etap1.build_inputs("S1", period="train4"), etap1.base_point("S1"), 0.0004)


def _x(ev, **pt):
    r = ev.evaluate_v2(pt, "train4")
    assert r[0]["status"] == "ok"
    return r[1]


def test_dannye_kx_menyaet_ryad(ev_s1):                                          # (а)+(б) на реальных данных
    assert not np.array_equal(_x(ev_s1), _x(ev_s1, k_x=1.0))


# Ж-087 п. 1: поля отчёта оси — не в ledger (формат v2.0 не меняется); строка при net = прежняя
POLYA_OTCHETA = {"dolya_tejk_y", "dolya_pribyl_y", "vyigrysh_sredn_y", "proigrysh_sredn_y", "top5_pnl_y", "n_bez_tejka"}


@pytest.mark.parametrize("t", ["net", "pol_puti"])
def test_ledger_stroka_bez_poley_otcheta(ev_s1, t):
    row = ev_s1.evaluate_v2({"vhod": 0.5, "proverka": "bars5m", "tejk_signal": t}, "train4")[0]
    assert not (POLYA_OTCHETA & set(row)), sorted(POLYA_OTCHETA & set(row))


@pytest.mark.parametrize("t", ["k_srednemu", "pol_puti"])
def test_dannye_tejk_menyaet_ryad(ev_s1, t):
    kw = dict(vhod=0.5, proverka="bars5m", tejk_signal=t)
    assert not np.array_equal(_x(ev_s1, vhod=0.5, proverka="bars5m"), _x(ev_s1, **kw))
