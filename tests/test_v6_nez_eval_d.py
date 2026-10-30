"""V6-12d, независимая проверка PR #700 (тестировщик, не автор). Синтетика, без данных (≤ 60 с).
(1) пивот S5 / VWAP S6 / pol_puti через оценщик: tejk_M, дошедший до движка, — уровни строки d_sig (не d+1), PivUrovni из SignalSet.extra;
(2) оси тренда на стоп-заявке доходят до Params движка (b0, b0_c с рангом дня старта #683, sp, step_s, pir, reentry_den);
(3) свёртки canon_v6: time_decay без tau / при tejk ≠ net → один канон; reentry при podtverzhdenie → один канон; N Соболя не уменьшается;
(4) fkey различает новые оси; (5) tejk_signal.otchet с ногой pir открытой позиции не падает и даёт верную сумму.
Мутации (в коде, по одной; откат git checkout -- fastbt/): пивот со строки d+1 (eval_v6_piv.tm: slice(s0 + 1, …)); снята свёртка time_decay (canon_v6);
ось тренда не передаётся в Params (eval_v6_trend.vernut / primenit) — каждая роняет файл."""
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest

from fastbt import canon_v5, canon_v6, canon_v6_nesovm as nesovm, eval_v5, eval_v6_piv as piv6, etap1_eval_v2, tejk_signal
from fastbt.eval_v5_osi import yakor
from fastbt.eval_v6 import EvaluatorV6, fkey
from fastbt.exec_e_piv import PivUrovni
from fastbt.exec_e_types import Day, Trade

N = 14
S0, B = 2, 9                                              # окно оценки: дни [S0; S0 + B]
EXTRA = {k: np.arange(40, dtype=float) * (3 + i) + 100 * i for i, k in enumerate(("P", "S1", "S2", "R1", "R2", "vwap", "dev"))}
SP = {"S3": dict(okno=5, podtv="none"), "S4": dict(period=5, mnozh=0.5),
      "S5": dict(pivot_type="classic", pivot_period="den", uroven=1, atr_filtr=0, filter_set="net"),
      "S6": dict(vwap_window=10, dev_mult=0.5, dir_bias="OFF", filter_set="net")}
TR = ("S3", "S4")


def _vykl(sid):
    return {n: v["setka"][0] for n, v in canon_v6.novye(canon_v6._kub(sid)).items()}


def _pt(sid, **kw):
    return {**yakor(sid), **SP[sid], **({"trail": "atr:1.0"} if sid in TR else {"k_e": 1.0}), **_vykl(sid), **kw}


class Zahvat:
    """Оценщик v6 без данных: движок подменён записью (Params, kw) того, что оценщик отдаёт exec_e."""
    def __init__(self, monkeypatch, sid):
        self.calls = []
        days = [Day(*(np.full(1, 100.0),) * 4) for _ in range(N)]
        self.sig, self.rg = np.zeros(N), np.linspace(0.1, 0.9, B + 1)
        self.sig[3:7] = 1.0
        inp = SimpleNamespace(sk=SimpleNamespace(tickers=["T"], _td={"T": (None, None)}, sig_names=()), tick={"T": 0.01}, bars={"T": days},
                              close_n6={"T": np.full(N, 100.0)}, div={"T": np.zeros(N)}, sigma={})
        e = EvaluatorV6(sid, inp, {})
        e._start = lambda period: {"T": (S0, B, 0, np.arange(B + 1), self.sig[:B + 1], np.full(B + 1, 0.01), None)}
        e._ranki = lambda info: {"T": (self.rg, np.ones(B + 1, bool))}
        e._urovni = lambda t, s0, b, al: None
        e._ed = lambda *a: None
        e._piv_kesh = {("T", ()): EXTRA}
        monkeypatch.setattr(piv6, "_extra", lambda evr, t: EXTRA)
        monkeypatch.setattr(etap1_eval_v2.EvaluatorV2, "evaluate_v2", lambda s, pt, period: s._rows(pt, period, False))
        monkeypatch.setattr(eval_v5, "run_rows", lambda t, days, sig, sg, p, **kw: self.calls.append((p, kw)) or "rows")
        monkeypatch.setattr(eval_v5, "_one_bar", lambda c: Day(*(np.full(1, c),) * 4))
        self.e = e

    def __call__(self, pt):
        self.calls.clear()
        self.e.evaluate_v2(pt, "train4")
        assert len(self.calls) == 1
        return self.calls[0]


# ---------------- (1) пивот / VWAP / pol_puti: уровни строки d_sig ----------------
def _urovni_d(lv) -> bool:
    return all(np.array_equal(getattr(lv, k), EXTRA[k][S0:S0 + B + 1]) for k in ("P", "S1", "S2", "R1", "R2", "vwap", "dev"))


def test_piv_urovni_stroka_d_a_ne_d_plyus_1(monkeypatch):
    p, kw = Zahvat(monkeypatch, "S5")(_pt("S5", stop_nach="piv:next"))
    assert isinstance(kw["tejk_M"], PivUrovni) and p.piv_sl == "next"
    assert _urovni_d(kw["tejk_M"])                                                # день d окна = строка d; сдвиг d+1 дал бы EXTRA[k][S0 + 1:…]


def test_vwap_s6_pol_puti_i_piv_tp(monkeypatch):
    z = Zahvat(monkeypatch, "S6")
    p, kw = z(_pt("S6", stop_nach="vwap:1.5"))
    assert (p.piv_sl, p.vwap_m) == ("vwap", 1.5) and np.array_equal(kw["tejk_M"].vwap, EXTRA["vwap"][S0:S0 + B + 1])
    p, kw = z(_pt("S6", tejk="pol_puti"))
    assert p.tejk == "pol_puti" and np.array_equal(kw["tejk_M"], EXTRA["vwap"][S0:S0 + B + 1])    # S6: M = VWAP строки d
    z5 = Zahvat(monkeypatch, "S5")
    p, kw = z5(_pt("S5", tejk="pol_puti"))
    assert np.array_equal(kw["tejk_M"], EXTRA["P"][S0:S0 + B + 1])
    p, kw = z5(_pt("S5", tejk="piv:R"))
    assert p.piv_tp == "R" and isinstance(kw["tejk_M"], PivUrovni)


def test_piv_nesovmestimoe_otkaz(monkeypatch):
    z = Zahvat(monkeypatch, "S5")
    for kw in (dict(trail="atr:1.0"), dict(pir=0.5), dict(razmer_tip="risk_sigma"), dict(tejk="pol_puti")):
        with pytest.raises(etap1_eval_v2.NeSchitana):
            z(_pt("S5", stop_nach="piv:next", **kw))


def test_mutatsiya_pivot_d_plyus_1_lovitsya(monkeypatch):
    z = Zahvat(monkeypatch, "S5")
    orig = piv6.tm
    monkeypatch.setattr(piv6, "tm", lambda evr, ext, t, s0, b: orig(evr, ext, t, s0 + 1, b))
    _, kw = z(_pt("S5", stop_nach="piv:next"))
    assert not _urovni_d(kw["tejk_M"])                                           # проверка (1) на сдвиге падает


# ---------------- (2) оси тренда доходят до движка ----------------
@pytest.mark.parametrize("sid", TR)
@pytest.mark.parametrize("ax,val,pole,ozid", [("bezubytok", "atr:1.0", "b0", 1.0), ("stop_v_plyus", "2:1", "sp", (2.0, 1.0)), ("pir", 0.5, "pir", 0.5),
                                              ("trail", "step:0.5", "step_s", 0.5)])
def test_os_trenda_v_params(monkeypatch, sid, ax, val, pole, ozid):
    z = Zahvat(monkeypatch, sid)
    assert getattr(z(_pt(sid))[0], pole) in (None, 0, 0.0)
    assert getattr(z(_pt(sid, **{ax: val}))[0], pole) == ozid, f"{sid}.{ax}: не дошла до Params"


@pytest.mark.parametrize("sid", TR)
def test_reentry_i_b0c_s_rangom_dnya_starta(monkeypatch, sid):
    z = Zahvat(monkeypatch, sid)
    assert z(_pt(sid, reentry="da"))[0].reentry_den is True and z(_pt(sid))[0].reentry_den is False
    p, kw = z(_pt(sid, bezubytok="rang:1"))
    assert p.b0_c == 1.0 and p.b0 is None and kw["rang"] is not None            # ранг передан движку
    rs = eval_v5.rang_starta(z.sig[:B + 1], z.rg)                                # известный ответ: дни 3..6 несут ранг дня 3 (старт), вне сигнала nan
    assert np.allclose(rs[3:7], z.rg[3]) and np.isnan(rs[2]) and np.isnan(rs[7])
    # вес по рангу дня старта: m_sily lin → вес дня 6 равен весу дня 3 (а не по собственному рангу дня 6)
    _, kw = z(_pt(sid, m_sily="lin", rang_tip="obshchiy"))
    w = np.asarray(kw["w"])
    assert w[3] > 0 and np.allclose(w[3:7], w[3]) and not np.isclose(w[3], w[8])


# ---------------- (3) свёртки canon_v6 ----------------
def _q(sid, **kw):
    kub = canon_v6.load_kub(sid)
    base = {n: g[0] for n, g in canon_v5.osi(kub["v5"])}
    return kub, {**base, **{n: v["setka"][0] for n, v in canon_v6.novye(kub).items()}, **kw}


@pytest.mark.parametrize("sid", canon_v5.SIDS)
def test_time_decay_bez_tau_ili_pri_tejke_odin_kanon(sid):
    kub, q = _q(sid)
    if "time_decay" not in canon_v6.novye(kub):
        pytest.skip("нет оси")
    off = canon_v6.novye(kub)["time_decay"]["setka"][0]
    cv = lambda **kw: canon_v6.canon_v6(sid, {**q, **kw}, kub)                  # noqa: E731
    assert cv(time_decay="da", tau="net", tejk="net") == cv(time_decay=off, tau="net", tejk="net") is not None    # без tau — «выкл.», точка остаётся
    on, of = cv(time_decay="da", tau=5, tejk="rr:2.0"), cv(time_decay=off, tau=5, tejk="rr:2.0")
    assert on == of                                                              # при tejk ≠ net — то же
    keep = cv(time_decay="da", tau=5, tejk="net")
    assert keep is None or keep.get("time_decay") == "da"                        # при tau и tejk = net ось жива


@pytest.mark.parametrize("sid", TR)
def test_reentry_pri_podtverzhdenii_odin_kanon(sid):
    kub, q = _q(sid)
    a = canon_v6.canon_v6(sid, {**q, "vhod_tip": "podtverzhdenie", "reentry": "da"}, kub)
    b = canon_v6.canon_v6(sid, {**q, "vhod_tip": "podtverzhdenie", "reentry": "net"}, kub)
    assert a == b and a is not None and "reentry" not in a
    assert canon_v6.canon_v6(sid, {**q, "reentry": "da"}, kub)["reentry"] == "da"


@pytest.mark.parametrize("sid", canon_v5.SIDS)
def test_N_sobolya_ne_umenshaetsya_ot_svertok(sid):
    kub = canon_v6.load_kub(sid)
    with mock.patch.object(nesovm, "PRAVILA", ()):                               # свёртки без правил несовместимости: смысл прежний
        pts0, vyb0 = canon_v6.tochki_v6(kub, 0, 512)
    assert len(pts0) / 512 > 0.45, (sid, len(pts0), vyb0)                        # свёртка склеивает каноны, а не выбрасывает точки (было ≈ 29 %)
    pts, vyb = canon_v6.tochki_v6(kub, 0, 512)
    assert len(pts) / 512 > 0.30, (sid, len(pts), vyb)                           # 30.10: правила несовместимости (canon_v6_nesovm) выбрасывают ещё 21–26 % (замер 512: S1 0.37 … S6 0.32; запас ≈ 7 п.п.)


# ---------------- (4) fkey ----------------
def test_fkey_razlichaet_novye_osi():
    ks = {fkey({}, {k: v}) for k, v in (("piv_buf", 0.1), ("piv_buf", 0.2), ("bezubytok", "atr:1.0"), ("bezubytok", "rang:1"), ("stop_v_plyus", "2:1"),
                                         ("pir", 0.5), ("pir", 1.0), ("reentry", "da"))}
    assert len(ks) == 8 and fkey({}, {}) == ()


# ---------------- (5) otchet с ногой pir ----------------
def test_otchet_noga_pir_summa():
    dates = np.array(["2018-01-02", "2018-01-03", "2019-01-04", "2019-01-07"], "datetime64[ns]")
    a = Trade("T", 0, 1, 0, 10.0, exit_day=2, exit_bar=0, exit_px=11.0, reason="take")
    b = Trade("T", 0, 1, 0, 10.0, exit_day=3, exit_bar=0, exit_px=9.0, reason="stop")
    noga = Trade("T", 0, 2, 0, 10.5, reason="pir", parent=0)                     # открытая нога: выхода нет
    o, o2 = tejk_signal.otchet([([a, noga, b], dates)], (2018, 2019)), tejk_signal.otchet([([a, b], dates)], (2018, 2019))
    assert o["dolya_tejk_y"]["2019"] == 0.5 and o["dolya_pribyl_y"]["2019"] == 0.5
    assert repr(o) == repr(o2)                                                   # нога открытой позиции ничего не добавляет
