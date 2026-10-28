"""Л-21: tejk_signal = rodnoy_uroven (Ж-091 п. 1.1, 1.3). Синтетика, данных нет.
python3 -m pytest tests/test_rodnoy_uroven.py -q --tb=short
Мутации (monkeypatch правил): уровень по d+1, уровень не обновлён, активен в баре выставления, t_vyhod не выключен, касание вместо прохода."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from fastbt import exec_e_run, fast_etap1
from fastbt import tejk_signal as ts
from fastbt.etap1_eval_v2 import EvaluatorV2
from fastbt.exec_e import Params, run
from fastbt.exec_e_rules import rodnoy_level, take_hit
from fastbt.exec_e_run import Run
from fastbt.exec_e_types import Day
from fastbt.ledger_v2 import canon_v2

from fastbt import bars5m, cache

_MAIN_BARS = cache.MAIN / "data" / "regen" / "D25" / "bars1m"
if not bars5m.SRC.exists() and _MAIN_BARS.exists():
    bars5m.SRC = _MAIN_BARS

TK = 0.01


def D(*rows):
    a = np.array(rows, float).reshape(-1, 4)
    return Day(a[:, 0], a[:, 1], a[:, 2], a[:, 3])


def flat(c=100.0):
    return D((c, c, c, c))


def go(days, L_long, sig, L_short=None, f=run, **kw):
    n = len(days)
    ls = np.full(n, np.nan) if L_short is None else np.asarray(L_short, float)
    p = Params(tick=TK, check="bary", comm=0.0, tejk="rodnoy_uroven", exit_signal=False, **kw)
    return f("T", days, np.asarray(sig, float), np.full(n, 0.02), p, tejk_M=np.vstack([np.asarray(L_long, float), ls]))


# день 0: вход по close 100; L[0]=105 (день 1), L[1]=103 (день 2), L[2]=101 (день 3)
DAYS = [flat(), D((100, 104, 99, 100)), D((100, 104, 99, 100)), D((100, 102, 99, 100)), flat()]
LL = [105.0, 103.0, 101.0, 101.0, 101.0]
SIG = [1, 1, 1, 1, 0]


def _exit(r):
    t = r.trades[0]
    return t.exit_day, t.exit_px, t.reason


def test_uroven_obnovlyaetsya_kazhdyj_vecher():
    # день 1: заявка 105 (high 104 — нет); день 2: заявка 103 (high 104 ≥ 103+тик) -> take по max(open 100, 103) = 103
    assert _exit(go(DAYS, LL, SIG)) == (2, 103.0, "take")


def test_gep_po_open_i_stop_pervym():
    days = [flat(), D((106, 107, 105, 106)), flat()]
    assert _exit(go(days, [105.0, 105.0, 105.0], [1, 1, 0])) == (1, 106.0, "take")        # гэп через уровень — по open
    days = [flat(), D((100, 110, 90, 100)), flat()]
    assert _exit(go(days, [105.0, 105.0, 105.0], [1, 1, 0], b=0.2))[2] == "stop"          # оба в баре — стоп первым


def test_shorty_beryut_svoyu_stroku():
    days = [flat(), D((100, 101, 94, 100)), flat()]
    assert _exit(go(days, [150.0] * 3, [-1, -1, 0], L_short=[95.0] * 3)) == (1, 95.0, "take")


def test_nan_i_pervyj_den_bez_zayavki():
    assert go(DAYS, [np.nan] * 5, SIG).trades[0].reason != "take"
    assert rodnoy_level(1, float("nan"), TK) is None and rodnoy_level(1, 100.001, TK) == pytest.approx(100.01)
    assert rodnoy_level(-1, 100.009, TK) == pytest.approx(100.0)


def test_signal_vyhod_vyklyuchen():
    n = len(DAYS)
    p = Params(tick=TK, check="bary", comm=0.0, tejk="rodnoy_uroven", exit_signal=True)
    r = run("T", DAYS, np.array([1, 0, 0, 0, 0.0]), np.full(n, .02), p, tejk_M=np.vstack([[200.0] * n, [np.nan] * n]))
    assert r.trades[0].reason is None                            # выход по сигналу выключен в движке даже при exit_signal=True
    p = Params(tick=TK, check="bary", comm=0.0, tejk="net", exit_signal=True)
    assert run("T", DAYS, np.array([1, 0, 0, 0, 0.0]), np.full(n, .02), p, tejk_M=np.full(n, 200.0)).trades[0].reason == "signal"   # контроль


def test_tau_ostayotsya():
    r = go(DAYS, [200.0] * len(DAYS), [1, 1, 1, 1, 1], tau=2)
    assert r.trades[0].reason == "vremya" and r.trades[0].exit_day == 2


def test_mutatsiya_signal_vyhod_vklyuchen_lovitsya(monkeypatch):
    n = len(DAYS)
    p = Params(tick=TK, check="bary", comm=0.0, tejk="rodnoy_uroven", exit_signal=True)
    args = ("T", DAYS, np.array([1, 0, 0, 0, 0.0]), np.full(n, .02), p)
    monkeypatch.setattr(Run, "_sig_exit", lambda s, i: s.p.exit_signal and s.dir(i) != s.tr.side)    # мутация: guard rodnoy снят
    assert run(*args, tejk_M=np.vstack([[200.0] * n, [np.nan] * n])).trades[0].reason == "signal"


def test_toch_tik_a_ne_kasanie():
    days = [flat(), D((100, 105.0, 99, 100)), flat()]           # high == уровню: касание — не исполнено
    assert go(days, [105.0] * 3, [1, 1, 0]).trades[0].reason is None
    days = [flat(), D((100, 105.01, 99, 100)), flat()]
    assert _exit(go(days, [105.0] * 3, [1, 1, 0]))[2] == "take"


def test_ne_aktiven_v_bare_vystavleniya():
    # лимитный вход (k) исполнен на баре 1 дня 1; бар 1 тоже ушёл выше уровня 103, бары 2.. — нет: тейк не должен сработать на баре входа
    days = [flat(), D((100, 100, 100, 100), (100, 104, 98, 100), (100, 100, 99, 100)), flat()]
    r = go(days, [103.0] * 3, [1, 1, 0], k=0.01, f=run)
    t = r.trades[0]
    assert t.entry_day == 1 and t.entry_bar == 1 and t.reason != "take"


def test_bystryj_sloj_medlennym_putyom():
    assert not fast_etap1.pokryto({"tejk_signal": "rodnoy_uroven"})
    a, b = go(DAYS, LL, SIG), go(DAYS, LL, SIG, f=fast_etap1.run)
    assert a.trades == b.trades


def test_tejk_M_forma():
    with pytest.raises(ValueError):                              # rodnoy требует [2, days]
        run("T", DAYS, np.array(SIG, float), np.full(5, .02), Params(tick=TK, tejk="rodnoy_uroven", exit_signal=False), tejk_M=np.array(LL))


# --- формулы уровня ---
def _dn(c, n=30):
    c = np.asarray(c, float)[None]
    return {"close": c, "high": c + 1, "low": c - 1, "volume": np.ones_like(c), "dates": np.datetime64("2018-01-01") + np.arange(c.shape[1])}


def test_formuly_S1_S2_S5_S6():
    dn = _dn(np.arange(1.0, 31.0))
    sma, std = 8.0, np.sqrt(2.5)
    assert ts.rodnoy_uroven("S1", {"ma_window": 5, "z_exit": 0.5}, dn)[0, 9] == pytest.approx(sma - 0.5 * std)
    assert ts.rodnoy_uroven("S1", {"ma_window": 5, "z_exit": 0.5}, dn, sd=-1)[0, 9] == pytest.approx(sma + 0.5 * std)
    assert ts.rodnoy_uroven("S2", {"bb_window": 5}, dn)[0, 9] == pytest.approx(8.0)
    sp5 = {"pivot_period": "weekly", "pivot_type": "classic"}
    assert np.allclose(ts.rodnoy_uroven("S5", sp5, dn), ts.uroven_M("S5", sp5, dn), equal_nan=True)
    sp6 = {"vwap_window": 5}
    vw, dv = ts._vwap(dn["close"][0], dn["high"][0], dn["low"][0], dn["volume"][0], 5)
    assert ts.rodnoy_uroven("S6", sp6, dn)[0, 9] == pytest.approx(vw[9] - 0.5 * dv[9])
    assert ts.rodnoy_uroven("S6", sp6, dn, sd=-1)[0, 9] == pytest.approx(vw[9] + 0.5 * dv[9])
    with pytest.raises(ValueError):
        ts.rodnoy_uroven("S3", {}, dn)


@pytest.mark.parametrize("sid,sp", [("S1", {"ma_window": 5, "z_exit": 0.5}), ("S2", {"bb_window": 5}),
                                    ("S5", {"pivot_period": "weekly", "pivot_type": "classic"}), ("S6", {"vwap_window": 5})])
def test_uroven_ne_smotrit_vpered(sid, sp):
    c = np.sin(np.arange(40) / 3.0) * 5 + 100
    c2 = c.copy()
    c2[20:] += 50.0
    a, b = (ts.rodnoy_uroven(sid, sp, _dn(x)) for x in (c, c2))
    assert np.allclose(a[:, :20], b[:, :20], equal_nan=True)


# --- схлопывание ---
def _canon_ok(f):
    q = f("S2", {"tejk_signal": "rodnoy_uroven", "R": 2.0, "k_x": 1.0, "t_vyhod": "1500", "tau": 3})
    return q["t_vyhod"] == "open2" and q["tau"] == 3 and q["R"] == "net" and q["k_x"] == "net"


def test_canon_skhlopyvaet_t_vyhod_tau_ostayotsya():
    assert _canon_ok(canon_v2)
    assert canon_v2("S2", {"tejk_signal": "pol_puti", "t_vyhod": "1500", "tau": 3})["t_vyhod"] == "1500"     # прочие значения — как были


def _params(pt_extra):
    pt = dict(k_s=2.0, tau=3, vhod="rynok", R="net", k_tr="net", bezubytok="net", proverka="bars5m")
    nov = dict(tejk_signal="rodnoy_uroven", k_x="net", zhizn="den", gap="vhod", t_vyhod="1500", **pt_extra)
    ev = SimpleNamespace(inp=SimpleNamespace(tick={"T": TK}), comm=0.0004)
    return EvaluatorV2._params(ev, pt, "T", 5.0, False, nov, {})


def _params_ok(p):
    return p.tejk == "rodnoy_uroven" and not p.exit_signal and p.tau == 3 and p.t_vyhod == "open2"


def test_evaluator_signal_vyklyuchen_tau_i_open2():
    assert _params_ok(_params({}))
    assert not _params_ok(replace(_params({}), tau=None))        # мутация: τ выключен — ловится


# --- мутации ---
def _mut_dplus1(s, j):
    sd = s.tr.side
    return rodnoy_level(sd, float(s.tejk_M[0 if sd > 0 else 1][s.ix[j]]), s.p.tick)


def _mut_fixed(s, j):
    sd = s.tr.side
    return rodnoy_level(sd, float(s.tejk_M[0 if sd > 0 else 1][s.ix[s.ej]]), s.p.tick)


@pytest.mark.parametrize("mut", [_mut_dplus1, _mut_fixed])
def test_mutatsii_urovnya_lovyatsya(monkeypatch, mut):
    ok = _exit(go(DAYS, LL, SIG))
    monkeypatch.setattr(Run, "_rodnoy", mut)
    assert _exit(go(DAYS, LL, SIG)) != ok


def test_mutatsiya_aktiven_v_bare_vystavleniya_lovitsya(monkeypatch):
    days = [flat(), D((100, 100, 100, 100), (100, 104, 98, 100), (100, 100, 99, 100)), flat()]
    ok = go(days, [103.0] * 3, [1, 1, 0], k=0.01).trades[0].reason
    orig = Run._scan
    monkeypatch.setattr(Run, "_scan", lambda s, i, j, d, b0: orig(s, i, j, d, max(b0 - 1, 0)))
    assert go(days, [103.0] * 3, [1, 1, 0], k=0.01).trades[0].reason != ok


def test_mutatsiya_kasanie_vmesto_prohoda_lovitsya(monkeypatch):
    days = [flat(), D((100, 105.0, 99, 100)), flat()]
    ok = go(days, [105.0] * 3, [1, 1, 0]).trades[0].reason
    monkeypatch.setattr(exec_e_run, "take_hit", lambda sd, o, h, l, T, tick: take_hit(sd, o, h, l, T, 0.0))
    assert go(days, [105.0] * 3, [1, 1, 0]).trades[0].reason != ok


@pytest.fixture(scope="module")
def ev():
    from fastbt import etap1
    return EvaluatorV2("S1", etap1.build_inputs("S1", period="train4"), etap1.base_point("S1"), 0.0004)


def test_realnaya_lenta_skhlopnutye_osi_ravny(ev):
    """Данные ≤ 2019. t_vyhod при rodnoy_uroven не влияет на строку (схлопнут в open2); тейки по заявке есть."""
    pt = dict(vhod=0.5, proverka="bars5m", tejk_signal="rodnoy_uroven", tau=3)
    a = ev.evaluate_v2({**pt, "t_vyhod": "open2"}, "train4")[0]
    d = ev.diag
    b = ev.evaluate_v2({**pt, "t_vyhod": "1500"}, "train4")[0]
    assert a["status"] == "ok" and a["by_quarter"] == b["by_quarter"] and any(v > 0 for v in d["dolya_tejk_y"].values())


def test_mutatsiya_t_vyhod_ne_vyklyuchen_lovitsya(monkeypatch):
    from fastbt import ledger_v2

    def bad(sid, pt):
        q = ledger_v2.canon_v2(sid, pt)
        q.update(t_vyhod=pt.get("t_vyhod", "close"))
        return q
    assert _canon_ok(canon_v2) and not _canon_ok(bad)
