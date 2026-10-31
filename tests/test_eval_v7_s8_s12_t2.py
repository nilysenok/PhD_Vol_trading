"""Лист B s8_s12 (часть 2): S11/S12 на реальных сигналах impuls_v7 и S10 в момент T, S9 (сечение), ключ точки (Д5) и детерминизм.
Синтетика с известным ответом, без сети. Лента: плоско 100, пробои в последнем полном баре дня: D=310 вверх (110), F=330 вверх 1ч при высоком шипе 2ч-канала
(S11 молчит), E=350 вниз (90). Ожидание ручное: S10 {310: +1, 330: +1, 350: −1}; S11 = S12 = {310: +1, 350: −1}.
python3 -m pytest tests/test_eval_v7_s8_s12_t2.py -q"""
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from fastbt import eval_v7, exec_e
from fastbt import eval_v7_impuls as SI
from fastbt import kross_v7 as kv
from fastbt.eval_v6 import EvaluatorV6, IstochnikiV6
from fastbt.eval_v7 import EvaluatorV7
from fastbt.exec_e_types import Day, Params
from fastbt.indicators import compute_base
from fastbt.signals import v7_kross as vk
from fastbt.signals.interface import WARMUP, TickerData

NT = 400
DATES = pd.bdate_range("2018-01-02", periods=NT)
MODS = np.arange(590, 1140, 5)
SDF = Path(__file__).resolve().parent.parent


def _b10(okna: dict, shipy: dict = ()):
    """Плоская лента 100; okna {день: (слот с, слот по, цена)}; shipy {день: слот} — один бар с ценой 120 (высокий шип)."""
    px = np.full((NT, 54), 100.0)
    for d, (a, b, v) in okna.items():
        px[d, a:b + 1] = v
    for d, s in dict(shipy).items():
        px[d, s] = 120.0
    return pd.DataFrame({"date": np.repeat(DATES.to_numpy(), 54), "slot": np.tile(np.arange(54), NT),
                         **{c: px.ravel() for c in ("open", "high", "low", "close")}, "volume": 1.0})


def _lenta(a=42, b=47):
    return _b10({310: (a, b, 110.0), 330: (a, b, 110.0), 350: (a, b, 90.0)}, {326: 20})


def _td(name, idx, n):
    x = 100 + np.arange(float(n))
    return TickerData(name, idx.to_numpy("datetime64[ns]"), x, x, x, x, np.ones(n), np.zeros(n))


def _ev(sid, ist7, tds):
    e = object.__new__(EvaluatorV7)
    e.sid, e.proxy, e.ist7, e._fk, e.v6, e.ist, e._rel, e._dan = sid, eval_v7.PROXY[sid], ist7, (), {}, IstochnikiV6(), {}, {}
    e.sk = SimpleNamespace(_td={t: (td, compute_base(td.close, td.high, td.low, td.open, td.volume)) for t, td in tds.items()}, tickers=list(tds))
    e.r5, e._okup_k, e.fdr = None, None, None
    return e


def _probe(mp, sid, pt, b10=None, ist7=None, tds=None):
    ist7 = ist7 or eval_v7.Ist7(b10={"T": _lenta() if b10 is None else b10})
    cap, ev = {}, _ev(sid, ist7, tds or {"T": _td("T", DATES, NT)})

    def evaluate(self, q, period, vozm=None):
        self.r5 = SimpleNamespace(nov={}, ext={}, sp=dict(q))
        cap["vse"] = self._signals5(dict(q))
        cap.update(ss=cap["vse"].get("T") and cap["vse"]["T"].ss, fkey=self._fkey)
        self._rows(q, period, None)
        return "ok"

    def rows(self, pt_, period, anchor):
        cap.update(ext=dict(self.r5.ext), cena=self.fdr_kw().get("cena_T"))
        return []

    mp.setattr(EvaluatorV6, "evaluate_v2", evaluate)
    mp.setattr(EvaluatorV6, "_rows", rows)
    assert ev.evaluate_v2(dict(pt), "train4") == "ok"
    return cap


def _trades(cap):
    ss, st, hold = cap["ss"], 0.0, np.zeros(NT)
    for i in range(NT):                                       # ручная модель positions: сигнал открывает, флаг exit обнуляет бар
        st = 0.0 if ss.exit_l[0][i] else (float(ss.sig[0][i]) or st)
        hold[i] = st
    days = [Day(o := 100.0 + k + (MODS - 590) / 1000.0, o + 0.2, o - 0.2, o + 0.1, False, MODS) for k in range(NT)]
    p = replace(Params(comm=0.0, tau=2), **cap["ext"])
    cena = None if cap["cena"] is None else np.array([d.o[98] for d in days])      # open 5-мин бара 18:00 (цена b10 здесь плоская 100, не по ценам баров)
    return exec_e.run("T", days, hold, np.full(NT, 0.01), p, cena_T=cena).trades


ZHDEM = {"S10": [(310, 1), (330, 1), (350, -1)], "S11": [(310, 1), (350, -1)], "S12": [(310, 1), (350, -1)]}


@pytest.mark.parametrize("sid", ["S10", "S11", "S12"])
@pytest.mark.parametrize("vhod,sdvig", [("d1", 1), ("T", 0)])
def test_g1_signal_sdelki_znak_po_ruchnomu_schetu(monkeypatch, sid, vhod, sdvig):
    cap = _probe(monkeypatch, sid, dict(tau=2, vhod=vhod, T="18:00"))
    assert [(i, int(cap["ss"].sig[0][i])) for i in np.flatnonzero(cap["ss"].sig[0])] == ZHDEM[sid]
    tr = _trades(cap)
    assert [(t.d_sig, t.side) for t in tr] == ZHDEM[sid]
    assert [t.entry_day for t in tr] == [d + sdvig for d, _ in ZHDEM[sid]], "d1 — не раньше d+1 (Д2); T — в тот же день"
    assert [t.exit_day for t in tr] == [d + sdvig + 2 - sdvig for d, _ in ZHDEM[sid]] and all(t.exit_px for t in tr)   # выход close d+τ


def test_g1_s10_moment_T_17_00_vidit_bar_do_T_a_18_30_ne_vidit(monkeypatch):
    b = _b10({310: (36, 41, 110.0)})                                   # пробой в 1ч-баре 16:00–17:00
    sig = lambda **pt: [int(v) for v in _probe(monkeypatch, "S10", dict(tau=2, **pt), b)["ss"].sig[0]]    # noqa: E731
    assert [i for i, v in enumerate(sig(vhod="T", T="17:00")) if v] == [310]
    assert not any(sig(vhod="d1", T="17:00")), "решение в 18:30 бар 16:00–17:00 последним полным не считает"


def test_g1_kanon_d1_bez_T_schitaetsya_i_T_ne_vliyaet(monkeypatch):
    """canon_v7b убирает T при vhod = d1 — точка пилота без T должна считаться (раньше ValueError «нет осей ['T']»); результат тот же, что с любым T."""
    from fastbt import canon_v7b as c7b
    kub = c7b.load_kub("S10")
    pt = c7b.canon_v7b("S10", {"tau": 3, "vhod": "d1", "T": "18:00", "k_okup": "net"}, kub)
    assert "T" not in pt and "k_okup" not in pt
    bez = _probe(monkeypatch, "S10", pt)
    s_T = _probe(monkeypatch, "S10", dict(pt, T="18:40"))
    assert [int(v) for v in bez["ss"].sig[0]] == [int(v) for v in s_T["ss"].sig[0]] and bez["ss"].sig[0].any()


def test_g3_sdvig_ruda_lovitsya(monkeypatch):
    sdv = _b10({311: (42, 47, 110.0)})                                 # импульс, известный на день позже (M-d1)
    assert [i for i, v in enumerate(_probe(monkeypatch, "S10", dict(tau=2, vhod="T", T="18:00"), sdv)["ss"].sig[0]) if v] != [310]


# ---------------- S9: сечение ----------------
N, NS = 8, 320
IDX = pd.bdate_range("2017-01-02", periods=NS)
COLS = [f"T{i}" for i in range(N)]
METKI = np.array(["A", "A", "A", "B", "B", "B", "C", "C"] * NS, dtype=object).reshape(NS, N)


def _s9(mp, rebal, hvost=1.0, mut=False):
    """Сигналы S9 по всем тикерам [тикер, день]; hvost умножает цены дней ≥ 250 (решение дня ≤ 249 от них зависеть не вправе)."""
    r = np.random.default_rng(3)
    z = np.exp(np.cumsum(r.normal(0, 0.01, (NS, N)), axis=0))
    z[250:] *= np.exp(hvost * np.random.default_rng(4).normal(0, 0.3, (NS - 250, N)))
    c = pd.DataFrame(100 * z, index=IDX, columns=COLS)
    m = pd.Series(1000 * np.exp(np.cumsum(r.normal(0, 0.005, NS))), index=IDX)
    if mut:                                                            # M-d1: скор панели берёт закрытие d+1
        orig = kv.panel
        mp.setattr(kv, "panel", lambda c_, im, me, sid: orig(c_.shift(-1).ffill(), im, me, sid))
    pn = vk.Panel(IDX.to_numpy("datetime64[ns]"), COLS, c, m, METKI, np.ones((NS, N)), np.ones((NS, N), bool))
    cap = _probe(mp, "S9", dict(rebal=rebal, porog=0.3, bufer=0.1, cap=2, tau=3), ist7=eval_v7.Ist7(panel=pn),
                 tds={t: _td(t, IDX, NS) for t in COLS})
    return np.array([cap["vse"][t].ss.sig[0] for t in COLS]), cap


@pytest.mark.parametrize("rebal", ["den", "nedelya"])
def test_g1_s9_nejtralnost_i_bez_budushchego(monkeypatch, rebal):
    sig, _ = _s9(monkeypatch, rebal)
    assert (sig != 0).sum() > 20 and (sig == 1).sum() > 0 and (sig == -1).sum() > 0
    assert ((sig == 1).sum(axis=0) == (sig == -1).sum(axis=0)).all(), "сумма весов 0: лонгов и шортов поровну в каждый день"
    sig2, _ = _s9(monkeypatch, rebal, hvost=1.0)                       # детерминизм: те же входы — те же сигналы
    assert np.array_equal(sig, sig2)
    ish = _s9(monkeypatch, rebal, hvost=0.0)[0]                        # хвост цен после 250 убран (множитель 1) — прошлое не меняется
    assert np.array_equal(sig[:, :250], ish[:, :250]), "сигнал дня d < 250 зависит от цен позже d"


def test_g1_s9_nedelya_menyaetsya_tolko_v_posledniy_den_nedeli(monkeypatch):
    sig, _ = _s9(monkeypatch, "nedelya")
    iso = IDX.isocalendar()
    posl = np.append(iso["week"].to_numpy()[1:] != iso["week"].to_numpy()[:-1], True)     # ручной счёт: последний торговый день недели
    izm = [i for i in np.flatnonzero((sig[:, 1:] != sig[:, :-1]).any(axis=0)) + 1 if i > WARMUP]      # день WARMUP — конец прогрева (sig[:WARMUP] = 0)
    assert len(izm) > 3 and all(posl[i] for i in izm)


@pytest.mark.parametrize("rebal", ["den", "nedelya"])
def test_g3_s9_mutaciya_d_plus_1_lovitsya(monkeypatch, rebal):
    sig = _s9(monkeypatch, rebal, hvost=1.0, mut=True)[0]
    ish = _s9(monkeypatch, rebal, hvost=0.0, mut=True)[0]
    assert not np.array_equal(sig[:, :250], ish[:, :250]), "сигнал дня 249 при d+1 видит цену дня 250"


# ---------------- ключ точки и детерминизм (Д5) ----------------
def test_g4_klyuch_soderzhit_semyu_i_vse_osi(monkeypatch):
    ks = [_probe(monkeypatch, sid, dict(tau=tau, vhod=v, T=T))["fkey"] for sid, tau, v, T in
          [("S10", 2, "T", "18:00"), ("S11", 2, "T", "18:00"), ("S12", 2, "T", "18:00"), ("S10", 3, "T", "18:00"),
           ("S10", 2, "d1", "18:00"), ("S10", 2, "T", "18:40")]]
    assert len(set(map(repr, ks))) == len(ks), "семья, tau, vhod, T — каждая ось меняет ключ"
    assert len(set(map(repr, [_probe(monkeypatch, "S10", dict(tau=2, vhod="T", T="18:00"), _b10({310: (42, 47, 110.0)}))["fkey"],
                              _probe(monkeypatch, "S10", dict(tau=2, vhod="T", T="18:00"), _b10({310: (42, 47, 111.0)}))["fkey"]]))) == 2


KOD = ("import sys; sys.path.insert(0, %r)\nimport numpy as np, pandas as pd\nfrom fastbt import eval_v7\n"
       "d = pd.bdate_range('2018-01-02', periods=30)\n"
       "b = pd.DataFrame({'date': np.repeat(d.to_numpy(), 54), 'slot': np.tile(np.arange(54), 30), 'open': 100.0, 'high': 101.0, 'low': 99.0, 'close': 100.5, 'volume': 1.0})\n"
       "print(eval_v7.Ist7(b10={'T': b}).sha)\n")


def test_g4_sha_istochnikov_odinakov_v_dvuh_processah():
    d = pd.bdate_range("2018-01-02", periods=30)
    b = pd.DataFrame({"date": np.repeat(d.to_numpy(), 54), "slot": np.tile(np.arange(54), 30), "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5, "volume": 1.0})
    ok = eval_v7.Ist7(b10={"T": b}).sha
    for seed in ("1", "777"):
        out = subprocess.run([sys.executable, "-c", KOD % str(SDF)], capture_output=True, text=True, cwd=SDF, check=True, env={"PYTHONHASHSEED": seed, "PATH": ""}).stdout.strip()
        assert out == ok
