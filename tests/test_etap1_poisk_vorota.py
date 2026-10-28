# -*- coding: utf-8 -*-
"""Ворота Ж-044/Ж-047, ничья по σ126, кэш вырожденных точек, реестр «только дополняется». Синтетика, мутации — monkeypatch."""
from __future__ import annotations

import math

import numpy as np
import pytest

from fastbt import etap1_poisk as P
from fastbt import etap1_poisk_vorota as V

NET = "net"
AX = {"a": [0, 1, 2, 3, 4], "b": [0, 1, 2, 3, 4], "k_s": [1.0, 2.0, 3.0], "R": [1.0, 2.0, 3.0], "tau": [1, 3, 5]}
PT = {"a": 2, "b": 2, "k_s": 2.0, "R": 2.0, "tau": 3}


def led_for(pos, sdelok=40):
    """mSR соседа +1, если pos(p) истинно, иначе -1; сама точка PT положительна."""
    def ev(p, period):
        m = 1.0 if (p == PT or pos(p)) else -1.0
        return dict(msr_by_year={2018: m, 2019: m}, sdelok_y={2018: sdelok, 2019: sdelok}, vyrozhdena=False)
    return V.Ledger(ev)


def diff(p, *names):
    return all((p[a] != PT[a]) == (a in names) for a in AX)


def osi_cases():
    """(k_s, R, τ) положительных соседей -> ожидаемое решение (по каждой оси >= 1 и всего >= 3)."""
    cases = {(2, 1, 0): False, (1, 1, 1): True, (2, 0, 1): False, (2, 2, 2): True, (1, 1, 0): False}
    out = {}
    for (nk, nr, nt), want in cases.items():
        def pos(p, nk=nk, nr=nr, nt=nt):
            return ((diff(p, "k_s") and p["k_s"] in [1.0, 3.0][:nk]) or (diff(p, "R") and p["R"] in [1.0, 3.0][:nr])
                    or (diff(p, "tau") and p["tau"] in [1, 5][:nt]))
        out[(nk, nr, nt)] = (V.gate_osi(led_for(pos), PT, AX)["ok"], want)
    return out


def test_plato_po_ks_R_tau_otdelno():
    assert all(got == want for got, want in osi_cases().values()), osi_cases()


def test_mutaciya_osi_schitayutsya_summoy_lovitsya(monkeypatch):
    def summa(led, pt, axes):
        per = {a: V.n_pos(led, V.neighbors(pt, [a], axes)) for a in V.OSI}
        return dict(per=per, ok=sum(per.values()) >= 3)
    monkeypatch.setattr(V, "gate_osi", summa)
    assert not all(got == want for got, want in osi_cases().values())


def test_osi_dosschet_do_6_tochek_i_R_vyrozhdena_pri_ks_net():
    led = led_for(lambda p: True)
    g = V.gate_osi(led, PT, AX)
    assert led.n_unique == 6 and set(g["per"]) == {"k_s", "R", "tau"}              # досчёт соседей идёт в N
    pt = dict(PT, k_s=NET)
    ax = dict(AX, k_s=[1.0, 2.0, NET])
    g2 = V.gate_osi(led_for(lambda p: True), pt, ax)
    assert "R" not in g2["per"] and g2["per"]["k_s_net"] is True          # Ж-048 п. 1: «нет» — одностороннее плато


def gate_cases():
    ok = lambda *, pbo=0.0, s9=True, led=None: V.gates(led or led_for(lambda p: True), PT, AX, ["a", "b"], pbo, lambda p: s9)   # noqa: E731
    pos2 = lambda p: diff(p, "a") and p["a"] == 1 or diff(p, "b") and p["b"] == 1        # noqa: E731  (только 2 положительных соседа плана)
    pos3 = lambda p: (diff(p, "a") and p["a"] == 1) or (diff(p, "b") and p["b"] in (1, 3))    # noqa: E731
    return {"base": ok()["G1_pbo"] and ok()["G2_sdelki"] and ok()["G3_plato"] and ok()["G5_s9"],
            "pbo=0.19": ok(pbo=0.19)["G1_pbo"], "pbo=0.2": not ok(pbo=0.2)["G1_pbo"], "pbo nan": not ok(pbo=float("nan"))["G1_pbo"],
            "sdelok 29": not ok(led=led_for(lambda p: True, 29))["G2_sdelki"], "sdelok 30": ok(led=led_for(lambda p: True, 30))["G2_sdelki"],
            "s9": not ok(s9=False)["G5_s9"], "plato 2": not ok(led=led_for(pos2))["G3_plato"], "plato 3": ok(led=led_for(pos3))["G3_plato"],
            "itog ok": ok()["ok"], "itog net pri s9": not ok(s9=False)["ok"]}


def test_vorota_kazhdoe_otdelno():
    bad = [k for k, v in gate_cases().items() if not v]
    assert not bad, bad


@pytest.mark.parametrize("attr,val", [("PBO_MAX", 0.5), ("MIN_SDELOK", 0)])
def test_mutaciya_porog_vorot_lovitsya(monkeypatch, attr, val):
    monkeypatch.setattr(V, attr, val)
    assert not all(gate_cases().values())


def test_mutaciya_plato_schitaet_otricatelnyh_lovitsya(monkeypatch):
    monkeypatch.setattr(V, "n_pos", lambda led, pts: len(pts))
    assert not all(gate_cases().values())


def test_nichya_reshaetsya_men_shey_sigma126():
    rows = [dict(score=1.00, sigma126=0.9), dict(score=0.96, sigma126=0.2), dict(score=0.90, sigma126=0.0)]
    assert V.pick_best(rows, "score")["sigma126"] == 0.2                       # 0,04 <= 0,05 — ничья; 0,10 — нет
    assert V.pick_best(rows[:1] + [dict(score=0.94, sigma126=0.1)], "score")["score"] == 1.0
    assert V.pick_best([dict(score=math.nan, sigma126=0.1)], "score") is None


def test_mutaciya_nichya_bez_sigma126_lovitsya(monkeypatch):
    rows = [dict(score=1.00, sigma126=0.9), dict(score=0.96, sigma126=0.2)]
    monkeypatch.setattr(V, "TIE", 0.0)
    assert V.pick_best(rows, "score", tol=V.TIE)["sigma126"] != 0.2


def test_sigma126_po_rolling_sharpe_obeih_let():
    rng = np.random.default_rng(1)
    calm = {2018: rng.normal(0.001, 0.01, 252), 2019: rng.normal(0.001, 0.01, 252)}
    shift = {2018: rng.normal(0.01, 0.01, 252), 2019: rng.normal(-0.01, 0.01, 252)}
    a, b = V.sigma126(dict(excess_d=calm)), V.sigma126(dict(excess_d=shift))
    s = np.r_[calm[2018], calm[2019]]
    r = [s[i:i + 126].mean() / s[i:i + 126].std(ddof=1) * math.sqrt(252) for i in range(len(s) - 125)]
    assert a == pytest.approx(np.std(r, ddof=1)) and b > a
    assert V.sigma126(dict(excess_d={2018: calm[2018]})) != pytest.approx(a)        # 2019 не выброшен


def test_vyrozhdennye_tochki_schitayutsya_odin_raz():
    calls = []
    led = V.Ledger(lambda p, per: calls.append(p) or dict(msr_by_year={2018: 1.0}))
    base = dict(PT, k_s=NET)
    led(dict(base, R=1.0), "train")
    led(dict(base, R=2.0), "train")
    led(dict(base, R=3.0), "ctrl2019")
    assert len(calls) == 2 and led.n_unique == 1 and calls[0]["R"] == NET


def test_mutaciya_vyrozhdenie_ne_svoditsya_lovitsya(monkeypatch):
    monkeypatch.setattr(V, "canon", lambda p: dict(p))
    led = V.Ledger(lambda p, per: dict(msr_by_year={2018: 1.0}))
    led(dict(PT, k_s=NET, R=1.0), "train")
    led(dict(PT, k_s=NET, R=2.0), "train")
    assert led.n_unique == 2


def test_reestr_tolko_dopolnyaetsya(tmp_path):
    f = tmp_path / "x.csv"
    P.append_csv(f, [dict(s="S1", v=1)])
    P.append_csv(f, [dict(s="S1", v=2)])
    txt = f.read_text(encoding="utf-8").splitlines()
    assert txt[0].startswith("#") and txt[1:] == ["s,v", "S1,1", "S1,2"]
    with pytest.raises(ValueError):
        P.append_csv(f, [dict(s="S1", w=3)])
    assert f.read_text(encoding="utf-8").splitlines() == txt


def test_mutaciya_reestr_perezapisyvaetsya_lovitsya(tmp_path, monkeypatch):
    def perezapis(path, rows):
        import pandas as pd
        pd.DataFrame(rows).to_csv(path, index=False)
    monkeypatch.setattr(P, "append_csv", perezapis)
    f = tmp_path / "x.csv"
    P.append_csv(f, [dict(s="S1", v=1)])
    P.append_csv(f, [dict(s="S1", v=2)])
    assert len(f.read_text(encoding="utf-8").splitlines()) != 4              # правильный реестр: шапка-комментарий + заголовок + 2 строки


def test_dsr_N_vliyaet_mutaciya_N1():
    rng = np.random.default_rng(3)
    M = rng.normal(0, 1, (48, 30)) + np.linspace(0, 0.3, 30)
    x = M[:, -1]
    many, one = V.psr_dsr(x, M, 5000), V.psr_dsr(x, M, 1)
    assert many["dsr"] < one["dsr"] == pytest.approx(one["psr"]) and many["T"] == 48 and many["N_dsr"] == 5000
    assert V.pbo(M[:, :1]) != V.pbo(M[:, :1])                                 # N < 2 -> nan, ворота PBO закрыты
