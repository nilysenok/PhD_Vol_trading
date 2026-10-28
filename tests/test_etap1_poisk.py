# -*- coding: utf-8 -*-
"""Драйвер поиска этапа 1 (Ж-047): синтетика с известным ответом, мутации — monkeypatch. Сети и данных нет."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
import yaml

from fastbt import etap1_poisk as P
from fastbt import sweep_drivers as sd

NET = "net"
AXES = {"a": list(range(7)), "b": list(range(5)), "k_s": [1.0, 2.0, NET], "R": [1.0, 2.0, NET], "tau": [1, 3, NET],
        "sigma": ["rv5", "rv22"], "proverka": ["zakrytie", "bars5m"]}
YAKOR = {"a": 3, "b": 2, "k_s": NET, "R": NET, "tau": NET, "sigma": "rv22", "proverka": "zakrytie"}
SETKA = dict(sid="S9", axes=AXES, yakor=YAKOR, r=20, seed=20261027)
NOISE = {y: np.random.default_rng(y).normal(size=24) for y in (2018, 2019)}


class World:
    """Известный ответ: плато вокруг (a=3, b=2), одиночный шумовой пик в (a=0, b=4, k_s=1, R=1, tau=1); важная ось — a."""

    def __init__(self, sdelok=40):
        self.calls, self.sdelok = [], sdelok

    @staticmethod
    def score(p):
        s = 1.5 - 0.15 * (p["a"] - 3) ** 2 - 0.05 * (p["b"] - 2) ** 2
        return s + (5.0 if (p["a"], p["b"], p["k_s"], p["R"], p["tau"]) == (0, 4, 1.0, 1.0, 1) else 0.0)

    def evaluate(self, p, period):
        self.calls.append(period)
        ys, s = ((2019,) if period == "ctrl2019" else (2018, 2019)), self.score(p)
        return dict(msr_by_year={y: s for y in ys}, excess_d={y: NOISE[y] + 0.001 * s for y in ys},
                    sdelok_y={y: self.sdelok for y in ys}, vyrozhdena=False)


def fabrika(sid):
    return World().evaluate, lambda p: True


def run(tmp_path, **kw):
    w = World()
    row = P.run_strategy("S9", w.evaluate, lambda p: True, out=tmp_path, run_id="t", setka=SETKA, **kw)
    return row, w


def csv(tmp_path, name):
    return pd.read_csv(tmp_path / f"S9_{name}.csv", comment="#")


def morris_ok(tmp_path):
    row, _ = run(tmp_path)
    m = csv(tmp_path, "morris")
    return m[m.rang == 1].os.item() == "a" and m[m.rang == 2].os.item() == "b" and row["N_morris"] == 20 * (len(AXES) + 1)


def test_morris_vazhnaya_os_pervoy_r20(tmp_path):
    assert morris_ok(tmp_path)


def test_mutaciya_morris_r_ne_uchten_lovitsya(tmp_path, monkeypatch):
    orig = sd.morris_screen
    monkeypatch.setattr(sd, "morris_screen", lambda ev, g, per, n_paths, **kw: orig(ev, g, per, n_paths=1, **kw))
    assert not morris_ok(tmp_path)


def test_mutaciya_morris_ranzhir_naoborot_lovitsya(tmp_path, monkeypatch):
    orig = sd.morris_screen

    def inv(*a, **kw):
        r = orig(*a, **kw)
        return dict(r, effects=dict(reversed(list(r["effects"].items()))))
    monkeypatch.setattr(sd, "morris_screen", inv)
    assert not morris_ok(tmp_path)


def plan_ok():
    s1 = ["k_s", "R", "tau", "q_sily", "sigma"]          # 7·6·7·5·4 = 5880 > 5000 (толкование #345) -> 4 оси
    ax = {"k_s": range(7), "R": range(6), "tau": range(7), "q_sily": range(5), "sigma": range(4), "k_tr": range(4), "bez": range(2)}
    p4, p5 = P.choose_axes(s1, ax), P.choose_axes(["k_s", "R", "k_tr", "sigma", "bez"], ax)
    return p4 == s1[:4] and len(p5) == 5 and all(math.prod(len(ax[a]) for a in p) <= 5000 for p in (p4, p5))


def test_faktor_ne_vyhodit_za_5000_pyat_osey_ili_chetyre():
    assert plan_ok()


def test_mutaciya_potolok_snyat_lovitsya(monkeypatch):
    orig = P.choose_axes
    monkeypatch.setattr(P, "choose_axes", lambda r, a, potolok=P.POTOLOK: orig(r, a, 10 ** 9))
    assert not plan_ok()


def test_faktor_end_to_end_ne_bolshe_potolka(tmp_path):
    row, _ = run(tmp_path)
    assert row["N_faktor"] <= 5000 and len(row["osi_plana"].split()) in (4, 5)
    assert len(csv(tmp_path, "faktor")) == row["N_faktor"]


def center_ok(tmp_path):
    row, _ = run(tmp_path)
    return (row["p_a"], row["p_b"]) == (3, 2) and row["obj_min_msr"] < 2.0     # пик (a=0, b=4) даёт 4,95


def test_centr_plato_a_ne_pik(tmp_path):
    assert center_ok(tmp_path)


def test_mutaciya_vybor_po_piku_lovitsya(tmp_path, monkeypatch):
    def peak(fact, k=3, **kw):
        return sorted(({**r, "plateau": r["objective"], "neighbors": 9, "edge": False} for r in fact["rows"]),
                      key=lambda r: -r["objective"])[:k]
    monkeypatch.setattr(sd, "pick_by_plateau", peak)
    monkeypatch.setattr(P, "TOPK", 1)
    assert not center_ok(tmp_path)


def morris19_ok(tmp_path):
    _, w = run(tmp_path, morris_2019=True)
    k = w.calls.index("train")
    return k >= 20 and set(w.calls[:k]) == {"ctrl2019"} and csv(tmp_path, "itog").morris_period.item() == "ctrl2019"


def test_morris_2019_ne_trogaet_2018(tmp_path):
    assert morris19_ok(tmp_path)
    row, w = run(tmp_path / "n")
    assert w.calls[0] == "train" and row["morris_period"] == "train"


def test_mutaciya_flag_morris_2019_ignoriruetsya_lovitsya(tmp_path, monkeypatch):
    orig = sd.morris_screen
    monkeypatch.setattr(sd, "morris_screen", lambda ev, g, per, **kw: orig(ev, g, "train", **kw))
    assert not morris19_ok(tmp_path)


def test_dsr_N_vse_tochki_i_spravochno(tmp_path, monkeypatch):
    row, _ = run(tmp_path)
    assert row["N_dsr"] == row["N_unikalnykh"] >= row["N_do_dosheta"] and row["dsr"] < row["psr"]
    orig = P.psr_dsr
    monkeypatch.setattr(P, "psr_dsr", lambda x, M, n: orig(x, M, 1))                    # мутация: N = 1
    row1, _ = run(tmp_path / "m")
    assert not row1["dsr"] < row1["psr"]


def test_main_jobs_ne_bolshe_6_i_tri_vyhoda(tmp_path, monkeypatch):
    seen = []

    class Pool:
        def __init__(self, max_workers):
            seen.append(max_workers)

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def map(self, f, it):
            return [f(x) for x in it]
    monkeypatch.setattr(P, "ProcessPoolExecutor", Pool)
    y = dict(signal={k: dict(ishodnoe=AXES[k][0], setka=AXES[k]) for k in ("a", "b")},
             obshchie={k: dict(setka=AXES[k], yakor=YAKOR[k]) for k in AXES if k not in ("a", "b")}, morris=dict(r=20, seed=1))
    (tmp_path / "zh047_S9.yaml").write_text(yaml.safe_dump(y), encoding="utf-8")
    out = tmp_path / "out"
    rows = P.main(["--strategii", "S9", "--fabrika", f"{__name__}:fabrika", "--jobs", "99", "--out", str(out), "--setki", str(tmp_path)])
    assert seen == [6] and rows[0]["strategiya"] == "S9"
    assert all((out / f"S9_{n}.csv").read_text(encoding="utf-8").startswith("# статус") for n in ("morris", "faktor", "itog"))


@pytest.mark.parametrize("sid,k", [("S1", 15), ("S2", 14), ("S3", 13), ("S4", 14), ("S5", 15), ("S6", 16)])
def test_setki_zh047(sid, k):
    S = P.load_setka(sid)
    y = yaml.safe_load((P.SETKI / f"zh047_{sid}.yaml").read_text(encoding="utf-8"))
    gen = {a: S["axes"][a] for a in y["obshchie"]}
    sig = math.prod(len(S["axes"][a]) for a in y["signal"])
    assert len(S["axes"]) == k and len(gen) == 12 and math.prod(len(v) for v in gen.values()) == 22579200
    assert y["morris"]["N"] == y["morris"]["r"] * (k + 1) == S["r"] * (k + 1) and S["seed"] == 20261027
    assert y["N_polnyy_kub"] == sig * 22579200
    for a, v in y["signal"].items():
        g = v["setka"]
        if len(g) == 7 and all(isinstance(x, (int, float)) for x in g):                  # ±50 %, 7 точек
            assert abs(g[0] - 0.5 * v["ishodnoe"]) <= 0.51 and abs(g[-1] - 1.5 * v["ishodnoe"]) <= 0.51, (sid, a)
