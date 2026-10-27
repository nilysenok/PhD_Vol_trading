"""Шаги 1–3 главы 4 (v3): сетка C на одной σ (наивная √RV_w, HAR-J (б), гибрид), 2018–2021.

Чистые функции над движком `s9.Engine` (нужны `sigma`, `rvw`, `n_t`, `g_row`, `port`, `A_row`, `pairs`, `r`, `first`,
`lr_row`, `dates`, `cnt`, `rf`, `kr`, `cal`) — поэтому проверяются на синтетическом движке (tests/test_s9_shag_c.py).
Сетка — fastbt/designs/s9/setka_C_v3.yaml; затворы — `s9.c_dop_gates` (ранг по окну [d−W, d−1], текущий день не входит);
к2 — `fast_kontrol` (seed 20261027); к3 — `shag0.zero_sids` по семьям из k3.yaml; Δ и ЛВ — `fast_boot` (= s9_stats).
Колонки комиссии — суффикс `_035`, `_040`, `_050` (0,035 / 0,04 / 0,05 %). Таблицы и отбор — fastbt/shag_c_tab.py.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from fastbt import fast_boot
from fastbt import fast_kontrol as fk
from fastbt import s9, shag0

GRID = Path(__file__).resolve().parent / "designs" / "s9" / "setka_C_v3.yaml"
COMM_MAIN = (0.00035, 0.0004)          # основные колонки
COMM_SEL, COMM_REF = 0.0004, 0.0005    # отбор по С8 v2 — при 0,04; справочно — при 0,05
SEED_K2 = 20261027
ALPHA = 0.05
SRC = {"naive": {1: "naive_h1"}, "harj": {1: "harj_h1", 5: "harj_h5"}, "hyb": {1: "hyb_h1", 5: "hyb_h5"}}
SHAG = {"naive": 1, "harj": 2, "hyb": 3}


def tag(c: float) -> str:
    return f"{round(c * 1e5):03d}"


def load_grid(path=GRID) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def plan(sigma: str, grid: dict) -> list:
    """Все точки сетки на σ (без чтения данных): h, src, family, W, p1, p2 (0 у фильтра), label."""
    out = []
    for h in grid["sigmy"][sigma]["h"]:
        for fam, f in grid["semyi"].items():
            if fam == "C*" and sigma == "naive":        # σ̂/√RV_w ≡ 1: вырождена (yaml: isklyucheno)
                continue
            for W in grid["okna"]:
                base = dict(h=h, src=SRC[sigma][h], family=fam, W=W)
                out += [dict(base, p1=p, p2=0, label=f"W{W}_p{p}") for p in f.get("filtr", [])]
                out += [dict(base, p1=a, p2=b, label=f"W{W}_p{a}_{b}") for a, b in f.get("chastichnaya", [])]
    return out


def gate_spec(grid: dict, with_rvw: bool) -> dict:
    semi = {}
    for fam, f in grid["semyi"].items():
        if f["signal"] == "sigma_over_rvw" and not with_rvw:
            continue
        for kind in ("filtr", "chastichnaya"):
            if f.get(kind):
                semi[f"{fam}|{kind}"] = dict(signal=f["signal"], okna=grid["okna"], **{kind: f[kind]})
    return dict(min_periods=grid["min_periods"], semi=semi)


def gates(eng, src: str, grid: dict, naive: bool) -> dict:
    """{(семья, метка): G[3, тикер, бар]} для σ-источника src; ранг — по окну без текущего дня."""
    S = eng.sigma(src)
    rv = None if naive else eng.rvw()
    out = {}
    for fam, pts in s9.c_dop_gates(S, rv, eng.n_t, gate_spec(grid, rv is not None)):
        for lab, G in pts:
            out[(fam.split("|")[0], lab)] = G
    return out


def znaki(p1, p0) -> dict:
    out = {}
    for y in shag0.YEARS:
        m = shag0.year_mask(p1, y)
        d = s9.sharpe(p1["excess"].values[m]) - s9.sharpe(p0["excess"].values[m])
        out[f"znak_{y}"] = "+" if d > 0 else "−"
    return out


def lw(p1, p0, reps: int) -> tuple:
    """(Δ Шарпа, p Ледуа–Вольфа, блок 5)."""
    d, _, pv = fast_boot.lw_sharpe_diff(p1["excess"].values, p0["excess"].values, reps, 5)
    return d, pv


class Base:
    """A, к3 и контроли к2 на одной комиссии."""

    def __init__(self, eng, comm: float, n_shift: int = fk.N_SHIFT):
        self.eng, self.comm = eng, comm
        self.A = np.asarray(eng.A_row, float)
        self.k3 = shag0.zero_sids(self.A, shag0.sid_row(eng), fk.k3_sids())
        self.pA, self.pK = eng.port(self.A, comm=comm), eng.port(self.k3, comm=comm)
        self.mA, self.mK = s9.metrics(self.pA), s9.metrics(self.pK)
        self.kon = fk.Kontrol(eng, comm, n_shift=n_shift, seed=SEED_K2)
        if self.kon.t_tick.min() >= 2 * fk.MIN_LAG and self.kon.LAG.min() < fk.MIN_LAG:
            raise ValueError("к2: лаг сдвига меньше 21 (сдвиг 0 даёт исходную маску)")


def evaluate(b: Base, g: np.ndarray, reps: int = 10000, light: bool = False) -> dict:
    """Метрики точки A·g на комиссии b.comm; условия С8 v2 по отдельности (§0.5): 1 — прежний С8,
    2 — Шарп > Шарп к3, 3 — p к к2 < 0,05. light — без бутстрапа и доп. метрик (справочный отбор)."""
    p = b.eng.port(b.A * g, comm=b.comm)
    m = s9.metrics(p)
    b.kon.check(g)
    sh2 = b.kon.k2_sharpes(g)
    p2 = fk.p_perm(m["sharpe"], sh2)
    c1, c2, c3 = bool(s9.c8(m, b.mA)["passed"]), bool(m["sharpe"] > b.mK["sharpe"]), bool(p2 < ALPHA)
    r = dict(sharpe=m["sharpe"], sharpe_A=b.mA["sharpe"], sharpe_k3=b.mK["sharpe"], p_vs_k2=p2,
             sharpe_k2_med=float(np.median(sh2)), usl1_c8=c1, usl2_k3=c2, usl3_k2=c3, proshla_c8v2=c1 and c2 and c3)
    if light:
        return r
    r.update(shag0.stats(p, np.ones(len(p), bool)))
    r["d_vs_A"], r["p_vs_A"] = lw(p, b.pA, reps)
    r.update(znaki(p, b.pA))
    r["d_vs_k3"], r["p_vs_k3"] = lw(p, b.pK, reps)
    return r


def point_row(eng, bases: dict, pt: dict, G: np.ndarray, sigma: str, ref: float, reps: int) -> dict:
    g = eng.g_row(G)
    row = dict(pometka=shag0.POMETKA, sigma=sigma, point=f"{pt['family']}:{pt['label']}", **pt)
    row.update(eng.gate_stats(g))
    for c, b in bases.items():
        row.update({f"{k}_{tag(c)}": v for k, v in evaluate(b, g, reps, light=(c == ref)).items()})
    return row


def run_setka(eng, sigma: str, grid: dict, reps: int = 10000, comms=COMM_MAIN, ref=COMM_REF, log=print,
              n_shift: int = fk.N_SHIFT):
    """Все точки сетки на σ: DataFrame (основные комиссии — полно, ref — только условия отбора) и словарь затворов."""
    bases = {c: Base(eng, c, n_shift) for c in (*comms, ref)}
    rows, gs = [], {}
    pts = plan(sigma, grid)
    for src in dict.fromkeys(pt["src"] for pt in pts):
        G = gates(eng, src, grid, sigma == "naive")
        for pt in (q for q in pts if q["src"] == src):
            gs[(pt["h"], pt["family"], pt["label"])] = G[(pt["family"], pt["label"])]
            rows.append(point_row(eng, bases, pt, gs[(pt["h"], pt["family"], pt["label"])], sigma, ref, reps))
        log(f"{src}: {len(rows)} точек")
    return pd.DataFrame(rows), gs, bases
