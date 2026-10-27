"""Шаг 4 разработки главы 4 (v3): сравнение прогнозов (H2), 2018–2021. Чистые функции; движок — `s9.Engine`.

Точка C — ключ s9: ("C1:W252_p70", "hyb_h5") = (семья:метка, источник σ). Затвор — только из `fastbt.shag_c`
(PR #270); без флага `--razreshayu-shag4` и пока его нет — отказ. Определение: C = A × затвор.
Лестница, «лучший с лучшим», связка с главой 3 — на готовых рядах/матрицах [тикер, бар], движок не нужен.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from fastbt import shag0
from fastbt.msp import sharpe
from fastbt.s9_stats import lw_sharpe_diff

YEARS = shag0.YEARS
POMETKA = shag0.POMETKA
COMMS = (0.00035, 0.0004)                     # Ъ 12:05: обе — основные колонки; H2 принимается, если держится на обеих
SIGMAS = ("naive", "harj", "hyb")
NAMES = ("C_naive", "C_harj", "C_hyb")
STEPS = ("A", "к3") + NAMES
NAN = float("nan")
NOTE = "C = A × затвор; ступень к3 → C смешивает эффект S5/S6 и фильтра; разложение — shagN_razlozhenie"


def parse_label(label: str):
    """'W252_p70' → (252, 0.70); 'W252_p70_90' → (252, (0.70, 0.90))."""
    w, *ps = label.split("_")
    p = [int(x.lstrip("p")) / 100 for x in ps]
    if not w.startswith("W") or len(p) not in (1, 2):
        raise ValueError(f"метка точки не разобрана: {label}")
    return int(w[1:]), (p[0] if len(p) == 1 else tuple(p))


def src_of(sigma: str, h: str) -> str:
    return f"{sigma}_{h}"


def _has_shag_c() -> bool:
    import importlib.util
    return importlib.util.find_spec("fastbt.shag_c") is not None


def shag_c(razreshayu: bool):
    """Модуль затвора C (PR #270). Отказ: без флага --razreshayu-shag4 или пока fastbt/shag_c.py нет в репозитории."""
    if not razreshayu:
        raise PermissionError("реальный запуск шага 4 закрыт: нужен флаг --razreshayu-shag4")
    if not _has_shag_c():
        raise RuntimeError("fastbt/shag_c.py нет в репозитории (PR #270 не слит): затвор и портфель не из чего брать")
    from fastbt import shag_c as mod
    return mod


def gate(eng, ckey: str, src: str, razreshayu: bool = False) -> np.ndarray:
    """G[3, тикер, бар] точки ckey на σ-источнике src — из fastbt.shag_c.gates (своего кода затвора здесь нет).
    C* на наивной σ не строится (ValueError)."""
    C = shag_c(razreshayu)
    fam, label = ckey.split(":", 1)
    parse_label(label)
    if fam.startswith("C*") and src.startswith("naive"):
        raise ValueError("C* не строится на наивной σ")
    cache = eng.__dict__.setdefault("_shag4_gates", {})
    if src not in cache:
        cache[src] = C.gates(eng, src, C.load_grid(), src.startswith("naive"))
    try:
        return cache[src][(fam, label)]
    except KeyError:
        raise ValueError(f"точки {ckey} нет в сетке shag_c для {src}") from None


def portfel(eng, ckey: str, src: str, comm: float, razreshayu: bool = False) -> pd.DataFrame:
    """Дневной ряд портфеля точки ckey на σ src: A × затвор (затвор — fastbt.shag_c), комиссия comm."""
    return eng.port(eng.A_row * eng.g_row(gate(eng, ckey, src, razreshayu)), comm=comm)


def compare(p1: pd.DataFrame, p0: pd.DataFrame, reps: int) -> dict:
    """Δ Шарпа p1 − p0 (разность Шарпов, поэтому ступени лестницы складываются), p ЛВ (блок 5), знак по годам."""
    x1, x0 = p1["excess"].values, p0["excess"].values
    out = dict(sharpe=sharpe(x1), sharpe_baza=sharpe(x0), d_sharpe=sharpe(x1) - sharpe(x0),
               p_lw=lw_sharpe_diff(x1, x0, reps, 5)[2])
    n = 0
    for y in YEARS:
        m = shag0.year_mask(p1, y)
        dy = sharpe(x1[m]) - sharpe(x0[m])
        out[f"znak_{y}"] = "+" if dy > 0 else "−"
        n += dy > 0
    out["n_let_plyus"] = int(n)
    return out


def _row(comm, **kw) -> dict:
    return dict(pometka=POMETKA, comm_pct=round(comm * 100, 6), **kw)


def luchshiy(P: dict, reps: int) -> list:
    """4.A. P[comm][имя точки] → ряд. 3 × 3: клетка (a, b) = Δ(a − b); диагональ — Шарп точки."""
    rows = []
    for comm, ports in P.items():
        for a in NAMES:
            for b in NAMES:
                if a == b:
                    rows.append(_row(comm, a=a, b=b, sharpe=sharpe(ports[a]["excess"].values)))
                else:
                    rows.append(_row(comm, a=a, b=b, **compare(ports[a], ports[b], reps)))
    return rows


def odna_tochka(P: dict, A: dict, reps: int) -> list:
    """4.B. P[comm][(имя точки, σ)] → ряд или None (C* на наивной); A[comm] — ряд A. Точка × σ, Δ к A."""
    rows = []
    for comm, ports in P.items():
        for (nm, sg), p in ports.items():
            r = _row(comm, tochka=nm, sigma=sg)
            rows.append({**r, "primechanie": "C* на наивной σ не строится"} if p is None
                        else {**r, **compare(p, A[comm], reps)})
    return rows


def lestnitsa(P: dict, reps: int) -> list:
    """4.C. P[comm][STEPS] → ряд. Каждая ступень: Δ к предыдущей, p, знаки по годам; d_k_A — накопленная Δ к A."""
    rows = []
    for comm, ports in P.items():
        sA = sharpe(ports["A"]["excess"].values)
        rows.append(_row(comm, stupen="A", sharpe=sA, d_k_A=0.0, primechanie=NOTE))
        for prev, cur in zip(STEPS[:-1], STEPS[1:]):
            c = compare(ports[cur], ports[prev], reps)
            rows.append(_row(comm, stupen=f"{prev} → {cur}", **c, d_k_A=c["sharpe"] - sA))
    return rows


def check_lestnitsa(rows: list, tol: float = 1e-9) -> None:
    """Сумма Δ по ступеням = итог − A (по каждой комиссии); иначе AssertionError."""
    for comm in {r["comm_pct"] for r in rows}:
        rs = [r for r in rows if r["comm_pct"] == comm]
        tot = sum(r["d_sharpe"] for r in rs if "d_sharpe" in r)
        assert abs(tot - rs[-1]["d_k_A"]) < tol, f"лестница не сложилась при {comm}%: {tot} ≠ {rs[-1]['d_k_A']}"


def check_odna_tochka(P_o: dict, P_l: dict) -> None:
    """В 4.B клетка (точка, своя σ) обязана быть тем же рядом, что точка в 4.A/4.C — иначе σ перепутана."""
    for comm, ports in P_o.items():
        for nm, sg in zip(NAMES, SIGMAS):
            own = ports[(nm, sg)]
            assert own is not None and np.array_equal(own["excess"].values, P_l[comm][nm]["excess"].values), \
                f"4.B: клетка {nm} × {sg} не равна ряду точки {nm} при {comm * 100}%"


