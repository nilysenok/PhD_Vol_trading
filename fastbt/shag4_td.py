"""Шаг 4, 4.D: QLIKE-выигрыш гибрида по тикер-дням (Ъ 27.10 15:10, определение (в)).
Тикер-день открыт, если ранг σ̂ точки (past_rank в окне W, текущий день не входит) R < 0,5; R ≥ 0,5 — закрыт
(R = 0,5 ровно — закрыт, считается отдельно); ранг не определён (разогрев окна) — вне групп, считается.
При p50 это состояние затвора контртренда G[1]; затвор тренда — его дополнение.
Выигрыш = QLIKE(база) − QLIKE(гибрид), > 0 — гибрид лучше. DM — по дневному ряду средней по тикерам разности
потерь внутри группы, NW-лаг h−1, нормальное приближение (как глава 3, d24_6.dm). Чистые функции над [тикер, бар]."""
from __future__ import annotations

import math

import numpy as np

from fastbt.shag4 import NAN, POMETKA, YEARS

MINP = {126: 50, 252: 50, 504: 252}                  # min_periods из setka_C_v3.yaml
MIN_DAYS = 30                                        # меньше дней в группе — p не считаем
THRESH = 0.5


def rank_matrix(S: np.ndarray, n_t, W: int, minp: int | None = None) -> np.ndarray:
    """Ранг σ̂ в окне [d−W, d−1] — тот же past_rank, по которому строится затвор C1."""
    from fastbt.s9_stats import past_rank
    R = np.full(S.shape, np.nan)
    for ti in range(S.shape[0]):
        n = int(n_t[ti])
        R[ti, :n] = past_rank(np.ascontiguousarray(S[ti, :n]), int(W), int(minp or MINP[W]))
    return R


def open_by_rank(R: np.ndarray):
    """→ (определён, открыт): открыт строго при R < 0,5; R = 0,5 — закрыт."""
    defined = np.isfinite(R)
    with np.errstate(invalid="ignore"):
        return defined, defined & (R < THRESH)


def ql_el(y, f):
    """Поэлементный QLIKE главы 3 (d26_lib.qlike: f снизу 1e-12)."""
    with np.errstate(all="ignore"):
        r = y / np.clip(f, 1e-12, None)
        return r - np.log(r) - 1


def dm_nw(days: np.ndarray, d: np.ndarray, lag: int):
    """(p, число дней): d усредняется по тикерам внутри дня; NW(lag); p двусторонний, нормальный."""
    u, inv = np.unique(days, return_inverse=True)
    ds = np.bincount(inv, d) / np.bincount(inv)
    n = len(u)
    if n < MIN_DAYS:
        return NAN, n
    e = ds - ds.mean()
    v = e @ e / n + 2 * sum((1 - k / (lag + 1)) * (e[k:] @ e[:-k]) / n for k in range(1, lag + 1))
    if v <= 0:
        return NAN, n
    return math.erfc(abs(ds.mean() / math.sqrt(v / n)) / math.sqrt(2)), n


def qlike_td(y, f_base, f_hyb, groups: dict, days, years, h: int, half=None, undefined=None, **meta) -> list:
    """Строки по периодам (2018…2021 и «2018–2021») × группам (groups: имя → маска; «все» — объединение).
    half — маска R = 0,5; undefined — маска «ранг не определён» (только счёт)."""
    fin = np.isfinite(f_base) & np.isfinite(f_hyb)
    ok = fin & np.isfinite(y) & (y > 1e-12)
    nan_y = fin & np.isnan(y)
    Lb, Lh = ql_el(y, f_base), ql_el(y, f_hyb)
    union = np.any(list(groups.values()), axis=0)
    half = np.zeros_like(union) if half is None else half
    undefined = np.zeros_like(union) if undefined is None else undefined
    rows = []
    for per in [*YEARS, "2018–2021"]:
        pm = (years == per) if per != "2018–2021" else np.isin(years, YEARS)
        n_all = int((ok & union & pm).sum())
        for lab, gm in {**groups, "все": union}.items():
            m = ok & gm & pm
            n = int(m.sum())
            qb, qh = (float(Lb[m].mean()), float(Lh[m].mean())) if n else (NAN, NAN)
            p, nd = dm_nw(days[m], (Lb - Lh)[m], h - 1) if n else (NAN, 0)
            rows.append(dict(pometka=POMETKA, **meta, h=h, period=str(per), gruppa=lab, n=n,
                             dolya=n / n_all if n_all else NAN, n_dney=nd, n_nan_celi=int((nan_y & gm & pm).sum()),
                             n_R_050=int((half & gm & pm).sum()), n_bez_ranga=int((undefined & ok & pm).sum()),
                             qlike_baza=qb, qlike_gibrid=qh, vyigrysh=qb - qh,
                             otn_vyigrysh_pct=100 * (qb - qh) / qb if n and qb else NAN, p_dm=p,
                             primechanie=f"мало дней (<{MIN_DAYS}), p не считается" if n and nd < MIN_DAYS else ""))
    return rows


REZ = "4.D тикер-день: ранг σ̂ точки < 0,5 — открыт"
BASES = (("HAR-J (б)", "harj"), ("наивная", "naive"))


def qlike_rows(eng, T: dict, valid, Y, D, target, razr: bool = False) -> tuple:
    """(основное, справочное). Основное — ранг σ̂ C_hyb (h=5). Справочно — C_hyb на h=1, C_harj и C_naive (h=1; ранг
    их σ̂, гибрид против баз на том же h) и портфельные определения (затвор C_hyb h=5). target(h) → цель [тикер, бар]."""
    from fastbt import shag4
    main, ref = [], []
    ck, src = T["C_hyb"]
    variants = [(main, "C_hyb", ck, src), (ref, "C_hyb h=1", ck, shag4.src_of("hyb", "h1")),
                (ref, "C_harj", *T["C_harj"]), (ref, "C_naive", *T["C_naive"])]
    for out, nm, pk, psrc in variants:
        h = psrc[-2:]
        R = rank_matrix(eng.sigma(psrc), eng.n_t, shag4.parse_label(pk.split(":", 1)[1])[0])
        defined, op = open_by_rank(R)
        groups = {"открыт": valid & op, "закрыт": valid & defined & ~op}
        f = lambda sg: eng.sigma(shag4.src_of(sg, h)) ** 2                # noqa: E731
        for bn, base in BASES:
            out.extend(qlike_td(target(h), f(base), f("hyb"), groups, D, Y, int(h[1:]), half=valid & (R == THRESH),
                                undefined=valid & ~defined, rezhim=REZ, tochka=f"{nm}: {pk} ({psrc})", baza=bn))
    h, r_ = src[-2:], eng.r
    f = lambda sg: eng.sigma(shag4.src_of(sg, h)) ** 2                    # noqa: E731
    G = shag4.gate(eng, ck, src, razr)
    for dn, gr in portfolio_groups(eng.A_row, G, r_["tick"], r_["bar"], r_["cat"], valid.shape, valid).items():
        for bn, base in BASES:
            ref.extend(qlike_td(target(h), f(base), f("hyb"), gr, D, Y, int(h[1:]), rezhim=f"4.D портфельное: {dn}",
                                tochka=f"C_hyb: {ck} ({src})", baza=bn))
    return main, ref


def portfolio_groups(A, G, tick, bar, cat, shape, valid) -> dict:
    """Справочные портфельные определения: имя → {открыт, закрыт} (маски [тикер, бар]); G[3, тикер, бар]."""
    nz = np.asarray(A) != 0
    npos, npas = np.zeros(shape, int), np.zeros(shape, int)
    np.add.at(npos, (tick[nz], bar[nz]), 1)
    pz = nz & (G[cat, tick, bar] > 0)
    np.add.at(npas, (tick[pz], bar[pz]), 1)
    has = valid & (npos > 0)
    return {"хотя бы одна позиция A": dict(открыт=has & (npas > 0), закрыт=has & (npas == 0)),
            "все позиции против ни одной": dict(открыт=has & (npas == npos), закрыт=has & (npas == 0)),
            "затвор тренда": dict(открыт=valid & (G[0] > 0), закрыт=valid & ~(G[0] > 0))}
