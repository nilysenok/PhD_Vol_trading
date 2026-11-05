"""Цепочка Г3–Г5 на готовых прогнозах (Ж-197 п. 3): чистые функции над массивами одной сетки дат.

Вход `a` (prep): d, td (цель), rv (цель, RV на td), rv_w (RV_w на d), m (closed), st, z, month.
Варианты `v` для тестов утечки (допуски в GATES, этап 3): work | oracle | full | perm.
Всё, что выбирается по QLIKE за весь период (спокойный эксперт C, основа Г5), передаётся в `choice` и в тестах фиксируется.
"""
from __future__ import annotations

import numpy as np

from models_v2 import metrics
from models_v2.s3 import post

FUT = 252     # full: окно [k-251; k+252] по индексу прогноза k — включает цель и 252 будущих прогноза
SHIFT = 100   # perm: ряд RV циклически сдвинут на 100 шагов внутри оценки параметров
HL = (10, 20, 60)
G1 = ("g1_c050", "g1_c075", "g1_c100")
G5 = tuple("g5_%s_h%d" % (k, h) for k in ("always", "stress") for h in HL)


def prep(d, td, rv, rv_w, rvi_high) -> dict:
    d, td = np.asarray(d), np.asarray(td)
    q = post.q67_prior(np.asarray(rv_w, float))
    st = post.stress(rvi_high, rv_w, q)
    with np.errstate(invalid="ignore", divide="ignore"):
        z = np.log(rv_w) - np.log(q)
    return {"d": d, "td": td, "rv": np.asarray(rv, float), "rv_w": np.asarray(rv_w, float), "q": q, "st": st, "z": z,
            "m": post.closed_count(d, td), "month": d.astype("datetime64[M]").astype(int)}


def qmean(rv, ln_s) -> float:
    return float(metrics.qlike(rv, np.exp(ln_s)).mean())


def pick(names, ln, rv) -> str:
    """argmin QLIKE(all); при равенстве — первое имя (выбор на оценочном периоде)."""
    q = [qmean(rv, ln[n]) for n in names]
    return names[int(np.argmin(q))]


def _mfit(a, v):
    n = len(a["rv"])
    if v == "oracle":
        return np.arange(n) + 1, 1
    if v == "full":  # окно [k-251; k+252] по индексу прогноза k (m[k] = k при последовательной сетке)
        return np.minimum(np.arange(n) + FUT + 1, n), 2 * post.WIN
    return a["m"], post.WIN


def ewma_two_sided(rv, s2, halflife: float, clip=post.CLIP) -> np.ndarray:
    """full для Г5: двустороннее ядро 0,5^(|i−j|/h) на окне [j−251; j+252] (включает цель j и будущее)."""
    n, r, k = len(rv), rv / s2, np.empty(len(rv))
    for j in range(n):
        i = np.arange(max(0, j - post.WIN + 1), min(n, j + FUT + 1))
        w = 0.5 ** (np.abs(i - j) / halflife)
        k[j] = np.clip((w * r[i]).sum() / w.sum(), clip[0], clip[1])
    return k


def _rvfit(a, v):
    return np.roll(a["rv"], SHIFT) if v == "perm" else a["rv"]


def g3(a, ln, v="work"):
    m, win = _mfit(a, v)
    w = post.g3_weights(ln["b"], ln["h"], _rvfit(a, v), m, win)
    return w * ln["b"] + (1 - w) * ln["h"], w


def g4(a, ln, c, v="work"):
    """-> (ln жёсткая смесь, ln логистика, a, b)."""
    n, st = len(a["rv"]), a["st"]
    if v == "oracle":  # эксперт выбирается по QLIKE в точке d+1
        better_h = metrics.qlike(a["rv"], np.exp(ln["h"])) < metrics.qlike(a["rv"], np.exp(ln[c]))
        o = post.g4_hard(ln["h"], ln[c], better_h)
        return o, o, np.full(n, np.nan), np.full(n, np.nan)
    m = np.full(n, n) if v == "full" else a["m"]
    lg, pa, pb = post.g4_logit(ln["h"], ln[c], _rvfit(a, v), a["z"], st, m, a["month"])
    return post.g4_hard(ln["h"], ln[c], st), lg, pa, pb


def g5(a, s2, v="work") -> dict:
    out, st = {}, a["st"]
    for h in HL:
        if v == "oracle":
            k = np.clip(a["rv"] / s2, *post.CLIP)
        elif v == "full":
            k = ewma_two_sided(a["rv"], s2, float(h))
        else:
            k = post.ewma_k(_rvfit(a, v), s2, a["m"], float(h))
        out["g5_always_h%d" % h], out["g5_stress_h%d" % h] = post.g5(s2, k), post.g5(s2, k, st)
    return out


def build(a, ln, choice=None, v="work") -> tuple:
    """ln: {'h','b', G1…, 'g2'} (ln σ̂²). -> (sigma2 {модель: массив}, aux). choice {'C','base'} — иначе по QLIKE(all)."""
    ch, lw = choice or {}, dict(ln)
    lw["g3"], w = g3(a, lw)  # рабочие эксперты и основа Г5 — всегда work; вариант v меняет только оценку правила своей строки
    c = ch.get("C") or pick(list(G1) + ["g2", "g3"], lw, a["rv"])
    lw["g4_hard"], lw["g4_logit"], pa, pb = g4(a, lw, c)
    base = ch.get("base") or pick(list(G1) + ["g2", "g3", "g4_hard", "g4_logit"], lw, a["rv"])
    l3, w = (lw["g3"], w) if v == "work" else g3(a, lw, v)
    h4, lg4, pa, pb = (lw["g4_hard"], lw["g4_logit"], pa, pb) if v == "work" else g4(a, lw, c, v)
    sig = {"g3": np.exp(l3), "g4_hard": np.exp(h4), "g4_logit": np.exp(lg4)}
    sig.update(g5(a, np.exp(lw[base]), v))
    return sig, {"w": w, "a": pa, "b": pb, "choice": {"C": c, "base": base}}
