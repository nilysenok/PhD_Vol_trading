# -*- coding: utf-8 -*-
"""progon_ustojch.py — Л-10: LOYO, LOTO, бутстрап по кварталам, тикерная поправка (прогон этапа 1, справочно).
Только ряды ledger (Chitatel.matrica / .tik) и vybor_region из Л-7; движок не считает. Исключение — поправка: точки
«кандидат × vhod × tau» считает переданная evaluate_fn (EvaluatorV2, Л-5), они идут в N с istochnik = tiker.
Поправка — выбор на тикер, §4 п. 3: в A′ и в кандидаты не идёт (❓2). Шапка таблиц: разработка 2016–2019, потолок."""
from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np

from fastbt.metriki_e import sr_dnevnoy
from fastbt.progon_region import NK, _kv_sr, ryad_regiona, vybor_region

SEED, N_BOOT, POROG, MIN_GODOV = 20261028, 10000, 0.2, 3
GODY = (2016, 2017, 2018, 2019)
SHAPKA = "# разработка 2016–2019, потолок"


def _tsentr(res, rows) -> tuple:
    """Центр региона как ((ось, индекс уровня), …) — сравним между фолдами, где ведущие оси могут отличаться."""
    return tuple(sorted(zip(res["osi"], res["centr"]))) if res else ()


def loyo(M, rows, daty) -> list:
    """4 фолда: регион выбран на 3 годах, ряд региона смотрится на оставленном годе. Строка: god, tsentr, sovpal, mediana_kv_goda."""
    M, daty = np.asarray(M, float), np.asarray(daty).astype("datetime64[D]")
    polnyj = _tsentr(vybor_region(M, rows, daty), rows)
    out = []
    for i, g in enumerate(GODY):
        ostav = list(range(4 * i, 4 * i + 4))
        res = vybor_region(M, rows, daty, [q for q in range(NK) if q not in ostav])
        med = float("nan")
        if res:
            sr = _kv_sr(ryad_regiona(M, res["chleny"])[:, None], daty, ostav)[0]
            med = float(np.median(sr)) if not np.isnan(sr).any() else float("nan")
        ts = _tsentr(res, rows)
        out.append(dict(god=g, tsentr=ts, sovpal=bool(res) and ts == polnyj, mediana_kv_goda=med))
    return out


def summy_tik(tik, chunk=512) -> tuple:
    """(сумма, число) по тикерам с учётом nan: форма N × T, float64. tik — N × 17 × T."""
    n, _, t = tik.shape
    s, c = np.zeros((n, t)), np.zeros((n, t))
    for a in range(0, n, chunk):
        x = np.asarray(tik[a:a + chunk], dtype=np.float64)
        s[a:a + chunk], c[a:a + chunk] = np.nansum(x, axis=1), np.isfinite(x).sum(axis=1)
    return s, c


def portfel_bez(tik, s, c, i, chunk=512) -> np.ndarray:
    """T × N: nanmean по тикерам без i. Знаменатель — число оставшихся тикеров с данными на дату (nan — если их нет)."""
    n, _, t = tik.shape
    out = np.full((t, n), np.nan)
    for a in range(0, n, chunk):
        x = np.asarray(tik[a:a + chunk, i], dtype=np.float64)
        ss, cc = s[a:a + chunk] - np.nan_to_num(x), c[a:a + chunk] - np.isfinite(x)
        with np.errstate(divide="ignore", invalid="ignore"):
            out[:, a:a + chunk] = np.where(cc > 0, ss / np.where(cc > 0, cc, 1), np.nan).T
    return out


def loto(tik, rows, daty, tikery) -> list:
    """Для каждого тикера: портфель всех точек без него, выбор региона повторён. Строка: tiker, tsentr, sovpal, celevaya."""
    daty = np.asarray(daty).astype("datetime64[D]")
    s, c = summy_tik(tik)
    polnyj = _tsentr(vybor_region(np.where(c > 0, s / np.where(c > 0, c, 1), np.nan).T, rows, daty), rows)
    out = []
    for i, name in enumerate(tikery):
        res = vybor_region(portfel_bez(tik, s, c, i), rows, daty)
        ts = _tsentr(res, rows)
        out.append(dict(tiker=name, tsentr=ts, sovpal=bool(res) and ts == polnyj, celevaya=res["score"] if res else float("nan")))
    return out


def butstrap(ryad, daty, n=N_BOOT, seed=SEED) -> dict:
    """Повтор с возвращением 16 кварталов (блоки — кварталы ряда региона). Выход: 5/50/95 % медианы и доли + кварталов."""
    sr = _kv_sr(np.asarray(ryad, float)[:, None], np.asarray(daty).astype("datetime64[D]"), list(range(NK)))[0]
    idx = np.random.default_rng(seed).integers(0, NK, size=(n, NK))
    v = sr[idx]
    med, dol = np.median(v, axis=1), (v > 0).mean(axis=1)
    p = lambda x: [float(q) for q in np.nanpercentile(x, [5, 50, 95])] if np.isfinite(x).any() else [math.nan] * 3  # noqa: E731
    return dict(mediana_p5_50_95=p(med), dolya_p5_50_95=p(dol), n=n, seed=seed)


def sr_god(x, daty, god) -> float:
    """Шарп годовой шкалы (ddof=1) ряда x за календарный год; nan-дни выброшены; меньше 2 дней — nan."""
    x = np.asarray(x, float)
    x = x[(np.asarray(daty).astype("datetime64[Y]").astype(int) + 1970 == god) & np.isfinite(x)]
    return float(sr_dnevnoy(x) * math.sqrt(252)) if len(x) > 1 else float("nan")


def vybrat(vyigrysh, gody_gain) -> bool:
    """Берётся вариант, только если выигрыш ≥ POROG (нестрого) и его знак (+) совпал в ≥ MIN_GODOV годах из 4."""
    return bool(vyigrysh >= POROG and sum(1 for g in gody_gain if g > 0) >= MIN_GODOV)


def _sr_all(v) -> float:
    v = v[np.isfinite(v)]
    return float(sr_dnevnoy(v) * math.sqrt(252)) if len(v) > 1 else float("nan")


def tiker_popravka(kandidat: dict, tikery, daty, evaluate_fn, vhod_znach, tau_znach) -> tuple:
    """evaluate_fn(pt) -> ex_tik (17 × T, ряды excess тикеров точки). Точки: кандидат × vhod × tau (кроме самого кандидата).
    Возврат: (строки по тикерам: tiker, vybor, vyigrysh, gody; точки с istochnik = tiker — для N)."""
    base = _sr_tik_all(evaluate_fn(kandidat), daty)
    var, tochki = {}, []
    for v in vhod_znach:
        for t in tau_znach:
            pt = {**kandidat, "vhod": v, "tau": t}
            if pt == kandidat:
                continue
            tochki.append(dict(pt=pt, istochnik="tiker"))
            var[(v, t)] = _sr_tik_all(evaluate_fn(pt), daty)
    out = []
    for i, name in enumerate(tikery):
        b_all, b_y = base[i]
        best = max(var.items(), key=lambda kv: (np.nan_to_num(kv[1][i][0] - b_all, nan=-np.inf)), default=None)
        row = dict(tiker=name, vybor="kandidat", vyigrysh=0.0, gody=0)
        if best is not None:
            vy = best[1][i][0] - b_all
            gg = [a - b for a, b in zip(best[1][i][1], b_y)]
            row["gody"] = sum(1 for g in gg if g > 0)
            if math.isfinite(vy) and vybrat(vy, [g for g in gg if math.isfinite(g)]):
                row.update(vybor=f"vhod={best[0][0]};tau={best[0][1]}", vyigrysh=float(vy))
            else:
                row["vyigrysh"] = float(vy) if math.isfinite(vy) else float("nan")
        out.append(row)
    return out, tochki


def _sr_tik_all(ex_tik, daty) -> list:
    return [(_sr_all(np.asarray(ex_tik[i], float)), [sr_god(ex_tik[i], daty, g) for g in GODY]) for i in range(len(ex_tik))]


def zapisat_csv(path, rows: list) -> None:
    """Таблица с шапкой SHAPKA; кортежи и списки в ячейках — строкой."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        f.write(SHAPKA + "\n")
        w = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else [])
        w.writeheader()
        w.writerows(rows)
