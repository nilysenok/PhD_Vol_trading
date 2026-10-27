"""Критерий «прошла» §3 (П1–П5), правило остановки Ж-054, ledger итераций и таблицы Ж-051. Чистые функции, без данных.

Проект: docs/otchety/2026-10-28_zh051_iteracii.md §3–§4. Контроли в `Ledger.cache` не пишутся — ни в N, ни в PBO.
"""
from __future__ import annotations

import hashlib
import math
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from fastbt import etap1_poisk_vorota as V

DELTA, PBO_SHAG, PBO_STOP, N_KONTROL = 0.10, 0.03, V.PBO_MAX, 20
SHAPKA = ("# статус: разработка, train 2018–2019; Ж-051; validation не открыта\n"
          "# PnL без дивидендов (находка #362): цены не поправлены, дивиденды не начисляются — цифры не для диссертации\n")
LED = Path("docs/tables/zh051/_ledger")


def _r(x: float) -> float:
    return round(float(x), 12)


def _gt(a: float, b: float) -> bool:                      # a ≥ b с допуском на плавающую точку; nan → ложь
    return math.isfinite(a) and math.isfinite(b) and _r(a) >= _r(b)


def kriteriy(d: dict) -> dict:
    """П1–П5 по §3. Вход: tsel_pi, tsel_p1, msr_pi{год}, msr_p1{год}, nb_tsel (список tsel соседей, без net),
    sdelok_pi{год}, s9_ok, g4_ok, pbo_do, pbo_posle, kontrol (список tsel 20 контролей Pi; None — не применимо)."""
    dpi = d["tsel_pi"] - d["tsel_p1"]
    p1 = _gt(dpi, DELTA)
    p2 = all(_gt(d["msr_pi"][y], d["msr_p1"][y] - DELTA) for y in (2018, 2019))
    nb = [x - d["tsel_p1"] for x in d["nb_tsel"]]
    p3 = bool(d["plato_ok"]) if "plato_ok" in d else sum(1 for x in nb if math.isfinite(x) and x > 0) >= min(2, len(nb))
    p4 = (all(v >= V.MIN_SDELOK for v in d["sdelok_pi"].values()) and bool(d["s9_ok"]) and bool(d["g4_ok"])
          and _gt(d["pbo_do"] + PBO_SHAG, d["pbo_posle"]))
    p5, pval = True, float("nan")
    if d.get("kontrol") is not None:
        dk = [x - d["tsel_p1"] for x in d["kontrol"]]
        pval = (1 + sum(1 for x in dk if not (x < dpi))) / (len(dk) + 1)
        p5 = len(dk) >= N_KONTROL and sum(1 for x in dk if dpi > x) >= 19 and pval <= 0.1
    r = dict(P1=p1, P2=p2, P3=p3, P4=p4, P5=p5)
    return {**r, "p_kontrol": pval, "delta": dpi, "proshla": all(r.values())}


def ostanovka(pbo: dict, proshli: dict) -> dict:
    """[Ж-054]: у всех стратегий общий PBO > 0,2 → «остановка, A′ = A». Иначе — дальше только непрошедшие."""
    if not pbo or any(not math.isfinite(p) for p in pbo.values()):
        raise ValueError("PBO не определён (nan): правило остановки не применяется")
    if all(p > PBO_STOP for p in pbo.values()):
        return dict(stop=True, itog="остановка, A′ = A", dalshe=[])
    return dict(stop=False, itog="продолжение: далее только непрошедшие", dalshe=sorted(s for s in pbo if not proshli.get(s)))


def elem_it2(r: dict) -> dict:
    """Элемент итерации 2 по §3 (П1–П4) из строки `it2/S{k}_itog.csv`; П5 к ансамблю неприменим."""
    return kriteriy(dict(
        tsel_pi=r["obj_min_msr"], tsel_p1=r["it1_obj"], msr_pi={2018: r["msr_2018"], 2019: r["msr_2019"]},
        msr_p1={2018: r["it1_msr_2018"], 2019: r["it1_msr_2019"]}, nb_tsel=[], plato_ok=bool(r["G3_plato_ansambl"]), sdelok_pi={2018: r["sdelok_2018"], 2019: r["sdelok_2019"]}, s9_ok=r["G5_s9"],
        g4_ok=r["G4_osi_centra"], pbo_do=r["pbo_it1"], pbo_posle=r["pbo_obshchaya"], kontrol=None))


def sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def sums_read(led: Path | None = None) -> dict:
    led, out = Path(led or LED), {}
    for ln in (led / "SHA256SUMS.txt").read_text("utf-8").splitlines():
        if ln and not ln.startswith("#"):
            h, f, n = ln.split()
            out[f] = (h, int(n.split("=")[1]))
    return out


def load_ledger(sid: str, name: str, evaluate, led: Path | None = None):
    """Ledger из pickle со сверкой sha256 до использования; хэша нет или не совпал — SystemExit."""
    led = Path(led or LED)
    p, key = led / sid / name, f"{sid}/{name}"
    s = sums_read(led).get(key)
    if s is None or not p.exists():
        raise SystemExit(f"{key}: нет файла или строки в SHA256SUMS — восстановить etap1_ansambl_run (restore), не считать молча")
    if sha(p) != s[0]:
        raise SystemExit(f"{key}: sha256 не совпал с SHA256SUMS.txt")
    L = V.Ledger(evaluate)
    L.cache = pickle.loads(p.read_bytes())
    return L


def save_ledger(sid: str, L, name: str, led: Path | None = None) -> None:
    """Пишет pickle и дописывает строку в SHA256SUMS.txt (только дополнение: строка с этим файлом уже есть — стоп)."""
    led = Path(led or LED)
    key, p = f"{sid}/{name}", led / sid / name
    if key in sums_read(led):
        raise SystemExit(f"{key}: уже в SHA256SUMS — реестр только дополняется")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(pickle.dumps(L.cache, protocol=4))
    with open(led / "SHA256SUMS.txt", "a", encoding="utf-8") as fh:
        fh.write(f"{sha(p)}  {key}  N={L.n_unique}\n")


def add_cfg(L, key: tuple, params: dict, res: dict) -> None:
    """Конфигурация вне пространства точек (итерация 10) — столбец матрицы и +1 в N; только дополнение."""
    k = (key, "train")
    if k in L.cache:
        raise ValueError(f"{key} уже в матрице: реестр только дополняется")
    L.cache[k] = (params, res)


def merge(a, b):
    """Объединяет кэши двух ledger (итерация 10: ключи 3 и 4 не пересекаются; совпавшие общие — из it2, должны быть равны)."""
    for k, v in b.cache.items():
        if k in a.cache and any(not np.array_equal(x, V.series(v[1])[y]) for y, x in V.series(a.cache[k][1]).items()):
            raise ValueError(f"ключ {k[0]} различается в двух ledger")
        a.cache.setdefault(k, v)
    return a


def sverka_ledger(L, evaluate, n: int = 5, seed: int = 20261028) -> list:
    """n случайных обычных строк ledger перевычисляются текущим кодом; расхождения `excess_d` — список ключей."""
    keys = sorted((k for k in L.cache if not isinstance(k[0][0], str)), key=repr)
    rng, bad = np.random.default_rng(seed), []
    for i in rng.choice(len(keys), size=min(n, len(keys)), replace=False):
        prm, old = L.cache[keys[i]]
        new = evaluate(prm, keys[i][1])
        if any(not np.array_equal(old["excess_d"][y], new["excess_d"][y]) for y in old["excess_d"]):
            bad.append(keys[i][0])
    return bad


def write_table(path: Path, rows: list) -> None:
    """Таблица итерации: шапка с пометкой про дивиденды; только дополнение."""
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    if new:
        path.write_text(SHAPKA, encoding="utf-8")
    pd.DataFrame(rows).to_csv(path, mode="a", index=False, header=new)
