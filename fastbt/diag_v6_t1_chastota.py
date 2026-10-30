"""Диагностика v6 train4 (Ж-145, Ж-151), Д-2 дополнение: Моррис по числу сделок и сравнение μ* сигнала со стопом и размером.

Единицы μ*: `|Δ Шарп дн.|` на ОДИН активный шаг траектории Морриса. Шаг = 2/3 нормированной сетки оси (morris_v6.DELTA), на Δ не делится;
Y — дневной Шарп портфеля (ddof=1, `sr_dnevnoy`), накопленный до конца года окна (≤2016/≤2017/≤2018), а не mSR (progon_morris_v6_status.ocenka).
Число сделок точки — поле `ns` строк `morris_y.jsonl` (Σ `sdelok_y` ledger за 2016–2019); траектории (`nom`, `por`) воспроизводятся
`progon_morris_v6.plan` по r/mnozh из шапки CSV и сверяются с `n_akt` CSV по окну ≤2016 (расхождение — ValueError).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from fastbt import diag_v6_t1 as t1
from fastbt import morris_v6 as mv

# ↺ Умолчание дирижёра (класс B, Ж-151), не решение владельца: ось «частота», если медиана сделок в год < 5 по всем её активным шагам.
KRIT_SDELOK_GOD = 5
GODOV = 4  # train4: 2016–2019
OKNO = "<=2018"
STOP = ("tejk", "trail", "bezubytok")  # плюс всё, что начинается с `stop`
RAZMER = ("razmer_tip", "m_sily", "chast", "pir")
ZAGOLOVOK = ("μ* = среднее |Δ Шарп дн.| на один активный шаг траектории Морриса (шаг = 2/3 нормированной сетки оси, на Δ не делится); "
             "Y — дневной Шарп (не mSR) до конца года окна; потолок: in-sample 2016–2019",
             f"вердикт: «частота» — медиана сделок в год < {KRIT_SDELOK_GOD} (умолчание дирижёра, класс B, Ж-151); «сигнал» — сделки есть")


def _y_ns(path: Path) -> tuple:
    """Y (n, 3) и число сделок ns (n) по уникальным точкам из morris_y.jsonl; нет поля ns — NaN."""
    rows = [json.loads(s) for s in path.read_text("utf-8").splitlines() if s.strip()]
    n = max(e["b"] for e in rows)
    Y, ns = np.full((n, 3), np.nan), np.full(n, np.nan)
    for e in rows:
        Y[e["a"]:e["b"]], ns[e["a"]:e["b"]] = e["y"], e.get("ns", np.nan)
    return Y, ns


def _shapka(path: Path) -> dict:
    return {m[1]: int(m[2]) for m in re.finditer(r"^# (r|mnozh|tochek): (\d+)", path.read_text("utf-8"), re.M)}


def plan_kanon(sid: str, r: int, mnozh: int) -> tuple:
    """(nom, por, имена осей, число уникальных точек) — траектории прогона Морриса (данных не читает)."""
    from fastbt import progon_morris_v6 as pm
    kub, ox = pm.setka(sid)
    nom, por, pts = pm.plan(sid, kub, ox, r, mnozh, int(kub["sobol"]["morris_seed"]))
    return nom, por, [n for n, _, _ in ox], len(pts)


def os_sdelki(nom, por, ns) -> dict:
    """По осям (индекс): n шагов, доля пар без сделок, μ*(сделки/год), медиана max(сделок пары)/год. Шаг активен: ключ изменился, ns обеих конечны."""
    a, b = ns[nom[:, :-1]], ns[nom[:, 1:]]
    akt = (nom[:, 1:] != nom[:, :-1]) & np.isfinite(a) & np.isfinite(b)
    out = {}
    for i in range(por.shape[1]):
        m = akt & (por == i)
        n = int(m.sum())
        out[i] = dict(n_shagov=n, dolya_par_bez_sdelok=float(((a == 0) & (b == 0))[m].mean()) if n else float("nan"),
                      mu_sdelki=float(np.abs(b - a)[m].mean() / GODOV) if n else float("nan"),
                      med_sdelok_god=float(np.median(np.maximum(a, b)[m]) / GODOV) if n else float("nan"))
    return out


def verdikt(med_sdelok_god: float) -> str:
    if not np.isfinite(med_sdelok_god):
        return "нет шагов"
    return "частота" if med_sdelok_god < KRIT_SDELOK_GOD else "сигнал"


def soglasovat(nom, por, Y, df, osi) -> None:
    """n_akt воспроизведённого плана = n_akt CSV (окно ≤2016) по каждой оси, иначе ValueError."""
    nol = (Y == 0).all(axis=1)
    n = np.isfinite(mv.effekty(Y[nom, 0], nom, por, nol[nom])).sum(0)
    d = df[df["okno"] == "<=2016"].set_index("os")["n_akt"]
    for i, o in enumerate(osi):
        if o not in d.index or int(d[o]) != int(n[i]):
            raise ValueError(f"план Морриса не совпал с CSV: ось {o}: n_akt план {int(n[i])}, CSV {d.get(o)}")


def tablica_chastota(koren, gruppy_csv=None, plan_fn=plan_kanon) -> pd.DataFrame:
    """Строка на сигнальную ось с «не измерено» (окно ≤2018) каждой Sk: доля пар без сделок, μ*(сделки/год), вердикт, μ* Шарпа."""
    koren = Path(koren)
    gr = t1._gruppy(Path(gruppy_csv) if gruppy_csv else t1.pr.CSV)
    rows = []
    for sid in t1._sk(koren):
        df = t1.df_sk(koren, sid, gr)
        sh = _shapka(koren / sid / f"morris_{sid}.csv")
        nom, por, osi, npts = plan_fn(sid, sh["r"], sh["mnozh"])
        Y, ns = _y_ns(koren / sid / "morris_y.jsonl")
        if len(Y) != npts or npts != sh["tochek"]:
            raise ValueError(f"{sid}: точек в плане {npts}, в morris_y {len(Y)}, в CSV {sh['tochek']}")
        soglasovat(nom, por, Y, df, osi)
        sd = os_sdelki(nom, por, ns)
        for _, r in df[(df["okno"] == OKNO) & (df["g"] == "signal") & ~df["izm"]].iterrows():
            s = sd[osi.index(r["os"])]
            rows.append(dict(sid=sid, os=r["os"], n_akt=int(r["n_akt"]), mu_sharpe=r["mu_star"], **s, verdikt=verdikt(s["med_sdelok_god"])))
    return pd.DataFrame(rows)


def _gruppa_osej(d: pd.DataFrame, nazv: str) -> pd.Series:
    if nazv == "signal":
        return d[d["g"] == "signal"]
    v = d[d["g"] == "vyhod_risk"]
    return v[v["os"].str.startswith("stop") | v["os"].isin(STOP)] if nazv == "stop" else v[v["os"].isin(RAZMER)]


def sravnenie(koren, gruppy_csv=None) -> pd.DataFrame:
    """Одна строка на Sk (окно ≤2018): μ* оценённых осей (n_akt > 0) сигнала, стопа и размера — среднее и максимум с именем оси."""
    gr = t1._gruppy(Path(gruppy_csv) if gruppy_csv else t1.pr.CSV)
    rows = []
    for sid in t1._sk(Path(koren)):
        df = t1.df_sk(Path(koren), sid, gr)
        d, r = df[df["okno"] == OKNO], dict(sid=sid)
        for nazv in ("signal", "stop", "razmer"):
            x = _gruppa_osej(d, nazv)
            x = x[np.isfinite(x["mu_star"]) & (x["n_akt"] > 0)]
            r.update({f"osej_{nazv}": len(_gruppa_osej(d, nazv)), f"ocen_{nazv}": len(x),
                      f"mu_{nazv}_sr": float(x["mu_star"].mean()) if len(x) else 0.0,
                      f"mu_{nazv}_max": float(x["mu_star"].max()) if len(x) else 0.0,
                      f"os_{nazv}_max": x.loc[x["mu_star"].idxmax(), "os"] if len(x) else ""})
        rows.append(r)
    return pd.DataFrame(rows)


def zapisat(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        f.write("".join(f"# {s}\n" for s in ZAGOLOVOK))
        df.to_csv(f, index=False)
