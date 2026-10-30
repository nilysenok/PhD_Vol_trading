"""Диагностика v6 train4 (Ж-145), лист Д-2: таблица (1) — μ* Морриса по группам осей.

Вход — каталог `<koren>/<Sk>/{morris_<Sk>.csv, zamorozka.yaml}` (только чтение). Группа оси — по `gruppy_osej.csv` (Д-0, #837),
а НЕ по колонке `gruppa` CSV Морриса (там раздел куба). «Измерено» — n_akt ≥ 20 и μ* конечно; оси из `ne_ocenyeny_net_sdelok` и
`mertvye_ne_ocenyeny` zamorozka.yaml — не измерены всегда. Σμ* и доли — по измеренным осям. Данных ledger ≥ 2020 модуль не касается.
"""
from __future__ import annotations

import io
import math
from pathlib import Path

import pandas as pd
import yaml

from fastbt import diag_v6_pravila as pr

N_AKT_MIN = 20
# ↺ В PRAVILA (толкование [31.10 00:12] (в)) правила для таблицы (1) нет; берутся пороги плана §3 (класс B, умолчание архитектора).
T1_PRAVILO = dict(okno="<=2018", top_n=2, ne_izmereno_dolya_max=0.5, max_mu_min=0.05)
ETIKETKA = {"signal": "сигнал", "vyhod_risk": "выход-риск", "kontekst": "контекст", "ispolnenie": "исполнение"}
KOLONKI = ["sid", "okno", "gruppa", "osej", "izmereno", "ne_izmereno", "ne_ocenyeny", "mertvye", "zamorozheno",
           "sum_mu", "dolya", "max_mu", "mediana_mu"]


def _chitat_csv(path: Path) -> pd.DataFrame:
    """CSV с шапкой-комментариями `# …` (запятые в них ломают разбор): строки `#` отбрасываются."""
    tekst = "".join(s for s in path.read_text(encoding="utf-8").splitlines(True) if not s.startswith("#"))
    return pd.read_csv(io.StringIO(tekst))


def _gruppy(path: Path) -> dict:
    return {(r.sid, r.os): r.gruppa for r in _chitat_csv(path).itertuples()}


def _gruppa(gr: dict, sid: str, os: str, razdel: str) -> str:
    if (sid, os) not in gr:
        raise ValueError(f"{sid}.{os}: оси нет в gruppy_osej.csv")
    return gr[(sid, os)]


def _sk(koren: Path) -> list:
    return sorted(p.name for p in koren.iterdir() if (p / f"morris_{p.name}.csv").exists())


def _stroki_sk(koren: Path, sid: str, gr: dict) -> list:
    df = _chitat_csv(koren / sid / f"morris_{sid}.csv")
    zam = yaml.safe_load((koren / sid / "zamorozka.yaml").read_text(encoding="utf-8")) or {}
    net_sdelok = set(zam.get("ne_ocenyeny_net_sdelok") or [])
    mertvye = set(zam.get("mertvye_ne_ocenyeny") or [])
    df["g"] = [_gruppa(gr, sid, r.os, r.gruppa) for r in df.itertuples()]
    ok = (df["n_akt"] >= N_AKT_MIN) & df["mu_star"].map(lambda v: math.isfinite(v)) & ~df["os"].isin(net_sdelok | mertvye)
    df["izm"] = ok
    df["ne_ocen"] = df["os"].isin(net_sdelok)
    df["mertv"] = df["os"].isin(mertvye)
    out = []
    for okno, d in df.groupby("okno", sort=True):
        tot = d.loc[d["izm"], "mu_star"].sum()
        for g in pr.GRUPPY:
            x = d[d["g"] == g]
            mu = x.loc[x["izm"], "mu_star"]
            s = float(mu.sum())
            out.append((sid, okno, g, len(x), int(x["izm"].sum()), int((~x["izm"]).sum()), int(x["ne_ocen"].sum()), int(x["mertv"].sum()),
                        int((x["zamorozit"].astype(str) == "True").sum()), s, s / tot if tot > 0 else 0.0,
                        float(mu.max()) if len(mu) else 0.0, float(mu.median()) if len(mu) else 0.0))
    return out


def vyvod(df: pd.DataFrame) -> str:
    """Одна строка вывода по правилу таблицы (1): «слабое место: …, Sk: …» или «не слабое место»."""
    t, prich = T1_PRAVILO, {}
    for sid, d in df[df["okno"] == t["okno"]].groupby("sid"):
        d = d.set_index("gruppa")
        rang = sorted(pr.GRUPPY, key=lambda g: (-d.loc[g, "sum_mu"], pr.GRUPPY.index(g)))
        s, p = d.loc["signal"], []
        if "signal" not in rang[:t["top_n"]]:
            p.append(f"сигнал не в двух верхних по Σμ* ({t['okno']})")
        if s["osej"] and s["ne_izmereno"] / s["osej"] > t["ne_izmereno_dolya_max"]:
            p.append("в сигнале > 50 % осей не измерено")
        if s["max_mu"] < t["max_mu_min"]:
            p.append("max μ* сигнала < 0,05")
        if p:
            prich[sid] = p
    if not prich:
        return "не слабое место"
    return "слабое место: " + "; ".join(f"{k}: {', '.join(v)}" for k, v in prich.items()) + f", Sk: {','.join(prich)}"


def tablica_1(koren, gruppy_csv=None) -> tuple:
    """(DataFrame по Sk × окно × группа, строка вывода). `koren` — каталог со всеми Sk; ни одной Sk — ValueError."""
    koren = Path(koren)
    gr = _gruppy(Path(gruppy_csv) if gruppy_csv else pr.CSV)
    sids = _sk(koren)
    if not sids:
        raise ValueError(f"{koren}: нет Sk с morris_<Sk>.csv")
    df = pd.DataFrame([r for sid in sids for r in _stroki_sk(koren, sid, gr)], columns=KOLONKI)
    return df, vyvod(df)


if __name__ == "__main__":
    import sys
    d, v = tablica_1(sys.argv[1])
    out = Path(sys.argv[2])
    out.parent.mkdir(parents=True, exist_ok=True)
    d.to_csv(out, index=False)
    print(d.to_string(index=False), "\n" + v)
