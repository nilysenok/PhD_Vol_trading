"""Диагностика v6 train4 (Ж-145), лист Д-5: таблицы (4) лонг/шорт по годам и (5) Sharpe/mSR без лучших/худших дней у топ-10 каждой Sk.
Вход — кэш пересчёта Д-4 (`fastbt.diag_v6_perescet`: <sid>_<seed>_<i>.npz/.json) и top10.csv. Данных ≥ 2020 не читает.
Запуск: python3 -m fastbt.diag_v6_t45 <top10.csv> <кэш> <t4.csv> <t5.csv>.
(4) Год — год ВЫХОДА сделки (открытая на конце окна — год последней даты окна). PnL сделки = pnl_до_издержек − комиссия, в единицах x
(ряд избыточной доходности портфеля); без ФДР и платы за заём — справочно к excess. Доли: сделок и Σ|PnL| стороны.
(5) «Без 10 лучших/худших» — дни вычеркнуты по x (как scripts/analysis/dve_metriki_a.stroka), mSR нед. — metriki_e.stroka_ned.
Колонки Ж-148 — только диагностика (fastbt/metriki_diag.py), в пороги не входят. Строки вывода — по diag_v6_pravila.PRAVILA."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from fastbt import metriki_diag as md, metriki_e as me
from fastbt.diag_v6_pravila import PRAVILA

STATUS = "разработка train4 2016–2019, потолок (in-sample), после аудита; диагностика, не отбор"
SEED = {"s1": "seed", "s2": "seed2"}   # прогон → ключ seed в кэше (progon_v6 sobol_klyuch)


def zagruzit(kesh, sid: str, seed: str, i: int) -> dict:
    nm = Path(kesh) / f"{sid}_{seed}_{int(i):06d}"
    z, j = np.load(f"{nm}.npz"), json.loads(Path(f"{nm}.json").read_text(encoding="utf-8"))
    return dict(sid=sid, seed=seed, i=int(i), x=z["x"], daty=z["daty"], tik_poz=z["tik_poz"], zhurnal=j["zhurnal"], klyuch_ok=j.get("klyuch_ok"))


def god_sdelki(tr: dict, daty) -> int:
    """Год ВЫХОДА сделки; нет даты выхода (открыта на конце окна) — год последней даты окна."""
    return int(tr["vyhod"][:4]) if tr.get("vyhod") else int(str(np.asarray(daty)[-1])[:4])


def pnl_sdelki(tr: dict) -> float:
    return float(tr["pnl_do_izderzhek"] - tr["komissiya"])


def tablica_4(tochki: list) -> pd.DataFrame:
    """Строка на (точку, год выхода): число сделок и PnL лонг/шорт, доли сторон, флаг «одна сторона > порога»; let_odna_storona — на точку."""
    rows = []
    for t in tochki:
        acc = {}
        for tr in t["zhurnal"]:
            a = acc.setdefault(god_sdelki(tr, t["daty"]), {1: [0, 0.0], -1: [0, 0.0]})[1 if tr["storona"] > 0 else -1]
            a[0] += 1
            a[1] += pnl_sdelki(tr)
        flagi = {g: _dolya(c[1][0], c[-1][0]) for g, c in acc.items()}
        let = sum(1 for g, c in acc.items() if max(flagi[g], 1 - flagi[g]) > PRAVILA["odna_storona_dolya_max"])
        for g in sorted(acc):
            c = acc[g]
            dp = _dolya(abs(c[1][1]), abs(c[-1][1]))
            rows.append(dict(sid=t["sid"], rang=t.get("rang"), sobol_i=t["i"], prohod=t.get("prohod"), god_vyhoda=g,
                             n_long=c[1][0], n_short=c[-1][0], dolya_long_sdelok=flagi[g], pnl_long=c[1][1], pnl_short=c[-1][1],
                             dolya_long_pnl_abs=dp, odna_storona=max(flagi[g], 1 - flagi[g]) > PRAVILA["odna_storona_dolya_max"],
                             let_odna_storona=let, slaboe_mesto=let >= PRAVILA["odna_storona_let_min"]))
    return pd.DataFrame(rows)


def _dolya(a: float, b: float) -> float:
    return float(a / (a + b)) if a + b > 0 else float("nan")


def _ubrat(x: np.ndarray, hvost: str | None, n: int) -> np.ndarray:
    """Маска дней: hvost 'luch' — вычеркнуть n лучших по x, 'hud' — n худших."""
    keep = np.ones(len(x), bool)
    if hvost:
        keep[np.argsort(x)[-n:] if hvost == "luch" else np.argsort(x)[:n]] = False
    return keep


def _metriki(x, daty, hvost, n) -> dict:
    keep = _ubrat(np.asarray(x, float), hvost, n)
    xs = np.asarray(x, float)[keep]
    ned = me.metriki_ned(xs, np.asarray(daty)[keep])
    return dict(sharpe_dn=float(me.sharpe(xs)), msr_dn=float(me.msr(xs)), msr_ned=float(ned["msr_ned"]))


def tablica_5(tochki: list) -> pd.DataFrame:
    """Строка на точку: Sharpe дн., mSR дн., mSR нед. — полные / без n лучших / без n худших дней; остаток Sharpe; Ж-148 (диагностика)."""
    n, rows = PRAVILA["bez_luchshih_dnej_n"], []
    for t in tochki:
        x, daty = np.asarray(t["x"], float), t["daty"]
        pol, bl, bh = (_metriki(x, daty, h, n) for h in (None, "luch", "hud"))
        ost = bl["sharpe_dn"] / pol["sharpe_dn"] if pol["sharpe_dn"] > 0 else float("nan")
        poz, pnl = np.asarray(t["tik_poz"], float).T, [pnl_sdelki(tr) for tr in t["zhurnal"]]
        let = len(x) / md.GOD
        row = dict(sid=t["sid"], rang=t.get("rang"), sobol_i=t["i"], prohod=t.get("prohod"), dni=len(x))
        for pr, m in (("poln", pol), ("bez_luchshih", bl), ("bez_hudshih", bh)):
            row.update({f"{k}_{pr}": v for k, v in m.items()})
        row.update(ostatok_bez_luchshih=ost,
                   zavisit_ot_dnej=bool(bl["sharpe_dn"] <= 0 or ost < PRAVILA["bez_luchshih_dnej_dolya_min"]),
                   sharpe_norm_diag=md.sharpe_norm(x, poz), sharpe_sdelok_diag=md.sharpe_sdelok(pnl, let), cena_prostoev_diag=md.cena_prostoev(x, poz),
                   prichina_nan_diag="; ".join(p for p in (md.prichina_norm(poz), md.prichina_sdelok(pnl, let)) if p))
        rows.append(row)
    return pd.DataFrame(rows)


def chitat_top10(path) -> list:
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(r for r in f if not r.startswith("#")))


def sobrat(top10: list, kesh) -> list:
    out = []
    for r in top10:
        t = zagruzit(kesh, r["sid"], SEED[r["prohod"]], int(r["sobol_i"]))
        if not t["klyuch_ok"]:
            raise ValueError(f"{r['sid']} i={r['sobol_i']}: klyuch не совпал")
        out.append({**t, "rang": int(r["rang"]), "prohod": r["prohod"]})
    return out


def zapisat(df: pd.DataFrame, path, shapka: str) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(f"# статус: {STATUS}\n# {shapka}\n# формат: \"v\": \"1\"\n")
        df.to_csv(f, index=False, lineterminator="\n")


def main(argv=None) -> int:
    a = list(sys.argv[1:] if argv is None else argv)
    if len(a) != 4:
        print(__doc__)
        return 2
    tochki = sobrat(chitat_top10(a[0]), a[1])
    zapisat(tablica_4(tochki), a[2], "таблица (4): лонг/шорт по году выхода сделки, топ-10 каждой Sk; PnL справочно (без ФДР и платы за заём)")
    zapisat(tablica_5(tochki), a[3], "таблица (5): Sharpe дн., mSR дн./нед. — полные, без 10 лучших, без 10 худших дней; *_diag — Ж-148, диагностика")
    print(f"OK: {len(tochki)} точек -> {a[2]}, {a[3]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
