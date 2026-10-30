"""Полная таблица голых идей S7–S12 (Ж-153 п. 1). Только чтение готовых таблиц docs/tables/golaya_v7/, сигналы не пересчитываются.
Выход: docs/tables/golaya_v7/polnaya.csv. Проценты — в год от капитала. Запуск без аргументов."""
from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TAB = ROOT / "docs" / "tables" / "golaya_v7"
PERIODY = {"train4": ("2016-01-01", 4, (2016, 2017, 2018, 2019)), "v2": ("2015-01-01", 5, (2015, 2016, 2017, 2018, 2019))}
N_TIK = 17  # вес тикера 1/N (отчёт golaya_s7_s12, п. 2)


def chitat(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(l for l in f if not l.startswith("#")))


def bez_progreva(rows: list[dict], period: str) -> list[dict]:
    return [r for r in rows if r["data_vhoda"] >= PERIODY[period][0]]


def _f(x) -> float:
    return float(x) if x not in ("", None) else float("nan")


def po_storonam(rows: list[dict], gody: float) -> dict:
    """Лонг/шорт: число, доля, PnL до и после издержек (%/год)."""
    out, n = {}, len(rows)
    for nm, sg in (("long", 1), ("short", -1)):
        rr = [r for r in rows if int(float(r["storona"])) == sg]
        out[f"n_{nm}"] = len(rr)
        out[f"dolya_{nm}"] = len(rr) / n if n else 0.0
        out[f"pnl_do_{nm}"] = 100 * sum(_f(r["pnl_do_izderzhek"]) for r in rr) / gody
        out[f"pnl_posle_{nm}"] = 100 * sum(_f(r["pnl_posle"]) for r in rr) / gody
    return out


def ekspoziciya(rows: list[dict], dni: float) -> float:
    """Средняя валовая доля капитала (|лонг|+|шорт|): Σ сила·удержание / N_тикеров / дней окна (оценка по журналу сделок)."""
    w = lambda r: 1.0 if r["sila"] == "" else _f(r["sila"])  # у S10–S12 силы нет — равный вес
    return sum(w(r) * _f(r["uderzhanie_dn"]) for r in rows) / N_TIK / dni if rows else 0.0


def dnevnoj_ryad(rows: list[dict], calendar: list[str]) -> np.ndarray:
    """PnL после издержек по дате выхода; дни без выходов — нули (приближение, как в таблице Ж-150)."""
    d = defaultdict(float)
    for r in rows:
        d[r["data_vyhoda"]] += _f(r["pnl_posle"])
    return np.array([d.get(c, 0.0) for c in calendar])


def sharpe_prib(x) -> float:
    x = np.asarray(x, float)
    if len(x) < 2:
        return 0.0
    s = x.std(ddof=1)
    return float(x.mean() / s * np.sqrt(252)) if s > 1e-12 else 0.0


def vne_oblasti(msr_dn: str) -> str:
    """mSR не определён (mVaR ≤ 0, т.е. z_cf ≥ 0 с точностью до знака среднего) — «вне области Корниша–Фишера»."""
    v = _f(msr_dn)
    return "да" if v != v else "нет"


def p_perm(k: int, n: int) -> float:
    return (k + 1) / (n + 1)


def sverit_p(r: dict) -> None:
    n = int(r["n"])
    k = round(_f(r["dolya_ne_huzhe_centra"]) * n)
    if abs(p_perm(k, n) - _f(r["p_perm"])) > 1e-9:
        raise ValueError(f"p не сходится: k={k}, n={n}, p={r['p_perm']}")


def kontrol(r: dict) -> dict:
    sverit_p(r)
    return {"med": _f(r["mediana"]), "min": _f(r["min"]), "max": _f(r["max"]), "p": _f(r["p_perm"]), "n": int(r["n"])}


def vyvod(sid: str, p_placebo: float) -> str:
    if sid == "S7":
        return f"не включено: S7, p {p_placebo:.3f}".replace(".", ",")
    if sid in ("S8", "S9"):
        return "включено в v7 (Ж-153), низкий приоритет"
    return "включено в v7 (Ж-153)"


def stroka(sv: dict, rows: list[dict], cal: list[str], period: str, ctl: dict) -> dict:
    _, gody, gody_spisok = PERIODY[period]
    sid, var = sv["strategiya"], sv["variant"]
    mine = [r for r in rows if r["sid"] == sid and r["variant"] == var]
    ryad = dnevnoj_ryad(mine, cal)
    out = {"sid": sid, "variant": var, "period": period, "sdelok_v_god": sv["sdelok_v_god"], "dolya_vremeni": sv["dolya_vremeni"],
           "ekspoziciya_valovaya_ocenka": ekspoziciya(mine, 252 * gody), **po_storonam(mine, gody),
           "mSR_dn": sv["mSR"], "msr_ned": sv["msr_ned"], "sharpe_dn_prib": sharpe_prib(ryad),
           "mSR_vne_oblasti_KF": vne_oblasti(sv["mSR"]),
           "brutto_pct_god": sv["pnl_pct_god"], "izderzhki_pct_god": sum(_f(sv[k]) for k in ("kom_pct_god", "zaem_pct_god", "stavka_pct_god")),
           "netto_pct_god": sv["excess_pct_god"]}
    for g in gody_spisok:
        gg = [r for r in mine if r["data_vyhoda"][:4] == str(g)]
        do, po = (100 * sum(_f(r[k]) for r in gg) for k in ("pnl_do_izderzhek", "pnl_posle"))
        out.update({f"mSR_{g}": sv[f"mSR_{g}"], f"brutto_{g}": do, f"izderzhki_{g}": do - po, f"netto_{g}": po})
    for pref, d in (("k2", ctl["k2"].get(sid)), ("plac", ctl["plac"].get(sid))):
        for k in ("med", "min", "max", "p"):
            out[f"{pref}_{k}"] = d[k] if d and var == "centr" else ""
    out["diag_sharpe_norm"] = sv["Sharpe норм. (при полной загрузке, диагностика)"]
    out["diag_sharpe_po_sdelkam"] = sv["Sharpe по сделкам (диагностика)"]
    out["diag_cena_prostoev"] = sv["цена простоев (диагностика)"]
    out["vyvod"] = vyvod(sid, ctl["plac"][sid]["p"]) if var == "centr" and period == "train4" else ""
    return out


def sobrat(period: str) -> list[dict]:
    rows = bez_progreva(chitat(TAB / f"sdelki_{period}.csv"), period)
    cal = sorted({r["data_vyhoda"] for r in rows})
    ctl = {k: {r["strategiya"]: kontrol(r) for r in chitat(TAB / f"{f}_{period}.csv")} for k, f in (("k2", "k2"), ("plac", "placebo"))}
    sv = [s for s in chitat(TAB / f"svod_{period}.csv") if s["variant"] == "centr" or s["variant"].startswith("vhod_T")]
    return [stroka(s, rows, cal, period, ctl) for s in sv]


def main() -> None:
    out = sobrat("train4") + sobrat("v2")
    cols = list(dict.fromkeys(k for r in out for k in r))
    with open(TAB / "polnaya.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, cols, restval="")
        w.writeheader()
        w.writerows(out)
    print(f"строк: {len(out)}, столбцов: {len(cols)}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        sys.exit("без аргументов")
    main()
