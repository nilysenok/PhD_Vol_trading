"""S-Z по годам (Ж-259 п. 1): итоговая таблица по СОХРАНЁННЫМ прогнозам; обучения и новых прогнозов нет.
Статус: исследовательский результат на раскрытой истории, не OOS; сохранённые прогнозы, без обучения.
Выходы: `S_Z_YEARTABLE.csv`, `S_Z_FINAL_TABLE.md` (генератор из CSV, без ручных чисел), ключ `S.status` в CHECKPOINT (слиянием).
Запуск: `python3 -W ignore -m models_v2.s18.z_years --write | --check` (`--check` только читает, sha256 выходов до и после)."""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from models_v2.s18 import r_reg as RR
from models_v2.s18 import z_eval_cfg as C

REP, CKPT = C.REP, C.CKPT
CSV, MD = REP / "S_Z_YEARTABLE.csv", REP / "S_Z_FINAL_TABLE.md"
STATS, TABLE = REP / "S_Z_STATS.csv", REP / "S_Z_TABLE.csv"
SERIES = ("K1", "K2", "K3", "X1", "W5-G", "S-RCK", "S-RCKZ", "S-RZ", "S-RCKZs")
ADD = ("S-RCKZ", "S-RZ", "S-RCKZs")
BASES = ("S-RCK", "W5-G", "X1", "K2", "K3")
YEARS = tuple(str(y) for y in range(2018, 2026)) + ("2026 до 27.02", "2026 F (с 02.03)")
HALVES = ("h1 2018-2021", "h2 2022-2026")
HEAD = ["# исследовательский результат на раскрытой истории, не OOS; сохранённые прогнозы, без обучения",
        "# D_M: t⁺ 2018-01-04…2026-11-10, n = 2198; l = RV/F − ln(RV/F) − 1, QLIKE = среднее дневных потерь (без обрезки); Δ к S-RCK в % QLIKE S-RCK этого периода, минус — ниже",
        "# вклад = n_периода/n_D_M · (QLIKE ряда − QLIKE S-RCK)_периода / QLIKE S-RCK(D_M) · 100, п. п.; Σ вкладов по годам (и по половинам) = Δ всего D_M"]


def periods(tp: pd.DatetimeIndex) -> dict:
    y, d = tp.year.to_numpy(), tp.strftime("%Y-%m-%d").to_numpy()
    p = {"весь D_M": np.ones(len(tp), bool)}
    for k in range(2018, 2026):
        p[str(k)] = y == k
    p["2026 до 27.02"], p["2026 F (с 02.03)"] = (y == 2026) & (d <= "2026-02-27"), (y == 2026) & (d >= "2026-03-02")
    p["h1 2018-2021"], p["h2 2022-2026"] = y <= 2021, y >= 2022
    if (p["2026 до 27.02"] | p["2026 F (с 02.03)"]).sum() != (y == 2026).sum():
        raise SystemExit("2026 разбит неполно")
    return p


def compute(inp: dict) -> pd.DataFrame:
    rv, tp, F = inp["rv"], inp["tp"], inp["F"]
    ls = {k: rv / F[k] - np.log(rv / F[k]) - 1.0 for k in SERIES}
    qall = float(ls["S-RCK"].mean())
    rows = []
    for k in SERIES:
        for name, m in periods(tp).items():
            q, qb = float(ls[k][m].mean()), float(ls["S-RCK"][m].mean())
            rows.append(dict(series=k, period=name, n=int(m.sum()), first_tplus=str(tp[m].min().date()), last_tplus=str(tp[m].max().date()), qlike=q,
                             d_SRCK_pct=100.0 * (q - qb) / qb, contrib_pp=100.0 * m.sum() / len(tp) * (q - qb) / qall))
    return pd.DataFrame(rows)


def to_csv_text(df: pd.DataFrame) -> str:
    return "\n".join(HEAD) + "\n" + df.to_csv(index=False, lineterminator="\n")


def read_csv() -> pd.DataFrame:
    return pd.read_csv(CSV, comment="#", keep_default_na=False)


def _interval(st: pd.DataFrame, s: str, b: str) -> str:
    r = st[(st["window"] == "M") & (st["series"] == s) & (st["base"] == b)]
    return "[%.3f; %.3f]" % (r["lo90"].iloc[0], r["hi90"].iloc[0]) if len(r) else "не считался"


def _tab_a(df: pd.DataFrame, st: pd.DataFrame) -> list:
    w = df[df["period"] == "весь D_M"].set_index("series")
    out = ["| ряд | QLIKE | t⁺ с | t⁺ по | n | " + " | ".join("Δ к %s, %% [90 %%]" % b for b in BASES) + " |", "|" + "---|" * (5 + len(BASES))]
    for s in SERIES:
        r, cells = w.loc[s], []
        for b in BASES:
            d = 100.0 * (r["qlike"] - w.loc[b, "qlike"]) / w.loc[b, "qlike"]
            cells.append("%.3f %s" % (d, _interval(st, s, b)) if s != b else "—")
        out.append("| %s | %.6f | %s | %s | %d | %s |" % (s, r["qlike"], r["first_tplus"], r["last_tplus"], r["n"], " | ".join(cells)))
    return out


def _tab_b(df: pd.DataFrame) -> list:
    out = ["| период | n | " + " | ".join(SERIES) + " |", "|" + "---|" * (2 + len(SERIES))]
    for p in list(YEARS) + list(HALVES) + ["весь D_M"]:
        g = df[df["period"] == p].set_index("series")
        out.append("| %s | %d | %s |" % (p, g["n"].iloc[0], " | ".join("%.4f" % g.loc[s, "qlike"] for s in SERIES)))
    return out


def _tab_c(df: pd.DataFrame) -> list:
    out = ["| период | n | " + " | ".join("%s Δ %%" % s + " | %s вклад, п. п." % s for s in ADD) + " |", "|" + "---|" * (2 + 2 * len(ADD))]
    for p in list(YEARS) + list(HALVES) + ["весь D_M"]:
        g = df[df["period"] == p].set_index("series")
        out.append("| %s | %d | %s |" % (p, g["n"].iloc[0], " | ".join("%.3f | %.4f" % (g.loc[s, "d_SRCK_pct"], g.loc[s, "contrib_pp"]) for s in ADD)))
    return out


def to_md(df: pd.DataFrame, st: pd.DataFrame) -> str:
    h = ["# S-Z: итоговая таблица по годам", "", "Исследовательский результат на раскрытой истории, не OOS; сохранённые прогнозы, без обучения. Статус: DEVELOPMENT / HISTORICAL WALK-FORWARD.",
         "Источник: `S_Z_YEARTABLE.csv` (генератор `models_v2/s18/z_years.py`), интервалы — `S_Z_STATS.csv`. l = RV/F − ln(RV/F) − 1; QLIKE = среднее дневных потерь, без обрезки. Δ в % QLIKE базы (этого периода), минус — ниже. Вклад — в п. п. QLIKE S-RCK всего D_M; Σ вкладов по годам = Δ всего D_M.", ""]
    return "\n".join(h + ["## (а) Весь период D_M", ""] + _tab_a(df, st) + ["", "## (б) QLIKE по периодам", ""] + _tab_b(df)
                     + ["", "## (в) Добавочный эффект к S-RCK по периодам", ""] + _tab_c(df)) + "\n"


def sha_all() -> dict:
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None for p in (CSV, MD, CKPT)}


def status_block(cp: dict) -> dict:
    n, nw = C.registered_n(CKPT)
    return dict(written_at=time.strftime("%d.%m.%Y %H:%M:%S"), candidate="S-RCK", comparison=["S-RL1", "S-RCKZ"], control_versions=["X1", "W5-G"],
                closed_stages={"S-R": "bd4d3a59", "S-D": "e683432b", "S-L": "10d4afc5", "S-Z": "604aa2ed"}, next="S-RCD (спецификация до счёта, Ж-259)", n_program=n, n_wide=nw)


def write_status() -> None:
    cp = json.loads(CKPT.read_text(encoding="utf-8"))
    new = dict(cp)
    new["S.status"] = status_block(cp)
    tmp = CKPT.with_name(CKPT.name + ".tmp")
    tmp.write_text(RR._dump(new), encoding="utf-8")
    if {k: v for k, v in json.loads(tmp.read_text(encoding="utf-8")).items() if k != "S.status"} != cp:
        tmp.unlink()
        raise SystemExit("слияние изменило другие ключи — запись отменена")
    os.replace(tmp, CKPT)


def write() -> None:
    inp = C.load_inputs()
    df = compute(inp)
    CSV.write_text(to_csv_text(df), encoding="utf-8")
    MD.write_text(to_md(read_csv(), pd.read_csv(STATS, comment="#")), encoding="utf-8")
    write_status()
    print("записано:", CSV.name, MD.name, "S.status")


def check() -> list:
    bad, before = [], sha_all()
    inp = C.load_inputs()  # даты, rv, cfg8 — побитно одни во всех рядах (SystemExit при расхождении)
    df, f = compute(inp), read_csv()
    if CSV.read_text(encoding="utf-8") != to_csv_text(df):
        bad.append("CSV ≠ пересчёту")
    if not np.allclose(f[["qlike", "d_SRCK_pct", "contrib_pp"]].to_numpy(float), df[["qlike", "d_SRCK_pct", "contrib_pp"]].to_numpy(float), rtol=0, atol=1e-12):
        bad.append("числа CSV расходятся с пересчётом")
    if MD.read_text(encoding="utf-8") != to_md(f, pd.read_csv(STATS, comment="#")):
        bad.append("MD ≠ генератору из CSV")
    tb = pd.read_csv(TABLE, comment="#").set_index("series")["qlike"]
    w = f[f["period"] == "весь D_M"].set_index("series")["qlike"]
    bad += ["QLIKE %s ≠ S_Z_TABLE.csv (%.3e)" % (s, abs(w[s] - tb[s])) for s in SERIES if not abs(w[s] - tb[s]) <= 1e-12]
    for s in SERIES:
        g = f[f["series"] == s].set_index("period")["contrib_pp"]
        for grp in (YEARS, HALVES):
            if abs(g[list(grp)].sum() - g["весь D_M"]) > 1e-12:
                bad.append("Σ вкладов %s (%s…) ≠ Δ D_M" % (s, grp[0]))
    if not (set(f["n"][f["period"] == "весь D_M"]) == {C.N_D} and set(f["series"]) == set(SERIES)):
        bad.append("n ≠ 2198 или набор рядов")
    cp = json.loads(CKPT.read_text(encoding="utf-8")).get("S.status", {})
    n, nw = C.registered_n(CKPT)
    if not (cp.get("n_program") == n and cp.get("n_wide") == nw and cp.get("candidate") == "S-RCK" and cp.get("next")):
        bad.append("ключ S.status отсутствует или не сходится с S.*.registered")
    if sha_all() != before:
        bad.append("--check изменил выходы")
    return bad


if __name__ == "__main__":
    if sys.argv[1:] == ["--write"]:
        write()
    elif sys.argv[1:] == ["--check"]:
        b = check()
        print("\n".join(["ОШИБКА: " + x for x in b]) or "z_years --check: ок (sha256 выходов до/после совпали)")
        sys.exit(1 if b else 0)
    else:
        sys.exit("аргумент: --write или --check")
