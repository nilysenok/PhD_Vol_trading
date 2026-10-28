"""Ж-066 п. 1: четыре диагностические таблицы по точкам итерации 1 S1–S6 (определения — tolkovaniya.md, [Ж-066 п. 1]).
Чистые функции над списками сделок/заявок (синтетика в tests/test_diag_zh066.py); прогон движка — diag_zh066_run.
Диагностика, не выбор: train 2018–2019, комиссия 0,04 %, без дивидендов, до аудита; в ledger и реестр отбора не входит.
Сборка: python3 -m fastbt.diag_zh066 <каталог с S{k}.pkl от diag_zh066_run>"""
from __future__ import annotations

import csv
import hashlib
import math
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/tables/diag"
STATUS = "диагностика, train 2018–2019, без дивидендов, до аудита; комиссия 0,04 %; в ledger и реестр отбора не добавлено"
NOISE = 10                                    # < 10 сделок в дециле — «шум»
NAMES = ["S1", "S2", "S3", "S4", "S5", "S6"]


def _pct(x: float) -> float:
    return 100.0 * x


def _stat(pnls: list) -> tuple:
    n = len(pnls)
    return (n, _pct(sum(pnls)), _pct(sum(pnls) / n) if n else float("nan"),
            sum(p > 0 for p in pnls) / n if n else float("nan"))


def deciles(trades: list) -> list:
    """(а) Дециль силы сигнала по всем сделкам стратегии: ранг по силе (при равенстве — порядок в журнале), k = 10·r // n.
    Сделки без силы (nan) — отдельная строка «нет». Строка: (дециль, n, сила_мин, сила_макс, Σpnl %, средняя %, доля плюс, метка)."""
    ok = [t for t in trades if math.isfinite(t["sila"])]
    bad = [t for t in trades if not math.isfinite(t["sila"])]
    order = sorted(range(len(ok)), key=lambda i: (ok[i]["sila"], i))
    grp = {k: [] for k in range(1, 11)}
    for r, i in enumerate(order):
        grp[10 * r // len(ok) + 1].append(ok[i])
    rows = []
    for k, g in grp.items():
        n, s, m, w = _stat([t["pnl"] for t in g])
        lo, hi = (min(t["sila"] for t in g), max(t["sila"] for t in g)) if g else (float("nan"),) * 2
        rows.append((str(k), n, lo, hi, s, m, w, "шум" if n < NOISE else ""))
    if bad:
        n, s, m, w = _stat([t["pnl"] for t in bad])
        rows.append(("нет", n, float("nan"), float("nan"), s, m, w, "нет силы"))
    return rows


def long_short(trades: list) -> list:
    """(в) Лонг/шорт отдельно: (сторона, n, Σpnl %, средняя сделка %, доля плюсовых)."""
    return [(name, *_stat([t["pnl"] for t in trades if sel(t["side"])]))
            for name, sel in (("long", lambda s: s > 0), ("short", lambda s: s < 0), ("все", lambda s: True))]


def corr_matrix(series: dict, method: str) -> pd.DataFrame:
    """(б) Корреляция дневных рядов (inner join по датам); method pearson | spearman."""
    return pd.DataFrame(series).dropna().corr(method=method)


def slip(o: dict) -> float:
    """Проскальзывание, %: (цена исполнения − цена решения close d)/close d, знак «против сделки» положителен."""
    return _pct(o["side"] * (o["px"] - o["close_d"]) / o["close_d"])


def gap_gain(o: dict) -> float:
    """Улучшение при гэпе, %: исполнение по open лучше уровня L. Всегда ≥ 0 (limit_px)."""
    return _pct(o["side"] * (o["L"] - o["px"]) / o["L"])


def limit_rows(orders: list) -> list:
    """(г) По тикерам и «все»: (тикер, выставлено, исполнено, доля, проскальзывание ср. %, исполнений с гэпом, улучшение при гэпе ср. %)."""
    out = []
    for t in sorted({o["tiker"] for o in orders}) + ["все"]:
        g = [o for o in orders if t in ("все", o["tiker"])]
        f = [o for o in g if o["isp"]]
        gp = [gap_gain(o) for o in f if gap_gain(o) > 1e-9]
        out.append((t, len(g), len(f), len(f) / len(g), float(np.mean([slip(o) for o in f])) if f else float("nan"),
                    len(gp), float(np.mean(gp)) if gp else float("nan")))
    return out


def reconcile(trades: list, journal: list, tol: float = 1e-9) -> float:
    """G2: сделки прогона совпадают с журналом S{k}_sdelki.csv по (вход, выход, тикер) и pnl; возвращает |Σ − Σ|."""
    key = lambda r: (r["d_in"], r["d_out"], r["tiker"])                           # noqa: E731
    a, b = sorted(trades, key=key), sorted(journal, key=key)
    if [key(x) for x in a] != [key(x) for x in b]:
        raise AssertionError("состав сделок не совпал с журналом")
    d = abs(sum(x["pnl"] for x in a) - sum(x["pnl"] for x in b))
    if d > tol or any(abs(x["pnl"] - y["pnl"]) > tol for x, y in zip(a, b)):
        raise AssertionError(f"PnL не совпал с журналом: {d}")
    return d


def write_csv(path: Path, meta: dict, cols: list, rows: list) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        for k, v in meta.items():
            fh.write(f"# {k}: {v}\n")
        w = csv.writer(fh)
        w.writerow(cols)
        for r in rows:
            w.writerow([f"{x:.8f}" if isinstance(x, float) else x for x in r])


def build(runs: dict, journals: dict, meta: dict) -> None:
    a, v, g, b = [], [], [], []
    for sid in NAMES:
        r = runs[sid]
        reconcile(r["trades"], journals[sid])
        d_rows, ls_rows = deciles(r["trades"]), long_short(r["trades"])
        tot = sum(x["pnl"] for x in journals[sid])                           # G2: суммы таблиц (а), (в) = Σ pnl журнала
        for s in (sum(x[4] for x in d_rows) / 100, ls_rows[2][2] / 100, (ls_rows[0][2] + ls_rows[1][2]) / 100):
            assert abs(s - tot) < 1e-9, (sid, s, tot)
        a += [(sid, *x) for x in d_rows]
        v += [(sid, *x) for x in ls_rows]
        orders = [o for o in r["orders"] if o["d_sig"] >= "2018-01-01"]
        g += [(sid, *x) for x in limit_rows(orders)] if orders else [(sid, "лимиток нет: вход рынком по close", 0, 0, float("nan"), float("nan"), 0, float("nan"))]
    ser = {s: pd.Series(runs[s]["excess"]) for s in NAMES}
    for m in ("pearson", "spearman"):
        c = corr_matrix(ser, m)
        b += [(m, s, *[float(c.loc[s, t]) for t in NAMES]) for s in NAMES]
    for name, cols, rows in (
            ("a_decili", ["S", "дециль", "n", "сила_мин", "сила_макс", "pnl_сумма_%", "pnl_сделка_%", "доля_плюс", "метка"], a),
            ("b_korr", ["метод", "S"] + NAMES, b),
            ("v_long_short", ["S", "сторона", "n", "pnl_сумма_%", "pnl_сделка_%", "доля_плюс"], v),
            ("g_limitki", ["S", "тикер", "выставлено", "исполнено", "доля_исполнения", "проскальзывание_ср_%", "исполнений_с_гэпом", "улучшение_гэп_ср_%"], g)):
        write_csv(OUT / f"zh066_{name}.csv", {"статус": STATUS, **meta}, cols, rows)


def main(d: str) -> None:
    from fastbt.diag_zh066_run import journal
    runs = {s: pickle.load(open(Path(d) / f"{s}.pkl", "rb")) for s in NAMES}
    jr = {s: journal(s) for s in NAMES}
    h = hashlib.sha256(b"".join((OUT / f"{s}_sdelki.csv").read_bytes() for s in NAMES)).hexdigest()[:16]
    code = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    dg = ",".join(f"{s}:{runs[s]['vhody']}" for s in NAMES)
    build(runs, jr, {"коммит кода": code, "хэш входов": f"журналы sha256[:16]={h}; digest входов движка {dg}"})


if __name__ == "__main__":
    main(sys.argv[1])
