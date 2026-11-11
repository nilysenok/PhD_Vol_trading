"""S-T (Ж-256 пп. 1, 4): сводная таблица и годы по СОХРАНЁННЫМ прогнозам, без обучения. D_M: t⁺ 2018-01-04…2026-11-10, n = 2198.
Статус: DEVELOPMENT / HISTORICAL WALK-FORWARD, исследовательский результат, не OOS. Функции потерь, периоды и вклады — из `a_eval_*`, `r_eval_*` (импорт, без правок).
Публичное API:
- `SERIES` — ряды со ролью; `MISSING` — «не рассчитывался» (S-RA0, S-RA1, S-RCD); `load_inputs()` — F и потери `ls` на D_M (сверка d, rv — SystemExit);
- `table(tp, F, ls)` — S_T_TABLE: QLIKE (среднее дневных потерь) и Δ в % QLIKE базы к K1, K2, K3, X1, W5-G; `years(tp, ls)` — S_T_YEARS (QLIKE по периодам и половинам, Δ и вклад к W5-G и X1 для S-A*, S-R*, S-C*);
- `build()` — все таблицы (кэш); `write()` / `check()`: `check` — только чтение (хэш выходов до и после).
Запуск: `python3 -W ignore -m models_v2.s18.t_table [--write | --check]`."""
from __future__ import annotations

import functools
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from models_v2.s16 import x1_w5_eval_t as T5
from models_v2.s18 import a_eval_yr as AY, r_eval as RE, r_eval_tab as RT

ROOT, REP = RE.ROOT, RE.REP
BASES = ("K1", "K2", "K3", "X1", "W5-G")
YB = ("W5-G", "X1")
CTRL = ("K1", "K2", "K3", "X1", "W5-G", "W8-G", "B_mix")
SA, SR, SC = ("S-A0", "S-A1", "S-A2", "S-A2p"), ("S-RA2", "S-RA2b", "S-RA2c", "S-RCK", "S-RCKb", "S-RCKc"), ("S-CD", "S-CK", "S-CKA", "S-C7", "S-CKL")
MISSING = ("S-RA0", "S-RA1", "S-RCD")
ROLES = {"K1": "контроль (К1)", "K2": "контроль (К2)", "K3": "контроль (К3)", "X1": "контрольная версия", "W5-G": "контрольная версия", "W8-G": "экспериментальная", "B_mix": "справочный"}
FILES = {"S-A0": "A/a0", "S-A1": "A/a1", "S-A2": "A/a2", "S-A2p": "A/a2p", "S-CD": "CM/cd", "S-CK": "CM/ck", "S-CKA": "CM/cka", "S-C7": "CM/c7", "S-CKL": "CM/ckl"}
DIR = ROOT / "data/regen/harj_struct"
OUTS = ("S_T_TABLE.csv", "S_T_YEARS.csv", "S_T_SLICES.csv", "S_T_TRIMSETS.csv", "S_T_REPORT.md")
MARK = "исследовательский результат, не независимый OOS; DEVELOPMENT / HISTORICAL WALK-FORWARD (Ж-256)"


def _role(s: str) -> str:
    return ROLES.get(s) or ("структурный S-A" if s in SA else "структурный S-R" if s in SR or s in MISSING else "структурный S-C")


def _frame_ok(a: pd.DataFrame, x: pd.DataFrame, name: str) -> None:
    if not (len(a) == len(x) and (a["d"].to_numpy() == x["d"].to_numpy()).all() and np.array_equal(a["rv"].to_numpy(), x["rv"].to_numpy()) and np.isfinite(a["F"]).all()):
        raise SystemExit("%s: даты или rv расходятся с окном M, или нечисловые прогнозы" % name)


def load_inputs() -> dict:
    inp = RE.load_inputs()
    F = dict(inp["F"])
    for v, f in FILES.items():
        if v in F:
            continue
        a = pd.read_parquet(DIR / (f + ".parquet"))
        _frame_ok(a, inp["x"], v)
        F[v] = a["F"].to_numpy(float)
    ls = RT.losses(inp["rv"], F)
    return dict(rv=inp["rv"], tp=inp["tp"], F=F, ls=ls)


def series_order() -> list:
    return list(CTRL) + list(SA) + ["S-RA0", "S-RA1", "S-RA2", "S-RA2b", "S-RA2c", "S-RCD", "S-RCK", "S-RCKb", "S-RCKc"] + list(SC)


def table(tp, ls: dict) -> pd.DataFrame:
    tp, rows = pd.DatetimeIndex(tp), []
    q = {k: float(v.mean()) for k, v in ls.items()}
    for s in series_order():
        if s in MISSING:
            rows.append(dict(series=s, role=_role(s), status="не рассчитывался", n=np.nan, first_tplus="", last_tplus="", qlike=np.nan, **{"d_%s_pct" % b: np.nan for b in BASES}))
            continue
        rows.append(dict(series=s, role=_role(s), status="рассчитан", n=len(tp), first_tplus=str(tp.min().date()), last_tplus=str(tp.max().date()), qlike=q[s],
                         **{"d_%s_pct" % b: 100.0 * (q[s] - q[b]) / q[b] for b in BASES}))
    return pd.DataFrame(rows)


def years(tp, ls: dict) -> pd.DataFrame:
    keep = [s for s in series_order() if s not in MISSING]
    sub = {k: ls[k] for k in keep}
    comps = [(s, b) for s in SA + SR + SC for b in YB]
    y = AY.years(tp, sub, comps)
    return y[y["base"].isin(["-"] + list(YB))].reset_index(drop=True)


def periods(tp) -> dict:
    return T5.masks(tp)


@functools.lru_cache(maxsize=1)
def build() -> dict:
    from models_v2.s18 import t_report as TR, t_slices as TS
    inp = load_inputs()
    t = dict(table=table(inp["tp"], inp["ls"]), years=years(inp["tp"], inp["ls"]))
    t["slices"], t["trimsets"] = TS.slices(inp["tp"], inp["ls"]), TS.trimsets(inp["tp"], inp["ls"])
    t["report"] = TR.report(t, inp["tp"])
    return dict(inp=inp, t=t, tx=texts(t, inp["tp"]))


def head(tp) -> str:
    tp = pd.DatetimeIndex(tp)
    return "# %s\n# D_M: t⁺ %s…%s, n = %d; l = RV/F − ln(RV/F) − 1, QLIKE = среднее дневных потерь (без обрезки); Δ в %% QLIKE базы, минус — лучше\n" % (MARK, tp.min().date(), tp.max().date(), len(tp))


def texts(t: dict, tp) -> dict:
    h = head(tp)
    out = {n: h + t[k].to_csv(index=False, lineterminator="\n", float_format="%.12g") for n, k in zip(OUTS[:4], ("table", "years", "slices", "trimsets"))}
    out[OUTS[4]] = t["report"]
    return out


def tree_hash(rep: Path = REP) -> dict:
    ps = [rep / n for n in OUTS if (rep / n).exists()] + sorted((DIR / "A").glob("*.parquet")) + sorted((DIR / "R").glob("*.parquet")) + sorted((DIR / "CM").glob("*.parquet"))
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in ps}


def write(rep: Path = REP) -> dict:
    b = build()
    for k, v in b["tx"].items():
        (rep / k).write_text(v, encoding="utf-8")
    return b


def check(rep: Path = REP) -> list:
    from models_v2.s18 import t_chk as C
    h0, b = tree_hash(rep), build()
    bad = [] if all((rep / n).exists() and (rep / n).read_text(encoding="utf-8") == v for n, v in b["tx"].items()) else ["таблицы ≠ пересчёту"]
    bad += C.problems(b["t"], b["inp"])
    return bad + ([] if tree_hash(rep) == h0 else ["--check изменил выходы"])


if __name__ == "__main__":
    if "--write" in sys.argv:
        write()
        print("S-T: записано", ", ".join(OUTS))
    elif "--check" in sys.argv:
        e = check()
        print("S-T --check: %s" % ("ok" if not e else "FAIL " + "; ".join(e)))
        sys.exit(1 if e else 0)
    else:
        sys.exit("укажите --write или --check")
