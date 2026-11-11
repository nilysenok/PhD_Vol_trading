"""W5-B, оценка (Ж-250, Ж-251, Ж-252; `W5_SPEC.md` разделы 5, 6, 11.6). Статус: DEVELOPMENT / HISTORICAL WALK-FORWARD, не независимый OOS. Читает СОХРАНЁННЫЕ выходы `x1_w5_gap` (parquet) и `W5_GAP_PIT.csv`.
Публичное API:
- `OUTS` — 8 CSV `docs/otchety/harj_x1/W5_{TABLE,STATS,YEARS,GAPDAYS,GUARD,SENS,CONC,MCS}.csv` в порядке 11.6; шапка каждого — пометки 11.5.
- `load_inputs(out=OUT, gap_path=GAP_PIT) -> dict`: окно M (n = 2198), ряды K1, K2, K3, X1, M, W5-G, W5-Ga, W5-Gb, W5-GK2, ZB на одних датах; сверка rv, X1, K2, меток gap с parquet и CSV; расхождение — SystemExit.
- `perm_q(inp)`: QLIKE W5-G при сдвиге меток 252 + 100·(j − 1), j = 1…19 (пересчёт `x1_w5_gap.walk`); в N не входят.
- `compute(inp) -> dict` таблиц (`table`, `stats`, `years`, `gapdays`, `guard`, `sens`, `conc`, `mcs`, `decision`); `texts(t) -> {имя файла: текст}`.
- `write()`: ключ `W5.eval` в `CHECKPOINT.json` (слиянием, прочие ключи побайтно) пишется ПЕРВЫМ, затем таблицы с p; `check()` — только чтение: пересчёт = файлам, Σ вкладов = Δ (1e-12), N = 798 / 3301,
  в днях gap = 0 разность потерь нулевая, ключ совпадает с хэшами файлов, хэш всех выходов до и после равен.
Запуск: `python3 -W ignore -m models_v2.s16.x1_w5_eval [--write | --check]`."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from models_v2.s16 import x1_w1_join as J, x1_w3_core as C, x1_w5_eval_t as T, x1_w5_gap as G, x1_w5_gap_cal as CAL, x1_w5_reg as REG

REP, OUT, GAP_PIT = CAL.REP, G.OUT, G.GAP_PIT
CKPT = REP / "CHECKPOINT.json"
OUTS = tuple("W5_%s.csv" % k for k in ("TABLE", "STATS", "YEARS", "GAPDAYS", "GUARD", "SENS", "CONC", "MCS"))
KEYS = ("table", "stats", "years", "gapdays", "guard", "sens", "conc", "mcs")
HEAD = "\n".join(CAL.MARKS) + "\n# DEVELOPMENT / HISTORICAL WALK-FORWARD, не независимый OOS; l = RV/F − ln(RV/F) − 1, среднее дневных потерь на общих датах окна M (t⁺ 2018-01-04…2026-11-10); Δ в % QLIKE базы, минус — лучше\n"
EPS = 1e-12


def _sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _zb(d) -> np.ndarray:
    a, n = J.read_fwd(J.MODELS / "zb.parquet", {}), J.read_fwd(J.NEW / "zb.parquet", {})
    z = J.stitch(a, n).set_index("d")["F"].reindex(pd.DatetimeIndex(d))
    if z.isna().any():
        raise SystemExit("ZB не покрывает окно M: нет %d дат" % int(z.isna().sum()))
    return z.to_numpy(float)


def load_inputs(out: Path = OUT, gap_path: Path = GAP_PIT) -> dict:
    for k in G.OUTS:
        REG.require(k)
    x, gp = C.load(), CAL.read_commented(gap_path)
    s = {k: pd.read_parquet(out / f) for k, f in G.OUTS.items()}
    gap = gp["gap_pit"].astype(int).to_numpy()
    for k, f in s.items():
        base = x["K2"] if k == "W5-GK2" else x["F"]
        ok = len(f) == len(x) == len(gp) and (f["d"].to_numpy() == x["d"].to_numpy()).all() and np.array_equal(f["rv"].to_numpy(), x["rv"].to_numpy())
        if not ok or not np.array_equal(f["K2" if k == "W5-GK2" else "X1"].to_numpy(), base.to_numpy()) or not np.array_equal(f["gap"].to_numpy(), G.shift_labels(gap) if k == "W5-Gb" else gap):
            raise SystemExit("%s: даты, rv, база или метка gap расходятся с окном M / W5_GAP_PIT.csv" % k)
    F = {"K1": x["K1"], "K2": x["K2"], "K3": x["K3"], "X1": x["F"], "M": x["M"], "ZB": _zb(x["d"]), **{k: f["F"] for k, f in s.items()}}
    return dict(x=x, F={k: np.asarray(v, float) for k, v in F.items()}, gap=gap, delta=gp["delta_days"].astype(int).to_numpy(), rv=x["rv"].to_numpy(float), tp=pd.DatetimeIndex(x["t_plus"]))


def perm_q(inp: dict) -> list:
    x, out = inp["x"], []
    for j in range(19):
        f = G.walk(x["d"], x["t_plus"], x["rv"], x["F"], G.shift_labels(inp["gap"], 252 + 100 * j))
        out.append(float(T.Q.ql(inp["rv"], f["F"].to_numpy()).mean()))
    return out


def compute(inp: dict) -> dict:
    rv, tp, F = inp["rv"], inp["tp"], inp["F"]
    t = dict(table=T.table(rv, tp, F), stats=T.stats(rv, tp, F, perm_q(inp)), years=T.years(rv, tp, F), gapdays=T.gapdays(rv, tp, F, inp["gap"], inp["delta"]), sens=T.sens(rv, tp, F), conc=T.conc(rv, tp, F), mcs=T.mcs(rv, F))
    pp = float(t["stats"].loc[t["stats"]["kind"] == "perm_p", "p_one"].iloc[0])
    g = T.guard(t["table"], t["years"], t["conc"], pp)
    t["decision"] = T.decide(g)
    t["guard"] = pd.concat([g, pd.DataFrame([dict(id="решение", condition="правило раздела 5: основной критерий и У1–У5", value=t["decision"], threshold="—", formula="выведено кодом из строк выше", passed=t["decision"].startswith("новая"))])], ignore_index=True)
    return t


def texts(t: dict) -> dict:
    return {n: HEAD + t[k].to_csv(index=False, lineterminator="\n") for k, n in zip(KEYS, OUTS)}


def merge_checkpoint(ckpt: Path, t: dict, tx: dict, out: Path = OUT, gap_path: Path = GAP_PIT) -> None:
    cp = json.loads(ckpt.read_text(encoding="utf-8"))
    code = Path(__file__).resolve().parent
    cp["W5.eval"] = dict(inputs={f: _sha(out / f) for f in G.OUTS.values()} | {"W5_GAP_PIT.csv": _sha(gap_path), "W5_CALENDAR_PIT.csv": _sha(CAL.CAL)},
                         outputs={k: hashlib.sha256(v.encode("utf-8")).hexdigest() for k, v in tx.items()}, code={p.name: _sha(p) for p in sorted(code.glob("x1_w5_*.py"))},
                         n_holm=T.N_HOLM, n_holm_wide=T.N_WIDE, decision=t["decision"], configurations=0, forecasts_written=0)
    ckpt.write_text(json.dumps(cp, indent=1, sort_keys=True, ensure_ascii=False), encoding="utf-8")


def write(rep: Path = REP, ckpt: Path = CKPT, out: Path = OUT, gap_path: Path = GAP_PIT) -> dict:
    t = compute(load_inputs(out, gap_path))
    tx = texts(t)
    merge_checkpoint(ckpt, t, tx, out, gap_path)             # ключ W5.eval — до таблиц с p-значениями
    for k, v in tx.items():
        (rep / k).write_text(v, encoding="utf-8")
    return t


def tree_hash(rep: Path = REP, ckpt: Path = CKPT, out: Path = OUT) -> dict:
    ps = [rep / n for n in OUTS if (rep / n).exists()] + [ckpt] + sorted(out.glob("*.parquet"))
    return {str(p): _sha(p) for p in ps}


def check(rep: Path = REP, ckpt: Path = CKPT, out: Path = OUT, gap_path: Path = GAP_PIT) -> list:
    h0, bad = tree_hash(rep, ckpt, out), []
    t = compute(load_inputs(out, gap_path))
    for n, v in texts(t).items():
        bad += [] if (rep / n).exists() and (rep / n).read_text(encoding="utf-8") == v else ["%s: текст ≠ пересчёту" % n]
    if T.identity_gap(t["years"]) > EPS:
        bad.append("Σ вкладов ≠ Δ (> 1e-12)")
    s = t["stats"]
    if not ((s["n_holm"].dropna() == 798).all() and (s["n_holm_wide"].dropna() == 3301).all()):
        bad.append("Холм: N ≠ 798 / N_wide ≠ 3301")
    if float(t["gapdays"].loc[t["gapdays"]["subset"] == "gap^PIT = 0", "max_abs_loss_diff"].iloc[0]) != 0.0:
        bad.append("в днях gap = 0 потери W5-G ≠ потерям X1")
    cp = json.loads(ckpt.read_text(encoding="utf-8")).get("W5.eval", {})
    want = {k: hashlib.sha256(v.encode("utf-8")).hexdigest() for k, v in texts(t).items()}
    bad += [] if cp.get("outputs") == want else ["CHECKPOINT W5.eval ≠ пересчёту"]
    print("решение:", t["decision"])
    return bad + ([] if tree_hash(rep, ckpt, out) == h0 else ["--check изменил выходы"])


if __name__ == "__main__":
    if "--write" in sys.argv:
        print("решение:", write()["decision"])
    elif "--check" in sys.argv:
        b = check()
        print("W5 eval --check: %s" % ("ok" if not b else "FAIL " + "; ".join(b)))
        sys.exit(1 if b else 0)
    else:
        sys.exit("укажите --write или --check")
