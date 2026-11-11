"""W8-B, оценка (Ж-253 п. 4; `W8_SPEC.md` разд. 5, 6, 9). Статус: DEVELOPMENT / HISTORICAL WALK-FORWARD, не независимый OOS. Читает СОХРАНЁННЫЕ выходы `x1_w8_gap` (parquet) и `w5g.parquet`.
Публичное API:
- `MARKS` — три пометки разд. 6 (текст без `# `); `HEAD` — шапка каждого CSV (пометки + статус); `OUTS` — 8 CSV `docs/otchety/harj_x1/W8_{TABLE,STATS,YEARS,CELLS,CONC,GUARD,MCS,BETA}.csv`.
- `load_inputs(out=OUT) -> dict`: `require` для W8-G и W8-Gb (первым действием); окно M (n = 2198, `x1_w8_diag_core.load`: типы перерыва, годы, половины), ряды K1, K2, K3, X1, M, ZB, W5-G, W8-G, W8-Gb, 19 сдвигов
  на одних датах; сверка d, rv, X1, меток gap с parquet — расхождение SystemExit.
- `compute(inp) -> dict` таблиц: `table`, `stats` (+ `sens`, `perm`, `perm_p`), `years`, `cells`, `conc`, `guard`, `mcs`, `beta`, `decision`, `identity` (макс. нарушение тождеств вкладов); `texts(t) -> {имя: текст}`.
- `write()`: ключ `W8.stage1` (с `written_at`, хэши входов, выходов и кода) в `CHECKPOINT.json` пишется ПЕРВЫМ (слиянием, прочие ключи побайтно), затем таблицы с p.
- `check()` — только чтение: пересчёт = файлам; Σ вкладов = Δ (1e-12); N = 814 / 3333; в днях gap = 0 разность потерь W8-G и X1 нулевая; константы порогов = 0,415 / 0,50 / 0,26 / 0,10; ключ = хэшам файлов; хэш выходов до и после равен.
Запуск: `python3 -W ignore -m models_v2.s16.x1_w8_eval [--write | --check]`."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from models_v2.s16 import x1_w5_eval as E5, x1_w8_diag_core as K, x1_w8_eval_t as T, x1_w8_gap as G, x1_w8_reg as REG

REP, OUT = REG.REP, G.OUT
CKPT = REP / "CHECKPOINT.json"
OUTS = tuple("W8_%s.csv" % k for k in ("TABLE", "STATS", "YEARS", "CELLS", "CONC", "GUARD", "MCS", "BETA"))
KEYS = ("table", "stats", "years", "cells", "conc", "guard", "mcs", "beta")
MARKS = ("признак IMOEX RV выбран из 10 после анализа истории окна M (W8 разд. 1.5); прохождение переноса на уровне семейства неотличимо от случайного (p перестановок 0,05, сдвигов 0,15)",
         "база W5-G: состояние gap выбрано из 24, оценщик β — после просмотра 24.02.2022",
         "исследовательский результат на раскрытой истории, не независимое подтверждение (Ж-250 п. 6)")
HEAD = "\n".join("# " + m for m in MARKS) + "\n# DEVELOPMENT / HISTORICAL WALK-FORWARD, не независимый OOS; l = RV/F − ln(RV/F) − 1, среднее дневных потерь на общих датах окна M (t⁺ 2018-01-04…2026-11-10); Δ в % QLIKE базы, минус — лучше\n"
EPS = 1e-12


def _sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load_inputs(out: Path = OUT) -> dict:
    for k in G.OUTS:
        REG.require(k)
    x, w = K.load(), pd.read_parquet(G.W5G)
    s = {k: pd.read_parquet(out / f) for k, f in G.OUTS.items()}
    sh = {j: pd.read_parquet(out / f) for j, f in G.SHIFTS.items()}
    gap = w["gap"].to_numpy(int)
    for k, f in list(s.items()) + [("shift_%02d" % j, f) for j, f in sh.items()]:
        ok = len(f) == len(x) and (f["d"].to_numpy() == x["d"].to_numpy()).all() and np.array_equal(f["rv"].to_numpy(), x["rv"].to_numpy()) and np.array_equal(f["X1"].to_numpy(), x["F"].to_numpy())
        if not ok or not np.array_equal(f["gap"].to_numpy(), gap):
            raise SystemExit("%s: даты, rv, X1 или метка gap расходятся с окном M / w5g.parquet" % k)
    F = {"K1": x["K1"], "K2": x["K2"], "K3": x["K3"], "X1": x["F"], "M": x["M"], "ZB": E5._zb(x["d"]), "W5-G": w["F"], **{k: f["F"] for k, f in s.items()}}
    return dict(x=x, F={k: np.asarray(v, float) for k, v in F.items()}, rv=x["rv"].to_numpy(float), tp=pd.DatetimeIndex(x["t_plus"]), gap=gap, w8g=s["W8-G"],
                shifts=[np.asarray(sh[j]["F"], float) for j in sorted(sh)])


def compute(inp: dict) -> dict:
    rv, tp, F = inp["rv"], inp["tp"], inp["F"]
    perm_q = [float(T.Q.ql(rv, f).mean()) for f in inp["shifts"]]
    t = dict(table=T.table(rv, tp, F), stats=T.stats(rv, tp, F, perm_q), years=T.years(rv, tp, F), conc=T.conc(rv, tp, F), mcs=T.mcs(rv, F), beta=T.beta_table(inp["w8g"]))
    t["cells"], t["identity"] = T.cells(inp["x"], F["W8-G"])
    pp = float(t["stats"].loc[t["stats"]["kind"] == "perm_p", "p_one"].iloc[0])
    g = T.guard(t["table"], t["years"], t["conc"], pp)
    t["decision"] = T.decide(g)
    t["guard"] = pd.concat([g, pd.DataFrame([dict(id="решение", condition="основной критерий и У1–У6 (W8_SPEC.md разд. 5)", value=t["decision"], threshold="—", formula="выведено кодом из строк выше",
                                                   passed=t["decision"].startswith("третья"))])], ignore_index=True)
    return t


def texts(t: dict) -> dict:
    return {n: HEAD + t[k].to_csv(index=False, lineterminator="\n", float_format="%.12g") for k, n in zip(KEYS, OUTS)}


def merge_checkpoint(ckpt: Path, t: dict, tx: dict, out: Path = OUT, now: str | None = None) -> None:
    cp = json.loads(ckpt.read_text(encoding="utf-8"))
    code = Path(__file__).resolve().parent
    cp["W8.stage1"] = dict(written_at=now or REG.now_text(), inputs={f: _sha(out / f) for f in list(G.OUTS.values()) + list(G.SHIFTS.values())} | {"w5g.parquet": _sha(G.W5G)},
                           outputs={k: hashlib.sha256(v.encode("utf-8")).hexdigest() for k, v in tx.items()}, code={p.name: _sha(p) for p in sorted(code.glob("x1_w8_*.py"))},
                           n_holm=T.N_HOLM, n_holm_wide=T.N_WIDE, decision=t["decision"], configurations=2, shifts=19)
    ckpt.write_text(json.dumps(cp, indent=1, sort_keys=True, ensure_ascii=False), encoding="utf-8")


def write(rep: Path = REP, ckpt: Path = CKPT, out: Path = OUT) -> dict:
    t = compute(load_inputs(out))
    tx = texts(t)
    merge_checkpoint(ckpt, t, tx, out)             # ключ W8.stage1 — до таблиц с p-значениями
    for k, v in tx.items():
        (rep / k).write_text(v, encoding="utf-8")
    return t


def tree_hash(rep: Path = REP, ckpt: Path = CKPT, out: Path = OUT) -> dict:
    ps = [rep / n for n in OUTS if (rep / n).exists()] + [ckpt] + sorted(out.glob("*.parquet"))
    return {str(p): _sha(p) for p in ps}


def check(rep: Path = REP, ckpt: Path = CKPT, out: Path = OUT) -> list:
    h0, bad = tree_hash(rep, ckpt, out), []
    inp = load_inputs(out)
    t = compute(inp)
    for n, v in texts(t).items():
        bad += [] if (rep / n).exists() and (rep / n).read_text(encoding="utf-8") == v else ["%s: текст ≠ пересчёту" % n]
    if T.identity_gap(t["years"]) > EPS or t["identity"] > EPS:
        bad.append("Σ вкладов ≠ Δ (> 1e-12)")
    s = t["stats"]
    if not ((s["n_holm"].dropna() == 814).all() and (s["n_holm_wide"].dropna() == 3333).all()):
        bad.append("Холм: N ≠ 814 / N_wide ≠ 3333")
    if T.THR != {"У1": 0.415, "У2": 0.50, "У3": 0.26, "У6": 0.10}:
        bad.append("пороги ≠ константам спецификации")
    off = inp["gap"] == 0
    if not (inp["F"]["W8-G"][off] == inp["F"]["X1"][off]).all():
        bad.append("в днях gap = 0 F W8-G ≠ X1")
    cp = json.loads(ckpt.read_text(encoding="utf-8")).get("W8.stage1", {})
    want = {k: hashlib.sha256(v.encode("utf-8")).hexdigest() for k, v in texts(t).items()}
    bad += [] if cp.get("outputs") == want else ["CHECKPOINT W8.stage1 ≠ пересчёту"]
    print("решение:", t["decision"])
    return bad + ([] if tree_hash(rep, ckpt, out) == h0 else ["--check изменил выходы"])


if __name__ == "__main__":
    if "--write" in sys.argv:
        print("решение:", write()["decision"])
    elif "--check" in sys.argv:
        b = check()
        print("W8 eval --check: %s" % ("ok" if not b else "FAIL " + "; ".join(b)))
        sys.exit(1 if b else 0)
    else:
        sys.exit("укажите --write или --check")
