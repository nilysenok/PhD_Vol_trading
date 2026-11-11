"""S-L-EVAL (Ж-256 п. 5; `S_L_SPEC.md`, разд. 5, 6): оценка S-RL1 и контролей S-RL1a (π := π̄), S-RL1b (π со сдвигом 252) на D_M (t⁺ 2018-01-04…2026-11-10, n = 2198).
Статус: DEVELOPMENT / HISTORICAL WALK-FORWARD, исследовательский результат, не OOS; S-RL1a и S-RL1b — контроли, не кандидаты. Читает СОХРАНЁННЫЕ `L/{rl1,rl1a,rl1b}.parquet`, `R/rck.parquet` и ряды X1, W5-G, K2, K3; прогнозов и реестр не трогает.
Публичное API:
- `KEYS` → 8 таблиц `docs/otchety/harj_struct/S_L_EVAL_{TABLE,DELTA,YEARS,SLICES,MECH,MONTHS,GUARD,VERDICT}.csv` (`OUTS`); отчёт `S_L_EVAL_REPORT.md` — `l_doc`; `KEY` = "S.L.eval".
- `load_inputs() -> dict`, `compute(inp) -> dict` (вердикты 5.2, 5.3, 5.4 и P-L1…P-L4 выводятся из таблиц; условие 3 — из `S_D_ELEMENTS.csv`), `texts`, `write()`: ключ `S.L.eval` пишется ПЕРВЫМ слиянием, затем таблицы с p; `check()` — только чтение (`l_eval_chk.problems`; sha256 выходов до и после).
Запуск: `python3 -W ignore -m models_v2.s18.l_eval [--write | --check]`."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from models_v2.s16 import x1_w8_diag_core as K
from models_v2.s18 import a_eval_stat as S
from models_v2.s18 import l_eval_par as LP
from models_v2.s18 import l_eval_tab as LT
from models_v2.s18 import l_reg as LR
from models_v2.s18 import l_reg2 as LR2
from models_v2.s18 import l_walk as LW
from models_v2.s18 import r_reg as RR

ROOT, REP, CKPT = RR.ROOT, RR.ROOT / "docs/otchety/harj_struct", RR.CKPT
KEY, FIRST, LAST, N_D = "S.L.eval", "2018-01-04", "2026-11-10", 2198
KEYS = ("table", "delta", "years", "slices", "mech", "months", "guard", "verdict")
OUTS = tuple("S_L_EVAL_%s.csv" % k.upper() for k in KEYS)
REPORT = "S_L_EVAL_REPORT.md"
MARK = "исследовательский результат, не OOS; S-RL1a, S-RL1b, S-RL1b2 — контроли, не кандидаты; критерии разд. 5 S_L_SPEC.md объявлены до счёта; «подтверждено» только при p Холма (N программы) < 0,05 и p SPA < 0,05"
FILES = {"S-RCK": ROOT / "data/regen/harj_struct/R/rck.parquet", **{IDS: LW.L_DIR / (v + ".parquet") for v, IDS in LW.IDS_L.items()}}


def _sha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def head(tp) -> str:
    tp = pd.DatetimeIndex(tp)
    return "# %s\n# D_M: t⁺ %s…%s, n = %d; l = RV/F − ln(RV/F) − 1, QLIKE = среднее дневных потерь (без обрезки); Δ в %% QLIKE базы, минус — лучше; DEVELOPMENT / HISTORICAL WALK-FORWARD\n" % (MARK, tp.min().date(), tp.max().date(), len(tp))


def registered_n(ckpt: Path = CKPT) -> tuple:
    cp = json.loads(ckpt.read_text(encoding="utf-8"))
    ks = [cp.get(k) or {} for k in (LR.KEY, LR2.KEY)]    # N программы после S-RL1b2 (Ж-257): последний ключ S-L — 852 / 3409
    if not all(isinstance(k.get("n"), int) and isinstance(k.get("n_wide"), int) for k in ks):
        raise SystemExit("нет ключей %s, %s с N — оценка отказана" % (LR.KEY, LR2.KEY))
    return max(k["n"] for k in ks), max(k["n_wide"] for k in ks)


def load_inputs(ckpt: Path = CKPT) -> dict:
    x = K.load()
    if not (len(x) == N_D and str(x["t_plus"].min().date()) == FIRST and str(x["t_plus"].max().date()) == LAST):
        raise SystemExit("окно M ≠ D_M (2198 пар, t⁺ %s…%s)" % (FIRST, LAST))
    F, fr = {"X1": x["F"], "W5-G": x["W5G"], "K2": x["K2"], "K3": x["K3"]}, {}
    for k, p in FILES.items():
        a = pd.read_parquet(p)
        ok = len(a) == N_D and (a["d"].to_numpy() == x["d"].to_numpy()).all() and np.array_equal(a["rv"].to_numpy(), x["rv"].to_numpy()) and np.allclose(a["cfg8"], x["C"], rtol=1e-12, atol=0)
        if not (ok and np.isfinite(a[["F", "X1_R"]].to_numpy()).all()):
            raise SystemExit("%s: даты, rv, cfg8 расходятся с окном M или есть нечисловые прогнозы" % k)
        fr[k], F[k], F["X1_R(%s)" % k] = a, a["F"], a["X1_R"]
    n, nw = registered_n(ckpt)
    return dict(x=x, rv=x["rv"].to_numpy(float), tp=pd.DatetimeIndex(x["t_plus"]), F={k: np.asarray(v, float) for k, v in F.items()}, frames={k: fr[k] for k in LT.SERIES[:4]}, N=n, N_wide=nw)


def elements() -> pd.DataFrame:
    return pd.read_csv(REP / "S_D_ELEMENTS.csv", comment="#")


def compute(inp: dict) -> dict:
    rv, tp, F = inp["rv"], inp["tp"], inp["F"]
    ls = LT.losses(rv, F)
    tb, dl = LT.table(rv, tp, F), LT.deltas_ext(ls, tp, inp["N"], inp["N_wide"])
    yrs, mech = LT.years(tp, ls, LT.PAIRS_Y), LP.mech(inp["frames"], F, rv)
    guard, conds, pred = LT.stage_rows(tp, ls), LT.cond_rows(tb, dl, elements()), LP.predictions(mech, yrs, F)
    code, text = LT.rck_verdict(dl)
    ver = pd.concat([conds.assign(section="5.2"), pd.DataFrame([dict(section="5.4", id="5.4", condition="S-RL1 против S-RCK (5 параметров C-K): " + text, value=code, threshold="—", passed=code in ("worse_noise", "needed"))]),
                     guard.tail(1).assign(section="5.3")[["section", "id", "condition", "value", "threshold", "passed"]],
                     pd.DataFrame(dict(section="5.5", id=pred["id"], condition=pred["statement"], value=pred["fulfilled"], threshold=pred["threshold"].astype(str), passed=pred["fulfilled"] == "выполнено"))], ignore_index=True)
    return dict(table=tb, delta=dl, years=yrs, slices=LT.slices(tp, ls, LT.PAIRS_S), mech=mech, months=LT.months(inp["frames"]), guard=guard, verdict=ver)


def texts(t: dict, tp) -> dict:
    h = head(tp)
    return {n: h + t[k].to_csv(index=False, lineterminator="\n", float_format="%.12g") for k, n in zip(KEYS, OUTS)}


def inputs_sha() -> dict:
    ps = [str(p.relative_to(ROOT)) for p in FILES.values()] + ["data/regen/harj_x1/W5/w5g.parquet", "docs/otchety/harj_struct/S_L_SPEC.md", "docs/otchety/harj_struct/S_D_ELEMENTS.csv", "models_v2/s18/l_model.py", "models_v2/s18/l_walk.py", "models_v2/s18/r_loss.py", "models_v2/s18/r_walk.py"]
    return {p: _sha(ROOT / p) for p in ps}


def merge_checkpoint(ckpt: Path, t: dict, tx: dict, inp: dict, now: str | None = None) -> None:
    text = ckpt.read_text(encoding="utf-8")
    cp = json.loads(text)
    if json.dumps(cp, indent=1, sort_keys=True, ensure_ascii=False) != text:
        raise SystemExit("CHECKPOINT.json: формат не indent 1 / sorted keys — слияние ключа сломало бы прочие ключи")
    code, v = Path(__file__).resolve().parent, t["verdict"].set_index("id")
    cp[KEY] = dict(written_at=now or RR.now_text(), inputs=inputs_sha(), outputs={k: hashlib.sha256(x.encode("utf-8")).hexdigest() for k, x in tx.items()}, code={p.name: _sha(p) for p in sorted(code.glob("l_eval*.py")) + sorted(code.glob("l_doc*.py"))},
                   boot=S.B, boot_block=S.MEAN_BLOCK, boot_seed=S.SEED, lag_main=S.nw_lag(N_D), n=N_D, n_holm=inp["N"], n_holm_wide=inp["N_wide"], ktrim=LT.KTRIM, interaction=str(v.loc["interaction", "value"]), interaction_ref=str(v.loc["interaction_ref", "value"]), stage=str(v.loc["этап S", "value"]),
                   rck_verdict=str(v.loc["5.4", "value"]), predictions={i: str(v.loc[i, "value"]) for i in ("P-L1", "P-L2", "P-L3", "P-L4")})
    ckpt.write_text(json.dumps(cp, indent=1, sort_keys=True, ensure_ascii=False), encoding="utf-8")


def write(rep: Path = REP, ckpt: Path = CKPT) -> dict:
    inp = load_inputs(ckpt)
    t = compute(inp)
    tx = texts(t, inp["tp"])
    merge_checkpoint(ckpt, t, tx, inp)                  # ключ S.L.eval — до таблиц с p-значениями
    for k, v in tx.items():
        (rep / k).write_text(v, encoding="utf-8")
    return t


def tree_hash(rep: Path = REP, ckpt: Path = CKPT) -> dict:
    ps = [rep / n for n in OUTS + (REPORT,) if (rep / n).exists()] + [ckpt] + list(FILES.values())
    return {str(p): _sha(p) for p in ps}


def check(rep: Path = REP, ckpt: Path = CKPT) -> list:
    from models_v2.s18 import l_eval_chk as C
    h0, inp = tree_hash(rep, ckpt), load_inputs(ckpt)
    t = compute(inp)
    tx = texts(t, inp["tp"])
    bad = [] if all((rep / n).exists() and (rep / n).read_text(encoding="utf-8") == v for n, v in tx.items()) else ["таблицы ≠ пересчёту"]
    bad += C.problems(t, inp, rep, ckpt, tx)
    v = t["verdict"].set_index("id")
    print("взаимодействие:", v.loc["interaction", "value"], "| этап S:", v.loc["этап S", "value"], "| 5.4:", v.loc["5.4", "value"], "| предсказания:", {i: v.loc[i, "value"] for i in ("P-L1", "P-L2", "P-L3", "P-L4")})
    return bad + ([] if tree_hash(rep, ckpt) == h0 else ["--check изменил выходы"])


if __name__ == "__main__":
    if "--write" in sys.argv:
        t = write()
        print(t["verdict"][["id", "value"]].to_string())
    elif "--check" in sys.argv:
        b = check()
        print("S-L eval --check: %s" % ("ok" if not b else "FAIL " + "; ".join(b)))
        sys.exit(1 if b else 0)
    else:
        sys.exit("укажите --write или --check")
