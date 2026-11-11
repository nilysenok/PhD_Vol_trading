"""S-D-EVAL (Ж-256; `S_D_SPEC.md`, разд. 5, 6): оценка абляций S-RCKm, S-RCKf на D_M (t⁺ 2018-01-04…2026-11-10, n = 2198). Статус: DEVELOPMENT / HISTORICAL WALK-FORWARD,
диагностика вклада элементов R, исследовательский результат, не OOS; абляции не кандидаты. Читает СОХРАНЁННЫЕ `R/{rck,rckb}.parquet`, `D/{rckm,rckf}.parquet` и ряды X1, W5-G, K2, K3; прогнозов и реестр не трогает.
Публичное API:
- `KEYS` → 7 таблиц `docs/otchety/harj_struct/S_D_{TABLE,DELTA,ELEMENTS,YEARS,SLICES,MECH,PRED}.csv` (`OUTS`); `S_D_REPORT.md` — `d_doc`; `KEY` = "S.D.eval".
- `load_inputs() -> dict`, `compute(inp) -> dict` таблиц (вердикты и предсказания выводятся из таблиц), `write()`: ключ `S.D.eval` (written_at, хэши входов, выходов, кода) в CHECKPOINT пишется ПЕРВЫМ слиянием, затем таблицы.
- `check()` — только чтение (`d_eval_chk.problems`; sha256 выходов до и после). Запуск: `python3 -W ignore -m models_v2.s18.d_eval [--write | --check]`."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from models_v2.s16 import x1_w8_diag_core as K
from models_v2.s18 import a_eval_stat as S
from models_v2.s18 import cm_layers as CL
from models_v2.s18 import d_eval_par as PAR
from models_v2.s18 import d_eval_tab as T
from models_v2.s18 import d_reg as DR
from models_v2.s18 import r_reg as RR

ROOT, REP, CKPT, D_DIR, R_DIR = RR.ROOT, RR.ROOT / "docs/otchety/harj_struct", RR.CKPT, DR.D_DIR, RR.R_DIR
KEY, FIRST, LAST, N_D = "S.D.eval", "2018-01-04", "2026-11-10", 2198
KEYS = ("table", "delta", "elements", "years", "slices", "mech", "pred")
OUTS = tuple("S_D_%s.csv" % k.upper() for k in KEYS)
FILES = {"S-RCK": R_DIR / "rck.parquet", "S-RCKb": R_DIR / "rckb.parquet", "S-RCKm": D_DIR / "rckm.parquet", "S-RCKf": D_DIR / "rckf.parquet"}
MARK = "диагностика вклада элементов R, исследовательский результат, не OOS; абляции S-RCKm, S-RCKf не кандидаты (Ж-256); критерии разд. 5 S_D_SPEC.md объявлены до счёта; Холма и SPA нет, p справочно"
XS, BASES = ("S-RCKm", "S-RCKf"), ("S-RCK", "W5-G", "X1", "K2", "K3")
PAIRS_D = [(x, b) for x in XS for b in BASES] + [(x, "S-RCKb") for x in XS] + [("X1_R(%s)" % x, "X1") for x in XS] + [("S-RCK", b) for b in BASES[1:] + ("S-RCKb",)] + [("X1_R(S-RCK)", "X1")]
PAIRS_Y = [(x, b) for x in XS for b in ("S-RCK", "W5-G")] + [("S-RCK", "W5-G")]
PAIRS_S = [(x, b) for x in XS for b in ("S-RCK", "W5-G", "X1")] + [("S-RCKm", "S-RCKb"), ("S-RCK", "W5-G")]


def _sha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def head(tp) -> str:
    tp = pd.DatetimeIndex(tp)
    return "# %s\n# D_M: t⁺ %s…%s, n = %d; l = RV/F − ln(RV/F) − 1, QLIKE = среднее дневных потерь (без обрезки); Δ в %% QLIKE базы, минус — лучше; DEVELOPMENT / HISTORICAL WALK-FORWARD\n" % (MARK, tp.min().date(), tp.max().date(), len(tp))


def load_inputs() -> dict:
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
    return dict(x=x, rv=x["rv"].to_numpy(float), tp=pd.DatetimeIndex(x["t_plus"]), F={k: np.asarray(v, float) for k, v in F.items()}, frames=fr)


def compute(inp: dict) -> dict:
    rv, tp, F = inp["rv"], inp["tp"], inp["F"]
    ls = T.losses(rv, F)
    t = dict(table=T.table(rv, tp, F), delta=T.deltas(ls, tp, PAIRS_D), years=T.years(tp, ls, PAIRS_Y), slices=T.slices(tp, ls, PAIRS_S), mech=PAR.mech(inp["frames"], inp["x"], CL.inputs()[1], F["W5-G"]))
    el = T.elements(t["delta"])
    codes = {"iii": el["code"].iloc[0], "i": el["code"].iloc[1]}
    t["elements"] = pd.concat([el, pd.DataFrame([dict(element="сводный вывод", series="", base="", code="(i) %s; (iii) %s" % (codes["i"], codes["iii"]), verdict=T.summary(codes), main="да")])], ignore_index=True)
    t["pred"] = PAR.predictions(t["mech"])
    ov = T.trim_overlap(tp, ls, ("S-RCKm", "S-RCK"), ("S-RCKf", "S-RCK"))
    t["mech"] = pd.concat([t["mech"], pd.DataFrame([dict(section="trim_overlap", series="S-RCKm|S-RCKf к S-RCK", item=k, value=float(v)) for k, v in ov.items()])], ignore_index=True)
    return t


def texts(t: dict, tp) -> dict:
    h = head(tp)
    return {n: h + t[k].to_csv(index=False, lineterminator="\n", float_format="%.12g") for k, n in zip(KEYS, OUTS)}


def inputs_sha() -> dict:
    ps = [str(p.relative_to(ROOT)) for p in FILES.values()] + ["data/regen/harj_x1/W5/w5g.parquet", "data/regen/harj_struct/CM/feat.parquet", "docs/otchety/harj_struct/S_D_SPEC.md", "models_v2/s18/r_walk.py", "models_v2/s18/r_walk_ctx.py", "models_v2/s18/r_loss.py"]
    return {p: _sha(ROOT / p) for p in ps}


def merge_checkpoint(ckpt: Path, t: dict, tx: dict, now: str | None = None) -> None:
    text = ckpt.read_text(encoding="utf-8")
    cp = json.loads(text)
    if json.dumps(cp, indent=1, sort_keys=True, ensure_ascii=False) != text:
        raise SystemExit("CHECKPOINT.json: формат не indent 1 / sorted keys — слияние ключа сломало бы прочие ключи")
    code, el = Path(__file__).resolve().parent, t["elements"]
    cp[KEY] = dict(written_at=now or RR.now_text(), inputs=inputs_sha(), outputs={k: hashlib.sha256(v.encode("utf-8")).hexdigest() for k, v in tx.items()}, code={p.name: _sha(p) for p in sorted(code.glob("d_eval*.py")) + sorted(code.glob("d_doc*.py"))},
                   boot=S.B, boot_block=S.MEAN_BLOCK, boot_seed=S.SEED, lag_main=S.nw_lag(N_D), n=N_D, ktrim=T.KTRIM, elements=dict(zip(el["series"].iloc[:2] + "|" + el["element"].iloc[:2], el["code"].iloc[:2])), summary=el["verdict"].iloc[-1],
                   predictions=dict(zip(t["pred"]["id"], t["pred"]["fulfilled"])))
    ckpt.write_text(json.dumps(cp, indent=1, sort_keys=True, ensure_ascii=False), encoding="utf-8")


def write(rep: Path = REP, ckpt: Path = CKPT) -> dict:
    inp = load_inputs()
    t = compute(inp)
    tx = texts(t, inp["tp"])
    merge_checkpoint(ckpt, t, tx)                  # ключ S.D.eval — до таблиц
    for k, v in tx.items():
        (rep / k).write_text(v, encoding="utf-8")
    return t


def tree_hash(rep: Path = REP, ckpt: Path = CKPT) -> dict:
    ps = [rep / n for n in OUTS + ("S_D_REPORT.md",) if (rep / n).exists()] + [ckpt] + list(FILES.values())
    return {str(p): _sha(p) for p in ps}


def check(rep: Path = REP, ckpt: Path = CKPT) -> list:
    from models_v2.s18 import d_eval_chk as C
    h0, inp = tree_hash(rep, ckpt), load_inputs()
    t = compute(inp)
    tx = texts(t, inp["tp"])
    bad = [] if all((rep / n).exists() and (rep / n).read_text(encoding="utf-8") == v for n, v in tx.items()) else ["таблицы ≠ пересчёту"]
    bad += C.problems(t, inp, rep, ckpt, tx)
    print("вердикты:", t["elements"][["element", "code"]].to_dict("records"), "| предсказания:", dict(zip(t["pred"]["id"], t["pred"]["fulfilled"])))
    return bad + ([] if tree_hash(rep, ckpt) == h0 else ["--check изменил выходы"])


if __name__ == "__main__":
    if "--write" in sys.argv:
        t = write()
        print("вердикты:", dict(zip(t["elements"]["element"], t["elements"]["code"])), "| сводный:", t["elements"]["verdict"].iloc[-1])
    elif "--check" in sys.argv:
        b = check()
        print("S-D eval --check: %s" % ("ok" if not b else "FAIL " + "; ".join(b)))
        sys.exit(1 if b else 0)
    else:
        sys.exit("укажите --write или --check")
