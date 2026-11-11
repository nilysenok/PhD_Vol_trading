"""S-R-EVAL (Ж-254 пп. 6, 10–12, 16; `S_R_SPEC.md`, разд. 4, 6): оценка процедуры R (S-RA2 / S-RCK и абляции) на D_M (t⁺ 2018-01-04…2026-11-10, n = 2198). Статус: DEVELOPMENT / HISTORICAL WALK-FORWARD,
исследовательский результат на раскрытой истории, процедура выбрана после разбора S-C-WHY, не независимый OOS. Читает СОХРАНЁННЫЕ прогнозы `data/regen/harj_struct/R/*.parquet`, `A/a2.parquet`, `CM/ck.parquet`, ряды X1, W5-G, W8-G, K1–K3, ZB; прогнозов и реестр не трогает.
Обвязка — `a_eval_*` (DM NW-8, стационарный бутстрап B = 10 000, Холм, SPA / StepM, MCS, мощность); здесь — параметры семейств R (`r_eval_tab`, `r_eval_yr`, `r_eval_par`).
Публичное API:
- `KEYS` — 12 таблиц `docs/otchety/harj_struct/S_R_{TABLE,STATS,PROC,ABL,YEARS,CONC,GUARD,SPA,MCS,POWER,PARAMS,SEL}.csv` (`OUTS`); `KEY` = "S.R.eval".
- `load_inputs() -> dict`: ряды F (K1…ZB, S-A2, S-CK, 6 вариантов R, слои X1_R, X1_A(S-A2), X1_V(S-CK), F_fr) на одних датах; сверка d, rv, cfg8 с окном M — расхождение SystemExit.
- `compute(inp) -> dict` таблиц, `decision` (NEWCTRL-R), `proc` (П1, П2), `identity`; `texts(t, tp)`.
- `write()`: ключ `S.R.eval` (written_at, хэши входов, выходов, кода) в `docs/otchety/harj_x1/CHECKPOINT.json` пишется ПЕРВЫМ (слиянием), затем таблицы с p. `check()` — только чтение: `r_eval_chk.problems`; `check_spa_cli()`.
Запуск: `python3 -W ignore -m models_v2.s18.r_eval [--write | --check | --check-spa]`."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from models_v2.s16 import x1_w5_eval as E5, x1_w8_diag_core as K
from models_v2.s18 import a_eval as AE, a_eval_spa as SP, a_eval_stat as S, a_reg as AR
from models_v2.s18 import r_eval_par as PAR, r_eval_tab as T, r_eval_yr as Y, r_walk as W

ROOT, A_DIR, CKPT = AR.ROOT, AR.A_DIR, AR.CKPT
R_DIR, REP = W.R_DIR, ROOT / "docs/otchety/harj_struct"
KEY = "S.R.eval"
KEYS = ("table", "stats", "proc", "abl", "years", "conc", "guard", "spa", "mcs", "power", "params", "sel")
OUTS = tuple("S_R_%s.csv" % k.upper() for k in KEYS)
FIRST, LAST, N_D, EPS = AE.FIRST, AE.LAST, AE.N_D, 1e-12
MCS_SET = ("S-RA2", "S-RCK", "S-RA2b", "S-RCKb", "S-RA2c", "S-RCKc", "S-A2", "S-CK", "W5-G", "X1", "W8-G", "K2", "K3", "B_mix")
MARK = "исследовательский результат, процедура выбрана после разбора S-C-WHY, не независимый OOS (Ж-254 п. 11)"
MARKS = (MARK, "критерии разд. 4 S_R_SPEC.md объявлены и зарегистрированы до счёта; «подтверждено» только при p Холма (N программы) < 0,05 и p SPA < 0,05 по своей строке")


def _sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def head(tp) -> str:
    tp = pd.DatetimeIndex(tp)
    return "\n".join("# " + m for m in MARKS) + "\n# D_M: t⁺ %s…%s, n = %d; l = RV/F − ln(RV/F) − 1, QLIKE = среднее дневных потерь (без обрезки); Δ в %% QLIKE базы, минус — лучше; DEVELOPMENT / HISTORICAL WALK-FORWARD\n" % (tp.min().date(), tp.max().date(), len(tp))


def _aligned(a: pd.DataFrame, x: pd.DataFrame, name: str, cfg: str, layer: str) -> None:
    ok = len(a) == len(x) and (a["d"].to_numpy() == x["d"].to_numpy()).all() and np.array_equal(a["rv"].to_numpy(), x["rv"].to_numpy())
    ok = ok and np.allclose(a[cfg], x["C"], rtol=1e-12, atol=0) and np.isfinite(a["F"]).all() and np.isfinite(a[layer]).all()
    if not ok:
        raise SystemExit("%s: даты, rv, cfg8 расходятся с окном M или есть нечисловые прогнозы" % name)


def load_inputs() -> dict:
    x, w8 = K.load(), pd.read_parquet(AE.W8)
    if not (len(x) == N_D and str(x["t_plus"].min().date()) == FIRST and str(x["t_plus"].max().date()) == LAST):
        raise SystemExit("окно M ≠ D_M (2198 пар, t⁺ %s…%s)" % (FIRST, LAST))
    F = {"K1": x["K1"], "K2": x["K2"], "K3": x["K3"], "X1": x["F"], "W5-G": x["W5G"], "W8-G": w8["F"], "B_mix": x["M"], "ZB": E5._zb(x["d"])}
    a2, ck = pd.read_parquet(A_DIR / "a2.parquet"), pd.read_parquet(ROOT / "data/regen/harj_struct/CM/ck.parquet")
    for n, df, lay in (("S-A2", a2, "X1_A"), ("S-CK", ck, "X1_V")):
        ok = len(df) == len(x) and (df["d"].to_numpy() == x["d"].to_numpy()).all() and np.array_equal(df["rv"].to_numpy(), x["rv"].to_numpy())
        if not ok:
            raise SystemExit("%s: даты или rv расходятся с окном M" % n)
        F[n], F["X1_A(S-A2)" if n == "S-A2" else "X1_V(S-CK)"] = df["F"], df[lay]
    frames = {}
    for v, k in PAR.KEY.items():
        a = pd.read_parquet(R_DIR / (k + ".parquet"))
        _aligned(a, x, v, "cfg8", "X1_R")
        frames[v], F[v], F["X1_R(%s)" % v], F["Ffr(%s)" % v] = a, a["F"], a["X1_R"], a["F_fr"]
    if not ((w8["d"].to_numpy() == x["d"].to_numpy()).all() and np.array_equal(w8["rv"].to_numpy(), x["rv"].to_numpy())):
        raise SystemExit("W8-G: даты или rv расходятся с окном M")
    return dict(x=x, rv=x["rv"].to_numpy(float), tp=pd.DatetimeIndex(x["t_plus"]), d=pd.DatetimeIndex(x["d"]), F={k: np.asarray(v, float) for k, v in F.items()}, frames=frames)


def compute(inp: dict) -> dict:
    rv, tp, F = inp["rv"], inp["tp"], inp["F"]
    ls = T.losses(rv, F)
    spa, pspa = T.spa_blocks(ls), T.proc_spa(ls)
    ctxs = {"A": W.make_ctx("A"), "CK": W.make_ctx("CK")}
    t = dict(table=T.table(rv, tp, F), stats=T.stats(rv, tp, F, ls, spa, pspa), years=Y.years(tp, ls), conc=Y.conc(tp, ls), spa=T.spa_table(spa, pspa), power=Y.power_table(tp, ls),
             params=PAR.params(inp["frames"], PAR.read_cost()), sel=PAR.sel(inp["frames"]), abl=PAR.decomp(ls, inp["d"]), mcs=S.mcs(rv, {k: F[k] for k in MCS_SET}))
    ins = PAR.insample(inp["frames"], ctxs)
    t["conc"] = pd.concat([t["conc"], pd.DataFrame(dict(series=ins["variant"], base="—", item="insample_" + ins["item"], key="", value=ins["value"]))], ignore_index=True)
    t["guard"] = Y.guard(t["table"], t["years"], t["conc"])
    t["proc"] = Y.proc_table(t["table"], ls, t["stats"])
    t["decision"], t["proc_decision"], t["identity"] = Y.decide(t["guard"]), Y.decide_proc(t["proc"]), Y.identity_gap(t["years"])
    return t


def texts(t: dict, tp) -> dict:
    h = head(tp)
    return {n: h + t[k].to_csv(index=False, lineterminator="\n", float_format="%.12g") for k, n in zip(KEYS, OUTS)}


def inputs_sha() -> dict:
    from models_v2.s18 import cm_eval as CE, r_loss_chk as LC
    ps = list(AE.SPEC_SHA) + list(CE.C_SHA) + list(AE.EXTRA) + list(LC.R_SHA) + ["data/regen/harj_struct/R/" + k + ".parquet" for k in PAR.KEY.values()]
    return {p: _sha(ROOT / p) for p in dict.fromkeys(ps)}


def merge_checkpoint(ckpt: Path, t: dict, tx: dict, now: str | None = None) -> None:
    text = ckpt.read_text(encoding="utf-8")
    cp = json.loads(text)
    if json.dumps(cp, indent=1, sort_keys=True, ensure_ascii=False) != text:
        raise SystemExit("CHECKPOINT.json: формат не indent 1 / sorted keys — слияние ключа сломало бы прочие ключи")
    code = Path(__file__).resolve().parent
    cp[KEY] = dict(written_at=now or AR.now_text(), inputs=inputs_sha(), outputs={k: hashlib.sha256(v.encode("utf-8")).hexdigest() for k, v in tx.items()}, code={p.name: _sha(p) for p in sorted(code.glob("r_eval*.py")) + sorted(code.glob("r_doc*.py"))},
                   n_holm=T.N_HOLM, n_holm_wide=T.N_WIDE, m_family=T.M_FAMILY, m_proc=T.M_PROC, boot=S.B, lag_main=S.nw_lag(N_D), ktrim=T.KTRIM, decision=t["decision"], proc=t["proc_decision"], n=N_D)
    ckpt.write_text(json.dumps(cp, indent=1, sort_keys=True, ensure_ascii=False), encoding="utf-8")


def write(rep: Path = REP, ckpt: Path = CKPT) -> dict:
    inp = load_inputs()
    t = compute(inp)
    tx = texts(t, inp["tp"])
    merge_checkpoint(ckpt, t, tx)                  # ключ S.R.eval — до таблиц с p-значениями
    for k, v in tx.items():
        (rep / k).write_text(v, encoding="utf-8")
    return t


def tree_hash(rep: Path = REP, ckpt: Path = CKPT) -> dict:
    ps = [rep / n for n in OUTS if (rep / n).exists()] + [ckpt] + sorted(R_DIR.glob("*.parquet")) + sorted(A_DIR.glob("*.parquet")) + sorted((ROOT / "data/regen/harj_struct/CM").glob("*.parquet")) + [rep / "S_R_REPORT.md"] * (rep / "S_R_REPORT.md").exists()
    return {str(p): _sha(p) for p in ps}


def check(rep: Path = REP, ckpt: Path = CKPT) -> list:
    from models_v2.s18 import r_eval_chk as C
    h0, inp = tree_hash(rep, ckpt), load_inputs()
    t = compute(inp)
    tx = texts(t, inp["tp"])
    bad = [] if all((rep / n).exists() and (rep / n).read_text(encoding="utf-8") == v for n, v in tx.items()) else ["таблицы ≠ пересчёту"]
    bad += C.problems(t, rep, ckpt, tx)
    print("решение:", t["decision"], "|", t["proc_decision"])
    return bad + ([] if tree_hash(rep, ckpt) == h0 else ["--check изменил выходы"])


def check_spa_cli() -> list:
    inp = load_inputs()
    rv, F = inp["rv"], inp["F"]
    d = S.Q.ql(rv, F["W5-G"]) - S.Q.ql(rv, F["S-RA2"])
    return SP.check_spa(S.sigma_lr(d, S.nw_lag(len(d))), float(S.Q.ql(rv, F["W5-G"]).mean()), len(d))


if __name__ == "__main__":
    if "--write" in sys.argv:
        t = write()
        print("решение:", t["decision"], "|", t["proc_decision"])
    elif "--check" in sys.argv:
        b = check()
        print("S-R eval --check: %s" % ("ok" if not b else "FAIL " + "; ".join(b)))
        sys.exit(1 if b else 0)
    elif "--check-spa" in sys.argv:
        b = check_spa_cli()
        print("S-R eval --check-spa: %s" % ("ok" if not b else "FAIL " + "; ".join(b)))
        sys.exit(1 if b else 0)
    else:
        sys.exit("укажите --write, --check или --check-spa")
