"""S-C-EVAL (Ж-254 пп. 9, 11, 12, 16; `S_C_SPEC.md`, разд. 4, 7): оценка компонентного гибрида S-CD / S-CK / S-CKA / S-C7 / S-CKL на D_M (t⁺ 2018-01-04…2026-11-10, n = 2198). Статус: DEVELOPMENT / HISTORICAL WALK-FORWARD,
исследовательский результат на раскрытой истории, не независимый OOS. Читает СОХРАНЁННЫЕ прогнозы `data/regen/harj_struct/CM/*.parquet`, ряды X1, W5-G, W8-G, K1–K3, ZB, для сводки — `A/a2.parquet`; прогнозов и реестр не трогает.
Обвязка — `a_eval_*` (DM NW-8, стационарный бутстрап B = 10 000, Холм, SPA / StepM, MCS, мощность); здесь — параметры семейств C (`cm_eval_tab`, `cm_eval_yr`, `cm_eval_par`).
Публичное API:
- `OUTS` — 10 CSV `docs/otchety/harj_struct/S_C_{TABLE,STATS,YEARS,CONC,GUARD,SPA,MCS,POWER,PARAMS,LNM}.csv`; `KEY` = "S.C.eval"; `C_SHA` — sha256 (16 знаков) входов C по спецификации разд. 7.
- `load_inputs() -> dict`: ряды F (K1, K2, K3, X1, W5-G, W8-G, B_mix, ZB, S-A2, S-CD…S-CKL и слои X1_C(S-C*)) на одних датах; сверка d, rv, cfg8 с окном M; расхождение — SystemExit.
- `compute(inp) -> dict` таблиц, `decision` (NEWCTRL), `decision_struct`, `identity`; `texts(t, tp)`.
- `write()`: ключ `S.C.eval` (written_at, хэши входов, выходов, кода) в `docs/otchety/harj_x1/CHECKPOINT.json` пишется ПЕРВЫМ (слиянием: прочие ключи побайтно; indent 1 / sorted keys), затем таблицы с p.
- `check()` — только чтение: `cm_eval_chk.problems`; `check_spa_cli()`: синтетика с известным ответом (`a_eval_spa.check_spa`) с σ_LR разности W5-G − S-CK.
Запуск: `python3 -W ignore -m models_v2.s18.cm_eval [--write | --check | --check-spa]`."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from models_v2.s16 import x1_w5_eval as E5, x1_w8_diag_core as K
from models_v2.s18 import a_eval as AE, a_eval_spa as SP, a_eval_stat as S, a_reg as R
from models_v2.s18 import cm_eval_par as PAR, cm_eval_tab as T, cm_eval_yr as Y

ROOT, A_DIR, CKPT = R.ROOT, R.A_DIR, R.CKPT
CM_DIR, REP = ROOT / "data/regen/harj_struct/CM", ROOT / "docs/otchety/harj_struct"
KEY = "S.C.eval"
KEYS = ("table", "stats", "years", "conc", "guard", "spa", "mcs", "power", "params", "lnm")
OUTS = tuple("S_C_%s.csv" % k.upper() for k in KEYS)
VFILES = {v: v.replace("S-", "").lower() + ".parquet" for v in T.VARS}
C_SHA = {"data/regen/harj_struct/C/C_sessions.parquet": "c71410531eec3275", "models_v2/s18/c_data.py": "bf8e49e0fb3addcd", "models_v2/s18/c_factor.py": "afdbb021b2dd08ea", "models_v2/s18/c_decomp.py": "1a711761edbb2c55"}
EXTRA = tuple(AE.EXTRA) + ("data/regen/harj_struct/A/a2.parquet", "data/regen/harj_struct/CM/feat.parquet")
FIRST, LAST, N_D, EPS = AE.FIRST, AE.LAST, AE.N_D, 1e-12
MCS_SET = ("S-CK", "S-CKA", "S-CD", "S-C7", "S-CKL", "W5-G", "X1", "W8-G", "K2", "K3", "B_mix")
MARKS = ("исследовательский результат на раскрытой истории, не независимый OOS (Ж-254 п. 11); архитектура выбрана после изучения истории",
         "структура S-CD/S-CK/S-CKA/S-C7/S-CKL и критерии разд. 4 S_C_SPEC.md объявлены и зарегистрированы до счёта; «подтверждено» только при p Холма (N программы) < 0,05 и p SPA < 0,05")


def _sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def head(tp) -> str:
    tp = pd.DatetimeIndex(tp)
    return "\n".join("# " + m for m in MARKS) + "\n# D_M: t⁺ %s…%s, n = %d; l = RV/F − ln(RV/F) − 1, QLIKE = среднее дневных потерь; Δ в %% QLIKE базы, минус — лучше; DEVELOPMENT / HISTORICAL WALK-FORWARD\n" % (
        tp.min().date(), tp.max().date(), len(tp))


def _aligned(a: pd.DataFrame, x: pd.DataFrame, name: str, cfg: bool) -> None:
    ok = len(a) == len(x) and (a["d"].to_numpy() == x["d"].to_numpy()).all() and np.array_equal(a["rv"].to_numpy(), x["rv"].to_numpy())
    ok = ok and (not cfg or np.allclose(a["cfg8"], x["C"], rtol=1e-12, atol=0))
    if not ok or not (np.isfinite(a["F"]).all() and np.isfinite(a["X1_V" if cfg else "X1_A"]).all()):
        raise SystemExit("%s: даты, rv, cfg8 расходятся с окном M или есть нечисловые прогнозы" % name)


def load_inputs() -> dict:
    x, w8 = K.load(), pd.read_parquet(AE.W8)
    if not (len(x) == N_D and str(x["t_plus"].min().date()) == FIRST and str(x["t_plus"].max().date()) == LAST):
        raise SystemExit("окно M ≠ D_M (2198 пар, t⁺ %s…%s)" % (FIRST, LAST))
    F = {"K1": x["K1"], "K2": x["K2"], "K3": x["K3"], "X1": x["F"], "W5-G": x["W5G"], "W8-G": w8["F"], "B_mix": x["M"], "ZB": E5._zb(x["d"])}
    a2 = pd.read_parquet(A_DIR / "a2.parquet")
    _aligned(a2, x, "S-A2", False)
    F["S-A2"], frames = a2["F"], {}
    for v, f in VFILES.items():
        a = pd.read_parquet(CM_DIR / f)
        _aligned(a, x, v, True)
        frames[v], F[v], F["X1_C(%s)" % v] = a, a["F"], a["X1_V"]
    if not ((w8["d"].to_numpy() == x["d"].to_numpy()).all() and np.array_equal(w8["rv"].to_numpy(), x["rv"].to_numpy())):
        raise SystemExit("W8-G: даты или rv расходятся с окном M")
    from models_v2.s18 import cm_feat as CF
    feat = CF.load_feat().reindex(pd.DatetimeIndex(x["d"]))
    return dict(x=x, rv=x["rv"].to_numpy(float), tp=pd.DatetimeIndex(x["t_plus"]), F={k: np.asarray(v, float) for k, v in F.items()}, frames=frames, feat_s=feat)


def compute(inp: dict) -> dict:
    rv, tp, F = inp["rv"], inp["tp"], inp["F"]
    ls = T.losses(rv, F)
    spa, sspa = T.spa_blocks(ls), T.struct_spa(ls)
    t = dict(table=T.table(rv, tp, F), stats=T.stats(rv, tp, F, ls, spa, sspa), years=Y.years(tp, ls), conc=Y.conc(tp, ls), spa=T.spa_table(spa, sspa), power=Y.power_table(tp, ls),
             params=PAR.params(inp["frames"]), lnm=PAR.lnm(inp["frames"], inp["feat_s"]), mcs=S.mcs(rv, {k: F[k] for k in MCS_SET}))
    t["guard"] = Y.guard(t["table"], t["years"], t["conc"], t["stats"])
    t["decision"], t["decision_struct"], t["identity"] = Y.decide(t["guard"]), Y.decide_struct(t["guard"]), Y.identity_gap(t["years"])
    return t


def texts(t: dict, tp) -> dict:
    h = head(tp)
    return {n: h + t[k].to_csv(index=False, lineterminator="\n", float_format="%.12g") for k, n in zip(KEYS, OUTS)}


def inputs_sha() -> dict:
    ps = list(AE.SPEC_SHA) + list(C_SHA) + list(EXTRA) + ["data/regen/harj_struct/CM/" + f for f in VFILES.values()]
    return {p: _sha(ROOT / p) for p in ps}


def merge_checkpoint(ckpt: Path, t: dict, tx: dict, now: str | None = None) -> None:
    text = ckpt.read_text(encoding="utf-8")
    cp = json.loads(text)
    if json.dumps(cp, indent=1, sort_keys=True, ensure_ascii=False) != text:
        raise SystemExit("CHECKPOINT.json: формат не indent 1 / sorted keys — слияние ключа сломало бы прочие ключи")
    code = Path(__file__).resolve().parent
    cp[KEY] = dict(written_at=now or R.now_text(), inputs=inputs_sha(), outputs={k: hashlib.sha256(v.encode("utf-8")).hexdigest() for k, v in tx.items()},
                   code={p.name: _sha(p) for p in sorted(code.glob("cm_eval*.py")) + sorted(code.glob("cm_doc*.py"))}, n_holm=T.N_HOLM, n_holm_wide=T.N_WIDE, m_family=T.M_FAMILY, m_struct=T.M_STRUCT,
                   boot=S.B, lag_main=S.nw_lag(N_D), decision=t["decision"], decision_struct=t["decision_struct"], n=N_D)
    ckpt.write_text(json.dumps(cp, indent=1, sort_keys=True, ensure_ascii=False), encoding="utf-8")


def write(rep: Path = REP, ckpt: Path = CKPT) -> dict:
    inp = load_inputs()
    t = compute(inp)
    tx = texts(t, inp["tp"])
    merge_checkpoint(ckpt, t, tx)                  # ключ S.C.eval — до таблиц с p-значениями
    for k, v in tx.items():
        (rep / k).write_text(v, encoding="utf-8")
    return t


def tree_hash(rep: Path = REP, ckpt: Path = CKPT) -> dict:
    ps = [rep / n for n in OUTS if (rep / n).exists()] + [ckpt] + sorted(CM_DIR.glob("*.parquet")) + sorted(A_DIR.glob("*.parquet")) + [rep / "S_C_REPORT.md"] * (rep / "S_C_REPORT.md").exists()
    return {str(p): _sha(p) for p in ps}


def check(rep: Path = REP, ckpt: Path = CKPT) -> list:
    from models_v2.s18 import cm_eval_chk as C
    h0, inp = tree_hash(rep, ckpt), load_inputs()
    t = compute(inp)
    tx = texts(t, inp["tp"])
    bad = [] if all((rep / n).exists() and (rep / n).read_text(encoding="utf-8") == v for n, v in tx.items()) else ["таблицы ≠ пересчёту"]
    bad += C.problems(t, rep, ckpt, tx)
    print("решение:", t["decision"], "|", t["decision_struct"])
    return bad + ([] if tree_hash(rep, ckpt) == h0 else ["--check изменил выходы"])


def check_spa_cli() -> list:
    inp = load_inputs()
    rv, F = inp["rv"], inp["F"]
    d = S.Q.ql(rv, F["W5-G"]) - S.Q.ql(rv, F["S-CK"])
    return SP.check_spa(S.sigma_lr(d, S.nw_lag(len(d))), float(S.Q.ql(rv, F["W5-G"]).mean()), len(d))


if __name__ == "__main__":
    if "--write" in sys.argv:
        t = write()
        print("решение:", t["decision"], "|", t["decision_struct"])
    elif "--check" in sys.argv:
        b = check()
        print("S-C eval --check: %s" % ("ok" if not b else "FAIL " + "; ".join(b)))
        sys.exit(1 if b else 0)
    elif "--check-spa" in sys.argv:
        b = check_spa_cli()
        print("S-C eval --check-spa: %s" % ("ok" if not b else "FAIL " + "; ".join(b)))
        sys.exit(1 if b else 0)
    else:
        sys.exit("укажите --write, --check или --check-spa")
