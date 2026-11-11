"""S-R-EVAL (`S_R_SPEC.md`, гейт E.G1): независимые проверки оценки R — `problems(t, rep, ckpt, tx)`; самотесты порядка, П и Δ_trim на синтетике. Только чтение."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from models_v2.s18 import a_eval_stat as S
from models_v2.s18 import r_eval as E
from models_v2.s18 import r_eval_tab as T
from models_v2.s18 import r_eval_yr as Y

SPEC_ORDER = ("S-RA2", "S-RCK")


def _g(cands: dict) -> pd.DataFrame:
    rows = [dict(candidate=c, pair="—", id="i%d" % i, condition="", value=0.0, threshold="—", formula="", passed=ok) for c, oks in cands.items() for i, ok in enumerate(oks)]
    return pd.DataFrame(rows)


def order_selftest() -> list:
    ok, no = [True] * 10, [True] * 9 + [False]
    r1, r2, r3 = Y.decide(_g({"S-RA2": ok, "S-RCK": ok})), Y.decide(_g({"S-RA2": no, "S-RCK": ok})), Y.decide(_g({"S-RA2": no, "S-RCK": no}))
    good = tuple(Y.ORDER) == SPEC_ORDER and r1.startswith("новая контрольная версия: S-RA2 ") and r2.startswith("новая контрольная версия: S-RCK ") and r3.startswith("не прошёл")
    return [] if good else ["порядок кандидатов S-RA2 → S-RCK или правило решения нарушены"]


def trim_selftest() -> list:
    """Известный ответ: весь выигрыш на одной дате — Δ_trim ≈ без неё; K = 22 с обеих сторон; база процентов — на оставшихся датах."""
    n, rng = 2198, np.random.default_rng(0)
    lb = np.ones(n)
    lh = lb + rng.normal(0, 0.01, n)
    lh[10] = -50.0
    d = lh - lb
    keep = np.ones(n, bool)
    o = np.argsort(d, kind="stable")
    keep[o[:22]], keep[o[-22:]] = False, False
    want = 100.0 * d[keep].sum() / lb[keep].sum()
    got = T.trim_delta(lh, lb)
    full = 100.0 * d.sum() / lb.sum()
    one_side = 100.0 * d[np.sort(o[22:])].sum() / lb[np.sort(o[22:])].sum()
    good = abs(got - want) < 1e-12 and got > full and abs(got - one_side) > 1e-9 and T.KTRIM == 22 and T.KTRIM == math.ceil(0.01 * n)
    return [] if good else ["Δ_trim ≠ известному ответу (симметрично, K = 22, база — оставшиеся даты)"]


def proc_selftest() -> list:
    """П: новый лучше по QLIKE, но Δ_trim ≥ 0 ⇒ П не выполнено; П считается против ПРЕЖНЕГО ряда, а не W5-G."""
    n, rng = 2198, np.random.default_rng(1)
    rv = np.exp(rng.normal(0, 0.5, n))
    base = np.ones(n)
    ls = {k: S.Q.ql(rv, base * f) for k, f in (("S-RA2", 1.0), ("S-RCK", 1.0), ("S-RA2b", 1.0), ("S-RCKb", 1.0), ("S-RA2c", 1.0), ("S-RCKc", 1.0), ("S-A2", 1.0), ("S-CK", 1.0), ("W5-G", 5.0))}
    ls["S-RA2"] = ls["S-A2"].copy()
    ls["S-RA2"][100] = ls["S-A2"][100] - 0.5                       # весь выигрыш — на одной дате (внутри обрезки)
    ls["S-RCK"] = ls["S-CK"] - 1e-3                                # равномерный выигрыш
    tbl = pd.DataFrame(dict(series=list(ls), qlike=[float(v.mean()) for v in ls.values()]))
    st = pd.DataFrame(dict(kind=["proc", "proc"], hypothesis=["П1", "П2"], p_holm_N=[0.01, 0.01], p_spa_c=[0.01, 0.01]))
    r = Y.proc_table(tbl, ls, st).set_index("pid")
    good = (not r.at["П1", "passed"]) and r.at["П1", "cond_qlike"] and (not r.at["П1", "cond_trim"]) and r.at["П2", "passed"] and r.at["П2", "confirmed"] and r.at["П2", "old"] == "S-CK"
    return [] if good else ["П1 / П2: правило (QLIKE ∧ Δ_trim, против прежнего ряда, «подтверждено» по своей строке) нарушено"]


def _holm_problems(st: pd.DataFrame) -> list:
    bad = []
    if not ((st["n_holm"] == 846).all() and (st["n_holm_wide"] == 3397).all() and (st["B"] == S.B).all() and S.B == 10000):
        bad.append("Холм: N ≠ 846 / N_wide ≠ 3397 или B ≠ 10 000")
    for kind, mm in (("main", 8), ("proc", 2)):
        p = st[st["kind"] == kind]
        if not (p["m_family"] == mm).all():
            bad.append("m ≠ %d для %s" % (mm, kind))
        for col, N in (("p_holm_m", mm), ("p_holm_N", 846), ("p_holm_Nwide", 3397)):
            if not np.allclose(p[col].to_numpy(), S.holm_n(p["p_dm"].to_numpy(), N), rtol=0, atol=1e-15, equal_nan=True):
                bad.append("%s ≠ Холм(p DM NW-8, N = %d) для %s" % (col, N, kind))
    m = st[st["kind"] == "main"]
    bad += [] if (m["lag"] == S.nw_lag(2198)).all() and S.nw_lag(2198) == 8 else ["лаг NW основной колонки ≠ 8"]
    conf = m[m["verdict"].str.contains("подтверждено") & ~m["verdict"].str.contains("не подтверждено")]
    ok = m[(m["delta_pct"] < 0) & (m["p_holm_N"] < 0.05) & (m["p_spa_c"] < 0.05)]
    bad += [] if len(conf) == len(ok) and ((conf["p_holm_N"] < 0.05) & (conf["p_spa_c"] < 0.05)).all() else ["«подтверждено» не по своей строке"]
    return bad


def problems(t: dict, rep: Path, ckpt: Path, tx: dict) -> list:
    bad = [] if t["identity"] <= E.EPS else ["Σ вкладов ≠ Δ (%.3g > 1e-12)" % t["identity"]]
    bad += _holm_problems(t["stats"])
    if Y.THR != {"У1-S": 0.415, "У2-S": 0.50, "У3-S": 1.00} or Y.ORDER != SPEC_ORDER:
        bad.append("пороги или порядок ≠ спецификации")
    g, pc = pd.read_csv(rep / "S_R_GUARD.csv", comment="#"), pd.read_csv(rep / "S_R_PROC.csv", comment="#")
    bad += [] if Y.decide(g) == t["decision"] == g[g["id"] == "решение"]["value"].iloc[0] else ["решение NEWCTRL-R не выведено из строк GUARD"]
    bad += [] if Y.decide_proc(pc) == t["proc_decision"] else ["решение PROC не выведено из строк PROC"]
    bad += order_selftest() + trim_selftest() + proc_selftest()
    inp = E.load_inputs()
    for r in t["table"].itertuples():
        ql = float(np.mean(inp["rv"] / inp["F"][r.series] - np.log(inp["rv"] / inp["F"][r.series]) - 1.0))
        bad += [] if abs(ql - r.qlike) <= 1e-12 * max(1.0, abs(ql)) else ["QLIKE %s ≠ независимому расчёту без обрезки" % r.series]
    bad += [] if (t["mcs"]["n_used"] == E.N_D).all() and (t["table"]["n"] == E.N_D).all() else ["MCS или таблица не на 2198 датах"]
    bad += [] if (t["table"]["first_tplus"] == E.FIRST).all() and (t["table"]["last_tplus"] == E.LAST).all() else ["границы дат таблицы ≠ D_M"]
    from models_v2.s18 import r_loss_chk as LC
    bad += LC.sha_bad()
    cp = json.loads(ckpt.read_text(encoding="utf-8"))
    reg, key = cp.get("S.R.registered", {}), cp.get(E.KEY, {})
    bad += [] if (reg.get("n"), reg.get("n_wide")) == (846, 3397) else ["S.R.registered: N ≠ 846 / 3397"]
    want = {k: hashlib.sha256(v.encode("utf-8")).hexdigest() for k, v in tx.items()}
    return bad + ([] if key.get("outputs") == want and key.get("n_holm") == 846 and key.get("ktrim") == 22 else ["CHECKPOINT %s ≠ пересчёту" % E.KEY])
