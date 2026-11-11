"""S-A-EVAL (Ж-254; `S_A_SPEC.md`, разд. 8, гейт E.G1): проверки `a_eval --check`, только чтение (ничего не пишут).
Публичное API: `problems(t, rep, ckpt, tx) -> list` замечаний: Σ вкладов = Δ (1e-12); лаг NW основной колонки = 8, B = 10 000; Холм N = 831 / 3367, m = 12 пересчитан из p; пороги = константам спецификации;
решение выведено из строк GUARD (в том числе прочитанных из CSV), порядок A2 → A1 (самотест на синтетических строках); «подтверждено» только при p Холма < 0,05 и p SPA < 0,05; sha входов = спецификации;
ключ `S.A.eval` = хэшам выходов; константы λ / seeds `a_eval_par` = `a_walk`; MCS на всех 2198 датах; границы дат и n в таблицах."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from models_v2.s18 import a_eval as E
from models_v2.s18 import a_eval_par as PAR, a_eval_stat as S, a_eval_tab as T, a_eval_yr as Y

IDS10 = (("М1", "М1"), ("У1-S", "М1"), ("У2-S", "М1"), ("У3-S", "М1"), ("М2", "М2"), ("У1-S", "М2"), ("У2-S", "М2"), ("У3-S", "М2"), ("У4-S", "—"), ("У5-S", "М2"))


def _syn_guard(ok: dict) -> pd.DataFrame:
    rows = [dict(candidate=c, pair=p, id=i, value=0.0, passed=bool(ok[c])) for c in Y.ORDER for i, p in IDS10]
    return pd.DataFrame(rows + [dict(candidate=c, pair="—", id="кандидат", value="", passed=bool(ok[c])) for c in Y.ORDER])


def order_selftest() -> list:
    bad = []
    for ok, want in (({"S-A2": 1, "S-A1": 1}, "S-A2"), ({"S-A2": 0, "S-A1": 1}, "S-A1"), ({"S-A2": 1, "S-A1": 0}, "S-A2")):
        d = Y.decide(_syn_guard(ok))
        bad += [] if d.startswith("новая контрольная версия: " + want) else ["порядок A2 → A1: %s → %s" % (ok, d)]
    return bad + ([] if Y.decide(_syn_guard({"S-A2": 0, "S-A1": 0})).startswith("не прошёл") else ["оба не прошли ⇒ «не прошёл»"])


def _stats_problems(st: pd.DataFrame) -> list:
    m, bad = st[st["kind"] == "main"], []
    if len(m) != T.M_FAMILY or not (m["window"] == "M").all():
        bad.append("гипотез семейства %d ≠ 12" % len(m))
    if not (m["lag"] == 8).all() or S.nw_lag(E.N_D) != 8:
        bad.append("лаг NW основной колонки ≠ 8")
    if not ((st["B"] == 10000).all() and S.B == 10000 and S.idx_for(E.N_D).shape == (10000, E.N_D)):
        bad.append("B ≠ 10 000")
    if not ((st["n_holm"] == 831).all() and (st["n_holm_wide"] == 3367).all() and (st["m_family"] == 12).all()):
        bad.append("Холм: N ≠ 831 / N_wide ≠ 3367 / m ≠ 12")
    for col, N in (("p_holm_m12", 12), ("p_holm_N", 831), ("p_holm_Nwide", 3367)):
        if not np.allclose(m[col].to_numpy(), S.holm_n(m["p_dm"].to_numpy(), N), rtol=0, atol=1e-15, equal_nan=True):
            bad.append("%s ≠ Холм(p DM NW-8, N = %d)" % (col, N))
    conf = m[m["verdict"].str.contains("подтверждено") & ~m["verdict"].str.contains("не подтверждено")]
    if not ((conf["p_holm_N"] < 0.05) & (conf["p_spa_c"] < 0.05) & (conf["delta_pct"] < 0)).all():
        bad.append("«подтверждено» без p Холма < 0,05 и p SPA < 0,05")
    ok = m[(m["delta_pct"] < 0) & (m["p_holm_N"] < 0.05) & (m["p_spa_c"] < 0.05)]
    return bad + ([] if len(ok) == len(conf) else ["не подтверждено при выполненных условиях"])


def problems(t: dict, rep: Path, ckpt: Path, tx: dict) -> list:
    bad = [] if t["identity"] <= E.EPS else ["Σ вкладов ≠ Δ (%.3g > 1e-12)" % t["identity"]]
    bad += _stats_problems(t["stats"])
    if Y.THR != {"У1-S": 0.415, "У2-S": 0.50, "У3-S": 1.00} or Y.ORDER != ("S-A2", "S-A1"):
        bad.append("пороги или порядок ≠ спецификации")
    g = pd.read_csv(rep / "S_A_GUARD.csv", comment="#")
    bad += [] if Y.decide(g) == t["decision"] == g[g["id"] == "решение"]["value"].iloc[0] else ["решение не выведено из строк GUARD"]
    bad += order_selftest()
    bad += [] if (t["mcs"]["n_used"] == E.N_D).all() and (t["table"]["n"] == E.N_D).all() else ["MCS или таблица не на 2198 датах"]
    bad += [] if (t["table"]["first_tplus"] == E.FIRST).all() and (t["table"]["last_tplus"] == E.LAST).all() else ["границы дат таблицы ≠ D_M"]
    sha = E.inputs_sha()
    bad += ["sha входа %s ≠ спецификации" % p for p, s in E.SPEC_SHA.items() if not sha[p].startswith(s)]
    from models_v2.s18 import a_walk as W
    bad += [] if (len(W.LAMS), len(W.SEEDS)) == (PAR.N_LAMS, PAR.N_SEEDS) else ["λ / seeds ≠ a_walk"]
    cp = json.loads(ckpt.read_text(encoding="utf-8")).get(E.KEY, {})
    want = {k: hashlib.sha256(v.encode("utf-8")).hexdigest() for k, v in tx.items()}
    return bad + ([] if cp.get("outputs") == want and cp.get("n_holm") == 831 and cp.get("n_holm_wide") == 3367 else ["CHECKPOINT %s ≠ пересчёту" % E.KEY])
