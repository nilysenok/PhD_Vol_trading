"""S-C-EVAL (Ж-254; `S_C_SPEC.md`, разд. 7, гейт E.G1): проверки `cm_eval --check`, только чтение (ничего не пишут).
Публичное API: `problems(t, rep, ckpt, tx) -> list` замечаний: Σ вкладов = Δ (1e-12); лаг NW основной колонки = 8, B = 10 000; Холм N = 836 / 3377, m = 8 (основное семейство) и m = 3 (структура) пересчитаны из p DM; пороги = 0,415 / 0,50 / 1,00;
решения NEWCTRL и STRUCT выведены из строк GUARD (в том числе прочитанных из CSV), порядок S-CK → S-CKA, S-CD не кандидат (самотесты на синтетических строках); «подтверждено» только при p Холма < 0,05 и p SPA < 0,05;
sha входов = спецификации; N в `S.C.registered` = 836 / 3377; ключ `S.C.eval` = хэшам выходов; константы λ / seeds `cm_eval_par` = `cm_model`; MCS на всех 2198 датах; границы дат и n в таблицах.
`order_selftest()`, `struct_selftest()` — решения на синтетике с известным ответом."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from models_v2.s18 import a_eval_stat as S
from models_v2.s18 import cm_eval as E
from models_v2.s18 import cm_eval_par as PAR, cm_eval_tab as T, cm_eval_yr as Y

IDS10 = (("М1", "М1"), ("У1-S", "М1"), ("У2-S", "М1"), ("У3-S", "М1"), ("М2", "М2"), ("У1-S", "М2"), ("У2-S", "М2"), ("У3-S", "М2"), ("У4-C", "—"), ("У5-S", "М2"))


SPEC_ORDER = ("S-CK", "S-CKA")                  # порядок спецификации разд. 4.6 — константа проверки, не Y.ORDER (иначе мутация порядка не видна)


def _syn_guard(ok: dict) -> pd.DataFrame:
    rows = [dict(candidate=c, pair=p, id=i, value=0.0, passed=bool(ok[c])) for c in SPEC_ORDER for i, p in IDS10]
    return pd.DataFrame(rows + [dict(candidate=c, pair="—", id="кандидат", value="", passed=bool(ok[c])) for c in SPEC_ORDER])


def order_selftest() -> list:
    bad = []
    for ok, want in (({"S-CK": 1, "S-CKA": 1}, "S-CK"), ({"S-CK": 0, "S-CKA": 1}, "S-CKA"), ({"S-CK": 1, "S-CKA": 0}, "S-CK")):
        d = Y.decide(_syn_guard(ok))
        bad += [] if d.startswith("новая контрольная версия: %s (" % want) else ["порядок S-CK → S-CKA: %s → %s" % (ok, d)]
    return bad + ([] if Y.decide(_syn_guard({"S-CK": 0, "S-CKA": 0})).startswith("не прошёл") else ["оба не прошли ⇒ «не прошёл»"])


def struct_selftest() -> list:
    def g(k1, k2, k3, conf):
        return pd.DataFrame([dict(candidate=Y.SD, id=i, passed=bool(p)) for i, p in (("К1", k1), ("К2", k2), ("К3", k3), ("STRUCT подтверждён", conf))])
    bad = []
    for args, want in (((1, 1, 1, 1), "эффект структуры: да (К1 ∧ К2 ∧ К3); подтверждён"), ((1, 1, 1, 0), "эффект структуры: да (К1 ∧ К2 ∧ К3); не подтверждён"), ((0, 1, 1, 0), "эффект структуры: нет (не выполнено: К1)"),
                       ((1, 0, 1, 1), "эффект структуры: нет (не выполнено: К2)"), ((1, 1, 0, 1), "эффект структуры: нет (не выполнено: К3)")):
        d = Y.decide_struct(g(*args))
        bad += [] if d.startswith(want) else ["STRUCT %s → %s" % (args, d)]
    return bad


def _stats_problems(st: pd.DataFrame) -> list:
    m, s, bad = st[st["kind"] == "main"], st[st["kind"] == "struct"], []
    if len(m) != T.M_FAMILY or not (m["window"] == "M").all() or len(s) != T.M_STRUCT:
        bad.append("гипотез семейства %d ≠ 8 или структуры %d ≠ 3" % (len(m), len(s)))
    if not (m["lag"] == 8).all() or S.nw_lag(E.N_D) != 8:
        bad.append("лаг NW основной колонки ≠ 8")
    if not ((st["B"] == 10000).all() and S.B == 10000 and S.idx_for(E.N_D).shape == (10000, E.N_D)):
        bad.append("B ≠ 10 000")
    if not ((st["n_holm"] == 836).all() and (st["n_holm_wide"] == 3377).all() and (m["m_family"] == 8).all() and (s["m_family"] == 3).all()):
        bad.append("Холм: N ≠ 836 / N_wide ≠ 3377 / m ≠ 8, 3")
    for part, mm in ((m, 8), (s, 3)):
        for col, N in (("p_holm_m", mm), ("p_holm_N", 836), ("p_holm_Nwide", 3377)):
            if not np.allclose(part[col].to_numpy(), S.holm_n(part["p_dm"].to_numpy(), N), rtol=0, atol=1e-15, equal_nan=True):
                bad.append("%s ≠ Холм(p DM NW-8, N = %d) для %s" % (col, N, part["kind"].iloc[0]))
    fam = pd.concat([m, s])
    conf = fam[fam["verdict"].str.contains("подтверждено") & ~fam["verdict"].str.contains("не подтверждено")]
    if not ((conf["p_holm_N"] < 0.05) & (conf["p_spa_c"] < 0.05) & (conf["delta_pct"] < 0)).all():
        bad.append("«подтверждено» без p Холма < 0,05 и p SPA < 0,05")
    ok = fam[(fam["delta_pct"] < 0) & (fam["p_holm_N"] < 0.05) & (fam["p_spa_c"] < 0.05)]
    return bad + ([] if len(ok) == len(conf) else ["не подтверждено при выполненных условиях"])


def _sha_problems() -> list:
    sha = E.inputs_sha()
    bad = ["sha входа %s ≠ спецификации" % p for p, s in {**E.AE.SPEC_SHA, **E.C_SHA}.items() if not sha[p].startswith(s)]
    from models_v2.s18 import cm_model as M
    return bad + ([] if (len(M.LAMS), len(M.SEEDS)) == (PAR.N_LAMS, PAR.N_SEEDS) else ["λ / seeds ≠ cm_model"])


def problems(t: dict, rep: Path, ckpt: Path, tx: dict) -> list:
    bad = [] if t["identity"] <= E.EPS else ["Σ вкладов ≠ Δ (%.3g > 1e-12)" % t["identity"]]
    bad += _stats_problems(t["stats"])
    if Y.THR != {"У1-S": 0.415, "У2-S": 0.50, "У3-S": 1.00} or Y.ORDER != ("S-CK", "S-CKA") or "S-CD" in Y.ORDER:
        bad.append("пороги или порядок ≠ спецификации")
    g = pd.read_csv(rep / "S_C_GUARD.csv", comment="#")
    bad += [] if Y.decide(g) == t["decision"] == g[g["id"] == "решение"]["value"].iloc[0] else ["решение NEWCTRL не выведено из строк GUARD"]
    bad += [] if Y.decide_struct(g) == t["decision_struct"] == g[g["id"] == "решение STRUCT"]["value"].iloc[0] else ["решение STRUCT не выведено из строк GUARD"]
    bad += order_selftest() + struct_selftest()
    bad += [] if (t["mcs"]["n_used"] == E.N_D).all() and (t["table"]["n"] == E.N_D).all() else ["MCS или таблица не на 2198 датах"]
    bad += [] if (t["table"]["first_tplus"] == E.FIRST).all() and (t["table"]["last_tplus"] == E.LAST).all() else ["границы дат таблицы ≠ D_M"]
    bad += _sha_problems()
    cp = json.loads(ckpt.read_text(encoding="utf-8"))
    reg, key = cp.get("S.C.registered", {}), cp.get(E.KEY, {})
    bad += [] if (reg.get("n"), reg.get("n_wide")) == (836, 3377) else ["S.C.registered: N ≠ 836 / 3377"]
    want = {k: hashlib.sha256(v.encode("utf-8")).hexdigest() for k, v in tx.items()}
    return bad + ([] if key.get("outputs") == want and key.get("n_holm") == 836 and key.get("n_holm_wide") == 3377 else ["CHECKPOINT %s ≠ пересчёту" % E.KEY])
