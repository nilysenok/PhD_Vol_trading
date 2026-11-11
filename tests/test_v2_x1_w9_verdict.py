"""W9, `--check-verdict`: решение «none» выводится из CSV (блоки T, EP, A_sum, NULL) и совпадает с V и DECISION; мутации копии CSV ловятся
(кандидат «проходит» по подменённому d_X1, подменённый p_shift, подменённая строка DECISION); реальный файл только читается."""
from __future__ import annotations

import hashlib

from models_v2.s16 import x1_w9_diag_v as V


def test_real_verdict_none_and_consistent() -> None:
    sha = hashlib.sha256(V.OUT.read_bytes()).hexdigest()
    df = V.read()
    got, bad = V.problems(df)
    assert (got, bad) == ("none", [])
    res = V.derive(df)
    assert not any(r["passed"] for r in res.values()) and len(res) == 6
    assert res["W9A"]["P"] == dict(P1=False, P2=False, P3=False, P4=False, P5=False, P6=True)      # W9_EXHAUSTION: 00000 (П6 выполнен)
    assert res["W9D2_encshare"]["P"]["P1"] and res["W9D2_encshare"]["P"]["P2"] and not res["W9D2_encshare"]["P"]["P3"]   # 11000
    assert V.main() == 0
    assert hashlib.sha256(V.OUT.read_bytes()).hexdigest() == sha                                  # файл не тронут


def test_mutation_candidate_made_passing_is_caught() -> None:
    df = V.read()
    t = (df["block"] == "T") & (df["cand"] == "W9B1_S") & (df["metric"].isin(["d_ctrl_pct", "d_X1_pct"]))
    df.loc[t, "value"] = -1.0                                                  # все Δ отрицательны: П1–П3, П5 выполнены
    got, bad = V.problems(df)
    assert got != "none" or bad                                                # решение меняется или расхождение с V/DECISION названо
    assert any("W9B1_S" in b for b in bad)


def test_mutation_decision_row_and_p_values_caught() -> None:
    df = V.read()
    m = (df["block"] == "NULL") & (df["cand"] == "W9D2_encshare") & (df["metric"] == "p_shift")
    assert float(df.loc[m, "value"].iloc[0]) == 0.2                            # W9_EXHAUSTION разд. 3: p_shift W9D2 = 0,20
    d = (df["block"] == "DECISION") & (df["cand"] == "W9D2_encshare") & (df["metric"] == "passed_with_null")
    df.loc[d, "value"] = 1.0                                                   # передекларированное «прошёл» не совпадает с выводом из таблицы
    got, bad = V.problems(df)
    assert got == "none" and any("DECISION W9D2_encshare" in b for b in bad)


def test_mutation_p_shift_makes_passing_candidate_pass_only_with_all_p() -> None:
    df = V.read()
    for cand in ("W9D2_encshare",):
        t = (df["block"] == "T") & (df["cand"] == cand) & (df["window"].isin(["T3", "T4"])) & (df["metric"].isin(["d_ctrl_pct", "d_X1_pct"]))
        df.loc[t, "value"] = -1.0                                              # П1–П3, П5 выполнены; П4 (порядок групп) остаётся ложным
    assert not V.derive(df)["W9D2_encshare"]["passed"]
    ep = (df["block"] == "EP") & (df["cand"] == "W9D2_encshare") & (df["window"] == "h2b22") & (df["metric"] == "kappa")
    h1 = df[(df["block"] == "EP") & (df["cand"] == "W9D2_encshare") & (df["window"] == "h1") & (df["metric"] == "kappa")].set_index("group")["value"]
    df.loc[ep, "value"] = [float(h1[g]) for g in df.loc[ep, "group"]]          # порядок групп в h2b22 = порядку в h1: П4 выполнен
    r = V.derive(df)["W9D2_encshare"]
    assert r["all"] and not r["passed"]                                        # p_shift = 0,20 > 0,10 — нуль не пройден
    m = (df["block"] == "NULL") & (df["cand"] == "W9D2_encshare") & (df["metric"] == "p_shift")
    df.loc[m, "value"] = 0.05
    assert V.derive(df)["W9D2_encshare"]["passed"] and V.pick(V.derive(df)) == "W9D2_encshare"


def test_mutation_chosen_row_swapped_caught() -> None:
    df = V.read()
    c = (df["block"] == "DECISION") & (df["metric"] == "chosen")
    df.loc[c, "cand"] = "W9A"
    got, bad = V.problems(df)
    assert got == "none" and any("chosen" in b for b in bad)
