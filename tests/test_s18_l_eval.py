"""S-L-EVAL (`S_L_SPEC.md` разд. 5, 6; Ж-257): оценка S-RL1 — честный пересчёт проходит независимые проверки, каждая мутация из спецификации роняет свою (№ 103, 125, 147, 158, 160)."""
import json
import shutil

import numpy as np
import pandas as pd
import pytest

from models_v2.s18 import l_doc as LD
from models_v2.s18 import l_eval as E
from models_v2.s18 import l_eval_chk as C
from models_v2.s18 import l_eval_tab as LT


@pytest.fixture(scope="module")
def ctx():
    inp = E.load_inputs()
    t = E.compute(inp)
    return inp, t, E.texts(t, inp["tp"])


def probs(ctx, t=None, tx=None, rep=E.REP, ckpt=E.CKPT):
    inp, t0, tx0 = ctx
    return C.problems(t or t0, inp, rep, ckpt, tx or tx0)


def cp(t):
    return {k: v.copy() for k, v in t.items()}


def test_honest_passes_and_check_changes_no_file(ctx):
    h0 = E.tree_hash()
    assert probs(ctx) == [] and E.check() == [] and LD.check() == [] and E.tree_hash() == h0


def test_n_of_program_is_852_and_holm_uses_it(ctx):
    inp, t, _ = ctx
    assert (inp["N"], inp["N_wide"]) == (852, 3409) and (t["delta"]["N"] == 852).all()


def test_mutation_sign_or_base_of_delta_is_caught(ctx):
    _, t, _ = ctx
    m = cp(t)
    m["delta"]["delta_pct"] = -m["delta"]["delta_pct"]
    assert any("Δ" in b for b in probs(ctx, m))
    m = cp(t)
    m["delta"]["delta_pct"] = 100.0 * (m["delta"]["q_series"] - m["delta"]["q_base"]) / m["delta"]["q_series"]
    assert any("Δ" in b for b in probs(ctx, m))


def test_mutation_qlike_as_mean_of_yearly_means_is_caught(ctx):
    inp, t, _ = ctx
    m = cp(t)
    y = pd.Series(inp["tp"].year)
    ls = LT.ql(inp["rv"], inp["F"]["S-RL1"])
    m["table"].loc[m["table"]["series"] == "S-RL1", "qlike"] = float(pd.Series(ls).groupby(y).mean().mean())
    assert any("QLIKE" in b for b in probs(ctx, m))


def test_mutation_holm_n_equals_row_count_is_caught(ctx):
    inp, t, _ = ctx
    m = cp(t)
    f = m["delta"]["m_family"].notna()
    m["delta"].loc[f, "p_holm_N"] = LT.S.holm_n(m["delta"].loc[f, "p_dm_nw8"].to_numpy(), 4)
    assert any("Холм" in b for b in probs(ctx, m))


def test_mutation_confirmed_at_holm_p_ge_005_is_caught(ctx):
    _, t, _ = ctx
    m = cp(t)
    i = m["delta"].index[m["delta"]["m_family"].notna()][1]
    m["delta"].loc[i, "verdict"] = "численно лучше базы; подтверждено (p Холма N < 0,05 и p SPA < 0,05)"
    assert any("158" in b for b in probs(ctx, m))


def test_mutation_trimmed_dates_in_main_delta_is_caught(ctx):
    _, t, _ = ctx
    m = cp(t)
    m["slices"].loc[m["slices"]["kind"] == "основной", "n_removed"] = 44
    assert any("обрезка" in b for b in probs(ctx, m))


def test_mutation_years_contrib_not_summing_to_delta_is_caught(ctx):
    _, t, _ = ctx
    m = cp(t)
    m["years"].loc[0, "contrib_pct"] += 0.01
    assert any("Σ вкладов" in b for b in probs(ctx, m))


def mk(q1, qa, qb2, qb, el):
    tb = pd.DataFrame(dict(series=["S-RL1", "W5-G", "S-RL1b2", "S-RL1b"], qlike=[q1, qa, qb2, qb]))
    w = pd.DataFrame([dict(series="S-RL1", base="W5-G", delta_pct=-1.0, p_holm_N=1.0, p_spa_c=1.0)])
    e = pd.DataFrame([dict(series="S-RCKf", base="S-RCK", delta_pct=el, code="worse_noise")])
    return LT.cond_rows(tb, w, e).set_index("id")


def test_condition_2_compares_with_b2_not_w5g_and_condition_3_reads_the_file():
    r = mk(1.0, 1.1, 0.9, 1.2, 0.15)    # S-RL1 лучше W5-G, но хуже S-RL1b2: основной критерий не выполнен, справочный (по S-RL1b) выполнен
    assert r.loc["interaction", "value"] == "не подтверждено (условие 2)" and r.loc["interaction_ref", "value"] == "да"
    assert mk(1.0, 1.1, 1.05, 1.2, 0.15).loc["interaction", "value"] == "да"
    assert mk(1.0, 1.1, 1.05, 1.2, -0.05).loc["interaction", "value"] == "не подтверждено (условие 3)"    # условие 3 не вписано вручную


def test_key_after_tables_is_caught(ctx, tmp_path):
    inp, t, tx = ctx
    ck = tmp_path / "CHECKPOINT.json"
    shutil.copy(E.CKPT, ck)
    k = json.loads(ck.read_text(encoding="utf-8"))
    k["S.L.eval"]["written_at"] = "31.12.2026 23:59:59"
    ck.write_text(json.dumps(k, indent=1, sort_keys=True, ensure_ascii=False), encoding="utf-8")
    assert any("позже самой ранней" in b for b in C.problems(t, inp, E.REP, ck, tx))


def test_report_number_and_better_worse_mutations_are_caught(tmp_path):
    for n in E.OUTS + (E.REPORT,):
        shutil.copy(E.REP / n, tmp_path / n)
    assert LD.check(tmp_path, E.CKPT) == []
    r = tmp_path / E.REPORT
    txt = r.read_text(encoding="utf-8")
    r.write_text(txt.replace("0.215166", "0.215167", 1), encoding="utf-8")
    assert LD.check(tmp_path, E.CKPT)
    r.write_text(txt.replace("без 5 параметров хуже", "без 5 параметров лучше", 1), encoding="utf-8")
    assert LD.check(tmp_path, E.CKPT)
    r.write_text(txt.replace("# исследовательский результат, не OOS", "# результат", 1), encoding="utf-8")
    assert LD.check(tmp_path, E.CKPT)
