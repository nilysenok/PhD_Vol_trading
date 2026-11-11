"""S-C-EVAL, `cm_eval --check` на реальных сохранённых прогонах (гейт E.G1; mutations.md № 103, 125, 140, 158, 160): пересчёт = таблицам на диске; Холм N = 836 / 3377; лаг NW-8 основной колонки; B = 10 000; Σ вкладов = Δ;
решения выведены из строк GUARD; sha входов = спецификации; ключ `S.C.eval` = хэшам выходов; мутации результата (Холм N → m, лаг 5, пороги, порядок, решение) ловятся `problems`; `--check` только читает.
Реальные файлы только читаются (хэши до и после равны); выходы не пишутся."""
from __future__ import annotations

import copy
import json

import numpy as np
import pandas as pd
import pytest

from models_v2.s18 import a_eval_stat as S
from models_v2.s18 import cm_eval as E
from models_v2.s18 import cm_eval_chk as C
from models_v2.s18 import cm_eval_yr as Y


@pytest.fixture(scope="module")
def real():
    h0 = E.tree_hash()
    inp = E.load_inputs()
    t = E.compute(inp)
    return dict(h0=h0, inp=inp, t=t, tx=E.texts(t, inp["tp"]))


def probs(real, t=None):
    return C.problems(real["t"] if t is None else t, E.REP, E.CKPT, real["tx"])


def test_problems_empty_and_tables_equal_files(real):
    assert probs(real) == []
    for n, v in real["tx"].items():
        assert (E.REP / n).read_text(encoding="utf-8") == v
    t = real["t"]
    assert (t["table"]["n"] == 2198).all() and t["identity"] <= 1e-12 and S.nw_lag(2198) == 8 and S.B == 10000


def test_checkpoint_key_and_program_n(real):
    cp = json.loads(E.CKPT.read_text(encoding="utf-8"))
    k = cp[E.KEY]
    assert (k["n_holm"], k["n_holm_wide"], k["m_family"], k["m_struct"], k["boot"], k["lag_main"], k["n"]) == (836, 3377, 8, 3, 10000, 8, 2198)
    assert cp["S.C.registered"]["n"] == 836 and cp["S.C.registered"]["n_wide"] == 3377
    assert k["decision"] == real["t"]["decision"] and "written_at" in k


def test_mutation_holm_n_replaced_by_family_size_is_caught(real):
    t = copy.deepcopy(real["t"])
    st = t["stats"]
    m = st["kind"] == "main"
    assert (st.loc[m, "p_holm_N"] == 1.0).all()                               # реальные p ≥ 0,26: Холм N и m оба равны 1 — мутация на реальных p невидима, нужны малые p
    p = np.linspace(1e-5, 1e-3, int(m.sum()))
    st.loc[m, "p_dm"], st.loc[m, "p_holm_N"] = p, S.holm_n(p, 8)             # № 103: N = числу строк семейства
    assert any("p_holm_N" in b for b in probs(real, t))


def test_mutation_lag5_in_main_column_is_caught(real):
    t = copy.deepcopy(real["t"])
    t["stats"].loc[t["stats"]["kind"] == "main", "lag"] = 5
    assert any("лаг NW" in b for b in probs(real, t))


def test_mutation_thresholds_order_and_decision_are_caught(real, monkeypatch):
    monkeypatch.setattr(Y, "THR", {"У1-S": 0.5, "У2-S": 0.5, "У3-S": 1.0})    # пороги подогнаны под результат
    assert any("пороги" in b for b in probs(real))
    monkeypatch.undo()
    monkeypatch.setattr(Y, "ORDER", ("S-CKA", "S-CK"))
    assert any("порядок" in b or "пороги" in b for b in probs(real))
    monkeypatch.undo()
    t = copy.deepcopy(real["t"])
    t["decision"] = "новая контрольная версия: S-CK (решение разработки, не «подтверждено»); подмена"
    assert any("NEWCTRL" in b for b in probs(real, t))


def test_mutation_identity_and_confirmed_without_both_p_are_caught(real):
    t = copy.deepcopy(real["t"])
    t["identity"] = 1e-6
    assert any("Σ вкладов" in b for b in probs(real, t))
    t = copy.deepcopy(real["t"])
    st = t["stats"]
    i = st[(st["kind"] == "main") & (st["delta_pct"] > 0)].index[0]
    st.loc[i, "verdict"] = "численно лучше базы; подтверждено (p Холма N < 0,05 и p SPA < 0,05)"   # «подтверждено» по чужой строке
    assert any("подтверждено" in b for b in probs(real, t))


def test_mutation_mcs_and_dates_are_caught(real):
    t = copy.deepcopy(real["t"])
    t["mcs"]["n_used"] = 2197
    assert any("MCS" in b for b in probs(real, t))
    t = copy.deepcopy(real["t"])
    t["table"].loc[0, "first_tplus"] = "2018-01-05"
    assert any("границы дат" in b for b in probs(real, t))


def test_check_reads_only_and_matches_hash(real):
    assert E.check() == []
    assert E.tree_hash() == real["h0"]                                        # CM, A, S_C_*, CHECKPOINT до и после равны
    assert len(list(E.CM_DIR.glob("*.parquet"))) == 6 and all(len(pd.read_parquet(E.CM_DIR / f)) == 2198 for f in E.VFILES.values())
