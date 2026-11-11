"""S-R-EVAL: боевая проверка на сохранённых рядах (E.G1, E.G2) и мутации «--check пишет», «решение не из таблицы», «пороги из результата», «обрезка в оценке», «знак Δ». Только чтение боевых файлов (копии — в tmp)."""
import shutil

import numpy as np
import pytest

from models_v2.s18 import a_eval_stat as S
from models_v2.s18 import r_eval as E
from models_v2.s18 import r_eval_chk as C
from models_v2.s18 import r_eval_tab as T
from models_v2.s18 import r_eval_yr as Y


@pytest.fixture(scope="module")
def real():
    inp = E.load_inputs()
    t = E.compute(inp)
    return inp, t, E.texts(t, inp["tp"])


@pytest.fixture()
def tmp_rep(tmp_path):
    for n in E.OUTS:
        shutil.copy(E.REP / n, tmp_path / n)
    shutil.copy(E.CKPT, tmp_path / "CKPT.json")
    return tmp_path, tmp_path / "CKPT.json"


def test_real_check_passes_and_changes_nothing(tmp_rep):
    rep, ck = tmp_rep
    assert E.check(rep, ck) == []


def test_spa_check_on_synthetic_with_data_sigma():
    assert E.check_spa_cli() == []


def test_real_tables_equal_files_and_decision_comes_from_guard(real):
    inp, t, tx = real
    assert all((E.REP / n).read_text(encoding="utf-8") == v for n, v in tx.items())
    assert t["decision"] == Y.decide(t["guard"]) and t["identity"] <= 1e-12 and (t["table"]["n"] == 2198).all()


def test_mut_check_that_writes_is_caught(tmp_rep, monkeypatch):
    rep, ck = tmp_rep
    orig = C.problems

    def writing(t, r, c, tx):
        (rep / "S_R_TABLE.csv").write_text("подмена", encoding="utf-8")
        return orig(t, r, c, tx)
    monkeypatch.setattr(C, "problems", writing)
    assert any("изменил выходы" in b for b in E.check(rep, ck))


def test_mut_decision_not_derived_from_guard_rows_is_caught(real, tmp_rep):
    inp, t, tx = real
    rep, ck = tmp_rep
    g = (rep / "S_R_GUARD.csv").read_text(encoding="utf-8").splitlines()
    i = next(k for k, ln in enumerate(g) if ln.startswith("S-RCK,М2,У3-S,") and ln.endswith("True"))
    g[i] = g[i][:-4] + "False"
    (rep / "S_R_GUARD.csv").write_text("\n".join(g) + "\n", encoding="utf-8")
    assert any("не выведено" in b for b in C.problems(t, rep, ck, tx))


def test_mut_thresholds_from_result_are_caught(real, tmp_rep, monkeypatch):
    inp, t, tx = real
    rep, ck = tmp_rep
    monkeypatch.setitem(Y.THR, "У1-S", 9.9)
    assert any("пороги" in b for b in C.problems(t, rep, ck, tx))


def test_mut_clipped_loss_in_evaluation_is_caught(real, tmp_rep, monkeypatch):
    inp, t, tx = real
    rep, ck = tmp_rep
    orig = T.T.Q.ql
    monkeypatch.setattr(T.T.Q, "ql", lambda rv, F: np.minimum(orig(rv, F), 3.0))
    t2 = dict(t)
    t2["table"] = T.table(inp["rv"], inp["tp"], inp["F"])
    monkeypatch.setattr(T.T.Q, "ql", orig)
    assert any("QLIKE" in b for b in C.problems(t2, rep, ck, tx))


def test_mut_trimmed_dates_in_main_delta_is_caught(real, tmp_rep):
    inp, t, tx = real
    rep, ck = tmp_rep
    t2 = dict(t)
    tb = t["table"].copy()
    i = tb.index[tb["series"] == "S-RCK"][0]
    tb.loc[i, "qlike"] = float(np.mean(np.sort(S.Q.ql(inp["rv"], inp["F"]["S-RCK"]))[:-22]))
    t2["table"] = tb
    assert any("QLIKE S-RCK" in b for b in C.problems(t2, rep, ck, tx))


def test_mut_wrong_sign_of_delta_in_table_is_caught(real):
    inp, t, _ = real
    tb = t["table"].set_index("series")
    q = {k: float(np.mean(S.Q.ql(inp["rv"], inp["F"][k]))) for k in ("S-RCK", "W5-G")}
    assert (tb.at["S-RCK", "d_W5-G_pct"] < 0) == (q["S-RCK"] < q["W5-G"]) and abs(tb.at["S-RCK", "d_W5-G_pct"] - 100 * (q["S-RCK"] - q["W5-G"]) / q["W5-G"]) < 1e-9
