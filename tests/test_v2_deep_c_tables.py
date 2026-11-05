"""D0 (Ж-204): строки C в таблицах B1 (HAR_BOOSTING_RESULTS, REGIME_MODEL_RESULTS) и в C_ALL.csv — состав, trial_id, правило «перспективно», Холм.
Строки B1 в этих же таблицах проверяет test_v2_deep_b1_tables.py. Без выходов счёта C тесты пропускаются."""
import json

import numpy as np
import pandas as pd
import pytest

from models_v2.s5 import common
from models_v2.s5.c import ctx_c, tab_c

FILES = {"har_boosting": "HAR_BOOSTING_RESULTS.csv", "regime_aware": "REGIME_MODEL_RESULTS.csv"}
C_ALL = common.REP / "C_ALL.csv"
pytestmark = pytest.mark.skipif(not (C_ALL.exists() and (ctx_c.DATA / "rows.json").exists()), reason="счёт C не выполнен")


def c_all():
    return pd.read_csv(C_ALL, comment="#")


def c_rows(dirn):
    df = common.read_comment_csv(common.REP / FILES[dirn])
    return df[df.row_id.str.match(r"c[123]_")].reset_index(drop=True)


@pytest.mark.parametrize("dirn", list(FILES))
def test_stroki_c_v_tablice_ravny_c_all(dirn):
    a = c_all()
    want = a[a.direction == dirn]
    df = c_rows(dirn)
    assert len(want) > 0 and set(df.row_id) == set(want.row_id) and df.row_id.is_unique
    m = df.merge(want, on="row_id", suffixes=("", "_c"))
    for col in ("qlike_all", "q_int", "q_int_b2", "d_harj_no_240222"):
        assert np.allclose(m[col], m[col + "_c"], rtol=0, atol=1e-12)


def test_c_all_sovpadaet_s_rows_json_i_reestrom_c():
    a = c_all()
    rows = json.loads((ctx_c.DATA / "rows.json").read_text())
    assert set(a.row_id) == set(rows) and a.row_id.is_unique
    reg = pd.read_csv(tab_c.TRIALS_C, dtype=str)
    assert reg["trial_id"].is_unique and len(reg) == 40
    own = a[a.row_id != "c1_opt"]   # c1_opt — строка-процедура, не trial
    assert set(own.trial_id.astype(str)) <= set(reg["trial_id"])


def test_perspective_c_ravno_pravilu():
    a = c_all()
    rule = (a.q_int_b2 - a.q_int > 1e-12) & (a.d_harj_no_240222 <= 0)
    assert (a.perspective.astype(bool).to_numpy() == rule.to_numpy()).all()


def test_holm_c_monotonen_i_N90_ne_menshe_N40():
    a = c_all()
    for k in ("harj", "b2"):
        assert (a["p_holm_" + k] >= a["p_" + k] - 1e-12).all()
        assert (a["p_holm_%s" % k] >= a["p_holm_%s_ref40" % k] - 1e-12).all()


def test_mut_holm_s_N1_lovitsya():
    p = pd.DataFrame({"p_harj": [0.01, 0.02, 0.03], "p_b2": [0.5, 0.6, 0.7]})
    big, one = tab_c.holm_cols(p, m=90), tab_c.holm_cols(p, m=1)
    assert (big.p_holm_harj > one.p_holm_harj + 1e-9).all() and big.p_holm_harj.iloc[0] >= 0.01 * 90 - 1e-9
