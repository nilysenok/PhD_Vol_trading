"""V6-6, G46–G48: A0g = S + GBM тем же путём, что A2 (побитно на S + Σ); подложенные ошибки — признаки кроме S, «закрыто + 1», база правила 2.2 = A0 вместо A0g; регистрация A0g с пометкой отступления; прежние строки против A0 — копия."""
import numpy as np
import pandas as pd
import pytest

from models_v2.s13 import v6_0_cp as CP0, v6_0_io as IO, v6_0_q as Q, v6_1_reg as RG, v6_1_stage as S1, v6_1_wf as WF, v6_1b_sig as SG, v6_4_stage as S4, v6_6_stage as ST


@pytest.fixture(scope="module")
def base():
    tab, S = S4.panel()
    a0g = pd.read_parquet(IO.OUT / "A0g.parquet").set_index("d")
    return tab, S, a0g


def _sigma(S):
    sf = pd.read_parquet(IO.OUT / "SIGMA_features.parquet").set_index("d")
    return pd.concat([S, sf[SG.SIGMA_NAMES].reindex(S.index)], axis=1)


def test_a0g_on_disk_is_the_gbm_path_on_s_bitwise(base):
    tab, S, a0g = base
    g = ST.run_gbm(tab, S)["g"]
    assert np.array_equal(g.reindex(a0g.index).to_numpy(), a0g["g"].to_numpy())
    assert len(S.columns) == 9                                                              # A0g = только сводка S (без Σ, Q, R)


def test_same_gbm_path_on_s_plus_sigma_gives_a2_bitwise_and_mutation_extra_features_caught(base):
    tab, S, a0g = base
    a2 = pd.read_parquet(IO.OUT / "A2.parquet").set_index("d")
    g2 = ST.run_gbm(tab, _sigma(S))["g"].reindex(a2.index).to_numpy()
    assert np.array_equal(g2, a2["g"].to_numpy())                                           # тот же код и параметры, что в A2
    assert not np.array_equal(g2, a0g["g"].reindex(a2.index).to_numpy())                    # мутация «A0g с признаками кроме S» даёт другой ряд


def test_mutation_closed_plus_one_caught(base, monkeypatch):
    tab, S, a0g = base
    orig = WF.closed_rows
    monkeypatch.setattr(WF, "closed_rows", lambda tp, d0: orig(tp, np.datetime64(d0) + np.timedelta64(1, "D")), raising=True)
    g = ST.run_gbm(tab, S)["g"].reindex(a0g.index).to_numpy()
    assert not np.array_equal(g, a0g["g"].to_numpy())                                       # пары «закрыто + 1» меняют прогноз -> тест ловит
    assert WF.closed_rows is not orig


def test_closed_rows_restored_after_mutation_test():
    tp = pd.bdate_range("2016-01-04", periods=30).to_numpy()
    assert len(WF.closed_rows(tp, tp[10])) == 11                                            # изоляция подмен (класс № 43): оригинал на месте


def test_rule_rows_use_a0g_base_and_known_answer_sign(base):
    tab, S, a0g = base
    w, cp0 = S1.controls(), CP0.read()["0"]
    prev = S4.prior_forecasts(w, CP0.read()["5"]["configs"])
    fs = {**prev, ST.CID: a0g["F"].reindex(w.index).to_numpy(), **{k: w[k].to_numpy() for k in RG.CTRL}}
    rule = ST.info_tables(fs, w, cp0)
    r = rule[(rule["role"] == ST.ROLE_RULE) & (rule["window"] == "2017_2021")].set_index("pair")
    assert set(r.index) == {"%s vs A0g" % x for x in ST.RULE_X}
    m = Q.wmask(w["t_plus"].to_numpy(), "2017_2021")
    for x in ST.RULE_X:
        d_g = Q.ql(w["rv"].to_numpy()[m], prev[x][m]).mean() - Q.ql(w["rv"].to_numpy()[m], fs[ST.CID][m]).mean()
        d_0 = Q.ql(w["rv"].to_numpy()[m], prev[x][m]).mean() - Q.ql(w["rv"].to_numpy()[m], prev["A0"][m]).mean()
        assert abs(r.loc["%s vs A0g" % x, "delta_qlike"] - d_g) <= 1e-12                    # Δ = QLIKE(X) − QLIKE(A0g): знак «X лучше» = отрицательный
        assert abs(d_g - d_0) > 1e-4                                                        # мутация «база A0 вместо A0g» даёт другое число


def test_old_rows_are_verbatim_copy_of_a_c_e_info():
    old = ST.old_rows()
    assert set(old["pair"]) == {"A2 vs A0", "C2 vs A0", "C4 vs A0", "E2 vs A0"} and set(old["window"]) == set(IO.read_csv("A_INFO.csv")["window"])
    a = IO.read_csv("A_INFO.csv")
    assert old[old["pair"] == "A2 vs A0"].reset_index(drop=True).equals(a[a["pair"] == "A2 vs A0"].reset_index(drop=True))


def test_a0g_registered_with_deviation_mark_and_registry_rules():
    assert ST.CID in RG.CONFIGS and ST.CID in RG.DECLARED and RG.CONFIGS[ST.CID]["features"] == ["S"] and RG.CONFIGS[ST.CID]["learner"] == "GBM"
    assert "ОТСТУПЛЕНИЕ" in RG.CONFIGS[ST.CID]["source"] and "вопрос 1" in RG.CONFIGS[ST.CID]["source"]
    txt = RG.trial_text(["A0", "A0g"])
    assert not RG.trials_bad(txt) and txt.count("Ж-227:") == 2
    assert RG.trials_bad(txt + RG.trial_text(["A0g"])[len(RG.trial_text([])):]) != []       # дубль A0g отвергается
    keys = [ln.split(",", 1)[0] for ln in CP0.ALL_TRIALS.read_text(encoding="utf-8").splitlines()[1:]]
    assert len(keys) == 18 == len(set(keys)) and "Ж-227:A0g" in keys and "Ж-227:CB2" not in keys
    cp = CP0.read()
    assert cp["6"]["used"] == 18 == int(cp["0"]["budget_V6"]) and cp["6"]["configs"] == keys_to_ids(keys)


def keys_to_ids(keys):
    return [k[6:] for k in keys]
