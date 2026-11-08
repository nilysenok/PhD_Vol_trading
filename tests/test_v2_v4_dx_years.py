"""V4-DX (G26): диагностика по годам — известные ответы, сверка O1–O3 с oracle (G7), вес HAR-J и бутстрап, мутации (знак D, clip, вес/ML-охват)."""
import numpy as np
import pandas as pd
import pytest

from models_v2.s11 import v4_dx_years as DY, v4_oracle as OR


def mk(rv, h, m, n_t=1, dates=None):
    rv, h, m = (np.asarray(x, float) for x in (rv, h, m))
    n = len(rv)
    d = pd.bdate_range("2018-01-02", periods=n) if dates is None else dates
    return pd.DataFrame({"ticker": "AFLT", "date": d, "t_plus": d + pd.Timedelta(days=1), "rv": rv, "h": h, "k2": m, "k3": m})


def test_known_answer_shares_and_o_values_equal_oracle():
    """H = 1, M = 3, RV = (2, 5, 0.5, 3): внутри диапазона 2 из 4, оба ниже 1 из 4, оба выше 1 из 4; O1–O3 = oracle."""
    df = mk([2, 5, 0.5, 3], np.ones(4), 3 * np.ones(4))
    s = DY.slice_stats(df, "K2", boot=False)
    assert s["share_in_range"] == 0.5 and s["share_both_under"] == 0.25 and s["share_both_over"] == 0.25 and s["share_outside"] == 0.5
    o = OR.oracle(df["rv"], df["h"], df["k2"])
    assert s["o1"] == pytest.approx(o["o1"], abs=1e-12) and s["o2"] == pytest.approx(o["o2"], abs=1e-12) and s["o3"] == pytest.approx(o["o3"], abs=1e-12)


def test_o_values_equal_oracle_on_random_data():
    rng = np.random.default_rng(1)
    rv = np.exp(rng.normal(0, 0.5, 300))
    h, m = rv * np.exp(rng.normal(0, 0.3, 300)), rv * np.exp(rng.normal(0, 0.3, 300))
    a, o = DY.o_values(rv, h, m), OR.oracle(rv, h, m)
    assert all(a[k] == pytest.approx(o[k], abs=1e-10) for k in ("o1", "o2", "o3"))


def test_d_sign_har_better_gives_negative_d():
    """RV = H точно: HAR-J лучше M (D < 0); перепутанный знак (D = QLIKE(M) − QLIKE(H)) дал бы плюс."""
    rng = np.random.default_rng(2)
    h = np.exp(rng.normal(0, 0.3, 200))
    df = mk(h, h, h * np.exp(rng.normal(0, 0.4, 200)))
    s = DY.slice_stats(df, "K2", boot=False)
    assert s["d_mean"] < 0 and s["share_d_neg"] > 0.9 and s["n_tickers_har_better"] == 1 and s["best"] == "H"


def synth_weight(w_m, n=800, noise=0.05, seed=3):
    rng = np.random.default_rng(seed)
    h = np.exp(rng.normal(0, 0.3, n))
    m = h * np.exp(rng.normal(0, 0.5, n))
    rv = np.exp((1 - w_m) * np.log(h) + w_m * np.log(m) + rng.normal(0, noise, n))
    return mk(rv, h, m)


@pytest.mark.parametrize("w_m", [0.3, 0.7])
def test_har_weight_known_answer_and_interval_covers_truth(w_m):
    df = synth_weight(w_m)
    r = DY.har_weight(df["rv"].to_numpy(), df["h"].to_numpy(), df["k2"].to_numpy(), df["date"].to_numpy())
    assert r["w_har"] == pytest.approx(1 - w_m, abs=0.03) and r["w_har_lo90"] <= 1 - w_m <= r["w_har_hi90"]
    assert not r["ml_encompasses_har"] and not r["har_encompasses_ml"]


def test_ml_encompasses_when_rv_follows_m_and_har_encompasses_when_rv_follows_h():
    a = synth_weight(1.0)
    ra = DY.har_weight(a["rv"].to_numpy(), a["h"].to_numpy(), a["k2"].to_numpy(), a["date"].to_numpy())
    b = synth_weight(0.0)
    rb = DY.har_weight(b["rv"].to_numpy(), b["h"].to_numpy(), b["k2"].to_numpy(), b["date"].to_numpy())
    assert ra["ml_encompasses_har"] and not ra["har_encompasses_ml"] and rb["har_encompasses_ml"] and not rb["ml_encompasses_har"]


def test_mutation_weight_of_m_instead_of_har_is_caught():
    """Мутация «вес M вместо веса HAR-J»: при истинном w_M = 0,3 результат ≈ 0,3, а не 0,7 — тест известного ответа это различает."""
    df = synth_weight(0.3)
    r = DY.har_weight(df["rv"].to_numpy(), df["h"].to_numpy(), df["k2"].to_numpy(), df["date"].to_numpy())
    assert abs(r["w_har"] - 0.3) > 0.3


def test_mutation_oracle_without_clip_is_caught(monkeypatch):
    """O2 без clip (identity) даёт QLIKE(RV, RV) = 0 -> O2 = 0 ≠ значению oracle; известный ответ на синтетике H = 1, M = 3."""
    df = mk([2, 5, 0.5, 3], np.ones(4), 3 * np.ones(4))
    good = DY.o_values(df["rv"].to_numpy(), df["h"].to_numpy(), df["k2"].to_numpy())["o2"]
    monkeypatch.setattr(DY.np, "clip", lambda x, lo, hi: x)
    bad = DY.o_values(df["rv"].to_numpy(), df["h"].to_numpy(), df["k2"].to_numpy())["o2"]
    assert good > 1e-3 and bad < 1e-12


def test_empty_or_short_slice_is_skipped():
    P = pd.concat([mk([1.0] * 10, [1.0] * 10, [1.0] * 10)])
    assert DY.years_table(P).empty
