"""G7 B2: строка таблицы — QLIKE пересчитывается из прогнозов, правило «перспективно» (мутант «с 24.02.2022» ловится), реестр trials."""
import numpy as np
import pandas as pd
import pytest

from models_v2 import metrics
from models_v2.s5 import registry
from models_v2.s5.b2 import engine, feat, tab

ST = {"mse_train": 0.1, "mse_block": 0.12, "trees": 50.0, "n_feat": 28, "n_fits": 2271, "secs": 1.0}


def frames(better_only_feb24=False, seed=0):
    td = pd.bdate_range("2017-01-02", periods=2272)[1:]
    rng = np.random.default_rng(seed)
    rv = np.exp(rng.normal(0, 0.5, 2271))
    mk = lambda s2: pd.DataFrame({"d": td - pd.Timedelta(days=1), "target_date": td, "sigma2": s2, "rv": rv})   # noqa: E731
    har = mk(rv * np.exp(rng.normal(0, 0.3, 2271)))
    b2 = mk(har["sigma2"].to_numpy() * 1.01)
    s2 = har["sigma2"].to_numpy() * 1.02
    if better_only_feb24:
        i = int(np.flatnonzero(td == tab.FEB24)[0])
        h2 = har["sigma2"].to_numpy().copy()
        s2 = h2 * np.exp(0.2)               # везде хуже HAR…
        s2[i] = rv[i]                       # …кроме 24.02.2022, где HAR катастрофически занижает
        har["sigma2"] = np.where(np.arange(2271) == i, rv / 500, h2)
    return mk(s2), har, b2


META = dict(trial_id="x", direction="d", crit_loyo=0.20, crit_loyo_b2=0.21, crit_wf_b2=0.21)


def test_row_qlike_recomputed_and_columns():
    fc, har, b2 = frames()
    r = tab.row("m", fc, har, b2, ST, META)
    assert r["qlike_all"] == pytest.approx(metrics.qlike(fc["rv"], fc["sigma2"]).mean(), abs=1e-12)
    m = (fc["target_date"].dt.year == 2018).to_numpy()
    assert tab.masks(fc["target_date"])["no_2022"].sum() == (fc["target_date"].dt.year != 2022).sum()
    assert r["mse_ratio"] == pytest.approx(1.2) and m.sum() > 0
    for k in ("dloss_harj_mean", "dloss_harj_dm_p", "dloss_b2_mean", "win_years_harj", "qlike_no_20220224", "qlike_no_2022", "d_b2_y2022"):
        assert k in r
    assert r["dloss_harj_mean"] == pytest.approx((metrics.qlike(fc["rv"], fc["sigma2"]) - metrics.qlike(har["rv"], har["sigma2"])).mean())


def test_grid_mismatch_is_error():
    fc, har, b2 = frames()
    with pytest.raises(ValueError):
        tab.row("m", fc.iloc[:-1], har.iloc[:-1], b2.iloc[:-1], ST, META)


def test_perspective_rule_and_feb24_mutant_caught():
    fc, har, b2 = frames(better_only_feb24=True)
    r = tab.row("m", fc, har, b2, ST, META)
    full = (metrics.qlike(fc["rv"], fc["sigma2"]) - metrics.qlike(har["rv"], har["sigma2"])).mean()
    assert full < 0 < r["dloss_harj_no_20220224"]        # «выигрыш» только в один день
    assert r["perspective"] is False and r["crit_kind"] == "loyo"
    mutant = full <= 0 and META["crit_loyo"] < META["crit_loyo_b2"]
    assert mutant is True                                # мутант (разность с 24.02.2022) объявил бы перспективной — ловится правилом


def test_perspective_requires_criterion_better_than_b2():
    fc, har, b2 = frames()
    r = dict(tab.row("m", fc, har, b2, ST, META))
    r["dloss_harj_no_20220224"] = -0.01
    assert tab.promising(r, META)[0] is True
    assert tab.promising(r, dict(META, crit_loyo_b2=0.19))[0] is False
    r["crit_loyo"] = float("nan")                       # weekly/monthly/CUSUM: crit_wf
    r["crit_wf"] = 0.20
    assert tab.promising(r, META) == (True, "wf") and tab.promising(r, dict(META, crit_wf_b2=0.19)) == (False, "wf")


def test_crit_wf_uses_complete_years_2017_2025():
    fc, _, _ = frames()
    yr = pd.DatetimeIndex(fc["target_date"]).year
    q = pd.Series(metrics.qlike(fc["rv"], fc["sigma2"]), index=yr).groupby(level=0).mean()
    assert tab.crit_wf(fc) == pytest.approx(q.loc[2017:2025].mean())


def test_trial_ids_unique_and_budget(tmp_path):
    acc = {g: c for g, c in feat.GROUPS.items()}
    cfgs = [engine.Cfg("b2_" + g.lower(), add=c) for g, c in acc.items()] + [engine.Cfg("b2_" + k, drop=k) for k in engine.DROP]
    cfgs += [engine.Cfg("b2_top2", add=("a",)), engine.Cfg("b2_top3", add=("a", "b"))]
    assert len(cfgs) == 12 <= 22 and len(engine.window_cfgs()) == 10 <= 17
    ids = [c.trial_id() for c in cfgs + engine.window_cfgs()]
    assert len(set(ids)) == len(ids)
    p = tmp_path / "t.csv"
    row = {c: "" for c in registry.TRIAL_COLS}
    assert tab.register([dict(row, trial_id="a"), dict(row, trial_id="b")], p) == 2
    assert tab.register([dict(row, trial_id="a"), dict(row, trial_id="c")], p) == 1     # только дописывание, без дублей
    assert list(pd.read_csv(p)["trial_id"]) == ["a", "b", "c"]
