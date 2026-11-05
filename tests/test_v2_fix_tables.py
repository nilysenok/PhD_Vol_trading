"""Таблицы Ж-200: сетка, пересчёт критерия независимо, шапки; критерий на синтетике с известным ответом."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from models_v2 import metrics
from models_v2.s4 import prov4, report

OUT, TAB = prov4.OUT, prov4.TAB
FILES = [OUT / ("fix_forecasts_%s.parquet" % s) for s in "ac"]
have = pytest.mark.skipif(not (OUT / "fix_forecasts_a.parquet").exists(), reason="счёт Ж-200 ещё не выполнен")


def synth_fc(sig_good, sig_bad_early):
    d = pd.bdate_range("2017-01-03", "2025-12-31")
    rv = np.full(len(d), 2.0)
    mk = lambda name, s: pd.DataFrame({"d": d, "target_date": d + pd.Timedelta(days=1), "model": name, "sigma2": s, "rv": rv})  # noqa: E731
    early = np.asarray(d.year <= 2021)
    return pd.concat([mk("harj_b", np.full(len(d), 2.0)), mk("m_good_b", sig_good), mk("m_bad_b", np.where(early, sig_bad_early, 2.0))])


def test_criterion_known_answer_and_thresholds():
    d = pd.bdate_range("2017-01-03", "2025-12-31")
    fc = synth_fc(np.full(len(d), 2.0), 4.0)
    leak = pd.DataFrame({"model": ["m_good", "m_bad", "m_noK1"], "work_daily": [1.0, 1.0, 1.0], "full": [0.9, 0.5, 0.96]})
    fc = pd.concat([fc, fc[fc.model == "m_good_b"].assign(model="m_noK1_b")])
    over = pd.DataFrame({"model": ["m_good", "m_bad", "m_noK1", "m_over"], "year": ["all"] * 4, "mse_train": 1.0,
                         "mse_es": [1.2, 1.1, 1.3, 1.31], "ratio_mean": [1.2, 1.1, 1.3, 1.31]})
    leak = pd.concat([leak, pd.DataFrame({"model": ["m_over"], "work_daily": [1.0], "full": [0.5]})])
    fc = pd.concat([fc, fc[fc.model == "m_good_b"].assign(model="m_over_b")])
    c = report.criterion(fc, leak, over).set_index("model")
    assert c.loc["m_good", "R"] and c.loc["m_good", "K2"] and c.loc["m_good", "fixed"]     # QLIKE = эталон, ratio 1,2
    assert c.loc["m_bad", "R"] and not c.loc["m_bad", "K2"] and not c.loc["m_bad", "fixed"]   # хуже HAR-J в 2017–2021
    assert not c.loc["m_noK1", "K1"] and c.loc["m_noK1", "fixed"]                          # К1 справочная, на итог не влияет; 1,3 — граница включена
    assert c.loc["m_noK1", "ratio_full_work"] == pytest.approx(0.96)
    assert not c.loc["m_over", "R"] and c.loc["m_over", "K1"] and not c.loc["m_over", "fixed"]   # ratio 1,31 > 1,3 даже при K1


@have
def test_forecasts_one_grid_positive_no_gap_pairs():
    p1 = pd.read_parquet(OUT / "forecasts_p1.parquet")
    grid = set(pd.to_datetime(p1[p1.model == "harj_b"]["d"]))
    for f in FILES:
        if not f.exists():
            continue
        fc = pd.read_parquet(f)
        n = fc.groupby("model").size()
        assert n.nunique() == 1 and n.iloc[0] == 2272 and (fc.sigma2 > 0).all() and np.isfinite(fc.sigma2).all()
        for _, g in fc.groupby("model"):
            assert set(pd.to_datetime(g["d"])) == grid
            assert not ((pd.to_datetime(g["d"]) <= "2022-02-25") & (pd.to_datetime(g["target_date"]) >= "2022-03-24")).any()


@have
def test_criterion_table_matches_independent_recompute():
    cr = pd.read_csv(TAB / "fix_criterion.csv", comment="#")
    fc = pd.concat([pd.read_parquet(f) for f in FILES if f.exists()])
    p1 = pd.read_parquet(OUT / "forecasts_p1.parquet")
    h = p1[p1.model == "harj_b"]
    for r in cr.itertuples():
        f = fc[fc.model == r.model + "_b"]
        y = pd.to_datetime(f["target_date"]).dt.year.to_numpy()
        hy = pd.to_datetime(h["target_date"]).dt.year.to_numpy()
        for lo, hi, col, hc in ((2017, 2021, r.qlike_2017_2021, r.harj_2017_2021), (2023, 2025, r.qlike_2023_2025, r.harj_2023_2025)):
            m, hm = (y >= lo) & (y <= hi), (hy >= lo) & (hy <= hi)
            assert col == pytest.approx(np.mean(f["rv"][m] / f["sigma2"][m] - np.log(f["rv"][m] / f["sigma2"][m]) - 1), rel=1e-9)
            assert hc == pytest.approx(metrics.qlike(h["rv"][hm], h["sigma2"][hm]).mean(), rel=1e-9)
        assert r.K1 == (r.qlike_full <= 0.95 * r.qlike_work) and r.R == (r.ratio_es_train <= 1.3)
        assert r.fixed == (r.R and r.K2) and r.K2 == (r.qlike_2017_2021 <= r.harj_2017_2021 and r.qlike_2023_2025 <= r.harj_2023_2025)


@have
def test_headers_carry_current_s4_sha():
    for p in sorted(TAB.glob("fix_*.csv")):
        head = Path(p).open(encoding="utf-8").readline()
        assert "models_v2/s4 sha256[:16]=%s" % prov4.code_sha4() in head, p.name
        assert "до аудита" in head
