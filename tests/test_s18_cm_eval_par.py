"""S-C-EVAL, `cm_eval_par` (S_C_SPEC разд. 2 «Учёт сложности», 4.5): ‖θ‖₀ (|θ| > 1e-6) по месяцам, число подгонок ((3 λ + 1)·5 при n_train ≥ 252, 5 при меньшем), S-C7 без обучения (0 подгонок), распределение ln 𝓜
(доля дат с |ln 𝓜| > 1e-3, по годам, активные даты) и доли компонент по годам. Синтетика с известным ответом; реальные таблицы читаются только на согласованность с файлами CM."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from models_v2.s18 import cm_eval_par as P


def frame(m, n_train=300, th=(0.0, 5e-7, 2e-6, 0.3, -0.2, 0.0)):
    """Синтетика: 2 года по 4 даты, 2 переоценки, колонки как в CM/*.parquet (параметры после `spread`)."""
    d = pd.to_datetime(["2021-12-06", "2021-12-07", "2022-01-10", "2022-01-11"] * 2)[:4]
    d = pd.DatetimeIndex(list(d) + list(pd.to_datetime(["2022-02-01", "2022-02-02", "2022-03-01", "2022-03-02"])))
    n = len(d)
    ref = np.where(d < pd.Timestamp("2022-02-01"), pd.Timestamp("2021-12-01"), pd.Timestamp("2022-02-01"))
    ref = np.where(d >= pd.Timestamp("2022-03-01"), pd.Timestamp("2022-03-01"), ref)
    df = pd.DataFrame(dict(t_plus=d, M=m, C=np.ones(n), F_c=np.full(n, 0.4), F_p=np.full(n, 0.6), active=True, refit_date=pd.DatetimeIndex(ref), lam=1e-3, n_train=n_train, seed_best=0.0, spread=0.0))
    for k, v in zip(("a", "b", "c", "d", "e", "f"), th):
        df[k] = v
    return df


def test_nz0_counts_modulus_above_1e6_and_year_mean():
    fr = frame(np.ones(8))
    p = P.params({"S-CK": fr})
    r = p[(p["variant"] == "S-CK") & (p["level"] == "refit")]
    assert list(r["nz0"]) == [3, 3, 3] and (r["n"] == 300).all()              # 2e-6, 0,3, −0,2; 5e-7 и 0 — нули
    y = p[p["level"] == "year"].set_index("key")
    assert y.at["2021", "n"] == 1 and y.at["2022", "n"] == 2 and y.at["2022", "nz0"] == 3.0 and y.at["2022", "a"] == 0.0 and y.at["2022", "d"] == pytest.approx(0.3)


def test_n_fits_boundary_252_and_c7_untrained():
    assert P.n_fits([251]) == 5 and P.n_fits([252]) == 20 and P.n_fits([124, 300, 400]) == 5 + 20 + 20
    fr7 = frame(np.ones(8))[["t_plus", "M", "C", "F_c", "F_p", "active", "refit_date", "lam", "n_train", "seed_best", "spread"]]
    p = P.params({"S-C7": fr7})
    f = p[p["level"] == "fits"].iloc[0]
    assert f["n"] == 0 and "параметров 0" in f["key"]
    fk = P.params({"S-CK": frame(np.ones(8))})
    assert fk[fk["level"] == "fits"].iloc[0]["n"] == P.n_fits([300, 300, 300]) == 60


def test_inactive_dates_are_not_refits():
    fr = frame(np.ones(8))
    fr.loc[:3, "active"] = False
    p = P.params({"S-CK": fr})
    assert list(p[p["level"] == "refit"]["key"]) == ["2022-02-01", "2022-03-01"]


def test_lnm_share_year_and_level_change_known_answer():
    m = np.array([1.0, np.exp(9e-4), np.exp(2e-3), np.exp(-0.05), 1.0, 1.0, 1.1, 0.9])
    fr = frame(m)
    s = pd.DataFrame(dict(sc=np.full(8, 0.3), sp=np.full(8, 0.7)))
    L = P.lnm({"S-CK": fr}, s)
    d = L[(L["kind"] == "dist") & (L["key"] == "D_M")].iloc[0]
    assert d["n"] == 8 and d["share_abs_gt_1e3"] == pytest.approx(4 / 8)       # |ln M| > 1e-3: e^{2e-3}, e^{−0,05}, 1,1, 0,9; e^{9e-4} ниже порога
    assert d["mean"] == pytest.approx(float(np.log(m).mean())) and d["min"] == pytest.approx(np.log(0.9)) and d["max"] == pytest.approx(np.log(1.1))
    y = L[L["kind"] == "year"].set_index("key")
    assert y.at["2021", "n"] == 2 and y.at["2022", "n"] == 6 and y.at["2022", "share_abs_gt_1e3"] == pytest.approx(4 / 6) and y.at["2021", "share_abs_gt_1e3"] == 0.0
    sh = L[L["kind"] == "share"].set_index("key")
    assert sh.at["2022", "fc_over_c"] == pytest.approx(0.4) and sh.at["2022", "fp_over_c"] == pytest.approx(0.6) and sh.at["2022", "s_c"] == pytest.approx(0.3)


def test_lnm_identity_model_is_exactly_zero():
    L = P.lnm({"S-C7": frame(np.ones(8))}, pd.DataFrame(dict(sc=np.zeros(8), sp=np.ones(8))))
    d = L[(L["kind"] == "dist") & (L["key"] == "D_M")].iloc[0]
    assert d["share_abs_gt_1e3"] == 0.0 and d["mean"] == 0.0 and d["max"] == 0.0 and not (L["kind"] == "share").any()


def test_real_tables_level_changes_and_files_equal():
    from models_v2.s18 import cm_doc_fmt as F
    L = F.read_all()["lnm"]
    d = L[(L["kind"] == "dist") & (L["key"] == "D_M")].set_index("variant")
    assert (d.loc[["S-CD", "S-CK", "S-CKA", "S-C7", "S-CKL"], "share_abs_gt_1e3"] > 0.5).all()     # общий уровень меняется
    ck = pd.read_parquet("data/regen/harj_struct/CM/ck.parquet")
    assert d.at["S-CK", "mean"] == pytest.approx(float(np.log(ck["M"]).mean()), abs=1e-9)
