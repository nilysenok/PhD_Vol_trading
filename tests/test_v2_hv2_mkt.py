"""Ж-214, лист 4, V10: рыночная общность SBER–IMOEX. Синтетика с известным ответом (точная линейная связь), тождество идиосинкратической и рыночной частей, асимметрия семиковариаций,
главный тест обрезки на каждой конфигурации E1…E9, мутации темы: признак дня d+1, идиосинкратическая часть не от синхронной RV, информативность без «без 24.02.2022», позиционное окно. Без сети."""
import dataclasses

import numpy as np
import pandas as pd

from models_v2 import har as H
from models_v2.s9 import hv2_dec as DC, hv2_mkt as MK, hv2_mktstage as MS_, hv2_msfeat as MS, hv2_synth as SB, hv2_synth_ml as SY, hv2_world as W

CFGS = [(MK.run, True, (), 22), (MK.run, True, ("semicov",), 22), (MK.run, True, ("idio",), 22), (MK.run, True, ("relrv",), 22), (MK.run, True, ("dmkt", "mktsemi"), 22),
        (MK.run, False, ("semicov", "idio"), 22), (MK.run, True, (), 63)]


def test_idio_plus_systematic_equals_synchronous_rv_and_mutation_caught():
    dw = SY.dworld()
    c = dw.comp
    cols = dict(MK.block_cols(dw, "idio"))
    sys_, idio = c["cov"] ** 2 / c["rv_m_sync"], np.exp(cols["ln_idio_d"])
    assert np.allclose(idio + sys_, c["rv_s_sync"], rtol=1e-12, atol=0) and np.allclose(np.exp(cols["ln_sys_d"]), sys_, rtol=1e-12)
    assert (idio > 0).all()
    wrong = dw.p.rv - sys_                                                        # МУТАНТ: идиосинкратическая часть от канонической RV, не от синхронной
    assert not np.allclose(wrong + sys_, c["rv_s_sync"], rtol=1e-6)


def test_semicov_asymmetry_known_answer_and_windows():
    dw = SY.dworld(n=80, tail=20)
    c = {k: v.copy() for k, v in dw.comp.items()}
    c["cov_pp"][:], c["cov_nn"][:], c["cov_mix"][:] = 3.0, 1.0, -1.0
    d2 = dataclasses.replace(dw, comp=c)
    cols = dict(MK.block_cols(d2, "semicov"))
    assert np.allclose(cols["semi_asym_d"][5:], 0.4) and np.allclose(cols["semi_mix_d"][5:], 0.2)
    assert np.allclose(cols["semi_asym_22"][30:], 0.4) and np.isnan(cols["semi_asym_22"][:21]).all()


def test_corr_beta_window_known_answer_constant_and_not_positional():
    dw = SY.dworld(n=80, tail=20)
    c = {k: v.copy() for k, v in dw.comp.items()}
    c["rv_s_sync"][:], c["rv_m_sync"][:], c["cov"][:] = 4.0, 1.0, 1.0          # corr = 1/√4 = 0,5; beta = 1
    cols = dict(MK.mb_cols(dataclasses.replace(dw, comp=c)))
    assert np.allclose(cols["corr22"][30:], 0.5) and np.allclose(cols["beta22"][30:], 1.0) and np.isnan(cols["corr22"][:21]).all()


def test_mb_known_answer_noiseless_forecast_equals_target():
    rng = np.random.default_rng(3)
    dw = SY.dworld(n=520, tail=120, seed=11)
    n = len(dw.p.rv)
    imo = np.exp(rng.normal(1.0, 0.5, n))
    ln = np.zeros(n)
    for t in range(1, n):
        ln[t] = 0.3 + 0.6 * ln[t - 1] + 0.3 * np.log(imo[t - 1])
    rv = np.exp(ln)
    seg = dw.p.seg
    X = np.column_stack([ln, np.log(DC.wmean(rv, seg, 5)), np.log(DC.wmean(rv, seg, 22)), dw.p.X[:, 3]])
    s = pd.Series(imo)
    comp = {**dw.comp, "imo_rv": imo, "imo_w": s.rolling(5, min_periods=1).mean().to_numpy(), "imo_m": s.rolling(22, min_periods=1).mean().to_numpy()}
    d2 = dataclasses.replace(dw, p=H.Panel(dw.p.dates, seg, rv, X), comp=comp, el=H.eligible(H.Panel(dw.p.dates, seg, rv, X))[-120:])
    s2, B, names, fb = MK.run(d2, True)
    assert not fb.any() and np.allclose(s2, rv[d2.el + 1], rtol=1e-6)
    b = dict(zip(names, B[-1]))
    assert abs(b["ln_rv_d"] - 0.6) < 1e-6 and abs(b["imo_d"] - 0.3) < 1e-6


def test_main_truncation_every_config_and_lead_mutation_caught():
    dw = SY.dworld()
    i0 = int(dw.el[-20])
    for fn, use_mb, blocks, win in CFGS:
        f = lambda x, a=(use_mb, blocks, win): MK.run(x, *a)[0]
        assert DC.trunc_diff(f, dw, i0) == 0.0, (use_mb, blocks, win)
    bad = lambda x: MK.run(x, True, ("semicov", "idio"), 22, lead=1)[0]            # МУТАНТ: признаки дня d+1
    assert DC.trunc_diff(bad, dw, i0) > 0.0


def test_nonfinite_feature_day_falls_back_to_a0_and_sigma_positive():
    dw = SY.dworld()
    d = int(dw.el[-6])
    c = {k: v.copy() for k, v in dw.comp.items()}
    c["cov"][d] = np.nan
    d2 = dataclasses.replace(dw, comp=c)
    s2, _, _, fb = MK.run(d2, True)
    a0 = DC.a0_forecast(d2)
    k = int(np.flatnonzero(d2.el == d)[0])
    assert fb[k] and s2[k] == a0[k] and np.isfinite(s2).all() and (s2 > 0).all()


def test_informative_rule_requires_all_four_conditions_mutation_caught():
    ok = {"d_crit_Mb": -0.001, "d_qlike_Mb": -0.002, "d_qlike_no_240222_Mb": -0.001, "years_better_Mb": 7}
    assert MK.informative(ok)
    assert not MK.informative({**ok, "d_crit_Mb": -0.0004})
    assert not MK.informative({**ok, "d_qlike_no_240222_Mb": 0.0002})              # МУТАНТ: выигрыш держится на 24.02.2022
    assert not MK.informative({**ok, "years_better_Mb": 5}) and not MK.informative({**ok, "d_qlike_Mb": 0.0})


def test_build_world_places_imoex_on_panel_dates_without_forward_lookup():
    days = SB.bdays("2018-01-02", 70)
    bars = SB.make([(d, {}) for d in days])
    ix = SB.make([(d, {"amp": 0.0005}) for d in days], seed=1)
    d, _ = __import__("models_v2.daily", fromlist=["x"]).build_daily(bars, index_days=None)
    w = W.build(bars, pd.DataFrame(index=d.index, data={"rq": 1.0, "rv_rel": 0.1}), ix)
    m10 = pd.DataFrame({k: 1.0 for k in MS.KEYS}, index=d.index)
    dw = MS.build_world(w, m10, ix)
    assert len(dw.comp["imo_rv"]) == len(dw.p.rv) and np.isfinite(dw.comp["imo_rv"]).all() and (dw.comp["imo_rv"] > 0).all()
    assert np.isnan(dw.comp["imo_m"][:21]).all() and np.isfinite(dw.comp["imo_m"][21:]).all()


def test_stage_configs_cover_min_set_and_no_extra_trials():
    assert MS_.MIN_SET == ("E1", "E2", "E3", "E4") and set(MS_.CFG) == {"E1", "E2", "E3", "E4", "E5", "E6", "E9"}
    assert MS_.CFG["E9"][3] == 63 and MS_.CFG["E1"][2] == ()
    from models_v2.s9 import hv2_ledger as L
    assert L.LIMITS["market"] == 10
