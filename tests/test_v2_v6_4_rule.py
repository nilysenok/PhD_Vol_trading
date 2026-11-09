"""V6-4, G37/G38: правила D1, D2, D3 и K9 на малом пуле с известным ответом (k, h, κ как в плане при подстановке малых значений; значения плана — по умолчанию в сигнатуре), clip [e⁻¹, e], реестр (K9 — не строка),
проверка «не сводится к режиму» (вывод «эквивалент калибровки по уровню» только при выигрыше в одном терциле), изоляция (только SBER), записанные выходы."""
from __future__ import annotations

import ast
import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from models_v2.s13 import v6_0_cp as CP0, v6_1_reg as RG, v6_4_an as AN, v6_4_regime as RGM, v6_4_stage as ST

ST_PATH = Path(AN.__file__).parent


def small_pool():
    """Пул из 5 точек на первой оси (0…4), остальные оси постоянны; z-нормировка: среднее 2, СКО √2; R = 1…5; запрос 2,1 (расстояния 0,1; 0,9; 1,1; 1,9; 2,1 до деления на √2)."""
    Z = np.zeros((5, 6))
    Z[:, 0] = np.arange(5)
    zq = np.zeros(6)
    zq[0] = 2.1
    return Z, np.arange(1.0, 6.0), zq


def test_plan_parameters_are_defaults():
    p = inspect.signature(AN.day_stats).parameters
    assert (p["k1"].default, p["k2"].default, p["nh"].default, p["kappa"].default) == (50, 100, 50, 50) == (AN.K_D1, AN.K_D2, AN.K_H, AN.KAPPA)
    assert AN.ZN == ["s1", "s2", "s4", "s5", "s7", "s9"] and AN.VAR_D1 == (25, 50, 100) and AN.VAR_D2 == (50, 100, 200) and AN.VAR_KAPPA == (25, 50, 100)


def test_d1_d2_d3_known_answer_small_pool():
    Z, R, zq = small_pool()
    o = AN.day_stats(Z, R, zq, k1=2, k2=3, nh=1, kappa=50)
    assert o["m1_2"] == 3.5 and o["m_glob"] == 3.0                                           # ближайшие: точки 2 и 3 (R = 3, 4)
    assert np.isclose(o["h"], 1 / np.sqrt(2))                                                # сосед каждой точки на расстоянии 1/√2, медиана
    w = np.exp(-np.array([0.1, 0.9, 1.1]) ** 2 / 2)                                          # dist²/(2h²) = Δ²/2 при h = 1/√2
    m2, neff = float((w * np.array([3.0, 4.0, 2.0])).sum() / w.sum()), float(w.sum() ** 2 / (w ** 2).sum())
    assert np.isclose(o["m2_3"], m2, rtol=1e-12) and np.isclose(o["neff_3"], neff, rtol=1e-12)
    lo, hi = sorted((m2, 3.0))
    assert np.isclose(o["m3_50"], (neff * m2 + 50 * 3.0) / (neff + 50), rtol=1e-12) and lo < o["m3_50"] < hi


def test_h_uses_the_50th_neighbour_not_another():
    Z, R, zq = small_pool()
    assert not np.isclose(AN.day_stats(Z, R, zq, k1=2, k2=3, nh=3, kappa=50)["h"], AN.day_stats(Z, R, zq, k1=2, k2=3, nh=1, kappa=50)["h"])      # мутация «другой сосед» (3-й вместо 1-го) меняет h
    with pytest.raises(ValueError):
        AN.h_scale(np.zeros((5, 6)), 5)                                                       # пул не больше nh — отказ, а не молчаливая подстановка


def test_k9_and_d3_limits():
    Z, R, zq = small_pool()
    o = AN.day_stats(Z, R, zq, k1=2, k2=3, nh=1, kappa=10 ** 9)
    assert np.isclose(o["m3_1000000000"], o["m_glob"], rtol=1e-6)                             # κ → ∞: D3 → глобальная калибровка
    o1 = AN.day_stats(Z, R, zq, k1=2, k2=3, nh=1, kappa=1)
    assert np.isclose(o1["m3_1"], (o1["neff_3"] * o1["m2_3"] + o1["m_glob"]) / (o1["neff_3"] + 1), rtol=1e-12)     # κ = 1: формула плана


def test_clip_and_forecast_form():
    h = np.array([2.0, 2.0, 2.0])
    g, f = ST.forecast_of(np.array([100.0, 1e-3, 1.0]), h)
    assert np.allclose(f, [2.0 * np.e, 2.0 / np.e, 2.0]) and f[2] == 2.0 and g[2] == 0.0       # clip [e⁻¹, e]; M = 1 -> F = H побитно
    assert np.allclose(AN.clip_m(np.array([100.0, 1e-3, 1.5])), [np.e, 1 / np.e, 1.5])
    assert (f > 0).all()


def test_registry_has_d_rows_and_k9_is_not_a_trial(tmp_path, monkeypatch):
    for c in ("D1", "D2", "D3"):
        assert c in RG.CONFIGS and c in RG.DECLARED and RG.CONFIGS[c]["base_rule22"].startswith("K9")
    assert "K9" not in RG.CONFIGS and "K9" not in RG.DECLARED and "K9" not in RG.CTRL           # контроль без бюджета: не конфигурация, не в N
    f = tmp_path / "ALL_TRIALS.csv"
    monkeypatch.setattr(CP0, "ALL_TRIALS", f)
    prior = ["A0", "A1", "A2", "A3", "C1", "C2", "C3", "C4", "E1", "E2", "E3"]
    f.write_text(RG.trial_text(prior + ["D1", "D2", "D3"]), encoding="utf-8")
    assert RG.trials_ok(prior + ["D1", "D2", "D3"], exact=True) and RG.trials_ok(prior) and not RG.trials_bad(f.read_text(encoding="utf-8"))
    f.write_text(f.read_text(encoding="utf-8") + "Ж-227:K9,Ж-227,x,y,{},trial\n", encoding="utf-8")
    assert RG.trials_bad(f.read_text(encoding="utf-8"))                                         # строка K9 в реестре недопустима


def synth_regime(better_in):
    rng = np.random.default_rng(7)
    n = 1500
    tp = pd.bdate_range("2017-01-02", periods=n)
    mu = rng.normal(0, 0.3, n)
    rv = np.exp(mu + rng.normal(0, 0.5, n))
    terc = rng.integers(1, 4, n)
    best, k9 = np.exp(mu + 0.125), np.exp(mu + 0.125 + 0.4)
    f = np.where(np.isin(terc, better_in), best, k9)
    return rv, f, k9, tp, terc, np.exp(mu), np.exp(mu)


def test_regime_verdict_only_one_tercile_is_level_calibration():
    t = RGM.with_verdict(RGM.table(*synth_regime([2])))
    assert RGM.EQ in t["verdict"].iloc[-1] and "терциле 2" in t["verdict"].iloc[-1]
    t3 = RGM.with_verdict(RGM.table(*synth_regime([1, 2, 3])))
    assert RGM.EQ not in t3["verdict"].iloc[-1] and t3["verdict"].iloc[-1].startswith("нет")
    t0 = RGM.with_verdict(RGM.table(*synth_regime([])))                                          # D2 = K9: выигрыша нет, проверка неприменима
    assert "неприменима" in t0["verdict"].iloc[-1] and len(t0) == 10


def test_regime_correlation_known_answer():
    rv, f, k9, tp, terc, h, _ = synth_regime([1, 2, 3])
    pos = RGM.table(rv, f, k9, tp, terc, h, h ** 2)
    neg = RGM.table(rv, f, k9, tp, terc, h, 1 / h)
    assert np.allclose(pos[pos["section"] == "корреляция"]["corr_lnM_lnH"], 1.0) and np.allclose(neg[neg["section"] == "корреляция"]["corr_lnM_lnH"], -1.0)


def test_leaf4_code_reads_only_sber_and_no_other_tickers():
    for p in sorted(ST_PATH.glob("v6_4_*.py")):
        tree = ast.parse(p.read_text(encoding="utf-8"))
        lits = {n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)}
        assert not (lits & {"GAZP", "LKOH", "GMKN", "IMOEX", "VTBR", "ROSN"}), p.name


def test_real_stage4_outputs_when_written():
    cp = CP0.read().get("4")
    if cp is None:
        pytest.skip("стадия 4 ещё не записана")
    assert cp["used"] == 14 and cp["N"] == 722 and cp["N_wide"] == 3122 and cp["code_hash_v6_4"] == ST.code_hash() and "K9" not in cp["configs"]
    for c in ("D1", "D2", "D3"):
        a = pd.read_parquet(Path(CP0.CP.parent.parent.parent.parent) / "data/regen/harj_hybrid_v6" / (c + ".parquet"))
        assert len(a) == 2271 and (a["F"] > 0).all() and np.isfinite(a["F"]).all()
