"""V6 Z2 (G62, G63): сравнительные таблицы на синтетике с известным ответом. Ответ: Δ = (QLIKE(X) − QLIKE(Y)) / QLIKE(K1) · 100 — знак (минус = X лучше) и база (уровень K1, не Y); Холм с N = 722 / 3122 (не с числом строк);
ряд класса «потолок» не может быть лучшим. Мутации: Δ от уровня Y, N Холма = числу строк, потолок в выборе, прогноз сдвинут на +1 день."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from models_v2.s5.d import stats_d as SD
from models_v2.s13 import v6_0_q as Q, v6_cmp_best as B, v6_cmp_reg as R, v6_cmp_tab as T


def synth(n: int = 2250, seed: int = 3) -> dict:
    """9 лет по 250 дат (2017–2025); rv логнормальна; K1/K2/K3 — грубые прогнозы, H — точный (близок к rv), CEIL — идеальный (QLIKE 0), но класс «потолок»."""
    rng = np.random.default_rng(seed)
    rv = np.exp(rng.normal(0, 0.5, n))
    tp = pd.date_range("2017-01-03", periods=n, freq="B")
    f = {"K1": rv * np.exp(rng.normal(0.5, 0.5, n)), "K2": rv * np.exp(rng.normal(0.2, 0.3, n)), "K3": rv * np.exp(rng.normal(0.25, 0.3, n)), "M": rv * np.exp(rng.normal(0.2, 0.3, n)),
         "K8": rv * np.exp(rng.normal(0.3, 0.3, n)), "V6:A2": rv * np.exp(rng.normal(0.1, 0.27, n)), "J211feat:lgbm_S345": rv.copy()}
    return dict(rv=rv, tp=tp.to_numpy(), f=f, dates=tp - pd.Timedelta(days=1))


def test_delta_known_answer_sign_and_k1_level():
    u = synth()
    r = T.one_test(u, "V6:A2", "K2", "all")
    qx, qy, k1 = (Q.ql(u["rv"], u["f"][s]).mean() for s in ("V6:A2", "K2", "K1"))
    assert r["delta_pct"] == pytest.approx(100 * (qx - qy) / k1, abs=1e-12)
    assert r["delta_pct"] < 0 and r["p_one"] < 0.01                      # точный ряд лучше грубого: минус и малый p
    rev = T.one_test(u, "K2", "V6:A2", "all")
    assert rev["delta_pct"] == pytest.approx(-r["delta_pct"], abs=1e-12)    # переставленные X и Y меняют только знак
    assert rev["p_one"] > 0.99         # обратная пара: «K2 лучше» почти невозможно


def test_mutation_delta_from_y_level_is_caught():
    u = synth()
    r = T.one_test(u, "V6:A2", "K2", "all")
    qx, qy = (Q.ql(u["rv"], u["f"][s]).mean() for s in ("V6:A2", "K2"))
    mutant = 100 * (qx - qy) / qy                                            # подложенная ошибка: уровень Y вместо K1
    assert abs(mutant - r["delta_pct"]) > 1.0


def reg_of() -> pd.DataFrame:
    rows = [dict(series=s, row="V6", program="V6", role="а", cls=R.PIT) for s in ("V6:A2",)]
    return pd.DataFrame(rows)


def test_holm_uses_program_n_not_row_count():
    t = T.tests(synth(), reg_of())
    for w, g in t.groupby("window"):
        assert g["p_holm_N722"].to_numpy() == pytest.approx(SD.holm_n(g["p_one"].to_numpy(), 722), abs=1e-15)
        assert g["p_holm_N3122"].to_numpy() == pytest.approx(SD.holm_n(g["p_one"].to_numpy(), 3122), abs=1e-15)
        assert (g["p_holm_N3122"] >= g["p_holm_N722"] - 1e-15).all()
        wrong = SD.holm_n(g["p_one"].to_numpy(), len(g))                      # мутация № 103: N = число строк таблицы
        assert (g["p_holm_N722"].to_numpy() > wrong * 10).all()           # N = 722 против N = 4 строк окна: строго консервативнее


def test_ceiling_series_cannot_be_best():
    u = synth()
    with pytest.raises(AssertionError):
        B.best_set(u, ["K1", "K2", "J211feat:lgbm_S345"], "all")         # у потолка QLIKE = 0 — был бы первым без проверки класса
    ok = B.best_set(u, ["K1", "K2", "V6:A2"], "all")
    assert ok[0][0] == "V6:A2"


def test_mutation_ceiling_in_selection_is_caught_by_candidates():
    reg = pd.DataFrame([dict(series="K1", role="контроль", cls=R.CTL), dict(series="J211feat:lgbm_S345", role="а", cls=R.CEIL), dict(series="V6:A2", role="а", cls=R.PIT)])
    assert R.candidates(reg) == ["K1", "V6:A2"]


def test_forecast_shift_plus_one_is_detected_by_repro():
    """Прогноз, сдвинутый на +1 день против даты цели, даёт другой QLIKE: сверка с числом программы обязана дать «расхождение»."""
    from models_v2.s13 import v6_cmp_repro as P
    u = R.universe()
    reg = R.build_registry(u)
    assert len(P.unexplained(P.repro(u, reg))) == 0
    bad = np.append(u["f"]["V6:A2"][1:], u["f"]["V6:A2"][-1])               # прогноз даты d+1 против цели даты d
    u2 = dict(u, f={**u["f"], "V6:A2": bad})
    assert "V6:A2" in set(P.unexplained(P.repro(u2, reg))["series"])


def test_target_date_is_after_forecast_date():
    u = R.universe()
    assert (pd.DatetimeIndex(u["tp"]) > u["dates"]).all()                   # цель t⁺ строго позже даты прогноза d (Д2)
