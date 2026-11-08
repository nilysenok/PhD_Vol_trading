"""V4-FIN (G45): таблицы итога на синтетике с известным ответом. Δ QLIKE к базе — в % уровня K1 того же окна (знак: минус = лучше базы); три лучшие — по crit_panel, а не по окнам 2022+;
сверка с таблицами этапов ловит подменённое число; frames_equal ловит правку числа. Тест в сеть не ходит; реальные таблицы — только чтение."""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_v2.s11 import v4_fin_check as CK, v4_fin_data as FD, v4_fin_stats as FS, v4_io as IO

WIN_Q = {"2017": 0.20, "2018": 0.30, "2017_2021": 0.25, "2022_2025": 0.35, "2026": 0.25, "all": 0.30}


def synth(q_x: float = 1.10, q_k2: float = 0.98) -> tuple:
    """16 бумаг M16; K1 = WIN_Q, K2 = q_k2·K1, X = q_x·K1 (во всех окнах); известный ответ Δ."""
    rows, cs = [], {"k1": 0.26, "k2": 0.27, "k3": 0.27, "d2": 0.26, "X": 0.27}
    for i in range(16):
        for w, q in WIN_Q.items():
            for s, m in (("k1", 1.0), ("k2", q_k2), ("k3", 1.0), ("d2", 1.0), ("X", q_x)):
                rows.append({"ticker": "T%d" % i, "group": "M16", "window": w, "series": s, "n": 10, "qlike": q * m, "rmse": 1.0, "mae": 1.0, "r2_rv": 0.5, "r2_ln": 0.5})
    c = pd.DataFrame({"group": ["M16"] * 3, "t_plus": pd.to_datetime(["2017-05-02", "2019-05-02", "2023-05-02"])})
    cond = pd.DataFrame({"series": list(cs), "crit_no2017": 0.3, "crit_trim1": 0.2, "c1": True, "c2": True, "c3": True, "go": True})
    return c, list(cs), pd.Series(cs), pd.DataFrame(rows), cond


def test_all_table_delta_is_in_percent_of_k1_level_known_answer():
    c, cols, cs, pt, cond = synth()
    a = FS.all_table(c, cols, cs, pt, cond).set_index(["series", "window"])
    for w, q in WIN_Q.items():
        assert abs(a.loc[("X", w), "d_qlike_k1_pct_of_k1"] - 10.0) < 1e-9                      # (1,10q − q)/q·100
        assert abs(a.loc[("X", w), "d_qlike_k2_pct_of_k1"] - 100 * (1.10 - 0.98)) < 1e-9       # (1,10q − 0,98q)/q·100
        assert abs(a.loc[("k2", w), "d_qlike_k1_pct_of_k1"] + 2.0) < 1e-9                      # K2 лучше K1 на 2 %
    assert a.loc[("X", "2017_2021"), "qlike_simple_2017_2021"] == a.loc[("X", "all"), "qlike_simple_2017_2021"] == 0.25 * 1.10


def test_mutation_sign_or_base_level_is_caught():
    """Мутации: Δ с обратным знаком; Δ в % уровня самой конфигурации (а не K1) — известный ответ их не пропускает."""
    c, cols, cs, pt, cond = synth()
    a = FS.all_table(c, cols, cs, pt, cond).set_index(["series", "window"])
    x, k2 = a.loc[("X", "all"), "qlike"], a.loc[("k2", "all"), "qlike"]
    assert 100 * (k2 - x) / 0.30 != a.loc[("X", "all"), "d_qlike_k2_pct_of_k1"]                # знак перепутан
    assert abs(100 * (x - k2) / x - a.loc[("X", "all"), "d_qlike_k2_pct_of_k1"]) > 1e-3         # база — уровень X, а не K1


def test_best3_uses_only_crit_panel_lowest_first_ties_by_name():
    """Тройка — три наименьших crit_panel среди 46 конфигураций (контроли не входят, даже если лучше); ничьи — по имени. Окна 2022+ в функцию не передаются."""
    cs = pd.Series({c: 0.30 for c in FD.CFGS + FD.CTRL})
    cs["I2"], cs["K8"], cs["B1"], cs["C1"], cs["E1"] = 0.25, 0.26, 0.26, 0.2, 0.1
    cs["d2"] = 0.05                                                                             # контроль лучше всех — в тройку не входит
    assert FS.best3(cs) == ["E1", "C1", "I2"]
    cs["E1"] = 0.9
    assert FS.best3(cs) == ["C1", "I2", "B1"]                                                   # ничья B1 = K8 — по имени


def test_registry_is_46_configurations_with_saved_forecasts():
    assert len(FD.CFGS) == len(set(FD.CFGS)) == 46
    assert sorted(FD.CFGS) == sorted(k.split(":")[1] for k in pd.read_csv(IO.REP / "ALL_TRIALS.csv")["key"] if k.startswith("Ж-222:"))
    assert all((FD.OUT / FD.file_of(c)).exists() for c in FD.CFGS)


def test_frames_equal_catches_changed_number_and_label():
    a = pd.DataFrame({"s": ["x", "y"], "v": [0.1, 0.2], "g": [True, False]})
    assert CK.frames_equal(a, a.copy())
    b = a.copy()
    b.loc[1, "v"] = 0.2 + 1e-6
    assert not CK.frames_equal(a, b)
    b = a.copy()
    b.loc[0, "g"] = False
    assert not CK.frames_equal(a, b)


def test_cross_stage_catches_substituted_number_in_fin_all():
    """FIN_ALL с подменённым crit_panel одного ряда расходится с таблицей этапа (допуск 1e-9)."""
    fin = IO.read_csv("FIN_ALL.csv")
    assert CK.cross_stage(fin) == []
    m = fin.copy()
    m.loc[(m["series"] == "K8") & (m["window"] == "all"), "crit_panel"] += 1e-6
    assert CK.cross_stage(m) != []
    z = fin.copy()
    z.loc[(z["series"] == "C8") & (z["window"] == "2022_2025"), "qlike"] *= 1.0001
    assert any("C8" in x for x in CK.cross_stage(z))
    assert np.isfinite(fin["crit_panel"]).all()
