"""LGBM волна D: правила — годовой выбор по pen своего года, crit_wf по прошлым годам, порог частоты, бюджет и реестр, фолдовый crit (MD5–MD7)."""
import json

import numpy as np
import pandas as pd
import pytest

from models_v2 import metrics
from models_v2.s6.lgbm import crit, d_comp, d_world, grid


def test_pick_uses_only_pen_of_own_year():
    pens = pd.DataFrame({"A": [0.30, 0.10, 0.30], "B": [0.20, 0.20, 0.20]}, index=[2017, 2018, 2019])
    pick = d_comp.pick_by_year(pens)
    assert pick == {2017: "B", 2018: "A", 2019: "B"}
    shifted = d_comp.pick_by_year(pens.shift(1).bfill())                                    # мутация: pen другого года
    assert shifted != pick
    fcs = {n: pd.DataFrame({"target_date": pd.to_datetime(["2017-03-01", "2018-03-01", "2019-03-01"]), "sigma2": [v] * 3, "rv": [1.0] * 3}) for n, v in (("A", 1.0), ("B", 2.0))}
    assert d_comp.compose_by_year(fcs, pick)["sigma2"].tolist() == [2.0, 1.0, 2.0]          # прогноз года Y — из ряда кандидата года Y
    assert pd.Series(pens.min(axis=1)).mean() == pytest.approx(np.mean([pens.loc[Y, n] for Y, n in pick.items()]))


def _fc(levels: dict) -> pd.DataFrame:
    """Ряд с известным годовым QLIKE: sigma2 = rv / r, QLIKE = r − ln r − 1."""
    d, s, v = [], [], []
    for Y, r in levels.items():
        d += [pd.Timestamp("%d-03-01" % Y), pd.Timestamp("%d-09-01" % Y)]
        s += [1.0 / r] * 2
        v += [1.0] * 2
    return pd.DataFrame({"target_date": d, "sigma2": s, "rv": v})


def test_crit_wf_uses_only_past_years_and_mutation_caught():
    fc = _fc({2017: 2.0, 2018: 1.5, 2019: 1.2, 2020: 3.0})
    q = {Y: float(metrics.qlike(np.array([1.0]), np.array([1.0 / r]))[0]) for Y, r in {2017: 2.0, 2018: 1.5, 2019: 1.2, 2020: 3.0}.items()}
    want = np.mean([crit.pen_year(np.array([q[y] for y in range(2017, Y)])) for Y in (2018, 2019, 2020)])
    assert d_comp.crit_wf(fc) == pytest.approx(want)                                          # известный ответ
    fc2 = _fc({2017: 2.0, 2018: 1.5, 2019: 1.2, 2020: 9.0})
    assert d_comp.crit_wf(fc2) == pytest.approx(want)                                         # последний год не входит в pen своего Y
    mut = np.mean([crit.pen_year(np.array([q[y] for y in range(2017, Y + 1)])) for Y in (2018, 2019, 2020)])    # мутация: включает год Y
    assert mut != pytest.approx(want)


def test_frequency_threshold_known_answer_and_mutation_caught():
    assert d_comp.frequency_verdict(0.1990, 0.2000, 0.2010) == "daily"
    assert d_comp.frequency_verdict(0.19995, 0.2000, 0.2010) == "weekly"                      # выигрыш 0,00005 < порога 0,0005 — не оправдывает ежедневное
    assert d_comp.frequency_verdict(0.19995, 0.20005, 0.2000) == "monthly"
    assert d_comp.frequency_verdict(0.19995, 0.2000, 0.2010, thr=0.0) == "daily"             # мутация: порог 0 принимает шум


def test_fold_crit_matches_crit_module():
    res = {(2018, 2017): (np.arange(3), np.array([0.1, 0.2, 0.3])), (2019, 2017): (np.arange(3), np.array([0.2, 0.2, 0.2])),
           (2019, 2018): (np.arange(3), np.array([0.4, 0.4, 0.4]))}
    c = d_comp.crit_of(d_comp.fold_frame(res), "x")
    want = crit.crit(crit.frame({("x", 2018, 2017): 0.2, ("x", 2019, 2017): 0.2, ("x", 2019, 2018): 0.4}, "x"))
    assert c["crit_pen"] == pytest.approx(want["crit_pen"]) and c["by_year"][2019] == pytest.approx(0.3 + grid.PEN * 0.1)


def _trials(path, rows):
    pd.DataFrame(rows, columns=list(d_world.tab.TRIAL_COLS)).to_csv(path, index=False)


def _row(i, direction, wave="D"):
    return {"trial_id": "t%03d" % i, "wave": wave, "direction": direction, "config": "c", "metric_inner": "m", "seed": 42, "time": "1"}


def test_register_budget_and_controls(tmp_path):
    p = tmp_path / "t.csv"
    _trials(p, [_row(i, "hp", "C") for i in range(33)] + [_row(100 + i, "obj_feat", "B") for i in range(22)] + [_row(200 + i, "temporal") for i in range(29)])
    new = [_row(300, "control"), _row(301, "hp"), _row(200, "temporal"), _row(302, "regime"), _row(302, "regime")]
    assert d_world.register_rows(new, p) == 1                                                  # контроль, hp, дубль и повтор не считаются
    with pytest.raises(ValueError):                                                            # 31-я конфигурация
        d_world.register_rows([_row(303, "regime")], p)
    assert len(pd.read_csv(p)) == 33 + 22 + 30


def test_checkpoint_counts_classes(tmp_path):
    p, c = tmp_path / "t.csv", tmp_path / "c.json"
    _trials(p, [_row(1, "hp", "C"), _row(2, "obj_feat", "B"), _row(3, "temporal"), _row(4, "regime"), _row(5, "regime")])
    c.write_text(json.dumps({"used": {"obj_feat": 0, "hp": 0, "temporal_regime": 0}, "limits": {}}))
    out = d_world.checkpoint(c, p)
    assert out["used"] == {"obj_feat": 1, "hp": 1, "temporal_regime": 3} and out["used_stream_d"]["temporal_regime"] == 3
    _trials(p, [_row(i, "temporal") for i in range(31)])
    with pytest.raises(ValueError):
        d_world.checkpoint(c, p)
