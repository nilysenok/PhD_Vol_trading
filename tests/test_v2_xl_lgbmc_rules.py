"""LGBM волна C: выбор базы C-L0, правило оси «с эффектом», бюджет и реестр C, интерполяция C-L2, годовой выбор по прошлым целям; мутации."""
import json

import numpy as np
import pandas as pd
import pytest

from models_v2.s6.lgbm import c_hp, c_l0, c_l1, c_l2, c_nbr, c_world, grid, tab

H = {Y: c_hp.lift(grid.Hp(8, 10, 2.0, 0.05, 1.0, 0.8, 3)) for Y in (2017, 2018)}


def test_choose_min_and_tie_rule():
    assert c_l0.choose({"own_top3": 0.208, "own_F4": 0.211, "inh_top3": 0.209, "inh_F4": 0.194}) == "inh_F4"
    assert c_l0.choose({"own_top3": 0.2, "own_F4": 0.3, "inh_top3": 0.2 + 1e-13, "inh_F4": 0.2}) == "own_top3"      # ничья -> собственные, top3 первым


def test_l1_items_at_most_18_and_unique_and_skip_duplicates(monkeypatch):
    monkeypatch.setattr(c_l0, "base_cfg", lambda W: ("B", object(), H))
    its = c_l1.items({})
    assert 0 < len(its) <= 18 and len({i.name for i in its}) == len(its)
    assert {i.direction for i in its} == {"hp"}
    one = {Y: c_hp.lift(grid.Hp(8, 10, 2.0, 0.05, 1.0, 1.0, 3)) for Y in (2017, 2018)}     # bag уже 1,0 -> верхняя точка оси bag = база, не trial
    monkeypatch.setattr(c_l0, "base_cfg", lambda W: ("B", object(), one))
    assert "C1_bag_hi" not in {i.name for i in c_l1.items({})}
    with pytest.raises(ValueError):
        c_l1.items({}, limit=3)                                                              # превышение бюджета поднимает ошибку


def test_axis_effect_rule(monkeypatch):
    crit = {"B": 0.2000, "C1_l2_lo": 0.1985, "C1_l2_hi": 0.2010, "C1_lr_lo": 0.1995, "C1_lr_hi": 0.2002}
    monkeypatch.setattr(c_world, "get", lambda n: (None, {"crit": {"crit_pen": crit[n]}}))
    its = [type("I", (), {"name": n})() for n in crit if n != "B"]
    t = c_l1.axis_table(its, "B").set_index("axis")
    assert bool(t.loc["l2", "effect"]) and t.loc["l2", "best_side"] == "lo"
    assert not bool(t.loc["lr", "effect"])                       # Δ = −0,0005 > −EFFECT: шум, ось без эффекта
    assert not bool(t.loc["ff", "effect"]) and np.isnan(t.loc["ff", "best_d_crit"])        # точек нет -> эффекта нет


def test_mutation_effect_threshold_zero_would_accept_noise(monkeypatch):
    monkeypatch.setattr(c_world, "get", lambda n: (None, {"crit": {"crit_pen": {"B": 0.2, "C1_lr_lo": 0.1999}[n]}}))
    its = [type("I", (), {"name": "C1_lr_lo"})()]
    assert not bool(c_l1.axis_table(its, "B").set_index("axis").loc["lr", "effect"])
    monkeypatch.setattr(c_l1, "EFFECT", 0.0)                      # мутация: порог снят — шум Δ = −0,0001 принят за эффект
    assert bool(c_l1.axis_table(its, "B").set_index("axis").loc["lr", "effect"])


def test_interp_known_answers_and_sign():
    h = c_hp.lift(grid.Hp(8, 10, 2.0, 0.05, 1.0, 0.8, 3))
    assert c_l2.interp(h, "min_data", 0.0).min_data == 10
    assert c_l2.interp(h, "min_data", -1.0).min_data == 5 and c_l2.interp(h, "min_data", 1.0).min_data == 20
    assert c_l2.interp(h, "min_data", 0.5).min_data == 15
    g = c_l2.interp(h, "num_leaves", 1.0)
    assert (g.num_leaves, g.max_depth) == (16, 4)                 # лист и глубина двигаются вместе
    assert c_l2.interp(c_hp.lift(grid.Hp(7, 30, 2.0, 0.05, 1.0, 0.8)), "max_depth", 0.1).max_depth == -1
    assert c_l2.interp(c_hp.lift(grid.Hp(7, 30, 2.0, 0.05, 1.0, 0.8)), "max_depth", -0.9).max_depth == 3


def test_build_applies_every_year_and_absolute_axes():
    b = c_l2.build(H, {"min_data": 1.0, "l1": 3.0, "bag": 0.7})
    assert all(h.min_data == 20 and h.l1 == 3.0 and h.bag == 0.7 for h in b.values())
    assert c_l2.N_TRIALS <= 20 and c_l2.N_STARTUP < c_l2.N_TRIALS and c_l2.SEED == 42


def test_register_only_hp_and_budget(tmp_path, monkeypatch):
    t, c = tmp_path / "T.csv", tmp_path / "C.json"
    b = [tab.trial_row("b%d" % i, "hp", "c", "m", 1.0) for i in range(15)]
    mk = lambda n: [{**tab.trial_row("c%d" % i, "hp", "c", "m", 1.0), "wave": "C"} for i in range(n)]   # noqa: E731
    c.write_text(json.dumps({"used": {"hp": 0, "obj_feat": 0}}))

    def run(rows):
        pd.DataFrame(rows).to_csv(t, index=False)
        return c_world.checkpoint(c, t)
    assert run(b)["used_stream_c"]["hp"] == 0                          # волна B (15) в лимит C не входит
    assert run(b + mk(40))["used_stream_c"]["hp"] == 40 and run(b + mk(40))["used"]["hp"] == 55
    with pytest.raises(ValueError):
        run(b + mk(41))                                                # 41 > 40
    with pytest.raises(ValueError):
        run(b + mk(3) + mk(1))                                         # дубль trial_id c0


def test_register_takes_only_hp_direction_controls_are_not_trials(monkeypatch):
    got = []
    monkeypatch.setattr(tab, "register", lambda rows: got.extend(rows) or len(rows))
    st = {"A": "hp", "B": "control"}
    monkeypatch.setattr(c_world, "get", lambda n: (None, {"direction": st[n], "trial_id": "id" + n, "crit": {"crit_pen": 0.2, "crit_mean": 0.1}, "desc": "d", "secs_total": 1.0}))
    assert c_world.register(["A", "B"]) == 1 and [r["trial_id"] for r in got] == ["idA"] and got[0]["wave"] == "C"


def test_annual_pick_uses_only_pen_of_own_year(tmp_path, monkeypatch):
    """Выбор для года Y — по pen_Y (цели < 01.01.Y). Мутация «pen года Y+1» даёт другой выбор — тест ловит."""
    d = pd.date_range("2017-01-02", periods=4, freq="365D")
    fc = lambda v: pd.DataFrame({"d": d, "target_date": d + pd.Timedelta(days=1), "sigma2": v, "rv": 1.0})   # noqa: E731
    st = lambda p17, p18: {"crit": {"crit_pen": 0.0, "by_year": {"2017": p17, "2018": p18}}, "kind": "ln", "trial_id": "x"}   # noqa: E731
    store = {"A": (fc(1.0), st(0.1, 0.9)), "B": (fc(2.0), st(0.5, 0.2))}
    monkeypatch.setattr(c_world, "get", lambda n: store[n])
    monkeypatch.setattr(c_world, "paths", lambda n, root=None: (tmp_path / (n + ".parquet"), tmp_path / (n + ".json")))
    c_nbr.annual_pick({}, ["A", "B"])
    out = pd.read_parquet(tmp_path / "C_annual_pick.parquet")
    st_pick = json.loads((tmp_path / "C_annual_pick.json").read_text())["hp_by_year"]
    assert st_pick == {"2017": "A", "2018": "B"}
    yr = pd.to_datetime(out["target_date"]).dt.year
    assert (out.loc[yr == 2017, "sigma2"] == 1.0).all() and (out.loc[yr == 2018, "sigma2"] == 2.0).all()
