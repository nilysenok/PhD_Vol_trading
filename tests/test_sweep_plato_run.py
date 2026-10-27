"""Э1-в plato: сквозной прогон на SynthEvaluator, факторный план на двух узлах, Optuna: предел активности и параметры с эффектом.
`test_*` — на коде проходит, `test_mut_*` — с подложенной ошибкой та же проверка обязана не пройти."""
import json

import pandas as pd
import pytest

from fastbt import sweep_drivers as sd
from fastbt import sweep_pts as pts
from fastbt import sweep_run_iter as it
from fastbt.sweep_synth import SynthEvaluator
from test_sweep_iter_util import run, setka

TOR = "# статус: тест\n# код\n# вход\nparam,objective_range,n\n"
BASE = dict(params=None, paths=4, trials=12, n_jobs=1, min_frac=None, tornado_min=None)


def tornado(out, rows):
    out.mkdir(parents=True, exist_ok=True)
    (out / "S1_oat_tornado_0.04.csv").write_text(TOR + "".join(f"{k},{v},5\n" for k, v in rows), "utf-8")


def mode(rezhim, out, **o):
    return it.run_mode("S1", setka(), rezhim, {**BASE, **o}, out, evaluator=SynthEvaluator())


def ledger_pts(res):
    return [dict(json.loads(k)) for k in res["ledger"]]


# ---- plato сквозной: соседи в ledger и в N, топ-3 дописывается ----
def test_plato_skvoznoi_sosedi_v_N_top3_dopisyvaetsya(tmp_path):
    run("optuna", tmp_path, trials=15)
    tornado(tmp_path, [("a", 1.0), ("w", 0.5), ("z", 0.3), ("mode", 0.0)])
    n0 = len(json.loads((tmp_path / "_ledger" / "S1" / "ledger.json").read_text("utf-8")))
    res = mode("plato", tmp_path)
    assert res["n_neighbors"] > 0 and len(res["ledger"]) > n0
    t = res["table"]
    assert set(t["N"]) == {len(res["ledger"])} and t["plateau"].notna().all()
    top = pd.read_csv(tmp_path / "S1_top3.csv", comment="#")
    assert set(top["mode"]) == {"optuna", "plato"}                            # прежние строки Optuna не стёрты
    p = top[top["mode"] == "plato"]
    assert 0 < len(p) <= 3 and (p["neighbors"] >= 3).all()
    assert {"tune_sharpe_c0.035", "ctrl_sharpe_c0.035", "dsr", "sdelok_y", "nb_min_objective"} <= set(p.columns)


def test_plato_bez_tornado_otkaz(tmp_path):
    run("optuna", tmp_path, trials=5)
    with pytest.raises(FileNotFoundError):
        mode("plato", tmp_path)


def test_plato_predel_aktivnosti_ubiraet_kandidatov(tmp_path):
    run("optuna", tmp_path, trials=15)
    tornado(tmp_path, [("a", 1.0), ("w", 0.5)])
    assert len(mode("plato", tmp_path, min_frac=1e6)["table"]) == 0           # предел недостижим


def test_plato_menshe_parametrov_s_effektom_menshe_sosedei(tmp_path):
    run("optuna", tmp_path / "x", trials=15)
    tornado(tmp_path / "x", [("a", 1.0), ("w", 0.5), ("z", 0.3)])
    wide = mode("plato", tmp_path / "x")["n_neighbors"]
    run("optuna", tmp_path / "y", trials=15)
    tornado(tmp_path / "y", [("a", 1.0)])
    assert mode("plato", tmp_path / "y")["n_neighbors"] < wide


# ---- факторный план: сетка из двух узлов ----
def test_dva_uzla_oba_bez_tsentra(tmp_path):
    assert pts.levels(["x", "y"], "x") == ["x", "y"] and pts.levels(["x", "y"], "y") == ["x", "y"]
    res = run("factorial", tmp_path, params="a,w,mode")
    assert len(res["done"][0.04]) == 3 * 3 * 2
    assert sorted({r["p.mode"] for r in res["done"][0.04]}) == ["off", "on"]


def test_mut_dva_uzla_tsentr_dublirovan(tmp_path, monkeypatch):
    monkeypatch.setattr(it, "levels", lambda v, b: [v[0], b, v[-1]] if len(v) == 2 else v[v.index(b) - 1:v.index(b) + 2])
    res = run("factorial", tmp_path, params="a,w,mode")
    assert len(res["done"][0.04]) != 3 * 3 * 2                                # три уровня вместо двух — тест выше упал бы


def test_odin_uzel_otkaz():
    with pytest.raises(ValueError):
        pts.levels(["x"], "x")


# ---- Optuna: предел активности и параметры с эффектом ----
def test_optuna_nedopustimye_probi_v_ledger_no_ne_v_tablitse(tmp_path):
    res = mode("optuna", tmp_path, min_frac=1e6)
    assert len(res["ledger"]) > 0 and res["done"][0.04] == []


def test_mut_optuna_bez_shtrafa(tmp_path, monkeypatch):
    monkeypatch.setattr(it, "penalized", lambda min_sd: sd.objective_stable)
    assert len(mode("optuna", tmp_path, min_frac=1e6)["done"][0.04]) > 0      # штрафа нет — тест выше упал бы


def test_optuna_tolko_parametry_s_effektom(tmp_path):
    tornado(tmp_path, [("a", 1.0), ("w", 0.01), ("z", 0.0)])
    ps = ledger_pts(mode("optuna", tmp_path, tornado_min=0.05, trials=20))
    assert {p["w"] for p in ps} == {1.0} and {p["z"] for p in ps} == {2} and len({p["a"] for p in ps}) > 1


def test_mut_optuna_vsya_setka(tmp_path, monkeypatch):
    tornado(tmp_path, [("a", 1.0), ("w", 0.01), ("z", 0.0)])
    monkeypatch.setattr(pts, "effect_params", lambda exp, out, name, c0, thr: list(exp["grid"]))
    ps = ledger_pts(mode("optuna", tmp_path, tornado_min=0.05, trials=20))
    assert len({p["w"] for p in ps}) > 1                                      # w бегает — тест выше упал бы


def test_effect_params_porog(tmp_path):
    tornado(tmp_path, [("a", 0.05), ("w", 0.049), ("z", 0.0)])
    exp = it.load(setka())[1]
    assert pts.effect_params(exp, tmp_path, "S1", 0.04, 0.05) == ["a"]
