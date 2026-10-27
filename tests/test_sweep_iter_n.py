"""PR #318, тестировщик T1: накопление N между режимами, повтор точки, seed Optuna, top3, неактивный условный параметр.
Синтетика (`SynthEvaluator`), tmp_path. Схема `ok_*` / `test_*` / `test_mut_*` как в test_sweep_iter.py."""
import itertools
import random
import uuid
from pathlib import Path

import pandas as pd

from fastbt import sweep_drivers as sd
from fastbt import sweep_run, sweep_run_iter as it
from fastbt.sweep import Sweep
from fastbt.sweep_spec import pkey
from fastbt.sweep_synth import SynthEvaluator
from test_sweep_iter_util import keys, run, setka


def fact_keys(a=(5, 8, 12), w=(0.5, 1.0, 1.5), z=(1, 2, 3)) -> set:
    base = {"a": 8, "w": 1.0, "z": 2, "mode": "on", "cb.q": 1}
    return {pkey({**base, "a": x, "w": y, "z": v}) for x, y, v in itertools.product(a, w, z)}


# ---- 4. N накапливается ----
def ok_nakoplenie(tmp) -> bool:
    oat = sweep_run.run_setka("S1", setka(), SynthEvaluator(), tmp)
    k_oat = set(oat["ledger"])
    f = run("factorial", tmp)
    ok = set(f["ledger"]) == k_oat | fact_keys() and len(f["ledger"]) > len(k_oat)
    o = run("optuna", tmp, trials=15)
    return ok and set(o["ledger"]) == set(f["ledger"]) | keys(o) and {r["N"] for r in o["done"][0.04]} == {len(o["ledger"])}


def ok_bez_zadvoeniya(tmp) -> bool:
    n1 = len(run("factorial", tmp)["ledger"])
    r2 = run("factorial", tmp)
    return n1 == 27 and len(r2["ledger"]) == 27 and r2["n_before"] == 27 and {r["N"] for r in r2["done"][0.04]} == {27}


def test_N_nakaplivaetsya_oat_factorial_optuna(tmp_path):
    assert ok_nakoplenie(tmp_path)


def test_mut_ledger_na_kazhdyi_zapusk(tmp_path, monkeypatch):
    monkeypatch.setattr(it, "ledger_dir", lambda out, name: str(Path(out) / "_ledger" / name / uuid.uuid4().hex))
    assert not ok_nakoplenie(tmp_path)


def test_povtor_toi_zhe_tochki_ne_zadvaivaet_N(tmp_path):
    assert ok_bez_zadvoeniya(tmp_path)


def test_mut_povtor_pishet_novyi_klyuch(tmp_path, monkeypatch):
    orig, n = Sweep.evaluate, itertools.count()

    def ev(self, params, period="train", comm_pct=None):
        r = orig(self, params, period, comm_pct)
        if period == "setup" and comm_pct == self.spec.comm_pct[0]:
            self.ledger[pkey({**self.spec.base, **params}) + f"#{next(n)}"] = r["sr_d"]
        return r
    monkeypatch.setattr(Sweep, "evaluate", ev)
    assert not ok_bez_zadvoeniya(tmp_path)


# ---- 6. seed Optuna ----
def ok_seed(tmp) -> bool:
    a, b = run("optuna", tmp / "1", trials=12), run("optuna", tmp / "2", trials=12)
    cols = ["objective", "tune_sharpe", "ctrl_sharpe", "dsr", "rank"]
    return set(a["ledger"]) == set(b["ledger"]) and [r["_pk"] for r in a["done"][0.04]] == [r["_pk"] for r in b["done"][0.04]] \
        and pd.DataFrame(a["done"][0.04])[cols].equals(pd.DataFrame(b["done"][0.04])[cols])


def test_optuna_odin_seed_odin_rezultat(tmp_path):
    assert ok_seed(tmp_path)


def test_mut_optuna_seed_sluchaen(tmp_path, monkeypatch):
    orig = sd.optuna_search

    def bad(ev, base, space, period, n, seed, **kw):
        return orig(ev, base, space, period, n, random.randrange(10**9), **kw)
    monkeypatch.setattr(sd, "optuna_search", bad)
    assert not ok_seed(tmp_path)


# ---- 7. top3 дополняется ----
def modes_in_top3(out) -> list:
    return sorted(pd.read_csv(Path(out) / "S1_top3.csv", comment="#")["mode"])


def ok_top3(tmp) -> bool:
    run("factorial", tmp)
    run("optuna", tmp, trials=12)
    both = modes_in_top3(tmp) == ["factorial"] * 3 + ["optuna"] * 3
    run("factorial", tmp)                                    # повтор режима заменяет свои строки, чужие остаются
    return both and modes_in_top3(tmp) == ["factorial"] * 3 + ["optuna"] * 3


def test_top3_dopolnyaetsya_a_ne_zatiraetsya(tmp_path):
    assert ok_top3(tmp_path)


def test_mut_top3_zatiraetsya(tmp_path, monkeypatch):
    orig = it.write

    def bad(out, name, mode, h, done, comms):
        (Path(out) / f"{name}_top3.csv").unlink(missing_ok=True)
        orig(out, name, mode, h, done, comms)
    monkeypatch.setattr(it, "write", bad)
    assert not ok_top3(tmp_path)


# ---- 8. неактивный условный параметр ----
def ok_neaktivnyi(tmp) -> bool:
    kw = dict(mode=("off",), mode0="off", a=(3, 5), a0=5, w=(1.0,), z=(2,))        # cb.q неактивен (mode=off) всегда
    o = run("optuna", tmp / "o", trials=40, **kw)
    m = run("morris", tmp / "m", params=None, paths=8, **kw)
    return all(len(r["ledger"]) <= 2 and all(dict(eval_k(k))["cb.q"] == 1 for k in r["ledger"]) for r in (o, m)) \
        and {r["N"] for r in o["done"][0.04]} == {len(o["ledger"])}


def eval_k(k):
    import json
    return json.loads(k)


def test_neaktivnyi_uslovnyi_parametr_ne_razduvaet_N(tmp_path):
    assert ok_neaktivnyi(tmp_path)


def test_mut_fix_ne_svodit_k_baze(tmp_path, monkeypatch):
    monkeypatch.setattr(it, "make_fix", lambda exp: (lambda p: dict(p)))
    assert not ok_neaktivnyi(tmp_path)
