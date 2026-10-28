"""PR #318, тестировщик T1: уровни факторного плана, комиссия 0,035, цель без 2019, лимит процессов, validation.
Синтетика (`SynthEvaluator`), tmp_path. Схема как в test_sweep_post_ledger: `ok_*` — проверка, `test_*` — на коде проходит,
`test_mut_*` — с подложенной ошибкой (monkeypatch) та же проверка обязана вернуть ложь / поймать."""
import argparse

import pytest

from fastbt import sweep_run_iter as it
from fastbt.sweep import Sweep
from fastbt.sweep_spec import pkey
from test_sweep_iter_util import keys, run

BAD_PERIOD = {"tsel": "tune2018", "kontrol": "ctrl2019", "validation": "otkryto"}
MODES = ["factorial", "morris", "optuna"]


def vals(res, k):
    return sorted({r[f"p.{k}"] for r in res["done"][0.04]})


# ---- 1. уровни ----
def ok_urovni(tmp) -> bool:
    res = run("factorial", tmp)
    return vals(res, "a") == [5, 8, 12] and vals(res, "w") == [0.5, 1.0, 1.5] and len(res["done"][0.04]) == 27


def ok_kray(tmp) -> bool:
    return vals(run("factorial", tmp / "l", a0=3), "a") == [3, 5, 8] and vals(run("factorial", tmp / "r", a0=20), "a") == [8, 12, 20]


def ok_vne_setki(tmp) -> bool:
    try:
        run("factorial", tmp, a0=7)
    except ValueError as e:
        return "факторный 3 уровня" in str(e)
    return False


def test_urovni_centr_plyus_minus_shag(tmp_path):
    assert ok_urovni(tmp_path)


def test_mut_urovni_sdvinuty(tmp_path, monkeypatch):
    monkeypatch.setattr(it, "levels", lambda v, b: v[:3])
    assert not ok_urovni(tmp_path)


def test_baza_na_kraju_tri_blizhaishikh_uzla(tmp_path):
    assert ok_kray(tmp_path)


def test_mut_kray_bez_zazhima(tmp_path, monkeypatch):
    monkeypatch.setattr(it, "levels", lambda v, b: v[v.index(b) - 1:v.index(b) + 2])   # на левом краю срез [-1:2] пуст
    with pytest.raises(Exception):
        assert ok_kray(tmp_path)


def test_baza_vne_setki_otkaz(tmp_path):
    assert ok_vne_setki(tmp_path)


def test_mut_baza_vne_setki_bez_otkaza(tmp_path, monkeypatch):
    monkeypatch.setattr(it, "levels", lambda v, b: v[:3])
    assert not ok_vne_setki(tmp_path)


# ---- 2. комиссия 0,035 ----
def ok_komissiya(tmp) -> bool:
    res = run("factorial", tmp)
    n = {r["N"] for r in res["done"][0.04]} | {r["N"] for r in res["done"][0.035]}
    return len(res["ledger"]) == 27 and n == {27} and keys(res, 0.04) == keys(res, 0.035)


def test_komissiya_0035_ne_uvelichivaet_N(tmp_path):
    assert ok_komissiya(tmp_path)


def test_mut_komissiya_0035_pishet_v_ledger(tmp_path, monkeypatch):
    orig = Sweep.evaluate

    def ev(self, params, period="train", comm_pct=None):
        r = orig(self, params, period, comm_pct)
        if period == "setup" and comm_pct is not None and comm_pct != self.spec.comm_pct[0]:
            self.ledger[pkey({**self.spec.base, **params}) + "@0.035"] = r["sr_d"]
        return r
    monkeypatch.setattr(Sweep, "evaluate", ev)
    assert not ok_komissiya(tmp_path)


# ---- 3. 2019 не в цели ----
def ok_cel_2018(tmp) -> bool:
    rows = run("factorial", tmp)["done"][0.04]
    return all(abs(r["objective"] - r["tune_sharpe_2018"]) < 1e-12 and r["ctrl_sharpe_2019"] != r["tune_sharpe_2018"] for r in rows)


def test_2019_ne_v_celi(tmp_path):
    assert ok_cel_2018(tmp_path)


def test_mut_cel_po_train(tmp_path, monkeypatch):
    monkeypatch.setattr(it, "periods", lambda raw: ("train", "control"))
    assert not ok_cel_2018(tmp_path)


# ---- 5. лимит процессов ----
def args(**kw):
    d = dict(jobs=1, n_jobs=1, mode="optuna", params=None, paths=2, trials=3, yaml=["S1_x.yaml"], out="o")
    return argparse.Namespace(**{**d, **kw})


def ok_limit(**kw) -> bool:
    try:
        it.run_cli(args(**kw))
    except ValueError as e:
        return "1..6" in str(e)
    except Exception:
        return False
    return False


@pytest.mark.parametrize("kw", [dict(n_jobs=7), dict(jobs=7), dict(n_jobs=0)])
def test_limit_processov_6(kw):
    assert ok_limit(**kw)


def test_mut_limit_podnyat_do_8(monkeypatch):
    monkeypatch.setattr(it, "MAX_JOBS", 8)
    assert not ok_limit(n_jobs=7)


def test_n_jobs_6_dopustimo_i_dohodit_do_optuna(monkeypatch):
    seen = []

    class Inline:
        def __init__(self, n): pass
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def map(self, f, it_): return [f(x) for x in it_]
    monkeypatch.setattr(it, "ProcessPoolExecutor", Inline)
    monkeypatch.setattr(it, "run_mode", lambda name, path, mode, o, out: seen.append(o) or dict(secs=0.0, n_before=0, ledger={}))
    it.run_cli(args(n_jobs=6))
    assert seen[0]["n_jobs"] == 6


# ---- 9. validation ----
def ok_val_flag(mode, tmp) -> bool:
    try:
        run(mode, tmp, period=BAD_PERIOD)
    except ValueError as e:
        return "validation" in str(e)
    return False


def ok_val_sweep(mode, tmp) -> bool:
    try:
        run(mode, tmp)
    except PermissionError as e:
        return "validation закрыта" in str(e)
    except Exception:                       # мутант: validation прочитана, ломается уже дальше — отказа нет
        return False
    return False


@pytest.mark.parametrize("mode", MODES)
def test_validation_otkrytaya_v_setke_otkaz(tmp_path, mode):
    assert ok_val_flag(mode, tmp_path)


@pytest.mark.parametrize("mode", MODES)
def test_mut_validation_flag_ignoriruetsya(tmp_path, monkeypatch, mode):
    monkeypatch.setattr(it, "periods", lambda raw: ("setup", "control"))
    assert not ok_val_flag(mode, tmp_path)


@pytest.mark.parametrize("mode", MODES)
def test_validation_bez_open_validation_otkaz(tmp_path, monkeypatch, mode):
    monkeypatch.setattr(it, "periods", lambda raw: ("validation", "control"))     # режим пытается читать 2020–2021
    assert ok_val_sweep(mode, tmp_path)


@pytest.mark.parametrize("mode", MODES)
def test_mut_validation_otkryta_molcha(tmp_path, monkeypatch, mode):
    orig = Sweep.__init__

    def init(self, *a, **k):
        orig(self, *a, **k)
        self._val_open = True
    monkeypatch.setattr(Sweep, "__init__", init)
    monkeypatch.setattr(it, "periods", lambda raw: ("validation", "control"))
    assert not ok_val_sweep(mode, tmp_path)
