"""P3-17, G17 (Ж-220, У-22): LOTO на 17 через боевой путь `hp_s17_loto.loto_ticker17` / `run_module` на синтетической панели («17» — A–D, «E11» — E, F; сеть и данные проекта не читаются).
Инварианты: прогноз A1 для бумаги i не зависит от её целей и от целей E11 (обучение на 16, выбор по 16); у D2 в выборе и обучении w нет i и E11; состав обучения восстанавливается.
Мутации: «LOTO обучен на 17 вместо 16» (исключённая бумага в обучении) — тест падает."""
from __future__ import annotations

import contextlib

import numpy as np
import pandas as pd
import pytest

from models_v2.s10 import hp_hyb_core as C, hp_hyb_out as O, hp_hyb_run as HR, hp_p17 as X, hp_paths as P, hp_s17_loto as L17

TK = "D"
HP = dict(lr=0.2, rounds=15, patience=5, min_leaf=10, l2=1.0, es_dates=20, min_train=100)
T17, T11 = ("A", "B", "C", "D"), ("E", "F")


def make_panel(seed: int = 0, n: int = 700, tickers=T17 + T11) -> pd.DataFrame:
    """Строка (бумага, дата): признаки z_*, цель RV_{t⁺}, база K1; бумаги вне 17 устроены иначе (знак связи обратный)."""
    dates = pd.bdate_range("2015-01-01", periods=n)
    parts = []
    for t in tickers:
        rng = np.random.default_rng(seed * 100 + ord(t))
        z = {c: rng.normal(size=n - 1) for c in C.Z28}
        k1 = np.exp(rng.normal(0, 0.2, n - 1))
        rv = k1 * np.exp((-1.0 if t in T11 else 1.0) * 0.4 * np.sign(z["z_x_d_m"]) + 0.3 * rng.normal(size=n - 1))
        parts.append(pd.DataFrame({"ticker": t, "date": dates[:-1], "t_plus": dates[1:], "rv_tplus": rv, "sigma2_k1": k1, "mkt_rv_tplus": 0.5 * rv * np.exp(0.2 * rng.normal(size=n - 1)),
                                   "x_d_m": z["z_x_d_m"], "x_w_m": rng.normal(size=n - 1), **z}))
    return pd.concat(parts, ignore_index=True).sort_values(["date", "ticker"]).reset_index(drop=True)


@pytest.fixture
def env(monkeypatch, tmp_path):
    monkeypatch.setattr(P, "OUT", tmp_path)
    monkeypatch.setattr(P, "REP", tmp_path)
    monkeypatch.setattr(X, "TRAIN", list(T17))

    def parts(panel):
        k = panel["sigma2_k1"].to_numpy(float)
        return panel[["ticker", "date"]].assign(hm=0.3 * k, hu=0.7 * k, b2=0.6)
    monkeypatch.setattr(HR, "parts_c", parts)
    return tmp_path


def swap(panel: pd.DataFrame, tickers) -> pd.DataFrame:
    q, m = panel.copy(), panel["ticker"].isin(tickers).to_numpy()
    q.loc[m, "rv_tplus"] = q.loc[m, "rv_tplus"].to_numpy() * np.exp(np.random.default_rng(5).normal(1.0, 0.8, m.sum()))
    return q


def loto(panel, names=("A1",), tk=TK, module="xgb"):
    return L17.loto_ticker17(module, list(names), tk, 2, None, panel, HP)


def test_loto_a1_ne_zavisit_ot_celej_i_i_E11(env):
    p = make_panel()
    a = loto(p)
    assert len(a) > 100 and (a["ticker"] == TK).all() and a["config"].eq("xgb_A1").all()
    assert np.array_equal(a["sigma2"].to_numpy(), loto(swap(p, [TK]))["sigma2"].to_numpy())          # цели i
    assert np.array_equal(a["sigma2"].to_numpy(), loto(swap(p, T11))["sigma2"].to_numpy())           # цели E11


def no_op(tk):
    return contextlib.nullcontext()


def test_mutaciya_loto_obuchen_na_17_lovitsya(env, monkeypatch):
    """Мутант: исключаемая бумага остаётся в обучении (состав 17) — прогноз начинает зависеть от её целей, инвариант выше падает."""
    p = make_panel()
    monkeypatch.setattr(L17, "train_without", no_op)
    a, b = loto(p), loto(swap(p, [TK]))
    assert not np.array_equal(a["sigma2"].to_numpy(), b["sigma2"].to_numpy())


def test_d2_vybor_i_obuchenie_w_bez_i_i_E11(env, monkeypatch):
    seen = []
    orig = X.sel_mask

    def spy(tickers):
        m = orig(tickers)
        seen.append(set(np.asarray(tickers)[m]))
        return m
    monkeypatch.setattr(X, "sel_mask", spy)
    d = loto(make_panel(), ("A1", "D2"), module="lgbm")
    assert set(d["config"]) == {"lgbm_A1", "lgbm_D2"} and (d["ticker"] == TK).all()
    assert seen and all(s == set(T17) - {TK} for s in seen)                              # выбор года и обучение w — только 16
    assert d.groupby("config")["sigma2"].apply(lambda s: bool(np.isfinite(s).all() and (s > 0).all())).all()


def test_d2_loto_otlichaetsya_ot_puula_i_mutaciya(env, monkeypatch):
    """Чувствительность: D2 без i в обучении не равен D2 пула (бумага в обучении)."""
    p = make_panel()
    real = loto(p, ("A1", "D2"))
    monkeypatch.setattr(L17, "train_without", no_op)
    pool = loto(p, ("A1", "D2"))
    r, q = (x[x["config"] == "xgb_D2"]["sigma2"].to_numpy() for x in (real, pool))
    assert len(r) == len(q) and not np.array_equal(r, q)


def test_sostav_obucheniya_vosstanavlivaetsya_i_chuzhaya_bumaga_otkaz(env):
    before = list(X.TRAIN)
    loto(make_panel())
    assert X.TRAIN == before
    with pytest.raises(RuntimeError):
        with L17.train_without(TK):
            assert TK not in X.TRAIN and len(X.TRAIN) == 3
            raise RuntimeError("сбой внутри")
    assert X.TRAIN == before                                                              # finally: класс № 43 / 52 / 110
    with pytest.raises(ValueError):
        with L17.train_without("E"):                                                      # E11 — не из 17
            pass


def test_run_module_pishet_17_bumag_reestr_odin_raz_i_ne_trogaet_28(env, monkeypatch):
    reg = env / "ALL_TRIALS.csv"
    reg.write_text("key,program,source,direction,config,kind\nЖ-217-LOTO:xgb:A1:abc,Ж-217,x,y,{},trial\n", encoding="utf-8")
    monkeypatch.setattr(O, "REG", reg)
    monkeypatch.setattr(O, "LOCK", env / "x.lock")
    p = make_panel()
    L17.run_module("xgb", ["A1"], 2, lambda m: None, panel=p, hp=HP)
    out = pd.read_parquet(env / "loto_p17_xgb.parquet")
    assert set(out["ticker"]) == set(T17) and set(out["config"]) == {"xgb_A1"}
    L17.run_module("xgb", ["A1"], 2, lambda m: None, panel=p, hp=HP)                       # повтор: из кэша, строка реестра не дублируется
    rows = reg.read_text(encoding="utf-8").splitlines()
    assert sum(r.startswith("Ж-220-p17-LOTO:xgb:A1:") for r in rows) == 1 and sum(r.startswith("Ж-217-LOTO:") for r in rows) == 1
    assert not (env / "loto_xgb.parquet").exists() and not (env / "loto.parquet").exists()      # файлы на 28 не созданы / не тронуты
