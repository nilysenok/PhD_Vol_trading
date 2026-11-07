"""P2, G9/G10: сквозной боевой путь `hp_hyb_run.Run` (A1, B1, B2, C1, C2, D1, D2; A2/A3 — тем же кодом) на синтетической панели с именами признаков панели движка — XGB и LGBM (сеть и данные проекта не читаются).
Обрезка будущего: повреждение ВСЕЙ панели после d (признаки строк после d, RV целей t⁺ > d, база K1, поправочные части) не меняет σ̂² на даты <= d ни одной конфигурации (побитно).
Реестр: строка конфигурации добавляется один раз, used считается по реестру; кэш проходов реагирует на подмену входа (Д5)."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from models_v2.s10 import hp_hyb_core as C, hp_hyb_out as O, hp_hyb_run as HR, hp_paths as P

HP = dict(lr=0.2, rounds=15, patience=5, min_leaf=10, l2=1.0, es_dates=20, min_train=100)
CFGS = ["A1", "B1", "B2", "C1", "C2", "D1", "D2"]


def make_panel(seed: int = 0, n: int = 800, tickers=("A", "B", "C", "D")) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2015-01-01", periods=n)
    parts = []
    for t in tickers:
        z = {c: rng.normal(size=n - 1) for c in C.Z28}
        k1 = np.exp(rng.normal(0, 0.2, n - 1))
        rv = k1 * np.exp(0.4 * np.sign(z["z_x_d_m"]) + 0.3 * rng.normal(size=n - 1))
        parts.append(pd.DataFrame({"ticker": t, "date": dates[:-1], "t_plus": dates[1:], "rv_tplus": rv, "sigma2_k1": k1, "mkt_rv_tplus": 0.5 * rv * np.exp(0.2 * rng.normal(size=n - 1)),
                                   "x_d_m": z["z_x_d_m"], "x_w_m": rng.normal(size=n - 1), **z}))
    return pd.concat(parts, ignore_index=True).sort_values(["date", "ticker"]).reset_index(drop=True)


def corrupt(p: pd.DataFrame, d0) -> pd.DataFrame:
    q, rng = p.copy(), np.random.default_rng(11)
    late, tgt = (q["date"] > d0).to_numpy(), (q["t_plus"] > d0).to_numpy()
    for c in C.Z28 + ["x_d_m", "x_w_m"]:
        q.loc[late, c] = rng.normal(size=late.sum()) * 3 + 2
    q.loc[late, "sigma2_k1"] = np.exp(rng.normal(size=late.sum()))
    for c in ("rv_tplus", "mkt_rv_tplus"):
        q.loc[tgt, c] = 20 * np.exp(rng.normal(size=tgt.sum()))
    return q


@pytest.fixture
def env(monkeypatch, tmp_path):
    monkeypatch.setattr(P, "OUT", tmp_path)
    monkeypatch.setattr(P, "REP", tmp_path)

    def parts(panel):                                                               # части K7: функция строки панели (в боевом пути — из дневных таблиц бумаг)
        k = panel["sigma2_k1"].to_numpy(float)
        return panel[C.KEY].assign(hm=0.3 * k, hu=0.7 * k, b2=0.6)
    monkeypatch.setattr(HR, "parts_c", parts)
    return tmp_path


def calc_all(panel: pd.DataFrame, module: str) -> dict:
    r = HR.Run(module, 2, lambda m: None, panel=panel, hp=HP)
    for n in CFGS:
        r.calc(n)
    return r.sig, r


@pytest.mark.parametrize("module", ["xgb", "lgbm"])
def test_obrezka_budushchego_panel_vseh_konfiguracij(module, env):
    p = make_panel()
    d0 = pd.Timestamp("2017-06-30")
    a, r = calc_all(p, module)
    b, _ = calc_all(corrupt(p, d0), module)
    past = (p["date"] <= d0).to_numpy()
    for n in CFGS:
        ok = past & np.isfinite(a[n])
        assert ok.sum() > 500, n
        assert np.array_equal(a[n][ok], b[n][ok]), n                                    # прогноз на d не видит данных после d
        assert (a[n][ok] > 0).all()
    assert any(not np.array_equal(a[n][~past & np.isfinite(a[n]) & np.isfinite(b[n])], b[n][~past & np.isfinite(a[n]) & np.isfinite(b[n])]) for n in CFGS)   # будущее — другое


def test_klyuch_kesha_reagiruet_na_podmenu_vhoda_i_modulya(env):
    p = make_panel(n=300)
    k1 = p["sigma2_k1"].to_numpy(float)
    base = C.cached_runs("xgb", "t", p, ["z_x_d_m"], k1, "s", "qlike", HP, (2,), 2)
    files = lambda: sorted(f.name for f in env.glob("hyb_run_*"))
    assert len(files()) == 1
    C.cached_runs("xgb", "t", p, ["z_x_d_m"], k1, "s", "qlike", HP, (2,), 2)
    assert len(files()) == 1                                                            # тот же вход — попадание в кэш
    q = p.copy()
    q.loc[100, "z_x_d_m"] += 1e-9
    C.cached_runs("xgb", "t", q, ["z_x_d_m"], k1, "s", "qlike", HP, (2,), 2)
    C.cached_runs("lgbm", "t", p, ["z_x_d_m"], k1, "s", "qlike", HP, (2,), 2)
    C.cached_runs("xgb", "t", p, ["z_x_d_m"], k1 * 1.0000001, "s", "qlike", HP, (2,), 2)
    assert len(files()) == 4 and base[2] is not None                                    # подмена признака, модуля, базы — промах


def test_reestr_odna_stroka_na_konfiguraciyu_i_used(monkeypatch, tmp_path):
    reg, cp, old = tmp_path / "ALL_TRIALS.csv", tmp_path / "CHECKPOINT.json", tmp_path / "old.csv"
    old.write_text("key,program,source,direction,config,kind\nЖ-214:x:1,Ж-214,s,d,\"{\"\"a\"\": 1}\",trial\n", encoding="utf-8")
    cp.write_text(json.dumps({"used": 0, "leaf": "P1"}))
    for name, val in (("REG_OLD", old), ("REG", reg), ("CP", cp), ("LOCK", tmp_path / "lock")):
        monkeypatch.setattr(O, name, val)
    assert O.register("xgb", "A1", "panel_global", {"n_rows": 5}) and not O.register("xgb", "A1", "panel_global", {"n_rows": 6})
    assert O.register("lgbm", "A1", "panel_global", {})
    rows = reg.read_text(encoding="utf-8").splitlines()
    assert rows[0].startswith("key,") and rows[1].startswith("Ж-214:x:1") and len(rows) == 4          # старая строка не тронута, повтор не дублируется
    c = json.loads(cp.read_text())
    assert c["used"] == 2 and c["leaf"] == "P2" and c["N_registry"] == 587
