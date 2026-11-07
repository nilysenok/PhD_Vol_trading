"""P0, G1–G3, G5–G6: инвентарь, тождества, проверка сдвига (Д1), панель — на синтетике + проверка наличия артефактов листа (после запуска стадий)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from models_v2.s10 import hp_daily as HD, hp_feat as HF, hp_har as HH, hp_inv as HI, hp_panel as PN, hp_paths as P, hp_port as PO, hp_shift as SH, hp_sync as SY, hp_synth as SS


@pytest.fixture(scope="module")
def mini():
    probe = SS.weekdays(900)
    bars, imo, cal = SS.world(900, ("A", "B", "C", "D", "E", "F"), skip={"A": [probe[i] for i in range(500, 508)]})
    daily = {t: HD.ticker_daily(b, imo) for t, b in bars.items()}
    return bars, imo, daily


def test_shift_plus_one_lovitsya_i_chestnyj_ryad_prohodit(mini):
    """Известный ответ: ln RV_d даёт максимум на k = 0; тот же ряд, сдвинутый на +1 строку (σ_{d+1} на дне d, мутация № 1), — на k = +1."""
    _, _, daily = mini
    d = daily["B"]
    y = np.log(d["rv"])
    ok = SH.verdict(SH.corr_k(y, y), len(y))
    leak = SH.verdict(SH.corr_k(y.shift(-1), y), len(y))
    assert ok["ok"] and ok["argmax"] == 0
    assert not leak["ok"] and leak["argmax"] == 1 and leak["diag"].startswith("РАЗБОР")


def test_table_one_vse_ryady_i_utechka_v_k1(mini):
    _, _, daily = mini
    t = daily["B"]
    k1 = HH.series(t)
    rows = SH.table_one("B", t, daily["C"]["rv"], k1, start="2017-01-03")
    assert {r["series"] for r in rows} >= {"ln_rv", "ln_rq", "ln_rv_w", "beta", "corr", "ln_imoex_rv", "k1_sigma2@d"}
    assert all(r["ok"] for r in rows if r["series"] != "ln_rv_rel")           # ln_rv_rel на синтетике не связан с RV (corr ≈ 0.01): шумовой разбор
    assert [r["diag"].startswith("слабая") for r in rows if r["series"] == "ln_rv_rel"] == [True]
    leak = k1.assign(sigma2=k1["sigma2"].shift(-1).ffill())                  # прогноз «из завтра»: σ̂ дня d+1 записан на d
    bad = [r for r in SH.table_one("B", t, daily["C"]["rv"], leak) if r["series"] == "k1_sigma2@d"][0]
    assert not bad["ok"] and bad["argmax"] == 1 and bad["diag"].startswith("РАЗБОР")


def test_tozhdestva_i_mutatsiya_chasti_ne_v_summe(mini):
    _, _, daily = mini
    t = daily["B"]
    idn = HD.identities(t)
    assert idn["all_ok"] and idn["parts_sum"] <= HD.TOL_ID and idn["c_plus_j"] <= HD.TOL_ID and idn["sign_sum"] <= HD.TOL_ID
    broken = t.assign(rv_open=t["rv_open"] * 1.001)
    assert not HD.identities(broken)["all_ok"]


def test_inventar_dyry_i_gap(mini, monkeypatch):
    bars, imo, daily = mini
    monkeypatch.setattr(P, "SECTOR", {**P.SECTOR, "A": "тест"})
    core_a = HD.core(bars["A"])
    r = HI.one("A", bars["A"], core_a, HD.core(imo, None).index)
    assert r["n_holes_vs_imoex"] == 8 and r["n_gap_pairs_gt5"] == 1 and r["n_days"] == len(core_a)
    gp = HI.gap_pairs(core_a.index)
    assert int((gp > P.GAP_DAYS).sum()) == 1 and int(gp.max()) >= 10


def test_poperechnye_rang_i_minimum_bumag():
    w = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, np.nan], "c": [2.0, 1.0], "d": [4.0, 5.0], "e": [0.0, 3.0], "f": [5.0, 0.0]})
    up = (w > 1).astype(float).where(w.notna())
    cs = SY.cross_stats(w, up)
    assert cs["m_med"].iloc[0] == 2.5 and cs["n_cross"].tolist() == [6, 5]
    assert cs["m_med"].iloc[1] == 2.0 and cs["m_share_up"].iloc[1] == pytest.approx(3 / 5)
    rank, dev = SY.cross_rank(w, cs["m_med"])
    assert np.isnan(rank["b"].iloc[1]) and dev["a"].iloc[0] == pytest.approx(1.0 - 2.5)
    few = SY.cross_stats(w.iloc[:, :3], up.iloc[:, :3])
    assert few["m_med"].isna().all()                                      # < 5 бумаг — NaN, не оценка по горстке


def test_panel_check_lovit_oshibki(mini, monkeypatch):
    bars, imo, daily = mini
    monkeypatch.setattr(P, "SECTOR", {**P.SECTOR, **{t: "тест" for t in daily}})
    imd = HD.index_daily(imo)
    k1 = {t: HH.series(d) for t, d in daily.items()}
    feats = PN.features_all(daily, imd, k1)
    rows = [PN.ticker_rows(t, daily[t], k1[t], feats[t], imd["rv"]) for t in daily]
    panel = pd.concat(rows, ignore_index=True)
    bad = PN.check(panel, daily)
    assert bad == ["бумаг 6 ≠ 28"]                                          # единственное замечание — число бумаг синтетики; всё остальное чисто
    broken = panel.copy()
    broken.loc[panel.index[10], "t_plus"] = broken.loc[panel.index[10], "t_plus"] + pd.Timedelta(days=1)
    assert any("не следующая строка" in b or "gap" in b for b in PN.check(broken, daily))
    assert set(HF.X_COLS + HF.M_COLS + HF.C_COLS) <= set(panel.columns)
    assert (panel["t_plus"] > panel["date"]).all()


def test_perenosimost_pravilo_na_sinteticheskih_faktah():
    f = {"has_cols": True, "share_days_px_ok": 1.0, "share_days_vol_pos": 1.0, "n_days_before_2015": 140, "share_days_sync_imoex": 1.0, "n_pairs_at_20170103": 400}
    assert PO.verdict(f, True)["K2_portable"]
    short = PO.verdict({**f, "n_days_before_2015": 10}, True)
    assert not short["K2_portable"] and short["K6_portable"] and "LOYO" in short["reason_not_portable"]
    assert not PO.verdict({**f, "share_days_sync_imoex": 0.5}, True)["K6_portable"]


def test_artefakty_lista_posle_zapuska():
    """После стадий run_hp: 28 бумаг в инвентаре и переносимости, признаки описаны в P0_FEATURES.md, панель с SHA — в отчёте."""
    need = [P.REP / n for n in ("P0_INVENTORY.csv", "P0_CALENDAR.csv", "P0_PORTABILITY.csv", "P0_FEATURES.md", "P0_SHIFT.csv", "P0_K1.csv", "CHECKPOINT.json")]
    if not all(p.exists() for p in need):
        pytest.skip("стадии run_hp ещё не запускались")
    assert len(pd.read_csv(P.REP / "P0_INVENTORY.csv", comment="#")) == 28 and len(pd.read_csv(P.REP / "P0_PORTABILITY.csv", comment="#")) == 28
    doc = (P.REP / "P0_FEATURES.md").read_text()
    assert all(c in doc for c in HF.X_COLS + HF.M_COLS + HF.C_COLS)
