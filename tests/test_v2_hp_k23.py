"""P1a, G7 (R10, У-3): K2/K3 на произвольной бумаге — через боевой путь (бары -> мир -> замороженные XGB/LGBM) на синтетике.
Мутации: «бумага по умолчанию» (обёртка читает не ту бумагу), «год Y видит цели года Y» (класс № 101), нарушение обрезки будущего. Малые деревья (max_trees 40) — быстрые тесты."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from models_v2.s4 import mkt, select
from models_v2.s4.fit import HP
from models_v2.s10 import hp_har as HH, hp_k23 as K, hp_k23_calc as KC, hp_k23_world as KW, hp_synth as SS

I2 = {Y: HP(2, 30.0, 2.0, 0.1, 0.8, 0.5, 40) for Y in KW.YEARS}
N = 900


@pytest.fixture(scope="module")
def sw():
    base = SS.weekdays(N)
    skip_b = [base[i] for i in range(300, 330)]
    skip_t = [d for d in base if d < pd.Timestamp("2017-10-02")]
    bars, imo, _ = SS.world(N, ("A", "B", "C", "T"), skip={"B": skip_b, "T": skip_t})
    return bars, imo


def pool_of(bars: dict) -> list:
    return [mkt.paper(bars[t]) for t in ("A", "B", "C")]


def panel_of(t: str, w: dict, step: int = 6) -> pd.DataFrame:
    g, F = w["grid"], w["F"]
    return pd.DataFrame({"ticker": t, "date": F.dates[g], "t_plus": F.dates[g + 1], "rv_tplus": F.rv[g + 1], "sigma2_k1": np.nan, "gap": False, "cross_seg": False}).iloc[::step].reset_index(drop=True)


def test_obrezka_buduschego_boevoj_put(sw):
    """Порча баров бумаги, IMOEX и пула после d0 не меняет σ̂² K2/K3 на датах <= d0 (и меняет после — тест не пустой)."""
    bars, imo = sw
    w = KW.build_world(bars["A"], imo, pool_of(bars))
    d0 = pd.Timestamp(w["F"].dates[w["grid"]][len(w["grid"]) // 2])
    rows = w["grid"][::9]
    a = KC.forecast_frame(w, I2, rows, 2)
    bad = {t: HH.corrupt_after(b, d0) for t, b in bars.items()}
    w2 = KW.build_world(bad["A"], HH.corrupt_after(imo, d0), pool_of(bad))
    b = KC.forecast_frame(w2, I2, rows, 2)
    before = (a["date"] <= d0).to_numpy()
    assert before.sum() > 20 and (~before).sum() > 20
    for c in ("sigma2_k2", "sigma2_k3"):
        assert np.array_equal(a.loc[before, c].to_numpy(), b.loc[before, c].to_numpy()), c
        assert np.abs(a.loc[~before, c].to_numpy() - b.loc[~before, c].to_numpy()).max() > 0.0


def test_bumaga_po_umolchaniyu_mutaciya_lovitsya(sw):
    """Обёртка обязана брать бары ПЕРЕДАННОЙ бумаги: мутант (всегда A) даёт другую сетку/RV цели -> отказ; боевой путь B проходит и отличается от A."""
    bars, imo = sw
    pool = pool_of(bars)
    wb = KW.build_world(bars["B"], imo, pool)
    pb = panel_of("B", wb, 8)
    out_b, meta = K.compute_ticker("B", pb, lambda t: bars[t], imo, pool, I2)
    wa = KW.build_world(bars["A"], imo, pool)
    out_a, _ = K.compute_ticker("A", panel_of("A", wa, 8), lambda t: bars[t], imo, pool, I2)
    assert np.isfinite(out_b["sigma2_k2"]).sum() > 20 and meta["y0_used"] == 2017
    assert not np.array_equal(out_a["rv_tplus"].to_numpy()[:20], out_b["rv_tplus"].to_numpy()[:20])
    with pytest.raises(ValueError, match="не совпали"):
        K.compute_ticker("B", pb, lambda t: bars["A"], imo, pool, I2)


def test_u3_pervyj_god_i_nan_do_nego(sw):
    """У-3: у бумаги с историей с 02.10.2017 пар до 01.01.2018 < 126, до 01.01.2019 >= 126 -> прогнозы только с 2019; раньше — NaN (не заполняется)."""
    bars, imo = sw
    pool = pool_of(bars)
    w = KW.build_world(bars["T"], imo, pool)
    assert KW.pairs_before(w, 2018) < 126 <= KW.pairs_before(w, 2019) and KW.first_year(w) == 2019
    out, meta = K.compute_ticker("T", panel_of("T", w, 3), lambda t: bars[t], imo, pool, I2)
    yr = out["t_plus"].dt.year
    assert meta["y0_rule"] == meta["y0_used"] == 2019 and (yr < 2019).sum() > 20 and (yr >= 2019).sum() > 20
    assert out.loc[yr < 2019, ["sigma2_k2", "sigma2_k3", "hp_year"]].isna().all().all()
    assert np.isfinite(out.loc[yr >= 2019, ["sigma2_k2", "sigma2_k3"]]).all().all() and (out.loc[yr >= 2019, "hp_year"] == 2019).all()


def test_first_year_monotonno(sw):
    bars, imo = sw
    w = KW.build_world(bars["A"], imo, pool_of(bars))
    assert KW.first_year(w) == 2017
    assert KW.first_year(w, need=KW.pairs_before(w, 2017) + 1) == 2018
    assert KW.first_year(w, need=10 ** 6) is None


SMALL = (HP(2, 30.0, 2.0, 0.1, 0.8, 0.5, 40), HP(3, 30.0, 2.0, 0.1, 0.8, 0.5, 40))


def _scores_2018(bars, imo, pool_fn, corrupt: bool) -> pd.DataFrame:
    d0 = pd.Timestamp("2017-12-31")
    b = {t: (HH.corrupt_after(x, d0) if corrupt else x) for t, x in bars.items()}
    w = KW.build_world(b["A"], HH.corrupt_after(imo, d0) if corrupt else imo, pool_fn(b))
    return KW.loyo_scores(w, (2018,), 2)


def test_loyo_ne_vidit_celi_goda_Y(sw, monkeypatch):
    """И2 года Y=2018 строится по целям < 01.01.2018: порча всех входов после 31.12.2017 не меняет оценки LOYO (боевой `select.run_loyo`)."""
    monkeypatch.setattr(select, "GRID132", SMALL)
    bars, imo = sw
    a, b = _scores_2018(bars, imo, pool_of, False), _scores_2018(bars, imo, pool_of, True)
    assert len(a) == 2 and np.isfinite(a["score"]).all() and a["score"].tolist() == b["score"].tolist()


def test_mutaciya_god_Y_vidit_celi_goda_Y_lovitsya(sw, monkeypatch):
    """Класс № 101: мутант `cut_row(F, Y+1)` (выбор года Y видит цели года Y) -> оценки зависят от данных после 31.12.2017 -> проверка инвариантности его ловит."""
    monkeypatch.setattr(select, "GRID132", SMALL)
    orig = select.cut_row
    monkeypatch.setattr(select, "cut_row", lambda F, Y: orig(F, Y + 1))
    bars, imo = sw
    a, b = _scores_2018(bars, imo, pool_of, False), _scores_2018(bars, imo, pool_of, True)
    assert a["score"].tolist() != b["score"].tolist()


def test_hp_ravny_i2_bumagi_i_sdvigi_x2(sw):
    """Знание ответа: K2 использует HP И2 бумаги со сдвигами X2_DECISION (не «жёсткие» SBER); метки HP пишутся в выход."""
    bars, imo = sw
    w = KW.build_world(bars["A"], imo, pool_of(bars))
    fr = KC.forecast_frame(w, I2, w["grid"][:3], 2)
    assert fr["hp_k3"].iloc[0] == KC.c_hp.lift_y({2017: KC.grid.from_i2(I2[2017])})[2017].label()
    moved = KC.c_space.apply({Y: KC.c_space.to_hpc(h) for Y, h in I2.items()}, KC.x2_moves())[2017]
    assert fr["hp_k2"].iloc[0] == KC.hpc_label(moved) and moved != KC.c_space.to_hpc(I2[2017])
