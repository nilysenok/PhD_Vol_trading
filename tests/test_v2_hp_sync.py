"""P0, G5 (R4, R5): синхронизация бумаг и признаки панели через БОЕВОЙ путь (бары -> hp_daily -> hp_har -> hp_panel.features_all).
Мутации (должны падать на мутанте и проходить на боевом коде): «чужой день t+1», «ffill отсутствующей бумаги в поперечных», «нормировка с днём t / с будущим»."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from models_v2.s10 import hp_daily as HD, hp_feat as HF, hp_har as HH, hp_panel as PN, hp_sync as SY, hp_synth as SS

T = ("A", "B", "C", "D", "E", "F")
D0_ROW = 700


def prep(bars: dict, imoex: pd.DataFrame) -> tuple:
    daily = {t: HD.ticker_daily(b, imoex) for t, b in bars.items()}
    return daily, HD.index_daily(imoex), {t: HH.series(d) for t, d in daily.items()}


def corrupt(bars: pd.DataFrame, lo: pd.Timestamp, hi: pd.Timestamp | None = None, seed: int = 3) -> pd.DataFrame:
    """Бары с датой > lo (и <= hi, если задано) умножены на случайный множитель."""
    day = bars["begin"].dt.normalize()
    m = (day > lo) if hi is None else ((day > lo) & (day <= hi))
    out, f = bars.copy(), np.exp(np.random.default_rng(seed).normal(0, 0.5, int(m.sum())))
    for c in ("open", "high", "low", "close"):
        out.loc[m, c] = out.loc[m, c].to_numpy() * f
    return out


@pytest.fixture(scope="module")
def base():
    bars, imo, cal = SS.world(900, T)
    return bars, imo, cal, prep(bars, imo)


def test_boevoj_put_futurnost_vseh_bumag(base):
    """Все бары ВСЕХ бумаг и индекса после d0 повреждены: признаки X/M/C (сырые и z) на датах <= d0 не меняются."""
    bars, imo, cal, (daily, imd, k1) = base
    d0 = cal[D0_ROW]
    bad = prep({t: corrupt(b, d0) for t, b in bars.items()}, corrupt(imo, d0))
    f0, f1 = PN.features_all(daily, imd, k1), PN.features_all(*bad)
    for t in T:
        a, b = f0[t].loc[:d0], f1[t].loc[:d0]
        pd.testing.assert_frame_equal(a, b, check_exact=True)
        assert a["z_x_hdev"].notna().any() and a["z_m_med"].notna().any()       # проверка осмысленная: z и K1-признаки определены
        assert not f0[t].loc[d0 + pd.Timedelta(days=1):].equals(f1[t].loc[d0 + pd.Timedelta(days=1):])   # порча действует после d0


def _foreign_day_changes_features(base, patch=None, monkeypatch=None) -> bool:
    """True, если подмена ОДНОГО чужого дня t+1 (бумага B, день d0+1) меняет признаки бумаги A или B на датах <= d0."""
    bars, imo, cal, (daily, imd, k1) = base
    d0, d1 = cal[D0_ROW], cal[D0_ROW + 1]
    bb = dict(bars)
    bb["B"] = corrupt(bars["B"], d0, d1)
    daily1, imd1, k11 = prep(bb, imo)
    if patch:
        monkeypatch.setattr(SY, "cross_stats", patch)
    f0, f1 = PN.features_all(daily, imd, k1), PN.features_all(daily1, imd1, k11)
    return any(not f0[t].loc[:d0].equals(f1[t].loc[:d0]) for t in ("A", "B"))


def test_chuzhoj_den_t_plus_1_boevoj_ne_menyaet(base):
    assert not _foreign_day_changes_features(base)


def test_mutaciya_chuzhoj_den_t_plus_1_lovitsya(base, monkeypatch):
    real = SY.cross_stats
    assert _foreign_day_changes_features(base, lambda lnrel, up: real(lnrel.shift(-1), up.shift(-1)), monkeypatch)


def _hole_world():
    """Бумага B не торгует в день t0 (остальные торгуют): поперечные величины дня t0 — по 5 бумагам."""
    probe = SS.weekdays(900)
    t0 = probe[400]
    bars, imo, cal = SS.world(900, T, skip={"B": [t0]})
    return t0, prep(bars, imo)


def _cross_at_hole(patch=None, monkeypatch=None) -> tuple:
    t0, (daily, imd, k1) = _hole_world()
    if patch:
        monkeypatch.setattr(PN, "cross_frames", patch)
    f = PN.features_all(daily, imd, k1)
    lnrel = [np.log(daily[t].loc[t0, "rv"] / daily[t].loc[t0, "rv_m"]) for t in T if t != "B"]
    assert t0 not in daily["B"].index and t0 not in f["B"].index
    return float(f["A"].loc[t0, "m_med"]), float(np.median(lnrel))


def test_otsutstvuyushchaya_bumaga_ne_vhodit_v_poperechnye():
    got, want = _cross_at_hole()
    assert got == pytest.approx(want, abs=1e-12)


def test_mutaciya_ffill_otsutstvuyushchej_lovitsya(monkeypatch):
    real = PN.cross_frames

    def ffilled(daily):
        a, b = real(daily)
        return a.ffill(), b.ffill()
    got, want = _cross_at_hole(ffilled, monkeypatch)
    assert abs(got - want) > 1e-6          # мутант заметно расходится с медианой по присутствующим бумагам


def _norm_probe(fn) -> dict:
    """Влияние порчи x[k] на медиану/шкалу в k, k−1 и на z до k−1: past_stats(x)[t] не должна зависеть от x[t], x[t+1], … (>= 252 наблюдений истории)."""
    x = pd.Series(np.random.default_rng(0).normal(size=600))
    k = 400
    y, w = x.copy(), x.copy()
    y[k] += 50.0
    w[k + 1:] += 50.0
    m0, s0 = fn(x)
    m1, s1 = fn(y)
    m2, s2 = fn(w)
    return {"uses_day_t": not (m0[k] == m1[k] and s0[k] == s1[k]), "uses_future": not (m0[:k + 1].equals(m2[:k + 1]) and s0[:k + 1].equals(s2[:k + 1])),
            "defined": bool(np.isfinite(m0[k]) and np.isfinite(s0[k]) and np.isnan(m0[251]) and np.isfinite(m0[252]))}


def test_normirovka_tolko_po_proshlomu():
    r = _norm_probe(HF.past_stats)
    assert r == {"uses_day_t": False, "uses_future": False, "defined": True}


def test_mutaciya_normirovka_s_dnem_t_lovitsya():
    def with_t(x, min_hist=HF.MIN_HIST):
        e = x.expanding(min_periods=min_hist)
        return e.median(), (e.quantile(0.75) - e.quantile(0.25)) / 1.349         # без shift(1): день t в статистике
    assert _norm_probe(with_t)["uses_day_t"]


def test_mutaciya_normirovka_s_budushchim_lovitsya():
    def with_future(x, min_hist=HF.MIN_HIST):
        e = x.expanding(min_periods=min_hist)
        return e.median().shift(-1), ((e.quantile(0.75) - e.quantile(0.25)) / 1.349).shift(-1)
    assert _norm_probe(with_future)["uses_future"]


def test_zscore_znanyj_otvet():
    """Известный ответ: на днях s < t x = 0, 1 чередуются; медиана 0,5; шкала 1/1.349; z_t = (x_t − 0,5)·1,349."""
    x = pd.Series(np.tile([0.0, 1.0], 300))
    z = HF.zscore(x)
    assert z.iloc[400] == pytest.approx((x.iloc[400] - 0.5) * 1.349, rel=1e-9)
    assert z.iloc[:252].isna().all()
