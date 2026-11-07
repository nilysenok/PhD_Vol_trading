"""P3-17, G19 (Ж-220, У-25): LOTO на 17 с поперечными признаками по 16 — боевой путь (бары -> hp_daily -> hp_har -> hp_s17_loto16.panel16 -> loto_ticker17), синтетика, сеть и данные проекта не читаются.
Инварианты: признаки бумаги j ≠ i в панели прогона без i = пересчёт по 17 \\ {i}; признаки i — относительно 16 (медиана, ранг, отклонение — вручную); данные i после d прогноз i на ≤ d не меняют (обрезка).
Мутации: исключённая бумага в поперечных признаках своего прогона (крючок `cross_without`) — ловится; «взгляд вперёд» в данных i — ловится обрезкой; состав обучения и крючок восстанавливаются."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from models_v2.s10 import hp_daily as HD, hp_hyb_run as HR, hp_har as HH, hp_p17 as X, hp_paths as P, hp_s17_loto16 as L16, hp_synth as SS

T = ("A", "B", "C", "D", "E", "F", "G", "H")
T17, T11 = T[:6], T[6:]
TK = "C"
HP = dict(lr=0.2, rounds=15, patience=5, min_leaf=10, l2=1.0, es_dates=20, min_train=100)
CROSS = ("m_med", "m_disp", "m_share_up", "c_rank", "c_dev_med")


@pytest.fixture(scope="module")
def world():
    bars, imo, _ = SS.world(900, T)
    daily = {t: HD.ticker_daily(b, imo) for t, b in bars.items()}
    return daily, HD.index_daily(imo), {t: HH.series(d) for t, d in daily.items()}


@pytest.fixture
def env(monkeypatch, tmp_path):
    monkeypatch.setattr(P, "TICKERS", sorted(T))
    monkeypatch.setattr(P, "SECTOR", {t: "s" for t in T})
    monkeypatch.setattr(P, "group", lambda t: "M16" if t in T17 else "E11")
    monkeypatch.setattr(P, "OUT", tmp_path)
    monkeypatch.setattr(X, "TRAIN", list(T17))
    monkeypatch.setattr(HR, "parts_c", lambda panel: panel[["ticker", "date"]].assign(hm=0.3 * panel["sigma2_k1"].to_numpy(float), hu=0.7 * panel["sigma2_k1"].to_numpy(float), b2=0.6))
    return tmp_path


def lnrel(daily, tickers):
    return pd.DataFrame({s: np.log(daily[s]["rv"] / daily[s]["rv_m"]) for s in tickers})


def row(panel, tk, date):
    return panel[(panel["ticker"] == tk) & (panel["date"] == date)].iloc[0]


def test_priznaki_j_po_17_bez_i_a_i_otnositelno_16(world, env):
    daily, imd, k1 = world
    pn = L16.panel16(TK, daily, imd, k1)
    rest = [s for s in T17 if s != TK]
    assert X.TRAIN == list(T17)                                                          # состав обучения глобально не менялся
    ln = lnrel(daily, rest)
    d = pn[(pn["ticker"] == "A") & pn["m_med"].notna()]["date"].iloc[300]
    med = float(np.median(ln.loc[d]))
    for t in ("A", "B", "G"):                                                            # j ≠ i (в т. ч. E11): медиана дня по 16
        assert row(pn, t, d)["m_med"] == pytest.approx(med, rel=1e-12)
    ri = row(pn, TK, d)                                                                  # i: все величины — относительно 16
    x = float(np.log(daily[TK].loc[d, "rv"] / daily[TK].loc[d, "rv_m"]))
    assert ri["m_med"] == pytest.approx(med, rel=1e-12) and ri["c_dev_med"] == pytest.approx(x - med, rel=1e-12)
    assert ri["c_rank"] == pytest.approx(((ln.loc[d] < x).sum() + 1) / (len(rest) + 1), rel=1e-12)
    full = X.build_panel(daily, imd, cross=list(T17), k1=k1)                             # по 17 (прежняя панель): у i медиана другая
    assert row(full, TK, d)["m_med"] != ri["m_med"] and row(full, TK, d)["m_med"] == pytest.approx(float(np.median(lnrel(daily, T17).loc[d])), rel=1e-12)


def test_ne_poperechnye_kolonki_ravny_panelyu_17(world, env):
    daily, imd, k1 = world
    pn, full = L16.panel16(TK, daily, imd, k1), X.build_panel(daily, imd, k1=k1)
    assert pn[["ticker", "date"]].equals(full[["ticker", "date"]])
    skip = set(CROSS) | {"z_" + c for c in CROSS}
    for c in full.columns:
        if full[c].dtype.kind == "f" and c not in skip:
            assert np.array_equal(pn[c].to_numpy(float), full[c].to_numpy(float), equal_nan=True), c
    assert not np.array_equal(pn["m_med"].to_numpy(float), full["m_med"].to_numpy(float), equal_nan=True)


def test_mutaciya_isklyuchennaya_v_poperechnyh_lovitsya(world, env, monkeypatch):
    daily, imd, k1 = world
    ok = L16.panel16(TK, daily, imd, k1)
    monkeypatch.setattr(L16, "cross_without", lambda tk: list(T17))                      # ошибка: i остаётся в поперечных признаках своего прогона
    bad = L16.panel16(TK, daily, imd, k1)
    d = ok[(ok["ticker"] == "A") & ok["m_med"].notna()]["date"].iloc[300]
    med16 = float(np.median(lnrel(daily, [s for s in T17 if s != TK]).loc[d]))
    assert row(ok, "A", d)["m_med"] == pytest.approx(med16, rel=1e-12)                  # инвариант на правке — держится
    assert row(bad, "A", d)["m_med"] != pytest.approx(med16, rel=1e-12)                  # на мутации — нарушен (и для j, и для i)
    assert not np.array_equal(ok["m_med"].to_numpy(float), bad["m_med"].to_numpy(float), equal_nan=True)


def forecast(daily, imd, k1, env_dir):
    out = L16.loto_ticker16("xgb", ["A1"], TK, daily, imd, 2, None, HP, k1)
    assert len(out) > 100 and (out["ticker"] == TK).all() and out["config"].eq("xgb_A1").all() and (out["sigma2"] > 0).all()
    return out.set_index("date")["sigma2"]


def cut_future(daily, date):
    """Данные бумаги TK после `date` испорчены (удаление данных после d эквивалентно подмене: прогноз на ≤ d не должен измениться)."""
    d2 = dict(daily)
    f = daily[TK].copy()
    num = f.columns[f.dtypes.map(lambda t: t.kind == "f")]
    f.loc[f.index > date, num] = f.loc[f.index > date, num].to_numpy() * 3.0
    d2[TK] = f
    return d2


def test_obrezka_dannye_i_posle_d_ne_menyayut_prognoz_i_mutaciya(world, env, monkeypatch):
    daily, imd, k1 = world
    base = forecast(daily, imd, k1, env)
    d = base.index[len(base) // 2]
    k1b = {**k1, TK: HH.series(cut_future(daily, d)[TK])}
    cut = forecast(cut_future(daily, d), imd, k1b, env)
    both = base.index[base.index <= d]
    assert len(both) > 50 and np.array_equal(base.loc[both].to_numpy(), cut.loc[both].to_numpy())   # обрезка: прогноз на ≤ d тот же побитно
    assert not np.array_equal(base.loc[base.index > d].to_numpy(), cut.loc[cut.index > d].to_numpy())   # а после d данные i учтены (тест чувствителен)
    real = L16.panel16

    def peek(tk, dl, im, k=None):                                                        # ошибка: признаки i сдвинуты на день вперёд (будущее)
        dl = dict(dl)
        dl[tk] = dl[tk].shift(-1)
        return real(tk, dl, im, None)
    monkeypatch.setattr(L16, "panel16", peek)
    bad = L16.loto_ticker16("xgb", ["A1"], TK, daily, imd, 2, None, HP, k1)
    badc = L16.loto_ticker16("xgb", ["A1"], TK, cut_future(daily, d), imd, 2, None, HP, k1b)
    bb, bc = bad.set_index("date")["sigma2"], badc.set_index("date")["sigma2"]
    idx = bb.index[bb.index <= d].intersection(bc.index)
    assert len(idx) > 50 and not np.array_equal(bb.loc[idx].to_numpy(), bc.loc[idx].to_numpy())   # мутант заглядывает вперёд — обрезка его ловит


def test_sostav_obucheniya_i_kryuchok_vosstanavlivayutsya(world, env):
    daily, imd, k1 = world
    L16.loto_ticker16("xgb", ["A1"], TK, daily, imd, 2, None, HP, k1)
    assert X.TRAIN == list(T17) and L16.cross_without(TK) == [t for t in T17 if t != TK] and TK not in L16.cross_without(TK)
