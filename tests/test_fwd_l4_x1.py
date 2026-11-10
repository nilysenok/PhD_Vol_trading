"""Л4: X1 (F = M^(1−π)·C^π) на рядах cfg8, K2, K3 и e_d Л3 — реальный архив. Цен новых файлов тест не читает (только begin; cut > 2026-02-27 даёт SealedError).
Гейты: архивные строки = `E1/x1.parquet` побитно; на обрезках (первая и последняя из 38) строка d и префикс = архив; состояние π₀, π₁, η, α₀, α₁ на последней строке = архив/пересчёт;
α меняется только в первый день месяца; будущее после d0 не меняет префикс; мутации «закрыто + 1» (`closed_date` = дата прогноза) и «α по всей выборке» ловятся."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from models_v2.s17 import fwd_cfg8 as F8, fwd_src as S, fwd_store as FS, fwd_x1 as X1  # noqa: I001  (fwd_cfg8 первым: torch до lightgbm)
from models_v2.s11.v4_ntp_state import qlike
from models_v2.s15 import de_e1_state as E1

R = S.ROOT
ARCH = R / "data/regen/harj_deep/E1/x1.parquet"
HAVE = ARCH.exists() and F8.OUT.exists() and X1.OUT.exists() and F8.K23.exists() and S.Paths().new_sber.exists()
pytestmark = pytest.mark.skipif(not HAVE, reason="нет архива E1, выходов Л4 (cfg8.parquet, x1.parquet), k23.parquet Л2 или begin новых файлов")
CUTS = ["2025-12-30", "2026-02-26"]
_C: dict = {}


@pytest.fixture(scope="module")
def src(tmp_path_factory):
    return S.Src(S.Paths(lock=tmp_path_factory.mktemp("l4x") / "no_lock.json"))


@pytest.fixture(scope="module")
def arch(src):
    store = FS.build_store(S.ARCHIVE_END, src)
    cfg8, tab, st = pd.read_parquet(F8.OUT), F8.tab_at(S.ARCHIVE_END, src), FS.build_states(S.ARCHIVE_END, src, store)
    x1, r = X1.run_x1(cfg8, tab, st)
    return dict(cfg8=cfg8, tab=tab, st=st, x1=x1, r=r)


def cut_x1(cut: str, src, a: dict) -> pd.DataFrame:
    """X1 на обрезке: K1 и e_d по барам <= cut, K2/K3 — сохранённые Л2, C — прогнозы cfg8 строк < d (архив) и строки d без цели."""
    if cut not in _C:
        c = pd.Timestamp(cut)
        st = FS.build_states(c, src)
        k1, daily = F8.k1_at(c, src)
        tab = F8.make_tab(k1, F8.mask_k23(pd.read_parquet(F8.K23), c), daily)
        cfg = a["cfg8"][a["cfg8"]["d"] <= c].copy()
        cfg.loc[cfg["d"] == c, "rv"] = np.nan
        assert np.array_equal(cfg["t_plus"].to_numpy("datetime64[ns]")[-1:], tab["t_plus"].to_numpy("datetime64[ns]")[-1:])
        _C[cut] = X1.build_x1(c, src, cfg, tab, st)
    return _C[cut]


def test_polnyj_hod_arhivnye_stroki_pobitno(arch):
    pd.testing.assert_frame_equal(arch["x1"], pd.read_parquet(ARCH), check_exact=True)
    pd.testing.assert_frame_equal(arch["x1"], pd.read_parquet(X1.OUT), check_exact=True)
    assert len(arch["x1"]) == 2018 and (arch["x1"]["F"] > 0).all()


def test_sostoyanie_pi_eta_alpha_ravno_arhivu(arch):
    x, st, ref = arch["x1"], X1.state_of(arch["r"]), pd.read_parquet(ARCH).iloc[-1]
    assert (st["pi0"], st["pi1"], st["a0"], st["a1"]) == (ref["pi0"], ref["pi1"], ref["a0"], ref["a1"]) and x["d"].iloc[-1] == pd.Timestamp("2026-02-26")
    closed = (x["t_plus"] <= x["d"].iloc[-1]).to_numpy()                                       # пары, закрытые к последнему d
    q = np.concatenate([qlike(x["rv"].to_numpy()[closed], x["C"].to_numpy()[closed]), qlike(x["rv"].to_numpy()[closed], x["M"].to_numpy()[closed])])
    assert st["eta"] == 1.0 / max(float(np.median(q)), E1.EPS)                                  # η = 1/медиана QLIKE обеих моделей по закрытым парам
    assert pd.Timestamp("2026-02-02") <= x["d"].iloc[-1] and st["a0"] in E1.GRID and st["a1"] in E1.GRID


def test_alpha_menyaetsya_tolko_v_pervyj_den_mesyaca_pi_ezhednevno(arch):
    x = arch["x1"]
    month_first = x.index[~x["d"].dt.to_period("M").duplicated()]
    for c in ("a0", "a1"):
        ch = x.index[1:][np.diff(x[c].to_numpy()) != 0]
        assert len(ch) >= 2 and set(ch) <= set(month_first)
    assert (np.diff(x["pi0"].to_numpy()[-60:]) != 0).mean() > 0.2                               # π0 обновляется почти каждый день, а не раз в месяц


@pytest.mark.parametrize("cut", CUTS)
def test_obrezka_stroka_d_i_prefiks_ravny_arhivu_pobitno(cut, src, arch):
    got, ref = cut_x1(cut, src, arch), pd.read_parquet(ARCH)
    assert got["d"].iloc[-1] == pd.Timestamp(cut) and np.isnan(got["rv"].iloc[-1])
    ref = ref[ref["d"] <= pd.Timestamp(cut)].reset_index(drop=True)
    pd.testing.assert_frame_equal(got.drop(columns="rv"), ref.drop(columns="rv"), check_exact=True)


def test_buduschee_posle_d0_ne_menyaet_prefiks(arch):
    cfg, tab, st = arch["cfg8"], arch["tab"], arch["st"]
    n0 = int((cfg["d"] <= pd.Timestamp("2026-01-15")).sum())
    rng = np.random.default_rng(5)
    bad, bt = cfg.copy(), tab.copy()
    for c in ("rv", "F"):
        bad.loc[n0:, c] = rng.uniform(0.05, 5.0, len(bad) - n0)
    bt[["K2", "K3"]] = np.where((bt.index > bad["d"].iloc[n0 - 1])[:, None], rng.uniform(0.05, 5.0, (len(bt), 2)), bt[["K2", "K3"]].to_numpy())
    st2 = st.copy()
    st2.loc[st2["d"] > bad["d"].iloc[n0 - 1], "e"] = 1 - st2["e"]
    got, _ = X1.run_x1(bad, bt, st2)
    pd.testing.assert_frame_equal(got.iloc[:n0].drop(columns="rv"), arch["x1"].iloc[:n0].drop(columns="rv"), check_exact=True)
    assert not np.array_equal(got["F"].to_numpy()[n0:], arch["x1"]["F"].to_numpy()[n0:])


def test_mutaciya_zakryto_plus_odin_lovitsya(arch, monkeypatch):
    monkeypatch.setattr(E1, "closed_date", lambda t_plus, date: date)                         # пара видна в день прогноза (t⁺ + 1 торговый день к ней)
    mut, _ = X1.run_x1(arch["cfg8"], arch["tab"], arch["st"])
    ref = pd.read_parquet(ARCH)
    assert not np.array_equal(mut["F"].to_numpy(), ref["F"].to_numpy()) and not np.array_equal(mut["pi0"].to_numpy(), ref["pi0"].to_numpy())


def test_mutaciya_alpha_po_vsej_vyborke_lovitsya(arch, monkeypatch):
    monkeypatch.setattr(E1, "cutoff", lambda dates, i: dates[-1])
    mut, _ = X1.run_x1(arch["cfg8"], arch["tab"], arch["st"])
    ref = pd.read_parquet(ARCH)
    assert not (np.array_equal(mut["a0"].to_numpy(), ref["a0"].to_numpy()) and np.array_equal(mut["a1"].to_numpy(), ref["a1"].to_numpy()))


def test_novye_dni_trebuyut_zamok(src, arch):
    with pytest.raises(S.SealedError):
        X1.build_x1(pd.Timestamp("2026-03-02"), src, arch["cfg8"], arch["tab"])
