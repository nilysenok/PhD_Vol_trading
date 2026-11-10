"""Л4: cfg8 на продлённой таблице с z из Л3 — реальный архив. Цены новых файлов тест не читает (только begin через Л1; cut > 2026-02-27 даёт SealedError).
Гейты: таблица = `load_tab`; прогнозы d = cut на обрезках (первая и последняя из 38) = архив побитно; даты переоценок нового периода = правило (календарь по begin);
порча баров SBER и IMOEX после d0 не меняет прогнозы блока ≤ d0; мутация «закрыто + 1» (подмена `closed_rows`) ловится. Полный ход и λ — `test_fwd_l4_cfg8_full.py`."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from models_v2.s17 import fwd_cfg8 as F8, fwd_cal as FC, fwd_src as S, fwd_store as FS  # noqa: I001  (fwd_cfg8 первым: torch до lightgbm)
from models_v2.s10 import hp_har as HH
from models_v2.s14 import v7_1m_data as DATA, v7_1m_wf as WF

R = S.ROOT
ARCH = R / "data/regen/harj_hybrid_v7/3/cfg8.parquet"
HAVE = ARCH.exists() and F8.K23.exists() and S.Paths().new_sber.exists() and (R / "data/regen/harj_hybrid_v7/stageA/tcn_w5_2026_s4.npz").exists()
pytestmark = pytest.mark.skipif(not HAVE, reason="нет архива V7, k23.parquet Л2 или begin новых файлов")
CUTS = ["2025-12-30", "2026-02-26"]
D0 = pd.Timestamp("2025-12-30")
_CACHE: dict = {}


@pytest.fixture(scope="module")
def src(tmp_path_factory):
    return S.Src(S.Paths(lock=tmp_path_factory.mktemp("l4c") / "no_lock.json"))


def cut_run(cut: str, src):
    if cut not in _CACHE:
        store = FS.build_store(pd.Timestamp(cut), src)
        inp = F8.inputs_at(pd.Timestamp(cut), src, F8.mask_k23(pd.read_parquet(F8.K23), cut), store, FS.build_xw5(store))
        _CACHE[cut] = (inp, F8.build_cfg8(pd.Timestamp(cut), src, inp))
    return _CACHE[cut]


def test_tablica_arhiv_ravna_load_tab(src):
    tab, ref = F8.tab_at(S.ARCHIVE_END, src), DATA.load_tab()
    pd.testing.assert_frame_equal(tab.iloc[: len(ref)], ref, check_exact=True)
    assert len(tab) == len(ref) == 2271


def test_daty_pereotsenok_novogo_perioda_po_kalendaryu_begin(src):
    cal = FC.build_cal(S.LAST_SESSION, src)
    days = pd.DatetimeIndex(cal["date"][cal["date"] > S.ARCHIVE_END])
    first = [days[days.to_period("M") == m][0].strftime("%m-%d") for m in days.to_period("M").unique()]
    assert first == ["03-02", "04-01", "05-04", "06-01", "07-01", "08-03", "09-01", "10-01", "11-02"]


@pytest.mark.parametrize("cut", CUTS)
def test_obrezka_prognoz_dnya_d_ravny_arhivu_pobitno(cut, src):
    inp, df = cut_run(cut, src)
    tab, ref = inp["tab"], pd.read_parquet(ARCH).set_index("d")
    assert tab.index[-1] == pd.Timestamp(cut) and np.isnan(tab["rv_next"].iloc[-1]) and (tab["t_plus"].iloc[:-1] <= pd.Timestamp(cut)).all()
    pd.testing.assert_frame_equal(tab.iloc[:-1], F8.tab_at(S.ARCHIVE_END, src).iloc[: len(tab) - 1], check_exact=True)
    assert df["d"].iloc[-1] == pd.Timestamp(cut) and np.isnan(df["rv"].iloc[-1]) and df["d"].iloc[0].month == pd.Timestamp(cut).month
    # ≤ 1e-12 отн.: ансамбль torch считает блок пакетом строк — длина блока (обрезок против полного хода) может менять последний разряд (по 38 обрезкам: F ≤ 1,1e-16, π, g ≤ 8e-14; L4_CUTS38.csv)
    pd.testing.assert_frame_equal(df.drop(columns="rv").set_index("d"), ref.loc[df["d"]].drop(columns="rv"), check_exact=False, rtol=1e-12, atol=0)


def test_obrezka_bary_posle_d0_ne_menyayut_blok_do_d0(tmp_path, src):
    (tmp_path / "b").mkdir()
    HH.corrupt_after(pd.read_parquet(src.archive_path("SBER")), D0).to_parquet(tmp_path / "b/SBER.parquet")
    HH.corrupt_after(pd.read_parquet(src.archive_path("IMOEX")), D0).to_parquet(tmp_path / "IMOEX.parquet")
    bad = S.Src(S.Paths(lock=tmp_path / "no_lock.json", bars_dir=tmp_path / "b", imoex_archive=tmp_path / "IMOEX.parquet"))
    k1, daily = F8.k1_at(S.ARCHIVE_END, bad)
    k23 = pd.read_parquet(F8.K23)
    late, rng = (k23["date"] > D0).to_numpy(), np.random.default_rng(3)
    k23.loc[late, ["sigma2_k2", "sigma2_k3"]] *= np.exp(rng.normal(0, 0.5, (int(late.sum()), 2)))
    k23["rv"] = k1.set_index("d")["rv"].reindex(k23["date"]).to_numpy()
    tab = F8.make_tab(k1, k23, daily)
    assert not np.array_equal(tab["H"].to_numpy()[-5:], F8.tab_at(S.ARCHIVE_END, src)["H"].to_numpy()[-5:])        # порча действует
    df = F8.build_cfg8(S.ARCHIVE_END, bad, dict(tab=tab, zmap=F8.z_maps(tab.index, S.ARCHIVE_END, bad)), at=int(tab.index.searchsorted(pd.Timestamp("2025-12-01"))))
    keep, ref = df[df["d"] <= D0].set_index("d"), pd.read_parquet(ARCH).set_index("d")
    assert len(keep) > 15 and keep.index[0] == pd.Timestamp("2025-12-01")
    pd.testing.assert_frame_equal(keep.drop(columns="rv"), ref.loc[keep.index].drop(columns="rv"), check_exact=True)   # rv строки d0 — цель t⁺ > d0 (испорчена по замыслу)


def _plus_one(tp, d0):
    """МУТАНТ «закрыто + 1»: пара строки d0 (t⁺ = следующий день) считается закрытой уже на d0."""
    tp = np.asarray(tp, "datetime64[ns]")
    return np.flatnonzero(tp <= tp[tp > np.datetime64(d0, "ns")].min())


def test_mutaciya_zakryto_plus_odin_lovitsya(src, monkeypatch):
    inp, honest = cut_run(CUTS[1], src)
    ref = pd.read_parquet(ARCH).set_index("d").loc[honest["d"]]
    close = lambda a, b: np.allclose(a.to_numpy(), b.to_numpy(), rtol=1e-12, atol=0)
    assert close(honest["G"], ref["G"]) and close(honest["F"], ref["F"])                    # честный путь = архив
    monkeypatch.setattr(WF, "closed_rows", _plus_one)
    mut = F8.build_cfg8(pd.Timestamp(CUTS[1]), src, inp)
    assert not close(mut["G"], ref["G"]) and not close(mut["F"], ref["F"])


def test_novye_dni_trebuyut_zamok(src):
    with pytest.raises(S.SealedError):
        F8.inputs_at(pd.Timestamp("2026-03-02"), src)
