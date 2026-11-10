"""Л4: ZB и ZB0 (X1·exp(e_d·(b + a·s_d)) по экстремальным парам, помесячно) на ряду X1 и e_d, s_d Л3 — реальный архив. Цен новых файлов тест не читает (SealedError при cut > 2026-02-27).
Гейты: архивные строки ZB и ZB0 = `Z/zb.parquet`, `zb0.parquet` побитно; на обрезках (первая и последняя из 38) строка d и префикс = архив; состояние λ, b, a = архив;
параметры меняются только в первый день месяца; будущее после d0 не меняет префикс; мутации «закрыто + 1» (`closed_date` = дата прогноза) и «пул по всем парам» ловятся."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from models_v2.s17 import fwd_src as S, fwd_store as FS, fwd_x1 as X1, fwd_zb as ZB  # noqa: I001  (fwd_zb импортирует torch первым через models_v2.s16)
from models_v2.s16 import x1_z_fit as FT, x1_z_walk as W

R = S.ROOT
Z = R / "data/regen/harj_x1/Z"
HAVE = (Z / "zb.parquet").exists() and (Z / "zb0.parquet").exists() and X1.OUT.exists() and (ZB.OUT_DIR / "zb.parquet").exists() and S.Paths().new_sber.exists()
pytestmark = pytest.mark.skipif(not HAVE, reason="нет архива Z, выходов Л4 (x1.parquet, zb.parquet) или begin новых файлов")
CUTS = ["2025-12-30", "2026-02-26"]
_C: dict = {}


@pytest.fixture(scope="module")
def src(tmp_path_factory):
    return S.Src(S.Paths(lock=tmp_path_factory.mktemp("l4z") / "no_lock.json"))


@pytest.fixture(scope="module")
def arch(src):
    store = FS.build_store(S.ARCHIVE_END, src)
    st, x1 = FS.build_states(S.ARCHIVE_END, src, store), pd.read_parquet(X1.OUT)
    return dict(store=store, st=st, x1=x1, zb=ZB.build_zb(S.ARCHIVE_END, src, x1, st, store, "ZB"), zb0=ZB.build_zb(S.ARCHIVE_END, src, x1, st, store, "ZB0"))


def cut_zb(cut: str, src, a: dict) -> pd.DataFrame:
    """ZB на обрезке: склад и e_d, s_d по барам <= cut; ряд X1 — архивные строки <= d, у строки d цели нет."""
    if cut not in _C:
        c = pd.Timestamp(cut)
        store = FS.build_store(c, src)
        x1 = a["x1"][a["x1"]["d"] <= c].copy()
        x1.loc[x1["d"] == c, "rv"] = np.nan
        _C[cut] = ZB.build_zb(c, src, x1, FS.build_states(c, src, store), store, "ZB")
    return _C[cut]


@pytest.mark.parametrize("kind,name", [("zb", "zb.parquet"), ("zb0", "zb0.parquet")])
def test_polnyj_hod_arhivnye_stroki_pobitno(arch, kind, name):
    pd.testing.assert_frame_equal(arch[kind], pd.read_parquet(Z / name), check_exact=True)
    pd.testing.assert_frame_equal(arch[kind], pd.read_parquet(ZB.OUT_DIR / name), check_exact=True)
    assert len(arch[kind]) == 2018 and (arch[kind]["F"] > 0).all()


def test_sostoyanie_lambda_b_a_ravno_arhivu_i_zb0_bez_a(arch):
    s, ref = ZB.state_of(arch["zb"]), pd.read_parquet(Z / "zb.parquet").iloc[-1]
    assert (s["lam"], s["b"], s["a"]) == (ref["lam"], ref["b"], ref["a"]) and s["active"] == 1 and s["n_ext"] >= FT.MIN_EXT and s["lam"] in FT.LAMS
    z0, act = arch["zb0"], arch["zb0"]["active"] == 1
    assert (z0.loc[act, "a"] == 0).all() and (arch["zb"].loc[arch["zb"]["active"] == 1, "a"] != 0).any()      # ZB0: только b


def test_parametry_menyayutsya_tolko_v_pervyj_den_mesyaca(arch):
    zb = arch["zb"]
    month_first = set(zb.index[~zb["d"].dt.to_period("M").duplicated()])
    for c, k in (("lam", 5), ("b", 20), ("a", 20)):
        v = zb[c].to_numpy()
        ch = zb.index[1:][~((v[1:] == v[:-1]) | (np.isnan(v[1:]) & np.isnan(v[:-1])))]
        assert len(ch) > k and set(ch) <= month_first, c


@pytest.mark.parametrize("cut", CUTS)
def test_obrezka_stroka_d_i_prefiks_ravny_arhivu_pobitno(cut, src, arch):
    got = cut_zb(cut, src, arch)
    ref = pd.read_parquet(Z / "zb.parquet")
    ref = ref[ref["d"] <= pd.Timestamp(cut)].reset_index(drop=True)
    assert got["d"].iloc[-1] == pd.Timestamp(cut) and np.isnan(got["rv"].iloc[-1]) and got["active"].iloc[-1] == 1
    pd.testing.assert_frame_equal(got.drop(columns="rv"), ref.drop(columns="rv"), check_exact=True)


def test_buduschee_posle_d0_ne_menyaet_prefiks(src, arch):
    x1, st = arch["x1"], arch["st"]
    n0 = int((x1["d"] <= pd.Timestamp("2026-01-15")).sum())
    bad, rng = x1.copy(), np.random.default_rng(7)
    bad.loc[n0:, ["rv", "F"]] = rng.uniform(0.05, 5.0, (len(bad) - n0, 2))
    st2 = st.copy()
    late = st2["d"] > bad["d"].iloc[n0 - 1]
    st2.loc[late, "e"], st2.loc[late, "s"] = 1 - st2.loc[late, "e"], rng.uniform(-3, 3, int(late.sum()))
    got = ZB.build_zb(S.ARCHIVE_END, src, bad, st2, arch["store"], "ZB")
    pd.testing.assert_frame_equal(got.iloc[:n0].drop(columns="rv"), arch["zb"].iloc[:n0].drop(columns="rv"), check_exact=True)
    assert not np.array_equal(got["F"].to_numpy()[n0:], arch["zb"]["F"].to_numpy()[n0:])


def test_mutaciya_zakryto_plus_odin_lovitsya(src, arch, monkeypatch):
    monkeypatch.setattr(W, "closed_date", lambda tp, d: d)                                    # пара видна в день прогноза d, а не с t⁺
    mut = ZB.build_zb(S.ARCHIVE_END, src, arch["x1"], arch["st"], arch["store"], "ZB")
    ref = pd.read_parquet(Z / "zb.parquet")
    assert not np.array_equal(mut["F"].to_numpy(), ref["F"].to_numpy()) and not np.array_equal(mut["b"].to_numpy(), ref["b"].to_numpy(), equal_nan=True)


def test_mutaciya_pul_po_vsem_param_lovitsya(src, arch, monkeypatch):
    monkeypatch.setattr(FT, "ext_pool", lambda pool, e: np.asarray(pool))                    # не только e = 1: все закрытые пары
    mut = ZB.build_zb(S.ARCHIVE_END, src, arch["x1"], arch["st"], arch["store"], "ZB")
    assert not np.array_equal(mut["F"].to_numpy(), pd.read_parquet(Z / "zb.parquet")["F"].to_numpy())


def test_tolko_zb_i_zb0_i_zamok(src, arch):
    with pytest.raises(ValueError):
        ZB.build_zb(S.ARCHIVE_END, src, arch["x1"], arch["st"], arch["store"], "ZC")
    with pytest.raises(S.SealedError):
        ZB.build_zb(pd.Timestamp("2026-03-02"), src, arch["x1"])
