"""Л1 (источник, календарь, K1) на РЕАЛЬНОМ архиве + begin новых файлов. Цен новых файлов тест не читает.
Гейты: архивные дни, t⁺, K1 = daily_SBER/har_SBER побитно; календарь нового периода = 180 (11-04 да; 05-01, 06-12 нет);
corrupt_after не меняет строки d <= d0; цены нового файла до замка — исключение (классы № 141/142)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from models_v2.s10 import hp_har as HH
from models_v2.s17 import fwd_cal as C, fwd_k1 as K, fwd_src as S

R = S.ROOT
PAN = R / "data/regen/harj_panel_hybrid"
HAVE = (PAN / "har_SBER.parquet").exists() and S.Paths().new_sber.exists() and (R / "data/fwd/mkt_m10/IMOEX_m10_fwd.parquet").exists()
pytestmark = pytest.mark.skipif(not HAVE, reason="нет архива или новых файлов")
D0 = pd.Timestamp("2025-12-30")
DCOLS = ["n_bars", "rv", "bv", "j", "seg", "rv_w", "rv_m"]


def nolock(tmp_path, **kw) -> S.Src:
    return S.Src(S.Paths(lock=tmp_path / "no_lock.json", **kw))


def same(a, b) -> bool:
    return np.array_equal(np.asarray(a), np.asarray(b), equal_nan=a.dtype.kind == "f")


@pytest.fixture(scope="module")
def src(tmp_path_factory):
    return nolock(tmp_path_factory.mktemp("l1"))


@pytest.fixture(scope="module")
def har():
    return pd.read_parquet(PAN / "har_SBER.parquet")


@pytest.fixture(scope="module")
def daily_arch():
    d = pd.read_parquet(PAN / "daily_SBER.parquet")
    return d if d.index.name == "date" else d.set_index("date")


@pytest.fixture(scope="module")
def k1(src):
    return K.build_k1(S.ARCHIVE_END, src)


def test_arhivnyj_dnevnoj_ryad_pobitno(src, daily_arch):
    d = K.build_daily_sber(S.ARCHIVE_END, src)
    assert d.index.equals(pd.DatetimeIndex(daily_arch.index))
    for c in DCOLS:
        assert same(d[c].to_numpy(), daily_arch[c].to_numpy()), c


def test_arhivnye_dni_i_t_plus_po_begin_ravny_arhivu(src, daily_arch, har):
    cal = C.build_cal(S.ARCHIVE_END, src)
    assert pd.DatetimeIndex(cal["date"]).equals(pd.DatetimeIndex(daily_arch.index))   # дни по begin = дни build_daily
    tp = dict(zip(cal["date"], cal["t_plus"]))
    assert all(tp[d] == t for d, t in zip(har["d"], har["target_date"]))                  # t⁺ = target_date K1 для каждой строки
    assert pd.isna(cal["t_plus"].iloc[-1]) and cal["t_plus"].iloc[-2] == cal["date"].iloc[-1]


def test_k1_arhiv_pobitno_har_sber(k1, har):
    assert list(k1.columns) == list(har.columns) and len(k1) == len(har)
    for c in har.columns:
        a, b = k1[c], har[c]
        assert same(a.to_numpy().astype("datetime64[ns]") if a.dtype.kind == "M" else a.to_numpy(),
                    b.to_numpy().astype("datetime64[ns]") if b.dtype.kind == "M" else b.to_numpy()), c


def test_kalendar_novogo_perioda_180_dnej_po_pravilu_imoex(src):
    cal = C.build_cal(S.LAST_SESSION, src)
    new = cal[cal["date"] > S.ARCHIVE_END]
    days = set(new["date"])
    assert pd.Timestamp("2026-11-04") in days
    assert not {pd.Timestamp("2026-05-01"), pd.Timestamp("2026-06-12")} & days
    pairs = C.new_pairs(cal)
    assert len(new) == 180 == len(pairs)
    assert pairs.iloc[0].tolist() == [S.ARCHIVE_END, new["date"].iloc[0]]
    assert pairs["t_plus"].iloc[-1] == S.LAST_SESSION == new["date"].iloc[-1]
    assert (pairs["t_plus"].shift(1).iloc[1:].to_numpy() == pairs["t_plus"].iloc[:-1].to_numpy()).all()
    assert (pairs["date"].iloc[1:].to_numpy() == pairs["t_plus"].iloc[:-1].to_numpy()).all()   # d следующей пары = t⁺ предыдущей
    assert pairs["t_plus"].dt.dayofweek.max() <= 4                                              # выходные сессии не входят


def test_calendar_parquet_ravno_pereschetu(src):
    f = C.OUT
    assert f.exists()
    pd.testing.assert_frame_equal(pd.read_parquet(f), C.build_cal(S.LAST_SESSION, src))
    assert int(pd.read_parquet(f)["new_pair"].sum()) == 180 and K.OUT.exists()
    pd.testing.assert_frame_equal(pd.read_parquet(K.OUT), K.build_k1(S.ARCHIVE_END, src))


def test_sdvig_dnej_do_d0_calendar_ne_zavisit_ot_novyh_dannyh(src):
    full = C.build_cal(S.LAST_SESSION, src)
    cut = C.build_cal(D0, src)
    a = full[full["date"] <= D0].iloc[:-1].reset_index(drop=True)          # у последней строки t⁺ ещё неизвестен
    b = cut.iloc[:-1].reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b)
    assert len(C.closed_pairs(full, D0)) == len(b) and C.closed_pairs(full, D0)["t_plus"].max() <= D0


def test_corrupt_after_ne_menyaet_stroki_d_do_d0(tmp_path, src):
    sber = pd.read_parquet(src.archive_path("SBER"))
    imx = pd.read_parquet(src.archive_path("IMOEX"))
    (tmp_path / "b").mkdir()
    HH.corrupt_after(sber, D0).to_parquet(tmp_path / "b/SBER.parquet")
    HH.corrupt_after(imx, D0).to_parquet(tmp_path / "IMOEX.parquet")
    bad = nolock(tmp_path, bars_dir=tmp_path / "b", imoex_archive=tmp_path / "IMOEX.parquet")
    clean = K.build_k1(D0, src)
    assert not np.array_equal(sber["close"][sber["begin"] > D0], HH.corrupt_after(sber, D0)["close"][sber["begin"] > D0])
    pd.testing.assert_frame_equal(K.build_k1(D0, bad), clean)                                # бары после d0 не входят
    late = K.build_k1(S.ARCHIVE_END, bad)                                                    # порча видна в строках после d0
    old = late[late["target_date"] <= D0].reset_index(drop=True)
    pd.testing.assert_frame_equal(old, clean)                                                # строки с t⁺ <= d0 не изменились
    assert not late["sigma2"].tail(20).equals(K.build_k1(S.ARCHIVE_END, src)["sigma2"].tail(20))


def test_mutaciya_cut_ne_obrezaet_lovitsya(tmp_path, src, monkeypatch):
    sber = pd.read_parquet(src.archive_path("SBER"))
    (tmp_path / "b").mkdir()
    HH.corrupt_after(sber, D0).to_parquet(tmp_path / "b/SBER.parquet")
    bad = nolock(tmp_path, bars_dir=tmp_path / "b", new_sber=tmp_path / "net.parquet", new_mkt_dir=tmp_path)   # только архив
    clean = K.build_k1(D0, src)["sigma2"]
    monkeypatch.setattr(S, "limit_of", lambda cut: pd.Timestamp.max)          # мутация: обрезка по cut снята
    assert not K.build_k1(D0, bad)["sigma2"].equals(clean)


def test_cena_novogo_fajla_do_zamka_isklyuchenie_a_begin_chitaetsya(src, monkeypatch):
    seen = []
    orig = pd.read_parquet
    monkeypatch.setattr(pd, "read_parquet", lambda p, *a, **kw: (seen.append((str(p), kw.get("columns"))), orig(p, *a, **kw))[1])
    for fn in (lambda: src.bars("SBER", S.LAST_SESSION), lambda: K.build_k1(pd.Timestamp("2026-03-10"), src),
               lambda: src.bars("IMOEX", S.LAST_SESSION), src.sealed_rv):
        with pytest.raises(S.SealedError):
            fn()
    seen.clear()
    assert len(src.begins("SBER", S.LAST_SESSION)) > 200000 and src.index_days(S.LAST_SESSION)
    newp = [c for p, c in seen if "/data/fwd/" in p]
    assert newp and all(c == ["begin"] for c in newp)                           # новые файлы — только begin
    assert not any(p.endswith("rv_fwd_SEALED.parquet") for p, _ in seen)
