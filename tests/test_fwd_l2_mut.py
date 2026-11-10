"""Л2: обрезка по моменту выпуска, бары вне основной сессии IMOEX (решение 7), правило IMOEX склейки (решение 8) и мутации.
Реальный архив; цен новых файлов тест не читает. Мутации: «бар d+1 в рыночном блоке», «вечер d в рыночном блоке», «бары вне основной сессии IMOEX входят»,
«правило IMOEX не подставлено», «index_rule не восстанавливает»."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from models_v2 import daily as D
from models_v2.s4 import mkt
from models_v2.s10 import hp_k23_world as KW
from models_v2.s17 import fwd_cal as C, fwd_k23 as K, fwd_src as S

R = S.ROOT
HAVE = (R / "data/regen/harj_panel_hybrid/k23_SBER.parquet").exists() and S.Paths().new_sber.exists() and (R / "data/fwd/mkt_m10/IMOEX_m10_fwd.parquet").exists()
pytestmark = pytest.mark.skipif(not HAVE, reason="нет архива или новых файлов")
D0 = pd.Timestamp("2026-02-10")      # день с данными и в следующую сессию (31.12.2025 данных нет)
LIM0 = D0 + pd.Timedelta(hours=18, minutes=40)
NAMES = KW.POOL + ["IMOEX"]


def same(a, b) -> bool:
    return np.array_equal(np.asarray(a), np.asarray(b), equal_nan=True)


def world_same(a: dict, b: dict) -> bool:
    return (same(a["F"].dates, b["F"].dates) and same(a["F"].X, b["F"].X) and same(a["dz0"].X, b["dz0"].X) and same(a["dz0"].y, b["dz0"].y)
            and same(a["feats"].to_numpy(), b["feats"].to_numpy()) and same(a["eps"], b["eps"]))


def src_of(tmp, **kw) -> S.Src:
    return S.Src(S.Paths(lock=tmp / "no_lock.json", **kw))


@pytest.fixture(scope="module")
def src(tmp_path_factory):
    return src_of(tmp_path_factory.mktemp("l2m"))


@pytest.fixture(scope="module")
def w_d0(src):
    return K.build_world_at(D0, src)


def corrupt_late(bars: pd.DataFrame, lim: pd.Timestamp, seed: int = 7) -> pd.DataFrame:
    """Бары строго после момента выпуска (вечер d тоже) умножены на случайный множитель."""
    late = (bars["begin"] > lim).to_numpy()
    f = np.exp(np.random.default_rng(seed).normal(0.0, 0.5, int(late.sum())))
    out = bars.copy()
    for c in ("open", "high", "low", "close"):
        out.loc[late, c] = out.loc[late, c].to_numpy() * f
    out.loc[late, "volume"] = (out.loc[late, "volume"].to_numpy() * f).astype(out["volume"].dtype)
    return out


@pytest.fixture(scope="module")
def bad(src, tmp_path_factory):
    tmp = tmp_path_factory.mktemp("l2bad")
    (tmp / "b").mkdir()
    for t in NAMES:
        corrupt_late(pd.read_parquet(src.archive_path(t)), LIM0).to_parquet(tmp / ("IMOEX.parquet" if t == "IMOEX" else f"b/{t}.parquet"))
    return src_of(tmp, bars_dir=tmp / "b", imoex_archive=tmp / "IMOEX.parquet")


def test_corrupt_after_18_instrumentov_ne_menyaet_mir_dnya_d(bad, w_d0):
    assert not same(pd.read_parquet(bad.archive_path("IMOEX"))["close"], pd.read_parquet(S.Paths().imoex_archive)["close"])    # порча реальна
    assert world_same(K.build_world_at(D0, bad), w_d0)


def test_dostupnost_k_vypusku_bar_instrumenta_ne_pozzhe_d_18_40(src):
    av = K.availability(D0, src)
    assert set(av) == set(NAMES) and len(av) == 18
    assert all(pd.notna(v) and v <= LIM0 and v.normalize() == D0 for v in av.values())             # у всех 18 есть бар основной сессии дня d, не позже 18:40


@pytest.mark.parametrize("nm,lim", [("bar_d_plus_1", lambda cut: pd.Timestamp(cut).normalize() + pd.Timedelta(days=1, hours=18, minutes=40)),
                                    ("vecher_d", lambda cut: pd.Timestamp(cut).normalize() + pd.Timedelta(hours=23, minutes=59))])
def test_mutaciya_bar_d_plus_1_ili_vecher_d_lovitsya(nm, lim, src, bad, w_d0, monkeypatch):
    monkeypatch.setattr(S, "limit_of", lim)
    if nm == "bar_d_plus_1":
        assert any(v > LIM0 for v in K.availability(D0, src).values())                              # проверка доступности ловит
        assert not world_same(K.build_world_at(D0, bad), w_d0)                                      # рыночный блок видит порчу
    else:
        monkeypatch.setattr(D, "T_TO", "23:50")                                                     # вечерние бары допущены в «основную сессию»
        clean = K.build_world_at(D0, src)
        assert not world_same(K.build_world_at(D0, bad), clean)                                     # порченый вечер d меняет признаки дня d


def off_session(imx: pd.DataFrame, extra_days, seed: int = 3) -> pd.DataFrame:
    """К барам IMOEX добавлены бары 07:00–09:40 и 19:00–23:50 каждого дня (и пустых дней extra_days) со случайными ценами."""
    rng = np.random.default_rng(seed)
    mins = list(range(7 * 60, 9 * 60 + 50, 10)) + list(range(19 * 60, 23 * 60 + 51, 10))
    days = list(pd.Index(imx["begin"].dt.normalize().unique())) + list(extra_days)
    parts = []
    for d in days:
        b = pd.DataFrame({c: np.repeat(imx[c].iloc[[0]].to_numpy(), len(mins)) for c in imx.columns})
        px = 2500.0 * np.exp(rng.normal(0.0, 0.7, len(mins)))
        b["begin"] = pd.to_datetime([d + pd.Timedelta(minutes=m) for m in mins]).astype(imx["begin"].dtype)
        b["open"], b["close"], b["high"], b["low"] = px, px * 1.01, px * 1.08, px * 0.93
        b["volume"] = np.abs(rng.normal(1e6, 5e5, len(mins))).astype(imx["volume"].dtype)
        parts.append(b)
    out = pd.concat([imx] + parts, ignore_index=True).drop_duplicates("begin", keep="first")        # дубли begin с реальными барами не нужны
    return out.sort_values("begin").reset_index(drop=True)


@pytest.fixture(scope="module")
def aug(src, tmp_path_factory):
    tmp = tmp_path_factory.mktemp("l2aug")
    imx = pd.read_parquet(src.archive_path("IMOEX"))
    off_session(imx, [pd.Timestamp("2025-05-01"), pd.Timestamp("2025-05-03")]).to_parquet(tmp / "IMOEX.parquet")
    return src_of(tmp, imoex_archive=tmp / "IMOEX.parquet")


def test_idx_i_kalendar_ne_menyayutsya_ot_barov_vne_osnovnoj_sessii_imoex(src, aug, w_d0):
    assert len(aug.begins("IMOEX", D0)) > len(src.begins("IMOEX", D0)) + 40000                      # бары вне 10:00–18:40 реально добавлены
    assert aug.index_days(D0) == src.index_days(D0)
    pd.testing.assert_frame_equal(C.build_cal(D0, aug), C.build_cal(D0, src))
    w = K.build_world_at(D0, aug)
    assert world_same(w, w_d0) and np.isfinite(w["dz0"].X[-300:, -6:]).all()                        # IDX (6 столбцов) и F4 побитно прежние


def test_mutaciya_bary_vne_sessii_vhodyat_lovitsya(src, aug, monkeypatch):
    monkeypatch.setattr(D, "T_FROM", "07:00")
    monkeypatch.setattr(D, "T_TO", "23:50")                                                         # «основная сессия» = все бары
    a, b = K.build_world_at(D0, aug), K.build_world_at(D0, src)
    assert not same(a["dz0"].X[:, -6:-3], b["dz0"].X[:, -6:-3])                                      # imoex_d/w/m при том же коде и разных барах вне сессии разные


def bars_3days() -> pd.DataFrame:
    rng = np.random.default_rng(1)
    b = [pd.Timestamp(d) + pd.Timedelta(minutes=600 + 10 * i) for d in ("2025-03-04", "2025-03-05", "2025-03-06") for i in range(40)]
    c = 300.0 * np.exp(np.cumsum(rng.normal(0, 0.001, len(b))))
    return pd.DataFrame({"begin": b, "open": c, "high": c * 1.001, "low": c * 0.999, "close": c, "volume": np.full(len(b), 1000)})


def test_index_rule_podstavlyaet_dni_imoex_i_vosstanavlivaet():
    bars, before = bars_3days(), D._IDAYS
    with K.index_rule({pd.Timestamp("2025-03-04"), pd.Timestamp("2025-03-06")}):
        assert list(mkt.paper(bars)["day"].index) == [pd.Timestamp("2025-03-04"), pd.Timestamp("2025-03-06")]    # 05.03 без сессии IMOEX исключён замороженным кодом
    assert D._IDAYS is before
    with pytest.raises(RuntimeError), K.index_rule({pd.Timestamp("2025-03-04")}):
        raise RuntimeError("сбой внутри")
    assert D._IDAYS is before                                                                        # восстановлено и при исключении


def test_mutaciya_index_rule_bez_vosstanovleniya_lovitsya():
    from contextlib import contextmanager

    @contextmanager
    def bad_rule(idays):
        D._IDAYS = set(idays)
        yield
    before = D._IDAYS
    try:
        with bad_rule({pd.Timestamp("2025-03-04")}):
            pass
        assert D._IDAYS is not before                                                                # мутант оставляет подмену — проверка выше упала бы
    finally:
        D._IDAYS = before
