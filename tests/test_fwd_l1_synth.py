"""Л1 на СИНТЕТИКЕ (известный ответ, без реальных новых данных): календарь по правилу IMOEX, замок, строки нового периода после замка,
t⁺ = следующий канонический день, закрытые пары по t⁺ (№ 142), обрезка. Мутации: «день без IMOEX оставлен», «бар вне основной сессии IMOEX
делает день рабочим», «замок не проверяется», «пары по d <= cut»."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from models_v2 import daily as D
from models_v2.s10 import hp_har as HH
from models_v2.s17 import fwd_cal as C, fwd_k1 as K, fwd_src as S

MAIN = pd.timedelta_range("10:00:00", "18:40:00", freq="10min")
OFF = list(pd.timedelta_range("19:00:00", "23:50:00", freq="10min")) + [pd.Timedelta("09:50:00")]
END_A = pd.Timestamp("2026-02-27")
HOL_A, HOL_N, SAT, SHORT = (pd.Timestamp(x) for x in ("2025-05-09", "2026-04-10", "2025-11-01", "2025-07-15"))
NEW_LAST = pd.Timestamp("2026-04-30")


def bars_of(days, seed, slots=lambda d: list(MAIN) + OFF, px0=100.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ts = [d + s for d in days for s in slots(d)]
    c = px0 * np.exp(np.cumsum(rng.normal(0, 0.002, len(ts))))
    o = np.r_[px0, c[:-1]]
    return pd.DataFrame({"begin": pd.DatetimeIndex(ts).astype("datetime64[ms]"), "open": o, "high": np.maximum(o, c) * 1.001,
                         "low": np.minimum(o, c) * 0.999, "close": c, "volume": rng.integers(1, 100, len(ts)).astype("int64"), "value": c * 10})


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    t = tmp_path_factory.mktemp("w")
    wd = pd.bdate_range("2024-06-03", NEW_LAST)
    sber_days = [d for d in wd if d not in ()] + [SAT]
    imx_days = [d for d in wd if d not in (HOL_A, HOL_N)] + [SAT]
    off_only = {HOL_N: OFF}                                                    # у IMOEX в праздник есть только бары вне основной сессии
    imx_slots = lambda d: off_only.get(d, list(MAIN) + OFF)
    s = bars_of(sorted(sber_days), 1, lambda d: list(MAIN)[:20] if d == SHORT else list(MAIN) + OFF)
    i = bars_of(sorted(imx_days + [HOL_N]), 2, imx_slots)
    paths = {}
    for nm, df in (("s", s), ("i", i)):
        for part, m in (("a", df["begin"].dt.normalize() <= END_A), ("n", df["begin"].dt.normalize() > END_A)):
            df[m].reset_index(drop=True).to_parquet(t / f"{nm}_{part}.parquet")
    (t / "mkt").mkdir()
    (t / "bars").mkdir()
    return t, s, i


def mk(t, lock=False) -> S.Src:
    (t / "bars" / "SBER.parquet").write_bytes((t / "s_a.parquet").read_bytes())
    (t / "mkt" / "IMOEX_m10_fwd.parquet").write_bytes((t / "i_n.parquet").read_bytes())
    lk = t / "lock.json"
    lk.unlink(missing_ok=True)
    if lock:
        lk.write_text("{}")
    return S.Src(S.Paths(bars_dir=t / "bars", imoex_archive=t / "i_a.parquet", new_sber=t / "s_n.parquet", new_mkt_dir=t / "mkt", lock=lk))


def days(cal, a=None) -> set:
    return set(cal["date"] if a is None else cal[cal["date"] > a]["date"])


def test_kalendar_po_begin_ravno_build_daily_i_pravilo_imoex(world):
    t, s, i = world
    cal = C.build_cal(NEW_LAST, mk(t))
    idays = set(D.main_session(i)["begin"].dt.normalize())
    full = D.build_daily(s, index_days=idays)[0]
    assert pd.DatetimeIndex(cal["date"]).equals(full.index)                      # begin-календарь = build_daily на этом мире
    ds = days(cal)
    assert HOL_A not in ds and HOL_N not in ds and SAT not in ds and SHORT not in ds   # без IMOEX / выходной / < 30 баров
    assert pd.Timestamp("2026-04-09") in ds and pd.Timestamp("2026-04-13") in ds


def test_mutaciya_den_bez_imoex_ostavlen_lovitsya(world, monkeypatch):
    t, s, _ = world
    src = mk(t)
    monkeypatch.setattr(S.Src, "index_days", lambda self, cut: set(s["begin"].dt.normalize()))   # мутация: правило IMOEX снято
    assert HOL_N in days(C.build_cal(NEW_LAST, src))                              # тест выше (HOL_N not in ds) на ней падает


def test_bary_imoex_vne_osnovnoj_sessii_kalendar_ne_menyayut(world, monkeypatch):
    t, _, i = world
    cal = C.build_cal(NEW_LAST, mk(t))
    assert HOL_N in set(i["begin"].dt.normalize()) and HOL_N not in days(cal)    # бары вне сессии есть, день не рабочий
    monkeypatch.setattr(D, "main_session", lambda b: b)                           # мутация: любой бар IMOEX = сессия
    assert HOL_N in days(C.build_cal(NEW_LAST, mk(t)))


def test_t_plus_sleduyushchij_kanonicheskij_den_i_zakrytye_pary_po_t_plus(world):
    t, _, _ = world
    cal = C.build_cal(NEW_LAST, mk(t))
    tp = dict(zip(cal["date"], cal["t_plus"]))
    assert tp[pd.Timestamp("2026-04-09")] == pd.Timestamp("2026-04-13")           # через праздник и выходные, не d+1
    cut = pd.Timestamp("2026-04-09")
    cl = C.closed_pairs(cal, cut)
    assert cl["t_plus"].max() == cut and cl["date"].max() < cut
    mutant = cal[cal["date"] <= cut]                                              # мутация № 142: d <= cut
    assert mutant["t_plus"].max() > cut and len(mutant) == len(cl) + 1


def test_zamok_novye_ceny_do_zamka_isklyuchenie_posle_zamka_stroki(world):
    t, s, i = world
    src = mk(t)
    cut = pd.Timestamp("2026-04-08")
    with pytest.raises(S.SealedError):
        K.build_k1(cut, src)
    with pytest.raises(S.SealedError):
        src.bars("SBER", cut)
    assert K.build_k1(END_A, src)["target_date"].max() == END_A                   # архив без замка читается
    src = mk(t, lock=True)
    k = K.build_k1(cut, src)
    idays = set(D.main_session(i)["begin"].dt.normalize())
    ref = HH.series(D.build_daily(s[s["begin"] <= cut + S.CUT_TIME], index_days=idays)[0])
    pd.testing.assert_frame_equal(k, ref)
    assert k["target_date"].max() == cut and (k["d"] <= END_A).sum() > 100 and (k["target_date"] > END_A).sum() >= 20


def test_mutaciya_zamok_ne_proveryaetsya_lovitsya(world, monkeypatch):
    t, _, _ = world
    src = mk(t)

    def sealed() -> bool:
        try:
            src.bars("SBER", NEW_LAST)
        except S.SealedError:
            return True
        return False
    assert sealed()
    monkeypatch.setattr(S.Src, "is_open", lambda self: True)
    assert not sealed()                                                           # мутация: замок «всегда открыт» — тест выше падает


def test_obrezka_corrupt_after_i_vechernie_bary_d(world):
    t, s, _ = world
    d0 = pd.Timestamp("2025-12-30")
    src = mk(t, lock=True)
    clean = K.build_k1(d0, src)
    HH.corrupt_after(s[s["begin"].dt.normalize() <= END_A], d0).to_parquet(t / "bars" / "SBER.parquet")
    late = K.build_k1(END_A, src)
    pd.testing.assert_frame_equal(late[late["target_date"] <= d0].reset_index(drop=True), clean)
    b = src.begins("SBER", d0)["begin"]
    assert b.max() == d0 + pd.Timedelta("18:40:00")                               # вечерние бары d и дни после d не входят
    assert src.bars("SBER", d0)["begin"].max() == d0 + pd.Timedelta("18:40:00")


def test_peresechenie_arhiva_i_novyh_otkaz(world, tmp_path):
    t, s, _ = world
    mk(t)
    s[s["begin"].dt.normalize() >= pd.Timestamp("2026-02-26")].reset_index(drop=True).to_parquet(tmp_path / "ov.parquet")
    src = S.Src(S.Paths(bars_dir=t / "bars", imoex_archive=t / "i_a.parquet", new_sber=tmp_path / "ov.parquet", new_mkt_dir=t / "mkt", lock=t / "lock.json"))
    with pytest.raises(ValueError):
        src.begins("SBER", NEW_LAST)
