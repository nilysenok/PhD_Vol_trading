"""G6 и мутации PR #804 (known_to в expirations.csv): до 2021 включительно выход каждого читателя
побайтно равен выходу на старом календаре (одна строка 29.03.2022 с known_from 2022-01-01, без колонки known_to);
версии MIX-3.22 имеют верные границы; читатели не оставили распаковки (d, kf) без known_to."""
from __future__ import annotations

import re
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "analysis"))
import d22_harxp as P  # noqa: E402
import d26_1 as Z  # noqa: E402
import d73_features as D73  # noqa: E402
import l2_lib as L  # noqa: E402
import r2_features as R2  # noqa: E402

CAL = ROOT / "data" / "calendar"
DATES = pd.bdate_range("2018-01-01", "2021-12-31")


@pytest.fixture(scope="module")
def stary() -> Path:
    """Копия data/calendar со старым expirations.csv: без known_to, без версий 17.03/24.03/30.03 (29.03 — с 2022-01-01)."""
    import tempfile
    d = Path(tempfile.mkdtemp())
    for f in CAL.glob("*.csv"):
        shutil.copy(f, d / f.name)
    e = pd.read_csv(CAL / "expirations.csv")
    e = e[~e["date"].isin(["2022-03-17", "2022-03-24", "2022-03-30"])].copy()
    e.loc[e["date"] == "2022-03-29", "known_from"] = "2022-01-01"
    e.drop(columns="known_to").to_csv(d / "expirations.csv", index=False)
    return d


def _eq(a: pd.DataFrame, b: pd.DataFrame) -> None:
    pd.testing.assert_frame_equal(a, b, check_exact=True)


def test_g6_d73_i_d7_block(stary: Path) -> None:
    for types in (D73.TYPES, ("cbr", "fomc", "cpi_us", "expirations", "imoex_rebal")):  # вторые — CAL_TYPES d7_block
        for h in (5, 22):
            n = D73.features_frame(DATES, h, D73.load_events(str(CAL), types))
            s = D73.features_frame(DATES, h, D73.load_events(str(stary), types))
            _eq(n, s)


def test_g6_d73_nezavisimyj_etalon_staroj_formuly(stary: Path) -> None:
    """Эталон без izvestna: старая формула kf <= t на старом календаре; ловит ошибку, общую для нового и старого пути."""
    e = pd.read_csv(stary / "expirations.csv", parse_dates=["date", "known_from"])
    got = D73.features_frame(DATES, 22, D73.load_events(str(CAL), ("expirations",)))
    for t in DATES[::7]:
        d = e[e["known_from"] <= t]["date"]
        assert got.loc[t, "n_expirations_h22"] == ((d > t) & (d <= t + pd.Timedelta(days=22))).sum(), t
        fut = d[d > t]
        ozh = (fut.min() - t).days if len(fut) else np.nan  # конец года: будущих известных нет
        assert (np.isnan(ozh) and np.isnan(got.loc[t, "dte_expirations"])) or got.loc[t, "dte_expirations"] == ozh, t


def test_load_v_staryh_testah_ne_oslablen() -> None:
    """Фильтр known_to.isna() в test_calendar_expirations._load отбрасывает ровно три замещённые версии марта 2022."""
    full = pd.read_csv(CAL / "expirations.csv", parse_dates=["date", "known_to"])
    assert sorted(full[full["known_to"].notna()]["date"].astype(str)) == ["2022-03-17", "2022-03-24", "2022-03-30"]


def test_g6_l2_f_cal_i_f_season(stary: Path) -> None:
    A = pd.DataFrame({"date": DATES})
    en, es = L.load_calendar(("expirations",), CAL), L.load_calendar(("expirations",), stary)
    _eq(L.f_cal(DATES, 22, en), L.f_cal(DATES, 22, es))
    _eq(L.f_season(A, en), L.f_season(A, es))


def test_g6_d22_i_d26_1(stary: Path) -> None:
    P._EV.clear()
    en = P._events(CAL)
    P._EV.clear()
    es = P._events(stary)
    P._EV.clear()
    for a, b in ((1, 5), (1, 22)):
        _eq(P.cal_window(DATES, a, b, en), P.cal_window(DATES, a, b, es))
    for j in (1, 3):
        _eq(Z.cal_lead(DATES, j, en), Z.cal_lead(DATES, j, es))


def test_g6_r2_zagruzchik_i_cal_window(stary: Path, monkeypatch) -> None:
    """Реальный load_inputs (parquet подменены заглушкой), I['EV'] -> cal_window; календарь — новый и старый."""
    cols = ["date", "ticker", "volume", "close_RVI", "rv_IMOEX", "imoex_low_bars", "close_IMOEX", "sp500", "eem", "dxy",
            "gold", "brent", "wti", "vix", "rv", "bv", "j"]
    stub = pd.DataFrame({c: [0.0] for c in cols}).assign(date=pd.Timestamp("2020-01-02"), ticker="SBER")
    monkeypatch.setattr(R2.pd, "read_parquet", lambda *a, **k: stub.copy())
    res = []
    for cal in (CAL, stary):
        monkeypatch.setattr(R2, "CALD", cal)
        ev = R2.load_inputs()["EV"]
        assert "known_to" in ev["expirations"]
        res.append(R2.cal_window(DATES, 1, 22, ev))
    _eq(*res)


def test_d26_3_i_d26_1_net_dvuhelementnoy_raspakovki() -> None:
    """d26_3.leak и все читатели получают от _events тройки: распаковка (d, kf) без known_to запрещена."""
    for f in ("d26_3.py", "d26_1.py", "d22_harxp.py", "l2_lib.py"):
        src = (ROOT / "scripts" / "analysis" / f).read_text(encoding="utf-8")
        assert not re.search(r"for \w+, \(d, kf\) in", src), f
    P._EV.clear()
    ev = P._events(CAL)
    P._EV.clear()
    assert all(len(v) == 3 for v in ev.values())
    ev0, ev2 = Z._perturb_events(pd.Timestamp("2021-06-30"), np.random.default_rng(1))
    assert all(len(v) == 3 for v in ev2.values())


def _versii() -> pd.DataFrame:
    d = pd.read_csv(CAL / "expirations.csv", parse_dates=["date", "known_from", "known_to"])
    return d[d["date"].astype(str).str[:7].eq("2022-03")].sort_values("known_from").reset_index(drop=True)


def test_granitsy_versiy_mix_3_22() -> None:
    """Каждая версия начинается в день раскрытия, а предыдущая кончается им же (известна: kf <= t < kt)."""
    v = _versii()
    assert [str(x.date()) for x in v["date"]] == ["2022-03-17", "2022-03-24", "2022-03-30", "2022-03-29"]
    assert [str(x.date()) for x in v["known_from"]] == ["2022-01-01", "2022-03-15", "2022-03-23", "2022-03-24"]
    assert [str(x.date()) for x in v["known_to"][:3]] == ["2022-03-15", "2022-03-23", "2022-03-24"]
    assert pd.isna(v["known_to"].iloc[3])
    assert (v["known_to"][:3].values == v["known_from"][1:].values).all()  # встык, без дыр и перекрытий


def test_mutatsiya_izvestna_nestrogo_lovitsya() -> None:
    """Мутация `<= known_to`: 15.03 была бы видна и старая, и новая версия."""
    v = _versii()
    t = pd.Timestamp("2022-03-15")
    ok = D73.izvestna(v["known_from"], v["known_to"], t)
    mut = (v["known_from"] <= t) & (pd.isna(v["known_to"]) | (v["known_to"] >= t))
    assert ok.sum() == 1 and mut.sum() == 2


def test_mutatsiya_pustoy_known_to_ne_deystvuet_lovitsya() -> None:
    """Пустой known_to = действует: последняя версия (29.03) видна на 28.03 и далее."""
    v = _versii()
    for t in ("2022-03-28", "2022-04-15"):
        assert D73.izvestna(v["known_from"], v["known_to"], pd.Timestamp(t)).tolist() == [False, False, False, True]


def test_mutatsiya_24_03_s_16_03_lovitsya() -> None:
    """Версия 24.03 с known_from 16.03 (на день позже раскрытия 15.03): 15.03 — ни одной даты марта."""
    v = _versii()
    t = pd.Timestamp("2022-03-15")
    assert D73.izvestna(v["known_from"], v["known_to"], t).tolist() == [False, True, False, False]
    v.loc[1, "known_from"] = pd.Timestamp("2022-03-16")
    assert not D73.izvestna(v["known_from"], v["known_to"], t)[1:].any()  # мутация даёт дыру; на этом тест и держится


def test_d26_1_proverka_utechki_uchityvaet_known_to() -> None:
    """_perturb_events возмущает только версии с known_from > cut и несёт known_to без изменений;
    мутация «кортеж без known_to» ломает распаковку в cal_lead."""
    cut = pd.Timestamp("2022-03-16")
    ev, ev2 = Z._perturb_events(cut, np.random.default_rng(7))
    d, kf, kt = ev2["expirations"]
    assert kt.astype(str).tolist() == ev["expirations"][2].astype(str).tolist()  # NaT == NaT ложно, сравниваем строками
    assert (kf > np.datetime64(cut.date())).any()
    t16 = [pd.Timestamp("2022-03-16")]
    P._EV.clear()
    ev = P._events(CAL)  # _perturb_events читает календарь из FASTBT_MAIN (тень), поэтому для мутации — календарь ветки
    P._EV.clear()
    ok = Z.cal_lead(t16, 1, ev).iloc[0]["cal_dte_expirations"]
    mut = Z.cal_lead(t16, 1, {k: (a, b, np.full(len(c), np.datetime64("NaT"), dtype=c.dtype)) for k, (a, b, c) in ev.items()})
    assert ok == 8 and mut.iloc[0]["cal_dte_expirations"] == 1  # утечка без known_to: виден и старый срок 17.03
    with pytest.raises(ValueError):
        Z.cal_lead(DATES[:5], 1, {k: v[:2] for k, v in ev2.items()})
