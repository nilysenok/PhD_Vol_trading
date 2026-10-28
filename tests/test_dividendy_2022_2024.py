"""Даты отсечки 2022–2024 по ценам (Ж-121..123, Д8; метрика #400, mutations.md № 34): эталон — цены, не last_cum().
Объявлено до прогона (метрика и окно — как tests/test_dividendy_gap.py):
  * день гэпа t — торговый день в окне ±4 бара вокруг бара даты реестра, где open(t)/close(t−1) − 1 ближе всего
    к −D/close(t−1); last_cum по цене = бар t−1;
  * «видимый» гэп: D > 2 % close (порог #400) и r(t) < 0; остальные строки — отдельным списком, не подгоняются;
  * движок: fastbt.dividendy.events() в зоне test1 (FASTBT_Z2_DIV = эта таблица, FASTBT_Z2_DAILY = ряд 18:50 2022–2024),
    правило T+2 до 31.07.2023, T+1 с T1_OT.
Данные: Z2 (< 2025-01-01), ничего ≥ 2025 не читается (проверка max(date)).
python3 -m pytest tests/test_dividendy_2022_2024.py -q -s"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fastbt import dividendy, zona

ROOT = Path(__file__).resolve().parent.parent
DIV = ROOT / "docs" / "issledovaniya" / "2026-10-29_dividendy_2022_2024.csv"
DAILY = Path("/Users/nilysenok/Desktop/Старое/PhD_Vol_trading/data/regen/test1_2022_2024/daily/ohlcv_daily_1850.parquet")
WIN, THR = 4, 0.02
T1 = pd.Timestamp(dividendy.T1_OT)


# Видимые (D > 2 %, r < 0), где день по цене ≠ день движка. Разобраны, не подгоняются:
#  MOEX 2024-06-14: 12.06 — выходной, на 13.06 открытие −10 % (санкции США против МОЭКС 12–13.06.2024: рыночный шок, не гэп D);
#                   по T+1 last_cum = 13.06, гэп 14.06 (−3,6 %, D = 7,2 %) — дата реестра таблицы согласуется с T+1.
#  NVTK 2023-10-10: D = 2,0 % (на пороге), r(10.10) = −1,2 % и r(12.10) = −1,25 % — оба в шуме, минимум на 12.10 случаен.
KNOWN_BAD = {("MOEX", "2024-06-14"), ("NVTK", "2023-10-10")}


def _gap_cum(b, rec, dps):
    """День перед гэпом по ценам: (дата, |ошибка|, вторая по близости ошибка, r(t))."""
    dt, c, o = b["date"].values, b["close"].values, b["open"].values
    i = int(np.searchsorted(dt, np.datetime64(rec), side="right")) - 1
    cand = sorted((abs(o[t] / c[t - 1] - 1 + dps / c[t - 1]), t) for t in range(max(i - WIN, 1), min(i + WIN + 1, len(c))))
    t = cand[0][1]
    return pd.Timestamp(dt[t - 1]), cand[0][0], cand[1][0], o[t] / c[t - 1] - 1


def _bar_before(b, rec, lag):
    i = int(np.searchsorted(b["date"].values, np.datetime64(rec), side="right")) - 1
    return b["date"][i - lag]


@pytest.fixture(scope="module")
def z2(request):
    if not (DIV.exists() and DAILY.exists()):
        pytest.skip("нет таблицы 2022–2024 или ряда 18:50 зоны Z2")
    mp = pytest.MonkeyPatch()
    mp.setenv("FASTBT_Z2_DIV", str(DIV))
    mp.setenv("FASTBT_Z2_DAILY", str(DAILY))
    zona.set_zone("test1")
    request.addfinalizer(lambda: (zona.set_zone("dev"), mp.undo()))
    d = pd.read_parquet(DAILY, columns=["ticker", "date", "open", "close"])
    d["date"] = pd.to_datetime(d["date"])
    zona.check(d["date"].max(), "ряд 18:50 Z2")                      # ≥ 2025 закрыт
    d = d.sort_values(["ticker", "date"])
    rows = pd.read_csv(DIV, parse_dates=["record_date"])
    out = []
    for _, r in rows.iterrows():
        b = d[d["ticker"] == r["ticker"]].reset_index(drop=True)
        e = dividendy.events(r["ticker"])
        rec = r["record_date"]
        m = e[(e["dps_rub"] == r["dps_rub"]) & (e["last_cum_date"] <= rec) & (e["last_cum_date"] > rec - pd.Timedelta(days=10))]
        got = m["last_cum_date"].iloc[0] if len(m) else pd.NaT
        if b.empty:                                                  # тикера нет в ряду 18:50 (GAZP): цен нет, гэп не проверить
            out.append(dict(tic=r["ticker"], rec=rec, b=b, got=got, want=pd.NaT, vis=False, dps=r["dps_rub"], nobars=True))
            continue
        want, err, err2, ret = _gap_cum(b, rec, r["dps_rub"])
        i = int(np.searchsorted(b["date"].values, np.datetime64(rec), side="right")) - 1
        vis = r["dps_rub"] / b["close"][i - 2] > THR and ret < 0
        out.append(dict(tic=r["ticker"], rec=rec, b=b, got=got, want=want, vis=vis, dps=r["dps_rub"], nobars=False))
    return out


def test_55_strok(z2):
    assert len(z2) == 55, len(z2)


def test_pravilo_sovpadaet_s_dnem_gepa(z2):
    vis = [x for x in z2 if x["vis"]]
    bad = [(x["tic"], str(x["rec"].date()), str(x["got"])[:10], str(x["want"].date())) for x in vis if x["got"] != x["want"]]
    nob = [(x["tic"], str(x["rec"].date())) for x in z2 if x["nobars"]]
    print("без цен (тикера нет в ряду 18:50):", len(nob), nob)
    inv = [x for x in z2 if not x["vis"] and not x["nobars"]]
    inv_bad = [(x["tic"], str(x["rec"].date()), str(x["got"])[:10], str(x["want"].date())) for x in inv if x["got"] != x["want"]]
    print(f"\nвидимых {len(vis)}, совпало {len(vis) - len(bad)}; невидимых {len(inv)}, совпало {len(inv) - len(inv_bad)}")
    print("несовпавшие видимые:", bad)
    print("несовпавшие невидимые (гэп в шуме, не подгоняются):", inv_bad)
    assert {(t, r) for t, r, _, _ in bad} == KNOWN_BAD, bad          # любое новое расхождение или исчезновение известного — красный


def _shifted(x, k):
    b = x["b"]
    j = int(np.searchsorted(b["date"].values, np.datetime64(x["got"])))
    return b["date"][j + k]


@pytest.mark.parametrize("k", [1, -1])
def test_mutaciya_data_otsechki_plus_minus_den_lovitsya(z2, k):
    vis = [x for x in z2 if x["vis"]]
    assert any(_shifted(x, k) != x["want"] for x in vis), f"сдвиг {k:+d} не пойман"
    assert sum(_shifted(x, k) == x["want"] for x in vis) < len(vis)


def test_mutaciya_t2_dlya_vseh_let_lovitsya(z2):
    late = [x for x in z2 if x["rec"] >= T1 and not x["nobars"]]
    assert late, "нет строк после 31.07.2023"
    t2 = [x for x in late if _bar_before(x["b"], x["rec"], 2) != x["want"]]
    assert t2, "T+2 для всех лет не отличается от цены ни на одной строке после T1_OT"
    assert any(x["vis"] for x in t2), "среди видимых после T1_OT T+2 не ловится"


def test_sintetika_gap_cum_nahodit_znanyj_gep():
    dates = pd.bdate_range("2023-09-01", periods=20)
    c = np.full(20, 100.0)
    o = c.copy()
    o[10:] -= 5.0
    c[10:] -= 5.0                              # гэп −5 на баре 10, D = 5
    b = pd.DataFrame({"date": dates, "open": o, "close": c})
    got, err, err2, ret = _gap_cum(b, dates[9], 5.0)
    assert got == dates[9] and err < 1e-12 and err2 > 0.01 and ret < 0
    assert _gap_cum(b, dates[9], 5.0)[0] != dates[10]       # мутация +1 различима
