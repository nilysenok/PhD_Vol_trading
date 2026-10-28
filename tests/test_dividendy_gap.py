"""Даты отсечки по ценовым гэпам (PR #400, Ж-072 п. 2): эталон — цены, а не last_cum() и не колонка таблицы.
Объявлено до прогона:
  * строки: выплаты с record_date < 2020-01-01 (2020–2021 — validation, цены не читаются) и D > 2 % close(t−1);
  * день гэпа t — торговый день в окне ±4 бара вокруг бара даты реестра, где r(t) = open(t)/close(t−1) − 1
    ближе всего к −D/close(t−1) (минимум |r + D/c|); окно строится от record_date, не от last_cum;
  * ожидание: last_cum(record_date) = бар перед t, 55 из 55.
python3 -m pytest tests/test_dividendy_gap.py -q"""
import numpy as np
import pandas as pd
import pytest

from fastbt import dividendy

CUT = pd.Timestamp("2020-01-01")
WIN = 4
# (тикер, реестр, last_cum литералом, основание). Литералы посчитаны по ценам 18:50 (гэп open/close(t−1) − 1 ≈ −D/c) и
# по календарю баров тикера, не вызовом last_cum(). "price" — гэп различим (ошибка минимума заметно меньше второго),
# "calendar" — D 0,7–1,4 % цены ниже шума, гэп не различим, дата закреплена по календарю баров (T+2).
SIX = [("MOEX", "2019-05-14", "2019-05-10", "price"),     # D 8,4 %: гэп 13.05
       ("TATN", "2019-01-09", "2019-01-04", "price"),     # D 3,0 %: гэп 08.01
       ("PHOR", "2018-03-12", "2018-03-07", "price"),     # D 0,84 %: минимум на 08.03, но запас слабый — и календарь T+2
       ("NVTK", "2018-05-03", "2018-04-30", "price"),     # D 1,0 %: слабый запас — и календарь T+2
       ("NVTK", "2019-05-06", "2019-05-02", "calendar"),  # D 1,4 %: гэп в шуме; майские праздники
       ("PHOR", "2018-06-13", "2018-06-09", "calendar")]  # D 0,65 %: гэп в шуме; 09.06.2018 — рабочая суббота


def _bars():
    from fastbt.daily_1850 import out_path
    if not out_path().exists():
        pytest.skip("нет ряда 18:50")
    d = pd.read_parquet(out_path(), columns=["ticker", "date", "open", "close"])
    d["date"] = pd.to_datetime(d["date"])
    return d[d["date"] < CUT].sort_values(["ticker", "date"])      # цены 2020+ не читаются


def _gap_cum(b, rec, dps):
    """День перед гэпом по ценам. Возвращает (дата, |ошибка|, вторая по близости ошибка)."""
    dt = b["date"].values
    i = int(np.searchsorted(dt, np.datetime64(rec), side="right")) - 1       # бар даты реестра (или предыдущий)
    c, o = b["close"].values, b["open"].values
    cand = [(abs(o[t] / c[t - 1] - 1 + dps / c[t - 1]), t) for t in range(max(i - WIN, 1), min(i + WIN + 1, len(c)))]
    cand.sort()
    return pd.Timestamp(dt[cand[0][1] - 1]), cand[0][0], cand[1][0]


def _rows():
    d = _bars()
    o = pd.read_csv(dividendy.TABLE, parse_dates=["record_date"])
    o = o[(o["dps_rub"] > 0) & (o["record_date"] < CUT)]
    for _, r in o.iterrows():
        b = d[d["ticker"] == r["ticker"]].reset_index(drop=True)
        i = int(np.searchsorted(b["date"].values, np.datetime64(r["record_date"]), side="right")) - 1
        if r["dps_rub"] / b["close"][i - 2] > 0.02:
            yield r, b


def _mismatch(cum_of):
    bad, n = [], 0
    for r, b in _rows():
        n += 1
        want = _gap_cum(b, r["record_date"], r["dps_rub"])[0]
        got = cum_of(r, b)
        if got != want:
            bad.append((r["ticker"], str(r["record_date"].date()), str(got)[:10], str(want.date())))
    return n, bad


def _shift(k):
    def f(r, b):
        j = int(np.searchsorted(b["date"].values, np.datetime64(_rule(r, b))))
        return b["date"][j + k]
    return f


def _table_col(r, b):
    t = pd.read_csv(dividendy.TABLE, parse_dates=["record_date", "last_cum_date"])
    return t[(t["ticker"] == r["ticker"]) & (t["record_date"] == r["record_date"])]["last_cum_date"].iloc[0]


def _engine(tic, rec, dps):
    """last_cum_date, которую видит движок (events → gap_divs): событие тикера с тем же D в 10 днях до реестра."""
    e = dividendy.events(tic)
    rec = pd.Timestamp(rec)
    m = e[(e["dps_rub"] == dps) & (e["last_cum_date"] <= rec) & (e["last_cum_date"] > rec - pd.Timedelta(days=10))]
    assert len(m) <= 1, (tic, str(rec.date()), len(m))
    return m["last_cum_date"].iloc[0] if len(m) else pd.NaT      # нет события в этом окне — расхождение


def _rule(r, b):
    return _engine(r["ticker"], r["record_date"], r["dps_rub"])


def test_pravilo_t2_sovpadaet_s_dnem_gepa_po_cenam_55_iz_55():
    n, bad = _mismatch(_rule)
    assert n == 55, n
    assert not bad, f"расхождения (тикер, реестр, last_cum, по гэпу): {bad}"


@pytest.mark.parametrize("name,fn", [("plus_den", _shift(1)), ("minus_den", _shift(-1)), ("kolonka_tablicy", _table_col)])
def test_mutatsii_lovyatsya(name, fn):
    n, bad = _mismatch(fn)
    assert bad, f"мутация {name} не поймана: {n} из {n} совпали"


def test_shest_strok_400_zakrepleny_literalami():
    d = _bars()
    for tic, rec, want, how in SIX:
        b = d[d["ticker"] == tic].reset_index(drop=True)
        dps = float(pd.read_csv(dividendy.TABLE).query("ticker == @tic and record_date == @rec")["dps_rub"].iloc[0])
        got = _engine(tic, rec, dps)                                         # дата движка (events), не колонка таблицы
        assert got == pd.Timestamp(want), (tic, rec, str(got)[:10], want)
        i = int(np.searchsorted(b["date"].values, np.datetime64(rec), side="right")) - 1
        assert b["date"][i - 2] == pd.Timestamp(want), (tic, rec)           # календарь баров тикера, независимо от кода
        if how == "price":
            assert _gap_cum(b, rec, dps)[0] == pd.Timestamp(want), (tic, rec)
