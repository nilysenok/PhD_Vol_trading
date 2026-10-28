"""Даты отсечки 2017–2021 по правилу T+2 баров (PR div-daty-t2, находка #394): колонка last_cum_date таблицы — только сверка.
Реальный календарь (ряд 18:50, чтение); без ряда тесты пропускаются. Значения цен не читаются, только даты.
python3 -m pytest tests/test_dividendy_daty.py -q"""
import numpy as np
import pandas as pd
import pytest

from fastbt import dividendy
from fastbt.daily_1850 import out_path

pytestmark = pytest.mark.skipif(not out_path().exists(), reason="нет ряда 18:50")


def _key_dates():
    """Возвращает last_cum из events() для MOEX 2019-05-14 и TATN 2019-01-09 (по сумме выплаты из таблицы)."""
    t = pd.read_csv(dividendy.TABLE)
    out = {}
    for tk, rec in (("MOEX", "2019-05-14"), ("TATN", "2019-01-09")):
        dps = float(t[(t.ticker == tk) & (t.record_date == rec)]["dps_rub"].iloc[0])
        e = dividendy.events(tk)
        # событие этого реестра — единственное с такой выплатой в окне ±10 дней от даты реестра
        near = e[(e["dps_rub"] == dps) & ((e["last_cum_date"] - pd.Timestamp(rec)).abs() < pd.Timedelta(days=10))]
        out[tk] = [str(x.date()) for x in near["last_cum_date"]]
    return out


def test_moex_tatn_data_otsechki_t_plus_2_bars():
    k = _key_dates()
    assert k["MOEX"] == ["2019-05-10"]          # до правки (колонка таблицы) — 2019-05-08
    assert k["TATN"] == ["2019-01-04"]          # до правки — 2018-12-28


def test_mutatsiya_kolonka_tablicy_dlya_2017_2021_lovitsya(monkeypatch):
    """Мутация: last_cum_date берётся из колонки таблицы — тест дат обязан упасть."""
    t = pd.read_csv(dividendy.TABLE, parse_dates=["record_date", "last_cum_date"])
    m = t.drop_duplicates("record_date").set_index("record_date")["last_cum_date"]     # дата реестра → колонка таблицы
    monkeypatch.setattr(dividendy, "last_cum", lambda rec: pd.Series(m.reindex(rec).to_numpy(), index=rec.index))
    with pytest.raises(AssertionError):
        test_moex_tatn_data_otsechki_t_plus_2_bars()


def test_golden_gap_divs_tolko_v_dni_otsechek_i_summa_sohranena():
    """Дни без отсечек побайтно нулевые; ненулевые — ровно первые бары после last_cum_date по правилу; сумма D не теряется."""
    cal = dividendy._calendar()
    dates = cal[(cal >= np.datetime64("2017-01-01")) & (cal <= np.datetime64("2021-12-31"))]
    t = pd.read_csv(dividendy.TABLE, parse_dates=["record_date"])
    t = t[(t["dps_rub"] > 0) & (t["record_date"] >= "2017-01-01") & (t["record_date"] < "2022-01-01")]
    t["cum"] = dividendy.last_cum(t["record_date"])
    for tk, g in t.groupby("ticker"):
        want = np.zeros(len(dates))
        for cum, dps in zip(g["cum"], g["dps_rub"]):
            i = int(np.searchsorted(dates, np.datetime64(cum), side="right"))
            if 0 < i < len(dates):
                want[i] += float(dps)
        assert dividendy.gap_divs(tk, dates).tobytes() == want.tobytes(), tk


def test_chislo_raskhozhdeniy_s_tablicey_po_godam():
    """Было/стало: 11 строк различаются (2017: 0, 2018: 3, 2019: 3, 2020: 2, 2021: 3), остальные 116 из 127 совпадают."""
    t = pd.read_csv(dividendy.TABLE, parse_dates=["record_date", "last_cum_date"])
    t = t[t["dps_rub"] > 0]
    diff = t[dividendy.last_cum(t["record_date"]).to_numpy() != t["last_cum_date"].to_numpy()]
    assert diff["record_date"].dt.year.value_counts().sort_index().to_dict() == {2018: 3, 2019: 3, 2020: 2, 2021: 3}
    assert len(t) == 127
