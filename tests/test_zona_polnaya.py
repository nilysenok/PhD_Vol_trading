"""Зона polnaya: во всех слоях чтения S1–S6 максимальная прочитанная дата < period_v2.chtenie_konec('polnaya') (2025-01-01 до акта).
Синтетика с хвостом до 2026-02, без сети; чтение ловится обёрткой pd.read_parquet / pq.read_table и результатом загрузчика.
Мутация (граница wf у слоя) — тот же слой при подмене zona.end/hard на 2026-03-01 обязан ронять проверку.
python3 -m pytest tests/test_zona_polnaya.py -q"""
import contextlib

import pandas as pd
import pyarrow.parquet as pq
import pytest

from fastbt import daily_1850, data, dividendy, e1_rel_sektor, e1_sektor, etap1_eval, period_v2, progon_v7, rezhim_e1, zona
from fastbt import vselennaya as vs

D = pd.Timestamp
DO = D("2025-01-01")
DNI = ["2014-09-01", "2016-06-01", "2020-03-02", "2024-12-20", "2024-12-30", "2025-01-10", "2025-06-02", "2026-02-20"]
READ: list = []


@pytest.fixture(autouse=True)
def _chistka(tmp_path, monkeypatch):
    READ.clear()
    p = tmp_path / "wf_zamok_snyat.json"
    p.write_text('{"preregistraciya_v2": "docs/preregistraciya_v2.md", "sha256": "%s", "data_snyatiya": "2026-10-30"}' % ("a" * 64))
    monkeypatch.setattr(period_v2, "OTMETKA", p)
    for f in (data.daily_df, dividendy._calendar, vs._sostav_wf):
        f.cache_clear()
    yield
    zona.set_zone("dev")
    for f in (data.daily_df, dividendy._calendar, vs._sostav_wf):
        f.cache_clear()


def _zapomnit(t) -> None:
    t = t if isinstance(t, pd.DataFrame) else t.to_pandas()
    for c in ("date", "begin", "timestamp"):
        if c in t.columns and len(t):
            READ.append(pd.to_datetime(t[c]).max())


@pytest.fixture(autouse=True)
def _obertka(monkeypatch):
    rp, rt = pd.read_parquet, pq.read_table

    def r1(*a, **k):
        t = rp(*a, **k)
        _zapomnit(t)
        return t

    def r2(*a, **k):
        t = rt(*a, **k)
        _zapomnit(t)
        return t

    monkeypatch.setattr(pd, "read_parquet", r1)
    monkeypatch.setattr(pq, "read_table", r2)


def _bary(tmp_path, nm="IMOEX"):
    ts = [D(f"{d} 10:00") for d in DNI]
    df = pd.DataFrame({"begin": [t.isoformat(sep=" ") for t in ts], "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "value": 1.0, "volume": 1})
    p = tmp_path / f"{nm}.parquet"
    df.to_parquet(p)
    return p


def _dnevka(tmp_path, kal=False):
    dd = pd.bdate_range("2014-09-01", "2024-12-31") if kal else pd.DatetimeIndex(DNI)
    dd = dd.append(pd.DatetimeIndex(["2025-01-10", "2026-02-20"])) if kal else dd
    p = tmp_path / "d.parquet"
    pd.DataFrame({"date": dd, "ticker": "SBER", "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1}).to_parquet(p)
    return p


def _l_read_cut(tp, mp):
    p = tp / "s.parquet"
    pd.DataFrame({"date": pd.to_datetime(DNI), "v": 1.0}).to_parquet(p)
    data._read_cut(p, "polnaya")


def _l_daily(tp, mp):
    mp.setattr(data, "OHLCV", _dnevka(tp))
    data.daily_df()


def _l_kalendar(tp, mp):
    mp.setattr(daily_1850, "out_path", lambda: _dnevka(tp, kal=True))
    dividendy._calendar()


def _l_dividendy(tp, mp):
    mp.setattr(daily_1850, "out_path", lambda: _dnevka(tp, kal=True))
    s = tp / "svod.csv"
    pd.DataFrame({"ticker": "ZZZZ", "record_date": ["2024-06-05", "2025-03-05", "2026-01-12"], "dps_rub": 1.0, "godna_korr": 1}).to_csv(s, index=False)
    mp.setattr(dividendy, "SVOD", s)
    READ.append(dividendy.events("ZZZZ")["last_cum_date"].max())


def _l_sektora_idx(tp, mp):
    _bary(tp, "MOEXFN")
    with contextlib.suppress(Exception):          # после чтения aggregate ≥ 2025 просит календарь MOEX — для проверки чтения не важно
        e1_sektor.load_indexes(["MOEXFN"], main=tp)


def _l_imoex(tp, mp):
    with contextlib.suppress(Exception):
        rezhim_e1.imoex_daily(_bary(tp))


def _l_rel(tp, mp):
    p = tp / "pit.csv"
    pd.DataFrame({"тикер": "SBER", "дата_ребалансировки": ["2022-06-20", "2024-12-20", "2025-12-22"], "индекс": "MOEXFN"}).to_csv(p, index=False)
    q = tp / "pit0.csv"
    pd.DataFrame({"тикер": "SBER", "дата_ребалансировки": ["2018-01-01"], "индекс": "MOEXFN"}).to_csv(q, index=False)
    mp.setattr(e1_sektor, "SEKTORY_PIT_2022", str(p))
    mp.setattr(e1_rel_sektor, "SEKTORY_CSV", str(q))
    READ.append(max(d for d, _ in e1_rel_sektor.sektor_pit(str(q))["SBER"]))


def _l_sostav(tp, mp):
    p = tp / "sostav.csv"
    pd.DataFrame({"date_eff": ["2022-03-18", "2024-12-20", "2025-03-21"], "known_from": ["2022-03-18", "2024-12-20", "2025-03-21"],
                  "tickers": "SBER GAZP"}).to_csv(p, index=False)
    mp.setattr(vs, "SOSTAV_2022", p)
    READ.append(D("2024-12-30"))                  # состав на последний день до границы читается
    assert vs.sostav(D("2024-12-30")) == frozenset({"SBER", "GAZP"})
    try:
        vs.sostav(D("2025-06-02"))
    except ValueError:
        return
    READ.append(D("2025-06-02"))                  # состав на 2025 отдан — слой читает ≥ 2025


def _l_periods(tp, mp):
    READ.append(D(f"{max(etap1_eval.years_of('polnaya'))}-12-31"))


SLOI = [_l_read_cut, _l_daily, _l_kalendar, _l_dividendy, _l_sektora_idx, _l_imoex, _l_rel, _l_sostav, _l_periods]
IDS = [f.__name__[3:] for f in SLOI]


def _progon(layer, tp, mp):
    with progon_v7.zona_ctx("polnaya"):
        for f in (data.daily_df, dividendy._calendar, vs._sostav_wf):
            f.cache_clear()
        layer(tp, mp)
        assert READ, f"{layer.__name__}: слой ничего не прочитал — проверка слепа"
        assert max(READ) < DO, f"{layer.__name__}: прочитана дата {max(READ).date()} >= {DO.date()}"


@pytest.mark.parametrize("layer", SLOI, ids=IDS)
def test_polnaya_chtenie_do_akta(layer, tmp_path, monkeypatch):
    _progon(layer, tmp_path, monkeypatch)


@pytest.mark.parametrize("layer", SLOI[:-1], ids=IDS[:-1])      # periods — константа импорта, не слой чтения: свой тест ниже
def test_mutaciya_granitsa_wf_u_sloya_lovitsya(layer, tmp_path, monkeypatch):
    wf = period_v2.PERIODY["wf"].konec + pd.Timedelta(days=1)
    monkeypatch.setattr(zona, "end", lambda: wf)
    monkeypatch.setattr(zona, "hard", lambda: wf)
    monkeypatch.setattr(period_v2, "chtenie_konec", lambda period: wf)
    with pytest.raises(AssertionError):
        _progon(layer, tmp_path, monkeypatch)


def test_zona_polnaya_granitsa_i_wf_train4_ne_tronuty():
    with progon_v7.zona_ctx("polnaya"):
        assert zona.name() == "polnaya" and zona.end() == period_v2.chtenie_konec("polnaya") == DO == zona.hard()
    with progon_v7.zona_ctx("wf"):
        assert zona.name() == "wf" and zona.end() == D("2026-03-01") == zona.hard()
    with progon_v7.zona_ctx("train4"):
        assert zona.name() == "dev" and zona.end() == D("2022-01-01")
    assert zona.name() == "dev"


def test_polnaya_bez_otmetki_otkaz(tmp_path, monkeypatch):
    monkeypatch.setattr(period_v2, "OTMETKA", tmp_path / "net.json")
    with pytest.raises(period_v2.ZamokVladeltsa):
        progon_v7.zona_ctx("polnaya").__enter__()
    assert zona.name() == "dev"


def test_periods_polnaya_2017_2024():
    assert etap1_eval.PERIODS["polnaya"] == (2017, 2024) and etap1_eval.years_of("polnaya")[0] == 2017
    assert etap1_eval.PERIODS["wf"] == (2015, 2026)