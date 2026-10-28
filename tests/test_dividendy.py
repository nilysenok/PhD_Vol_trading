"""Дивиденды (Ж-057 п. 1, Ж-059 п. 2, Ж-060): слой данных fastbt/dividendy.py, сигналы на скорректированном ряду, ключи кэша (Д5).
Синтетика, данные не читаются. python3 -m pytest tests/test_dividendy.py -q"""
import dataclasses

import numpy as np
import pandas as pd
import pytest

from fastbt import cache as fcache
from fastbt import dividendy, etap1, s9, sigprov
from fastbt.indicators import compute_base
from fastbt.signals.interface import TickerData

N = 300
DATES = pd.bdate_range("2018-01-01", periods=N).values        # пн-пт


def _table(tmp_path, monkeypatch, rows):
    f = tmp_path / "div.csv"
    f.write_text("ticker,record_date,last_cum_date,announce_date,dps_rub\n"
                 + "".join(f"{t},{c},{c},{c},{d}\n" for t, c, d in rows))
    monkeypatch.setattr(dividendy, "TABLE", f)
    return f


def _walk(seed=1):
    r = np.random.default_rng(seed)
    c = 100 * np.exp(np.cumsum(r.normal(0, 0.01, N)))
    lr = np.zeros(N)
    lr[:-1] = np.log(c[1:] / c[:-1])
    return c, lr


# ---------- lr ----------
def test_golden_dni_bez_otsechki_pobaytno_prezhnie(tmp_path, monkeypatch):
    c, lr = _walk()
    _table(tmp_path, monkeypatch, [("T", "2018-06-15", 3.0), ("T", "2018-10-10", 2.0)])
    out = dividendy.adjust_lr("T", DATES, c, lr)
    ch = np.flatnonzero(out != lr)
    g = [int(np.searchsorted(DATES, np.datetime64(x), side="right")) - 1 for x in ("2018-06-15", "2018-10-10")]
    assert ch.tolist() == g
    assert out.tobytes()[:8 * g[0]] == lr.tobytes()[:8 * g[0]]
    keep = np.setdiff1d(np.arange(N), g)
    assert out[keep].tobytes() == lr[keep].tobytes()


def test_tiker_bez_sobytiy_i_pustaya_tablitsa_ne_menyayut_lr(tmp_path, monkeypatch):
    c, lr = _walk()
    _table(tmp_path, monkeypatch, [("OTHER", "2018-06-15", 3.0)])
    assert dividendy.adjust_lr("T", DATES, c, lr).tobytes() == lr.tobytes()


def test_den_gepa_pervyy_torgovyy_posle_last_cum_date(tmp_path, monkeypatch):
    """last_cum_date = пятница → гэп в понедельник; = суббота (не торговый) → тоже понедельник; = дата вне ряда → без поправки."""
    c, lr = _walk()
    fri, sat = "2018-06-15", "2018-06-16"
    for cum in (fri, sat):
        _table(tmp_path, monkeypatch, [("T", cum, 3.0)])
        d = dividendy.gap_divs("T", DATES)
        assert np.flatnonzero(d).tolist() == [int(np.searchsorted(DATES, np.datetime64("2018-06-18")))]
    _table(tmp_path, monkeypatch, [("T", "2030-01-01", 3.0), ("T", "2010-01-01", 3.0)])
    assert not dividendy.gap_divs("T", DATES).any()


def test_formula_i_masshtab_D_delitsya_na_tsenu(tmp_path, monkeypatch):
    """lr[g−1] = log((c[g] + D·0,87)/c[g−1]); цены и D ×10 дают то же lr (мутация «D в рублях без деления на цену» ломает)."""
    c, lr = _walk()
    _table(tmp_path, monkeypatch, [("T", "2018-06-15", 3.0)])
    g = int(np.searchsorted(DATES, np.datetime64("2018-06-15"), side="right"))
    out = dividendy.adjust_lr("T", DATES, c, lr)
    assert out[g - 1] == pytest.approx(np.log((c[g] + 3.0 * 0.87) / c[g - 1]), abs=1e-15)
    _table(tmp_path, monkeypatch, [("T", "2018-06-15", 30.0)])
    out10 = dividendy.adjust_lr("T", DATES, c * 10, lr)
    assert out10[g - 1] == pytest.approx(out[g - 1], abs=1e-13)
    assert abs(np.log(1 + 3.0 * 0.87) - np.log(1 + 0.87 * 3.0 / c[g])) > 0.1          # мутация заметна


def test_flag_div_nalog_polnyy_D(tmp_path, monkeypatch):
    c, lr = _walk()
    _table(tmp_path, monkeypatch, [("T", "2018-06-15", 3.0)])
    g = int(np.searchsorted(DATES, np.datetime64("2018-06-15"), side="right"))
    full = dividendy.adjust_lr("T", DATES, c, lr, nalog=1.0)
    main = dividendy.adjust_lr("T", DATES, c, lr)
    assert main[g - 1] == dividendy.adjust_lr("T", DATES, c, lr, nalog=0.87)[g - 1]
    assert full[g - 1] > main[g - 1]


# ---------- ряд для сигналов ----------
def test_obratnaya_korrektirovka_tsen(tmp_path, monkeypatch):
    c, _ = _walk()
    _table(tmp_path, monkeypatch, [("T", "2018-06-15", 3.0)])
    g = int(np.searchsorted(DATES, np.datetime64("2018-06-15"), side="right"))
    (adj,) = dividendy.adjust_ohlc("T", DATES, c, c)
    assert adj[g - 1] == pytest.approx(c[g - 1] - 3.0, abs=1e-12)          # c_cum·(1 − D/c_cum) = c_cum − D
    assert adj[g:].tobytes() == c[g:].tobytes()                             # с дня гэпа — без изменений
    assert adj[: g - 1] == pytest.approx(c[: g - 1] * (1 - 3.0 / c[g - 1]))


def _td(o, h, l, c, v):
    return TickerData("T", DATES, o, h, l, c, v, np.zeros(N), {})


def _signals(td, monkeypatch, sid):
    monkeypatch.setattr(fcache, "load", lambda k: None)                    # общий кэш не читаем и не пишем
    monkeypatch.setattr(fcache, "save", lambda k, a: None)
    ss = sigprov.get_signals(sid, td, compute_base(td.close, td.high, td.low, td.open, td.volume))
    return np.asarray(ss.sig)


def _series(div):
    """Реальный ряд, у которого после скорректировки получается путь p (сигналы не должны зависеть от D)."""
    r = np.random.default_rng(7)
    p = 100 * np.exp(np.cumsum(r.normal(0, 0.012, N)))
    g = 200
    f = p[g - 1] / (p[g - 1] + div)
    raw = np.where(np.arange(N) < g, p / f, p)                              # реальный ряд: до отсечки цены выше на D
    v = r.uniform(1e6, 2e6, N)
    return p, raw, v, g


@pytest.mark.parametrize("sid", ["S4", "S5", "S6"])
def test_signal_posle_otsechki_ne_zavisit_ot_razmera_D(tmp_path, monkeypatch, sid):
    """D = 0 и D > 0 при одном и том же «скорректированном» пути дают одинаковые сигналы; на реальном ряду — другие (мутация)."""
    D_ = 25.0
    p, raw, v, g = _series(D_)
    _table(tmp_path, monkeypatch, [("T", str(DATES[g - 1])[:10], D_)])
    o, h, l, c = dividendy.adjust_ohlc("T", DATES, raw, raw, raw * 1.01, raw * 0.99, raw)
    assert c == pytest.approx(p, rel=1e-12)
    s_adj = _signals(_td(o, h * 1, l * 1, c, v), monkeypatch, sid)
    s_zero = _signals(_td(p, p * 1.01, p * 0.99, p, v), monkeypatch, sid)
    assert np.array_equal(s_adj, s_zero) and s_zero.any()
    s_raw = _signals(_td(raw, raw * 1.01, raw * 0.99, raw, v), monkeypatch, sid)  # мутация: сигналы на реальном ряду
    assert not np.array_equal(s_raw, s_zero)


# ---------- ключи кэша (Д5) ----------
def _s9_env(tmp_path, monkeypatch):
    from fastbt import data as fd
    src = {}
    for lab, col in (("hyb_h1", "sigma_pred_h1"), ("hyb_h5", "sigma_pred_h5")):
        p = tmp_path / f"{lab}.parquet"
        pd.DataFrame({"date": pd.bdate_range("2019-01-01", periods=5), "ticker": "ZZZ", col: np.arange(5, dtype=float)}).to_parquet(p)
        src[lab] = (p, col, lab[-2:])
    monkeypatch.setattr(s9, "SOURCES", src)
    ohlcv, corp = tmp_path / "ohlcv.parquet", tmp_path / "corp.csv"
    pd.DataFrame({"date": pd.bdate_range("2019-01-01", periods=5), "ticker": "ZZZ", "close": np.arange(5.0)}).to_parquet(ohlcv)
    corp.write_text("ticker,date,factor,status\nZZZ,2019-01-03,0.5,applied\n")
    monkeypatch.setattr(fd, "OHLCV", ohlcv)
    monkeypatch.setattr(fd, "CORP", corp)


def test_klyuch_s9_menyaetsya_ot_tablitsy_i_nalogovogo_mnozhitelya(tmp_path, monkeypatch):
    _s9_env(tmp_path, monkeypatch)
    f = _table(tmp_path, monkeypatch, [("T", "2018-06-15", 3.0)])
    k0 = s9.cache_key()
    assert s9.cache_key() == k0
    f.write_text(f.read_text().replace("3.0", "3.5"))                      # подмена таблицы
    k1 = s9.cache_key()
    assert k1 != k0
    monkeypatch.setattr(dividendy, "DIV_NALOG", 1.0)                       # чувствительность с полным D — другой кэш
    assert s9.cache_key() != k1


def test_mutatsiya_klyuch_s9_bez_dividendov_ne_lovit_by_podmenu(tmp_path, monkeypatch):
    """Контроль теста: если digest() константа (ключ не видит таблицу), подмена таблицы не меняет ключ — ровно это тест ловит."""
    _s9_env(tmp_path, monkeypatch)
    f = _table(tmp_path, monkeypatch, [("T", "2018-06-15", 3.0)])
    monkeypatch.setattr(dividendy, "digest", lambda: "const")
    k0 = s9.cache_key()
    f.write_text(f.read_text().replace("3.0", "3.5"))
    assert s9.cache_key() == k0


def test_klyuch_signalov_vidit_skorrektirovannyy_ryad(tmp_path, monkeypatch):
    """Кэш сигналов: td.digest() берёт скорректированные цены, значит таблица дивидендов меняет ключ."""
    c, _ = _walk()
    v = np.full(N, 1e6)
    _table(tmp_path, monkeypatch, [("T", "2018-06-15", 3.0)])
    td0 = _td(*dividendy.adjust_ohlc("T", DATES, c, c, c, c, c), v)
    _table(tmp_path, monkeypatch, [("T", "2018-06-15", 4.0)])
    td1 = _td(*dividendy.adjust_ohlc("T", DATES, c, c, c, c, c), v)
    assert td0.digest() != td1.digest()


def test_klyuch_etap1_vidit_dividendy():
    base = dict(sk=None, sigma={}, bars={}, close_n6={}, rezhim={}, sobytiya={}, tick={}, rate=pd.Series([0.1], index=[pd.Timestamp("2018-01-01")]),
                meta={})
    e0 = etap1.E1Inputs(**base, div={"T": np.array([0.0, 3.0])})
    e1 = etap1.E1Inputs(**base, div={"T": np.array([0.0, 3.5])})
    assert e0.digest() == etap1.E1Inputs(**base, div={"T": np.array([0.0, 3.0])}).digest()
    assert e0.digest() != e1.digest()
