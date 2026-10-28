"""К-24а: внешние ряды на 18:50 (синтетика с известным ответом + три обязательные мутации)."""
import numpy as np
import pandas as pd

from fastbt import vneshnie_1850 as v


def _bars(day, rows):
    return pd.DataFrame([{"timestamp": pd.Timestamp(f"{day} {t}"), "open": c, "high": c, "low": c,
                          "close": c, "volume": 1} for t, c in rows])


# --- мутация 1: значение после 18:50 попадает в d --------------------------------------------
def test_vechernie_bary_ne_vhodyat_v_d(monkeypatch):
    b = _bars("2020-07-01", [("18:30", 9.0), ("18:40", 10.0), ("19:00", 99.0), ("23:40", 77.0)])
    assert v.close_1850(b, "x").iloc[0] == 10.0
    monkeypatch.setattr(v, "CUT_MIN", 24 * 60)
    assert v.close_1850(b, "x").iloc[0] == 77.0      # мутация меняет ответ -> тест выше её ловит


# --- склейка Brent -------------------------------------------------------------------------
def _contracts():
    lst = ["2020-03-20", "2020-04-20"]
    t = pd.DataFrame({"secid": ["A", "B"], "lsttrade": pd.to_datetime(lst)})
    t["roll"] = [v.roll_date(x) for x in t["lsttrade"]]
    return t


def _per():
    base = pd.Series(np.log(50) + np.cumsum(np.full(60, 0.001)), index=pd.bdate_range("2020-02-03", periods=60))
    b = np.exp(base[base.index >= "2020-02-17"])
    b = b[b.index <= "2020-04-20"]
    a = np.exp(base[base.index <= "2020-03-20"]) + 5.0            # другой уровень: контанго
    a.loc[a.index >= "2020-03-13"] = 500.0                         # мусор в последние 5 дней A
    return {"A": a, "B": b}, b


def test_sklejka_po_dohodnostyam_bez_skachka():
    per, b = _per()
    out = v.splice_brent(per, _contracts()).set_index("date")
    assert out.loc["2020-03-12", "secid"] == "A" and out.loc["2020-03-13", "secid"] == "B"
    assert out.loc["2020-03-13", "close"] == b["2020-03-13"]       # мусор A (500) не виден
    exp = np.log(b["2020-03-13"] / b["2020-03-12"])
    assert abs(out.loc["2020-03-13", "ret"] - exp) < 1e-12         # на смене контракта доходность B, без скачка
    assert out["ret"].abs().max() < 0.05


def test_mutatsiya_sklejka_urovnem_lovitsya(monkeypatch):
    per, _ = _per()
    monkeypatch.setattr(v, "SPLICE_BY_LEVEL", True)
    out = v.splice_brent(per, _contracts()).set_index("date")
    assert abs(out.loc["2020-03-13", "ret"]) > 0.05                 # скачок от смены уровня — тест выше упал бы


def test_mutatsiya_bez_perehoda_na_sleduyuschiy_kontrakt_lovitsya(monkeypatch):
    per, _ = _per()
    monkeypatch.setattr(v, "ROLL_DAYS", 0)
    t = _contracts()
    out = v.splice_brent(per, t).set_index("date")
    assert out.loc["2020-03-13", "secid"] == "A" and out.loc["2020-03-13", "close"] == 500.0


def test_net_ffill_dni_bez_brent_schitayutsya():
    per, _ = _per()
    per["B"] = per["B"].drop(pd.Timestamp("2020-03-18"))
    out = v.splice_brent(per, _contracts())
    assert pd.Timestamp("2020-03-18") not in set(out["date"])
    cal = pd.bdate_range("2020-03-02", "2020-03-31")
    assert v.missing_days(out, cal).to_dict() == {2020: 1}


def test_posle_poslednego_kontrakta_pusto():
    assert v.active_secid(pd.to_datetime(["2020-04-30"]), _contracts()).iloc[0] is None


# --- мутация 3: чтение ≥ 2022 --------------------------------------------------------------
def _write(tmp_path, name, as_str):
    ts = pd.to_datetime(["2021-12-30 18:40", "2022-01-10 10:00"])
    begin = ts.strftime("%Y-%m-%d %H:%M:%S") if as_str else ts
    p = tmp_path / name
    pd.DataFrame({"begin": begin, "close": [1.0, 2.0]}).to_parquet(p)
    return p


def test_chtenie_obrezaet_2022_obe_formy_begin(tmp_path):
    for s in (True, False):
        b = v.read_bars(_write(tmp_path, f"x{s}.parquet", s))
        assert b["timestamp"].max() < pd.Timestamp("2022-01-01") and len(b) == 1


def test_mutatsiya_chteniya_za_predelami_lovitsya(tmp_path, monkeypatch):
    p = _write(tmp_path, "m.parquet", True)
    monkeypatch.setattr(v, "END", pd.Timestamp("2030-01-01"))
    assert v.read_bars(p)["timestamp"].max() >= pd.Timestamp("2022-01-01")


# --- Д1: проверка сдвига -------------------------------------------------------------------
def test_sdvig_maksimum_na_nule_i_utechka_lovitsya():
    r = np.random.default_rng(1)
    idx = pd.bdate_range("2018-01-01", periods=400)
    y = pd.Series(r.normal(size=400), index=idx)
    ok = y + pd.Series(r.normal(scale=0.3, size=400), index=idx)
    leak = y.shift(-1).fillna(0.0)                                  # x_d = y_{d+1}: знание завтрашнего дня
    for x, zero in ((ok, True), (leak, False)):
        c = v.shift_corr(x, y)
        assert (max(c, key=lambda k: abs(c[k])) == 0) is zero
    assert v.max_at_zero(pd.DataFrame({"ryad": "a", "cel": "b", "sdvig": [-2, -1, 0, 1],
                                       "corr": [0.1, 0.2, 0.9, 0.3]}))[("a", "b")]
