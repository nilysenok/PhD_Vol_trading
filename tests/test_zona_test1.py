"""Реверс раунд 1, лист R-P: период test1 (2022–2024, прогрев 2021) и граница чтения ≥ 2025. Синтетика, данные ≥ 2022 не читаются.
Умолчание и train4 — побайтно прежние (test_s9_scope, test_s9_a_train4, test_yakor_v1); здесь — только новый механизм.
python3 -m pytest tests/test_zona_test1.py -q"""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from fastbt import daily_1850, data, etap1, etap1_eval, s9, zona


@pytest.fixture(autouse=True)
def _restore():
    old = s9.PERIOD, s9.YEARS, s9.VAL_START
    yield
    s9._restore_period(*old)
    zona.set_zone("dev")


def _bars(path, days):
    ts = pd.to_datetime([f"{d} 10:00" for d in days])
    pd.DataFrame({"timestamp": ts, "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0}).to_parquet(path)


def test_umolchanie_dev_2022_i_z2_2025():
    assert zona.name() == "dev" and zona.end() == pd.Timestamp("2022-01-01")
    with zona.zone_ctx("test1"):
        assert zona.end() == pd.Timestamp("2025-01-01")
    assert zona.name() == "dev"


def test_chtenie_bars_granitsa_zony(tmp_path):
    p = tmp_path / "X.parquet"
    _bars(p, ["2021-12-30", "2022-01-03", "2024-12-30", "2025-01-09"])
    assert len(daily_1850.read_bars(p)) == 1                          # dev: только < 2022
    with zona.zone_ctx("test1"):
        d = daily_1850.read_bars(p)
        assert [str(x.date()) for x in d["timestamp"]] == ["2021-12-30", "2022-01-03", "2024-12-30"]


def test_mutatsiya_chteniya_za_2025_lovitsya(tmp_path):
    """Мутация: явный end за HARD (в любой зоне) — отказ; строка 2025 не попадает в зону Z2."""
    p = tmp_path / "X.parquet"
    _bars(p, ["2024-12-30", "2025-01-09"])
    for z in ("dev", "test1"):
        with zona.zone_ctx(z), pytest.raises(PermissionError):
            daily_1850.read_bars(p, pd.Timestamp("2025-02-01"))
    with zona.zone_ctx("test1"):
        assert pd.Timestamp("2025-01-09") not in set(daily_1850.read_bars(p)["timestamp"])
    with pytest.raises(PermissionError):
        zona.check("2025-01-01")
    zona.check("2024-12-31")


def test_daily_df_z2_put_i_granitsa(tmp_path, monkeypatch):
    ok, bad = tmp_path / "ok.parquet", tmp_path / "bad.parquet"
    pd.DataFrame({"ticker": "A", "date": pd.to_datetime(["2024-12-30"]), "close": 1.0}).to_parquet(ok)
    pd.DataFrame({"ticker": "A", "date": pd.to_datetime(["2025-01-09"]), "close": 1.0}).to_parquet(bad)
    with zona.zone_ctx("test1"):
        monkeypatch.delenv("FASTBT_Z2_DAILY", raising=False)
        with pytest.raises(FileNotFoundError):                         # тихого чтения файла dev нет
            data.daily_df()
        monkeypatch.setenv("FASTBT_Z2_DAILY", str(bad))
        with pytest.raises(PermissionError):
            data.daily_df()
        monkeypatch.setenv("FASTBT_Z2_DAILY", str(ok))
        assert str(data.daily_df()["date"].max().date()) == "2024-12-30"
    assert data.daily_df()["date"].max() < pd.Timestamp("2022-01-01")  # после выхода из зоны — прежний ряд, кэш сброшен


def test_set_period_test1_menyaet_zonu_i_period_ctx_vozvrashchaet():
    with s9.period_ctx("test1"):
        assert (s9.PERIOD, s9.YEARS, s9.VAL_START, zona.name()) == ("test1", (2022, 2023, 2024), 2021, "test1")
    assert (s9.PERIOD, zona.name()) == ("train", "dev")
    with pytest.raises(RuntimeError), s9.period_ctx("test1"):
        raise RuntimeError("x")
    assert (s9.PERIOD, zona.name()) == ("train", "dev")                # и при ошибке


def test_cache_key_test1_otlichaetsya_i_sleduet_za_ryadom(tmp_path, monkeypatch):
    k0 = s9.cache_key()
    daily, div = tmp_path / "d.parquet", tmp_path / "div.csv"
    daily.write_bytes(b"1")
    div.write_text("ticker,record_date,dps_rub\n")
    monkeypatch.setenv("FASTBT_Z2_DAILY", str(daily))
    monkeypatch.setenv("FASTBT_Z2_DIV", str(div))
    s9.set_period("test1")
    k1 = s9.cache_key()
    daily.write_bytes(b"2")                                            # подмена входа при том же периоде — промах кэша
    assert k1 != k0 and s9.cache_key() != k1
    s9.set_period("train")
    assert s9.cache_key() == k0


def test_konec_z2_trebuet_obshchego_dnya():
    mk = lambda last: SimpleNamespace(dates=np.array(["2024-12-29", last], "datetime64[ns]"))     # noqa: E731
    ev = SimpleNamespace(tickers=["A", "B"], _td={"A": (mk("2024-12-30"),), "B": (mk("2024-12-30"),)})
    assert etap1._konec_z2(ev) == pd.Timestamp("2024-12-30")
    ev._td["B"] = (mk("2024-12-27"),)
    with pytest.raises(ValueError):
        etap1._konec_z2(ev)
    ev._td["B"] = (mk("2025-01-09"),)
    with pytest.raises((ValueError, PermissionError)):
        etap1._konec_z2(ev)


def test_b_test1_poslednyaya_stroka_ryada():
    d = pd.to_datetime(["2024-12-27", "2024-12-30"]).values
    ev = SimpleNamespace(inp=SimpleNamespace(meta={"A": {"dates": d}}))
    assert etap1_eval.Evaluator._b(ev, "A", "test1") == 1             # ряд кончается в 2024: b = последний день, не за концом
    d2 = pd.to_datetime(["2019-12-30", "2020-01-02"]).values
    ev.inp.meta["A"]["dates"] = d2
    assert etap1_eval.Evaluator._b(ev, "A", "train4") == 1            # прежнее поведение: первый день после окна


def test_umolchanie_perioda_chitaetsya_pri_vyzove(monkeypatch):
    """AS.PER = "test1" после импорта действует на вызовы без period (мутация: умолчание, связанное при импорте, роняет тест)."""
    from fastbt import a_strogaya as AS
    seen = []

    def bi(sid, ev=None, period=None):
        seen.append(period)
        raise StopIteration
    monkeypatch.setattr(AS.etap1, "build_inputs", bi)
    monkeypatch.setattr(AS, "PER", "test1")
    with pytest.raises(StopIteration):
        AS.evaluatory("S1")
    assert seen == ["test1"]

    def ev2(pt, per):
        seen.append(per)
        raise StopIteration
    with pytest.raises(StopIteration):
        AS.schitat(SimpleNamespace(comm=0.0004, evaluate_v2=ev2), False)
    assert seen[-1] == "test1"
    r = dict(dates=np.array(["2022-01-03"], "datetime64[D]"), x=np.zeros(1), baza=np.zeros(1), expo=np.ones(1),
             cnt=np.zeros(1), god=np.array([2022]))
    assert {row["god"] for row in AS.pasport("A", "v", r)} == {2022, 2023, 2024, "все"}


def test_periods_test1_v_oschenshchike():
    assert etap1_eval.years_of("test1") == (2022, 2023, 2024)
    assert etap1_eval.years_of("train4") == (2016, 2017, 2018, 2019)
