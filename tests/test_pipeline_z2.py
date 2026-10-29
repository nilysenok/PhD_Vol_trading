"""Режим Z2 в fastbt.pipeline (реверс р.1, сверка 5а): σ̂ режется < 2025 на чтении, зона ставится в воркере, годы < 2025.
Синтетика; данные ≥ 2022 основной копии не читаются. python3 -m pytest tests/test_pipeline_z2.py -q"""
import pandas as pd
import pytest

from fastbt import data, pipeline, zona
from fastbt.config import Config


@pytest.fixture(autouse=True)
def _dev():
    yield
    zona.set_zone("dev")
    data.vpred_df.cache_clear()


def _vpred(path, days):
    pd.DataFrame({"date": pd.to_datetime(days), "ticker": "SBER", "sigma_h1": 0.01,
                  "sigma_h5": 0.01, "sigma_h22": 0.01}).to_parquet(path)


def _bez_holdout(df) -> bool:
    return bool((pd.to_datetime(df["date"]) < zona.HARD).all())


def test_vpred_z2_otsechka_na_chtenii(tmp_path, monkeypatch):
    p = tmp_path / "vp.parquet"
    _vpred(p, ["2021-12-30", "2024-12-30", "2025-01-09", "2026-02-02"])
    monkeypatch.delenv("FASTBT_Z2_R2FINAL", raising=False)
    monkeypatch.setenv("FASTBT_Z2_VPRED", str(p))
    with zona.zone_ctx("test1"):
        v = data.vpred_df()
        assert [str(d.date()) for d in v["date"]] == ["2021-12-30", "2024-12-30"]
        assert _bez_holdout(v)


def test_vpred_dev_kak_bylo(tmp_path, monkeypatch):
    """Умолчание (зона dev) — побайтно прежнее чтение: файл целиком, без фильтра."""
    p = tmp_path / "vp.parquet"
    _vpred(p, ["2021-12-30", "2025-01-09"])
    monkeypatch.setattr(data, "VPRED", p)
    data.vpred_df.cache_clear()
    assert zona.name() == "dev" and len(data.vpred_df()) == 2


def test_mutatsiya_chtenie_bez_otsechki_lovitsya(tmp_path, monkeypatch):
    """Мутация: read_parquet без filters (строка ≥ 2025 попала в память) — ловится: отказ или строка за границей."""
    p = tmp_path / "vp.parquet"
    _vpred(p, ["2024-12-30", "2025-01-09"])
    orig = pd.read_parquet

    def bez_filtra(path, *a, filters=None, **k):            # подложенная ошибка: фильтр отброшен
        return orig(path, *a, **k)

    monkeypatch.setattr(data.pd, "read_parquet", bez_filtra)
    with zona.zone_ctx("test1"):
        with pytest.raises(PermissionError):
            data._read_cut(p)
    monkeypatch.setattr(data.pd, "read_parquet", orig)
    with zona.zone_ctx("test1"):
        assert _bez_holdout(data._read_cut(p))


def test_z2_config_gody_do_2025():
    c = pipeline.z2_config(Config())
    assert c.years_a == (2020, 2021, 2022, 2023, 2024) and c.years_bcd == (2022, 2023, 2024)
    assert pipeline.z2_config(Config.gate()).mask_old is True          # остальное не трогается


def test_run_ticker_stavit_zonu_v_vorkere(monkeypatch):
    """Воркер spawn начинает в зоне dev: зона приходит в задании и ставится до расчёта; без зоны — как было."""
    vidno = []

    def fake(ticker, sid, cfg, td=None):
        vidno.append(zona.name())
        return pd.DataFrame(columns=pipeline.COLS), {}

    monkeypatch.setattr(pipeline, "run_unit", fake)
    pipeline.run_ticker(("SBER", ["S1"], Config(), "test1"))
    zona.set_zone("dev")
    pipeline.run_ticker(("SBER", ["S1"], Config()))
    assert vidno == ["test1", "dev"]


def test_run_all_z2_peredaet_zonu_i_gody(monkeypatch):
    zadaniya = []

    def fake(args):
        zadaniya.append(args)
        return args[0], pd.DataFrame(columns=pipeline.COLS), {}, 0.0

    monkeypatch.setattr(pipeline, "run_ticker", fake)
    pipeline.run_all(Config(), ["SBER"], ["S1"], workers=1, log=lambda m: None, zone="test1")
    pipeline.run_all(Config(), ["SBER"], ["S1"], workers=1, log=lambda m: None)
    (t, s, c, z), stary = zadaniya
    assert z == "test1" and max(c.years_a) == 2024 and max(c.years_bcd) == 2024
    assert len(stary) == 3 and stary[2] == Config()                    # умолчание: задание и годы прежние
