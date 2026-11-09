"""V7 лист 2a, G27: запись и `--check` конфигураций 4–7 на синтетике (боевые функции `run_prod`, `record`, `check_cfg`, артефакты Stage A подменены загрузчиком, выходы — во временном каталоге).
Проверка читает и ничего не пишет; ловит: подмену прогноза (sha256), подмену прогноза с пересчитанным sha (переоценка побитно), F ≠ H при G = 0, даты не как у конфигурации 1, число параметров."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from models_v2.s14 import v7_0_io as IO, v7_1b_cfgrun as CR, v7_1m_forms as FM, v7_1m_syn as SY, v7_2_cfg as C2, v7_2_run as R2

S = SY.make(720, 1)
TAB, SEEDS = S["tab"], (0,)


def fake_load(path):
    """Артефакт энкодера года Y: z для всех дней ≤ конца года; z даты d — S['z'][d] (не зависит от года и seed)."""
    year = int(path.name.removesuffix(".npz").split("_")[2])
    days = TAB.index[TAB.index < pd.Timestamp(year + 1, 1, 1)]
    return dict(days=days.to_numpy("datetime64[ns]").astype(np.int64), z=S["z"][:len(days)])


@pytest.fixture(scope="module")
def made(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("v72")
    mp = pytest.MonkeyPatch()
    mp.setattr(IO, "ROOT", tmp)
    mp.setattr(IO, "OUT", tmp / "out")
    mp.setattr(R2, "same_dates", lambda df: True)
    out = {}
    for cfg in ("4", "6"):
        df, per = C2.run_prod(cfg, TAB, "tcn", load=fake_load, seeds=SEEDS)
        out[cfg] = (R2.record(cfg, df, per, "tcn", 1.0), df)
    yield out, tmp
    mp.undo()


def chk(cfg, rec, mp):
    return R2.check_cfg(cfg, {"cfg" + cfg: rec}, "tcn", TAB, load=fake_load, seeds=SEEDS)


def test_honest_files_pass_check_for_flat_and_moe_and_params_recorded(made, monkeypatch):
    out, tmp = made
    monkeypatch.setattr(IO, "ROOT", tmp)
    monkeypatch.setattr(IO, "OUT", tmp / "out")
    monkeypatch.setattr(R2, "same_dates", lambda df: True)
    for cfg in out:
        assert chk(cfg, out[cfg][0], monkeypatch) == []
    assert out["4"][0]["params"] == FM.counts()["flat"] and out["6"][0]["params"] == FM.counts()["router"]["B"] + FM.counts()["experts"]
    assert out["4"][0]["grid"] == [[0.1, 0.0], [1.0, 0.0]] and out["6"][0]["fitted_share"] > 0.5 and set(out["6"][0]["lam_selected"]) <= {"0.1/0", "0.1/0.1", "1/0", "1/0.1"}


def test_check_writes_nothing(made, monkeypatch):
    out, tmp = made
    monkeypatch.setattr(IO, "ROOT", tmp)
    monkeypatch.setattr(IO, "OUT", tmp / "out")
    monkeypatch.setattr(R2, "same_dates", lambda df: True)
    boom = lambda *a, **k: (_ for _ in ()).throw(AssertionError("проверка пишет"))
    for obj, name in ((pd.DataFrame, "to_parquet"), (pd.DataFrame, "to_csv"), (IO, "merge_checkpoint"), (IO, "write_csv"), (np, "savez"), (np, "savez_compressed")):
        monkeypatch.setattr(obj, name, boom)
    before = IO.outputs_digest()
    assert chk("6", out["6"][0], monkeypatch) == [] and IO.outputs_digest() == before


def test_mutation_substituted_forecast_caught_by_sha(made, monkeypatch):
    out, tmp = made
    monkeypatch.setattr(IO, "ROOT", tmp)
    monkeypatch.setattr(IO, "OUT", tmp / "out")
    monkeypatch.setattr(R2, "same_dates", lambda df: True)
    rec, df = out["6"]
    bad = df.copy()
    bad.loc[len(bad) - 1, "G"] += 1e-9
    bad.to_parquet(R2.path("6"), index=False)
    try:
        assert any("sha256" in m for m in chk("6", rec, monkeypatch))
        bad["G"] = bad["G"] + 1e-9                                                                               # подмена внутри проверяемого блока, sha подогнан — ловит пересчёт переоценки
        bad.to_parquet(R2.path("6"), index=False)
        rec2 = dict(rec, sha256_content=CR.content_hash(bad), sha256_file=IO.sha_file(R2.path("6")))
        assert any("побитно" in m for m in chk("6", rec2, monkeypatch))
    finally:
        df.to_parquet(R2.path("6"), index=False)


def test_mutations_F_not_H_at_zero_G_dates_and_pi_simplex_caught(made, monkeypatch):
    out, tmp = made
    monkeypatch.setattr(IO, "ROOT", tmp)
    monkeypatch.setattr(IO, "OUT", tmp / "out")
    rec, df = out["6"]
    monkeypatch.setattr(R2, "same_dates", lambda d: False)
    assert any("даты" in m for m in chk("6", rec, monkeypatch))
    monkeypatch.setattr(R2, "same_dates", lambda d: True)
    try:
        bad = df.copy()
        bad.loc[bad.index[:3], "G"] = 0.0                                              # FX (версия 2): пары 2017 года с z — строк с G = 0 нет, подкладываем сами
        bad.loc[bad.index[:3], "F"] = bad.loc[bad.index[:3], "H"] * 1.01
        bad.to_parquet(R2.path("6"), index=False)
        assert any("F ≠ H" in m for m in R2.check_cfg("6", {"cfg6": dict(rec, sha256_content=CR.content_hash(bad), sha256_file=IO.sha_file(R2.path("6")))}, "tcn", TAB, load=fake_load, seeds=SEEDS))
        bad = df.copy()
        bad.loc[bad["fitted"] > 0, "pi1"] += 0.1
        bad.to_parquet(R2.path("6"), index=False)
        assert any("симплекс" in m for m in R2.check_cfg("6", {"cfg6": dict(rec, sha256_content=CR.content_hash(bad), sha256_file=IO.sha_file(R2.path("6")))}, "tcn", TAB, load=fake_load, seeds=SEEDS))
    finally:
        df.to_parquet(R2.path("6"), index=False)


def test_mutation_encoder_of_other_seed_changes_forecast_so_recount_catches(made, monkeypatch):
    """Ключ z = энкодер того же seed: подмена z у переоценки меняет прогноз, и побитная сверка это видит."""
    out, tmp = made
    monkeypatch.setattr(IO, "ROOT", tmp)
    monkeypatch.setattr(IO, "OUT", tmp / "out")
    monkeypatch.setattr(R2, "same_dates", lambda df: True)
    shifted = lambda p: dict(fake_load(p), z=fake_load(p)["z"] + 0.5)
    got = R2.check_cfg("6", {"cfg6": out["6"][0]}, "tcn", TAB, load=shifted, seeds=SEEDS)
    assert any("побитно" in m for m in got)
