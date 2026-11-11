"""W5-B, `x1_w5_gap` на реальных рядах окна M (W5_SPEC 3, 11.7; B.G1/B.G2): n = 2198; в днях gap^PIT = 0 и до включения F = X1 побитно; β ∈ [0; 0,5]; включение ≥ 2018-07-02 и ≥ 30 пар в каждой группе;
Ga — все дни, ±0,5; Gb — метка сдвинута на 252 (сдвиг 0 даёт W5-G побитно и ловится); GK2 над K2; `--check-pit`; `--check` только читает (хэши); реестр до прогноза (прогон без строки W5-G в копии реестра
отказывает); существующий выход с иным содержимым не перезаписывается. Выходы строятся только во временной папке; реальные parquet, реестр и CHECKPOINT не тронуты (autouse-хэши)."""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
import pytest

from models_v2.s16 import x1_w5_gap as G
from models_v2.s16 import x1_w5_gap_cal as CAL
from models_v2.s16 import x1_z_walk as ZW

REAL = (G.GAP_PIT, G.REG.TRIALS, G.REG.CKPT)


@pytest.fixture(autouse=True)
def real_files_untouched():
    def st():
        s = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None for p in REAL}
        s["W5"] = sorted((str(p), hashlib.sha256(p.read_bytes()).hexdigest()) for p in G.OUT.rglob("*") if p.is_file()) if G.OUT.exists() else []
        return s
    before = st()
    yield
    assert st() == before


R = G.compute()
SER, TBL = R["series"], R["tbl"]


def test_series_shapes_labels_and_bitwise_identity_off_gap_and_before_activation() -> None:
    assert len(R["x"]) == 2198 and R["x"]["t_plus"].min() == pd.Timestamp("2018-01-04") and R["x"]["t_plus"].max() == pd.Timestamp("2026-11-10")
    assert set(SER) == {"W5-G", "W5-Ga", "W5-Gb", "W5-GK2"} and all(len(f) == 2198 for f in SER.values())
    assert G.check_series(SER, 2198) == []
    g = SER["W5-G"]
    off = ((g["gap"] == 0) | g["refit_date"].isna()).to_numpy()
    assert off.sum() > 1500 and (g.loc[off, "F"].to_numpy() == R["x"]["F"].to_numpy()[off]).all()      # 11.7
    assert g["gap"].tolist() == TBL["gap_pit"].tolist()
    assert (SER["W5-Gb"]["gap"].to_numpy() == np.roll(TBL["gap_pit"].to_numpy(), 252)).all() and G.SHIFT == 252
    assert not g["F"].equals(SER["W5-Gb"]["F"])                                      # контроль (б) ≠ W5-G
    assert (SER["W5-GK2"]["K2"].to_numpy() == R["x"]["K2"].to_numpy()).all() and "X1" not in SER["W5-GK2"].columns
    for k in ("W5-G", "W5-Gb", "W5-GK2"):
        assert SER[k]["beta"].between(0, 0.5).all()
    assert SER["W5-Ga"]["beta"].between(-0.5, 0.5).all()
    ga = SER["W5-Ga"]
    assert (ga.loc[ga["refit_date"].notna(), "F"].to_numpy() == ga.loc[ga["refit_date"].notna(), "X1"].to_numpy() * np.exp(ga.loc[ga["refit_date"].notna(), "beta"].to_numpy())).all()   # Ga — все дни


def test_activation_not_before_2018_07_02_and_thirty_pairs_in_each_group_on_every_refit() -> None:
    g, x = SER["W5-G"], R["x"]
    refits = sorted(g["refit_date"].dropna().unique())
    assert pd.Timestamp(refits[0]) >= pd.Timestamp("2018-07-02") and g.loc[g["d"] < pd.Timestamp("2018-07-02"), "refit_date"].isna().all()
    gap = TBL["gap_pit"].to_numpy(int)
    for D in refits:
        pool = ZW.closed_pool(x["t_plus"], x["d"], D)
        assert (gap[pool] == 1).sum() >= 30 and (gap[pool] == 0).sum() >= 30
        assert (x["t_plus"].iloc[pool] <= D).all()                                    # пары только закрытые к D


def test_real_pit_check_and_closed_plus_one_mutation(monkeypatch) -> None:
    assert G.check_pit(step=6) == []                                                  # B.G2: порча RV пар с t⁺ > D не меняет β_m и прогнозы месяца
    monkeypatch.setattr(ZW, "closed_date", lambda tp, d: d)                          # мутация № 164: «закрыто + 1»
    assert any("порча RV" in b for b in G.check_pit(step=6))


def test_zero_shift_control_and_gap_zero_correction_mutations_are_visible_through_compute(monkeypatch) -> None:
    orig_shift, orig_walk = G.shift_labels, G.walk
    monkeypatch.setattr(G, "shift_labels", lambda gap, shift=0: orig_shift(gap, 0))
    m = G.compute()["series"]
    assert m["W5-Gb"]["F"].to_numpy().tobytes() == m["W5-G"]["F"].to_numpy().tobytes()       # W5-Gb = W5-G побитно: тест «контроль ≠ W5-G» выше на этой мутации падает
    assert not SER["W5-Gb"]["F"].equals(SER["W5-G"]["F"])
    monkeypatch.setattr(G, "shift_labels", orig_shift)
    monkeypatch.setattr(G, "walk", lambda d, tp, rv, base, gap, beta_fn=G.beta_gap, all_days=False, pool_fn=None, base_name="X1": orig_walk(d, tp, rv, base, gap, beta_fn, True, pool_fn, base_name))
    assert G.check_series(G.compute()["series"], 2198)                                # мутация «поправка в днях gap = 0»: B.G1 ловит


def build(tmp):
    G.write(tmp / "W5", tmp / "W5_GAP_PIT.csv")
    return tmp / "W5", tmp / "W5_GAP_PIT.csv"


def test_write_then_check_ok_only_reads_and_equals_in_memory(tmp_path, capsys) -> None:
    out, gp = build(tmp_path)
    assert sorted(p.name for p in out.glob("*.parquet")) == sorted(G.OUTS.values())
    h = G.tree_hash(out, gp)
    assert G.check(out, gp) == [] and G.tree_hash(out, gp) == h                       # B.G1: только чтение
    assert "месяц включения 2018-" in capsys.readouterr().out
    pd.testing.assert_frame_equal(pd.read_parquet(out / "w5g.parquet"), SER["W5-G"], check_exact=True)
    assert gp.read_text(encoding="utf-8").splitlines()[:3] == list(CAL.MARKS)         # пометки 11.5 в шапке W5_GAP_PIT.csv


def test_check_that_writes_is_caught_by_hash_and_missing_or_tampered_output_is_reported(tmp_path, monkeypatch) -> None:
    out, gp = build(tmp_path)
    orig = G.compute
    monkeypatch.setattr(G, "compute", lambda: (out.joinpath("extra.parquet").write_bytes(b"x"), orig())[1])     # мутация № 123: check пишет
    assert any("изменил выходы" in b for b in G.check(out, gp))
    monkeypatch.setattr(G, "compute", orig)
    (out / "extra.parquet").unlink()
    t = pd.read_parquet(out / "w5g.parquet")
    t.loc[t.index[-1], "F"] *= 1.0000001
    t.to_parquet(out / "w5g.parquet", index=False)
    assert any("w5g.parquet" in b for b in G.check(out, gp))
    (out / "w5ga.parquet").unlink()
    assert any("w5ga.parquet" in b for b in G.check(out, gp))
    gp.write_text(gp.read_text(encoding="utf-8").replace(",1,", ",0,", 1), encoding="utf-8")
    assert any("W5_GAP_PIT.csv" in b for b in G.check(out, gp))


def test_write_refuses_to_overwrite_different_output_and_is_idempotent(tmp_path) -> None:
    out, gp = build(tmp_path)
    before = (out / "w5g.parquet").read_bytes()
    G.write(out, gp)                                                                  # то же содержимое — можно
    t = pd.read_parquet(out / "w5g.parquet")
    t["beta"] = 0.123
    t.to_parquet(out / "w5g.parquet", index=False)
    changed = (out / "w5g.parquet").read_bytes()
    with pytest.raises(SystemExit):
        G.write(out, gp)
    assert (out / "w5g.parquet").read_bytes() == changed != before


def test_registry_before_forecast_compute_refuses_without_w5g_row_in_registry_copy(tmp_path, monkeypatch) -> None:
    cur = G.REG._read(G.REG.TRIALS)
    ids = [i for i in cur["id"] if i != "W5-G"]
    copy = tmp_path / "ALL_TRIALS.csv"
    copy.write_text(G.REG.TRIALS.read_text(encoding="utf-8").replace("W5-G,", "W5-Gx,", 1), encoding="utf-8")   # копия реестра без строки с id W5-G
    assert "W5-G" not in set(G.REG._read(copy)["id"]) and len(ids) == len(cur) - 1
    orig = G.REG.require
    monkeypatch.setattr(G.REG, "require", lambda k, trials=copy: orig(k, trials=trials))
    with pytest.raises(SystemExit):
        G.compute()
    with pytest.raises(SystemExit):
        G.write(tmp_path / "W5", tmp_path / "g.csv")
    assert not (tmp_path / "W5").exists() or not list((tmp_path / "W5").glob("*.parquet"))              # ничего не записано
