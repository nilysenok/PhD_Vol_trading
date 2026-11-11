"""W8-B, проход W8-G на реальных рядах окна M (гейты B.G1, B.G2; mutations.md № 142, 164): 2198 дат; gap = 0 ⇒ F = X1 побитно; группа не включена ⇒ F = W5-G побитно; β ∈ [0; 0,5]; месяцы включения групп
напечатаны; сохранённые parquet = пересчёту; W8-Gb = тот же конвейер на ряде z, сдвинутом на 252; порча RV пар с t⁺ > D и баров IMOEX после d 18:30 не меняет β_{m,G} и прогнозы месяца; z_d — только бары 10:00…18:30.
Мутации: «закрыто + 1», метки по поздним порогам, бар 18:40 в z, сдвиг 0 — monkeypatch (№ 43). Реальные файлы только читаются."""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
import pytest

from models_v2.s16 import x1_w8_gap as G
from models_v2.s16 import x1_w8_gap_z as Z
from models_v2.s16 import x1_z_walk as ZW


@pytest.fixture(autouse=True)
def real_files_untouched():
    def st():
        s = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (G.REG.TRIALS, G.REG.CKPT) if p.exists()}
        s.update({str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(G.REG.REP.glob("W8_*")) if p.is_file()})
        s["W8"] = {str(p.name): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(G.OUT.rglob("*")) if p.is_file()} if G.OUT.exists() else {}
        return s
    before = st()
    yield
    assert st() == before


@pytest.fixture(scope="module")
def comp():
    return G.compute()


def test_series_bitwise_identities_bounds_dates_and_groups(comp) -> None:
    x, w5 = comp["x"], comp["w5"]
    allf = dict(comp["series"]) | {"shift_%02d" % j: f for j, f in comp["shifts"].items()}
    assert len(allf) == 2 + 19
    for k, f in allf.items():
        assert len(f) == 2198 and (f["d"].to_numpy() == x["d"].to_numpy()).all(), k
        off = ((f["gap"] == 0) | f["refit_date"].isna()).to_numpy()
        assert (f.loc[off, "F"].to_numpy() == f.loc[off, "X1"].to_numpy()).all() and (f.loc[off, "X1"].to_numpy() == x["F"].to_numpy()[off]).all(), k       # gap = 0 / до включения: F = X1 побитно
        inc = np.zeros(len(f), bool)
        for T in G.GROUPS:
            inc |= ((f["group"] == T) & (f["n_" + T] >= 30)).to_numpy()
        nz = ~off & ~inc
        assert nz.any() and (f.loc[nz, "F"].to_numpy() == w5.loc[nz, "F"].to_numpy()).all(), k                      # группа не включена ⇒ F = W5-G побитно
        assert f["beta"].between(0.0, 0.5).all() and G.check_series(f, w5, 2198, k) == [], k
        on = (f["gap"] == 1).to_numpy() & ~off | ((f["gap"] == 1) & f["refit_date"].notna()).to_numpy()
        assert (f.loc[on, "F"].to_numpy() == (f.loc[on, "X1"] * np.exp(f.loc[on, "beta"])).to_numpy()).all(), k      # F = X1·exp(β) в днях gap = 1
    g = comp["series"]["W8-G"]
    assert (g["gap"].to_numpy() == w5["gap"].to_numpy()).all() and (g["refit_date"].dropna().drop_duplicates().to_numpy() == w5["refit_date"].dropna().drop_duplicates().to_numpy()).all()   # включение как у W5-G
    inc = G.inclusion(g)
    assert set(inc) == {"T1", "T2", "T3"} and all(v == "нет" or v >= "2018-07-02" for v in inc.values())


def test_group_beta_changes_forecast_somewhere_and_w8gb_equals_roll_by_252(comp) -> None:
    g, gb, w5 = comp["series"]["W8-G"], comp["series"]["W8-Gb"], comp["w5"]
    assert not np.array_equal(g["F"], w5["F"]) and not np.array_equal(g["F"], gb["F"])           # группа действует; контроль (б) отличается
    x = comp["x"]
    w = comp["w5"]
    again = G._run(x, w, Z.shift_z(comp["z"], 252))
    assert again["F"].to_numpy().tobytes() == gb["F"].to_numpy().tobytes()                       # W8-Gb = конвейер на np.roll(z, 252)
    zz = comp["z"]
    assert (Z.shift_z(zz, 252)[252:][np.isfinite(zz[:-252])] == zz[:-252][np.isfinite(zz[:-252])]).all()
    shifts = [comp["shifts"][j]["F"].to_numpy().tobytes() for j in range(1, 20)]
    assert len(set(shifts)) == 19 and gb["F"].to_numpy().tobytes() not in shifts[1:]               # 19 разных сдвигов (252 + 100·(j − 1)); j = 1 — тот же 252


def test_shift_zero_mutation_makes_control_equal_main_and_is_caught(comp, monkeypatch) -> None:
    monkeypatch.setattr(Z, "SHIFT", 0)
    mut = G._run(comp["x"], comp["w5"], Z.shift_z(comp["z"], 0))
    assert mut["F"].to_numpy().tobytes() == comp["series"]["W8-G"]["F"].to_numpy().tobytes()      # сдвиг 0 → W8-Gb ≡ W8-G: именно это проверка У4/сверки обязана ловить
    assert comp["series"]["W8-Gb"]["F"].to_numpy().tobytes() != mut["F"].to_numpy().tobytes()


def test_check_gate_b1_reads_only_and_prints_inclusion(capsys) -> None:
    h0 = G.tree_hash()
    assert G.check() == [] and "месяцы включения групп" in capsys.readouterr().out and G.tree_hash() == h0
    assert len(h0) == 21 and G.OUTS == {"W8-G": "w8g.parquet", "W8-Gb": "w8gb.parquet"} and list(G.SHIFTS.values())[0] == "shift_01.parquet"


def test_check_catches_changed_parquet_in_a_copy(tmp_path) -> None:
    for p in G.OUT.glob("*.parquet"):
        (tmp_path / p.name).write_bytes(p.read_bytes())
    assert G.check(tmp_path) == []
    f = pd.read_parquet(tmp_path / "w8g.parquet")
    f.loc[f.index[-1], "F"] *= 1.0000001                                                           # подмена одного числа
    f.to_parquet(tmp_path / "w8g.parquet", index=False)
    assert any("w8g.parquet" in b for b in G.check(tmp_path))
    (tmp_path / "shift_05.parquet").unlink()
    assert any("shift_05" in b for b in G.check(tmp_path))                                         # нет файла сдвига


def test_write_refuses_to_overwrite_different_output(tmp_path, monkeypatch) -> None:
    f = pd.read_parquet(G.OUT / "w8g.parquet")
    f.loc[f.index[-1], "F"] *= 2.0
    f.to_parquet(tmp_path / "w8g.parquet", index=False)
    with pytest.raises(SystemExit, match="перезапись отказана"):
        G.write(tmp_path)                                                                          # существующий выход с иным содержимым — отказ
    assert pd.read_parquet(tmp_path / "w8g.parquet")["F"].iloc[-1] == f["F"].iloc[-1]


# ---------- B.G2: PIT на реальных рядах ----------

def test_check_pit_honest_is_clean_and_closed_plus_one_mutation_is_caught(monkeypatch) -> None:
    assert G.check_pit(step=6) == []                                                               # порча RV пар с t⁺ > D и баров после d 18:30 не меняет β_{m,G} и F до D
    monkeypatch.setattr(ZW, "closed_date", lambda tp, d: d)                                        # мутация № 164: «закрыто + 1»
    assert G.check_pit(step=6)


def test_check_pit_catches_late_threshold_labels_after_corrupt_bars_fix(monkeypatch) -> None:
    whole = lambda z: np.where(np.isfinite(z), np.where(z <= np.nanquantile(z, 1 / 3), "T1", np.where(z <= np.nanquantile(z, 2 / 3), "T2", "T3")), "na").astype(object)      # noqa: E731
    monkeypatch.setattr(G, "labels", whole)                                                        # мутация: метки пар пересчитаны по порогам всей выборки; теперь B.G2 это видит
    assert G.check_pit(step=6)


@pytest.fixture(scope="module")
def pit_inputs():
    x, w = G._inputs()
    return x, w, G.load_bars()


def strong(bars: pd.DataFrame, D) -> pd.DataFrame:
    """Порча, которая МЕНЯЕТ RV: close ×7 (open не трогаем) у всех баров дней > D и у баров после 18:30 дней ≤ D (независимая от `corrupt_bars` порча: у неё свой тест ниже)."""
    b = bars.copy()
    tod = b["begin"] - b["begin"].dt.normalize()
    m = (b["begin"].dt.normalize() > pd.Timestamp(D)) | (tod > pd.Timedelta(hours=18, minutes=30))
    b.loc[m, "close"] *= 7.0
    return b


def z_leaks(x, w, bars) -> bool:
    """Порча баров IMOEX после d 18:30 и баров дней > D не должна менять β_{m,G} и F строк d ≤ D."""
    cols = ["group", "beta", "F", "b_T1", "b_T2", "b_T3"]
    ref = G._run(x, w, G.z_series(x["d"], bars))
    for D in [x["d"][p] for p in ZW.month_starts(x["d"])[6::24]]:
        f = G._run(x, w, G.z_series(x["d"], strong(bars, D)))
        m = (x["d"] <= D).to_numpy()
        if not f.loc[m, cols].equals(ref.loc[m, cols]):
            return True
    return False


def test_strong_bar_corruption_honest_is_clean_and_mutations_142_and_late_labels_are_caught(pit_inputs, monkeypatch) -> None:
    x, w, bars = pit_inputs
    assert z_leaks(x, w, bars) is False
    whole = lambda z: np.where(np.isfinite(z), np.where(z <= np.nanquantile(z, 1 / 3), "T1", np.where(z <= np.nanquantile(z, 2 / 3), "T2", "T3")), "na").astype(object)      # noqa: E731
    monkeypatch.setattr(G, "labels", whole)                                                        # мутация: метки пар пересчитаны по порогам всей выборки (поздним)
    assert z_leaks(x, w, bars) is True
    monkeypatch.undo()
    monkeypatch.setattr(Z, "CUT_BAR", pd.Timedelta(hours=18, minutes=40))                          # мутация № 142: бар с началом 18:40 в z_d
    assert z_leaks(x, w, bars) is True
    monkeypatch.undo()
    real_rv = Z.day_rv
    monkeypatch.setattr(Z, "day_rv", lambda b: real_rv(b).shift(-1))                               # мутация № 142: в день d отдан RV дня d + 1
    assert z_leaks(x, w, bars) is True


def test_corrupt_bars_changes_rv_of_future_days() -> None:
    """Регрессия находки тестировщика (исправлено кодером 15:33): прежняя `corrupt_bars` умножала open и close ×7 одинаково — RV дня не менялась, и B.G2 по барам была пуста."""
    b = pd.concat([bars("2024-03-04", None), bars("2024-03-05", None)], ignore_index=True)
    d2 = pd.Timestamp("2024-03-05")
    assert not np.isclose(Z.day_rv(Z.corrupt_bars(b, "2024-03-04")).loc[d2], Z.day_rv(b).loc[d2], rtol=1e-6, atol=0)      # порча действует на RV будущего дня (сейчас — нет, с точностью до округления)


# ---------- z_d: известный ответ ----------

def bars(day: str, extra) -> pd.DataFrame:
    t = pd.date_range(day + " 10:00", day + " 18:30", freq="10min")
    c = 3000.0 * np.exp(0.0007 * np.sin(np.arange(len(t)) * 1.3))
    b = pd.DataFrame(dict(begin=t, open=np.r_[3000.0 * 0.996, c[:-1]], close=c))
    return pd.concat([b, pd.DataFrame(extra, columns=["begin", "open", "close"])], ignore_index=True) if extra else b


def test_day_rv_known_answer_session_window_and_z_nan_when_no_day() -> None:
    b = bars("2024-03-04", [(pd.Timestamp("2024-03-04 09:50"), 2000.0, 2500.0), (pd.Timestamp("2024-03-04 18:40"), 3000.0, 3300.0), (pd.Timestamp("2024-03-04 21:00"), 3300.0, 3600.0)])
    core = b[(b["begin"].dt.hour * 60 + b["begin"].dt.minute >= 600) & (b["begin"].dt.hour * 60 + b["begin"].dt.minute <= 18 * 60 + 30)].sort_values("begin")
    lr = np.r_[np.log(core["close"].iloc[0] / core["open"].iloc[0]), np.diff(np.log(core["close"].to_numpy()))]
    rv = Z.day_rv(b)
    assert list(rv.index) == [pd.Timestamp("2024-03-04")] and rv.iloc[0] == pytest.approx(float((lr ** 2).sum()), rel=1e-12)    # бары 09:50, 18:40, 21:00 не вошли
    z = Z.z_series(pd.to_datetime(["2024-03-04", "2024-03-05"]), b)
    assert z[0] == pytest.approx(np.log(rv.iloc[0]), rel=1e-12) and np.isnan(z[1])                 # дня нет → NaN
    c = Z.corrupt_bars(b, "2024-03-03")                                                            # порча всех баров дней > D и баров вне 10:00…18:30
    assert (c["close"] != b["close"]).all() and not np.isclose(Z.day_rv(c).iloc[0], rv.iloc[0], rtol=1e-3) and Z.day_rv(Z.corrupt_bars(b, "2024-03-04")).iloc[0] == rv.iloc[0]      # будущее испорчено так, что RV меняется; день D и бары в сессии не тронуты
    late_only = Z.corrupt_bars(b, "2024-03-04")
    assert (late_only["close"] != b["close"]).sum() == 2 and (late_only["close"][b["begin"].dt.hour * 60 + b["begin"].dt.minute <= 18 * 60 + 30] == b["close"][b["begin"].dt.hour * 60 + b["begin"].dt.minute <= 18 * 60 + 30]).all()


def test_day_rv_mutation_cut_bar_1840_changes_rv(monkeypatch) -> None:
    b = bars("2024-03-04", [(pd.Timestamp("2024-03-04 18:40"), 3000.0, 3300.0)])
    good = Z.day_rv(b).iloc[0]
    monkeypatch.setattr(Z, "CUT_BAR", pd.Timedelta(hours=18, minutes=40))
    assert Z.day_rv(b).iloc[0] > 100 * good                                                        # скачок 10 % в баре 18:40 вошёл бы в z_d
