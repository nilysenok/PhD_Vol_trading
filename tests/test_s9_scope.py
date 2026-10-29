"""С9 --scope test: замок и неизменность режима разработки (поручение владельца 26.10, 09:50).

1) Без docs/otchety/FINAL_DESIGN_strategies.md любой этап `python3 -m fastbt.s9 … --scope test`
   отказывает (fastbt.dev.TestLocked) до чтения данных; настройки модуля остаются dev.
2) При снятом замке (подменённый путь FINAL во временной папке) переключаются годы, срез σ,
   источники и суффикс; выбор B/C/D по С8 берётся из таблиц разработки и равен журналу final3.
3) Режим разработки на σ_pred final3 воспроизводит все 10 таблиц docs/tables/strat_dev/s9_*_final3.csv
   побайтно (cache + run, ~1–3 мин; пропуск, если нет общих данных или S9_REPRO=0).
python3 -m pytest tests/test_s9_scope.py -q
"""
import importlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fastbt import dev, s9

ROOT = Path(__file__).resolve().parent.parent
TABLES10 = ("s9_B_targetirovanie", "s9_C_rezhim", "s9_C3_tochnost_v_rezultat", "s9_D_zatvor",
            "s9_D_zatvor_harj", "s9_P1_vesa_tikerov", "s9_P2_multistrategiya", "s9_P3_fko",
            "s9_T1_komissii", "s9_T2_podperiody")


@pytest.fixture
def fresh_s9():
    """После теста — модуль С9 в исходном состоянии (dev, final2)."""
    yield s9
    importlib.reload(s9)


@pytest.fixture
def locked(monkeypatch, tmp_path):
    monkeypatch.setattr(dev, "FINAL", tmp_path / "FINAL_DESIGN_strategies.md")


@pytest.fixture
def unlocked(monkeypatch, tmp_path):
    f = tmp_path / "FINAL_DESIGN_strategies.md"
    f.write_text("тест замка\n")
    monkeypatch.setattr(dev, "FINAL", f)


def test_real_lock_closed():
    """В репозитории замок закрыт: FINAL_DESIGN_strategies.md нет."""
    if dev.FINAL.exists():
        pytest.skip("замок снят владельцем")
    with pytest.raises(dev.TestLocked):
        s9.main(["run", "--scope", "test"])


@pytest.mark.parametrize("stage", ["sources", "cache", "check", "run"])
def test_lock_refuses_every_stage(stage, locked, fresh_s9, monkeypatch):
    touched = []
    for fn in ("build_sources", "build_cache", "load_cache", "run_all_experiments", "load_source"):
        monkeypatch.setattr(s9, fn, lambda *a, _fn=fn, **k: touched.append(_fn))
    with pytest.raises(dev.TestLocked):
        s9.main([stage, "--scope", "test", "--sigma-h1", "final3"])
    assert touched == []                                   # данные не читались
    assert s9.SCOPE == "dev" and s9.YEARS == (2018, 2019, 2020, 2021)
    assert s9.CUT == pd.Timestamp("2021-12-31")


def test_set_scope_locked(locked, fresh_s9):
    with pytest.raises(dev.TestLocked):
        s9.set_scope("test")
    assert s9.SCOPE == "dev"


def test_lock_is_systemexit_for_cli(locked):
    r = subprocess.run([sys.executable, "-c",
                        "import sys; from fastbt import dev, s9; from pathlib import Path; "
                        "dev.FINAL = Path(sys.argv[1]); s9.main(['cache', '--scope', 'test'])",
                        str(dev.FINAL)], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode != 0 and "контрольный период закрыт" in r.stderr


def test_dev_scope_unchanged(fresh_s9):
    s9.set_scope("dev")
    s9.use_sigma_h1("final3")
    assert s9.SCOPE == "dev" and s9.YEARS == (2018, 2019, 2020, 2021) and s9.VAL_START == 2017
    assert s9.CUT == pd.Timestamp("2021-12-31") and s9.SUFFIX == "_final3"
    assert {"seed42_h1", "seed43_h1", "seed44_h1"} <= set(s9.SOURCES)
    assert s9.T2_CUT == "2020-03-01"


def test_unlocked_switches_scope(unlocked, fresh_s9):
    s9.set_scope("test")
    s9.use_sigma_h1("final3")
    assert s9.YEARS == (2022, 2023, 2024, 2025, 2026)
    assert (s9.VAL_START, s9.VAL_END) == (2018, 2021)      # окно заморожено на разработке
    assert s9.CUT == pd.Timestamp("2026-02-03")
    assert s9.SUFFIX == "_final3_test"
    assert not any(k.startswith("seed") for k in s9.SOURCES)
    assert s9.SOURCES["naive_h1"][0].name == "sigma_naive_rvw_test.parquet"
    assert s9.RVI.name == "rvi_1850_test.parquet"
    assert s9.SOURCES["hyb_h1"][0].name == "sigma_pred_h1_final3.parquet"
    assert s9.TABLES != s9.TABLES_DEV
    s9.use_sigma_h1("final3_full")                          # σ теста по решению 10:05, п. 6–8
    assert s9.SUFFIX == "_final3_test"
    assert s9.SOURCES["hyb_h1"][0].name == "sigma_pred_h1_final3_full.parquet"


def test_configs_vpred_final3_full():
    """п. 7: vpred во всех трёх fastbt/designs/final/*.yaml — final3_full, побитно = final3."""
    import yaml
    for f in sorted((ROOT / "fastbt" / "designs" / "final").glob("*.yaml")):
        v = Path(yaml.safe_load(f.read_text())["vpred"])
        assert v.name == "sigma_pred_h1_final3_full.parquet", f.name
    p = s9.REGEN / "sigma_pred_h1_final3_full.parquet"
    if p.exists():
        assert p.read_bytes() == (s9.REGEN / "sigma_pred_h1_final3.parquet").read_bytes()


ARHIV_N5 = ROOT / "docs" / "otchety" / "arhiv_do_audita" / "posle_N5"


def _frozen_choice(tables_dir, monkeypatch):
    monkeypatch.setattr(s9, "TABLES_DEV", tables_dir)
    s9.set_scope("test")
    s9.use_sigma_h1("final3")
    k = s9.frozen_keys()
    choice = dict(D=f"{k['D'][0]} на {k['D'][1]}", B=f"{k['B'][0]} на {k['B'][1]}",
                  C=f"{k['C'][0]} на {k['C'][1]}")
    return choice, (k["D1"], k["B1"])


def test_frozen_keys_equal_archive_choice(unlocked, fresh_s9, monkeypatch):
    """Ключи v1 (main.yaml, заморожены) = выбор по эталонам final3 того периода, которому они
    соответствовали, — архив posle_N5 (код 9b85bff, до #259). Правило владельца 26.10, 23:15."""
    if not (ARHIV_N5 / "s9_D_zatvor_final3.csv").exists():
        pytest.skip("нет архива posle_N5")
    choice, (d1, b1) = _frozen_choice(ARHIV_N5, monkeypatch)
    assert choice == {"D": "D6 на hyb_h1", "B": "cap1_band20 на hyb_h1",
                      "C": "C1:W504_p70 на hyb_h5"}
    assert (d1, b1) == ("D6", "cap1_band20")


ARHIV_DIV = s9.MAIN / "arhiv" / "final3_do_dividendov_2026-10-28"   # вне git; хэши — docs/otchety/2026-10-28_arhiv_final3_do_dividendov.md


def test_frozen_keys_equal_dev_choice(unlocked, fresh_s9, monkeypatch):
    """Замороженный выбор B/C/D (журнал заморозки run_log_final3.json, поле choice) = выбор по
    эталонам final3 ДО дивидендов (архив, #370): D6lag1 / cap1_band20 / TEST_C. Тест не перевыбирает
    по новым таблицам (см. test_dev_*_choice_* ниже)."""
    log = s9.OUT / "run_log_final3.json"
    if not log.exists() or not (ARHIV_DIV / "s9_D_zatvor_final3.csv").exists():
        pytest.skip("нет журнала заморозки или архива до дивидендов")
    frozen = json.loads(log.read_text())["choice"]
    assert frozen == {"D": "D6lag1 на hyb_h1", "B": "cap1_band20 на hyb_h1",
                      "C": "C1:W504_p70 на hyb_h5"}
    assert s9.TEST_C == ("C1:W504_p70", "hyb_h5")
    choice, (d1, b1) = _frozen_choice(ARHIV_DIV, monkeypatch)
    assert choice == frozen
    assert (d1, b1) == ("D6lag1", "cap1_band20")


def _record_select_keys(monkeypatch):
    """Обёртка select_keys: пишет выбор по таблицам разработки, не меняя его."""
    rec, orig = [], s9.select_keys
    monkeypatch.setattr(s9, "select_keys", lambda *a: rec.append(orig(*a)) or rec[-1])
    return rec


def _dev_choice(monkeypatch):
    if not (s9.TABLES_DEV / "s9_C_rezhim_final3.csv").exists():
        pytest.skip("нет таблиц разработки final3")
    log = s9.OUT / "run_log_final3.json"
    if not log.exists():
        pytest.skip("нет журнала заморозки")
    return _frozen_choice(s9.TABLES_DEV, monkeypatch)[0], json.loads(log.read_text())["choice"]


def test_dev_c_choice_equals_test_c(unlocked, fresh_s9, monkeypatch):
    """Ж-072: выбор C по таблицам разработки = TEST_C, защита frozen_keys не отказывает.
    Мутация «TEST_C ≠ выбор по таблицам» ловится: защита отказывает (SystemExit)."""
    choice, _ = _dev_choice(monkeypatch)
    assert choice["C"] == f"{s9.TEST_C[0]} на {s9.TEST_C[1]}"
    monkeypatch.setattr(s9, "TEST_C", ("C1:W252_p70", "hyb_h5"))
    with pytest.raises(SystemExit, match="C в тест задан владельцем"):
        _frozen_choice(s9.TABLES_DEV, monkeypatch)


def test_dev_b_choice_equals_freeze_log(unlocked, fresh_s9, monkeypatch):
    """Ж-072: выбор B по таблицам разработки = журнал заморозки."""
    choice, frozen = _dev_choice(monkeypatch)
    assert choice["B"] == frozen["B"]


def test_dev_d_choice_differs_from_freeze_log(unlocked, fresh_s9, monkeypatch):
    """Тест-факт: выбор D по таблицам после дивидендов ≠ журнал (D6lag1) — известно, вопрос
    владельцу о заморозке D в пакете 18:00 (Ж-072). Только имена ключей."""
    choice, frozen = _dev_choice(monkeypatch)
    assert frozen["D"] == "D6lag1 на hyb_h1" and choice["D"] != frozen["D"]


def test_split_c_one_main_point():
    """Владелец 10:00: в тесте основная строка C — одна точка C1 W504 p70 h=5, остальное — чувствительность."""
    p = s9.TABLES_DEV / "s9_C_rezhim_final3.csv"
    if not p.exists():
        pytest.skip("нет таблиц разработки final3")
    dfc = pd.read_csv(p, sep=";", decimal=",", encoding="utf-8-sig")
    main, sens = s9.split_c_test(dfc, s9.TEST_C)
    assert len(main) == 1 and len(sens) == len(dfc) - 1
    assert main.rol.iloc[0].startswith("основная")
    assert sens.rol.str.startswith("чувствительность").all()
    with pytest.raises(SystemExit):
        s9.split_c_test(dfc, ("C1:W999_p70", "hyb_h5"))


SMOKE = """
import sys
from pathlib import Path
import pandas as pd
from fastbt import dev, s9
tmp = Path(sys.argv[1])
(tmp / "FINAL_DESIGN_strategies.md").write_text("смоук\\n")
dev.FINAL = tmp / "FINAL_DESIGN_strategies.md"          # замок снят только в этом процессе
frozen = sys.argv[2] == "frozen"
# ни одного года ≥ 2022. expanding — окна как в разработке (сверка с ней);
# frozen — «псевдотест»: окна калибровки заморожены на 2017 (аналог 2018–2021 для теста 2022–2026)
dev.SCOPES["test"] = dict(dev.SCOPES["dev"], val_end=2017 if frozen else 0)
s9.TEST_YEARS, s9.TEST_CUT = s9.YEARS, s9.CUT
s9.TEST_VAL_START, s9.TEST_VAL_END = 2017, (2017 if frozen else 0)
s9.OUT, s9.TEST_TABLES = tmp, tmp
for st in (["sources"], ["cache", "--workers", "1"], ["run", "--boot-reps", "300"]):
    s9.main(st + ["--scope", "test"])                       # σ по умолчанию — final3_full
"""


def _as_dev(df: pd.DataFrame) -> pd.DataFrame:
    """Таблица ветки test → вид разработки: p_*_iid (IID, справочно) на место p_* (ЛВ), без rol."""
    df = df.copy()
    for c in [c for c in df.columns if c.endswith("_iid")]:
        df[c[:-4]] = df.pop(c)
    return df.drop(columns=[c for c in ("rol",) if c in df])


@pytest.mark.xfail(
    strict=True,
    reason="эталон предзащиты final3 зафиксирован до #678 (обратная дивидендная корректировка); воспроизведение — на коммите 5ac7815f; решение класса A у владельца",
)
@pytest.mark.skipif(os.environ.get("S9_REPRO") == "0", reason="S9_REPRO=0")
def test_test_path_on_dev_years(tmp_path):
    """Код ветки test (источники *_test, кэш с A без фильтра, замороженный выбор, ЛВ-бутстрап,
    одна точка C + чувствительность) на годах разработки 2018–2021 — ни одного года ≥ 2022.
    Таблицы с IID на месте p совпадают с прогоном разработки тех же 300 повторов."""
    if not (s9.REGEN / "sigma_pred_h1_final3_full.parquet").exists():
        pytest.skip("нет общих данных")
    env = dict(os.environ, NUMBA_CACHE_DIR=str(s9.MAIN / "data" / "fastbt" / "numba_cache"))
    r = subprocess.run([sys.executable, "-W", "ignore", "-c", SMOKE, str(tmp_path), "expanding"],
                       cwd=ROOT, env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    assert "(из разработки)" in r.stdout
    dv = tmp_path / "dev"
    for args in (["cache", "--workers", "3"], ["run", "--boot-reps", "300"]):
        r = subprocess.run([sys.executable, "-W", "ignore", "-m", "fastbt.s9", *args,
                            "--sigma-h1", "final3_full", "--suffix", "_final3pt"], cwd=ROOT,
                           env=dict(env, S9_TABLES=str(dv)), capture_output=True, text=True)
        assert r.returncode == 0, r.stderr[-2000:]
    rd = lambda p: pd.read_csv(p, sep=";", decimal=",", encoding="utf-8-sig", dtype=str)
    for t in TABLES10:
        if t in ("s9_C_rezhim", "s9_T2_podperiody"):
            continue
        a, b = _as_dev(rd(tmp_path / f"{t}_final3_test.csv")), rd(dv / f"{t}_final3pt.csv")
        pd.testing.assert_frame_equal(a[b.columns], b, obj=t)
    assert not (tmp_path / "s9_C_rezhim_final3_test.csv").exists()
    main = rd(tmp_path / "s9_C_osnovnaya_final3_test.csv")
    sens = rd(tmp_path / "s9_C_chuvstvitelnost_final3_test.csv")
    ref = rd(dv / "s9_C_rezhim_final3pt.csv")
    assert len(main) == 1 and len(main) + len(sens) == len(ref)
    assert (main.family.iloc[0], main.point.iloc[0], main.sigma.iloc[0]) == ("C1", "W504_p70", "hyb_h5")
    assert sens.rol.str.startswith("чувствительность").all()
    osn = rd(tmp_path / "s9_osnovnye_final3_test.csv")
    assert list(osn.approach) == ["A", "A без фильтра", "B", "C", "D"]
    assert osn.set_index("approach").loc["B", "note"].startswith("лучшая по С8")
    assert "p_vs_A_iid" in osn.columns
    for f in ("sigma_naive_rvw_test.parquet", "rvi_1850_test.parquet"):
        assert pd.read_parquet(tmp_path / f)["date"].max() <= pd.Timestamp("2021-12-31")


@pytest.mark.skipif(os.environ.get("S9_REPRO") == "0", reason="S9_REPRO=0")
def test_s9_pseudotest_frozen_windows(tmp_path):
    """Псевдотест С9 на 2018–2021: окна калибровки заморожены (2017), пустых окон нет, вывод
    теста полный; результаты отличаются от расширяющихся окон (заморозка действует)."""
    if not (s9.REGEN / "sigma_pred_h1_final3_full.parquet").exists():
        pytest.skip("нет общих данных")
    env = dict(os.environ, NUMBA_CACHE_DIR=str(s9.MAIN / "data" / "fastbt" / "numba_cache"))
    r = subprocess.run([sys.executable, "-W", "ignore", "-c", SMOKE, str(tmp_path), "frozen"],
                       cwd=ROOT, env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    rd = lambda p: pd.read_csv(p, sep=";", decimal=",", encoding="utf-8-sig")
    osn = rd(tmp_path / "s9_osnovnye_final3_test.csv")
    assert list(osn.approach) == ["A", "A без фильтра", "B", "C", "D"]
    assert osn["sharpe"].notna().all() and osn["p_vs_A"].iloc[2:].between(0, 1).all()
    assert len(rd(tmp_path / "s9_C_osnovnaya_final3_test.csv")) == 1
    ref = rd(ROOT / "docs" / "tables" / "strat_dev" / "s9_D_zatvor_final3.csv")
    assert not rd(tmp_path / "s9_D_zatvor_final3_test.csv")["sharpe"].equals(ref["sharpe"])


def test_config_val_bounds():
    """Config.val_end: 0 — окно до Y−1 (разработка, как прежде); 2021 — заморожено; пустое — ошибка."""
    from fastbt.config import Config
    from fastbt.data import year_bounds
    d = pd.bdate_range("2014-06-01", "2026-02-03").values.astype("datetime64[ns]")
    c0 = Config(val_start=2017)
    assert c0.val_bounds(d, 2020) == (year_bounds(d, 2017)[0], year_bounds(d, 2020)[0])
    c1 = Config(val_start=2018, val_end=2021)
    for Y in (2022, 2024, 2026):
        assert c1.val_bounds(d, Y) == (year_bounds(d, 2018)[0], year_bounds(d, 2022)[0])
        assert c1.cal_end_year(Y) == 2022
    with pytest.raises(ValueError):
        Config(val_start=2022, val_end=2021).val_bounds(d, 2022)
    from fastbt import dev
    t = dev.make_config({}, "test")
    assert (t.val_start, t.val_end, t.years_a[0], t.years_bcd[0]) == (2018, 2021, 2022, 2022)
    assert dev.make_config({}, "dev").val_end == 0


def test_chapter4_a_locked(locked):
    from fastbt import chapter4_a
    with pytest.raises(dev.TestLocked):
        chapter4_a.main(["--workers", "1"])
    assert chapter4_a.YEARS_A == (2022, 2023, 2024, 2025, 2026)
    assert "2020–2026" not in chapter4_a.WINDOWS


def test_final_test_locked(locked):
    from fastbt import final_test
    with pytest.raises(dev.TestLocked):
        final_test.main(["--main", "x.yaml", "--harj-vpred", "y.parquet"])
    assert list(final_test.windows("test")) == ["2022–2026", "2025–01.2026"]


def test_ce_boot_block_and_iid_unchanged():
    """ce_boot: block=1 — прежний IID (тот же ГСЧ); block=5 — круговой блочный."""
    from fastbt.final_test import ce, ce_boot
    rng = np.random.RandomState(3)
    a, b = rng.normal(3e-4, 0.01, 600), rng.normal(1e-4, 0.01, 600)
    obs, pv, lo, hi = ce_boot(a, b, 5, 400)
    r = np.random.RandomState(42)
    idx = r.randint(0, 600, (400, 600))
    d = (ce(a[idx], 5) - ce(b[idx], 5)) * 252 * 100
    assert pv == float(np.mean(d <= 0))
    o5, p5, _, _ = ce_boot(a, b, 5, 400, block=5)
    assert o5 == obs and 0 <= p5 <= 1 and p5 != pv


FT_PSEUDO = """
import sys
from pathlib import Path
from fastbt import cache, dev, final_test
tmp = Path(sys.argv[1])
cache.MAIN = tmp                                           # позиции псевдотеста — во временную папку
dev.SCOPES["dev"] = dict(years_a=tuple(range(2017, 2022)), years_bcd=tuple(range(2018, 2022)),
                         val_start=2017, val_end=2017)     # окна заморожены, ни одного года ≥ 2022
final_test.main(["--main", sys.argv[2], "--harj-vpred", sys.argv[3], "--scope", "dev",
                 "--boot-reps", "200", "--workers", "3"])
"""


@pytest.mark.skipif(os.environ.get("S9_REPRO") == "0", reason="S9_REPRO=0")
def test_final_test_pseudotest(tmp_path):
    """Движок final_test на 2018–2021 с замороженными окнами: ЛВ основной + IID рядом, без ошибок."""
    harj = s9.REGEN / "sigma_pred_h1_harj_final2.parquet"
    if not harj.exists():
        pytest.skip("нет общих данных")
    env = dict(os.environ, NUMBA_CACHE_DIR=str(s9.MAIN / "data" / "fastbt" / "numba_cache"))
    r = subprocess.run([sys.executable, "-W", "ignore", "-c", FT_PSEUDO, str(tmp_path),
                        str(ROOT / "fastbt" / "designs" / "final" / "main.yaml"), str(harj)],
                       cwd=ROOT, env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-3000:]
    assert "p_iid" in r.stdout and "p_ce_g3_iid" in r.stdout
    assert (tmp_path / "data" / "fastbt" / "final_dev_main.parquet").exists()


@pytest.mark.xfail(
    strict=True,
    reason="эталон предзащиты final3 зафиксирован до #678 (обратная дивидендная корректировка); воспроизведение — на коммите 5ac7815f; решение класса A у владельца",
)
@pytest.mark.skipif(os.environ.get("S9_REPRO") == "0", reason="S9_REPRO=0")
def test_dev_reproduces_final3_tables(tmp_path):
    """Разработка воспроизводит 10 из 10 эталонных таблиц `docs/tables/strat_dev/s9_*_final3.csv`.
    Эталоны пересчитаны на коде после #259 и #260 (ЭТ2, 27.10) на σ final3_full, комиссия 0,05%;
    прежние — docs/otchety/arhiv_do_audita/ (до аудита), posle_N1_N3/ (после Н1–Н3, PR #247) и
    posle_N5/ (код 9b85bff, после Н1–Н5; ЭТ), с SHA256.txt."""
    if not (s9.REGEN / "sigma_pred_h1_final3_full.parquet").exists():
        pytest.skip("нет общих данных")
    env = dict(os.environ, S9_TABLES=str(tmp_path),
               NUMBA_CACHE_DIR=str(s9.MAIN / "data" / "fastbt" / "numba_cache"))
    for args in (["cache", "--workers", "3"], ["run", "--boot-reps", "1000"]):
        r = subprocess.run([sys.executable, "-W", "ignore", "-m", "fastbt.s9", *args,
                            "--sigma-h1", "final3_full", "--suffix", "_final3pt"],
                           cwd=ROOT, env=env, capture_output=True, text=True)
        assert r.returncode == 0, r.stderr[-2000:]
    same = [t for t in TABLES10 if (tmp_path / f"{t}_final3pt.csv").read_bytes()
            == (ROOT / "docs" / "tables" / "strat_dev" / f"{t}_final3.csv").read_bytes()]
    assert len(same) == 10, sorted(set(TABLES10) - set(same))
