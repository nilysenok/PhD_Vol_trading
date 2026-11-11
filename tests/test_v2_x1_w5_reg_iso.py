"""W5-A, изоляция и «только читает» регистратора `x1_w5_reg` и аудита меток `x1_w5_gaps`: `check`/`check_gaps` ничего не пишут; реальные ALL_TRIALS.csv, CHECKPOINT.json, W5_GAPS.csv и каталог
`data/regen/harj_x1/W5` не тронуты тестами; в тестах W3/W4/W5 нет присваиваний атрибутам модулей (только monkeypatch, класс № 43); реестр W5 на реальном файле — всё или ничего, по id (№ 164);
аудит перерывов — известный ответ на синтетике и мутации (пятница → понедельник, остановка 2022, 66 пар, подмена строки файла). Реальные файлы только читаются."""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from models_v2.s16 import x1_w5_gaps as G
from models_v2.s16 import x1_w5_reg as W

ROOT = Path(W.__file__).resolve().parents[2]
TESTS = (sorted(ROOT.glob("tests/test_v2_x1_w3_*.py")) + sorted(ROOT.glob("tests/test_v2_x1_w4_reg*.py")) + sorted(ROOT.glob("tests/test_v2_x1_w5_*.py")))
REAL = (W.TRIALS, W.CKPT, W.GAPS)


def real_state() -> dict:
    s = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in REAL if p.exists()}
    s["W5"] = {str(p.relative_to(W.W5)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(W.W5.rglob("*")) if p.is_file()} if W.W5.exists() else {}
    return s


@pytest.fixture(autouse=True)
def real_files_untouched():
    before = real_state()
    yield
    assert real_state() == before                                                         # реальные реестр, CHECKPOINT, W5_GAPS.csv и каталог W5 прежние


def attr_assignments(src: str, mods: set) -> list:
    out = []
    for n in ast.walk(ast.parse(src)):
        for t in (n.targets if isinstance(n, ast.Assign) else [n.target] if isinstance(n, (ast.AugAssign, ast.AnnAssign)) else []):
            if isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name) and t.value.id in mods:
                out.append(ast.unparse(n))
    return out


def test_no_module_attribute_assignment_in_w3_w4_w5_tests_and_scanner_sees_one() -> None:
    assert len(TESTS) >= 8
    for p in TESTS:
        src = p.read_text(encoding="utf-8")
        mods = {n.asname or n.name for t in ast.walk(ast.parse(src)) if isinstance(t, ast.ImportFrom) and (t.module or "").startswith("models_v2") for n in t.names}
        assert attr_assignments(src, mods) == [], p.name
    assert attr_assignments("from models_v2.s16 import x1_w5_reg as W\nW.register = lambda: 0\n", {"W"})      # сканер видит подложенное присваивание


def test_swapped_functions_are_originals_after_other_tests() -> None:
    names = {f: getattr(W, f).__code__.co_name for f in ("register", "require", "check", "check_gaps", "merge_checkpoint", "_read", "_parquet")}
    assert all(k == v for k, v in names.items())                                          # подмены monkeypatch не остались в модуле
    from models_v2.s16 import x1_w5_diag as D
    assert D.compute.__code__.co_name == "compute" and D.build.__code__.co_name == "build" and D.check.__code__.co_name == "check"
    assert G.C.load.__code__.co_name == "load" and D.terc_past.__code__.co_name == "terc_past" and D._cal.__code__.co_name == "_cal"


def seed(t: Path) -> Path:
    t.write_text("id,name,formula,params,status,registered_at\nY1,a,b,c,зарегистрирована,10.11.2026 04:21:03\n", encoding="utf-8")
    return t


def test_check_only_reads(tmp_path) -> None:
    t, w5 = seed(tmp_path / "ALL_TRIALS.csv"), tmp_path / "W5"
    W.register("11.11.2026 15:00:00", trials=t, w5=w5, ckpt=None)
    snap = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in tmp_path.rglob("*") if p.is_file()}
    assert W.check(trials=t, w5=w5) == []
    assert snap == {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in tmp_path.rglob("*") if p.is_file()} and not w5.exists()


def test_check_that_writes_is_caught_by_hash(tmp_path, monkeypatch) -> None:
    t, w5 = seed(tmp_path / "ALL_TRIALS.csv"), tmp_path / "W5"
    W.register("11.11.2026 15:00:00", trials=t, w5=w5, ckpt=None)
    before = hashlib.sha256(t.read_bytes()).hexdigest()
    orig = W._parquet
    monkeypatch.setattr(W, "_parquet", lambda d: (t.write_text(t.read_text(encoding="utf-8") + "# изменено проверкой\n", encoding="utf-8"), orig(d))[1])      # мутация: check пишет
    W.check(trials=t, w5=w5)
    assert hashlib.sha256(t.read_bytes()).hexdigest() != before                           # внешний хэш выхода это видит


def write_calls(src: str) -> list:
    return [n.func.attr for n in ast.walk(ast.parse(src)) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr in ("write_text", "write_bytes", "open", "to_csv", "dump")]


def func_src(src: str, name: str) -> str:
    return next(ast.unparse(f) for f in ast.parse(src).body if isinstance(f, ast.FunctionDef) and f.name == name)


def test_readers_have_no_writes_and_scanner_sees_mutation() -> None:
    src = Path(W.__file__).read_text(encoding="utf-8")
    for f in ("check", "require", "_read", "_parquet", "check_gaps"):
        assert write_calls(func_src(src, f)) == [], f
    assert write_calls(func_src(src, "register")) and write_calls(func_src(src, "merge_checkpoint"))        # писатели — только эти две
    assert write_calls(func_src(src.replace("    cur = _read(trials)\n    rows", "    trials.write_text('x')\n    rows", 1), "check"))
    g = Path(G.__file__).read_text(encoding="utf-8")
    for f in ("odd_pairs", "check", "_data_lines", "to_csv_text"):
        assert [c for c in write_calls(func_src(g, f)) if c != "to_csv"] == [], f
    assert "write_text" in write_calls(func_src(g, "write"))                              # единственный писатель аудита — write (флаг --write)


# ---------- реальный реестр: всё или ничего, по id ----------

def test_real_registry_is_all_or_nothing_and_by_id() -> None:
    cur = W._read(W.TRIALS)
    mine = cur[cur["id"].isin(W.IDS)]
    assert len(mine) in (0, 28)                                                           # частичной регистрации нет
    other = cur[~cur["id"].isin(W.IDS)]
    assert not other["id"].str.match(r"^(W5D[0-9]+|W5-)").any()                           # чужих W5-строк (вне раздела 7) нет
    if len(mine) == 28:
        assert mine["registered_at"].nunique() == 1                                       # одна registered_at
        pq = W._parquet(W.W5)
        if pq:                                                                            # режим «после прогноза»: старый check() законно отказывает, регистрация раньше прогнозов
            assert any("parquet" in b for b in W.check())
            reg_at = pd.to_datetime(mine["registered_at"].iloc[0], format="%d.%m.%Y %H:%M:%S")
            first = min(pd.Timestamp.fromtimestamp((W.W5 / n).stat().st_mtime) for n in pq)
            assert reg_at < first
            if "after" in W.check.__code__.co_varnames[:W.check.__code__.co_argcount]:
                assert W.check(after=True) == []
        else:
            assert W.check() == []                                                        # A.G4: прогнозов нет
        assert len(cur[cur["id"].str.match(r"^(W5D[0-9]+|W5-)")]) == 28                  # A.G3
        cp = json.loads(W.CKPT.read_text(encoding="utf-8"))
        r = cp["W5.registered"]
        assert (r["n"], r["n_wide"], r["rows"], r["registered_at"]) == (798, 3301, 28, mine["registered_at"].iloc[0])
        assert cp["W5"]["forecasts_written"] == 0 and cp["W5"]["configurations"] == 0
    else:
        assert W.check() and not W._parquet(W.W5)                                         # не зарегистрировано, а прогнозов нет — отказывает check по строкам


# ---------- аудит метки gap (x1_w5_gaps) ----------

def test_real_gaps_file_equals_recount_66_pairs_and_every_pair_has_a_weekday_between() -> None:
    assert G.check() == [] and W.check_gaps() == []
    df = G.odd_pairs()
    assert len(df) == G.N_EXPECTED == 66
    assert not ((df["d_weekday"] == "Fri") & (df["delta_days"] == 2)).any()               # простые выходные не входят
    weekend_only = df[~df["nonworking_between"].apply(lambda s: any(not b.endswith(("Sat", "Sun")) for b in s.split("|")))]
    # НАХОДКА тестировщика: 5 из 66 — рабочая суббота → понедельник (между ними одно воскресенье), не праздник; спецификация (0.4, A.G4) называет все 66 плановыми праздниками.
    assert weekend_only["d"].astype(str).tolist() == ["2018-04-28", "2018-06-09", "2021-02-20", "2024-04-27", "2024-12-28"] and set(weekend_only["d_weekday"]) == {"Sat"}
    assert len(df) - len(weekend_only) == 61                                              # у остальных 61 есть будний нерабочий день между d и t⁺
    assert df["delta_days"].min() >= 1 and pd.Timestamp("2022-02-25").date() not in set(df["d"])


def x_of(pairs):
    return pd.DataFrame(dict(d=pd.to_datetime([a for a, _ in pairs]), t_plus=pd.to_datetime([b for _, b in pairs])))


def test_check_gaps_catches_tampered_missing_extra_and_stop_2022(tmp_path, monkeypatch) -> None:
    good = tmp_path / "W5_GAPS.csv"
    good.write_bytes(G.GAPS.read_bytes())
    assert G.check(good) == []
    lines = good.read_text(encoding="utf-8").splitlines()
    good.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")                       # убрана последняя пара
    assert any("не равны пересчёту" in b for b in G.check(good))
    good.write_text(G.GAPS.read_text(encoding="utf-8").replace("Wed", "Thu", 1), encoding="utf-8")        # подменён день недели в строке данных
    assert any("не равны пересчёту" in b for b in G.check(good))
    assert any("нет файла" in b for b in G.check(tmp_path / "missing.csv"))
    real_load = G.C.load
    monkeypatch.setattr(G.C, "load", lambda: pd.concat([real_load(), x_of([("2022-02-25", "2022-02-28")])], ignore_index=True))      # пара через остановку 2022 (мутация)
    assert any("2022-02-25" in b for b in G.check(G.GAPS))


def test_check_gaps_catches_wrong_count_and_friday_in_list(monkeypatch) -> None:
    few = x_of([("2024-01-09", "2024-01-10"), ("2024-02-21", "2024-02-26")])             # одна пара не по выходным вместо 66
    monkeypatch.setattr(G.C, "load", lambda: few)
    bad = G.check(G.GAPS)
    assert any("пар с перерывом не по выходным 1, ожидалось 66" in b for b in bad)
    assert G.odd_pairs(few)["d"].astype(str).tolist() == ["2024-02-21"]
