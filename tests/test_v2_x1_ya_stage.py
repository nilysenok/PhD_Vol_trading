"""Y-A (Ж-243), стадия: порядок записи (прогнозы → `YA` → `YA.eval` без p → таблицы с p → отчёт → `YA.rules`), `--check` ТОЛЬКО читает и ловит подмену (число в таблице/отчёте, p в `YA.eval`, порядок ключей,
запись во время проверки, прогноз ≠ sha, реестр позже прогноза); модули листа не пишут в каталоги V6/V7/DE и в x1.parquet. Запись и проверка идут в ВРЕМЕННЫЕ каталоги (пути стадии подменяются), реальные файлы не трогаются."""
from __future__ import annotations

import ast
import json
import shutil
from pathlib import Path

import pandas as pd
import pytest

import models_v2.s16  # noqa: F401
from models_v2.s16 import x1_reg as R, x1_ya_stage as S


def set_env(mp, root: Path) -> None:
    mp.setattr(S, "REP", root / "rep"), mp.setattr(S, "OUT", root / "out"), mp.setattr(S, "CP_FILE", root / "rep/CHECKPOINT.json"), mp.setattr(S, "REPORT", root / "rep/YA_REPORT.md"), mp.setattr(S, "WATCH", [])


@pytest.fixture(scope="module")
def honest(tmp_path_factory):
    root, log = tmp_path_factory.mktemp("ya"), []
    mp = pytest.MonkeyPatch()
    set_env(mp, root)
    (root / "rep").mkdir()
    orig = S.merge_cp
    mp.setattr(S, "merge_cp", lambda k, v: (log.append((k, sorted(p.name for p in root.rglob("*") if p.is_file()))), orig(k, v))[1])
    S.write()
    mp.undo()
    return root, log


def with_env(tmp_path, honest_root, mp) -> Path:
    root = tmp_path / "w"
    shutil.copytree(honest_root, root)
    set_env(mp, root)
    return root


def test_write_order_and_clean_check(honest, monkeypatch, tmp_path) -> None:
    root, log = honest
    keys = [k for k, _ in log]
    assert keys == ["YA", "YA.eval", "YA.rules"]
    files = dict(log)
    assert [n for n in files["YA"] if n.endswith(".parquet")] == ["y1.parquet", "y2.parquet"] and not [n for n in files["YA"] if n.startswith("YA_")]
    assert not [n for n in files["YA.eval"] if n.startswith("YA_") and n.endswith(".csv")]            # ключ YA.eval раньше ЛЮБОЙ таблицы с p
    assert {n for n in files["YA.rules"] if n.startswith("YA_")} >= {k + ".csv" for k in S.TABLES} and "YA_REPORT.md" in files["YA.rules"]
    cp = json.loads((root / "rep/CHECKPOINT.json").read_text(encoding="utf-8"))
    assert "p_holm" not in json.dumps(cp["YA.eval"]["numbers"]) and "p_one" not in json.dumps(cp["YA.eval"]["numbers"]) and "p_holm" in json.dumps(cp["YA.rules"]["rules"])
    with_env(tmp_path, root, monkeypatch)
    assert S.check() == []


def bad_of(tmp_path, honest, mp, mutate) -> list:
    root = with_env(tmp_path, honest[0], mp)
    mutate(root)
    return S.check()


def test_check_catches_substituted_table_number(honest, monkeypatch, tmp_path) -> None:
    def mut(root):
        p = root / "rep/YA_PAIRS.csv"
        lines = p.read_text(encoding="utf-8").split("\n")
        lines[3] = lines[3].replace("0.", "0.9", 1)
        p.write_text("\n".join(lines), encoding="utf-8")
    assert any("YA_PAIRS.csv" in b for b in bad_of(tmp_path, honest, monkeypatch, mut))


def test_check_catches_substituted_number_in_report(honest, monkeypatch, tmp_path) -> None:
    def mut(root):
        p = root / "rep/YA_REPORT.md"
        t = p.read_text(encoding="utf-8")
        i = t.index("| Y1 | 0.") + 9
        p.write_text(t[:i] + "9" + t[i + 1:], encoding="utf-8")
    assert any("YA_REPORT.md" in b for b in bad_of(tmp_path, honest, monkeypatch, mut))


def test_check_catches_p_in_eval_and_wrong_key_order(honest, monkeypatch, tmp_path) -> None:
    def put_p(root):
        cp = json.loads((root / "rep/CHECKPOINT.json").read_text(encoding="utf-8"))
        cp["YA.eval"]["numbers"]["p_holm_N"] = 0.01
        (root / "rep/CHECKPOINT.json").write_text(json.dumps(cp, ensure_ascii=False, sort_keys=True), encoding="utf-8")

    def late_eval(root):
        cp = json.loads((root / "rep/CHECKPOINT.json").read_text(encoding="utf-8"))
        cp["YA.eval"]["written_at"] = "31.12.2099 00:00:00"
        (root / "rep/CHECKPOINT.json").write_text(json.dumps(cp, ensure_ascii=False, sort_keys=True), encoding="utf-8")

    assert any("порядок/состав" in b for b in bad_of(tmp_path, honest, monkeypatch, put_p))
    monkeypatch.undo()
    assert any("порядок/состав" in b for b in bad_of(tmp_path / "b", honest, monkeypatch, late_eval))


def test_check_catches_forecast_substitution_and_late_registry(honest, monkeypatch, tmp_path) -> None:
    def forecast(root):
        p = root / "out/y1.parquet"
        d = pd.read_parquet(p)
        d.loc[d.index[-1], "F"] *= 1.01
        d.to_parquet(p, index=False)

    def early_forecast(root):
        cp = json.loads((root / "rep/CHECKPOINT.json").read_text(encoding="utf-8"))
        cp["YA"]["computed_at"] = "01.01.2026 00:00:00"
        (root / "rep/CHECKPOINT.json").write_text(json.dumps(cp, ensure_ascii=False, sort_keys=True), encoding="utf-8")

    assert any("y1.parquet" in b for b in bad_of(tmp_path, honest, monkeypatch, forecast))
    monkeypatch.undo()
    assert any("позже прогноза" in b for b in bad_of(tmp_path / "b", honest, monkeypatch, early_forecast))


def test_check_that_writes_a_file_is_caught(honest, monkeypatch, tmp_path) -> None:
    root = with_env(tmp_path, honest[0], monkeypatch)
    orig = S.compute
    monkeypatch.setattr(S, "WATCH", [root / "rep"])
    monkeypatch.setattr(S, "compute", lambda *a, **k: ((root / "rep/leak.txt").write_text("x"), orig(*a, **k))[1])        # мутация: проверка пишет файл
    assert any("изменил выходы" in b for b in S.check())


def write_calls(src: str) -> list:
    out = []
    for n in ast.walk(ast.parse(src)):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute):
            if n.func.attr in ("write_text", "write_bytes", "to_parquet", "savez", "save", "dump") or (n.func.attr == "to_csv" and n.args) or (n.func.attr == "open" and "'w" in ast.dump(n)):
                out.append(ast.unparse(n)[:80])
    return out


def test_pure_modules_do_not_write_and_stage_writes_only_to_ya_paths() -> None:
    d = Path(S.__file__).parent
    for name in ("feat", "fit", "walk", "eval", "doc", "synth"):
        assert write_calls((d / ("x1_ya_%s.py" % name)).read_text(encoding="utf-8")) == []
    assert write_calls("import numpy\nnumpy.save('data/regen/harj_deep/E1/x1.parquet', 1)\np.write_text('x')")        # сканер видит подложенную запись
    for src in ((d / "x1_ya_stage.py").read_text(encoding="utf-8"), (d / "x1_reg.py").read_text(encoding="utf-8")):
        for c in write_calls(src):
            assert any(k in c for k in ("OUT", "REP", "CP_FILE", "REPORT", "trials", "f.to_parquet")) and "harj_hybrid" not in c and "harj_deep" not in c
    assert R.TRIALS.parent == S.REP and S.OUT.parts[-2:] == ("harj_x1", "YA")


def test_compute_does_not_change_v6_v7_de_or_x1_files() -> None:
    watch = [S.ROOT / p for p in ("data/regen/harj_hybrid_v6", "data/regen/harj_hybrid_v7", "data/regen/harj_deep", "docs/otchety/harj_hybrid_v6", "docs/otchety/harj_hybrid_v7", "docs/otchety/harj_deep")]
    before, x1 = S.digest(watch), S.sha_file(S.X1_FILE)
    S.compute()
    assert S.digest(watch) == before and S.sha_file(S.X1_FILE) == x1
