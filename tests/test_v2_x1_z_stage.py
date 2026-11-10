"""Этап Z (Ж-246), стадия: порядок записи (прогнозы → `Z` → `Z.eval` без p → таблицы с p → отчёт → `Z.rules`), `--check` ТОЛЬКО читает и ловит подмену (число в таблице Z_PAIRS/Z_CRIT/отчёте, p в `Z.eval`, порядок ключей,
запись во время проверки, прогноз ≠ sha, реестр позже прогноза); модули листа не пишут в каталоги V6/V7/DE/YA/YB, в x1.parquet и в ZD; отказ для конфигураций сверх пяти. Запись и проверка идут во ВРЕМЕННЫЕ каталоги, реальные файлы не трогаются."""
from __future__ import annotations

import ast
import json
import shutil
from pathlib import Path

import pandas as pd
import pytest

import models_v2.s16  # noqa: F401
from models_v2.s16 import x1_z_stage as S, x1_z_walk as W


def set_env(mp, root: Path) -> None:
    mp.setattr(S, "REP", root / "rep"), mp.setattr(S, "OUT", root / "out"), mp.setattr(S, "CP_FILE", root / "rep/CHECKPOINT.json"), mp.setattr(S, "REPORT", root / "rep/Z_REPORT.md"), mp.setattr(S, "WATCH", [])


@pytest.fixture(scope="module")
def honest(tmp_path_factory):
    root, log = tmp_path_factory.mktemp("z"), []
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


def bad_of(tmp_path, honest, mp, mutate) -> list:
    root = with_env(tmp_path, honest[0], mp)
    mutate(root)
    return S.check()


def edit_cp(root, fn) -> None:
    p = root / "rep/CHECKPOINT.json"
    cp = json.loads(p.read_text(encoding="utf-8"))
    fn(cp)
    p.write_text(json.dumps(cp, ensure_ascii=False, sort_keys=True), encoding="utf-8")


def test_write_order_and_clean_check(honest, monkeypatch, tmp_path) -> None:
    root, log = honest
    assert [k for k, _ in log] == ["Z", "Z.eval", "Z.rules"]
    files = dict(log)
    assert [n for n in files["Z"] if n.endswith(".parquet")] == ["za.parquet", "zb.parquet", "zb0.parquet", "zc.parquet", "zcp.parquet"] and not [n for n in files["Z"] if n.startswith("Z_")]
    assert not [n for n in files["Z.eval"] if n.startswith("Z_") and n.endswith(".csv")]            # ключ Z.eval раньше ЛЮБОЙ таблицы с p
    assert {n for n in files["Z.rules"] if n.startswith("Z_")} >= {k + ".csv" for k in S.TABLES} and "Z_REPORT.md" in files["Z.rules"]
    cp = json.loads((root / "rep/CHECKPOINT.json").read_text(encoding="utf-8"))
    assert "p_holm" not in json.dumps(cp["Z.eval"]["numbers"]) and "p_one" not in json.dumps(cp["Z.eval"]["numbers"]) and "p_holm" in json.dumps(cp["Z.rules"]["rules"])
    assert cp["Z.eval"]["run"]["n_holm"] == 749 and cp["Z.eval"]["run"]["n_wide"] == 3203 and cp["Z.eval"]["run"]["configs_spent"] == 5
    with_env(tmp_path, root, monkeypatch)
    assert S.check() == []


def test_report_first_line_and_sections(honest) -> None:
    txt = (honest[0] / "rep/Z_REPORT.md").read_text(encoding="utf-8")
    assert txt.splitlines()[0].startswith("Статус: разработка, историческая пошаговая проверка, не независимая; ZB, ZC — после просмотра")
    for h in ("## 1. QLIKE", "## 4. Контрасты", "## 7. Состояния E", "## 9. Критерии K1–K5", "## 5. Правило 6.2"):
        assert h in txt
    crit = pd.read_csv(honest[0] / "rep/Z_CRIT.csv", comment="#")
    assert set(crit["y"]) == {"ZA", "ZB", "ZC"} and set(crit.loc[crit["crit"] == "итог", "name"]) <= {"продолжить", "закрыть"}


def test_check_catches_substituted_table_number_and_substituted_crit(honest, monkeypatch, tmp_path) -> None:
    def mut(name):
        def f(root):
            p = root / ("rep/%s.csv" % name)
            lines = p.read_text(encoding="utf-8").split("\n")
            lines[3] = lines[3].replace("0.", "0.9", 1) if "0." in lines[3] else lines[3] + "9"
            p.write_text("\n".join(lines), encoding="utf-8")
        return f
    assert any("Z_PAIRS.csv" in b for b in bad_of(tmp_path, honest, monkeypatch, mut("Z_PAIRS")))
    monkeypatch.undo()
    assert any("Z_CRIT.csv" in b for b in bad_of(tmp_path / "b", honest, monkeypatch, mut("Z_CRIT")))


def test_check_catches_substituted_number_in_report(honest, monkeypatch, tmp_path) -> None:
    def mut(root):
        p = root / "rep/Z_REPORT.md"
        t = p.read_text(encoding="utf-8")
        i = t.index("| ZA | 0.") + 9
        p.write_text(t[:i] + "9" + t[i + 1:], encoding="utf-8")
    assert any("Z_REPORT.md" in b for b in bad_of(tmp_path, honest, monkeypatch, mut))


def test_check_catches_p_in_eval_and_wrong_key_order(honest, monkeypatch, tmp_path) -> None:
    assert any("порядок/состав" in b for b in bad_of(tmp_path, honest, monkeypatch, lambda r: edit_cp(r, lambda cp: cp["Z.eval"]["numbers"].update(p_holm_N=0.01))))
    monkeypatch.undo()
    assert any("порядок/состав" in b for b in bad_of(tmp_path / "b", honest, monkeypatch, lambda r: edit_cp(r, lambda cp: cp["Z.eval"].update(written_at="31.12.2099 00:00:00"))))


def test_check_catches_forecast_substitution_and_late_registry(honest, monkeypatch, tmp_path) -> None:
    def forecast(root):
        p = root / "out/zb.parquet"
        d = pd.read_parquet(p)
        d.loc[d.index[-1], "F"] *= 1.01
        d.to_parquet(p, index=False)
    assert any("zb.parquet" in b for b in bad_of(tmp_path, honest, monkeypatch, forecast))
    monkeypatch.undo()
    assert any("позже прогноза" in b for b in bad_of(tmp_path / "b", honest, monkeypatch, lambda r: edit_cp(r, lambda cp: cp["Z"].update(computed_at="01.01.2026 00:00:00"))))


def test_check_that_writes_a_file_is_caught_by_hash(honest, monkeypatch, tmp_path) -> None:
    root = with_env(tmp_path, honest[0], monkeypatch)
    orig = S.compute
    monkeypatch.setattr(S, "WATCH", [root / "rep"])
    monkeypatch.setattr(S, "compute", lambda *a, **k: ((root / "rep/leak.txt").write_text("x"), orig(*a, **k))[1])        # мутация: проверка пишет файл
    assert any("изменил выходы" in b for b in S.check())


def test_hash_ignores_zd_files_and_keys_but_sees_other_files(honest, monkeypatch, tmp_path) -> None:
    root = with_env(tmp_path, honest[0], monkeypatch)
    h = S.rep_digest()
    (root / "rep/ZD_X.csv").write_text("x")
    edit_cp(root, lambda cp: cp.update({"ZD": {"a": 1}}))
    assert S.rep_digest() == h
    (root / "rep/YA_X.csv").write_text("x")
    assert S.rep_digest() != h


def write_calls(src: str) -> list:
    return [ast.unparse(n)[:80] for n in ast.walk(ast.parse(src)) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
            and (n.func.attr in ("write_text", "write_bytes", "to_parquet", "savez", "save", "dump") or (n.func.attr == "to_csv" and n.args) or (n.func.attr == "open" and "'w" in ast.dump(n)))]


def test_pure_modules_do_not_write_and_stage_writes_only_to_z_paths() -> None:
    d = Path(S.__file__).parent
    for name in ("feat", "fit", "walk", "eval", "crit", "doc", "synth"):
        assert write_calls((d / ("x1_z_%s.py" % name)).read_text(encoding="utf-8")) == []
    assert write_calls("import numpy\nnumpy.save('data/regen/harj_deep/E1/x1.parquet', 1)")                       # сканер видит подложенную запись
    for c in write_calls((d / "x1_z_stage.py").read_text(encoding="utf-8")):
        assert any(k in c for k in ("CP_FILE", "REP", "REPORT", "f.to_parquet")) and "harj_hybrid" not in c and "harj_deep" not in c and "ZD" not in c
    src = (d / "x1_z_reg.py").read_text(encoding="utf-8")
    assert "trials.open(" in src and S.ZR.TRIALS == S.REP / "ALL_TRIALS.csv" and S.OUT.parts[-2:] == ("harj_x1", "Z")


def test_extra_configuration_refused_and_compute_does_not_change_foreign_files() -> None:
    for st in ("Y5", "Y6"):
        with pytest.raises(SystemExit):
            S.main(["--stage", st])
    with pytest.raises(SystemExit):
        S.ZR.guard("Z6")
    assert W.KINDS == ("ZA", "ZB", "ZB0", "ZC", "ZCp") and S.ZR.IDS == W.KINDS
    before = S.SA.digest(S.WATCH + [S.YA_OUT, S.YB_OUT, S.SA.X1_FILE])
    S.compute()
    assert S.SA.digest(S.WATCH + [S.YA_OUT, S.YB_OUT, S.SA.X1_FILE]) == before
