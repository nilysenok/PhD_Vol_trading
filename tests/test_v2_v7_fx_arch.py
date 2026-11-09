"""V7 лист FX, G34: архив версии 1 во временном каталоге — прогнозы перенесены (на старом месте пусто), таблицы скопированы, ключи `1b_v1`/`2_v1` с sha256, в `1b` остался Stage A/E0 без v1-ключей,
строки реестра добавлены (старые не удалены), повтор — без изменений. Мутации: подмена файла архива и пропавший файл ловит `check_archive`."""
from __future__ import annotations

import json

import pytest

from models_v2.s14 import v7_0_io as IO, v7_1m_reg as RG, v7_fx_arch as AR


@pytest.fixture()
def world(tmp_path, monkeypatch):
    rep, out = tmp_path / "rep", tmp_path / "out"
    for d in (rep, out / "1b", out / "2"):
        d.mkdir(parents=True)
    monkeypatch.setattr(IO, "REP", rep)
    monkeypatch.setattr(IO, "OUT", out)
    monkeypatch.setattr(AR, "ARCH", out / "archive_v1")
    monkeypatch.setattr(RG, "TRIALS", rep / "ALL_TRIALS.csv")
    RG.ensure(rep / "ALL_TRIALS.csv", "2026-11-09T00:00:00+03:00")
    cp = {"1b": {"stageA": {"n": 1}, "E0": {"x": 1}, "eval": {"v": 1}, "rules": {}, "choice3": {"choice": "tcn"}}, "2": {}}
    for i in "123":
        (out / "1b" / ("cfg%s.parquet" % i)).write_bytes(b"v1-" + i.encode())
        cp["1b"]["cfg" + i] = dict(sha256_content="c" + i, qlike_2018_2021=0.1, n_rows=5)
    for i in "4567":
        (out / "2" / ("cfg%s.parquet" % i)).write_bytes(b"v1-" + i.encode())
        cp["2"]["cfg" + i] = dict(sha256_content="c" + i, qlike_2018_2021=0.1, n_rows=5)
    (rep / "1b_METRICS.csv").write_text("a\n1\n")
    (rep / "2a_QLIKE.csv").write_text("b\n2\n")
    (rep / "LEAF1B_REPORT.md").write_text("r\n")
    (rep / "CHECKPOINT.json").write_text(json.dumps(cp))
    return rep, out


def test_archive_moves_forecasts_copies_tables_keeps_registry_rows_and_is_idempotent(world):
    rep, out = world
    n_before = len(RG.read())
    assert AR.archive() == "архивировано"
    assert not any(list((out / d).glob("cfg*.parquet")) for d in ("1b", "2")) and len(list((out / "archive_v1").rglob("cfg*.parquet"))) == 7
    assert (rep / "1b_METRICS.csv").exists() and (out / "archive_v1/otchety/1b_METRICS.csv").exists() and (out / "archive_v1/README.md").exists()
    cp = json.loads((rep / "CHECKPOINT.json").read_text())
    assert set(cp["1b"]) == {"stageA", "E0"} and cp["2"] == {} and "choice3" in cp["1b_v1"] and "cfg4" in cp["2_v1"] and "reason" in cp["2_v1"]
    rows = RG.read()
    assert len(rows) == n_before + 7 and sum(r[0].endswith(RG.ARCH_SUFFIX) for r in rows) == 7 and all(r[7] == RG.STATUS_ARCH for r in rows if r[0].endswith(RG.ARCH_SUFFIX))
    assert RG.bad() == [] and AR.check_archive(cp) == []
    assert AR.archive() == "уже архивировано" and len(RG.read()) == n_before + 7


def test_mutation_changed_or_missing_archive_file_is_caught(world):
    rep, out = world
    AR.archive()
    cp = json.loads((rep / "CHECKPOINT.json").read_text())
    (out / "archive_v1/1b/cfg1.parquet").write_bytes(b"changed")
    assert any("cfg1" in m for m in AR.check_archive(cp))
    (out / "archive_v1/2/cfg7.parquet").unlink()
    assert any("cfg7" in m for m in AR.check_archive(cp))


def test_registry_v2_row_is_valid_and_old_v1_row_is_kept(world):
    RG.add_result("1", dict(a=1))
    assert RG.add_result("1", dict(a=2), suffix=RG.V2_SUFFIX) == "дописана" and RG.add_result("1", dict(a=2), suffix=RG.V2_SUFFIX) == "уже есть"
    assert RG.bad() == [] and [r[0] for r in RG.read()[1:] if r[0].startswith("Ж-233:1:")] == ["Ж-233:1:результат", "Ж-233:1:результат_v2"]
