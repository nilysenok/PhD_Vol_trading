"""W4-B2, решение ступени 1, реестр до прогноза и «--check только читает» (B.G4, B.G5): решение выведено из CSV по 2.5 (известный ответ, границы, подмена числа в CSV);
прогон без строки W4-S1… в реестре отказывает до счёта (копия реестра через monkeypatch, по id, не по позиции — № 164); `check` не пишет выходы. Реальные файлы только читаются."""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from models_v2.s16 import x1_w4_comp_run as RUN
from models_v2.s16 import x1_w4_comp_stat as ST
from models_v2.s16 import x1_w4_reg as REG

ROOT = Path(REG.__file__).resolve().parents[2]
REP = ROOT / "docs/otchety/harj_x1"
REAL = [REP / "ALL_TRIALS.csv", REP / "CHECKPOINT.json", REP / "W4_STAGE1.csv", REP / "W4_STAGE1_REPORT.md", ROOT / "data/regen/harj_x1/W4/mplus_S1.parquet"]
ORIG_READ = REG._read


def real_state() -> dict:
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None for p in REAL}


@pytest.fixture(autouse=True)
def real_files_untouched():
    before = real_state()
    yield
    assert real_state() == before


def frame(m=(1.0, 1.0, 1.0), ka=(1.02, 1.02, 1.02), s1=(0.98, 0.99, 0.97), s2=(0.97, 1.0051, 0.96)) -> pd.DataFrame:
    rows = []
    for name, v in (("M", m), ("Ka", ka), ("S1", s1), ("S2", s2)):
        rows += [ST._row("window", name, k, n=100, qlike=q) for k, q in zip(("all", "h1", "h2"), v)]
    return pd.DataFrame(rows)[ST.COLS]


def test_decide_known_answer_s1_passes_s2_fails_on_half_excess() -> None:
    d = ST.decide(frame())
    assert d["passed"] == ["S1"] and d["chosen"] == "S1" and d["stage2"] == "считается"
    assert d["sets"]["S2"]["c1_better_than_M"] and d["sets"]["S2"]["c3_better_than_Ka"] and not d["sets"]["S2"]["c2_ok"]


def test_decide_boundaries_half_percent_and_both_pass_chooses_s2() -> None:
    assert ST.decide(frame(s2=(0.97, 1.004999, 0.96)))["passed"] == ["S1", "S2"]
    assert ST.decide(frame(s2=(0.97, 1.004999, 0.96)))["chosen"] == "S2"           # оба прошли — S2 (раздел 3)
    assert ST.decide(frame(s2=(0.97, 1.0, 1.005001)))["passed"] == ["S1"]          # превышение во второй половине > 0,5 %


def test_decide_each_condition_is_needed_and_none_means_stage2_not_computed() -> None:
    assert ST.decide(frame(s1=(1.0, 0.99, 0.97)))["sets"]["S1"]["c1_better_than_M"] is False        # Q = Q(M): не лучше
    assert ST.decide(frame(ka=(0.95, 1.02, 1.02)))["sets"]["S1"]["passed"] is False                 # не лучше K_a (Д-1)
    none = ST.decide(frame(s1=(1.01, 1.0, 1.0), s2=(1.02, 1.0, 1.0)))
    assert none["passed"] == [] and none["chosen"] is None and none["stage2"] == "не считается, условие ложно"


def test_decision_is_derived_from_csv_text_roundtrip_and_substituted_number_changes_it(tmp_path) -> None:
    df = frame()
    p = tmp_path / "W4_STAGE1.csv"
    p.write_text(ST.csv_text(df), encoding="utf-8")
    assert ST.decide(pd.read_csv(p)) == json.loads(json.dumps(ST.decide(df)))
    bad = df.copy()
    bad.loc[(bad["series"] == "S1") & (bad["key"] == "all"), "qlike"] = 1.5                          # подмена числа в CSV
    p.write_text(ST.csv_text(bad), encoding="utf-8")
    assert ST.decide(pd.read_csv(p))["passed"] == [] and ST.decide(pd.read_csv(p)) != ST.decide(df)


def _ckpt(tmp: Path, df: pd.DataFrame, monkeypatch) -> tuple:
    csv, ck = tmp / "W4_STAGE1.csv", tmp / "CHECKPOINT.json"
    ck.write_text(json.dumps({"other": {"k": 1}}, indent=1, sort_keys=True, ensure_ascii=False), encoding="utf-8")
    text = ST.csv_text(df)
    csv.write_text(text, encoding="utf-8")
    ST.write_stage1(ck, ST.decide(df), hashlib.sha256(text.encode()).hexdigest(), {})
    monkeypatch.setattr(RUN, "CSV", csv)
    monkeypatch.setattr(RUN, "CKPT", ck)
    return csv, ck


def test_check_decision_ok_and_tampered_csv_is_caught_both_ways(tmp_path, monkeypatch) -> None:
    df = frame()
    csv, ck = _ckpt(tmp_path, df, monkeypatch)
    assert RUN.check_decision_keys() == [] and json.loads(ck.read_text(encoding="utf-8"))["other"] == {"k": 1}
    bad = df.copy()
    bad.loc[(bad["series"] == "S1") & (bad["key"] == "all"), "qlike"] = 1.5
    text = ST.csv_text(bad)
    csv.write_text(text, encoding="utf-8")
    assert any("sha256" in b or "решение" in b for b in RUN.check_decision_keys())                  # подмена в CSV при старом ключе
    ck.write_text(json.dumps({"other": {"k": 1}}, indent=1, sort_keys=True, ensure_ascii=False), encoding="utf-8")   # (write_stage1 не умеет перезаписывать ключ — см. отчёт)
    ST.write_stage1(ck, ST.decide(df), hashlib.sha256(text.encode()).hexdigest(), {})                # подделан и sha в ключе, решение осталось старым
    assert "решение ≠ вывод из CSV" in RUN.check_decision_keys()


def test_check_decision_without_key_is_failure(tmp_path, monkeypatch) -> None:
    csv, ck = _ckpt(tmp_path, frame(), monkeypatch)
    ck.write_text(json.dumps({"other": 1}, indent=1, sort_keys=True), encoding="utf-8")
    assert RUN.check_decision_keys() == ["нет ключа W4.stage1"]


@pytest.mark.skipif(not (REP / "W4_STAGE1.csv").exists(), reason="выходы ступени 1 ещё не посчитаны")
def test_real_csv_decision_equals_key_in_checkpoint() -> None:
    assert RUN.check_decision_keys() == []
    d = ST.decide(pd.read_csv(REP / "W4_STAGE1.csv"))
    assert d["stage2"] == ("считается" if d["passed"] else "не считается, условие ложно")


def _registry(tmp: Path, drop=None, reverse=False) -> Path:
    lines = (REP / "ALL_TRIALS.csv").read_text(encoding="utf-8").splitlines()
    h = next(i for i, x in enumerate(lines) if x.startswith("id,"))                                  # строки до шапки — комментарий
    head, body = lines[: h + 1], [x for x in lines[h + 1:] if not (drop and x.startswith(drop + ","))]
    p = tmp / "ALL_TRIALS.csv"
    p.write_text("\n".join(head + (body[::-1] if reverse else body)) + "\n", encoding="utf-8")
    return p


def _use(monkeypatch, path: Path) -> None:
    monkeypatch.setattr(REG, "_read", lambda trials: ORIG_READ(path))


def test_registry_present_in_any_row_order_passes(tmp_path, monkeypatch) -> None:
    _use(monkeypatch, _registry(tmp_path, reverse=True))
    RUN.require_all()                                                                                 # по id, не по позиции


@pytest.mark.parametrize("missing", RUN.IDS)
def test_run_without_registered_row_refuses_before_any_count(tmp_path, monkeypatch, missing) -> None:
    _use(monkeypatch, _registry(tmp_path, drop=missing))
    monkeypatch.setattr(RUN, "compute", lambda sp: (_ for _ in ()).throw(AssertionError("счёт начат без реестра")))
    with pytest.raises(SystemExit, match="реестр до прогноза"):
        RUN.run()
    with pytest.raises(SystemExit, match=missing):
        RUN.require_all()


def _stub_check(monkeypatch, tmp: Path, writer=None) -> list:
    outs = [tmp / "mplus_S1.parquet", tmp / "W4_STAGE1.csv", tmp / "W4_STAGE1_REPORT.md"]
    for p in outs:
        p.write_text("CSV" if p.suffix == ".csv" else "MD" if p.suffix == ".md" else "PQ", encoding="utf-8")
    monkeypatch.setattr(RUN, "outputs", lambda: outs)
    monkeypatch.setattr(RUN, "CSV", outs[1])
    monkeypatch.setattr(RUN, "MD", outs[2])
    monkeypatch.setattr(RUN, "require_all", lambda: None)
    monkeypatch.setattr(RUN, "load_series", lambda: ({}, {}))
    for f in ("check_rho_zero", "check_dates", "check_first_refit", "check_decision_keys"):
        monkeypatch.setattr(RUN, f, lambda *a, **k: [])
    monkeypatch.setattr(RUN, "check_vs_fit_gbm", lambda: [])
    monkeypatch.setattr(RUN, "build_outputs", lambda: (None, {}, "MD"))
    monkeypatch.setattr(ST, "csv_text", lambda df: "CSV")
    monkeypatch.setattr(RUN, "compute", lambda sp: (writer(outs) if writer else None, {})[1])
    return outs


def test_check_reads_only_and_mutation_check_that_writes_is_caught(tmp_path, monkeypatch, capsys) -> None:
    outs = _stub_check(monkeypatch, tmp_path)
    before = RUN.sha(outs)
    assert RUN.check() == 0 and RUN.sha(outs) == before and "хэши выходов до = после True" in capsys.readouterr().out
    t2 = tmp_path / "w"
    t2.mkdir()
    outs = _stub_check(monkeypatch, t2, writer=lambda o: o[1].write_text("CSV изменён проверкой", encoding="utf-8"))   # мутация: --check пишет
    assert RUN.check() == 1 and "до = после False" in capsys.readouterr().out


def test_check_with_missing_outputs_refuses(tmp_path, monkeypatch) -> None:
    outs = _stub_check(monkeypatch, tmp_path)
    outs[0].unlink()
    with pytest.raises(SystemExit, match="нет выходов"):
        RUN.check()


def test_frame_roundtrip_has_no_nan_in_window_rows() -> None:
    df = pd.read_csv(io.StringIO(ST.csv_text(frame())))
    assert not df["qlike"].isna().any() and np.isfinite(df["qlike"]).all()
