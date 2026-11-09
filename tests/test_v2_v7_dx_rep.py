"""V7-DX (G37–G41): правила вывода, отчёт из таблиц (подмена числа ловится), модули DX не пишут прогнозов и не трогают ключи 1b/2, стадия 2 пишет оценку, `--check` не пишет, реестр 2b."""
import ast
import json
from pathlib import Path

from models_v2.s14 import run_v7, v7_0_io as IO, v7_1m_reg as RG, v7_dx_rep as REP, v7_dx_run as R

S14 = Path(__file__).resolve().parents[1] / "models_v2" / "s14"
NUM = dict(enc_share_beats=0.5, sig_in={"1": 1.0, "4": 1.0, "6": 1.0}, expost_partial_r2_best=0.0, sweep_red={"0": 5.0, "0.01": 5.0, "0.1": 5.0, "1": 4.0}, pit_r2_dev=0.0, z_r2_on_controls=0.5)


def test_verdict_boundaries_a():
    v = lambda x: REP.verdicts(dict(NUM, enc_share_beats=x))["a"]
    assert (v(0.8), v(0.2), v(0.5)) == ("отброшена", "подтверждена", "не различимо")
    assert v(0.79) == "не различимо" and v(0.21) == "не различимо"


def test_verdict_boundaries_b_c_d():
    b = lambda r, ins=1.0: REP.verdicts(dict(NUM, expost_partial_r2_best=r, sig_in={"1": ins}))["b"]
    assert b(0.004, 0.0) == "подтверждена" and b(0.02) == "отброшена" and b(0.01) == "не различимо"
    assert b(0.004, 1.0).startswith("подтверждена (")                              # в выборке снижение есть — пометка
    c = lambda d1, d0: REP.verdicts(dict(NUM, sweep_red={"0": 5.0 + d0, "0.01": 5.0 + d1, "0.1": 5.0, "1": 4.0}))["c"]
    assert c(0.4, 0.9) == "отброшена" and c(0.6, 0.9) == "не различимо" and c(0.4, 1.1) == "не различимо" and c(3.0, 3.0) == "не различимо"
    d = lambda p, r: REP.verdicts(dict(NUM, pit_r2_dev=p, z_r2_on_controls=r))["d"]
    assert d(0.005, 0.3) == "подтверждена" and d(0.005, 0.29) == "не различимо" and d(0.01, 0.9) == "отброшена" and d(-0.2, 0.9) == "подтверждена"


def test_head_line_names_confirmed_first():
    s = REP.head_line(dict(a="отброшена", b="подтверждена (пометка)", c="не различимо", d="подтверждена"))
    assert s.index("подтверждена — ") < s.index("отброшена — ") < s.index("не различимо — ") and "(б)" in s and "(г)" in s


def _tables():
    return {k: R.roundtrip(IO.read_csv(R.name(k))) for k in R.KEYS}


def test_report_is_rendered_from_tables_and_substituted_number_is_caught():
    t = _tables()
    text = REP.render(t, R.K.STATUS)
    assert (IO.REP / R.REPORT).read_text(encoding="utf-8") == text
    assert text.splitlines()[4].startswith("Причина")                                # первая содержательная строка — причина
    mut = {k: v.copy() for k, v in t.items()}
    mut["ENC"].loc[:, "enc_beats_all_naive"] = False                                  # подмена: энкодер нигде не лучше наивных
    assert REP.render(mut, R.K.STATUS) != text and "подтверждена — (а)" in REP.render(mut, R.K.STATUS).splitlines()[4]
    mut2 = {k: v.copy() for k, v in t.items()}
    mut2["PIT"]["r2_oos"] = mut2["PIT"]["r2_oos"] + 0.5
    assert REP.render(mut2, R.K.STATUS) != text


def test_status_in_every_csv_header():
    for k in R.KEYS:
        assert (IO.REP / R.name(k)).read_text(encoding="utf-8").splitlines()[0].startswith("# диагностика, не отбор")


def calls_of(src: str):
    return [n for n in ast.walk(ast.parse(src)) if isinstance(n, ast.Call)]


def forbidden(src: str) -> list:
    """Запись прогнозов: to_parquet / np.save* / записи в CHECKPOINT не под ключом DX."""
    bad = []
    for c in calls_of(src):
        f = c.func.attr if isinstance(c.func, ast.Attribute) else getattr(c.func, "id", "")
        if f in ("to_parquet", "save", "savez", "savez_compressed"):
            bad.append(f)
        if f == "merge_checkpoint" and not (c.args and isinstance(c.args[0], ast.Constant) and c.args[0].value == "DX"):
            bad.append("merge_checkpoint не DX")
    return bad


def test_dx_modules_do_not_write_forecasts_or_other_checkpoint_keys():
    files = sorted(S14.glob("v7_dx_*.py"))
    assert len(files) >= 7
    for f in files:
        assert forbidden(f.read_text(encoding="utf-8")) == [], f.name
    assert forbidden("df.to_parquet('x')") == ["to_parquet"] and forbidden("IO.merge_checkpoint('2', {})") == ["merge_checkpoint не DX"]     # детектор ловит подложенное


def test_protected_hash_catches_change_in_1b_or_2():
    cp = json.loads((IO.REP / "CHECKPOINT.json").read_text(encoding="utf-8"))
    h = R.protected_hash(cp)
    cp["2"]["cfg4"]["sha256_file"] = "0" * 64
    assert R.protected_hash(cp) != h
    cp2 = json.loads((IO.REP / "CHECKPOINT.json").read_text(encoding="utf-8"))
    cp2["1b"]["cfg1"]["sha256_file"] = "0" * 64
    assert R.protected_hash(cp2) != h
    assert json.loads((IO.REP / "CHECKPOINT.json").read_text(encoding="utf-8"))["DX"]["protected_1b_2_sha256"] == h


def test_stage_2_write_calls_evrun_and_check_does_not(monkeypatch):
    from models_v2.s14 import v7_1m_stage as S1
    calls = []
    monkeypatch.setattr(S1, "run", lambda stage, check, config=None: 0)
    monkeypatch.setattr(R, "write_2b", lambda: calls.append("write") or 0)
    assert run_v7.main(["--stage", "2"]) == 0 and calls == ["write"]
    assert run_v7.main(["--stage", "2", "--check"]) == 0 and calls == ["write"]       # --check: запись не вызывается
    assert run_v7.main(["--stage", "2", "--config", "4"]) == 0 and calls == ["write"]
    monkeypatch.setattr(S1, "run", lambda stage, check, config=None: 2)
    assert run_v7.main(["--stage", "2"]) == 2 and calls == ["write"]                  # отказ записи стадии — оценка не пишется


def test_write_2b_calls_write_eval_then_rows(monkeypatch):
    from models_v2.s14 import v7_2_evrun as EVR
    order = []
    monkeypatch.setattr(EVR, "write_eval", lambda: order.append("eval") or 0)
    monkeypatch.setattr(R, "add_2b_rows", lambda cp, path=None: order.append("rows") or [])
    assert R.write_2b() == 0 and order == ["eval", "rows"]
    monkeypatch.setattr(EVR, "write_eval", lambda: order.append("eval2") or 1)
    assert R.write_2b() == 1 and order[-1] == "eval2"


def test_2b_rows_in_registry_idempotent_and_registration_rows_untouched(tmp_path):
    src = (IO.REP / "ALL_TRIALS.csv").read_text(encoding="utf-8")
    p = tmp_path / "ALL_TRIALS.csv"
    p.write_text("".join(l for l in src.splitlines(keepends=True) if ":результат_2b" not in l), encoding="utf-8")
    before = p.read_bytes()
    cp = json.loads((IO.REP / "CHECKPOINT.json").read_text(encoding="utf-8"))
    assert R.add_2b_rows(cp, p) == ["дописана"] * 7
    assert R.add_2b_rows(cp, p) == ["уже есть"] * 7
    assert p.read_bytes().startswith(before) and RG.bad(p) == []                        # прежние байты (регистрация, результаты) не тронуты; реестр цел
    rows = RG.read(p)[1:]
    assert sum(r[0].endswith(RG.R2B_SUFFIX) for r in rows) == 7 and len({r[0] for r in rows}) == len(rows)


def test_real_registry_has_2b_rows_and_is_clean():
    assert R.check_rows_2b(json.loads((IO.REP / "CHECKPOINT.json").read_text(encoding="utf-8"))) == []


def test_check_dx_passes_on_saved_tables_writes_nothing_and_catches_substitution(monkeypatch):
    """Быстрый вариант: пересчёт (`compute`) подменён таблицами из файлов; полный `--stage DX --check` — `test_v2_v7_dx_check.py` и команда."""
    cp = json.loads((IO.REP / "CHECKPOINT.json").read_text(encoding="utf-8"))
    before = IO.outputs_digest()
    monkeypatch.setattr(R, "compute", _tables)
    assert R.check_dx(cp) == [] and IO.outputs_digest() == before
    t = _tables()
    t["PIT"]["r2_oos"] = t["PIT"]["r2_oos"] + 0.5
    monkeypatch.setattr(R, "compute", lambda: t)
    assert any("DX_PIT.csv" in m for m in R.check_dx(cp)) and any("DX.numbers" in m or "verdicts" in m for m in R.check_dx(cp))
