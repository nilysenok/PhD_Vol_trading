"""E5: условие X5 — механическое, по таблицам E2–E4 на окне 2018–2021 (синтетика во временном каталоге + чтение реальных таблиц листов)."""
import json

import pandas as pd
import pytest

from models_v2.s15 import de0_io as IO, de_e5_stage as S5, de_reg as R, run_de


def _met(c, q18, q22):
    return pd.DataFrame([dict(series=c, window="2018_2021", qlike=q18), dict(series=c, window="2022_2025", qlike=q22)])


def _pair(c, rule18, rule22="—"):
    base = dict(x=c, y="V6:A0g", delta_pct=-1.0, lo90=-2.0, hi90=0.5, p_one=0.2, years_neg=1, n_years=4)
    return pd.DataFrame([dict(base, window="2018_2021", rule62=rule18), dict(base, window="2022_2025", rule62=rule22)])


def tables(q18, q22, rules):
    return ({c: _met(c, q18[c], q22[c]) for c in S5.CONF}, {c: _pair(c, rules[c], "да") for c in S5.CONF})


@pytest.fixture()
def world(tmp_path, monkeypatch):
    monkeypatch.setattr(IO, "REP", tmp_path)
    monkeypatch.setattr(R, "TRIALS", tmp_path / "ALL_TRIALS.csv")
    monkeypatch.setattr(IO, "CP_FILE", tmp_path / "CHECKPOINT.json")
    R.register("01.01.2020 10:00:00")
    return tmp_path


def _put(world, q18, rules):
    for i, c in enumerate(S5.CONF):
        _met(c, q18[c], 9.0).to_csv(world / ("E%d_METRICS.csv" % (i + 2)), index=False)
        _pair(c, rules[c]).to_csv(world / ("E%d_PAIRS.csv" % (i + 2)), index=False)


def test_best_by_2018_2021_and_rule_62_decide():
    m, p = tables({"X2": 0.17, "X3": 0.16, "X4": 0.165}, {"X2": 0.1, "X3": 0.3, "X4": 0.3}, {"X2": "да", "X3": "нет", "X4": "да"})
    c = S5.condition_from(m, p)
    assert c["best"] == "X3" and c["condition"] is False                  # лучший — X3 (у X2 правило «да», но он не лучший)
    m, p = tables({"X2": 0.17, "X3": 0.16, "X4": 0.165}, {"X2": 0.1, "X3": 0.3, "X4": 0.3}, {"X2": "нет", "X3": "да", "X4": "нет"})
    assert S5.condition_from(m, p)["condition"] is True


def test_mutation_condition_by_2022_plus_is_caught(monkeypatch):
    """Условие по 2022+ (лучший по 2022–2025 — X2): даёт другой ответ, чем честное окно 2018–2021 → тест на окно падает на мутанте."""
    m, p = tables({"X2": 0.17, "X3": 0.16, "X4": 0.165}, {"X2": 0.1, "X3": 0.3, "X4": 0.3}, {"X2": "да", "X3": "нет", "X4": "да"})
    honest = S5.condition_from(m, p)
    assert S5.WINDOW == "2018_2021" and honest["best"] == "X3" and honest["condition"] is False
    monkeypatch.setattr(S5, "WINDOW", "2022_2025")
    mutant = S5.condition_from(m, p)
    assert mutant["best"] == "X2" and mutant["condition"] != honest["condition"]      # мутант отличим: у него условие истинно из-за 2022+


def test_real_tables_condition_is_false_x3():
    """Реальные таблицы листов E2–E4: лучший — X3 (QLIKE 2018–2021), правило 6.2 против A0g «нет», условие ложно."""
    c = S5.compute()
    assert c["best"] == "X3" and c["rule62"] == "нет" and c["condition"] is False and c["qlike"]["X3"] < c["qlike"]["X4"] < c["qlike"]["X2"]


def test_missing_tables_refuse_with_condition_message(world):
    with pytest.raises(SystemExit, match="условие"):
        S5.compute()
    with pytest.raises(SystemExit, match="условие"):
        run_de.main(["--stage", "E5"])


def test_false_condition_writes_checkpoint_registry_and_runner_refuses_x5(world):
    _put(world, {"X2": 0.17, "X3": 0.16, "X4": 0.165}, {"X2": "да", "X3": "нет", "X4": "да"})
    S5.write()
    cp = json.loads((world / "CHECKPOINT.json").read_text(encoding="utf-8"))["E5"]
    assert cp["condition"] is False and cp["best"] == "X3" and cp["x5_computed"] is False
    reg = R.read()
    row = reg[(reg["id"] == "X5") & (reg["status"] == "не считается")]
    assert len(row) == 1 and "не считается: условие ложно" in row.iloc[0]["formula"]
    txt = R.TRIALS.read_text(encoding="utf-8")
    S5.write()
    assert R.TRIALS.read_text(encoding="utf-8") == txt                      # реестр только дополняется, повтор строки не добавляет
    with pytest.raises(SystemExit, match="условие"):
        R.guard("X5")
    assert S5.check() == []


def test_true_condition_is_not_served_and_nothing_written(world):
    _put(world, {"X2": 0.17, "X3": 0.16, "X4": 0.165}, {"X2": "нет", "X3": "да", "X4": "нет"})
    before = R.TRIALS.read_text(encoding="utf-8")
    with pytest.raises(SystemExit, match="истинно"):
        S5.write()
    assert R.TRIALS.read_text(encoding="utf-8") == before and not (world / "CHECKPOINT.json").exists()


def test_check_catches_tampered_checkpoint_registry_and_x5_shown_computed(world):
    _put(world, {"X2": 0.17, "X3": 0.16, "X4": 0.165}, {"X2": "да", "X3": "нет", "X4": "да"})
    S5.write()
    cp = json.loads((world / "CHECKPOINT.json").read_text(encoding="utf-8"))
    cp["E5"]["condition"] = True                                              # условие подменено
    (world / "CHECKPOINT.json").write_text(json.dumps(cp), encoding="utf-8")
    assert any("E5" in b for b in S5.check())
    cp["E5"]["condition"] = False
    (world / "CHECKPOINT.json").write_text(json.dumps(cp), encoding="utf-8")
    assert S5.check() == []
    R.TRIALS.write_text(R.TRIALS.read_text(encoding="utf-8").replace(",не считается,", ",посчитана,"), encoding="utf-8")      # X5 показана как посчитанная
    assert any("не считается" in b for b in S5.check())


def test_check_does_not_write(world):
    _put(world, {"X2": 0.17, "X3": 0.16, "X4": 0.165}, {"X2": "да", "X3": "нет", "X4": "да"})
    S5.write()
    h = (IO.sha_file(IO.CP_FILE), IO.sha_file(R.TRIALS))
    S5.check()
    assert (IO.sha_file(IO.CP_FILE), IO.sha_file(R.TRIALS)) == h
