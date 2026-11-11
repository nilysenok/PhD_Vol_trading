"""W5-B, оценка «от реальных рядов до файлов» во временной папке (W5_SPEC 5, 6, 11.5–11.6; B.G3): ряды и таблицы строит боевой код в tmp; `--check` только читает (хэши до и после); пометки 11.5 в шапке каждого CSV;
Σ вкладов = Δ из файла (1e-12), выброшенная строка ломает тождество; Холм N = 798 / 3301 в файле; решение выводится из `W5_GUARD.csv`: подмена числа ловится `--check`, подмена флага меняет решение; пороги
У1–У3 в файле равны 0,415 / 0,50 / 0,26, мутация «порог из результата» ловится; GK2 критериев не получает; ключ `W5.eval` пишется до таблиц с p и его отсутствие ловится; реестр до прогноза. Реальные выходы
только читаются (autouse-хэши), тест реальных файлов — в конце."""
from __future__ import annotations

import hashlib
import json
import shutil

import numpy as np
import pandas as pd
import pytest

from models_v2.s16 import x1_w5_eval as E
from models_v2.s16 import x1_w5_eval_t as T
from models_v2.s16 import x1_w5_gap as G
from models_v2.s16 import x1_w5_gap_cal as CAL

MARKS = ["# состояние gap выбрано из 24 после анализа истории окна M", "# способ оценки β выбран после просмотра 24.02.2022 (Д-2)",
         "# исследовательский результат на раскрытой истории, не новое независимое подтверждение (Ж-250 п. 6)"]
REAL = [E.REP / n for n in E.OUTS] + [E.REP / "W5_REPORT.md", E.CKPT, G.REG.TRIALS, G.GAP_PIT]


def real_state() -> dict:
    s = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None for p in REAL}
    s["W5"] = sorted((str(p), hashlib.sha256(p.read_bytes()).hexdigest()) for p in G.OUT.rglob("*") if p.is_file()) if G.OUT.exists() else []
    return s


@pytest.fixture(autouse=True)
def real_files_untouched():
    before = real_state()
    yield
    assert real_state() == before


def build(tmp):
    out, gp, ck = tmp / "W5", tmp / "W5_GAP_PIT.csv", tmp / "CHECKPOINT.json"
    G.write(out, gp)
    shutil.copy(E.CKPT, ck)
    t = E.write(tmp, ck, out, gp)
    return dict(rep=tmp, out=out, gp=gp, ck=ck, t=t)


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    return build(tmp_path_factory.mktemp("w5e"))


def chk(e) -> list:
    return E.check(e["rep"], e["ck"], e["out"], e["gp"])


def csv(e, name) -> pd.DataFrame:
    return pd.read_csv(e["rep"] / name, comment="#")


def derived_ok(g: pd.DataFrame, d: str) -> bool:
    """Решение d согласовано с флагами `passed` таблицы (формат чисел при чтении CSV другой: сравниваем по смыслу — начало и перечень невыполненных id)."""
    c = g[~g["id"].isin(["справка", "решение"])]
    bad = c[~c["passed"].astype(bool)]["id"].tolist()
    if not bad:
        return d.startswith("новая контрольная точка")
    return d.startswith("X1 остаётся контрольной точкой") and all(i in d for i in bad) and not any(i in d.split("не прошёл:")[1] for i in c["id"] if i not in bad)


def snap(e) -> dict:
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(e["rep"].rglob("*")) if p.is_file()}


def test_check_ok_only_reads_and_every_csv_has_three_marks_from_spec(env) -> None:
    h = snap(env)
    assert chk(env) == [] and snap(env) == h
    for n in E.OUTS:
        assert (env["rep"] / n).read_text(encoding="utf-8").splitlines()[:3] == MARKS, n


def test_check_that_writes_is_caught(env, monkeypatch) -> None:
    orig, p = E.compute, env["rep"] / "W5_STATS.csv"
    keep = p.read_bytes()
    monkeypatch.setattr(E, "compute", lambda inp: (p.write_bytes(keep + "# изменено проверкой\n".encode("utf-8")), orig(inp))[1])      # мутация № 123
    try:
        assert any("изменил выходы" in b or "текст ≠ пересчёту" in b for b in chk(env))
    finally:
        p.write_bytes(keep)
    monkeypatch.setattr(E, "compute", orig)
    assert chk(env) == []


def mutated(env, name, old, new, count=1):
    p = env["rep"] / name
    keep = p.read_text(encoding="utf-8")
    assert old in keep, (name, old)
    p.write_text(keep.replace(old, new, count), encoding="utf-8")
    try:
        return chk(env)
    finally:
        p.write_text(keep, encoding="utf-8")


def test_identity_from_file_and_dropped_row_and_holm_n_in_file(env) -> None:
    y = csv(env, "W5_YEARS.csv")
    assert T.identity_gap(y) < 1e-12                                                  # Σ вкладов = Δ из сохранённого файла
    k = y[(y["series"] == "W5-G") & (y["base"] == "X1") & (y["kind"] == "period")].index[2]
    assert T.identity_gap(y.drop(index=k)) > 1e-9                                     # выброшенная строка (дата/период) ломает тождество
    s = csv(env, "W5_STATS.csv")
    m = s[s["kind"] == "main"]
    assert m["base"].tolist() == ["X1", "K2", "K3"] and (m["n_holm"] == 798).all() and (m["n_holm_wide"] == 3301).all()
    first = s[s["kind"] == "main"].iloc[0]
    assert any("W5_STATS.csv" in b for b in mutated(env, "W5_STATS.csv", "798", "3"))                  # N = числу строк: файл ≠ пересчёту
    assert first["delta_pct"] != 0 and any("W5_YEARS.csv" in b for b in mutated(env, "W5_YEARS.csv", "W5-G,X1,period", "W5-G,X1,half"))


def test_gk2_and_references_get_no_criteria(env) -> None:
    s, g = csv(env, "W5_STATS.csv"), csv(env, "W5_GUARD.csv")
    ref = s[s["kind"] == "reference"]
    assert set(ref["hybrid"] + ">" + ref["base"]) == {"W5-G>M", "W5-G>W5-Ga", "W5-G>W5-Gb", "W5-GK2>K2"}
    assert ref["p_holm_N"].isna().all() and (ref["verdict"] == "справочно, критериев не получает").all()
    assert not g["id"].isin(["GK2", "W5-GK2"]).any() and not g["condition"].str.contains("GK2").any()


def test_decision_is_derived_from_guard_file_and_threshold_mutation_is_caught(env, monkeypatch) -> None:
    g = csv(env, "W5_GUARD.csv")
    d = g[g["id"] == "решение"]["value"].iloc[0]
    assert d == env["t"]["decision"] and derived_ok(g, d)                             # решение = функция строк таблицы (флагов passed)
    thr = g.set_index("id")["threshold"]
    assert (float(thr["У1"]), float(thr["У2"]), float(thr["У3"])) == (0.415, 0.50, 0.26)      # константы спецификации, не данные
    one = g[g["id"].isin(["основной", "У1", "У2", "У3", "У4", "У5"])].index[0]
    flip = g.copy()
    flip.loc[one, "passed"] = not bool(flip.loc[one, "passed"])
    assert not derived_ok(flip, d)                                                    # подмена флага в таблице меняет решение: записанное решение перестаёт ей соответствовать
    val = g.loc[g["id"] == "У1", "value"].iloc[0]
    assert any("W5_GUARD.csv" in b for b in mutated(env, "W5_GUARD.csv", ",%s," % val, ",%s," % (float(val) + 0.5)))          # подмена числа ловится --check
    monkeypatch.setattr(T, "THR", {"У1": float(val) + 1.0, "У2": 9.0, "У3": 9.0})        # мутация: порог взят из результата
    assert any("W5_GUARD.csv" in b for b in chk(env))


def test_eval_key_is_written_before_tables_and_missing_key_or_wrong_hash_is_caught(tmp_path, monkeypatch) -> None:
    e2 = dict(rep=tmp_path, out=tmp_path / "W5", gp=tmp_path / "W5_GAP_PIT.csv", ck=tmp_path / "CHECKPOINT.json")
    G.write(e2["out"], e2["gp"])
    shutil.copy(E.CKPT, e2["ck"])
    before = json.loads(e2["ck"].read_text(encoding="utf-8"))
    orig = E.merge_checkpoint
    seen = []

    def spy(ckpt, t, tx, out=E.OUT, gap_path=E.GAP_PIT):
        seen.append([n for n in E.OUTS if (tmp_path / n).exists()])
        return orig(ckpt, t, tx, out, gap_path)
    monkeypatch.setattr(E, "merge_checkpoint", spy)
    E.write(tmp_path, e2["ck"], e2["out"], e2["gp"])
    assert seen == [[]]                                                               # в момент записи ключа таблиц с p ещё нет
    after = json.loads(e2["ck"].read_text(encoding="utf-8"))
    assert {k: v for k, v in after.items() if k != "W5.eval"} == {k: v for k, v in before.items() if k != "W5.eval"}      # прочие ключи не тронуты
    assert chk(e2) == [] and after["W5.eval"]["n_holm"] == 798 and after["W5.eval"]["n_holm_wide"] == 3301 and after["W5.eval"]["forecasts_written"] == 0
    after["W5.eval"]["outputs"]["W5_GUARD.csv"] = "0" * 64
    e2["ck"].write_text(json.dumps(after, indent=1, sort_keys=True, ensure_ascii=False), encoding="utf-8")
    assert any("CHECKPOINT" in b for b in chk(e2))
    del after["W5.eval"]
    e2["ck"].write_text(json.dumps(after, indent=1, sort_keys=True, ensure_ascii=False), encoding="utf-8")
    assert any("CHECKPOINT" in b for b in chk(e2))                                    # мутация: ключ не записан


def test_inputs_mismatch_and_registry_before_forecast_refuse(env, tmp_path, monkeypatch) -> None:
    t = pd.read_parquet(env["out"] / "w5g.parquet")
    t.loc[t.index[5], "rv"] *= 1.01
    bad = tmp_path / "W5"
    shutil.copytree(env["out"], bad)
    t.to_parquet(bad / "w5g.parquet", index=False)
    with pytest.raises(SystemExit):
        E.load_inputs(bad, env["gp"])                                                 # rv сохранённого ряда ≠ окну M
    copy = tmp_path / "ALL_TRIALS.csv"
    copy.write_text(G.REG.TRIALS.read_text(encoding="utf-8").replace("W5-G,", "W5-Gx,", 1), encoding="utf-8")        # реестр без строки W5-G
    orig = E.REG.require
    monkeypatch.setattr(E.REG, "require", lambda k, trials=copy: orig(k, trials=trials))
    with pytest.raises(SystemExit):
        E.load_inputs(env["out"], env["gp"])


def test_real_outputs_are_consistent_with_recount_and_registered_thresholds() -> None:
    assert E.check() == []                                                            # B.G3 на реальных выходах (только чтение)
    g = pd.read_csv(E.REP / "W5_GUARD.csv", comment="#")
    cp = json.loads(E.CKPT.read_text(encoding="utf-8"))["W5.eval"]
    assert derived_ok(g, cp["decision"]) and g[g["id"] == "решение"]["value"].iloc[0] == cp["decision"]
    thr = g.set_index("id")["threshold"]
    assert (float(thr["У1"]), float(thr["У2"]), float(thr["У3"])) == (0.415, 0.50, 0.26)
    assert T.identity_gap(pd.read_csv(E.REP / "W5_YEARS.csv", comment="#")) < 1e-12 and np.isfinite(cp["n_holm"]) and cp["n_holm"] == 798 and cp["n_holm_wide"] == 3301
    assert CAL.MARKS == tuple(MARKS)
