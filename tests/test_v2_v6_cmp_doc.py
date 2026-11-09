"""V6 Z2 (G60, G61, G64): реестр, документ из таблиц, `--check` только читает, изоляция. Документ на диске = пересборка из таблиц `cmp/` (число за числом и побайтно); таблицы на диске = пересчёт из сохранённых прогнозов (1e-9).
Мутации: подменённое число в документе и в таблице; перевёрнутые «лучше/хуже» и знак Δ; ряд-потолок выбран лучшим; `--check` пишет файл; импорт обучателя; запись в проверке."""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

from models_v2.s13 import run_v6_cmp as RUN, v6_cmp_best as B, v6_cmp_doc2 as D2, v6_cmp_io as C, v6_cmp_reg as R

DIR = Path(__file__).resolve().parents[1] / "models_v2/s13"
FILES = sorted(DIR.glob("v6_cmp_*.py")) + [DIR / "run_v6_cmp.py"]
TRAIN = ("v6_1_fit", "v6_1_wf", "lightgbm", "xgboost", "sklearn", "v6_1_feat", "v6_4_an", "v6_5_fit")
WRITERS = ("write_text", "write_csv", "to_csv", "to_parquet", "write_bytes")


@pytest.fixture(scope="module")
def fresh():
    return RUN.compute()


def doc() -> str:
    return C.DOC.read_text(encoding="utf-8")


def test_disk_equals_recompute(fresh):
    assert RUN.check_all(fresh) == []


def test_document_numbers_checked_one_by_one(fresh):
    n, bad = D2.compare(doc(), D2.build(fresh))
    assert n > 500 and bad == 0 and doc().splitlines()[2].startswith("Результат:") and "не независимый OOS" in doc()[:1500]


def test_static_texts_have_no_decimal_numbers():
    for s in D2.NOT_COMP + D2.DEVIATIONS:
        assert not D2.NUM.search(s), s


def test_mutation_substituted_number_in_document_is_caught(fresh):
    d = doc()
    q = d.split("QLIKE ")[1][:6]                                              # первое число QLIKE после слова
    assert D2.compare(d.replace("QLIKE " + q, "QLIKE " + q[:-1] + ("0" if q[-1] != "0" else "1"), 1), D2.build(fresh))[1] >= 1


def test_mutation_substituted_number_in_table_is_caught(fresh):
    t = {k: v.copy() if hasattr(v, "copy") else v for k, v in fresh.items()}
    t["BEST"]["best_qlike"] = t["BEST"]["best_qlike"] + 0.0001
    assert D2.compare(doc(), D2.build(t))[1] >= 1


def test_mutation_flipped_sign_of_delta_is_caught(fresh):
    t = {k: v.copy() if hasattr(v, "copy") else v for k, v in fresh.items()}
    t["TESTS"]["delta_pct"] = -t["TESTS"]["delta_pct"]
    assert D2.compare(doc(), D2.build(t))[1] >= 1 and doc() != D2.build(t)


def test_mutation_flipped_better_worse_words_is_caught(fresh):
    d = doc().replace("Минус — лучше", "Минус — хуже").replace("статистически не отделим от", "статистически отделён от")
    assert d != D2.build(fresh) and "Минус — хуже" in d


def test_mutation_ceiling_chosen_best_is_caught(fresh, monkeypatch):
    u = R.universe()
    real = fresh["BEST"]
    real = real[real["set"].str.startswith("основное (") & (real["window"] == "2017_2021") & (real["ref_kind"] == "второй")].iloc[0]["best"]
    ceil = "J211feat:lgbm_S345"
    assert R.klass(ceil)[0] == R.CEIL and B.best_set(u, [real, "K2"], "2017_2021")[0][0] == real
    with pytest.raises(AssertionError):
        B.best_set(u, [real, ceil], "2017_2021")                             # честный код отказывает
    monkeypatch.setattr(R, "klass", lambda s: (R.PIT, ""))                    # подложенная ошибка: класс «потолок» не учитывается
    assert B.best_set(u, [real, ceil], "2017_2021")[0][0] == ceil != real     # …и потолок становится лучшим — сверка с таблицей это ловит


def test_registry_rules(fresh):
    reg, cand = fresh["SERIES"], R.candidates(fresh["SERIES"])
    assert not set(reg[reg["cls"] == R.CEIL]["series"]) & set(cand)
    assert not (reg[reg["program"].isin(["V1", "V2"])]["role"] == "б").any()   # у V1, V2 нет PIT-гибридов: (б) отсутствует
    assert "V6:A0g" not in set(reg["series"]) and set(reg[reg["program"] == "V6"]["role"]) <= {"а", "б"}
    assert (reg["n_dates"] == 2271).all() and reg["sha256_file"].str.len().eq(64).all() and reg["sha256_forecast"].str.len().eq(64).all()


def test_n_coverage_and_controls_identity(fresh):
    cov = fresh["COV"].set_index("row")
    assert cov["in_N722"].all() and cov["in_N3122"].all() and cov.loc["V5", "unique_keys"] == 24 and cov.loc["V6", "unique_keys"] == 18 and cov.loc["V3", "unique_keys"] == 634
    r = fresh["REPRO"]
    assert (r[r["kind"] == "identity"]["status"] == "совпало").all() and len(RUN.P.unexplained(r)) == 0


def test_check_changes_no_file_and_mutation_writing_check_is_caught(monkeypatch, capsys):
    before = C.digest()
    assert RUN.main_check() == 0 and C.digest() == before
    stray = C.CMP / "STRAY.tmp"
    orig = RUN.compute

    def writing_compute():
        stray.write_text("x")                                                 # подложенная ошибка: проверка пишет файл
        return orig()
    monkeypatch.setattr(RUN, "compute", writing_compute)
    try:
        assert RUN.main_check() == 1
    finally:
        stray.unlink(missing_ok=True)
    assert "--check изменил файлы" in capsys.readouterr().out


def imports(src: str) -> set:
    out = set()
    for n in ast.walk(ast.parse(src)):
        if isinstance(n, ast.Import):
            out |= {a.name.split(".")[-1] for a in n.names}
        elif isinstance(n, ast.ImportFrom):
            out |= {n.module.split(".")[-1]} | {a.name for a in n.names}
    return out


def calls(src: str, names=WRITERS) -> set:
    """Вызовы записи; `to_csv` без пути (строка в памяти) записью не считается."""
    return {n.func.attr for n in ast.walk(ast.parse(src)) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr in names
            and not (n.func.attr == "to_csv" and not n.args and not any(k.arg == "path_or_buf" for k in n.keywords))}


def test_no_training_imports_and_check_path_does_not_write():
    for f in FILES:
        assert not imports(f.read_text(encoding="utf-8")) & set(TRAIN), f.name
    src = (DIR / "run_v6_cmp.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    for fn in ("main_check", "check_all", "compare_tables", "evidence", "roundtrip", "load_disk", "compute"):
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == fn)
        assert not calls(ast.unparse(node)), fn
    assert calls("def f(p):\n    p.write_text('x')") == {"write_text"}                                              # детектор видит запись (мутация: запись в проверке)
    assert "v6_1_fit" in imports("from models_v2.s13 import v6_1_fit") & set(TRAIN)                                 # детектор видит импорт обучателя (мутация)
