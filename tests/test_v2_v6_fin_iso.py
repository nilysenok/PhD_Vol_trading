"""V6-FIN (G53, G55): изоляция и реестр. Итог не обучает (модули FIN не импортируют обучатели и движок прогноза), `--check` и модули сборки документа не пишут, реестр — 18 строк без дублей, ключ FIN в CHECKPOINT, REPRODUCTION содержит команды всех стадий.
Мутации: импорт обучателя в модуль FIN, запись в проверке, лишняя строка реестра."""
from __future__ import annotations

import ast
import json
from pathlib import Path

from models_v2.s13 import v6_0_cp as CP0, v6_0_io as IO, v6_1_reg as RG, v6_fin_asm as A, v6_fin_data as D

DIR = Path(__file__).resolve().parents[1] / "models_v2/s13"
FIN_FILES = sorted(DIR.glob("v6_fin_*.py"))
TRAIN = ("v6_1_fit", "v6_1_wf", "lightgbm", "xgboost", "sklearn", "v6_1_feat", "v6_4_an", "v6_5_fit")
WRITERS = ("write_text", "write_csv", "to_csv", "to_parquet", "write_docs", "write_bytes")


def imports(src: str) -> set:
    out = set()
    for n in ast.walk(ast.parse(src)):
        if isinstance(n, ast.Import):
            out |= {a.name.split(".")[-1] for a in n.names}
        elif isinstance(n, ast.ImportFrom):
            out |= {n.module.split(".")[-1]} | {a.name for a in n.names}
    return out


def calls(src: str) -> set:
    return {n.func.attr for n in ast.walk(ast.parse(src)) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}


def test_fin_modules_do_not_import_trainers():
    assert FIN_FILES
    for p in FIN_FILES:
        assert not (imports(p.read_text(encoding="utf-8")) & set(TRAIN)), p.name


def test_mutation_trainer_import_is_detected():
    assert imports("from models_v2.s13 import v6_1_fit as FT") & set(TRAIN)
    assert imports("import lightgbm") & set(TRAIN)


def test_check_and_doc_modules_never_write():
    """`--check` (v6_fin_check, v6_fin_docheck, v6_fin_ev) и блоки документа только читают; пишут лишь v6_fin_stage.write и v6_fin_asm.write_docs / v6_fin_data.write_csv."""
    for n in ("v6_fin_check.py", "v6_fin_docheck.py", "v6_fin_ev.py", "v6_fin_doc.py", "v6_fin_doc2.py", "v6_fin_doc3.py", "v6_fin_tabs.py", "v6_fin_abl.py"):
        assert not (calls((DIR / n).read_text(encoding="utf-8")) & set(WRITERS)), n


def test_mutation_write_in_check_is_detected():
    assert calls("def check(r):\n    CP.write_text('x')\n") & set(WRITERS)


def test_registry_is_18_declared_rows_without_duplicates_and_a0g_marked():
    txt = CP0.ALL_TRIALS.read_text(encoding="utf-8")
    keys = [ln.split(",", 1)[0] for ln in txt.splitlines()[1:]]
    assert keys == ["Ж-227:%s" % c for c in D.CFG] and len(set(keys)) == 18 and RG.trials_bad(txt) == []
    assert "ОТСТУПЛЕНИЕ" in RG.CONFIGS["A0g"]["source"] and "Ж-227:A0g," in txt


def test_mutation_extra_or_duplicate_row_is_detected():
    txt = CP0.ALL_TRIALS.read_text(encoding="utf-8")
    assert RG.trials_bad(txt + "Ж-227:ZZ,Ж-227,x,y,{},trial\n") != []
    assert RG.trials_bad(txt + txt.splitlines()[1] + "\n") != []


def test_checkpoint_has_keys_0_to_6_and_fin_with_zero_configs_and_table_hashes():
    cp = json.loads(CP0.CP.read_text())
    assert all(k in cp for k in ("0", "1a", "1b", "2", "3", "4", "5", "6", "FIN"))
    fin = cp["FIN"]
    assert fin["configs_spent_by_FIN"] == 0 and fin["used_total"] == 18 and fin["budget"] == 18 and set(fin["tables_sha"]) == set(D.TABLES)
    for n in D.TABLES:
        assert fin["tables_sha"][n] == IO.sha_file(IO.REP / (n + ".csv"))[:16]
    assert {"code_hash_v6_1"} <= set(cp["6"]) and cp["6"]["code_hash_v6_1"] == RG.code_hash()


def test_reproduction_lists_every_stage_with_check():
    txt = A.REPRO.read_text(encoding="utf-8")
    for s in A.STAGES:
        assert "--stage %s\n" % s in txt and "--stage %s --check" % s in txt
