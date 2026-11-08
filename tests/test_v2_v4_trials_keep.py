"""V4, G22: стадия не удаляет строки других стадий из ALL_TRIALS.csv; прежний способ записи (перезапись) ловится тем же инвариантом."""
from __future__ import annotations

import pytest

from models_v2.s11 import v4_b_stage as BS, v4_c_stage as CA, v4_cb_stage as CB, v4_e_stage as ES, v4_io as IO, v4_trials as TR

SEED = ["Ж-222:%s:aaa,Ж-222,x,d,cfg,trial" % c for c in ["B1", "B2", "C1", "C2", "C5", "C10", "E1", "E2"]]


@pytest.fixture()
def reg(tmp_path, monkeypatch):
    monkeypatch.setattr(IO, "REP", tmp_path)
    (tmp_path / "ALL_TRIALS.csv").write_text(TR.HEADER + "".join(x + "\n" for x in SEED))
    return tmp_path / "ALL_TRIALS.csv"


def rows(f):
    return f.read_text().splitlines()[1:]


def others(f, prefixes):
    return [x for x in rows(f) if not x.startswith(tuple(prefixes))]


def old_overwrite(f, own):
    """Прежнее поведение (Ca): оставить только строки «своей» стадии по грубому префиксу и дописать новые — чужие строки теряются."""
    f.write_text(TR.HEADER + "".join(x + "\n" for x in [y for y in rows(f) if y.startswith("Ж-222:C")] + own))


def test_ca_rerun_keeps_other_stages(reg):
    CA.write_trials("k" * 64)
    CA.write_trials("k" * 64)
    r = rows(reg)
    assert [x for x in r if x.startswith("Ж-222:C5") or x.startswith("Ж-222:C10")] != []
    assert others(reg, ["Ж-222:C1:", "Ж-222:C2:", "Ж-222:C3:", "Ж-222:C4:"]) == [x for x in SEED if not x.startswith(("Ж-222:C1:", "Ж-222:C2:"))]
    assert sum(x.startswith(("Ж-222:C1:", "Ж-222:C2:", "Ж-222:C3:", "Ж-222:C4:")) for x in r) == 4


def test_cb_rerun_keeps_other_stages(reg):
    CB.write_trials("k" * 64)
    CB.write_trials("k" * 64)
    assert [x for x in rows(reg) if x.startswith(("Ж-222:B", "Ж-222:E", "Ж-222:C1:", "Ж-222:C2:"))] == [x for x in SEED if x.startswith(("Ж-222:B", "Ж-222:E", "Ж-222:C1:", "Ж-222:C2:"))]
    assert sum(x.startswith("Ж-222:C") for x in rows(reg)) == 2 + 6


def test_b_and_e_rerun_keep_other_stages(reg):
    ES.write_trials("k" * 64)
    ES.write_trials("k" * 64)
    assert sum(x.startswith("Ж-222:E") for x in rows(reg)) == 6
    assert [x for x in rows(reg) if x.startswith(("Ж-222:B", "Ж-222:C"))] == [x for x in SEED if x.startswith(("Ж-222:B", "Ж-222:C"))]
    BS.write_trials(["B1", "B2"], "k" * 64)
    assert sum(x.startswith("Ж-222:E") for x in rows(reg)) == 6 and sum(x.startswith("Ж-222:C") for x in rows(reg)) == 4


def test_replace_in_place_and_idempotent(reg):
    new = ["Ж-222:C1:new,a,b,c,d,trial"]
    TR.replace_rows(["Ж-222:C1:"], new)
    a = rows(reg)
    TR.replace_rows(["Ж-222:C1:"], new)
    assert rows(reg) == a and a.index(new[0]) == 2 and len(a) == len(SEED)


def test_mutation_overwrite_loses_other_stages(reg):
    old_overwrite(reg, ["Ж-222:C1:new,a,b,c,d,trial"])
    assert not [x for x in rows(reg) if x.startswith(("Ж-222:B", "Ж-222:E"))]        # прежний способ теряет B и E; инвариант выше на исправлении проходит
