"""Л6: отказы скрипта открытия до замка (протокол, раздел 6, шаги 1–4): нет «да», чужой sha кода/входов/манифеста/энкодера, неполное замыкание импортов, пакеты,
репетиция не «0 расхождений», pending у Л1–Л5, замок или выходы уже есть. Каждый отказ: код 2, замка нет, цены не читались. Синтетический мир из test_fwd_l6_open."""
from __future__ import annotations

import pytest

from test_fwd_l6_open import CLOSURE, VERS, Spy, edit_freeze, fake_result, make_world  # noqa: I001  (первым: добавляет scripts/fwd в sys.path)
import otkrytie_harj_fwd as OP  # noqa: E402
import otkrytie_proverki as PV  # noqa: E402


def mut_owner(c): edit_freeze(c, lambda d: d.pop("owner_yes"))  # noqa: E704
def mut_not_in_journal(c): c.resheniya.write_text(c.resheniya.read_text(encoding="utf-8").replace(PV.YES_PHRASE, "x"), encoding="utf-8")  # noqa: E704
def mut_yes_other_date(c): edit_freeze(c, lambda d: d["owner_yes"].update(date="2026-11-12"))  # noqa: E704
def mut_yes_other_zh(c): edit_freeze(c, lambda d: d["owner_yes"].update(zh="Ж-247"))  # noqa: E704
def mut_code(c): (c.root / "models_v2/m.py").write_text("x = 2\n", encoding="utf-8")  # noqa: E704
def mut_archive(c): (c.root / "data/regen/a.bin").write_text("tampered", encoding="utf-8")  # noqa: E704
def mut_manifest(c): (c.root / "data/fwd/sber_m10/MANIFEST.json").write_text('{"x": 1}', encoding="utf-8")  # noqa: E704
def mut_npz_hash(c): edit_freeze(c, lambda d: d["inputs_archive_sha256"].update({"data/regen/stageA/tcn_w5_2026_s0.npz": "d" * 64}))  # noqa: E704
def mut_closure(c): c.closure = lambda: CLOSURE | {"models_v2/s17/fwd_novyj.py"}  # noqa: E704
def mut_package(c): c.versions = lambda: dict(VERS, torch="9.9")  # noqa: E704
def mut_rehearsal(c): edit_freeze(c, lambda d: d["rehearsal"].update(result="3 расхождения"))  # noqa: E704
def mut_rehearsal_file(c): c.rehearsal.write_text("day,node,diff\n1,x,1\n", encoding="utf-8")  # noqa: E704
def mut_power_file(c): c.power.write_text("# c\nkind\n", encoding="utf-8")  # noqa: E704
def mut_pending(c): c.gates.write_text(c.gates.read_text(encoding="utf-8").replace("| Л3.1 | c | e | passed |", "| Л3.1 | c | e | pending |"), encoding="utf-8")  # noqa: E704
def mut_gate_missing(c): c.gates.write_text(c.gates.read_text(encoding="utf-8").replace("## Л5. лист", "## Л7. лист"), encoding="utf-8")  # noqa: E704
def mut_lock(c): c.lock.parent.mkdir(parents=True, exist_ok=True); c.lock.write_text("{}")  # noqa: E702,E704
def mut_outputs(c): (c.rep_dir / "FWD_REPORT.md").write_text("old", encoding="utf-8")  # noqa: E704
def mut_candidates(c): edit_freeze(c, lambda d: d.update(candidates=["X1", "ZB", "ZE"]))  # noqa: E704


REFUSALS = [(mut_owner, "owner_yes"), (mut_not_in_journal, "строки «да»"), (mut_yes_other_date, "строки «да»"), (mut_yes_other_zh, "строки «да»"), (mut_code, "sha256"),
            (mut_archive, "sha256"), (mut_manifest, "sha256"), (mut_npz_hash, "sha256"), (mut_closure, "код вне code_sha256"), (mut_package, "пакет torch"), (mut_rehearsal, "rehearsal.result"),
            (mut_rehearsal_file, "rehearsal: sha256"), (mut_power_file, "power: sha256"), (mut_pending, "Л3: незакрыто"), (mut_gate_missing, "нет раздела Л5"),
            (mut_lock, "замок"), (mut_outputs, "выход уже есть"), (mut_candidates, "кандидаты")]


@pytest.mark.parametrize("mut,frag", REFUSALS, ids=[m.__name__ for m, _ in REFUSALS])
def test_otkaz_do_zamka(tmp_path, capsys, mut, frag):
    ctx = make_world(tmp_path)
    mut(ctx)
    lock_before = ctx.lock.exists()
    spy = Spy(ctx, fake_result())
    assert OP.run(ctx, run_nodes=spy, dry=False) == 2
    assert frag in capsys.readouterr().err and spy.calls == [] and ctx.lock.exists() == lock_before  # ни замка, ни чтения; открытие не потрачено
    assert not list(ctx.new_dir.glob("*")) if ctx.new_dir.exists() else True
