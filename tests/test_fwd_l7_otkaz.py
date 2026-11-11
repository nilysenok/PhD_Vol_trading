"""Л7.2: повтор отказывает ДО замка r2 (код 2, замка r2 нет, цены не читались, ни один файл не изменён): нет «да» Ж-249; первый замок отсутствует / не «сбой» / иной sha;
замок r2 уже есть; чужой sha кода; существующий выходной файл — и отказы по самому FREEZE_v2 (нет файла, научный ключ, v1 изменён, models_v2, diff_v1, outputs.lock, owner, версия, гейт Л7)."""
from __future__ import annotations

import pytest

from test_fwd_l6_open import Spy, fake_result, snapshot  # noqa: I001 (первым: sys.path на scripts/fwd)
from test_fwd_l7_world import edit_v2, make_povtor_world, write_lock1  # noqa: E402
import otkrytie_harj_fwd as OP  # noqa: E402
import otkrytie_povtor as PT  # noqa: E402
import otkrytie_proverki as PV  # noqa: E402


def mut_no_yes(w): w.ctx.resheniya.write_text(w.ctx.resheniya.read_text(encoding="utf-8").replace(PT.YES_PHRASE, "x"), encoding="utf-8")  # noqa: E704
def mut_lock1_absent(w): w.lock1.unlink()  # noqa: E704
def mut_lock1_started(w): write_lock1(w.root, status="начато")  # noqa: E704
def mut_lock1_done(w): write_lock1(w.root, status="выполнено")  # noqa: E704
def mut_lock1_other_sha(w): write_lock1(w.root, why="другая причина")  # статус «сбой», но другие байты  # noqa: E704
def mut_lock2_exists(w): w.ctx.lock.parent.mkdir(parents=True, exist_ok=True); w.ctx.lock.write_text("{}")  # noqa: E702,E704
def mut_lock2_err(w): w.ctx.lock.parent.mkdir(parents=True, exist_ok=True); w.ctx.lock.with_suffix(".err").write_text("{}")  # noqa: E702,E704
def mut_code(w): (w.root / "models_v2/m.py").write_text("x = 2\n", encoding="utf-8")  # noqa: E704
def mut_script_code(w): (w.root / "scripts/fwd/otkrytie_harj_fwd.py").write_text("# other\n", encoding="utf-8")  # noqa: E704
def mut_closure(w): w.ctx.closure = lambda: {"models_v2/m.py", "scripts/fwd/otkrytie_harj_fwd.py", "scripts/fwd/freeze_v2.py"}  # noqa: E704
def mut_out_report(w): (w.ctx.rep_dir / "FWD_REPORT.md").write_text("old", encoding="utf-8")  # noqa: E704
def mut_out_new(w): w.ctx.new_dir.mkdir(parents=True, exist_ok=True); (w.ctx.new_dir / "x1.parquet").write_text("old")  # noqa: E702,E704
def mut_v2_absent(w): w.ctx.freeze.unlink()  # noqa: E704
def mut_v2_science(w): edit_v2(w.ctx, lambda d: d["calendar"].update(n_new=179))  # noqa: E704
def mut_v2_models(w): edit_v2(w.ctx, lambda d: d["code_sha256"].update({"models_v2/m.py": "9" * 64}))  # noqa: E704
def mut_v2_diff(w): edit_v2(w.ctx, lambda d: d.update(diff_v1=[]))  # noqa: E704
def mut_v2_lock(w): edit_v2(w.ctx, lambda d: d["outputs"].update(lock=PT.rel(w.lock1, w.root)))  # noqa: E704
def mut_v2_owner(w): edit_v2(w.ctx, lambda d: d["owner_yes"].update(zh="Ж-248"))  # noqa: E704
def mut_v2_version(w): edit_v2(w.ctx, lambda d: d.update(v=1))  # noqa: E704
def mut_v2_prev(w): edit_v2(w.ctx, lambda d: d.pop("prev_lock"))  # noqa: E704
def mut_v1_changed(w): w.ctx.prev_freeze.write_text(w.ctx.prev_freeze.read_text(encoding="utf-8") + " ", encoding="utf-8")  # noqa: E704
def mut_gate_l7(w): w.ctx.gates.write_text(w.ctx.gates.read_text(encoding="utf-8").replace("| Л7.1 | c | e | passed |", "| Л7.1 | c | e | pending |"), encoding="utf-8")  # noqa: E704
def mut_gate_l5(w): w.ctx.gates.write_text(w.ctx.gates.read_text(encoding="utf-8").replace("| Л5.1 | c | e | passed |", "| Л5.1 | c | e | pending |"), encoding="utf-8")  # noqa: E704


REFUSALS = [(mut_no_yes, "строки «да»"), (mut_lock1_absent, "первого замка"), (mut_lock1_started, "статусе «начато»"), (mut_lock1_done, "статусе «выполнено»"),
            (mut_lock1_other_sha, "первый замок: sha256"), (mut_lock2_exists, "замок OPENED_HARJ_FWD_r2.json уже есть"), (mut_lock2_err, "замок OPENED_HARJ_FWD_r2.json уже есть"),
            (mut_code, "sha256"), (mut_script_code, "sha256"), (mut_closure, "код вне code_sha256"), (mut_out_report, "выход уже есть"), (mut_out_new, "выход уже есть"),
            (mut_v2_absent, "нет FREEZE_v2.json"), (mut_v2_science, "научный ключ calendar"), (mut_v2_models, "models_v2: models_v2/m.py"), (mut_v2_diff, "diff_v1"),
            (mut_v2_lock, "outputs.lock"), (mut_v2_owner, "owner_yes.zh повтора"), (mut_v2_version, "версия"), (mut_v2_prev, "нет ключа prev_lock"),
            (mut_v1_changed, "FREEZE v1: sha256"), (mut_gate_l7, "Л7: незакрыто"), (mut_gate_l5, "Л5: незакрыто")]


def test_chestnyj_mir_povtora_prokhodit_proverki_i_dry_nichego_ne_pishet(tmp_path):
    w = make_povtor_world(tmp_path)
    assert PV.preflight(w.ctx) == []
    before = snapshot(w.root)
    assert OP.run(w.ctx, dry=True) == 0
    assert snapshot(w.root) == before and not w.ctx.lock.exists()


@pytest.mark.parametrize("mut,frag", REFUSALS, ids=[m.__name__ for m, _ in REFUSALS])
def test_otkaz_povtora_do_zamka(tmp_path, capsys, mut, frag):
    w = make_povtor_world(tmp_path)
    mut(w)
    after_mut = snapshot(w.root)
    spy = Spy(w.ctx, fake_result())
    assert OP.run(w.ctx, run_nodes=spy, dry=False) == 2
    assert frag in capsys.readouterr().err and spy.calls == [] and snapshot(w.root) == after_mut  # ни замка r2, ни чтения, ни одного изменённого файла


def test_gejt_l78_isklyuchen_po_id_a_l7_proveryaetsya_tolko_v_povtore(tmp_path):
    w = make_povtor_world(tmp_path)
    w.ctx.gates.write_text(w.ctx.gates.read_text(encoding="utf-8").replace("запускает дирижёр", "pending"), encoding="utf-8")
    assert PV.preflight(w.ctx) == []  # Л7.8 — по id, как L6.8
    mut_gate_l7(w)
    assert any("Л7: незакрыто" in m for m in PV.preflight(w.ctx))
