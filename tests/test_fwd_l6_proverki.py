"""Л6: правки по замечаниям проверяющего: ключ `rehearsal.result` (сквозной тест «писатель Л5 → preflight»), обязательные входы, разбор REHEARSAL.csv кодом Л5,
pending в любой форме, замыкание импортов после прогона, битый замок (.err). Синтетика и tmp; реальный замок не трогается."""
from __future__ import annotations

import json

import pandas as pd
import pytest

from test_fwd_l6_open import CH, CLOSURE, VERS, Spy, fake_result, make_world, sha  # noqa: I001 (первым: sys.path на scripts/fwd)
import otkrytie_harj_fwd as OP  # noqa: E402
import otkrytie_proverki as PV  # noqa: E402


def test_pisatel_fwd_freeze_build_prokhodit_preflight(tmp_path, monkeypatch):
    """FREEZE строит настоящий `fwd_freeze.build` (тяжёлые разделы подменены на tmp); ключи писателя читает preflight."""
    from models_v2.s17 import fwd_freeze as FZ
    ctx = make_world(tmp_path)
    r = ctx.root
    old = json.loads(ctx.freeze.read_text(encoding="utf-8"))
    pairs = pd.DataFrame(dict(date=pd.to_datetime(["2026-02-27"]), t_plus=pd.to_datetime(["2026-03-02"])))
    monkeypatch.setattr(FZ.P, "calendar_section", lambda src: (dict(days=[], n_new=1), pairs))
    monkeypatch.setattr(FZ.P, "import_chain", lambda: None)
    monkeypatch.setattr(FZ.P, "loaded_files", lambda: old["code_sha256"])
    monkeypatch.setattr(FZ.P, "git_section", lambda files: {})
    monkeypatch.setattr(FZ.P, "packages", lambda: dict(VERS))
    monkeypatch.setattr(FZ.P, "inputs_archive", lambda: old["inputs_archive_sha256"])
    monkeypatch.setattr(FZ.P, "derived_sha", lambda: {})
    monkeypatch.setattr(FZ.P, "inputs_new", lambda: old["inputs_new_sha256"])
    monkeypatch.setattr(FZ.P, "params", lambda: {})
    monkeypatch.setattr(FZ.P, "snapshot", lambda src: {})
    monkeypatch.setattr(FZ.RH, "OUT", ctx.rehearsal)
    monkeypatch.setattr(FZ.PW, "OUT", ctx.power)
    frz = FZ.build(None)
    assert frz["rehearsal"]["result"] == "0 расхождений" and "itog" not in frz["rehearsal"]
    ctx.freeze.write_text(json.dumps(frz, ensure_ascii=False), encoding="utf-8")
    assert PV.preflight(ctx) == [] and r.exists()
    frz["rehearsal"] = dict(sha256=frz["rehearsal"]["sha256"], itog="0 расхождений")  # мутант: старый ключ — отказ
    ctx.freeze.write_text(json.dumps(frz, ensure_ascii=False), encoding="utf-8")
    assert any("rehearsal.result" in m for m in PV.preflight(ctx))


def edit(ctx, fn):
    d = json.loads(ctx.freeze.read_text(encoding="utf-8"))
    fn(d)
    ctx.freeze.write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")


def test_obyazatelnye_vkhody_i_pustota(tmp_path):
    ctx = make_world(tmp_path)
    edit(ctx, lambda d: d.update(inputs_new_sha256={}))
    assert any("inputs_new_sha256: пусто" in m for m in PV.preflight(ctx))
    ctx = make_world(tmp_path / "b")
    need = (["data/regen/a.bin", "data/regen/k23_meta_SBER.json"], ["data/fwd/sber_m10/MANIFEST.json"])  # k23_meta в заморозке не записан
    ctx.required = lambda: need
    assert any("нет обязательного входа data/regen/k23_meta_SBER.json" in m for m in PV.preflight(ctx))


def test_rehearsal_razbiraetsya_kodom_l5(tmp_path):
    ctx = make_world(tmp_path)
    ctx.rh_problems = lambda p: ["d1: нет узла K2", "d2 K3: got≠ref"]
    ms = PV.preflight(ctx)
    assert any("REHEARSAL: d1: нет узла K2" in m for m in ms) and any("всего замечаний 2" in m for m in ms)

    def boom(p):
        raise ValueError("не csv")
    ctx.rh_problems = boom
    assert any("не разобран" in m for m in PV.preflight(ctx))


@pytest.mark.parametrize("old,new,frag", [
    ("| Л2.1 | c | e | passed |", "| Л2.1 | c | e | ABANDON: Л2.1 причина |", "незакрыто"),
    ("| Л4.1 | c | e | passed |", "| Л4.1 | c | e |  |", "пустой EVIDENCE"),
    ("## Л1. лист\n", "## Л1. лист\n\n- гейт Л1.0: pending\n", "незакрыто"),
    ("| Л5.1 | c | e | passed |", "| Л5.1 | c | e | PENDING |", "незакрыто"),
])
def test_pending_v_lyuboj_forme(tmp_path, old, new, frag):
    ctx = make_world(tmp_path)
    ctx.gates.write_text(ctx.gates.read_text(encoding="utf-8").replace(old, new), encoding="utf-8")
    assert any(frag in m for m in PV.preflight(ctx))


def test_gejt_l68_isklyuchen_po_id(tmp_path):
    ctx = make_world(tmp_path)
    ctx.gates.write_text(ctx.gates.read_text(encoding="utf-8").replace("| Л5.1 | c | e | passed |", "| Л5.1 | c | e | passed |\n| L6.8 открытие | c | e | pending |"), encoding="utf-8")
    assert PV.preflight(ctx) == []


def test_zamykanie_posle_progona_lishnij_modul_prichina_v_zamok(tmp_path):
    ctx = make_world(tmp_path)
    spy = Spy(ctx, fake_result())
    base = ctx.closure

    def run_nodes(c, fz):
        ctx.closure = lambda: base() | {"models_v2/s17/fwd_pozdnij.py"}  # модуль подгружен уже в прогоне
        return spy(c, fz)

    assert OP.run(ctx, run_nodes=run_nodes, dry=False) == 3
    lk = json.loads(ctx.lock.read_text(encoding="utf-8"))
    assert lk["status"] == "нарушение заморозки: models_v2/s17/fwd_pozdnij.py" and lk["outputs_sha256"]  # выходы записаны
    first = (ctx.rep_dir / "FWD_REPORT.md").read_text(encoding="utf-8").splitlines()[0]
    assert first.startswith("НАРУШЕНИЕ ЗАМОРОЗКИ: ") and "fwd_pozdnij" in first


def test_narushenie_tolko_vmeste_so_statusom_zamka():
    from models_v2.s17 import fwd_doc
    from test_fwd_l6_eval import make_frame
    from models_v2.s17 import fwd_eval as E
    t = {k: v for k, v in E.all_tables(make_frame(n=40)).items() if k != "pairs"}
    v = ["models_v2/s17/fwd_pozdnij.py"]
    text = fwd_doc.render(t, None, v)
    assert fwd_doc.verify(text, t, None, v, "нарушение заморозки: " + v[0]) == []
    assert fwd_doc.verify(text, t, None, v, "выполнено")  # пометка в отчёте без статуса замка
    assert fwd_doc.verify(fwd_doc.render(t), t, None, None, "нарушение заморозки: x")  # статус замка без пометки
    assert fwd_doc.verify(text, t, None, None, None)  # пометка без нарушения


def test_gejt_l68_tochnyj_id_a_ne_podstroka(tmp_path):
    ctx = make_world(tmp_path)
    ctx.gates.write_text(ctx.gates.read_text(encoding="utf-8").replace("| Л5.1 | c | e | passed |", "| Л5.1 | см. L6.8 | e | pending |"), encoding="utf-8")
    assert any("Л5" in m and "незакрыто" in m for m in PV.preflight(ctx))  # L6.8 в тексте чужой строки — не исключение


@pytest.mark.parametrize("boom", [False, True])
def test_bitoj_zamok_pishet_err_i_povtor_nevozmozhen(tmp_path, boom):
    ctx = make_world(tmp_path)

    def run_nodes(c, fz):
        c.lock.chmod(0o644)
        c.lock.write_text("", encoding="utf-8")  # пустой замок посреди прогона
        if boom:
            raise ValueError("сбой в счёте")
        return fake_result()

    assert OP.run(ctx, run_nodes=run_nodes, dry=False) == 1
    err = json.loads(ctx.lock.with_suffix(".err").read_text(encoding="utf-8"))
    assert "lock_error" in err and err["status"] == ("сбой" if boom else "выполнено")
    assert OP.run(ctx, run_nodes=Spy(ctx, fake_result()), dry=False) == 2  # повтор: замок/.err есть
    ctx.lock.unlink()
    assert any(".err" in m or "замок" in m for m in PV.preflight(ctx))  # одного .err достаточно для отказа
    assert sha(ctx.freeze) and CH and CLOSURE
