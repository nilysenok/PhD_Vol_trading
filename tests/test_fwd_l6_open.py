"""Л6: скрипт открытия на синтетике во временном каталоге (РЕАЛЬНЫЙ замок, данные и FREEZE не трогаются, `main` не вызывается).
Отказы до замка (нет «да», чужой sha, замыкание импортов, пакеты, репетиция, pending, замок/выходы уже есть), замок ДО чтения цен (№ 30),
один прогон, сбой после замка, `--check` ничего не пишет."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/fwd"))
import otkrytie_harj_fwd as OP  # noqa: E402
import otkrytie_proverki as PV  # noqa: E402

from models_v2.s17 import fwd_eval as E  # noqa: E402
from models_v2.s17 import fwd_src as S  # noqa: E402
from test_fwd_l6_eval import make_frame  # noqa: E402

REAL_LOCK = ROOT / "data/fwd/models/OPENED_HARJ_FWD.json"
REAL_LOCK_EXISTED = REAL_LOCK.exists()
VERS = {n: "1.0" for n in PV.PACKAGES}
YES = "- **«Да» владельца на открытие** — Ж-248 п. 4: «После фиксации протокола и технической репетиции разрешаю проверить доступный новый период до последней сессии»."
GATES = "# GATES_FWD\n\n" + "".join(f"## {g}. лист\n\n| Гейт | CHECK | EXPECT | EVIDENCE |\n|---|---|---|---|\n| {g}.1 | c | e | passed |\n\n" for g in PV.G.GATE_LISTS) + "## Л6. Открытие\n\n| Гейт | EVIDENCE |\n|---|---|\n| L6.9 открытие | pending, запускает дирижёр |\n"
FILES = {"models_v2/m.py": "x = 1\n", "scripts/fwd/otkrytie_harj_fwd.py": "# s\n", "data/regen/a.bin": "archive", "data/fwd/sber_m10/MANIFEST.json": "{}",
         "data/regen/stageA/tcn_w5_2026_s0.npz": "npz", "docs/otchety/harj_fwd/REHEARSAL.csv": "day,node,diff\n", "docs/otchety/harj_fwd/POWER.csv": "# c\nkind,window,hyp,n_new,block,power_holm7\n"}
CLOSURE = {"models_v2/m.py", "scripts/fwd/otkrytie_harj_fwd.py"}
CH = "c" * 64


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def make_world(tmp: Path):
    root = tmp / "repo"
    for rel, txt in FILES.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(txt, encoding="utf-8")
    jr = root / "docs/resheniya/resheniya.md"
    jr.parent.mkdir(parents=True, exist_ok=True)
    jr.write_text(f"# Журнал\n\n### Ж-247 — 10.11.2026, 23:00 MSK\n- блок\n\n### Ж-248 — 11.11.2026, 00:23 MSK (по `date`)\n{YES}\n", encoding="utf-8")
    (root / "docs/otchety/harj_fwd/GATES_FWD.md").write_text(GATES, encoding="utf-8")
    h = lambda rel: sha(root / rel)  # noqa: E731
    fz = dict(v=1, code_sha256={k: h(k) for k in CLOSURE}, packages=VERS, inputs_archive_sha256={"data/regen/a.bin": h("data/regen/a.bin"), "data/regen/stageA/tcn_w5_2026_s0.npz": CH},
              inputs_new_sha256={"data/fwd/sber_m10/MANIFEST.json": h("data/fwd/sber_m10/MANIFEST.json")}, calendar=dict(days=[], n_new=0), dates={}, candidates=["X1", "ZB"], controls=["K1", "K2", "K3"],
              family={}, outputs={}, rehearsal=dict(sha256=h("docs/otchety/harj_fwd/REHEARSAL.csv"), result="0 расхождений"), power=dict(sha256=h("docs/otchety/harj_fwd/POWER.csv")),
              owner_yes=dict(zh="Ж-248", date="2026-11-11", punkt=4))
    (root / "docs/otchety/harj_fwd/FREEZE.json").write_text(json.dumps(fz, ensure_ascii=False), encoding="utf-8")
    return PV.Ctx(root=root, versions=lambda: dict(VERS), content_hash=lambda p: CH, closure=lambda: set(CLOSURE), required=lambda: (list(fz["inputs_archive_sha256"]), list(fz["inputs_new_sha256"])), rh_problems=lambda p: [])


def edit_freeze(ctx, fn):
    d = json.loads(ctx.freeze.read_text(encoding="utf-8"))
    fn(d)
    ctx.freeze.write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")


def fake_result(verdict_mut: bool = False) -> dict:
    t = E.all_tables(make_frame(n=60))
    t["checks"] = pd.DataFrame([dict(check="календарь", status="ok", detail="дней 60")])
    if verdict_mut:
        i = t["family"]["p_holm"].idxmax()
        assert t["family"].loc[i, "p_holm"] >= 0.05
        t["family"].loc[i, "verdict"] = E.CONF  # подмена вывода при p Холма ≥ 0,05
    return dict(new={k: pd.DataFrame({"d": [pd.Timestamp("2026-03-02")], "F": [1.0]}) for k in OP.NEW_NODES}, tables=t)


class Spy:
    def __init__(self, ctx, result=None, boom=None):
        self.ctx, self.result, self.boom, self.calls = ctx, result, boom, []

    def __call__(self, ctx, fz):
        self.calls.append(ctx.lock.exists())  # замок на момент первого чтения должен уже стоять
        if self.boom:
            raise self.boom
        return self.result


def snapshot(root: Path) -> dict:
    return {str(p.relative_to(root)): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()}


def test_chestnyj_mir_proveryaetsya_i_dry_nichego_ne_pishet(tmp_path):
    ctx = make_world(tmp_path)
    assert PV.preflight(ctx) == []
    before = snapshot(ctx.root)
    assert OP.run(ctx, dry=True) == 0
    assert snapshot(ctx.root) == before and not ctx.lock.exists()


def test_zamok_do_chteniya_odin_progon_i_sha_vykhodov(tmp_path):
    ctx = make_world(tmp_path)
    spy = Spy(ctx, fake_result())
    assert OP.run(ctx, run_nodes=spy, dry=False) == 0
    assert spy.calls == [True]  # к моменту чтения замок уже есть (№ 30)
    lk = json.loads(ctx.lock.read_text(encoding="utf-8"))
    assert lk["status"] == "выполнено" and lk["freeze_sha256"] == sha(ctx.freeze) and not (ctx.lock.stat().st_mode & 0o222)
    assert len(lk["outputs_sha256"]) == len(OP.NEW_NODES) + len(OP.CSV) + 1
    for rel, h in lk["outputs_sha256"].items():
        assert sha(ctx.root / rel) == h
    assert (ctx.rep_dir / "FWD_REPORT.md").read_text(encoding="utf-8").splitlines()[0].startswith("Статус: проверка на новом периоде ")
    assert len(pd.read_csv(ctx.rep_dir / "FWD_FAMILY.csv")) == 7 and sorted(p.stem for p in ctx.new_dir.glob("*.parquet")) == sorted(OP.NEW_NODES)
    snap = snapshot(ctx.root)
    spy2 = Spy(ctx, fake_result())
    assert OP.run(ctx, run_nodes=spy2, dry=False) == 2 and spy2.calls == [] and snapshot(ctx.root) == snap  # повтор невозможен


def test_sboj_posle_zamka_prichina_v_zamok_povtor_zapreshchen(tmp_path):
    ctx = make_world(tmp_path)
    assert OP.run(ctx, run_nodes=Spy(ctx, boom=ValueError("archive mismatch")), dry=False) == 1
    lk = json.loads(ctx.lock.read_text(encoding="utf-8"))
    assert lk["status"] == "сбой" and "archive mismatch" in lk["prichina"] and not list(ctx.rep_dir.glob("FWD_*"))
    assert OP.run(ctx, run_nodes=Spy(ctx, fake_result()), dry=False) == 2


def test_podmena_vyvoda_v_tablice_ostanavlivaet_otchet(tmp_path):
    ctx = make_world(tmp_path)
    assert OP.run(ctx, run_nodes=Spy(ctx, fake_result(verdict_mut=True)), dry=False) == 1
    lk = json.loads(ctx.lock.read_text(encoding="utf-8"))
    assert lk["status"] == "сбой" and "не согласованы" in lk["prichina"] and not (ctx.rep_dir / "FWD_REPORT.md").exists()


def test_zamok_o_excl_dva_zapuska_ne_prokhodyat_oba(tmp_path):
    ctx = make_world(tmp_path)
    OP.acquire_lock(ctx, {"a": 1})
    with pytest.raises(FileExistsError):
        OP.acquire_lock(ctx, {"a": 2})
    assert json.loads(ctx.lock.read_text(encoding="utf-8"))["a"] == 1


def new_src_paths(ctx) -> S.Paths:
    d = ctx.root / "data/iss_check"
    d.mkdir(parents=True, exist_ok=True)
    (ctx.root / "data/fwd/sber_m10").mkdir(parents=True, exist_ok=True)
    arch = pd.DataFrame(dict(begin=pd.to_datetime(["2026-02-27 10:00", "2026-02-27 18:40"]).as_unit("ms"), close=[300.0, 301.0]))
    new = pd.DataFrame(dict(begin=pd.to_datetime(["2026-03-02 10:00", "2026-03-02 18:40"]).as_unit("ms"), close=[302.0, 303.0]))
    (d / "bars").mkdir(exist_ok=True)
    arch.to_parquet(d / "bars/SBER.parquet")
    new.to_parquet(ctx.root / "data/fwd/sber_m10/SBER_m10_fwd.parquet")
    return S.Paths(bars_dir=d / "bars", imoex_archive=d / "I.parquet", new_sber=ctx.root / "data/fwd/sber_m10/SBER_m10_fwd.parquet", new_mkt_dir=ctx.root / "data/fwd/mkt_m10",
                   sealed_rv=ctx.root / "data/fwd/sber_m10/rv.parquet", lock=ctx.lock)


def test_realnyj_src_cheny_novogo_perioda_tolko_pod_zamkom_mutatsiya_poryadka_lovitsya(tmp_path):
    ctx = make_world(tmp_path)
    paths = new_src_paths(ctx)
    cut = pd.Timestamp("2026-03-02")
    with pytest.raises(S.SealedError):
        S.Src(paths).bars("SBER", cut)  # без замка цены новых файлов не читаются

    def reader(c, fz):
        got = S.Src(paths).bars("SBER", cut)  # с замком — читаются
        assert got["close"].tolist() == [300.0, 301.0, 302.0, 303.0]
        return fake_result()

    assert OP.run(ctx, run_nodes=reader, dry=False) == 0


def test_mutatsiya_zamok_posle_chteniya_lovitsya(tmp_path):
    """Мутант: счёт раньше замка. Реальный `Src` отказывает, то есть порядок «замок → чтение» защищён не только текстом скрипта."""
    ctx = make_world(tmp_path)
    paths = new_src_paths(ctx)

    def run_late(c, run_nodes):
        run_nodes(c, {})  # чтение до замка
        OP.acquire_lock(c, {})

    with pytest.raises(S.SealedError):
        run_late(ctx, lambda c, fz: S.Src(paths).bars("SBER", pd.Timestamp("2026-03-02")))
    assert not ctx.lock.exists()


def test_zamok_skripta_est_zamok_src_i_vykhody_po_protokolu():
    c = PV.Ctx()
    assert c.lock == S.Paths().lock == ROOT / "data/fwd/models/OPENED_HARJ_FWD.json"
    assert c.new_dir == ROOT / "data/fwd/models/new" and c.rep_dir == ROOT / "docs/otchety/harj_fwd" and c.freeze.name == "FREEZE.json"
    assert {v for v in OP.CSV.values()} | {OP.REPORT} >= {"FWD_FAMILY.csv", "FWD_REPORT.md"} and all(v.startswith("FWD_") for v in OP.CSV.values())


def test_zamykanie_importov_polnoe_i_realnyj_zamok_ne_tronut():
    import models_v2.s17.fwd_doc  # noqa: F401
    got = PV.closure_files()
    assert {"models_v2/s17/fwd_eval.py", "models_v2/s17/fwd_doc.py", "scripts/fwd/otkrytie_harj_fwd.py", "scripts/fwd/otkrytie_proverki.py", "scripts/fwd/otkrytie_raschet.py"} <= got
    assert REAL_LOCK.exists() == REAL_LOCK_EXISTED  # тесты никогда не создают реальный замок


def test_izolyaciya_ast_ne_obuchaet_i_ne_pishet_arkhiv():
    import ast
    banned = {"train_encoder", "fit_pca_past", "train_missing", "merge_cp", "emit"}
    for f in ("otkrytie_harj_fwd", "otkrytie_proverki", "otkrytie_raschet"):
        tree = ast.parse((ROOT / f"scripts/fwd/{f}.py").read_text(encoding="utf-8"))
        names = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)} | {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
        assert not (banned & names), f
        assert not [n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str) and n.value.startswith("data/regen/")]  # архив только читается, пути не зашиты
