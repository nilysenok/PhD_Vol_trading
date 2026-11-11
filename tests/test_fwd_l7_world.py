"""Л7: общий синтетический мир повтора (не тесты): первый замок «сбой», FREEZE v1 → v2 настоящим писателем `freeze_v2.build`, журнал с «да» Ж-249, гейты Л7.
Всё в tmp; реальные замок, FREEZE и данные не трогаются."""
from __future__ import annotations

import dataclasses
import json
import types

from test_fwd_l6_open import GATES, VERS, make_world, sha  # noqa: I001 (первым: добавляет scripts/fwd в sys.path)
import freeze_v2 as FV  # noqa: E402
import otkrytie_povtor as PT  # noqa: E402

L7 = "## Л7. Повтор\n\n| Гейт | CHECK | EXPECT | EVIDENCE |\n|---|---|---|---|\n| Л7.1 | c | e | passed |\n| Л7.8 повтор выполнен | c | e | запускает дирижёр |\n\n"
LOCK1_REL = "data/fwd/models/OPENED_HARJ_FWD.json"
FIX = "b" * 40
STAMP = "2026-11-11T08:00:00+03:00"


def write_lock1(root, status: str = PT.LOCK1_STATUS, why: str = "KeyError: 'date'"):
    p = root / LOCK1_REL
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists():
        p.chmod(0o644)
    p.write_text(json.dumps(dict(status=status, prichina=why), ensure_ascii=False, indent=1), encoding="utf-8")
    p.chmod(0o444)
    return p


def fresh_of(root, v1: dict) -> dict:
    """Технические разделы «после правки»: скрипт открытия изменён (другой sha), models_v2 и входы те же."""
    (root / "scripts/fwd/otkrytie_harj_fwd.py").write_text("# s fixed\n", encoding="utf-8")
    code = dict(v1["code_sha256"], **{"scripts/fwd/otkrytie_harj_fwd.py": sha(root / "scripts/fwd/otkrytie_harj_fwd.py")})
    return dict(code_sha256=code, git=dict(commit=FIX), packages=dict(VERS), **{g: dict(v1[g]) for g in FV.INPUTS})


def make_povtor_world(tmp):
    """-> SimpleNamespace(ctx, v1, v2, lock1, root, fresh). Ctx повтора: FREEZE_v2.json, замок r2 (его нет), первый замок и v1 на месте."""
    ctx1 = make_world(tmp)
    root = ctx1.root
    v1 = json.loads(ctx1.freeze.read_text(encoding="utf-8"))
    v1.update(params={"p": 1}, initial_state={"x": 2.5}, not_admitted={"ZE": "нет"}, derived_archive_sha256={}, git=dict(commit="a" * 40), outputs=dict(models="data/fwd/models/", lock=LOCK1_REL))
    ctx1.freeze.write_text(json.dumps(v1, ensure_ascii=False), encoding="utf-8")
    ctx1.resheniya.write_text(ctx1.resheniya.read_text(encoding="utf-8") + f"\n### Ж-249 — 11.11.2026, 07:31 MSK\nЪ: «{PT.OWNER_QUOTE}»\n", encoding="utf-8")
    ctx1.gates.write_text(GATES + L7, encoding="utf-8")
    lock1 = write_lock1(root)
    fresh = fresh_of(root, v1)
    prev = dict(path=PT.rel(ctx1.freeze, root), sha256=sha(ctx1.freeze), v=1)
    v2 = FV.build(v1, PT.lock_info(lock1, root), prev, fresh, FIX, STAMP)
    ctx = dataclasses.replace(ctx1, povtor=True, freeze=None, lock=None, prev_lock=None, prev_freeze=None)
    ctx.freeze.write_text(json.dumps(v2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    return types.SimpleNamespace(ctx=ctx, v1=v1, v2=v2, lock1=lock1, root=root, fresh=fresh, prev=prev)


def edit_v2(ctx, fn) -> None:
    d = json.loads(ctx.freeze.read_text(encoding="utf-8"))
    fn(d)
    ctx.freeze.write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")
