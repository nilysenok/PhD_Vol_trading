"""Л7: проверки технического повтора открытия (Ж-249) ДО замка r2. Ничего не пишет, значений новых файлов не читает.

Повтор идёт по `FREEZE_v2.json` со ссылками на предыдущие: первый замок `OPENED_HARJ_FWD.json` (статус «сбой», sha256 как в `prev_lock`) и `FREEZE.json` v1 остаются нетронутыми.
Научные ключи v2 = v1 по значению, файлы `models_v2/**` в `code_sha256` = v1; отличаются только технические ключи (`diff_v1` их перечисляет).
"""
from __future__ import annotations

import json
from pathlib import Path

SCI = ("calendar", "dates", "candidates", "controls", "family", "params", "initial_state", "not_admitted", "rehearsal", "power", "inputs_archive_sha256", "inputs_new_sha256", "derived_archive_sha256")
TECH = ("v", "written_at", "git", "code_sha256", "packages", "outputs", "owner_yes", "prev_freeze", "prev_lock", "prev_commit", "fix_commit")
REQ_V2 = ("prev_freeze", "prev_lock", "prev_commit", "fix_commit", "diff_v1")
LOCK1_STATUS = "сбой"
OWNER_ZH = "Ж-249"
OWNER_QUOTE = "Разрешаю один повтор запуска проверки на новом периоде после устранения технической ошибки."
YES_PHRASE = "один повтор запуска проверки на новом периоде после устранения технической ошибки"
LOCK2 = "data/fwd/models/OPENED_HARJ_FWD_r2.json"


def sha(p) -> str:
    from otkrytie_proverki import sha256_file  # отложенный импорт: otkrytie_proverki сам импортирует этот модуль
    return sha256_file(p)


def rel(p, root) -> str:
    try:
        return Path(p).resolve().relative_to(Path(root).resolve()).as_posix()
    except ValueError:
        return str(p)


def lock_info(path: Path, root) -> dict:
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    return dict(path=rel(path, root), sha256=sha(path), status=d.get("status"), prichina=d.get("prichina"))


def diff_list(v1: dict, v2: dict) -> list:
    """Ключи, значения которых в v2 не те, что в v1 (кроме `diff_v1`), с пометкой «технический»; нетехнические и научные попадут в `check_diff` как нарушения."""
    return [dict(key=k, kind="технический") for k in sorted(set(v1) | set(v2)) if k != "diff_v1" and v1.get(k) != v2.get(k)]


def check_diff(v1: dict, v2: dict) -> list:
    """v2 против v1: научные ключи равны; `models_v2/**` в code_sha256 равны; `outputs` отличается только замком; отличия — только технические и перечислены в `diff_v1`."""
    bad = [f"научный ключ {k} отличается от v1" for k in SCI if v1.get(k) != v2.get(k)]
    m = lambda z: {k: v for k, v in (z.get("code_sha256") or {}).items() if k.startswith("models_v2/")}  # noqa: E731
    m1, m2 = m(v1), m(v2)
    bad += [f"models_v2: {k} отличается от v1" for k in sorted(set(m1) | set(m2)) if m1.get(k) != m2.get(k)]
    o = lambda z: {k: v for k, v in (z.get("outputs") or {}).items() if k != "lock"}  # noqa: E731
    bad += [] if o(v1) == o(v2) else ["outputs отличается не только замком"]
    d = diff_list(v1, v2)
    bad += [f"нетехнический ключ отличается от v1: {x['key']}" for x in d if x["key"] not in TECH + SCI]
    return bad + ([] if v2.get("diff_v1") == d else ["diff_v1 не совпадает с фактическими отличиями от v1"])


def check_prev_lock(fz: dict, ctx) -> list:
    pl = fz.get("prev_lock") or {}
    if not ctx.prev_lock.is_file():
        return [f"первого замка {ctx.prev_lock.name} нет: повтор возможен только после зафиксированного сбоя"]
    try:
        st = json.loads(ctx.prev_lock.read_text(encoding="utf-8")).get("status")
    except ValueError:
        st = "не разобран"
    bad = [] if st == LOCK1_STATUS else [f"первый замок в статусе «{st}», не «{LOCK1_STATUS}»"]
    bad += [] if pl.get("sha256") and sha(ctx.prev_lock) == pl["sha256"] else ["первый замок: sha256 не совпадает с prev_lock в FREEZE_v2"]
    return bad + ([] if pl.get("status") == LOCK1_STATUS and pl.get("path") == rel(ctx.prev_lock, ctx.root) else ["prev_lock: статус или путь не по первому замку"])


def check_prev_freeze(fz: dict, ctx) -> list:
    pf = fz.get("prev_freeze") or {}
    if not ctx.prev_freeze.is_file():
        return [f"нет {ctx.prev_freeze.name} (v1)"]
    bad = [] if pf.get("sha256") and sha(ctx.prev_freeze) == pf["sha256"] else ["FREEZE v1: sha256 не совпадает с prev_freeze в FREEZE_v2"]
    bad += [] if pf.get("path") == rel(ctx.prev_freeze, ctx.root) else ["prev_freeze: путь не по v1"]
    return bad + check_diff(json.loads(ctx.prev_freeze.read_text(encoding="utf-8")), fz)


def check_povtor(fz: dict, ctx) -> list:
    """Шаг повтора: v2 со ссылками на первый замок и v1; owner_yes Ж-249; замок r2 ≠ первому и записан в `outputs.lock`."""
    bad = [f"в FREEZE_v2.json нет ключа {k}" for k in REQ_V2 if k not in fz]
    bad += [] if (fz.get("owner_yes") or {}).get("zh") == OWNER_ZH else [f"owner_yes.zh повтора не {OWNER_ZH}"]
    bad += [] if (fz.get("outputs") or {}).get("lock") == rel(ctx.lock, ctx.root) and ctx.lock != ctx.prev_lock else ["outputs.lock не равен замку r2"]
    return bad + check_prev_lock(fz, ctx) + check_prev_freeze(fz, ctx)
