"""Л7: писатель `docs/otchety/harj_fwd/FREEZE_v2.json` для технического повтора открытия (Ж-249). Пишет только этот файл; `FREEZE.json` v1 и первый замок только читаются.

Запуск (только дирижёр, после коммита с правкой):
    python3 scripts/fwd/freeze_v2.py --write --fix-commit <коммит>
Пересчитываются технические поля: `code_sha256` (замыкание импортов), `git`, `packages`, `written_at`. Научные поля копируются из v1; входы перехэшируются и обязаны совпасть с v1
(иначе отказ). Отказ также: FREEZE_v2.json уже есть; первый замок не в статусе «сбой»; файл `models_v2/**` отличается от v1; любое отличие научного ключа.
"""
from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (str(ROOT), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import otkrytie_povtor as PT  # noqa: E402

REP = ROOT / "docs/otchety/harj_fwd"
OUT, V1, LOCK1 = REP / "FREEZE_v2.json", REP / "FREEZE.json", ROOT / "data/fwd/models/OPENED_HARJ_FWD.json"
INPUTS = ("inputs_archive_sha256", "inputs_new_sha256", "derived_archive_sha256")


class Refusal(RuntimeError):
    """Условие записи FREEZE_v2 не выполнено."""


def collect_fresh(commit: str | None) -> dict:
    """Технические разделы заново (как `fwd_freeze.build`) и перехэшированные входы; значения новых баров не читаются."""
    from models_v2.s17 import fwd_frz_parts as P
    P.import_chain()
    code = P.loaded_files()
    return dict(code_sha256=code, git=P.git_section(list(code), commit), packages=P.packages(), inputs_archive_sha256=P.inputs_archive(), inputs_new_sha256=P.inputs_new(), derived_archive_sha256=P.derived_sha())


def build(v1: dict, lock1: dict, prev_freeze: dict, fresh: dict, fix_commit: str, written_at: str, lock2: str = PT.LOCK2) -> dict:
    if lock1.get("status") != PT.LOCK1_STATUS:
        raise Refusal(f"первый замок в статусе «{lock1.get('status')}», не «{PT.LOCK1_STATUS}»: повтор не разрешён")
    for g in INPUTS:
        d = sorted(k for k in set(v1[g]) | set(fresh[g]) if v1[g].get(k) != fresh[g].get(k))
        if d:
            raise Refusal(f"{g}: sha256 не совпадает с v1 ({len(d)}): {d[:3]}")
    v2 = {k: copy.deepcopy(v1[k]) for k in PT.SCI}
    v2.update(v=2, written_at=written_at, git=fresh["git"], code_sha256=fresh["code_sha256"], packages=fresh["packages"], outputs=dict(v1["outputs"], lock=lock2),
              owner_yes=dict(zh=PT.OWNER_ZH, date="2026-11-11", punkt=1, quote=PT.OWNER_QUOTE), prev_freeze=prev_freeze, prev_lock=lock1, prev_commit=(v1.get("git") or {}).get("commit"), fix_commit=fix_commit)
    v2 = json.loads(json.dumps(v2, ensure_ascii=False))  # как после записи: типы JSON
    v2["diff_v1"] = PT.diff_list(v1, v2)
    bad = PT.check_diff(v1, v2)
    if bad:
        raise Refusal("FREEZE_v2 не собран: " + "; ".join(bad))
    return v2


def write(path: Path = OUT, v1_path: Path = V1, lock1_path: Path = LOCK1, fix_commit: str = "", root: Path = ROOT, fresh_fn=collect_fresh, lock2: str = PT.LOCK2) -> dict:
    if path.exists():
        raise Refusal(f"{path.name} уже есть: заморозка не перезаписывается")
    if not fix_commit:
        raise Refusal("не назван коммит с правкой (fix_commit)")
    v1 = json.loads(v1_path.read_text(encoding="utf-8"))
    prev = dict(path=PT.rel(v1_path, root), sha256=PT.sha(v1_path), v=v1.get("v"))
    stamp = datetime.now(ZoneInfo("Europe/Moscow")).isoformat(timespec="seconds")
    v2 = build(v1, PT.lock_info(lock1_path, root), prev, fresh_fn(fix_commit), fix_commit, stamp, lock2)
    tmp = path.with_name("tmp_" + path.name)
    tmp.write_text(json.dumps(v2, ensure_ascii=False, indent=1, sort_keys=True), encoding="utf-8")
    try:
        os.link(tmp, path)  # атомарно и без перезаписи: появился между проверкой и записью — FileExistsError
    finally:
        tmp.unlink()
    return v2


def main(argv: list) -> int:
    if "--write" not in argv or "--fix-commit" not in argv:
        print(__doc__)
        return 2
    full = subprocess.run(["git", "rev-parse", argv[argv.index("--fix-commit") + 1] + "^{commit}"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    try:
        v2 = write(fix_commit=full)
    except Refusal as e:
        print(f"ОТКАЗ (FREEZE_v2.json не записан): {e}", file=sys.stderr)
        return 2
    g = v2["git"]
    print(f"FREEZE_v2.json записан: fix_commit {full[:12]}, prev_commit {str(v2['prev_commit'])[:12]}; файлов в замыкании {len(v2['code_sha256'])}; не в коммите {len(g['not_in_commit'])}, отличаются {len(g['differs_from_commit'])}; отличий от v1: {[x['key'] for x in v2['diff_v1']]}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
