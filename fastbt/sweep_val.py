"""Validation A′ (Ж-031 Э1-д, Ж-037): один взгляд на 2020–2021 под замком. Реальный прогон запускает лид после «да»; здесь `--dry`.

Порядок: (1) сверка «да»: строка в `resheniya.md` под Ж-номером, позже заморозки списка кандидатов, с полным sha списка;
(2) замок `O_EXCL` от корня основной копии ДО любого чтения рядов 2020–2021; повтор — только с `reopen_note` (один раз, N×2);
(3) реестр `docs/otbor/validation_reestr.csv` — первая строка пишется под замком; (4) кандидаты и таблицы — в отдельный каталог,
которого до взгляда нет; (5) сбой — каталог удаляется без чтения, в реестр «прервано до просмотра».
Ворота и статистика — `sweep_val_stat`. Не прошёл кандидат — остаётся исходная версия.
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import pandas as pd

from fastbt import sweep_val_stat as vs


class Stop(SystemExit):
    def __init__(self, msg):
        super().__init__(2)
        self.msg = msg
        print(f"СТОП: {msg}", file=sys.stderr, flush=True)


def koren_repo(root=None) -> Path:
    root = Path(__file__).resolve().parent.parent if root is None else Path(root)
    r = subprocess.run(["git", "-C", str(root), "rev-parse", "--git-common-dir"], capture_output=True, text=True, timeout=20)
    if r.returncode != 0 or not r.stdout.strip():
        raise Stop(f"{root}: не git-репозиторий — путь замка не определить")
    g = Path(r.stdout.strip())
    return (g if g.is_absolute() else Path(root) / g).resolve().parent


@dataclasses.dataclass
class Puti:
    koren: Path
    zhurnal: Path | None = None
    lock: Path | None = None
    reestr: Path | None = None
    out: Path | None = None

    @classmethod
    def default(cls, root=None) -> "Puti":
        """Замок и реестр — от корня основной копии (git common-dir), а не от рабочей копии worktree."""
        return cls(koren=koren_repo(root))

    def __post_init__(self):
        k = Path(self.koren)
        self.zhurnal = self.zhurnal or k / "docs" / "resheniya" / "resheniya.md"
        self.lock = self.lock or k / "data" / "fastbt" / "VALIDATION_A_OPENED.json"
        self.reestr = self.reestr or k / "docs" / "otbor" / "validation_reestr.csv"
        self.out = self.out or k / "docs" / "tables" / "validation_a"


def sha_kandidatov(meta: list) -> str:
    return hashlib.sha256(json.dumps(sorted(meta, key=lambda m: m["id"]), sort_keys=True, default=str).encode()).hexdigest()


def _t(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d %H:%M")


def check_da(spec: dict, sha: str, P: Puti) -> None:
    """Образец — `scripts/otkrytie_gl4.check_da`: строка «да» в журнале под Ж-номером, позже заморозки, с полным sha."""
    if spec.get("zamorozheno_sha") != sha:
        raise Stop("список кандидатов изменён после заморозки (sha не равен zamorozheno_sha): новая заморозка, не перевыпуск «да»")
    d = spec.get("da_vladeltsa") or {}
    stroka, zh = str(d.get("stroka") or "").strip(), str(d.get("zhurnal") or "")
    if not stroka or not re.search(r"(?<![А-Яа-яЁё])да(?![А-Яа-яЁё])", stroka, re.I):
        raise Stop("нет строки «да» владельца")
    if not re.fullmatch(r"Ж-\d{3,}", zh) or not Path(P.zhurnal).exists():
        raise Stop("нет номера Ж-NNN или журнала решений")
    m = re.search(rf"^### {re.escape(zh)}\b.*?(?=^### |\Z)", Path(P.zhurnal).read_text("utf-8"), re.M | re.S)
    if not m or stroka not in m.group(0):
        raise Stop(f"строки «да» нет в журнале под {zh}")
    h = re.match(rf"### {re.escape(zh)}\s*·\s*(\d{{2}})\.(\d{{2}})\.(\d{{4}}),\s*(\d{{1,2}}):(\d{{2}})", m.group(0))
    if not h:
        raise Stop(f"заголовок {zh} без времени «ДД.ММ.ГГГГ, ЧЧ:ММ»")
    if _t(f"{h[3]}-{h[2]}-{h[1]} {int(h[4]):02d}:{h[5]}") <= _t(spec["zamorozheno_v"]):
        raise Stop("«да» не позже заморозки списка кандидатов")
    if sha not in stroka:
        raise Stop("строка «да» не содержит полный sha списка кандидатов")


def acquire_lock(P: Puti, spec: dict, sha: str) -> int:
    """Замок до чтения данных. Первый взгляд — O_EXCL на замок; второй — только с reopen_note и O_EXCL на `.2`; третий — отказ."""
    Path(P.lock).parent.mkdir(parents=True, exist_ok=True)
    lock, info = Path(P.lock), dict(sha=sha, zhurnal=spec["da_vladeltsa"]["zhurnal"], vremya=datetime.now().isoformat())
    for k, f in enumerate((lock, lock.with_name(lock.name + ".2")), 1):
        if k == 2 and not spec.get("reopen_note"):
            raise Stop("validation уже открыта; повторно — только с заранее записанным изменением (reopen_note), N в DSR ×2")
        try:
            fd = os.open(str(f), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o444)
        except FileExistsError:
            continue
        with os.fdopen(fd, "w") as fh:
            fh.write(json.dumps(dict(info, note=spec.get("reopen_note")), ensure_ascii=False))
        return k
    raise Stop("validation открыта дважды; третьего взгляда нет")


def reestr(P: Puti, sha: str, zh: str, status: str) -> None:
    f = Path(P.reestr)
    f.parent.mkdir(parents=True, exist_ok=True)
    new = not f.exists()
    with f.open("a", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["vremya", "sha_kandidatov", "zhurnal_da", "status"])
        w.writerow([datetime.now().strftime("%d.%m.%Y %H:%M:%S"), sha, zh, status])


def reshenie(rows: list) -> dict:
    """Группа (стратегия или A) -> id прошедшего кандидата с наибольшим Δ; иначе «ishodnaya» (остаётся исходная версия)."""
    out = {}
    for r in rows:
        g = r["id"].split(":")[0]
        best = out.get(g)
        if r["proshel"] and (best in (None, "ishodnaya") or r["delta"] > next(x["delta"] for x in rows if x["id"] == best)):
            out[g] = r["id"]
        else:
            out.setdefault(g, "ishodnaya")
    return out


def run(P: Puti, spec: dict, meta: list, factory, reps_lw: int = vs.REPS_LW, reps_k2: int = vs.REPS_K2) -> dict:
    """factory(meta, reopen_note) -> [Cand] читает данные — вызывается ТОЛЬКО под замком."""
    sha = sha_kandidatov(meta)
    check_da(spec, sha, P)
    acquire_lock(P, spec, sha)
    zh = spec["da_vladeltsa"]["zhurnal"]
    reestr(P, sha, zh, "начато")
    try:
        cands = factory(meta, spec.get("reopen_note"))
        for sw in {id(x): x for c in cands for x in (c.sw, c.orig_sw)}.values():
            sw.open_validation()
        rows = [vs.evaluate_candidate(c, reps_lw, reps_k2) for c in cands]
        res = reshenie(rows)
        Path(P.out).mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(Path(P.out) / "kandidaty.csv", index=False)
        (Path(P.out) / "reshenie.json").write_text(json.dumps(res, ensure_ascii=False), "utf-8")
    except BaseException:
        shutil.rmtree(P.out, ignore_errors=True)          # без чтения; результаты не показываются
        reestr(P, sha, zh, "прервано до просмотра")
        raise RuntimeError("прервано до просмотра: результаты удалены") from None
    reestr(P, sha, zh, "просмотр состоялся")
    return dict(rows=rows, reshenie=res)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dry", action="store_true", help="синтетика во временном каталоге; реальные данные не читаются")
    if not ap.parse_args(argv).dry:
        raise Stop("реальный прогон запускает лид после «да» владельца (список кандидатов и фабрика данных — вне этого PR)")
    from fastbt.sweep_val_dry import dry_run
    with tempfile.TemporaryDirectory() as td:
        res = dry_run(Path(td))
    print(pd.DataFrame(res["rows"])[["id", "delta", "p_lw", "p_k2", "pbo", "dsr", "N", "proshel"]].round(3).to_string())
    print(res["reshenie"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
