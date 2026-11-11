"""Л6: проверки скрипта открытия ДО замка (протокол HARJ_HYBRID_SBER_FWD, раздел 6, шаги 1–4; `otbor_statistika.md` п. 6).

Ничего не пишет и не читает значений цен/объёмов/RV новых файлов (sha256 считается по байтам, как в `otkrytie_gl4`). `preflight` возвращает список
причин отказа (пустой — можно открывать): нет `owner_yes` или строки «да» в журнале; чужой sha кода (и неполное замыкание импортов), входов, манифестов,
версий пакетов, REHEARSAL и POWER; репетиция не «0 расхождений»; `pending` у гейтов Л1–Л5; замок или выходы уже есть.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Callable, Optional

import otkrytie_gates as G
import otkrytie_povtor as PT

ROOT = Path(__file__).resolve().parents[2]
YES_PHRASE = "разрешаю проверить доступный новый период"
YES_PHRASES = {PT.OWNER_ZH: PT.YES_PHRASE}   # Л7: повтор после сбоя (Ж-249); остальные номера — фраза Ж-248
PACKAGES = ("python", "numpy", "pandas", "scipy", "torch", "xgboost", "lightgbm")
REQ = ("v", "code_sha256", "packages", "inputs_archive_sha256", "inputs_new_sha256", "calendar", "dates", "candidates", "controls", "family", "outputs", "rehearsal", "power", "owner_yes")
CLOSURE_DIRS = ("models_v2", "scripts/fwd")


def sha256_file(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def closure_files(root: Path = ROOT) -> set:
    """Файлы кода бегуна в `sys.modules` (models_v2/** и scripts/fwd/*.py) — относительные пути; `code_sha256` обязан их покрывать."""
    out = set()
    for m in list(sys.modules.values()):
        f = getattr(m, "__file__", None)
        if not f:
            continue
        try:
            rel = Path(f).resolve().relative_to(root).as_posix()
        except ValueError:
            continue
        if rel.endswith(".py") and any(rel.startswith(d + "/") for d in CLOSURE_DIRS):
            out.add(rel)
    return out | {p.relative_to(root).as_posix() for p in (root / "scripts/fwd").glob("*.py")}  # как `fwd_frz_parts.loaded_files`


def versions() -> dict:
    import importlib.metadata as md
    import importlib.util as iu
    out = {"python": sys.version.split()[0]}
    for n in PACKAGES[1:]:
        m = sys.modules.get(n)
        out[n] = (getattr(m, "__version__", None) or md.version(n)) if (m or iu.find_spec(n)) else None
    return out


def content_hash(p: Path) -> str:
    from models_v2.s17 import fwd_z
    return fwd_z.STA.content_hash(fwd_z.STA.load(p))


@dataclass
class Ctx:
    root: Path = ROOT
    freeze: Optional[Path] = None
    resheniya: Optional[Path] = None
    gates: Optional[Path] = None
    lock: Optional[Path] = None
    rehearsal: Optional[Path] = None
    power: Optional[Path] = None
    new_dir: Optional[Path] = None
    rep_dir: Optional[Path] = None
    versions: Callable = versions
    content_hash: Callable = content_hash
    closure: Optional[Callable] = None
    required: Optional[Callable] = None
    rh_problems: Callable = G.rehearsal_problems
    povtor: bool = False                  # Л7: повтор по FREEZE_v2 и замку r2 (первый замок и v1 — только чтение)
    prev_lock: Optional[Path] = None
    prev_freeze: Optional[Path] = None

    def __post_init__(self):
        self.required = self.required or (lambda: G.required_inputs(self.root))
        r, rep = self.root, self.root / "docs/otchety/harj_fwd"
        self.rep_dir = self.rep_dir or rep
        self.freeze = self.freeze or rep / ("FREEZE_v2.json" if self.povtor else "FREEZE.json")
        self.resheniya = self.resheniya or r / "docs/resheniya/resheniya.md"
        self.gates = self.gates or rep / "GATES_FWD.md"
        self.prev_lock = self.prev_lock or r / "data/fwd/models/OPENED_HARJ_FWD.json"
        self.prev_freeze = self.prev_freeze or rep / "FREEZE.json"
        self.lock = self.lock or (r / PT.LOCK2 if self.povtor else self.prev_lock)
        self.rehearsal = self.rehearsal or rep / "REHEARSAL.csv"
        self.power = self.power or rep / "POWER.csv"
        self.new_dir = self.new_dir or r / "data/fwd/models/new"
        self.closure = self.closure or (lambda: closure_files(r))


def check_owner(fz: dict, ctx: Ctx) -> list:
    """Шаг 1: `owner_yes` = {zh, date, punkt} и строка «да» с этой датой в журнале под этим номером."""
    oy = fz.get("owner_yes")
    if not isinstance(oy, dict) or not all(oy.get(k) for k in ("zh", "date", "punkt")):
        return ["в FREEZE.json нет owner_yes (zh, date, punkt)"]
    if not re.fullmatch(r"Ж-\d+", str(oy["zh"])):
        return [f"owner_yes.zh не вида Ж-NNN: {oy['zh']}"]
    try:
        dd = date.fromisoformat(str(oy["date"])).strftime("%d.%m.%Y")
    except ValueError:
        return [f"owner_yes.date не ГГГГ-ММ-ДД: {oy['date']}"]
    if not ctx.resheniya.is_file():
        return [f"нет журнала {ctx.resheniya}"]
    phrase = YES_PHRASES.get(oy["zh"], YES_PHRASE)
    parts = re.split(r"(?m)^### ", ctx.resheniya.read_text(encoding="utf-8"))
    for body in parts[1:]:
        head = body.split("\n", 1)[0]
        if head.startswith(f"{oy['zh']} ") and dd in head and phrase in body:
            return []
    return [f"в журнале нет строки «да» {oy['zh']} от {dd} ({phrase})"]


def _hash_problem(ctx: Ctx, key: str, want) -> Optional[str]:
    p = Path(key) if Path(key).is_absolute() else ctx.root / key
    if not p.is_file():
        return f"{key}: файла нет"
    want = want.get("sha256", "") if isinstance(want, dict) else want
    if not isinstance(want, str) or len(want) < 16:
        return f"{key}: в FREEZE нет sha256"
    got = ctx.content_hash(p) if p.suffix == ".npz" and "stageA" in p.parts else sha256_file(p)
    return None if got.startswith(want.lower()) else f"{key}: sha256 {got[:12]} ≠ заморожённому {want[:12]}"


def check_sha(fz: dict, ctx: Ctx) -> list:
    """Шаг 2: код (с полнотой замыкания), входы архива, манифесты и файлы новых данных (по байтам), пакеты, REHEARSAL, POWER."""
    bad = []
    code = fz.get("code_sha256") or {}
    bad += [f"код вне code_sha256: {p}" for p in sorted(ctx.closure() - set(code))]
    for grp in ("code_sha256", "inputs_archive_sha256", "derived_archive_sha256", "inputs_new_sha256"):
        bad += [m for k, v in (fz.get(grp) or {}).items() if (m := _hash_problem(ctx, k, v))]
    have = ctx.versions()
    want = fz.get("packages") or {}
    bad += [f"пакет {n}: {have.get(n)} ≠ заморожённому {want.get(n)}" for n in PACKAGES if want.get(n) is None or have.get(n) != want.get(n)]
    for key, path in (("rehearsal", ctx.rehearsal), ("power", ctx.power)):
        w = (fz.get(key) or {}).get("sha256")
        if not path.is_file() or not w or sha256_file(path) != w:
            bad.append(f"{key}: sha256 {path.name} не совпадает с FREEZE")
    return bad


def check_free(ctx: Ctx) -> list:
    """Шаг 4: замка (и файла причины .err) нет — открытие одно; выходов нет."""
    bad = [f"замок {ctx.lock.name} уже есть — открытие было"] if ctx.lock.exists() or ctx.lock.with_suffix(".err").exists() else []
    bad += [f"выход уже есть: {p}" for p in sorted(ctx.rep_dir.glob("FWD_*"))]
    bad += [f"выход уже есть: {p}" for p in sorted(ctx.new_dir.glob("*"))] if ctx.new_dir.exists() else []
    return bad


def preflight(ctx: Ctx) -> list:
    if not ctx.freeze.is_file():
        return [f"нет {ctx.freeze.name}"]
    fz = json.loads(ctx.freeze.read_text(encoding="utf-8"))
    bad = [f"в {ctx.freeze.name} нет ключа {k}" for k in REQ if k not in fz]
    if fz.get("v") != (2 if ctx.povtor else 1) or fz.get("candidates") != ["X1", "ZB"] or fz.get("controls") != ["K1", "K2", "K3"]:
        bad.append("FREEZE: версия, кандидаты (X1, ZB) или контроли (K1, K2, K3) не по протоколу")
    gates = G.check_gates(ctx.gates, *((G.POVTOR_LISTS, G.POVTOR_SKIP) if ctx.povtor else ()))
    return (bad + (PT.check_povtor(fz, ctx) if ctx.povtor else []) + check_owner(fz, ctx) + G.check_required(fz, ctx.required()) + check_sha(fz, ctx)
            + G.check_rehearsal(fz, ctx.rehearsal, ctx.rh_problems) + gates + check_free(ctx))
