"""S-D-IMPL-2 (Ж-256; `S_D_SPEC.md`, разд. 4): регистратор 2 строк абляций S-RCKm, S-RCKf и сторож «реестр до прогноза». По образцу `r_reg`.
Публичное API:
- `ROWS` / `IDS` — 2 строки разд. 4 по id; `KEY` = "S.D.registered"; `SPEC`, `D_DIR` = data/regen/harj_struct/D; `ADD` = 2, `ADD_WIDE` = 4.
- `register(now=None, trials=TRIALS, d_dir=D_DIR, ckpt=CKPT, spec=SPEC) -> 2`: дописывает обе строки разом с ОДНОЙ `registered_at` (часы машины) и сливает ключ = {spec_sha256, registered_at, n, n_wide, n_base, n_wide_base, rows, prior_bytes, prior_sha256}.
  N берётся из CHECKPOINT: максимум `n` / `n_wide` по ключам `S.*.registered` (кроме своего) + 2 / + 4. Отказ до любой записи: в `d_dir` есть `*.parquet` («реестр позже прогноза», № 30); id уже в реестре; формат CHECKPOINT чужой.
- `require(cfg, trials=TRIALS)`: отказ, если id нет в реестре (первое действие `d_walk.run`).
- `check(trials, d_dir, ckpt) -> list`, только чтение: ровно 2 id (по id, не по позиции), без дублей, одна `registered_at`; ключ: rows = 2, n = n_base + 2, n_wide = n_wide_base + 4; прежняя часть реестра по prior_sha256;
  `registered_at` раньше mtime самого раннего parquet в `d_dir` и раньше времени ключа `S.D.eval`.
Запуск: `python3 -W ignore -m models_v2.s18.d_reg [--check]`; без флага — регистрация (однократная)."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

from models_v2.s18 import r_reg as RR

ROOT, TRIALS, COLS, now_text, CKPT = RR.ROOT, RR.TRIALS, RR.COLS, RR.now_text, RR.CKPT
SPEC = ROOT / "docs/otchety/harj_struct/S_D_SPEC.md"
D_DIR = ROOT / "data/regen/harj_struct/D"
KEY, ADD, ADD_WIDE = "S.D.registered", 2, 4
ROWS = [
    ("S-RCKm", "S: абляция S-RCK — без поправки уровня m", "S-RCK при m_T := 1 (обрезка c_T = Q₀,₉₉ RV/W5-G сохраняется), потеря ρ_{c_T}(ln(RV/F^fr))", "6; включение 630", "абляция (диагностика вклада, не кандидат)"),
    ("S-RCKf", "S: абляция S-RCK — потеря на уровне C", "S-RCK с потерей обучения ρ_κ(ln(RV/(m·C))), c, m по RV/cfg8 (π := 1, β := 0 только в обучении); валидация — QLIKE уровня F, как в R", "6; включение 630", "абляция (диагностика вклада, не кандидат)"),
]
IDS = tuple(r[0] for r in ROWS)
assert len(ROWS) == 2 and len(set(IDS)) == 2


def _base(cp: dict, field: str) -> int:
    v = [k[field] for n, k in cp.items() if n.startswith("S.") and n.endswith(".registered") and n != KEY and isinstance(k, dict) and field in k]
    if not v:
        raise SystemExit("в CHECKPOINT нет ключей S.*.registered — N не определить")
    return max(v)


def merge_checkpoint(ckpt: Path, now: str, spec: Path, prior: bytes) -> dict:
    cp = json.loads(ckpt.read_text(encoding="utf-8"))
    nb, wb = _base(cp, "n"), _base(cp, "n_wide")
    cp[KEY] = dict(spec_sha256=RR._sha(spec), registered_at=now, n=nb + ADD, n_wide=wb + ADD_WIDE, n_base=nb, n_wide_base=wb, rows=len(ROWS), prior_bytes=len(prior), prior_sha256=hashlib.sha256(prior).hexdigest())
    ckpt.write_text(RR._dump(cp), encoding="utf-8")
    return cp[KEY]


def register(now: str | None = None, trials: Path = TRIALS, d_dir: Path = D_DIR, ckpt: Path | None = CKPT, spec: Path = SPEC) -> int:
    pq = RR._parquet(d_dir)
    if pq:
        raise SystemExit("реестр позже прогноза: в %s уже есть %s — регистрация отказана" % (d_dir, pq))
    have = sorted(set(IDS) & set(RR._read(trials)["id"]))
    if have:
        raise SystemExit("уже зарегистрированы: %s — повторная регистрация отказана" % have)
    if ckpt is not None and RR._dump(json.loads(ckpt.read_text(encoding="utf-8"))) != ckpt.read_text(encoding="utf-8"):
        raise SystemExit("CHECKPOINT.json: формат не совпадает с indent 1 / sorted keys")
    now = now or now_text()
    prior = trials.read_bytes()
    with trials.open("a", encoding="utf-8", newline="") as fh:
        fh.write(("" if prior.endswith(b"\n") else "\n") + pd.DataFrame([dict(zip(COLS, (*r, now))) for r in ROWS])[COLS].to_csv(index=False, header=False, lineterminator="\n"))
    if ckpt is not None:
        merge_checkpoint(ckpt, now, spec, prior)
    return len(ROWS)


def require(cfg: str, trials: Path = TRIALS) -> None:
    if cfg not in set(RR._read(trials)["id"]):
        raise SystemExit("%s: нет в реестре ALL_TRIALS.csv — счёт отказан (реестр до прогноза)" % cfg)


def check(trials: Path = TRIALS, d_dir: Path = D_DIR, ckpt: Path = CKPT) -> list:
    cur = RR._read(trials)
    rows = cur[cur["id"].isin(IDS)]
    bad = []
    if len(rows) != len(IDS) or set(rows["id"]) != set(IDS) or rows["id"].duplicated().any():
        bad.append("строк S-D в реестре %d из %d (ожидаются ровно id разд. 4)" % (len(rows), len(IDS)))
    reg = sorted(rows["registered_at"].unique())
    if len(reg) != 1:
        bad.append("registered_at не одна: %s" % reg)
    cp = json.loads(ckpt.read_text(encoding="utf-8")) if ckpt.exists() else {}
    k = cp.get(KEY) or {}
    ok = k.get("rows") == len(IDS) and [k.get("registered_at")] == reg and isinstance(k.get("n_base"), int) and isinstance(k.get("n_wide_base"), int)
    if not (ok and k.get("n") == k["n_base"] + ADD and k.get("n_wide") == k["n_wide_base"] + ADD_WIDE):
        bad.append("ключ %s в CHECKPOINT: %s (ожидается rows = 2, n = n_base + 2, n_wide = n_wide_base + 4, registered_at строк реестра)" % (KEY, k))
    raw = trials.read_bytes() if trials.exists() else b""
    if "prior_bytes" in k and hashlib.sha256(raw[: k["prior_bytes"]]).hexdigest() != k.get("prior_sha256"):
        bad.append("прежняя часть реестра изменена (sha256 первых %s байт ≠ записанному)" % k["prior_bytes"])
    ps = [d_dir / n for n in RR._parquet(d_dir)]
    if ps and len(reg) == 1:
        import datetime as dt
        first = min(dt.datetime.fromtimestamp(p.stat().st_mtime) for p in ps)
        bad += [] if RR._t(reg[0]) < first else ["registered_at %s не раньше mtime самого раннего parquet %s" % (reg[0], first)]
        kt = RR._key_time(cp, "S.D.eval")
        bad += [] if kt is None or RR._t(reg[0]) < kt else ["registered_at не раньше времени ключа S.D.eval"]
    return bad


if __name__ == "__main__":
    if "--check" in sys.argv:
        b = check()
        r = RR._read(TRIALS)
        r = r[r["id"].isin(IDS)]
        k = (json.loads(CKPT.read_text(encoding="utf-8")).get(KEY) or {})
        print("check %s: строк %d; registered_at %s; parquet в D: %d; N %s / N_wide %s" % ("ok" if not b else "FAIL " + "; ".join(b), len(r), sorted(r["registered_at"].unique()), len(RR._parquet(D_DIR)), k.get("n"), k.get("n_wide")))
        sys.exit(1 if b else 0)
    print("реестр S-D: добавлено строк %d" % register())
