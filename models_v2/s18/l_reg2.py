"""S-L-IMPL-2, добавка по Ж-257 п. 3: регистратор 1 строки S-RL1b2 (контроль (б) со сдвигом π на 252 строки БЕЗ переноса через конец; первые 252 пары исключены из пула обучения) до счёта `rl1b2.parquet`.
Строка и ключ `S.L2.registered` — отдельной записью; `l_reg` и зарегистрированные S-RL1, S-RL1a, S-RL1b не меняются. N = max `n` по `S.*.registered` (кроме своего: 851) + 1 = 852; N_wide + 2 (3409).
API как у `l_reg`: `ROWS`, `IDS`, `KEY`, `ADD` = 1, `ADD_WIDE` = 2, `register`, `require`, `check` (registered_at раньше mtime `rl1b2.parquet` и ключа `S.L.eval`; отказ регистрации, если `rl1b2.parquet` уже есть). Запуск: `python3 -W ignore -m models_v2.s18.l_reg2 [--check]`."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

from models_v2.s18 import l_reg as LR
from models_v2.s18 import r_reg as RR

TRIALS, COLS, CKPT, SPEC, L_DIR = LR.TRIALS, LR.COLS, LR.CKPT, LR.SPEC, LR.L_DIR
KEY, ADD, ADD_WIDE, OUT = "S.L2.registered", 1, 2, "rl1b2.parquet"
ROWS = [("S-RL1b2", "S: контроль (б2) к S-RL1 — π на 252 строки назад без переноса через конец", "как S-RL1b, но π_i := π_{i−252} (shift, не roll); первые 252 строки D_M исключены из пула обучения (не заполняются, Д4); c, m по RV/F_train(0); валидация и слои — настоящий π",
         "1; включение 630", "контроль (не кандидат); добавлен по Ж-257 п. 3 до счёта; заменяет циклический S-RL1b как основной контроль (б)")]
IDS = tuple(r[0] for r in ROWS)


def merge_checkpoint(ckpt: Path, now: str, spec: Path, prior: bytes) -> dict:
    cp = json.loads(ckpt.read_text(encoding="utf-8"))
    nb, wb = _max(cp, "n"), _max(cp, "n_wide")
    cp[KEY] = dict(spec_sha256=RR._sha(spec), registered_at=now, n=nb + ADD, n_wide=wb + ADD_WIDE, n_base=nb, n_wide_base=wb, rows=len(ROWS), prior_bytes=len(prior), prior_sha256=hashlib.sha256(prior).hexdigest())
    ckpt.write_text(RR._dump(cp), encoding="utf-8")
    return cp[KEY]


def _max(cp: dict, field: str) -> int:
    v = [k[field] for n, k in cp.items() if n.startswith("S.") and n.endswith(".registered") and n != KEY and isinstance(k, dict) and field in k]
    if not v:
        raise SystemExit("в CHECKPOINT нет ключей S.*.registered — N не определить")
    return max(v)


def register(now: str | None = None, trials: Path = TRIALS, l_dir: Path = L_DIR, ckpt: Path | None = CKPT, spec: Path = SPEC) -> int:
    if (l_dir / OUT).exists():
        raise SystemExit("реестр позже прогноза: %s уже есть — регистрация отказана" % (l_dir / OUT))
    if set(IDS) & set(RR._read(trials)["id"]):
        raise SystemExit("уже зарегистрированы: %s — повторная регистрация отказана" % list(IDS))
    if ckpt is not None and RR._dump(json.loads(ckpt.read_text(encoding="utf-8"))) != ckpt.read_text(encoding="utf-8"):
        raise SystemExit("CHECKPOINT.json: формат не совпадает с indent 1 / sorted keys")
    now, prior = now or RR.now_text(), trials.read_bytes()
    with trials.open("a", encoding="utf-8", newline="") as fh:
        fh.write(("" if prior.endswith(b"\n") else "\n") + pd.DataFrame([dict(zip(COLS, (*r, now))) for r in ROWS])[COLS].to_csv(index=False, header=False, lineterminator="\n"))
    if ckpt is not None:
        merge_checkpoint(ckpt, now, spec, prior)
    return len(ROWS)


def require(cfg: str, trials: Path = TRIALS) -> None:
    LR.require(cfg, trials)


def check(trials: Path = TRIALS, l_dir: Path = L_DIR, ckpt: Path = CKPT) -> list:
    cur = RR._read(trials)
    rows = cur[cur["id"].isin(IDS)]
    bad = [] if len(rows) == 1 and not rows["id"].duplicated().any() else ["строк S-RL1b2 в реестре %d из 1" % len(rows)]
    reg = sorted(rows["registered_at"].unique())
    cp = json.loads(ckpt.read_text(encoding="utf-8")) if ckpt.exists() else {}
    k = cp.get(KEY) or {}
    ok = k.get("rows") == 1 and [k.get("registered_at")] == reg and isinstance(k.get("n_base"), int) and isinstance(k.get("n_wide_base"), int)
    if not (ok and k.get("n") == k["n_base"] + ADD and k.get("n_wide") == k["n_wide_base"] + ADD_WIDE):
        bad.append("ключ %s в CHECKPOINT: %s (ожидается rows = 1, n = n_base + 1, n_wide = n_wide_base + 2)" % (KEY, k))
    raw = trials.read_bytes() if trials.exists() else b""
    if "prior_bytes" in k and hashlib.sha256(raw[: k["prior_bytes"]]).hexdigest() != k.get("prior_sha256"):
        bad.append("прежняя часть реестра изменена (sha256 первых %s байт ≠ записанному)" % k["prior_bytes"])
    if (l_dir / OUT).exists() and len(reg) == 1:
        first = dt.datetime.fromtimestamp((l_dir / OUT).stat().st_mtime)
        bad += [] if RR._t(reg[0]) < first else ["registered_at %s не раньше mtime %s" % (reg[0], OUT)]
    kt = RR._key_time(cp, "S.L.eval")
    bad += [] if kt is None or not reg or RR._t(reg[0]) < kt else ["registered_at не раньше времени ключа S.L.eval"]
    return bad


if __name__ == "__main__":
    if "--check" in sys.argv:
        b = check()
        k = json.loads(CKPT.read_text(encoding="utf-8")).get(KEY) or {}
        print("check %s: N %s / N_wide %s; rl1b2.parquet: %s" % ("ok" if not b else "FAIL " + "; ".join(b), k.get("n"), k.get("n_wide"), (L_DIR / OUT).exists()))
        sys.exit(1 if b else 0)
    print("реестр S-RL1b2: добавлено строк %d" % register())
    k = json.loads(CKPT.read_text(encoding="utf-8"))[KEY]
    print("N = %s / N_wide = %s" % (k["n"], k["n_wide"]))
