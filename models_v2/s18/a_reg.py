"""S-A-IMPL-3 (Ж-254; `S_A_SPEC.md`, разд. 7): регистратор 7 строк S-A и сторож «реестр до прогноза». По образцу `s16/x1_w5_reg`.
Публичное API:
- `ROWS` / `IDS` — 7 строк разд. 7 по id (S-A0, S-A1, S-A2, S-A2p, S-A3, S-A2s, S-A2g); S-A3, S-A2s, S-A2g — «не запускается в пилоте; в N не входит, пока не посчитана». `N_PROGRAM` = 831, `N_WIDE` = 3367, `SPEC` = docs/otchety/harj_struct/S_A_SPEC.md, `A_DIR` = data/regen/harj_struct/A, `KEY` = "S.A.registered".
- `register(now=None, trials=TRIALS, a_dir=A_DIR, ckpt=CKPT, spec=SPEC) -> 7`: дописывает ВСЕ 7 строк разом с ОДНОЙ `registered_at` (часы машины, формат «ДД.ММ.ГГГГ ЧЧ:ММ:СС») в ALL_TRIALS.csv и сливает в CHECKPOINT.json ключ `S.A.registered` = {spec_sha256, registered_at, n, n_wide, rows}
  (прочие ключи — побайтно; формат файла indent 1, sorted keys; несовпадение формата — отказ до любой записи). Отказ (SystemExit): в корне `a_dir` уже есть `*.parquet` (реестр позже прогноза), или хоть один из id уже в реестре. `ckpt=None` — без записи CHECKPOINT.
- `merge_checkpoint(ckpt, now, spec=SPEC)`: только ключ `S.A.registered`.
- `require(cfg, trials=TRIALS)`: отказ (SystemExit), если id нет в реестре (для a_walk — первым действием).
- `check(trials=TRIALS, a_dir=A_DIR, ckpt=CKPT) -> list` замечаний, только чтение: ровно 7 id (по id, не по позиции), без дублей, одна `registered_at`; ключ `S.A.registered` с n = 831, n_wide = 3367, rows = 7 и той же `registered_at`;
  если в корне `a_dir` есть `*.parquet` — `registered_at` раньше mtime самого раннего из них (и раньше времени ключа `S.A.eval`, если у него есть время).
Запуск: `python3 -W ignore -m models_v2.s18.a_reg [--check]`; без флага — регистрация (однократная)."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

from models_v2.s16 import x1_reg as R

ROOT, REP = R.ROOT, R.REP
TRIALS, COLS, now_text = R.TRIALS, R.COLS, R.now_text
CKPT = REP / "CHECKPOINT.json"
SPEC = ROOT / "docs/otchety/harj_struct/S_A_SPEC.md"
A_DIR = ROOT / "data/regen/harj_struct/A"
KEY = "S.A.registered"
N_PROGRAM, N_WIDE = 831, 3367
NOT_RUN = "не запускается в пилоте; в N не входит, пока не посчитана"
ROWS = [
    ("S-A0", "S: стандартная память + корректор X1, переоценка под QLIKE", "F = W5(X1(B_mix, C_A)); C_A = H_A·exp((w⁸+δw)ℓ_A + (1+δκ)m); H_A — разд. 3.1 при a = HAR, b = 1", "δ (5), δw, δκ; λ PIT", "зарегистрирована"),
    ("S-A1", "S: обучаемая лаговая память", "как S-A0, a = (1−ω)R + ωG(ψ), L = 5/22/66", "+ ω, ψ (9) = 16", "зарегистрирована"),
    ("S-A2", "S: совместная лаговая и внутридневная память (основная)", "как S-A1, RV_b с b_k(τ) = exp(h_k)/mean_present, h — натуральный сплайн, 5 узлов", "+ c (12) = 28", "зарегистрирована"),
    ("S-A2p", "S: контроль (б) — τ переставлено", "S-A2 с τ_{π(j)}, π = default_rng(0).permutation(53)", "28", "контроль"),
    ("S-A3", "S: условная по состоянию память", "ω_k = ω⁰_k·σ(γᵀz_t), z_t = (ln RV_d − PIT-медиана, e_d)", "—", NOT_RUN),
    ("S-A2s", "S: одна b на три памяти", "b_s = b_m = b_l", "—", NOT_RUN),
    ("S-A2g", "S: бустинг-корректор с cross-fitting", "корректор — бустинг с временным cross-fitting вместо дифференцируемого g", "—", NOT_RUN),
]
IDS = tuple(r[0] for r in ROWS)
assert len(ROWS) == 7 and len(set(IDS)) == 7


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _read(trials: Path) -> pd.DataFrame:
    return pd.read_csv(trials, comment="#", dtype=str, keep_default_na=False) if trials.exists() else pd.DataFrame(columns=COLS)


def _parquet(a_dir: Path) -> list:
    return sorted(p.name for p in a_dir.glob("*.parquet")) if a_dir.is_dir() else []


def _dump(cp: dict) -> str:
    return json.dumps(cp, indent=1, sort_keys=True, ensure_ascii=False)


def merge_checkpoint(ckpt: Path, now: str, spec: Path = SPEC) -> None:
    """Только ключ `S.A.registered`, слиянием (indent 1, sorted keys); прочие ключи сохраняются."""
    cp = json.loads(ckpt.read_text(encoding="utf-8"))
    cp[KEY] = dict(spec_sha256=_sha(spec), registered_at=now, n=N_PROGRAM, n_wide=N_WIDE, rows=len(ROWS))
    ckpt.write_text(_dump(cp), encoding="utf-8")


def register(now: str | None = None, trials: Path = TRIALS, a_dir: Path = A_DIR, ckpt: Path | None = CKPT, spec: Path = SPEC) -> int:
    """Дописывает ВСЕ 7 строк разом (одна `registered_at`), затем сливает ключ CHECKPOINT. Отказ (SystemExit): в корне a_dir есть *.parquet, любой id уже в реестре, формат CHECKPOINT не indent 1 / sorted keys."""
    pq = _parquet(a_dir)
    if pq:
        raise SystemExit("реестр позже прогноза: в %s уже есть %s — регистрация отказана" % (a_dir, pq))
    have = sorted(set(IDS) & set(_read(trials)["id"]))
    if have:
        raise SystemExit("уже зарегистрированы: %s — повторная регистрация отказана" % have)
    if ckpt is not None and _dump(json.loads(ckpt.read_text(encoding="utf-8"))) != ckpt.read_text(encoding="utf-8"):
        raise SystemExit("CHECKPOINT.json: формат не совпадает с indent 1 / sorted keys — слияние ключа сломало бы прочие ключи")
    now = now or now_text()
    text = trials.read_text(encoding="utf-8")
    with trials.open("a", encoding="utf-8", newline="") as fh:
        fh.write(("" if text.endswith("\n") else "\n") + pd.DataFrame([dict(zip(COLS, (*r, now))) for r in ROWS])[COLS].to_csv(index=False, header=False, lineterminator="\n"))
    if ckpt is not None:
        merge_checkpoint(ckpt, now, spec)
    return len(ROWS)


def require(cfg: str, trials: Path = TRIALS) -> None:
    if cfg not in set(_read(trials)["id"]):
        raise SystemExit("%s: нет в реестре ALL_TRIALS.csv — счёт отказан (реестр до прогноза)" % cfg)


def _t(s: str) -> dt.datetime:
    return dt.datetime.strptime(s, "%d.%m.%Y %H:%M:%S")


def _key_time(cp: dict, key: str):
    v = cp.get(key)
    for f in ("written_at", "time", "timestamp", "created_at", "at"):
        if isinstance(v, dict) and isinstance(v.get(f), str):
            return _t(v[f])
    return None


def check(trials: Path = TRIALS, a_dir: Path = A_DIR, ckpt: Path = CKPT) -> list:
    cur = _read(trials)
    rows = cur[cur["id"].isin(IDS)]
    bad = []
    if len(rows) != len(IDS) or set(rows["id"]) != set(IDS) or rows["id"].duplicated().any():
        bad.append("строк S-A в реестре %d из %d (ожидаются ровно id разд. 7)" % (len(rows), len(IDS)))
    reg = sorted(rows["registered_at"].unique())
    if len(reg) != 1:
        bad.append("registered_at не одна: %s" % reg)
    cp = json.loads(ckpt.read_text(encoding="utf-8")) if ckpt.exists() else {}
    k = cp.get(KEY) or {}
    if not (k.get("n") == N_PROGRAM and k.get("n_wide") == N_WIDE and k.get("rows") == len(IDS) and [k.get("registered_at")] == reg):
        bad.append("ключ %s в CHECKPOINT: %s (ожидается n = %d, n_wide = %d, rows = 7, registered_at строк реестра)" % (KEY, k, N_PROGRAM, N_WIDE))
    ps = [a_dir / n for n in _parquet(a_dir)]
    if ps and len(reg) == 1:
        first = min(dt.datetime.fromtimestamp(p.stat().st_mtime) for p in ps)
        bad += [] if _t(reg[0]) < first else ["registered_at %s не раньше mtime самого раннего parquet %s" % (reg[0], first)]
        kt = _key_time(cp, "S.A.eval")
        bad += [] if kt is None or _t(reg[0]) < kt else ["registered_at не раньше времени ключа S.A.eval"]
    return bad


if __name__ == "__main__":
    if "--check" in sys.argv:
        b = check()
        r = _read(TRIALS)
        r = r[r["id"].isin(IDS)]
        print("check %s: строк %d; registered_at %s; parquet в A: %d" % ("ok" if not b else "FAIL " + "; ".join(b), len(r), sorted(r["registered_at"].unique()), len(_parquet(A_DIR))))
        sys.exit(1 if b else 0)
    print("реестр S-A: добавлено строк %d" % register())
