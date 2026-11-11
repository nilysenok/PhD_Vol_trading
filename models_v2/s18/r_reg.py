"""S-R-IMPL-2 (Ж-254 пп. 6, 10–13; `S_R_SPEC.md`, разд. 5): регистратор 12 строк (6 S-R + 2 «не запускается» + 4 S-B) и сторож «реестр до прогноза». По образцу `cm_reg`.
Публичное API:
- `ROWS` / `IDS` — 12 строк разд. 5 по id; `KEY` = "S.R.registered"; `SPEC`, `R_DIR` = data/regen/harj_struct/R; `ADD` = 10 (6 S-R + 4 S-B; S-RA2u, S-RCKu в N не входят), `ADD_WIDE` = 20.
- `register(now=None, trials=TRIALS, r_dir=R_DIR, ckpt=CKPT, spec=SPEC) -> 12`: дописывает ВСЕ 12 строк разом с ОДНОЙ `registered_at` (часы машины) и сливает ключ = {spec_sha256, registered_at, n, n_wide, n_base, n_wide_base, rows,
  prior_bytes, prior_sha256 (прежняя часть реестра — для проверки «побайтно не изменена»)}. N берётся из CHECKPOINT: максимум `n` / `n_wide` по ключам `S.*.registered` (кроме своего) + 10 / + 20.
  Отказ до любой записи: в `r_dir` есть `*.parquet` («реестр позже прогноза», № 30); любой id уже в реестре; формат CHECKPOINT чужой.
- `require(cfg, trials=TRIALS)`: отказ, если id нет в реестре (первое действие `r_walk.run`).
- `check(trials, r_dir, ckpt) -> list`, только чтение: ровно 12 id (по id, не по позиции), без дублей, одна `registered_at`; ключ: rows = 12, n = n_base + 10, n_wide = n_wide_base + 20; прежняя часть реестра по prior_sha256;
  `registered_at` раньше mtime самого раннего parquet в `r_dir` и раньше времени ключа `S.R.eval`.
Запуск: `python3 -W ignore -m models_v2.s18.r_reg [--check]`; без флага — регистрация (однократная)."""
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
SPEC = ROOT / "docs/otchety/harj_struct/S_R_SPEC.md"
R_DIR = ROOT / "data/regen/harj_struct/R"
KEY, ADD, ADD_WIDE = "S.R.registered", 10, 20
NOT_RUN = "не запускается; в N не входит, пока не посчитана"
SB_F = "`S_B_DECISION.md` разд. 2.1; критерий sha256 e2953ec7…f7844"
SB_S = "диагностика; после расчёта диагностики (11.11 18:08); прогнозов нет; не «проверено отрицательно»"
FR = "F = W5(X1(B_mix, C_A)), C_A как S-A2; θ = argmin mean ρ_κ(ln(RV/(m·F^fr))) + pen; F^fr = B_mix^(1−π)C_A^π e^{βg} (π, β сохранённые PIT); c = Q₀,₉₉ PIT, m = mean min(x⁰, c); Λ = {0, 1e-1, 1e-2, 1e-3}; 3 блока × 126, допуск 2 из 3"
ROWS = [
    ("S-RA2", "S: A2 + устойчивая процедура R (основная)", FR, "28; включение 630 пар", "зарегистрирована"),
    ("S-RCK", "S: C-K + устойчивая процедура R", "как S-RA2 с C = cfg8·𝓜_K (S-CK, μ = 0)", "6; включение 630", "зарегистрирована"),
    ("S-RA2b", "S: абляция — R без обрезки", "S-RA2 с ℓ вместо ρ_κ (m = 1)", "28", "абляция"),
    ("S-RCKb", "S: абляция — R без обрезки", "S-RCK с ℓ вместо ρ_κ", "6", "абляция"),
    ("S-RA2c", "S: абляция — обрезка без блоков", "S-RA2, одна валидация 126, argmin, ничья — θ = 0", "28", "абляция"),
    ("S-RCKc", "S: абляция — обрезка без блоков", "S-RCK, одна валидация 126", "6", "абляция"),
    ("S-RA2u", "S: R на уровне C", "S-RA2 при π := 1, β := 0", "—", NOT_RUN),
    ("S-RCKu", "S: R на уровне C", "S-RCK при π := 1, β := 0", "—", NOT_RUN),
    ("S-B1", "S: диагностика B-lag, A1 × ln q5 (оракул в выборке)", SB_F, "E −0,0203 %, p 0,527, Холм m = 4: 1,000", SB_S),
    ("S-B2", "S: диагностика B-lag, A1 × u", SB_F, "E −0,2677 %, p 0,920, Холм m = 4: 1,000", SB_S),
    ("S-B3", "S: диагностика B-lag, A2 × ln q5", SB_F, "E +0,0107 %, p 0,479, Холм m = 4: 1,000", SB_S),
    ("S-B4", "S: диагностика B-lag, A2 × u", SB_F, "E −0,1152 %, p 0,663, Холм m = 4: 1,000", SB_S),
]
IDS = tuple(r[0] for r in ROWS)
assert len(ROWS) == 12 and len(set(IDS)) == 12


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _read(trials: Path) -> pd.DataFrame:
    return pd.read_csv(trials, comment="#", dtype=str, keep_default_na=False) if trials.exists() else pd.DataFrame(columns=COLS)


def _parquet(r_dir: Path) -> list:
    return sorted(p.name for p in r_dir.glob("*.parquet")) if r_dir.is_dir() else []


def _dump(cp: dict) -> str:
    return json.dumps(cp, indent=1, sort_keys=True, ensure_ascii=False)


def _base(cp: dict, field: str) -> int:
    v = [k[field] for n, k in cp.items() if n.startswith("S.") and n.endswith(".registered") and n != KEY and isinstance(k, dict) and field in k]
    if not v:
        raise SystemExit("в CHECKPOINT нет ключей S.*.registered — N не определить")
    return max(v)


def merge_checkpoint(ckpt: Path, now: str, spec: Path, prior: bytes) -> dict:
    cp = json.loads(ckpt.read_text(encoding="utf-8"))
    nb, wb = _base(cp, "n"), _base(cp, "n_wide")
    cp[KEY] = dict(spec_sha256=_sha(spec), registered_at=now, n=nb + ADD, n_wide=wb + ADD_WIDE, n_base=nb, n_wide_base=wb, rows=len(ROWS), prior_bytes=len(prior), prior_sha256=hashlib.sha256(prior).hexdigest())
    ckpt.write_text(_dump(cp), encoding="utf-8")
    return cp[KEY]


def register(now: str | None = None, trials: Path = TRIALS, r_dir: Path = R_DIR, ckpt: Path | None = CKPT, spec: Path = SPEC) -> int:
    pq = _parquet(r_dir)
    if pq:
        raise SystemExit("реестр позже прогноза: в %s уже есть %s — регистрация отказана" % (r_dir, pq))
    have = sorted(set(IDS) & set(_read(trials)["id"]))
    if have:
        raise SystemExit("уже зарегистрированы: %s — повторная регистрация отказана" % have)
    if ckpt is not None and _dump(json.loads(ckpt.read_text(encoding="utf-8"))) != ckpt.read_text(encoding="utf-8"):
        raise SystemExit("CHECKPOINT.json: формат не совпадает с indent 1 / sorted keys")
    now = now or now_text()
    prior = trials.read_bytes()
    with trials.open("a", encoding="utf-8", newline="") as fh:
        fh.write(("" if prior.endswith(b"\n") else "\n") + pd.DataFrame([dict(zip(COLS, (*r, now))) for r in ROWS])[COLS].to_csv(index=False, header=False, lineterminator="\n"))
    if ckpt is not None:
        merge_checkpoint(ckpt, now, spec, prior)
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


def check(trials: Path = TRIALS, r_dir: Path = R_DIR, ckpt: Path = CKPT) -> list:
    cur = _read(trials)
    rows = cur[cur["id"].isin(IDS)]
    bad = []
    if len(rows) != len(IDS) or set(rows["id"]) != set(IDS) or rows["id"].duplicated().any():
        bad.append("строк S-R / S-B в реестре %d из %d (ожидаются ровно id разд. 5)" % (len(rows), len(IDS)))
    reg = sorted(rows["registered_at"].unique())
    if len(reg) != 1:
        bad.append("registered_at не одна: %s" % reg)
    cp = json.loads(ckpt.read_text(encoding="utf-8")) if ckpt.exists() else {}
    k = cp.get(KEY) or {}
    ok = k.get("rows") == len(IDS) and [k.get("registered_at")] == reg and isinstance(k.get("n_base"), int) and isinstance(k.get("n_wide_base"), int)
    if not (ok and k.get("n") == k["n_base"] + ADD and k.get("n_wide") == k["n_wide_base"] + ADD_WIDE):
        bad.append("ключ %s в CHECKPOINT: %s (ожидается rows = 12, n = n_base + 10, n_wide = n_wide_base + 20, registered_at строк реестра)" % (KEY, k))
    raw = trials.read_bytes() if trials.exists() else b""
    if "prior_bytes" in k and hashlib.sha256(raw[: k["prior_bytes"]]).hexdigest() != k.get("prior_sha256"):
        bad.append("прежняя часть реестра изменена (sha256 первых %s байт ≠ записанному)" % k["prior_bytes"])
    ps = [r_dir / n for n in _parquet(r_dir)]
    if ps and len(reg) == 1:
        first = min(dt.datetime.fromtimestamp(p.stat().st_mtime) for p in ps)
        bad += [] if _t(reg[0]) < first else ["registered_at %s не раньше mtime самого раннего parquet %s" % (reg[0], first)]
        kt = _key_time(cp, "S.R.eval")
        bad += [] if kt is None or _t(reg[0]) < kt else ["registered_at не раньше времени ключа S.R.eval"]
    return bad


if __name__ == "__main__":
    if "--check" in sys.argv:
        b = check()
        r = _read(TRIALS)
        r = r[r["id"].isin(IDS)]
        k = (json.loads(CKPT.read_text(encoding="utf-8")).get(KEY) or {})
        print("check %s: строк %d; registered_at %s; parquet в R: %d; N %s / N_wide %s" % ("ok" if not b else "FAIL " + "; ".join(b), len(r), sorted(r["registered_at"].unique()), len(_parquet(R_DIR)), k.get("n"), k.get("n_wide")))
        sys.exit(1 if b else 0)
    print("реестр S-R: добавлено строк %d" % register())
