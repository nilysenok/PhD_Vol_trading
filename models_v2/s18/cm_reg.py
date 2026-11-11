"""S-C-IMPL-3 (Ж-254 пп. 8–12; `S_C_SPEC.md`, разд. 6): регистратор 9 строк S-C и сторож «реестр до прогноза». По образцу `a_reg` (S-A).
Публичное API:
- `ROWS` / `IDS` — 9 строк разд. 6 по id: S-CD, S-CK, S-CKA, S-C7, S-CKL (считаются), S-C2f, S-C7i, S-CDx, S-CDw («не запускается; в N не входит, пока не посчитана»). `KEY` = "S.C.registered", `SPEC`, `CM_DIR` = data/regen/harj_struct/CM,
  `N_PROGRAM` = 836 и `N_WIDE` = 3377 — значения на момент спецификации (831 + 5, 3367 + 10); при регистрации N берётся из CHECKPOINT: максимум `n` / `n_wide` по ключам `S.*.registered` (кроме своего) + 5 / + 10.
- `register(now=None, trials=TRIALS, cm_dir=CM_DIR, ckpt=CKPT, spec=SPEC) -> 9`: дописывает ВСЕ 9 строк разом с ОДНОЙ `registered_at` (часы машины, «ДД.ММ.ГГГГ ЧЧ:ММ:СС») и сливает ключ `S.C.registered` = {spec_sha256, registered_at, n, n_wide, n_base, n_wide_base, rows} (прочие ключи побайтно; формат indent 1 / sorted keys).
  Отказ (SystemExit) до любой записи: в корне `cm_dir` есть `*.parquet`; любой id уже в реестре; формат CHECKPOINT чужой. `ckpt=None` — без записи CHECKPOINT.
- `require(cfg, trials=TRIALS)`: отказ, если id нет в реестре (первое действие `cm_walk.run`).
- `check(trials=TRIALS, cm_dir=CM_DIR, ckpt=CKPT) -> list` замечаний, только чтение: ровно 9 id (по id, не по позиции), без дублей, одна `registered_at`; ключ с rows = 9, n = n_base + 5, n_wide = n_wide_base + 10 и той же `registered_at`;
  если в корне `cm_dir` есть `*.parquet` — `registered_at` раньше mtime самого раннего (и раньше времени ключа `S.C.eval`, если у него есть время).
Запуск: `python3 -W ignore -m models_v2.s18.cm_reg [--check]`; без флага — регистрация (однократная)."""
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
SPEC = ROOT / "docs/otchety/harj_struct/S_C_SPEC.md"
CM_DIR = ROOT / "data/regen/harj_struct/CM"
KEY = "S.C.registered"
N_PROGRAM, N_WIDE = 836, 3377
NOT_RUN = "не запускается; в N не входит, пока не посчитана"
ROWS = [
    ("S-CD", "S: прямая коррекция на признаках компонент", "F = W5(X1(B_mix, cfg8·exp(θ0 + ρ_c ln(Ĥ_c/A_c) + ρ_p ln(Ĥ_p/A_p) + θ_c1 ln(c_d/c_m) + θ_p1 ln(p_d/p_m) + θ_s logit ŝ_c)))", "6; λ PIT", "контроль"),
    ("S-CK", "S: компонентный гибрид, потеря по сумме (основная)", "F = W5(X1(B_mix, cfg8·(1 + Σ_{c,p} ŝ_X expm1(η_X)))), η_X = γ_X ln(Ĥ_X/A_X) + θ_X0 + θ_X1 ln(X_d/X_m), A_X = ŝ_X·cfg8", "6; λ PIT", "зарегистрирована"),
    ("S-CKA", "S: компонентный гибрид + компонентная потеря", "как S-CK, L += 0,5·½Σ_{c,p} ℓ(X_{t⁺}, F_X)", "6; μ = 0,5 фикс.", "зарегистрирована"),
    ("S-C7", "S: контроль K7-класса с фактором этапа", "F = W5(X1(B_mix, Ĥ_c + Ĥ_p + ŝ_u·cfg8))", "0", "контроль"),
    ("S-CKL", "S: контроль (б) — разложение сдвинуто на 252 сессии назад", "S-CK на c̃_s = (RV_s − u_s)·c_{s−252}/(c + p)_{s−252}, p̃_s = (RV_s − u_s) − c̃_s", "6", "контроль"),
    ("S-C2f", "S: два фактора", "S-CK при k = 2", "—", NOT_RUN),
    ("S-C7i", "S: K7 по IMOEX (V3)", "β²·Ĥ(IMOEX) + Ĥ(u), `hp_ctrl.k7_series`", "—", NOT_RUN),
    ("S-CDx", "S: C-D со взаимодействиями долей", "C-D + ŝ_X·z_X", "—", NOT_RUN),
    ("S-CDw", "S: C-D поверх выхода W5-G", "W5-G·exp(θᵀz)", "—", NOT_RUN),
]
IDS = tuple(r[0] for r in ROWS)
assert len(ROWS) == 9 and len(set(IDS)) == 9


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _read(trials: Path) -> pd.DataFrame:
    return pd.read_csv(trials, comment="#", dtype=str, keep_default_na=False) if trials.exists() else pd.DataFrame(columns=COLS)


def _parquet(cm_dir: Path) -> list:
    return sorted(p.name for p in cm_dir.glob("*.parquet")) if cm_dir.is_dir() else []


def _dump(cp: dict) -> str:
    return json.dumps(cp, indent=1, sort_keys=True, ensure_ascii=False)


def _base(cp: dict, field: str, default: int) -> int:
    v = [k[field] for n, k in cp.items() if n.startswith("S.") and n.endswith(".registered") and n != KEY and isinstance(k, dict) and field in k]
    return max(v) if v else default


def merge_checkpoint(ckpt: Path, now: str, spec: Path = SPEC) -> dict:
    """Только ключ `S.C.registered`, слиянием; N — из файла (текущее + 5 / + 10)."""
    cp = json.loads(ckpt.read_text(encoding="utf-8"))
    nb, wb = _base(cp, "n", N_PROGRAM - 5), _base(cp, "n_wide", N_WIDE - 10)
    cp[KEY] = dict(spec_sha256=_sha(spec), registered_at=now, n=nb + 5, n_wide=wb + 10, n_base=nb, n_wide_base=wb, rows=len(ROWS))
    ckpt.write_text(_dump(cp), encoding="utf-8")
    return cp[KEY]


def register(now: str | None = None, trials: Path = TRIALS, cm_dir: Path = CM_DIR, ckpt: Path | None = CKPT, spec: Path = SPEC) -> int:
    pq = _parquet(cm_dir)
    if pq:
        raise SystemExit("реестр позже прогноза: в %s уже есть %s — регистрация отказана" % (cm_dir, pq))
    have = sorted(set(IDS) & set(_read(trials)["id"]))
    if have:
        raise SystemExit("уже зарегистрированы: %s — повторная регистрация отказана" % have)
    if ckpt is not None and _dump(json.loads(ckpt.read_text(encoding="utf-8"))) != ckpt.read_text(encoding="utf-8"):
        raise SystemExit("CHECKPOINT.json: формат не совпадает с indent 1 / sorted keys")
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


def check(trials: Path = TRIALS, cm_dir: Path = CM_DIR, ckpt: Path = CKPT) -> list:
    cur = _read(trials)
    rows = cur[cur["id"].isin(IDS)]
    bad = []
    if len(rows) != len(IDS) or set(rows["id"]) != set(IDS) or rows["id"].duplicated().any():
        bad.append("строк S-C в реестре %d из %d (ожидаются ровно id разд. 6)" % (len(rows), len(IDS)))
    reg = sorted(rows["registered_at"].unique())
    if len(reg) != 1:
        bad.append("registered_at не одна: %s" % reg)
    cp = json.loads(ckpt.read_text(encoding="utf-8")) if ckpt.exists() else {}
    k = cp.get(KEY) or {}
    ok = k.get("rows") == len(IDS) and [k.get("registered_at")] == reg and isinstance(k.get("n_base"), int) and isinstance(k.get("n_wide_base"), int)
    ok = ok and k.get("n") == k["n_base"] + 5 and k.get("n_wide") == k["n_wide_base"] + 10
    if not ok:
        bad.append("ключ %s в CHECKPOINT: %s (ожидается rows = 9, n = n_base + 5, n_wide = n_wide_base + 10, registered_at строк реестра)" % (KEY, k))
    ps = [cm_dir / n for n in _parquet(cm_dir)]
    if ps and len(reg) == 1:
        first = min(dt.datetime.fromtimestamp(p.stat().st_mtime) for p in ps)
        bad += [] if _t(reg[0]) < first else ["registered_at %s не раньше mtime самого раннего parquet %s" % (reg[0], first)]
        kt = _key_time(cp, "S.C.eval")
        bad += [] if kt is None or _t(reg[0]) < kt else ["registered_at не раньше времени ключа S.C.eval"]
    return bad


if __name__ == "__main__":
    if "--check" in sys.argv:
        b = check()
        r = _read(TRIALS)
        r = r[r["id"].isin(IDS)]
        k = (json.loads(CKPT.read_text(encoding="utf-8")).get(KEY) or {})
        print("check %s: строк %d; registered_at %s; parquet в CM: %d; N %s / N_wide %s" % ("ok" if not b else "FAIL " + "; ".join(b), len(r), sorted(r["registered_at"].unique()), len(_parquet(CM_DIR)), k.get("n"), k.get("n_wide")))
        sys.exit(1 if b else 0)
    print("реестр S-C: добавлено строк %d" % register())
