"""S-Z-IMPL-2 (Ж-258; `S_Z_SPEC.md`, разд. 10): регистратор трёх строк S-RCKZ, S-RZ, S-RCKZs и сторож «реестр до прогноза». По образцу `r_reg`.
Публичное API:
- `ROWS` / `IDS` — три строки разд. 10; `KEY` = "S.Z.registered"; `SPEC`, `Z_DIR` = data/regen/harj_struct/Z; `ADD` = 3, `ADD_WIDE` = 6.
- `register(now=None, trials=TRIALS, z_dir=Z_DIR, ckpt=CKPT, spec=SPEC) -> 3`: дописывает все три строки разом с ОДНОЙ `registered_at` и сливает ключ {spec_sha256, registered_at, n, n_wide, n_base, n_wide_base, rows, prior_bytes, prior_sha256}
  (слияние ключа, остальные ключи CHECKPOINT не меняются). N = max `n` / `n_wide` по ключам `S.*.registered` (кроме своего) + 3 / + 6 (на 12.11: 852 + 3 = 855, 3409 + 6 = 3415).
  Отказ до любой записи: в `z_dir` есть `*.parquet` или `zmaps.npz` (№ 30); любой id уже в реестре; формат CHECKPOINT чужой.
- `require(cfg, trials=TRIALS)`: отказ, если id нет в реестре (первое действие прохода).
- `check(trials, z_dir, ckpt) -> list`, только чтение: ровно три id (по id, не по позиции), без дублей, одна `registered_at`; ключ: rows = 3, n = n_base + 3, n_wide = n_wide_base + 6; прежняя часть реестра по prior_sha256;
  `registered_at` раньше mtime `zmaps.npz`, любого `Z/*.parquet` и ключа `S.Z.eval`.
Запуск: `python3 -W ignore -m models_v2.s18.z_reg [--check]`; без флага — регистрация (однократная)."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

from models_v2.s16 import x1_reg as R
from models_v2.s18 import r_reg as RR

ROOT, REP = R.ROOT, R.REP
TRIALS, COLS, now_text = R.TRIALS, R.COLS, R.now_text
CKPT, _read, _dump, _sha, _t, _key_time = RR.CKPT, RR._read, RR._dump, RR._sha, RR._t, RR._key_time
SPEC = ROOT / "docs/otchety/harj_struct/S_Z_SPEC.md"
Z_DIR = ROOT / "data/regen/harj_struct/Z"
KEY, ADD, ADD_WIDE = "S.Z.registered", 3, 6
ROWS = [
    ("S-RCKZ", "S: C-K + линейная поправка ln C по z энкодера M10 под R (кандидат)",
     "C = cfg8·𝓜_CK(θ_CK)·exp(Σ_j θ_Z,j z̃_j); z — TCN-w5 Stage A года переоценки, seed s; z̃ = clip((z − μ_T)/σ_T, ±3), μ, σ по T; голова на seed, среднее ln C; F = W5(X1(B_mix, C)); потеря ρ_κ уровня F с сохранёнными π, β; c = Q₀,₉₉, m; Λ = {0; 0,1; 0,01; 0,001}; 3 блока × 126, 2 из 3",
     "10 на голову × 5; включение 630", "кандидат"),
    ("S-RZ", "S: абляция — только z, 𝓜_CK ≡ 1", "C = cfg8·exp(Σ θ_Z,j z̃_j), процедура S-RCKZ", "4 × 5", "абляция"),
    ("S-RCKZs", "S: контроль — z сессии на 252 раньше без переноса",
     "как S-RCKZ, z каждой даты (пул, блоки, прогноз) := z сессии pos − 252 по сетке сессий SBER, тот же энкодер; пары без сдвинутого z исключаются (Д4)", "10 × 5", "контроль (не кандидат)"),
]
IDS = tuple(r[0] for r in ROWS)
assert len(ROWS) == 3 and len(set(IDS)) == 3


def _files(z_dir: Path) -> list:
    return sorted([p.name for p in z_dir.glob("*.parquet")] + [p.name for p in z_dir.glob("zmaps.npz")]) if z_dir.is_dir() else []


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


def register(now: str | None = None, trials: Path = TRIALS, z_dir: Path = Z_DIR, ckpt: Path | None = CKPT, spec: Path = SPEC) -> int:
    if _files(z_dir):
        raise SystemExit("реестр позже прогноза: в %s уже есть %s — регистрация отказана" % (z_dir, _files(z_dir)))
    have = sorted(set(IDS) & set(_read(trials)["id"]))
    if have:
        raise SystemExit("уже зарегистрированы: %s — повторная регистрация отказана" % have)
    if ckpt is not None and _dump(json.loads(ckpt.read_text(encoding="utf-8"))) != ckpt.read_text(encoding="utf-8"):
        raise SystemExit("CHECKPOINT.json: формат не совпадает с indent 1 / sorted keys")
    if ckpt is not None:     # база N — до первой записи (отказ без ключей S.*.registered не должен оставить строк в реестре)
        cp0 = json.loads(ckpt.read_text(encoding="utf-8"))
        _base(cp0, "n"), _base(cp0, "n_wide")
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


def check(trials: Path = TRIALS, z_dir: Path = Z_DIR, ckpt: Path = CKPT) -> list:
    cur = _read(trials)
    rows, bad = cur[cur["id"].isin(IDS)], []
    if len(rows) != len(IDS) or set(rows["id"]) != set(IDS) or rows["id"].duplicated().any():
        bad.append("строк S-Z в реестре %d из %d (ожидаются ровно id разд. 10)" % (len(rows), len(IDS)))
    reg = sorted(rows["registered_at"].unique())
    if len(reg) != 1:
        bad.append("registered_at не одна: %s" % reg)
    cp = json.loads(ckpt.read_text(encoding="utf-8")) if ckpt.exists() else {}
    k = cp.get(KEY) or {}
    ok = k.get("rows") == len(IDS) and [k.get("registered_at")] == reg and isinstance(k.get("n_base"), int) and isinstance(k.get("n_wide_base"), int)
    if not (ok and k.get("n") == k["n_base"] + ADD and k.get("n_wide") == k["n_wide_base"] + ADD_WIDE):
        bad.append("ключ %s в CHECKPOINT: %s (ожидается rows = 3, n = n_base + 3, n_wide = n_wide_base + 6, registered_at строк реестра)" % (KEY, k))
    raw = trials.read_bytes() if trials.exists() else b""
    if "prior_bytes" in k and hashlib.sha256(raw[: k["prior_bytes"]]).hexdigest() != k.get("prior_sha256"):
        bad.append("прежняя часть реестра изменена (sha256 первых %s байт ≠ записанному)" % k["prior_bytes"])
    ps = [z_dir / n for n in _files(z_dir)]
    if ps and len(reg) == 1:
        first = min(dt.datetime.fromtimestamp(p.stat().st_mtime) for p in ps)
        bad += [] if _t(reg[0]) < first else ["registered_at %s не раньше mtime самого раннего файла Z %s" % (reg[0], first)]
        kt = _key_time(cp, "S.Z.eval")
        bad += [] if kt is None or _t(reg[0]) < kt else ["registered_at не раньше времени ключа S.Z.eval"]
    return bad


if __name__ == "__main__":
    if "--check" in sys.argv:
        b = check()
        r = _read(TRIALS)
        r = r[r["id"].isin(IDS)]
        k = (json.loads(CKPT.read_text(encoding="utf-8")).get(KEY) or {})
        print("check %s: строк %d; registered_at %s; файлов в Z: %s; N %s / N_wide %s" % ("ok" if not b else "FAIL " + "; ".join(b), len(r), sorted(r["registered_at"].unique()), _files(Z_DIR), k.get("n"), k.get("n_wide")))
        sys.exit(1 if b else 0)
    print("реестр S-Z: добавлено строк %d" % register())
