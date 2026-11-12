"""S-RCD-IMPL-2 (Ж-259; `S_RCD_SPEC.md`, разд. 10): регистратор трёх строк S-RCD, S-CDF, S-CKF и сторож «реестр до прогноза». По образцу `r_reg`, с проверкой всех отказов ДО первой записи.
Публичное API:
- `ROWS` / `IDS` — 3 строки разд. 10; `KEY` = "S.RCD.registered"; `SPEC`, `RD_DIR` = data/regen/harj_struct/RD; `ADD` = 3, `ADD_WIDE` = 6.
- `base_n(ckpt) -> (n, n_wide)`: максимум по ключам `S.*.registered` (кроме своего); отказ, если таких ключей нет.
- `register(now=None, trials=TRIALS, rd_dir=RD_DIR, ckpt=CKPT, spec=SPEC) -> 3`: отказ ДО любой записи: есть `*.parquet` в `rd_dir` (№ 30); id уже в реестре; ключ уже есть; нет ключей `S.*.registered`; формат CHECKPOINT чужой.
  Дописывает 3 строки с одной `registered_at`, затем СЛИВАЕТ ключ в CHECKPOINT (остальные ключи не трогает; запись через `.tmp` и `replace`). Печатает N и N_wide.
- `require(cfg, trials=TRIALS)`; `check(trials, rd_dir, ckpt) -> list`, только чтение: ровно 3 id (по id, не по позиции), одна `registered_at`, ключ (rows = 3, n = n_base + 3, n_wide = n_wide_base + 6), прежняя часть реестра по prior_sha256,
  `registered_at` раньше mtime самого раннего parquet в `rd_dir` и раньше ключа `S.RCD.eval`.
Запуск: `python3 -W ignore -m models_v2.s18.rd_reg [--check]`; без флага — регистрация (однократная)."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

from models_v2.s18 import r_reg as RR

ROOT, TRIALS, COLS, now_text, CKPT = RR.ROOT, RR.TRIALS, RR.COLS, RR.now_text, RR.CKPT
SPEC = ROOT / "docs/otchety/harj_struct/S_RCD_SPEC.md"
RD_DIR = ROOT / "data/regen/harj_struct/RD"
KEY, ADD, ADD_WIDE = "S.RCD.registered", 3, 6
HYP = "гипотеза выбрана после просмотра результатов S-R, S-D, S-L, S-Z"
FORM = "C = cfg8·exp(θ0 + ρ_c ln(Ĥ_c/A_c) + ρ_p ln(Ĥ_p/A_p) + θ_c1 ln(c_d/c_m) + θ_p1 ln(p_d/p_m) + θ_s logit ŝ_c) (форма S-CD)"
ROWS = [
    ("S-RCD", "S: прямая коррекция C-D + устойчивая процедура R (Ж-259)", FORM + "; F = W5(X1(B_mix, C)); процедура S-RCK: потеря ρ_κ уровня F с сохранёнными π, β; c = Q₀,₉₉, m; Λ = {0; 0,1; 0,01; 0,001}; 3 блока × 126, 2 из 3",
     "6; включение 630", "кандидат на проверку; " + HYP),
    ("S-CDF", "S: контроль — C-D без R, обычная QLIKE уровня F (Ж-259 п. 2)", "форма S-CD; L = mean ℓ(RV, F(θ)) + λ‖θ‖², F с сохранёнными π, β; λ ∈ {1e-4; 1e-3; 1e-2} по QLIKE уровня F на последних 126 закрытых парах, ничья — больший λ, < 252 пар — 1e-2; включение 2018-07-02; 5 перезапусков",
     "6; λ PIT", "контроль (не кандидат)"),
    ("S-CKF", "S: контроль уровня потери — C-K без R, обычная QLIKE уровня F", "как S-CDF с 𝓜_CK (S-CK, μ = 0)", "6; λ PIT", "контроль (диагностика, не кандидат)"),
]
IDS = tuple(r[0] for r in ROWS)
assert len(ROWS) == 3 and len(set(IDS)) == 3
_read, _parquet, _dump, _t, _sha = RR._read, RR._parquet, RR._dump, RR._t, RR._sha


def base_n(cp: dict) -> tuple:
    ks = [k for n, k in cp.items() if n.startswith("S.") and n.endswith(".registered") and n != KEY and isinstance(k, dict) and "n" in k and "n_wide" in k]
    if not ks:
        raise SystemExit("в CHECKPOINT нет ключей S.*.registered — N не определить")
    return max(k["n"] for k in ks), max(k["n_wide"] for k in ks)


def register(now: str | None = None, trials: Path = TRIALS, rd_dir: Path = RD_DIR, ckpt: Path = CKPT, spec: Path = SPEC) -> int:
    pq = _parquet(rd_dir)
    if pq:
        raise SystemExit("реестр позже прогноза: в %s уже есть %s — регистрация отказана" % (rd_dir, pq))
    have = sorted(set(IDS) & set(_read(trials)["id"]))
    if have:
        raise SystemExit("уже зарегистрированы: %s — повторная регистрация отказана" % have)
    raw = ckpt.read_text(encoding="utf-8")
    cp = json.loads(raw)
    if _dump(cp) != raw:
        raise SystemExit("CHECKPOINT.json: формат не совпадает с indent 1 / sorted keys")
    if KEY in cp:
        raise SystemExit("ключ %s уже есть в CHECKPOINT — регистрация отказана" % KEY)
    nb, wb = base_n(cp)     # отказ здесь — до любой записи
    now, prior = now or now_text(), trials.read_bytes()
    with trials.open("a", encoding="utf-8", newline="") as fh:
        fh.write(("" if prior.endswith(b"\n") else "\n") + pd.DataFrame([dict(zip(COLS, (*r, now))) for r in ROWS])[COLS].to_csv(index=False, header=False, lineterminator="\n"))
    cp = json.loads(ckpt.read_text(encoding="utf-8"))     # слияние по свежему содержимому: ключи других листов не теряются
    cp[KEY] = dict(spec_sha256=_sha(spec), registered_at=now, n=nb + ADD, n_wide=wb + ADD_WIDE, n_base=nb, n_wide_base=wb, rows=len(ROWS), prior_bytes=len(prior), prior_sha256=hashlib.sha256(prior).hexdigest())
    tmp = ckpt.with_suffix(".json.tmp")
    tmp.write_text(_dump(cp), encoding="utf-8")
    tmp.replace(ckpt)
    print("N = %d (%d + %d), N_wide = %d (%d + %d); registered_at %s" % (nb + ADD, nb, ADD, wb + ADD_WIDE, wb, ADD_WIDE, now))
    return len(ROWS)


def require(cfg: str, trials: Path = TRIALS) -> None:
    if cfg not in set(_read(trials)["id"]):
        raise SystemExit("%s: нет в реестре ALL_TRIALS.csv — счёт отказан (реестр до прогноза)" % cfg)


def check(trials: Path = TRIALS, rd_dir: Path = RD_DIR, ckpt: Path = CKPT) -> list:
    cur = _read(trials)
    rows = cur[cur["id"].isin(IDS)]
    bad = []
    if len(rows) != len(IDS) or set(rows["id"]) != set(IDS) or rows["id"].duplicated().any():
        bad.append("строк S-RCD / S-CDF / S-CKF в реестре %d из %d (ожидаются ровно id разд. 10)" % (len(rows), len(IDS)))
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
    ps = [rd_dir / n for n in _parquet(rd_dir)]
    if ps and len(reg) == 1:
        first = min(dt.datetime.fromtimestamp(p.stat().st_mtime) for p in ps)
        bad += [] if _t(reg[0]) < first else ["registered_at %s не раньше mtime самого раннего parquet %s" % (reg[0], first)]
        kt = RR._key_time(cp, "S.RCD.eval")
        bad += [] if kt is None or _t(reg[0]) < kt else ["registered_at не раньше времени ключа S.RCD.eval"]
    return bad


if __name__ == "__main__":
    if "--check" in sys.argv:
        b = check()
        r = _read(TRIALS)
        r = r[r["id"].isin(IDS)]
        k = (json.loads(CKPT.read_text(encoding="utf-8")).get(KEY) or {})
        print("check %s: строк %d; registered_at %s; parquet в RD: %d; N %s / N_wide %s" % ("ok" if not b else "FAIL " + "; ".join(b), len(r), sorted(r["registered_at"].unique()), len(_parquet(RD_DIR)), k.get("n"), k.get("n_wide")))
        sys.exit(1 if b else 0)
    print("реестр S-RCD: добавлено строк %d" % register())
