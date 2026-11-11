"""W4-A (Ж-250 пп. 3–4; `docs/otchety/harj_x1/W4_SPEC.md`, раздел 7): регистратор и сторож «реестр до прогноза».
`register()` дописывает 19 строк (W3D1–W3D11, W4-S1, W4-S2, W4-Ka, W4-Kb1, W4-Kb2, W4-X, W4-Xa, W4-Xb) в `ALL_TRIALS.csv` с ОДНОЙ меткой `registered_at` (часы машины) и сливает в
`CHECKPOINT.json` ключи `W3` и `W4.registered` (прочие ключи — побайтно). Отказ: в `data/regen/harj_x1/W4/` (только корень, без подкаталогов) уже есть `*.parquet`, или хоть один id уже в реестре.
`require(id)` — для W4-B/W4-C: отказ, если id нет в реестре. `check()` / `--check` только читают. Запуск: `python3 -W ignore -m models_v2.s16.x1_w4_reg [--check]`."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

from models_v2.s16 import x1_reg as R

ROOT, REP = R.ROOT, R.REP
TRIALS, COLS, now_text = R.TRIALS, R.COLS, R.now_text
CKPT = REP / "CHECKPOINT.json"
W4 = ROOT / "data/regen/harj_x1/W4"
N_PROGRAM, N_WIDE = 770, 3245
DIAG = "диагностика W3, критериев не получает"
POST = "диагностика W3, после просмотра"
ROWS = [
    ("W3D1", "W3: DFTL126", "π = argmin дисконтированной (λ = 0,5^(1/126)) суммы QLIKE по сетке π по закрытым парам состояния", "—", DIAG),
    ("W3D2", "W3: AH", "AdaHedge над 21 π-экспертом, fixed share 0,01, по состоянию", "—", "диагностика W3"),
    ("W3D3", "W3: DFTL63", "как W3D1, полураспад 63", "—", "диагностика W3"),
    ("W3D4", "W3: DFTL252", "как W3D1, полураспад 252", "—", "диагностика W3"),
    ("W3D5", "W3: DFTLinf", "как W3D1, без дисконта", "—", "диагностика W3"),
    ("W3D6", "W3: HEDGE0", "проход X1 с α = 0", "—", "диагностика W3"),
    ("W3D7", "W3: π = доля лет", "π = доля прошлых лет с QLIKE(C) < QLIKE(M) в состоянии", "—", "диагностика W3"),
    ("W3D8", "W3: N05", "в N π = 0,5, в E — π X1", "—", POST),
    ("W3D9", "W3: N_AH", "в N π AH, в E — π X1", "—", POST),
    ("W3D10", "W3: N_DFTL126", "в N π DFTL126, в E — π X1", "—", POST),
    ("W3D11", "W3: E_M", "в E π = 0, в N — π X1", "—", POST),
    ("W4-S1", "Б6 ступень 1: M⁺ на S", "M⁺ = M·exp(clip(ρĝ(S), ±1)), ĝ — `v6_1_fit.fit_gbm` на e = ln RV − ln M", "деревья ≤ 300 и ρ ∈ {0; 0,25; 0,5; 0,75; 1} — PIT по последним 252 закрытым парам", "зарегистрирована"),
    ("W4-S2", "Б6 ступень 1: M⁺ на S + Σ", "как W4-S1, признаки S + 12 площадей Σ (набор A2)", "как W4-S1", "зарегистрирована"),
    ("W4-Ka", "контроль (а) ступени 1", "g = ρ·ln mean(RV/M) по закрытым парам", "ρ", "контроль, критериев не получает"),
    ("W4-Kb1", "контроль (б) к W4-S1", "W4-S1 на S со сдвигом 252", "как W4-S1", "контроль"),
    ("W4-Kb2", "контроль (б) к W4-S2", "W4-S2 на S + Σ со сдвигом 252", "как W4-S2", "контроль"),
    ("W4-X", "Б6 ступень 2: X1 на M⁺", "F = (M⁺)^(1−π)·C^π, слой X1 PIT", "набор — правило раздела 3", "условная: считается, если прошёл ≥ 1 набор ступени 1"),
    ("W4-Xa", "контроль (а) ступени 2", "слой X1 над K_a", "—", "условная, контроль"),
    ("W4-Xb", "контроль (б) ступени 2", "слой X1 над K_b выбранного набора", "—", "условная, контроль"),
]
IDS = tuple(r[0] for r in ROWS)


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _read(trials: Path) -> pd.DataFrame:
    return pd.read_csv(trials, comment="#", dtype=str, keep_default_na=False) if trials.exists() else pd.DataFrame(columns=COLS)


def _parquet(w4: Path) -> list:
    return sorted(p.name for p in w4.glob("*.parquet")) if w4.is_dir() else []


def merge_checkpoint(ckpt: Path, now: str, rep: Path = REP, code: Path = Path(__file__).resolve().parent) -> None:
    """Только ключи W3 и W4.registered, слиянием (формат файла: indent 1, sorted keys); прочие ключи сохраняются побайтно."""
    cp = json.loads(ckpt.read_text(encoding="utf-8"))
    cp["W3"] = dict(sha256={"W3_DIAG.csv": _sha(rep / "W3_DIAG.csv"), "W3_SPEC.md": _sha(rep / "W3_SPEC.md")},
                    code={n: _sha(code / n) for n in ("x1_w3_core.py", "x1_w3_diag.py")}, configurations=0, forecasts_written=0)
    cp["W4.registered"] = dict(spec_sha256=_sha(rep / "W4_SPEC.md"), registered_at=now, n=N_PROGRAM, n_wide=N_WIDE, rows=len(ROWS))
    ckpt.write_text(json.dumps(cp, indent=1, sort_keys=True, ensure_ascii=False), encoding="utf-8")


def register(now: str | None = None, trials: Path = TRIALS, w4: Path = W4, ckpt: Path | None = CKPT) -> int:
    """Дописывает ВСЕ 19 строк разом (одна `registered_at`), затем сливает ключи CHECKPOINT. Отказ (SystemExit): в W4 уже есть *.parquet (реестр позже прогноза) или любой из id уже в реестре."""
    pq = _parquet(w4)
    if pq:
        raise SystemExit("реестр позже прогноза: в %s уже есть %s — регистрация отказана" % (w4, pq))
    cur = _read(trials)
    have = sorted(set(IDS) & set(cur["id"]))
    if have:
        raise SystemExit("уже зарегистрированы: %s — повторная регистрация отказана" % have)
    now = now or now_text()
    text = trials.read_text(encoding="utf-8")
    with trials.open("a", encoding="utf-8", newline="") as fh:
        fh.write(("" if text.endswith("\n") else "\n") + pd.DataFrame([dict(zip(COLS, (*r, now))) for r in ROWS])[COLS].to_csv(index=False, header=False, lineterminator="\n"))
    if ckpt is not None:
        merge_checkpoint(ckpt, now)
    return len(ROWS)


def require(cfg: str, trials: Path = TRIALS) -> None:
    """Для W4-B/W4-C: отказ (SystemExit), если id нет в реестре."""
    if cfg not in set(_read(trials)["id"]):
        raise SystemExit("%s: нет в реестре ALL_TRIALS.csv — счёт отказан (реестр до прогноза)" % cfg)


def check(trials: Path = TRIALS, w4: Path = W4) -> list:
    """Только чтение. -> список замечаний: 19 строк по id (не по позиции), одна `registered_at`, в корне W4 нет *.parquet."""
    cur = _read(trials)
    rows = cur[cur["id"].isin(IDS)]
    bad = []
    if len(rows) != len(IDS) or set(rows["id"]) != set(IDS) or rows["id"].duplicated().any():
        bad.append("строк W3/W4 в реестре %d из %d (ожидаются ровно id раздела 7)" % (len(rows), len(IDS)))
    if rows["registered_at"].nunique() != 1:
        bad.append("registered_at не одна: %s" % sorted(rows["registered_at"].unique()))
    pq = _parquet(w4)
    if pq:
        bad.append("в W4 есть parquet: %s" % pq)
    return bad


if __name__ == "__main__":
    if "--check" in sys.argv:
        b = check()
        print("check %s: строк %d; registered_at %s; parquet в W4: %d" % ("ok" if not b else "FAIL " + "; ".join(b), len(_read(TRIALS)[lambda d: d["id"].isin(IDS)]),
                                                                          sorted(_read(TRIALS)[lambda d: d["id"].isin(IDS)]["registered_at"].unique()), len(_parquet(W4))))
        sys.exit(1 if b else 0)
    print("реестр W4-A: добавлено строк %d" % register())
