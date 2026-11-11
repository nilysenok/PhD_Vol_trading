"""W6-reg (Ж-250 п. 6; `docs/otchety/harj_x1/W6_EXHAUSTION.md`, раздел 1): регистратор 4 диагностических рядов W6D-GJS, W6D-ZJS, W6D-U1, W6D-U2.
Регистрация ПОСЛЕ расчёта диагностики (расчёт — в scratchpad, прогнозных файлов нет), поэтому сторож «реестр до прогноза» не применяется.
Публичное API:
- `ROWS` / `IDS` — 4 строки; `N_PROGRAM` = 802, `N_WIDE` = 3309.
- `register(now=None, trials=TRIALS, ckpt=CKPT) -> 4`: дописывает 4 строки с ОДНОЙ `registered_at`, сливает в `CHECKPOINT.json` ключ `W6.registered` (прочие ключи побайтно).
  Отказ (SystemExit), если хоть один id уже в реестре.
- `check(trials=TRIALS) -> list` замечаний, только чтение: ровно 4 id (по id, не по позиции), без дублей, одна `registered_at`, пометка «ПОСЛЕ расчёта» в каждой строке.
Запуск: `python3 -W ignore -m models_v2.s16.x1_w6_reg [--check]`."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

from models_v2.s16 import x1_reg as R

REP, TRIALS, COLS, now_text = R.REP, R.TRIALS, R.COLS, R.now_text
CKPT = REP / "CHECKPOINT.json"
DOC = REP / "W6_EXHAUSTION.md"
N_PROGRAM, N_WIDE = 802, 3309
MARK = "ПОСЛЕ расчёта"
DIAG = "диагностика W6, критериев не получает; зарегистрировано ПОСЛЕ расчёта диагностики (Ж-250 п. 6, решение дирижёра 14:48)"
ROWS = [
    ("W6D-GJS", "W6: W5-G со сжатием по SE", "F = X1·exp(gap·β̃), β̃ — β W5-G, сжатая по стандартной ошибке (PIT κ)", "—", DIAG),
    ("W6D-ZJS", "W6: ZB со сжатием по Вальду", "F = X1·exp(e·(b̃ + ã·s)), (b, a) ZB, сжатые по статистике Вальда (PIT κ)", "—", DIAG),
    ("W6D-U1", "W6: θᵀ[e, e·s, gap], без сжатия", "F = X1·exp(θᵀ[e, e·s, gap]); без штрафа и клипа; D ≥ 2018-07-02, пул не меньше 30 пар E и 30 пар gap", "—", DIAG),
    ("W6D-U2", "W6: θᵀ[e, e·s, gap], общее сжатие", "то же, что W6D-U1, с общим сжатием θ", "—", DIAG),
]
IDS = tuple(r[0] for r in ROWS)
assert len(set(IDS)) == 4


def _read(trials: Path) -> pd.DataFrame:
    return pd.read_csv(trials, comment="#", dtype=str, keep_default_na=False) if trials.exists() else pd.DataFrame(columns=COLS)


def merge_checkpoint(ckpt: Path, now: str, doc: Path = DOC) -> None:
    """Только ключ `W6.registered`, слиянием (indent 1, sorted keys); прочие ключи сохраняются."""
    cp = json.loads(ckpt.read_text(encoding="utf-8"))
    sha = hashlib.sha256(doc.read_bytes()).hexdigest() if doc.exists() else ""
    cp["W6.registered"] = dict(spec_sha256=sha, registered_at=now, n=N_PROGRAM, n_wide=N_WIDE, rows=len(ROWS))
    ckpt.write_text(json.dumps(cp, indent=1, sort_keys=True, ensure_ascii=False), encoding="utf-8")


def register(now: str | None = None, trials: Path = TRIALS, ckpt: Path | None = CKPT, doc: Path = DOC) -> int:
    have = sorted(set(IDS) & set(_read(trials)["id"]))
    if have:
        raise SystemExit("уже зарегистрированы: %s — повторная регистрация отказана" % have)
    now = now or now_text()
    text = trials.read_text(encoding="utf-8")
    with trials.open("a", encoding="utf-8", newline="") as fh:
        fh.write(("" if text.endswith("\n") else "\n") + pd.DataFrame([dict(zip(COLS, (*r, now))) for r in ROWS])[COLS].to_csv(index=False, header=False, lineterminator="\n"))
    if ckpt is not None:
        merge_checkpoint(ckpt, now, doc)
    return len(ROWS)


def check(trials: Path = TRIALS) -> list:
    """Только чтение: 4 строки по id, одна `registered_at`, пометка «ПОСЛЕ расчёта» в каждой."""
    cur = _read(trials)
    rows = cur[cur["id"].isin(IDS)]
    bad = []
    if len(rows) != len(IDS) or set(rows["id"]) != set(IDS) or rows["id"].duplicated().any():
        bad.append("строк W6D в реестре %d из %d" % (len(rows), len(IDS)))
    if rows["registered_at"].nunique() != 1:
        bad.append("registered_at не одна: %s" % sorted(rows["registered_at"].unique()))
    no = [i for i, s in zip(rows["id"], rows["status"]) if MARK not in s]
    if no:
        bad.append("нет пометки «%s»: %s" % (MARK, no))
    return bad


if __name__ == "__main__":
    if "--check" in sys.argv:
        b = check()
        r = _read(TRIALS)
        r = r[r["id"].isin(IDS)]
        print("check %s: строк %d; registered_at %s" % ("ok" if not b else "FAIL " + "; ".join(b), len(r), sorted(r["registered_at"].unique())))
        sys.exit(1 if b else 0)
    print("реестр W6-reg: добавлено строк %d" % register())
