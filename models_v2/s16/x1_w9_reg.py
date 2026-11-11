"""W9-reg (`docs/otchety/harj_x1/W9_EXHAUSTION.md`, разд. 8; `W9_DECL.md`): регистратор 13 диагностических рядов W9.
Регистрация ПОСЛЕ расчёта диагностики (решение дирижёра 11.11 16:22; `W9_DIAG.csv`), поэтому сторож «реестр до прогноза» не применяется.
Состав (по объявлению и по перечню W9_EXHAUSTION разд. 8, без домыслов):
- 6 строк по объявлению: W9A, W9D1, W9D2, W9D3, W9B1, W9B2 (статус «критериев по объявлению не прошла»; `DECISION,none`);
- 7 строк «ПОСЛЕ ПРОСМОТРА»: из Д-2 четыре (PH-E, PH-E0, PH-R, PH-R0) и три из чувствительности PH2 (PH-R126, PH-R504, PH-R+W5G).
Контроли A0, B0, C0, 19 сдвигов, 500 перестановок — нулевые, в N не входят. N = 814 + 13 = 827; N_wide = 3333 + 26 = 3359 (по 2 на строку, как в W8).
Публичное API:
- `ROWS` / `IDS` — 13 строк; `N_PROGRAM` = 827, `N_WIDE` = 3359.
- `register(now=None, trials=TRIALS, ckpt=CKPT) -> 13`: дописывает 13 строк с ОДНОЙ `registered_at`, сливает в `CHECKPOINT.json` ключ `W9.registered` (прочие ключи побайтно).
  Отказ (SystemExit), если хоть один id уже в реестре.
- `check(trials=TRIALS) -> list` замечаний, только чтение: ровно 13 id (по id, не по позиции), без дублей, одна `registered_at`, пометка в каждой строке, «ПОСЛЕ ПРОСМОТРА» — ровно у 7.
Запуск: `python3 -W ignore -m models_v2.s16.x1_w9_reg [--check]`."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

from models_v2.s16 import x1_reg as R

REP, TRIALS, COLS, now_text = R.REP, R.TRIALS, R.COLS, R.now_text
CKPT = REP / "CHECKPOINT.json"
DOC = REP / "W9_EXHAUSTION.md"
DECL = REP / "W9_DECL.md"
N_PROGRAM, N_WIDE = 827, 3359
MARK = "зарегистрировано после расчёта диагностики (решение дирижёра 16:22)"
POST = "ПОСЛЕ ПРОСМОТРА"
DIAG = "диагностика W9, критериев по объявлению не прошла; " + MARK
PH = "диагностика W9, критериев не получает; " + POST + " (блок PH/PH2 W9_DIAG.csv, вердикт «none» уже выведен); " + MARK
ROWS = [
    ("W9A", "W9: разделение HAR-части и энкодера C", "ln F = ln M + π·(κ_H·D_H + κ_G·D_G), D_G = G − w·ℓ (Д-1); сетка κ 0…1,5 шаг 0,05; перенос h1 ↔ h2", "(κ_H, κ_G)", DIAG),
    ("W9D1", "W9: доверие к C по согласию HAR и энкодера", "ln F = ln M + κ_g·π·D, g = 1[D_G·D_H > 0]", "κ_g ∈ {0; 0,05; …; 1,5}", DIAG),
    ("W9D2", "W9: доверие к C по доле энкодера", "ln F = ln M + κ_g·π·D, g — PIT-терцили |D_G|/(|D_G| + |D_H|)", "κ_g ∈ {0; 0,05; …; 1,5}", DIAG),
    ("W9D3", "W9: доверие к C по z-неустойчивости", "ln F = ln M + κ_g·π·D, g — PIT-терцили расстояния z энкодера; выбран после просмотра DE0", "κ_g ∈ {0; 0,05; …; 1,5}", DIAG),
    ("W9B1", "W9: компонент под остаток X1 на S", "F = X1·exp(clip(θ0 + θᵀz, ±1)), ридж λ = 0,01, s1…s9", "θ (10)", DIAG),
    ("W9B2", "W9: компонент под остаток X1 на S + Σ", "то же, S + 12 площадей Σ", "θ (22)", DIAG),
    ("W9-PH-E", "W9 после просмотра: PIT разделённых весов, расширяющийся пул", "ln F = ln M + π·(κ_H,m·D_H + κ_G,m·D_G), argmin по закрытым парам t⁺ ≤ D_m, помесячно; от 126 пар", "(κ_H, κ_G) по месяцам", PH),
    ("W9-PH-E0", "W9 после просмотра: PH-E с общим κ", "то же при κ_H = κ_G", "κ по месяцам", PH),
    ("W9-PH-R", "W9 после просмотра: PIT разделённых весов, пул 252", "как PH-E, пул — 252 последние закрытые пары", "(κ_H, κ_G) по месяцам", PH),
    ("W9-PH-R0", "W9 после просмотра: PH-R с общим κ", "то же при κ_H = κ_G", "κ по месяцам", PH),
    ("W9-PH-R126", "W9 после просмотра: PH-R, пул 126", "как PH-R, пул — 126 последних закрытых пар (чувствительность)", "(κ_H, κ_G) по месяцам", PH),
    ("W9-PH-R504", "W9 после просмотра: PH-R, пул 504", "как PH-R, пул — 504 последние закрытые пары (чувствительность)", "(κ_H, κ_G) по месяцам", PH),
    ("W9-PH-R+W5G", "W9 после просмотра: PH-R плюс β W5-G", "F = PH-R·(F_W5-G / X1)", "параметры PH-R и W5-G", PH),
]
IDS = tuple(r[0] for r in ROWS)
assert len(set(IDS)) == 13 and sum(POST in r[4] for r in ROWS) == 7 and all(MARK in r[4] for r in ROWS)


def _read(trials: Path) -> pd.DataFrame:
    return pd.read_csv(trials, comment="#", dtype=str, keep_default_na=False) if trials.exists() else pd.DataFrame(columns=COLS)


def merge_checkpoint(ckpt: Path, now: str, doc: Path = DOC, decl: Path = DECL) -> None:
    """Только ключ `W9.registered`, слиянием (indent 1, sorted keys); прочие ключи сохраняются."""
    cp = json.loads(ckpt.read_text(encoding="utf-8"))
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else ""  # noqa: E731
    cp["W9.registered"] = dict(spec_sha256=sha(doc), decl_sha256=sha(decl), registered_at=now, n=N_PROGRAM, n_wide=N_WIDE, rows=len(ROWS),
                               rows_declared=6, rows_post_viewing=7)
    ckpt.write_text(json.dumps(cp, indent=1, sort_keys=True, ensure_ascii=False), encoding="utf-8")


def register(now: str | None = None, trials: Path = TRIALS, ckpt: Path | None = CKPT, doc: Path = DOC, decl: Path = DECL) -> int:
    have = sorted(set(IDS) & set(_read(trials)["id"]))
    if have:
        raise SystemExit("уже зарегистрированы: %s — повторная регистрация отказана" % have)
    now = now or now_text()
    text = trials.read_text(encoding="utf-8")
    with trials.open("a", encoding="utf-8", newline="") as fh:
        fh.write(("" if text.endswith("\n") else "\n") + pd.DataFrame([dict(zip(COLS, (*r, now))) for r in ROWS])[COLS].to_csv(index=False, header=False, lineterminator="\n"))
    if ckpt is not None:
        merge_checkpoint(ckpt, now, doc, decl)
    return len(ROWS)


def check(trials: Path = TRIALS) -> list:
    """Только чтение: 13 строк по id, одна `registered_at`, пометка в каждой, «ПОСЛЕ ПРОСМОТРА» ровно у 7."""
    cur = _read(trials)
    rows = cur[cur["id"].isin(IDS)]
    bad = []
    if len(rows) != len(IDS) or set(rows["id"]) != set(IDS) or rows["id"].duplicated().any():
        bad.append("строк W9 в реестре %d из %d" % (len(rows), len(IDS)))
    if rows["registered_at"].nunique() != 1:
        bad.append("registered_at не одна: %s" % sorted(rows["registered_at"].unique()))
    no = [i for i, s in zip(rows["id"], rows["status"]) if MARK not in s]
    if no:
        bad.append("нет пометки «%s»: %s" % (MARK, no))
    post = sorted(i for i, s in zip(rows["id"], rows["status"]) if POST in s)
    if post != sorted(i for i, *_, s in ROWS if POST in s):
        bad.append("«%s» не у тех строк: %s" % (POST, post))
    return bad


if __name__ == "__main__":
    if "--check" in sys.argv:
        b = check()
        r = _read(TRIALS)
        r = r[r["id"].isin(IDS)]
        print("check %s: строк %d; registered_at %s" % ("ok" if not b else "FAIL " + "; ".join(b), len(r), sorted(r["registered_at"].unique())))
        sys.exit(1 if b else 0)
    print("реестр W9-reg: добавлено строк %d" % register())
