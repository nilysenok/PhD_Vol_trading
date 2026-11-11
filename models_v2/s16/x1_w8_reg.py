"""W8-A (Ж-253 п. 4; `docs/otchety/harj_x1/W8_SPEC.md`, разд. 8): регистратор 12 строк и сторож «реестр до прогноза». По образцу `x1_w5_reg`.
Публичное API:
- `ROWS` / `IDS` — 12 строк разд. 8: W8D1…W8D10 (диагностика, признаки по порядку: type, len, holiday, lvl, shock, e, imoex_absr, imoex_rv, season, cold), W8-G, W8-Gb.
  `N_PROGRAM` = 814 (798 W5 + 4 W6D + 10 W8D + 2), `N_WIDE` = 3333 (3301 + 8 + 2·12). Строки W8D дополнены пометкой «ПОСЛЕ расчёта» (диагностика посчитана до регистрации, `W8_DECOMP.csv`; отступление
  от слов разд. 8 «до любого прогноза»: прогнозов W8D нет — это сохранённые ряды); строки W8-G и W8-Gb — статус разд. 8 («зарегистрирована», «контроль»).
- `register(now=None, trials=TRIALS, w8=W8, ckpt=CKPT) -> 12`: дописывает ВСЕ 12 строк разом с ОДНОЙ `registered_at` (часы машины, формат ДД.ММ.ГГГГ ЧЧ:ММ:СС) и сливает в `CHECKPOINT.json` ключ `W8.registered`
  (spec_sha256 `W8_SPEC.md`, decomp_sha256 `W8_DECOMP.csv`, registered_at, n, n_wide, rows; прочие ключи побайтно, indent 1, sorted keys). Отказ (SystemExit): в корне `w8` (`data/regen/harj_x1/W8/`,
  подкаталоги не считаются) есть `*.parquet` (реестр позже прогноза) или хоть один id уже в реестре. `ckpt=None` — без записи CHECKPOINT.
- `merge_checkpoint(ckpt, now, rep=REP)`: только ключ `W8.registered`.
- `require(cfg, trials=TRIALS)`: SystemExit, если id нет в реестре (для W8-B).
- `check(trials=TRIALS, w8=W8, after=False, ckpt=CKPT) -> list` замечаний, только чтение: ровно 12 id (по id, не по позиции), без дублей, одна `registered_at`; по умолчанию — в корне `w8` нет `*.parquet`;
  `after=True` (`--check-after`) — registered_at раньше mtime самого раннего parquet и раньше `written_at` ключа `W8.stage1` (если он есть).
Запуск: `python3 -W ignore -m models_v2.s16.x1_w8_reg [--check | --check-after]`; без флага — регистрация (однократная)."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

from models_v2.s16 import x1_reg as R
from models_v2.s16 import x1_w5_reg as W5R

ROOT, REP = R.ROOT, R.REP
TRIALS, COLS, now_text = R.TRIALS, R.COLS, R.now_text
CKPT = REP / "CHECKPOINT.json"
W8 = ROOT / "data/regen/harj_x1/W8"
N_PROGRAM, N_WIDE = 814, 3333
DIAG = "диагностика, критериев не получает; ПОСЛЕ расчёта (W8_DECOMP.csv, сохранённые ряды)"
FEATS = ("type", "len", "holiday", "lvl", "shock", "e", "imoex_absr", "imoex_rv", "season", "cold")
ROWS = (
    [("W8D%d" % (i + 1), "W8 поиск: перенос группового β W5-G по признаку %s" % s, "β_G = clip(Δū + ½Δs², 0, 0,5) по группе, перенос h1 ↔ h2; признаки по порядку: type, len, holiday, lvl, shock, e, imoex_absr, imoex_rv, season, cold",
      "—", DIAG) for i, s in enumerate(FEATS)]
    + [("W8-G", "W8: β W5-G по терцилю RV IMOEX дня d", "F = X1·exp(gap·β_{m,G}); группа не включена ⇒ β W5-G", "нет сетки", "зарегистрирована"),
       ("W8-Gb", "W8: контроль (б)", "W8-G с рядом z, сдвинутым на 252", "—", "контроль")]
)
IDS = tuple(r[0] for r in ROWS)
assert len(ROWS) == 12 and len(set(IDS)) == 12


def _sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def merge_checkpoint(ckpt: Path, now: str, rep: Path = REP) -> None:
    """Только ключ `W8.registered`, слиянием (indent 1, sorted keys); прочие ключи сохраняются."""
    cp = json.loads(ckpt.read_text(encoding="utf-8"))
    cp["W8.registered"] = dict(spec_sha256=_sha(rep / "W8_SPEC.md"), decomp_sha256=_sha(rep / "W8_DECOMP.csv"), registered_at=now, n=N_PROGRAM, n_wide=N_WIDE, rows=len(ROWS))
    ckpt.write_text(json.dumps(cp, indent=1, sort_keys=True, ensure_ascii=False), encoding="utf-8")


def register(now: str | None = None, trials: Path = TRIALS, w8: Path = W8, ckpt: Path | None = CKPT) -> int:
    """Дописывает ВСЕ 12 строк разом (одна `registered_at`), затем ключ CHECKPOINT. Отказ: в W8 есть *.parquet или любой из id уже в реестре."""
    pq = W5R._parquet(w8)
    if pq:
        raise SystemExit("реестр позже прогноза: в %s уже есть %s — регистрация отказана" % (w8, pq))
    have = sorted(set(IDS) & set(W5R._read(trials)["id"]))
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
    """Для W8-B: отказ (SystemExit), если id нет в реестре."""
    if cfg not in set(W5R._read(trials)["id"]):
        raise SystemExit("%s: нет в реестре ALL_TRIALS.csv — счёт отказан (реестр до прогноза)" % cfg)


def check(trials: Path = TRIALS, w8: Path = W8, after: bool = False, ckpt: Path = CKPT) -> list:
    """Только чтение: 12 строк по id (не по позиции), одна `registered_at`; по умолчанию в корне W8 нет *.parquet; `after=True` — registered_at раньше parquet и `written_at` ключа `W8.stage1`."""
    cur = W5R._read(trials)
    rows = cur[cur["id"].isin(IDS)]
    bad = []
    if len(rows) != len(IDS) or set(rows["id"]) != set(IDS) or rows["id"].duplicated().any():
        bad.append("строк W8 в реестре %d из %d (ожидаются ровно id разд. 8)" % (len(rows), len(IDS)))
    if rows["registered_at"].nunique() != 1:
        bad.append("registered_at не одна: %s" % sorted(rows["registered_at"].unique()))
    if after:
        return bad + W5R._after(sorted(rows["registered_at"].unique()), w8, ckpt, "W8.stage1")
    pq = W5R._parquet(w8)
    return bad + (["в W8 есть parquet: %s" % pq] if pq else [])


if __name__ == "__main__":
    if "--check" in sys.argv or "--check-after" in sys.argv:
        b = check(after="--check-after" in sys.argv)
        r = W5R._read(TRIALS)
        r = r[r["id"].isin(IDS)]
        print("check %s: строк %d; registered_at %s; parquet в W8: %d" % ("ok" if not b else "FAIL " + "; ".join(b), len(r), sorted(r["registered_at"].unique()), len(W5R._parquet(W8))))
        sys.exit(1 if b else 0)
    print("реестр W8-A: добавлено строк %d" % register())
