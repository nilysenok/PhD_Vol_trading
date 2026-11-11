"""W5-A (Ж-250, Ж-251; `docs/otchety/harj_x1/W5_SPEC.md`, раздел 7): регистратор и сторож «реестр до прогноза». По образцу `x1_w4_reg`.
Публичное API:
- `ROWS` / `IDS` — 28 строк раздела 7: W5D1…W5D18 (поиск: множитель уровня X1 по состоянию, порядок absCM, absK23, dow, e, err, ev_any, ev_cbr, ev_cpius, ev_exp, ev_fomc,
  ev_rebal, gap, lvl, pi, sgnCM, tCM, tK1M, v), W5D19…W5D24 (постоянный π по группе: absCM, e, pi, sgnCM, tCM, tK1M), W5-G, W5-Ga, W5-Gb, W5-GK2. `N_PROGRAM` = 798, `N_WIDE` = 3301.
- `register(now=None, trials=TRIALS, w5=W5, ckpt=CKPT) -> 28` дописывает ВСЕ 28 строк разом с ОДНОЙ `registered_at` (часы машины) и сливает в `CHECKPOINT.json` ключи `W5` и
  `W5.registered` (прочие ключи — побайтно). Отказ (SystemExit): в корне `w5` (`data/regen/harj_x1/W5/`, подкаталоги не считаются) уже есть `*.parquet` (реестр позже прогноза)
  или хоть один из id уже в реестре. `ckpt=None` — без записи CHECKPOINT.
- `merge_checkpoint(ckpt, now, rep=REP, code=<каталог модуля>)`: ключ `W5` = sha256 файлов `W5_DIAG.csv`, `W5_TABLE0.csv`, `W5_GAPS.csv`, `W5_SPEC.md` из `rep` и всех `x1_w5_*.py` из `code`,
  `configurations` 0, `forecasts_written` 0; ключ `W5.registered` = spec_sha256, registered_at, n, n_wide, rows. Формат файла: indent 1, sorted keys.
- `require(cfg, trials=TRIALS)`: отказ (SystemExit), если id нет в реестре (для W5-B).
- `check(trials=TRIALS, w5=W5, after=False, ckpt=CKPT) -> list`; `after=True` (`--check-after`, после прогноза): registered_at раньше mtime самого раннего parquet в W5 и раньше времени ключа `W5.eval` (если оно есть) — замечаний, только чтение: ровно 28 id раздела 7 (по id, не по позиции), без дублей, одна `registered_at`, в корне `w5` нет `*.parquet`.
- `check_gaps(gaps=GAPS) -> list` замечаний, только чтение: `x1_w5_gaps.check` — перечень пар с перерывом не по выходным пересчитан и равен файлу `W5_GAPS.csv`, 66 пар.
Запуск: `python3 -W ignore -m models_v2.s16.x1_w5_reg [--check]`; без флага — регистрация (однократная)."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

from models_v2.s16 import x1_reg as R
from models_v2.s16 import x1_w5_gaps as G

ROOT, REP = R.ROOT, R.REP
TRIALS, COLS, now_text = R.TRIALS, R.COLS, R.now_text
CKPT = REP / "CHECKPOINT.json"
GAPS = REP / "W5_GAPS.csv"
W5 = ROOT / "data/regen/harj_x1/W5"
N_PROGRAM, N_WIDE = 798, 3301
DIAG = "диагностика W5, критериев не получает"
LEVEL = ("absCM", "absK23", "dow", "e", "err", "ev_any", "ev_cbr", "ev_cpius", "ev_exp", "ev_fomc", "ev_rebal", "gap", "lvl", "pi", "sgnCM", "tCM", "tK1M", "v")
TRUST = ("absCM", "e", "pi", "sgnCM", "tCM", "tK1M")
ROWS = (
    [("W5D%d" % (i + 1), "W5 поиск: множитель уровня X1, состояние %s" % s, "F = X1·c_G, c_G = mean(RV/X1) группы состояния %s; перенос между половинами" % s, "—", DIAG)
     for i, s in enumerate(LEVEL)]
    + [("W5D%d" % (19 + i), "W5 поиск: постоянный π по группе, состояние %s" % s, "F = M^(1−π_G)·C^π_G, π_G — лучший на сетке 0…1 шаг 0,05 группы состояния %s" % s, "—", DIAG)
       for i, s in enumerate(TRUST)]
    + [
        ("W5-G", "W5: уровень X1 после перерыва торгов", "F = X1·exp(gap·β), gap = 1[Δ_d > 0], β = clip((ū₁ − ū₀) + ½(s²₁ − s²₀), 0; 0,5) по закрытым парам переоценки, u = ln(RV/X1)", "нет сетки", "зарегистрирована"),
        ("W5-Ga", "W5: контроль (а)", "F = X1·exp(clip(ū + ½s², −0,5; 0,5)) на все дни", "—", "контроль, критериев не получает"),
        ("W5-Gb", "W5: контроль (б)", "W5-G с меткой gap, сдвинутой на 252 строки окна M", "—", "контроль"),
        ("W5-GK2", "W5: справочно", "W5-G над K2: u = ln(RV/K2)", "—", "справочно, критериев не получает"),
    ]
)
IDS = tuple(r[0] for r in ROWS)
assert len(ROWS) == 28 and len(set(IDS)) == 28


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _read(trials: Path) -> pd.DataFrame:
    return pd.read_csv(trials, comment="#", dtype=str, keep_default_na=False) if trials.exists() else pd.DataFrame(columns=COLS)


def _parquet(w5: Path) -> list:
    return sorted(p.name for p in w5.glob("*.parquet")) if w5.is_dir() else []


def merge_checkpoint(ckpt: Path, now: str, rep: Path = REP, code: Path = Path(__file__).resolve().parent) -> None:
    """Только ключи W5 и W5.registered, слиянием (формат файла: indent 1, sorted keys); прочие ключи сохраняются побайтно."""
    cp = json.loads(ckpt.read_text(encoding="utf-8"))
    names = ("W5_DIAG.csv", "W5_TABLE0.csv", "W5_GAPS.csv", "W5_SPEC.md")
    cp["W5"] = dict(sha256={n: _sha(rep / n) for n in names}, code={p.name: _sha(p) for p in sorted(code.glob("x1_w5_*.py"))}, configurations=0, forecasts_written=0)
    cp["W5.registered"] = dict(spec_sha256=_sha(rep / "W5_SPEC.md"), registered_at=now, n=N_PROGRAM, n_wide=N_WIDE, rows=len(ROWS))
    ckpt.write_text(json.dumps(cp, indent=1, sort_keys=True, ensure_ascii=False), encoding="utf-8")


def register(now: str | None = None, trials: Path = TRIALS, w5: Path = W5, ckpt: Path | None = CKPT) -> int:
    """Дописывает ВСЕ 28 строк разом (одна `registered_at`), затем сливает ключи CHECKPOINT. Отказ (SystemExit): в W5 уже есть *.parquet или любой из id уже в реестре."""
    pq = _parquet(w5)
    if pq:
        raise SystemExit("реестр позже прогноза: в %s уже есть %s — регистрация отказана" % (w5, pq))
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
    """Для W5-B: отказ (SystemExit), если id нет в реестре."""
    if cfg not in set(_read(trials)["id"]):
        raise SystemExit("%s: нет в реестре ALL_TRIALS.csv — счёт отказан (реестр до прогноза)" % cfg)


def _t(s: str) -> dt.datetime:
    return dt.datetime.strptime(s, "%d.%m.%Y %H:%M:%S")


def _key_time(ckpt: Path, key: str):
    """Время записи ключа стадии в CHECKPOINT (поля written_at / time / timestamp / created_at / at), если оно есть; иначе None."""
    v = json.loads(ckpt.read_text(encoding="utf-8")).get(key) if ckpt.exists() else None
    for f in ("written_at", "time", "timestamp", "created_at", "at"):
        if isinstance(v, dict) and isinstance(v.get(f), str):
            return _t(v[f])
    return None


def _after(reg_at: list, w5: Path, ckpt: Path, key: str) -> list:
    """Режим «после прогноза»: registered_at раньше mtime самого раннего parquet и раньше времени записи ключа стадии (если оно есть). mtime — по часам файловой системы (копирование его меняет)."""
    ps = sorted(w5.glob("*.parquet")) if w5.is_dir() else []
    if not ps:
        return ["режим after: в %s нет parquet, сравнивать не с чем" % w5]
    try:
        r = _t(reg_at[0])
    except (ValueError, IndexError):
        return ["registered_at не разобрана: %s" % reg_at]
    first = min(dt.datetime.fromtimestamp(p.stat().st_mtime) for p in ps)
    bad = [] if r < first else ["registered_at %s не раньше mtime самого раннего parquet %s" % (r, first)]
    kt = _key_time(ckpt, key)
    return bad + ([] if kt is None or r < kt else ["registered_at %s не раньше времени записи ключа %s (%s)" % (r, key, kt)])


def check(trials: Path = TRIALS, w5: Path = W5, after: bool = False, ckpt: Path = CKPT) -> list:
    """Только чтение. -> список замечаний: 28 строк по id (не по позиции), одна `registered_at`; по умолчанию — в корне W5 нет *.parquet.
    `after=True` (CLI `--check-after`, после прогноза): вместо «parquet нет» — registered_at раньше mtime самого раннего parquet в W5 и раньше времени записи ключа `W5.eval` в CHECKPOINT (если в ключе есть время)."""
    cur = _read(trials)
    rows = cur[cur["id"].isin(IDS)]
    bad = []
    if len(rows) != len(IDS) or set(rows["id"]) != set(IDS) or rows["id"].duplicated().any():
        bad.append("строк W5 в реестре %d из %d (ожидаются ровно id раздела 7)" % (len(rows), len(IDS)))
    if rows["registered_at"].nunique() != 1:
        bad.append("registered_at не одна: %s" % sorted(rows["registered_at"].unique()))
    if after:
        return bad + _after(sorted(rows["registered_at"].unique()), w5, ckpt, "W5.eval")
    pq = _parquet(w5)
    if pq:
        bad.append("в W5 есть parquet: %s" % pq)
    return bad


def check_gaps(gaps: Path = GAPS) -> list:
    """Только чтение: перечень перерывов не по выходным пересчитан и равен файлу; 66 пар."""
    return G.check(gaps)


if __name__ == "__main__":
    if "--check" in sys.argv or "--check-after" in sys.argv:
        b = check(after="--check-after" in sys.argv) + check_gaps()
        r = _read(TRIALS)
        r = r[r["id"].isin(IDS)]
        print("check %s: строк %d; registered_at %s; parquet в W5: %d; пар в W5_GAPS.csv: %d" % (
            "ok" if not b else "FAIL " + "; ".join(b), len(r), sorted(r["registered_at"].unique()), len(_parquet(W5)), len(pd.read_csv(GAPS, comment="#")) if GAPS.exists() else -1))
        sys.exit(1 if b else 0)
    print("реестр W5-A: добавлено строк %d" % register())
