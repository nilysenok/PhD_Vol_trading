"""V7-FX: архив версии 1 (результаты 1–7 и E0, посчитанные с несогласованными координатами z энкодеров разных лет). Ничего не удаляется:
 · прогнозы `1b/cfg1..3.parquet`, `2/cfg4..7.parquet` — ПЕРЕНОС в `data/regen/harj_hybrid_v7/archive_v1/{1b,2}/` (на их место версия 2);
 · `1b_*.csv` (в т. ч. E0), `2a_QLIKE.csv`, `LEAF1B_REPORT.md` — КОПИИ в `archive_v1/otchety/` (оценка версии 2 — лист 2b);
 · CHECKPOINT: ключи `1b` и `2` копируются в `1b_v1` и `2_v1` с sha256 каждого файла архива; в `1b` остаются stageA, stageA_w1, E0 до пересчёта; `eval`, `rules`, cfg1–3, choice3 версии 1 живут только в `1b_v1`;
 · `ALL_TRIALS.csv` — строки «результат v1 архивирован: дефект согласования z» (строки версии 1 не удаляются); `archive_v1/README.md` — sha256 и причина.
`check_archive` ничего не пишет: сверяет sha256 файлов архива с CHECKPOINT."""
from __future__ import annotations

import copy
import json
import shutil

from models_v2.s14 import v7_0_io as IO, v7_1m_reg as RG

ARCH = IO.OUT / "archive_v1"
MOVE = {"1b": ("cfg1.parquet", "cfg2.parquet", "cfg3.parquet"), "2": ("cfg4.parquet", "cfg5.parquet", "cfg6.parquet", "cfg7.parquet")}
COPY = ("1b_*.csv", "2a_QLIKE.csv", "LEAF1B_REPORT.md")
V1_ONLY = ("cfg1", "cfg2", "cfg3", "choice3", "eval", "rules")
REASON = ("дефект согласования z (Ж-237, «Моя ошибка [V7]» 18:13): в версии 1 строка даты года y брала z от энкодера года y, так что при переоценке в году Y корректор учился на парах прошлых лет в координатах других энкодеров "
          "(знак, порядок и масштаб компонент не согласованы; `docs/otchety/harj_hybrid_v7/FX_DIAG.csv`), а у пар 2017 года z не было совсем (F = H до конца 2018)")


def files() -> list:
    return [ARCH / d / n for d, ns in MOVE.items() for n in ns] + [ARCH / "otchety" / f.name for f in sorted(ARCH.glob("otchety/*"))]


def sha_table() -> dict:
    return {str(f.relative_to(ARCH)): IO.sha_file(f) for f in files()}


def readme(table: dict) -> str:
    rows = "\n".join("| `%s` | %s |" % (k, v) for k, v in table.items())
    return "# Архив версии 1 (результаты 1–7 и E0 до исправления согласования z)\n\n**Причина.** %s.\n\nЧто здесь: прогнозы 1–7 (перенесены), таблицы `1b_*.csv`, `2a_QLIKE.csv`, отчёт `LEAF1B_REPORT.md` (копии). Ключи CHECKPOINT: `1b_v1`, `2_v1` (те же sha256).\n\n| файл | sha256 |\n|---|---|\n%s\n" % (REASON, rows)


def archive(when_note: str = "") -> str:
    """Перенос/копирование и запись ключей. Повтор — 'уже архивировано' (README существует). -> 'архивировано' | 'уже архивировано'."""
    if (ARCH / "README.md").exists():
        return "уже архивировано"
    cp = json.loads((IO.REP / "CHECKPOINT.json").read_text(encoding="utf-8"))
    for d, ns in MOVE.items():
        (ARCH / d).mkdir(parents=True, exist_ok=True)
        for n in ns:
            shutil.move(str(IO.OUT / d / n), str(ARCH / d / n))
    (ARCH / "otchety").mkdir(parents=True, exist_ok=True)
    for pat in COPY:
        for f in sorted(IO.REP.glob(pat)):
            shutil.copy2(f, ARCH / "otchety" / f.name)
    table = sha_table()
    for key, old in (("1b_v1", "1b"), ("2_v1", "2")):
        v = copy.deepcopy(cp[old])
        v["archive_sha256"] = {k: s for k, s in table.items() if k.startswith(old + "/") or (old == "1b" and k.startswith("otchety/"))}
        v["reason"] = REASON
        IO.merge_checkpoint(key, v)
    IO.merge_checkpoint("1b", {k: v for k, v in cp["1b"].items() if k not in V1_ONLY})
    IO.merge_checkpoint("2", {})
    for cfg in "1234567":
        rec = (cp["1b"] if cfg in "123" else cp["2"])["cfg" + cfg]
        RG.add_result(cfg, dict(reason="результат v1 архивирован: дефект согласования z", sha256_content=rec["sha256_content"], qlike_2018_2021=rec["qlike_2018_2021"], n_rows=rec["n_rows"]), suffix=RG.ARCH_SUFFIX)
    (ARCH / "README.md").write_text(readme(table), encoding="utf-8")
    return "архивировано"


def check_archive(cp: dict) -> list:
    """-> замечания (пусто = ок): ключи `1b_v1`, `2_v1`, каждый файл архива есть и sha256 = записанному, README на месте. Ничего не пишет."""
    bad = []
    if "1b_v1" not in cp or "2_v1" not in cp:
        return ["CHECKPOINT: нет ключей 1b_v1/2_v1 (архив версии 1)"]
    rec = {**cp["1b_v1"].get("archive_sha256", {}), **cp["2_v1"].get("archive_sha256", {})}
    for k, s in rec.items():
        f = ARCH / k
        if not f.exists() or IO.sha_file(f) != s:
            bad.append("архив v1: %s нет или sha256 ≠ CHECKPOINT" % k)
    if len([k for k in rec if k.endswith(".parquet")]) != 7 or not (ARCH / "README.md").exists():
        bad.append("архив v1: нет семи прогнозов или README.md")
    return bad
