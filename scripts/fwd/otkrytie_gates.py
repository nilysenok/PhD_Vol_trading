"""Л6: проверки состава заморозки, гейтов и репетиции для `otkrytie_proverki.preflight` (протокол, раздел 6, шаги 2–3). Ничего не пишет.
Допущение: sha256 новых файлов и `rv_fwd_SEALED.parquet` считается по байтам (как в `otkrytie_gl4`), значения не разбираются.
"""
from __future__ import annotations

import re
from pathlib import Path

GATE_LISTS = ("Л1", "Л2", "Л3", "Л4", "Л5")
ZERO_DIFF = "0 расхождений"
BAD = re.compile(r"\bpending\b|\bABANDON\b|\bTODO\b", re.I)
SKIP_ID = "L6.8"   # гейт «открытие выполнено» — по id, запускает дирижёр
MANIFEST_DIRS = ("sber_m10", "mkt_m10")


def check_gates(gates: Path) -> list:
    """У Л1–Л5: нет pending / ABANDON / TODO в любой строке (таблица, список, текст) и нет пустого EVIDENCE (последняя ячейка строки таблицы)."""
    if not gates.is_file():
        return ["нет GATES_FWD.md"]
    secs = {m.group(1): b for b in re.split(r"(?m)^## ", gates.read_text(encoding="utf-8"))[1:] for m in [re.match(r"(Л\d)\.", b)] if m}
    bad = []
    for g in GATE_LISTS:
        if g not in secs:
            bad.append(f"нет раздела {g} в GATES_FWD.md")
            continue
        for ln in secs[g].splitlines():
            s = ln.strip()
            if re.match(rf"^[|\-*\s]*{re.escape(SKIP_ID)}(?![\w.])", s):  # id гейта — первая метка строки, точно
                continue
            if BAD.search(s):
                bad.append(f"{g}: незакрыто (pending/ABANDON/TODO): {s[:70]}")
            elif s.startswith("|") and not re.fullmatch(r"[|\s:\-]+", s) and s.strip("|").split("|")[-1].strip() in ("", "—", "-"):
                bad.append(f"{g}: пустой EVIDENCE: {s[:70]}")
    return bad


def rehearsal_problems(path: Path) -> list:
    """Разбор REHEARSAL.csv кодом Л5 (`fwd_rehearsal.problems`): пустой список — «0 расхождений»."""
    import pandas as pd
    from models_v2.s17 import fwd_rehearsal as RH
    return RH.problems(pd.read_csv(path, comment="#"))


def required_inputs(root: Path) -> tuple:
    """Обязательные ключи заморозки: (архив, новые). Архив — бары 17 бумаг и IMOEX, k23_meta, 5 артефактов энкодера; новые — манифесты, SBER, rv_fwd_SEALED, 16 бумаг и IMOEX."""
    from models_v2.s10 import hp_k23_world as KW
    from models_v2.s14 import v7_1b_sta as STA
    from models_v2.s17 import fwd_src as S
    src = S.Src()
    rel = lambda p: str(Path(p).resolve().relative_to(Path(root).resolve()))  # noqa: E731
    arch = [rel(src.archive_path(t)) for t in KW.POOL + ["IMOEX"]] + ["data/regen/harj_panel_hybrid/k23_meta_SBER.json"]
    arch += [f"data/regen/harj_hybrid_v7/stageA/{STA.name('tcn', 2026, s)}" for s in range(5)]
    new = [f"data/fwd/{d}/MANIFEST.json" for d in MANIFEST_DIRS] + ["data/fwd/sber_m10/SBER_m10_fwd.parquet", "data/fwd/sber_m10/rv_fwd_SEALED.parquet"]
    new += [f"data/fwd/mkt_m10/{t}_m10_fwd.parquet" for t in KW.POOL + ["IMOEX"] if t != "SBER"]
    return arch, new


def check_required(fz: dict, req: tuple) -> list:
    bad = []
    for grp, need in (("inputs_archive_sha256", req[0]), ("inputs_new_sha256", req[1])):
        have = fz.get(grp) or {}
        bad += [f"{grp}: пусто"] if not have else []
        bad += [f"{grp}: нет обязательного входа {k}" for k in need if k not in have]
    return bad


def check_rehearsal(fz: dict, path: Path, rh_problems) -> list:
    """Строка FREEZE `rehearsal.result` и сам REHEARSAL.csv: 38 дней × узлы, 0 расхождений (не верим одной строке)."""
    bad = [] if (fz.get("rehearsal") or {}).get("result") == ZERO_DIFF else [f"rehearsal.result не «{ZERO_DIFF}»"]
    if not path.is_file():
        return bad + ["нет REHEARSAL.csv"]
    try:
        p = rh_problems(path)
    except Exception as e:  # noqa: BLE001  нечитаемая таблица — тоже отказ
        return bad + [f"REHEARSAL.csv не разобран: {type(e).__name__}: {e}"]
    return bad + [f"REHEARSAL: {m}" for m in p[:5]] + ([f"REHEARSAL: всего замечаний {len(p)}"] if p else [])
