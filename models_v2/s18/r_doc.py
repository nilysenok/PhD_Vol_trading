"""S-R-EVAL, документ (Ж-254 пп. 11, 16; `S_R_SPEC.md`, разд. 6, гейт E.G3): `docs/otchety/harj_struct/S_R_REPORT.md` строится ТОЛЬКО из таблиц `S_R_*.csv` и ключа `S.R.eval` (числа вручную не вписываются).
Статус: DEVELOPMENT / HISTORICAL WALK-FORWARD; первая строка отчёта — «исследовательский результат, процедура выбрана после разбора S-C-WHY, не независимый OOS (Ж-254 п. 11)».
Порядок: 1 таблица п. 16; 2 процедура (П1, П2, абляции); 3 статистика; 4 концентрация; 5 защита и решения; 6 диагностика; 7 параметры; 8 мощность; 9 стоимость и воспроизводимость.
Публичное API: `MARK`, `build(T, ck) -> str`, `confirmed(T) -> int`, `write()` (отказ SystemExit, если ключа `S.R.eval` нет), `check() -> list` (только чтение): отчёт = перестроению из таблиц; первая строка — пометка; таблица п. 16 первая, ряды по порядку;
в каждом разделе с таблицами перед первой таблицей — строка «D_M: t⁺ …, n = …»; «подтверждено» в строках — только у гипотез с Δ < 0, p Холма N < 0,05 и p SPA < 0,05 (по своей строке) и у П по своей строке; «лучше / хуже» — по знаку Δ; решения NEWCTRL-R и PROC = строкам GUARD и PROC; хэш выходов до и после равен.
Запуск: `python3 -W ignore -m models_v2.s18.r_doc [--write | --check]`."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

from models_v2.s18 import r_doc_fmt as F, r_doc_s1 as S1, r_doc_s2 as S2, r_doc_s3 as S3

REP = F.OUT_DIR
REPORT = REP / "S_R_REPORT.md"
CKPT = REP.parent / "harj_x1/CHECKPOINT.json"
KEY = "S.R.eval"
MARK = "исследовательский результат, процедура выбрана после разбора S-C-WHY, не независимый OOS (Ж-254 п. 11)"
ORDER = tuple(S1.LABEL[k] for k in S1.MAIN)
confirmed = F.confirmed


def _key() -> dict:
    ck = json.loads(CKPT.read_text(encoding="utf-8")).get(KEY)
    if not ck:
        raise SystemExit("нет ключа S.R.eval в CHECKPOINT.json: сначала `r_eval --write`")
    return ck


def build(T: dict, ck: dict) -> str:
    head = (MARK + "\n\n# S_R_REPORT — устойчивая к одной паре процедура обучения R для архитектур A2 и C-K: RV SBER, h = 1 (S-RA2 / S-RCK и абляции)\n\n"
            "Статус: DEVELOPMENT / HISTORICAL WALK-FORWARD. Спецификация `S_R_SPEC.md` (коммит 2f004d42), критерии разд. 4 объявлены и зарегистрированы до счёта (`S.R.registered`) и после результатов не менялись. "
            "Числа ниже — из `S_R_{TABLE,STATS,PROC,ABL,YEARS,CONC,GUARD,SPA,MCS,POWER,PARAMS,SEL}.csv` (генератор `models_v2/s18/r_doc.py`). «Подтверждено» — только при p Холма (N программы = 846) < 0,05 и p SPA < 0,05 по своей строке; "
            "численное улучшение при больших p не называется превосходством, отсутствие значимости не называется отсутствием выигрыша. Отдельно: техническая корректность (проверки `r_loss`, `r_walk`, `r_eval`), численное улучшение (разделы 1–2, 5), "
            "ограничения (разделы 4, 6, 8), статистическая подтверждённость (раздел 3), независимое подтверждение (раздел 5). Новая контрольная версия не требует значимости; X1 и W5-G остаются контрольными версиями в любом исходе, W8-G — экспериментальная.\n\n")
    return head + "\n\n".join([S1.sec1(T), S1.sec2(T), S1.sec3(T), S2.sec4(T), S2.sec5(T), S2.sec6(T), S3.sec7(T), S3.sec8(T), S3.sec9(T, ck)]) + "\n"


def write() -> None:
    T, ck = F.read_all(), _key()
    REPORT.write_text(build(T, ck), encoding="utf-8")


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _tree() -> dict:
    return {str(p): _sha(p) for p in sorted(REP.glob("S_R_*")) + [CKPT]}


def _sections(text: str) -> list:
    return text.split("\n## ")[1:]


def _row_checks(T: dict, text: str) -> list:
    """«подтверждено» — только на своей строке; «лучше / хуже» — по знаку Δ."""
    bad = []
    for r in T["stats"][T["stats"]["kind"] == "main"].itertuples():
        want, got = bool(r.delta_pct < 0 and r.p_holm_N < 0.05 and r.p_spa_c < 0.05), "подтверждено" in r.verdict.replace("не подтверждено", "")
        bad += [] if want == got else ["«подтверждено» не на своей строке: %s" % r.hypothesis]
        bad += [] if r.verdict.startswith("численно лучше") == (r.delta_pct < 0) else ["«лучше / не лучше» не по знаку Δ: %s" % r.hypothesis]
    for r in T["proc"].itertuples():
        want = bool(r.qlike_new < r.qlike_old and r.delta_trim_pct < 0 and r.p_holm_N < 0.05 and r.p_spa_c < 0.05)
        bad += [] if want == (str(r.confirmed) == "True") else ["«подтверждено» (П) не на своей строке: %s" % r.pid]
    sec2 = text.split("\n## 2. ")[1].split("\n## 3.")[0]
    for ln in sec2.splitlines():
        cl = [x.strip() for x in ln.strip("| ").split(" | ")]
        if len(cl) == 14 and cl[-1] in ("лучше", "хуже", "равно"):
            d = float(cl[5].replace(",", "."))
            bad += [] if cl[-1] == ("лучше" if d < 0 else "хуже" if d > 0 else "равно") else ["«лучше / хуже» не по знаку Δ: %s" % cl[0]]
    return bad


def _layout(T: dict, text: str) -> list:
    secs, bad = _sections(text), []
    first = [ln.split(" | ")[0].lstrip("| ").strip() for ln in secs[0].splitlines() if ln.startswith("| ") and not ln.startswith("| Ряд")]
    order = [next((i for i, n in enumerate(first) if n == k), -1) for k in ORDER]
    bad += [] if secs[0].startswith("1. Таблица") and any(b.startswith("| Ряд | QLIKE") for b in secs[0].split("\n\n")) else ["первой не таблица Ж-254 п. 16"]
    bad += [] if order == sorted(order) and -1 not in order else ["ряды таблицы 1 не в порядке п. 16"]
    bad += ["раздел «%s…» без строки D_M (даты, n)" % s[:12] for s in secs if any(ln.startswith("| ") for ln in s.splitlines()) and F.dm_line(T) not in s.split("\n| ")[0]]
    n_conf = sum(1 for ln in text.splitlines() if ln.startswith("| ") and "подтверждено" in ln.replace("не подтверждено", "").replace("не подтверждено статистически", ""))
    return bad + ([] if n_conf == confirmed(T) else ["«подтверждено» в таблицах: %d строк, условию отвечают %d" % (n_conf, confirmed(T))])


def check() -> list:
    h0, T, ck = _tree(), F.read_all(), _key()
    text = REPORT.read_text(encoding="utf-8")
    bad = [] if text == build(T, ck) else ["отчёт ≠ перестроению из таблиц (число подменено или таблица изменена)"]
    bad += [] if text.split("\n", 1)[0] == MARK else ["первая строка ≠ пометке «исследовательский, не OOS»"]
    bad += _layout(T, text) + _row_checks(T, text)
    g = T["guard"]
    bad += [] if "**" + g[g["id"] == "решение"]["value"].iloc[0] + "**" in text else ["решение NEWCTRL-R ≠ строке GUARD"]
    bad += [] if all("%s: %s" % (r.pid, ("да" + (", подтверждено" if r.confirmed else ", не подтверждено статистически")) if r.passed else "нет") in text for r in T["proc"].itertuples()) else ["решение PROC ≠ строкам PROC"]
    bad += [] if "Независимое подтверждение: нет" in text else ["нет строки «независимое подтверждение: нет»"]
    return bad + ([] if _tree() == h0 else ["--check изменил выходы"])


if __name__ == "__main__":
    if "--write" in sys.argv:
        write()
        print("отчёт записан: %s" % REPORT)
    elif "--check" in sys.argv:
        b = check()
        print("S-R doc --check: %s" % ("ok" if not b else "FAIL " + "; ".join(b)))
        sys.exit(1 if b else 0)
    else:
        sys.exit("укажите --write или --check")
