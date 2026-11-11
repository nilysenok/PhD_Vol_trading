"""S-A-EVAL, документ (Ж-254 пп. 11, 16; `S_A_SPEC.md`, разд. 8, гейт E.G3): `docs/otchety/harj_struct/S_A_REPORT.md` строится ТОЛЬКО из таблиц `S_A_*.csv` и ключа `S.A.eval` (числа вручную не вписываются).
Статус: DEVELOPMENT / HISTORICAL WALK-FORWARD; первая строка отчёта — «исследовательский результат на раскрытой истории, не независимый OOS (Ж-254 п. 11); архитектура выбрана после изучения истории».
Порядок: 1 таблица рядов (HAR-J / XGBoost / LightGBM / X1 / W5-G / W8-G как экспериментальный контроль / новые гибриды) и слой X1_A против X1; 2 статистика и мощность; 3 годы и вклады; 4 абляции; 5 У1-S…У5-S и решение; 6 концентрация; 7 параметры, стоимость, воспроизводимость; 8 ответ.
Публичное API:
- `MARK` — текст первой строки; `build(T, ck) -> str`: текст отчёта по таблицам `T` (`a_doc_fmt.read_all`) и ключу `ck`; `confirmed(T) -> int`: число подтверждённых гипотез (Δ < 0, p Холма N < 0,05 и p SPA < 0,05).
- `write()`: пишет отчёт (после таблиц и ключа `S.A.eval`; отказ SystemExit, если ключа нет).
- `check() -> list` (только чтение): отчёт = перестроению из таблиц; первая строка — пометка; таблица Ж-254 п. 16 первая, ряды в порядке п. 16; в каждом разделе с таблицами стоит строка «D_M: t⁺ 2018-01-04…2026-11-10, n = 2198»;
  «подтверждено» в строках таблиц — только у гипотез с p Холма < 0,05 и p SPA < 0,05; решение = строка GUARD; ответ «независимое подтверждение: нет»; хэш выходов до и после равен.
Запуск: `python3 -W ignore -m models_v2.s18.a_doc [--write | --check]`."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

from models_v2.s18 import a_doc_fmt as F, a_doc_s1 as S1, a_doc_s2 as S2, a_doc_s3 as S3

REP = F.OUT_DIR
REPORT = REP / "S_A_REPORT.md"
CKPT = REP.parent / "harj_x1/CHECKPOINT.json"
MARK = "исследовательский результат на раскрытой истории, не независимый OOS (Ж-254 п. 11); архитектура выбрана после изучения истории"
ORDER = ("HAR-J (K1)", "XGBoost (K2)", "LightGBM (K3)", "X1", "W5-G", "W8-G (экспериментальный контроль)", "S-A0", "S-A1", "S-A2")


def confirmed(T: dict) -> int:
    m = T["stats"][T["stats"]["kind"] == "main"]
    return int(((m["delta_pct"] < 0) & (m["p_holm_N"] < 0.05) & (m["p_spa_c"] < 0.05)).sum())


def _key() -> dict:
    ck = json.loads(CKPT.read_text(encoding="utf-8")).get("S.A.eval")
    if not ck:
        raise SystemExit("нет ключа S.A.eval в CHECKPOINT.json: сначала `a_eval --write`")
    return ck


def build(T: dict, ck: dict) -> str:
    head = (MARK + "\n\n# S_A_REPORT — пилот S-A0 / S-A1 / S-A2: совместная внутридневная агрегация и память HAR (SBER, h = 1)\n\n"
            "Статус: DEVELOPMENT / HISTORICAL WALK-FORWARD. Спецификация `S_A_SPEC.md` (коммит 12d098dc), критерии разд. 6 объявлены и зарегистрированы до счёта и после результатов не менялись. "
            "Числа ниже — из `S_A_{TABLE,STATS,YEARS,CONC,GUARD,SPA,MCS,POWER,PARAMS}.csv` (генератор `models_v2/s18/a_doc.py`). «Подтверждено» — только при p Холма (N программы = 831) < 0,05 и p SPA < 0,05; "
            "численное улучшение при больших p не называется превосходством, отсутствие значимости не называется отсутствием выигрыша. Новая контрольная версия не требует значимости; X1 и W5-G остаются контрольными версиями в любом исходе, W8-G — экспериментальная.\n\n")
    s7 = S3.sec7(T, ck).replace("## 7. Параметры, число обученных вариантов, стоимость, воспроизводимость\n\n", "## 7. Параметры, число обученных вариантов, стоимость, воспроизводимость\n\n" + F.dm_line(T) + " Таблицы параметров — по датам переоценки.\n\n")
    return head + "\n\n".join([S1.sec1(T), S1.sec2(T), S1.sec3(T), S2.sec4(T), S2.sec5(T), S2.sec6(T), s7, S3.sec8(T)]) + "\n"


def write() -> None:
    T, ck = F.read_all(), _key()
    REPORT.write_text(build(T, ck), encoding="utf-8")


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _tree() -> dict:
    return {str(p): _sha(p) for p in sorted(REP.glob("S_A_*")) + [CKPT]}


def _sections(text: str) -> list:
    return [s for s in text.split("\n## ")[1:]]


def _row_checks(T: dict, text: str) -> list:
    """По строкам: «подтверждено» — только на своей строке (Δ < 0, p Холма N < 0,05, p SPA < 0,05); «численно лучше / не лучше» — по знаку Δ; «лучше / хуже» абляций — по знаку Δ."""
    bad = []
    for r in T["stats"][T["stats"]["kind"] == "main"].itertuples():
        want, got = bool(r.delta_pct < 0 and r.p_holm_N < 0.05 and r.p_spa_c < 0.05), "подтверждено" in r.verdict.replace("не подтверждено", "")
        bad += [] if want == got else ["«подтверждено» не на своей строке: %s (Δ %.4f, p Холма %.4f, p SPA %.4f)" % (r.hypothesis, r.delta_pct, r.p_holm_N, r.p_spa_c)]
        bad += [] if r.verdict.startswith("численно лучше") == (r.delta_pct < 0) else ["«лучше / не лучше» не по знаку Δ: %s" % r.hypothesis]
    if "\n## 4. " not in text or "\n## 5." not in text:
        return bad + ["нет разделов 4–5 отчёта"]
    sec = text.split("\n## 4. ")[1].split("\n## 5.")[0]
    for ln in sec.splitlines():
        cl = [x.strip() for x in ln.strip("| ").split(" | ")]
        if len(cl) == 8 and cl[-1] in ("лучше", "хуже", "равно"):
            d = float(cl[4].replace(",", "."))
            bad += [] if cl[-1] == ("лучше" if d < 0 else "хуже" if d > 0 else "равно") else ["«лучше / хуже» не по знаку Δ: %s" % cl[0]]
    return bad


def check() -> list:
    h0, T, ck = _tree(), F.read_all(), _key()
    text = REPORT.read_text(encoding="utf-8")
    bad = [] if text == build(T, ck) else ["отчёт ≠ перестроению из таблиц (число подменено или таблица изменена)"]
    bad += [] if text.split("\n", 1)[0] == MARK else ["первая строка ≠ пометке «исследовательский, не OOS»"]
    secs = _sections(text)
    t1 = secs[0].split("\n\n")
    first = [ln.split(" | ")[0].lstrip("| ").strip() for ln in secs[0].splitlines() if ln.startswith("| ") and not ln.startswith("| Ряд")]
    names = [n for n in (x for x in first) if n]
    bad += [] if secs[0].startswith("1. Таблица") and any(b.startswith("| Ряд | QLIKE") for b in t1) else ["первой не таблица Ж-254 п. 16"]
    order = [next((i for i, n in enumerate(names) if n.startswith(k)), -1) for k in ORDER]
    bad += [] if order == sorted(order) and -1 not in order else ["ряды таблицы 1 не в порядке п. 16"]
    for s in secs:
        if any(ln.startswith("| ") for ln in s.splitlines()) and F.dm_line(T) not in s.split("\n| ")[0]:
            bad.append("раздел «%s…» без строки D_M (даты, n)" % s[:12])
    n_conf_lines = sum(1 for ln in text.splitlines() if ln.startswith("| ") and "подтверждено" in ln.replace("не подтверждено", ""))
    bad += [] if n_conf_lines == confirmed(T) else ["«подтверждено» в таблицах: %d строк, условию (p Холма < 0,05 и p SPA < 0,05) отвечают %d" % (n_conf_lines, confirmed(T))]
    bad += _row_checks(T, text)
    g = T["guard"]
    bad += [] if "**" + g[g["id"] == "решение"]["value"].iloc[0] + "**" in text else ["решение ≠ строке GUARD"]
    bad += [] if "Независимое подтверждение: нет" in text else ["нет строки «независимое подтверждение: нет»"]
    return bad + ([] if _tree() == h0 else ["--check изменил выходы"])


if __name__ == "__main__":
    if "--write" in sys.argv:
        write()
        print("отчёт записан: %s" % REPORT)
    elif "--check" in sys.argv:
        b = check()
        print("S-A doc --check: %s" % ("ok" if not b else "FAIL " + "; ".join(b)))
        sys.exit(1 if b else 0)
    else:
        sys.exit("укажите --write или --check")
