#!/usr/bin/env python3
"""Сверка блоков журнала решений (Ж-025 п. 1.1, 1.2, 1.4, 2.4).

Читает docs/resheniya/resheniya.md. Блок: заголовок уровня 2 или 3
`### Ж-NNN · ДД.ММ.ГГГГ, ЧЧ:ММ · …` или `### Ж-б/н-ДД.ММ-ЧЧ:ММ · …` (без номера),
текст, ровно одна строка `Квитанция: …` («принят Ж-NNN, ЧЧ:ММ» или
«квитанции не было — не дошёл (ретроспективно)»).
Выход 1: журнал пуст; блок без квитанции; не одна строка `Квитанция:`; номер
в квитанции не совпал; пропуск числового номера; убывание времени.
"""
import re
import sys

HEADER_RE = re.compile(
    r"^#{2,3}[ \t]+Ж-(\d{3}|б/н-\d{1,2}\.\d{1,2}-\d{1,2}:\d{2})[ \t]*(?:·(.*))?$", re.M)
LINE_RE = re.compile(r"^Квитанция:(.*)$", re.M)
RECEIPT_RE = re.compile(r"принят\s+Ж-(\d{3})\s*,\s*\d{1,2}:\d{2}", re.I)
RETRO_RE = re.compile(r"квитанции\s+не\s+было\s*[—–-]\s*не\s+дошёл\s*\(ретроспективно\)", re.I)
DATE_RE = re.compile(r"(\d{1,2})\.(\d{1,2})\.(\d{4})")
TIME_RE = re.compile(r"(\d{1,2}):(\d{2})")
DEFAULT = "docs/resheniya/resheniya.md"


def _vremya(rest):
    """Ключ (ГГГГ, ММ, ДД, ч, мин) из первого поля после «·» или None."""
    field = (rest or "").split("·")[0]
    d, t = DATE_RE.search(field), TIME_RE.search(DATE_RE.sub("", field))
    if not d or not t:
        return None
    return (int(d.group(3)), int(d.group(2)), int(d.group(1)), int(t.group(1)), int(t.group(2)))


def razobrat(text):
    """Блоки: {id, n (число или None), t, lines (строки «Квитанция:»), receipt, retro}."""
    heads = list(HEADER_RE.finditer(text))
    blocks = []
    for i, m in enumerate(heads):
        end = heads[i + 1].start() if i + 1 < len(heads) else len(text)
        lines = LINE_RE.findall(text[m.end():end])
        ids = m.group(1)
        blocks.append({
            "id": "Ж-" + ids,
            "n": int(ids) if ids.isdigit() else None,
            "t": _vremya(m.group(2)),
            "lines": lines,
            "receipt": [int(r.group(1)) for ln in lines for r in RECEIPT_RE.finditer(ln)],
            "retro": any(RETRO_RE.search(ln) for ln in lines),
        })
    return blocks


RETRO_DO = (2026, 10, 27, 15, 40)  # с этого момента «не дошёл» не принимается


def retro_dopustima(b):
    """Пометка допустима только для блока со временем раньше RETRO_DO."""
    return b["retro"] and b["t"] is not None and b["t"] < RETRO_DO


def bez_kvitantsii(blocks):
    return [b for b in blocks if not b["receipt"] and not retro_dopustima(b)]


def oshibki_kvitantsii(blocks):
    errs = []
    for b in blocks:
        if len(b["lines"]) > 1:
            errs.append(f"{b['id']}: строк «Квитанция:» {len(b['lines'])}, нужна одна")
        if b["receipt"] and b["receipt"] != [b["n"]]:
            errs.append(f"{b['id']}: номер в квитанции {b['receipt']} не совпал с блоком")
    return errs


def oshibki_poryadka(blocks):
    """Числовые номера подряд (по множеству); время не убывает в порядке файла."""
    errs = []
    nums = sorted(b["n"] for b in blocks if b["n"] is not None)
    for a, b in zip(nums, nums[1:]):
        if b != a + 1:
            errs.append(f"номер: после Ж-{a:03d} идёт Ж-{b:03d}")
    timed = [b for b in blocks if b["t"] is not None]
    for a, b in zip(timed, timed[1:]):
        if b["t"] < a["t"]:
            errs.append(f"время убывает: {b['id']} раньше {a['id']}")
    return errs


def sverka(text):
    """(строки отчёта, код выхода)."""
    blocks = razobrat(text)
    if not blocks:  # Ж-025 п. 2.4: пустой вывод — ошибка, не успех
        return ["блоков всего: 0", "журнал пуст или не найден"], 1
    bad = bez_kvitantsii(blocks)
    errs = oshibki_kvitantsii(blocks) + oshibki_poryadka(blocks)
    a = sum(1 for b in blocks if b["n"] is not None)
    out = [
        f"блоков всего: {len(blocks)} (с номером: {a}, без номера: {len(blocks) - a})",
        f"блоков без квитанции: {len(bad)}",
        f"ретроспективно не дошедших: {sum(1 for b in blocks if retro_dopustima(b))}",
    ]
    out += [f"без квитанции: {b['id']}" for b in bad]
    out += errs
    return out, (1 if bad or errs else 0)


def main(argv):
    path = argv[1] if len(argv) > 1 else DEFAULT
    try:
        with open(path, encoding="utf-8") as f:
            text = f.read()
    except OSError:
        text = ""
    lines, code = sverka(text)
    print("\n".join(lines))
    return code


if __name__ == "__main__":
    sys.exit(main(sys.argv))
