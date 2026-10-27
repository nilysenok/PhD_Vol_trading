#!/usr/bin/env python3
"""Сверка блоков журнала решений (Ж-025 п. 1.1, 1.4).

Читает docs/resheniya/resheniya.md. Блок: заголовок `## Ж-NNN · ЧЧ:ММ`, текст,
строка «принят Ж-NNN, ЧЧ:ММ» или «квитанции не было — не дошёл (ретроспективно)».
Выход 1, если есть блок без квитанции/пометки, пропуск номера или убывание времени.
"""
import re
import sys

HEADER_RE = re.compile(r"^##\s+Ж-(\d{3})\s*·\s*(\d{1,2}):(\d{2})", re.M)
RECEIPT_RE = re.compile(r"принят\s+Ж-(\d{3})\s*,\s*\d{1,2}:\d{2}", re.I)
RETRO_RE = re.compile(r"квитанции\s+не\s+было\s*[—–-]\s*не\s+дошёл\s*\(ретроспективно\)", re.I)
DEFAULT = "docs/resheniya/resheniya.md"


def razobrat(text):
    """Список блоков: {n, t (минуты), receipt, retro}."""
    heads = list(HEADER_RE.finditer(text))
    blocks = []
    for i, m in enumerate(heads):
        end = heads[i + 1].start() if i + 1 < len(heads) else len(text)
        body = text[m.end():end]
        n = int(m.group(1))
        blocks.append({
            "n": n,
            "t": int(m.group(2)) * 60 + int(m.group(3)),
            "receipt": any(int(r.group(1)) == n for r in RECEIPT_RE.finditer(body)),
            "retro": bool(RETRO_RE.search(body)),
        })
    return blocks


def bez_kvitantsii(blocks):
    return [b for b in blocks if not b["receipt"] and not b["retro"]]


def oshibki_poryadka(blocks):
    """Пропуски/повторы номеров и убывание времени."""
    errs = []
    for a, b in zip(blocks, blocks[1:]):
        if b["n"] != a["n"] + 1:
            errs.append(f"номер: после Ж-{a['n']:03d} идёт Ж-{b['n']:03d}")
        if b["t"] < a["t"]:
            errs.append(f"время убывает: Ж-{b['n']:03d} раньше Ж-{a['n']:03d}")
    return errs


def sverka(text):
    """(строки отчёта, код выхода)."""
    blocks = razobrat(text)
    if not blocks:  # Ж-025 п. 2.4: пустой вывод — ошибка, не успех
        return ["блоков всего: 0", "журнал пуст или не найден"], 1
    bad = bez_kvitantsii(blocks)
    errs = oshibki_poryadka(blocks)
    retro = [b for b in blocks if b["retro"] and not b["receipt"]]
    out = [
        f"блоков всего: {len(blocks)}",
        f"блоков без квитанции: {len(bad)}",
        f"ретроспективно не дошедших: {len(retro)}",
    ]
    out += [f"без квитанции: Ж-{b['n']:03d}" for b in bad]
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
