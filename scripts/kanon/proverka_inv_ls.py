#!/usr/bin/env python3
"""Гейты Л-С отчёта инвентаризации (G1…G4).

Использование: proverka_inv_ls.py <отчёт.md> G1|G2|G3|G4
G1 — номера разделов «## П. N.» подряд; G2 — строки таблицы п. 8: знак и
(«оценки нет» либо число со ссылкой на .csv/.md/.py); G3 — у каждого вопроса
п. 9 есть «Умолчание:»; G4 — все строки с 2020 или 2021 (просмотр глазами).
"""
import re
import sys

ZNAKI = ("в плюс", "в минус", "не определён", "0")
SSYLKA = re.compile(r"\.(csv|md|py)\b")


def razdel(lines, nachalo, konec):
    """Строки между заголовками «## П. nachalo.» и «## П. konec.»."""
    a = next(i for i, s in enumerate(lines) if s.startswith(f"## П. {nachalo}."))
    b = next((i for i, s in enumerate(lines) if s.startswith(f"## П. {konec}.")), len(lines))
    return lines[a + 1:b]


def g1(lines):
    nomera = "".join(m.group(1) for s in lines if (m := re.match(r"^## П\. (\d)\.", s)))
    return nomera


def g2(lines):
    plohie, n = [], 0
    for s in razdel(lines, 8, 9):
        yacheiki = [c.strip() for c in s.strip().strip("|").split("|")]
        if not re.match(r"^\|\s*\d+\s*\|", s) or len(yacheiki) < 5:
            continue
        n += 1
        znak, poryadok = yacheiki[2], yacheiki[3]
        ok_znak = any(znak.startswith(z) for z in ZNAKI)
        ok_por = "оценки нет" in poryadok or (
            re.search(r"\d", poryadok) and SSYLKA.search(s))
        if not (ok_znak and ok_por):
            plohie.append(yacheiki[0])
    return f"strok={n} bez_znaka_ili_poryadka={plohie}"


def g3(lines):
    plohie, n = [], 0
    for s in razdel(lines, 9, 10):
        m = re.match(r"^(\d+)\. ", s)
        if m:
            n += 1
            if "Умолчание:" not in s:
                plohie.append(m.group(1))
    return f"voprosov={n} bez_umolchaniya={plohie}"


def g4(lines):
    return "\n".join(f"{i}: {s}" for i, s in enumerate(lines, 1) if re.search(r"202[01]", s))


GATES = {"G1": g1, "G2": g2, "G3": g3, "G4": g4}


def main(argv):
    if len(argv) != 3 or argv[2] not in GATES:
        print(__doc__)
        return 2
    with open(argv[1], encoding="utf-8") as f:
        lines = f.read().split("\n")
    print(GATES[argv[2]](lines))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
