import os
import re
import sys
import unittest
from unittest import mock

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "..", "scripts", "kanon"))
import sverka_blokov as sb  # noqa: E402

# Копия 5 блоков реального журнала (2 без номера, Ж-024…Ж-026).
with open(os.path.join(HERE, "fixtures", "resheniya_fragment.md"), encoding="utf-8") as _f:
    OK = _f.read()


def sdelat(**kw):
    """Мутации текста эталона: ключ — имя, значение — (что, чем)."""
    return {n: OK.replace(a, b) for n, (a, b) in kw.items()}


MUTATED = sdelat(
    bez_kvitantsii=("Квитанция: принят Ж-026, 16:31", "Квитанция: "),
    chuzhoy_nomer=("принят Ж-026", "принят Ж-025"),
    propusk=("### Ж-025 ·", "### Ж-027 ·"),
    vremya_nazad=("27.10.2026, 16:20", "27.10.2026, 00:05"),
)
MUTATED["dve_kvitantsii"] = OK + "Квитанция: принят Ж-026, 16:32\n"


def pogreshnosti():
    """Набор проверок; пустой список = все прошли. Мутант обязан вернуть непустой."""
    f = []
    lines, code = sb.sverka(OK)
    if lines != ["блоков всего: 5 (с номером: 3, без номера: 2)",
                 "блоков без квитанции: 0",
                 "ретроспективно не дошедших: 2"] or code != 0:
        f.append("эталон: %r %r" % (lines, code))
    for name, text in MUTATED.items():
        if sb.sverka(text)[1] != 1:
            f.append(name + " не ловится")
    if "блоков без квитанции: 1" not in sb.sverka(MUTATED["bez_kvitantsii"])[0]:
        f.append("квитанция не засчитывается / отсутствие не считается")
    if sb.sverka("")[1] != 1:
        f.append("пустой журнал даёт 0")
    return f


class T(unittest.TestCase):
    def test_original_clean(self):
        self.assertEqual(pogreshnosti(), [])

    def test_mutant_receipt_not_counted(self):
        with mock.patch.object(sb, "RECEIPT_RE", re.compile(r"никогда_не_совпадёт")):
            self.assertTrue(pogreshnosti())

    def test_mutant_any_receipt(self):
        orig = sb.razobrat
        with mock.patch.object(sb, "razobrat",
                               lambda t: [dict(b, receipt=[b["n"]]) for b in orig(t)]):
            self.assertTrue(pogreshnosti())

    def test_mutant_gap_and_time_not_caught(self):
        with mock.patch.object(sb, "oshibki_poryadka", lambda blocks: []):
            self.assertTrue(pogreshnosti())

    def test_mutant_receipt_number_not_checked(self):
        with mock.patch.object(sb, "oshibki_kvitantsii", lambda blocks: []):
            self.assertTrue(pogreshnosti())

    def test_mutant_empty_gives_zero(self):
        orig = sb.sverka
        mutant = lambda t: (lambda r: (r[0], 0) if not sb.razobrat(t) else r)(orig(t))
        with mock.patch.object(sb, "sverka", mutant):
            self.assertTrue(pogreshnosti())

    def test_empty_log_is_error(self):
        for text in ("", "# Журнал без блоков\nтекст\n"):
            lines, code = sb.sverka(text)
            self.assertEqual(code, 1)
            self.assertIn("журнал пуст или не найден", lines)

    def test_missing_file_is_error(self):
        self.assertEqual(sb.main(["x", "/nonexistent/resheniya.md"]), 1)

    def test_levels_2_and_3_and_spaces(self):
        t = ("##  Ж-001  ·  27.10.2026, 9:05\nтекст\nКвитанция: Принят  Ж-001 , 9:06\n"
             "### Ж-002 · 27.10.2026, 9:10\nКвитанция: принят Ж-002, 9:11\n")
        self.assertEqual(sb.sverka(t)[1], 0)


if __name__ == "__main__":
    unittest.main()
