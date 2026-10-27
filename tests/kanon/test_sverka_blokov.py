import os
import re
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "scripts", "kanon"))
import sverka_blokov as sb  # noqa: E402

OK = """# Журнал
## Ж-001 · 10:00
текст
принят Ж-001, 10:01
## Ж-002 · 10:30
текст
квитанции не было — не дошёл (ретроспективно)
## Ж-003 · 11:00
текст
принят Ж-003, 11:02
"""


def pogreshnosti(text):
    """Набор проверок; пустой список = все прошли. Мутант обязан вернуть непустой."""
    f = []
    lines, code = sb.sverka(text)
    if lines[:3] != ["блоков всего: 3", "блоков без квитанции: 0",
                     "ретроспективно не дошедших: 1"] or code != 0:
        f.append("эталон")
    no_rcpt = OK.replace("принят Ж-003, 11:02", "")
    if sb.sverka(no_rcpt)[1] != 1 or "блоков без квитанции: 1" not in sb.sverka(no_rcpt)[0]:
        f.append("квитанция не засчитывается / отсутствие не ловится")
    wrong_no = OK.replace("принят Ж-003, 11:02", "принят Ж-002, 11:02")
    if sb.sverka(wrong_no)[1] != 1:
        f.append("квитанция чужого номера")
    gap = OK.replace("Ж-003 · 11:00", "Ж-004 · 11:00").replace("принят Ж-003", "принят Ж-004")
    if sb.sverka(gap)[1] != 1:
        f.append("пропуск номера не ловится")
    back = OK.replace("Ж-003 · 11:00", "Ж-003 · 09:00")
    if sb.sverka(back)[1] != 1:
        f.append("убывание времени не ловится")
    if sb.sverka("")[1] != 1:
        f.append("пустой журнал даёт 0")
    return f


class T(unittest.TestCase):
    def test_original_clean(self):
        self.assertEqual(pogreshnosti(OK), [])

    def test_mutant_receipt_not_counted(self):
        with mock.patch.object(sb, "RECEIPT_RE", re.compile(r"никогда_не_совпадёт")):
            self.assertTrue(pogreshnosti(OK))

    def test_mutant_any_receipt_number(self):
        # квитанция засчитывается без сверки номера: блок Ж-003 с чужой квитанцией проходит
        orig = sb.razobrat

        def mutant(text):
            return [dict(b, receipt=True) for b in orig(text)]
        with mock.patch.object(sb, "razobrat", mutant):
            self.assertTrue(pogreshnosti(OK))

    def test_mutant_gap_not_caught(self):
        with mock.patch.object(sb, "oshibki_poryadka", lambda blocks: []):
            self.assertTrue(pogreshnosti(OK))

    def test_empty_log_is_error(self):
        for text in ("", "# Журнал без блоков\nтекст\n"):
            lines, code = sb.sverka(text)
            self.assertEqual(code, 1)
            self.assertIn("журнал пуст или не найден", lines)

    def test_missing_file_is_error(self):
        self.assertEqual(sb.main(["x", "/nonexistent/resheniya.md"]), 1)

    def test_mutant_empty_gives_zero(self):
        # мутант: пустой журнал → 0; проверка пустоты должна это поймать
        orig = sb.sverka
        mutant = lambda t: (lambda r: (r[0], 0) if not sb.razobrat(t) else r)(orig(t))
        with mock.patch.object(sb, "sverka", mutant):
            self.assertTrue(pogreshnosti(OK))

    def test_tolerant_spaces(self):
        t = "##  Ж-001  ·  9:05\nтекст\nПринят  Ж-001 , 9:06\n"
        self.assertEqual(sb.sverka(t)[1], 0)


if __name__ == "__main__":
    unittest.main()
