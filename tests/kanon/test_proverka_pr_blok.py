import os
import subprocess
import sys
import unittest
from unittest import mock

SCRIPT = os.path.join(os.path.dirname(__file__), "..", "..", "scripts", "kanon", "proverka_pr_blok.py")
sys.path.insert(0, os.path.dirname(SCRIPT))
import proverka_pr_blok as pb  # noqa: E402

GOOD = "Описание\n\nBlok: Ж-025\nPunkt: Ж-025#1.1, #1.4\n"


def pogreshnosti():
    f = []
    if pb.oshibki(GOOD):
        f.append("эталон отвергнут")
    if not pb.oshibki("Punkt: Ж-025#1.1\n"):
        f.append("PR без Blok: проходит")
    if not pb.oshibki("Blok: Ж-025\n"):
        f.append("PR без Punkt: проходит")
    if not pb.oshibki("Blok: Ж-25\nPunkt: Ж-025#1.1\n"):
        f.append("двузначный номер проходит")
    if not pb.oshibki("Blok: Ж-0256\nPunkt: Ж-025#1.1\n"):
        f.append("четырёхзначный номер проходит")
    if not pb.oshibki("Blok: Ж-025\nPunkt: Ж-025\n"):
        f.append("Punkt без раздела проходит")
    if not pb.oshibki(""):
        f.append("пустое описание проходит")
    return f


class T(unittest.TestCase):
    def test_original_clean(self):
        self.assertEqual(pogreshnosti(), [])

    def test_mutant_blok_not_required(self):
        with mock.patch.object(pb, "BLOK_RE", pb.PUNKT_RE):
            self.assertTrue(pogreshnosti())

    def test_mutant_no_check(self):
        with mock.patch.object(pb, "oshibki", lambda text: []):
            self.assertTrue(pogreshnosti())

    def test_exit_codes_stdin(self):
        run = lambda s: subprocess.run([sys.executable, SCRIPT], input=s,
                                       capture_output=True, text=True).returncode
        self.assertEqual(run(GOOD), 0)
        self.assertEqual(run("Punkt: Ж-025#1.1\n"), 2)


if __name__ == "__main__":
    unittest.main()
