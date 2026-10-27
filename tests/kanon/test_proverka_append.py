import os
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

HERE = os.path.dirname(__file__)
SCRIPT = os.path.join(HERE, "..", "..", "scripts", "kanon", "proverka_append.py")
sys.path.insert(0, os.path.dirname(SCRIPT))
import proverka_append as pa  # noqa: E402

OLD = "a\nb\nc\n"


def pogreshnosti():
    f = []
    if not pa.proverit(OLD, OLD + "d\n")[0]:
        f.append("дописывание отвергнуто")
    if not pa.proverit("", "x\n")[0]:
        f.append("создание отвергнуто")
    for name, new in [("правка старой строки", "a\nB\nc\n"),
                      ("удаление строки", "a\nc\n"),
                      ("удаление из конца", "a\nb\n"),
                      ("вставка в начало", "z\n" + OLD)]:
        ok, msg = pa.proverit(OLD, new)
        if ok:
            f.append(name + " проходит")
        elif "Ж-025 п. 1.1" not in msg:
            f.append(name + ": нет ссылки на правило")
    if pa.proverit("a\nb", "a\nbX\n")[0]:
        f.append("дописывание в хвост незавершённой строки проходит")
    return f


class T(unittest.TestCase):
    def test_original_clean(self):
        self.assertEqual(pogreshnosti(), [])

    def test_mutant_edit_passes(self):
        with mock.patch.object(pa, "proverit", lambda old, new: (True, "")):
            self.assertTrue(pogreshnosti())

    def test_mutant_only_length_check(self):
        # «новый длиннее старого» вместо сравнения начала — правка с добавлением проходит
        with mock.patch.object(pa, "proverit",
                               lambda o, n: (len(n) >= len(o), pa.MSG)):
            self.assertTrue(pogreshnosti())

    def test_exit_codes(self):
        with tempfile.TemporaryDirectory() as d:
            o, n = os.path.join(d, "o"), os.path.join(d, "n")
            open(o, "w", encoding="utf-8").write(OLD)
            open(n, "w", encoding="utf-8").write(OLD + "d\n")
            run = lambda: subprocess.run([sys.executable, SCRIPT, o, n],
                                         capture_output=True, text=True)
            self.assertEqual(run().returncode, 0)
            open(n, "w", encoding="utf-8").write("a\nX\nc\nd\n")
            r = run()
            self.assertEqual(r.returncode, 2)
            self.assertIn("Ж-025 п. 1.1", r.stderr)


if __name__ == "__main__":
    unittest.main()
