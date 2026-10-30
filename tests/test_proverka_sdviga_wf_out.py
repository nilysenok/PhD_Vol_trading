"""proverka_sdviga_wf: таблица wf пишется только в --out; без --out для wf отказ; 2021 — путь по умолчанию (лист «мелочи wf»).
Мутация: wf пишет в docs/tables (OUT) -> тест падает."""
import importlib.util
from pathlib import Path

import pandas as pd

from fastbt import period_v2

_P = Path(__file__).resolve().parent.parent / "scripts" / "proverka_sdviga_wf.py"
_s = importlib.util.spec_from_file_location("proverka_sdviga_wf_out", _P)
P = importlib.util.module_from_spec(_s)
_s.loader.exec_module(P)


def _podgotovka(monkeypatch, tmp_path):
    ok = pd.DataFrame([P.stroka("a", "t", "c", {-2: 0.1, -1: 0.3, 0: 0.9, 1: 0.3})])
    monkeypatch.setattr(P, "OUT", tmp_path / "umolchanie.csv")
    monkeypatch.setattr(P, "shapka", lambda *a: "# s\n")
    monkeypatch.setattr(P, "sobrat", lambda m, d: ok)
    monkeypatch.setattr(period_v2, "proverit_chtenie", lambda *a, **k: None)


def test_wf_bez_out_otkaz_i_nichego_ne_pishet(monkeypatch, tmp_path):
    _podgotovka(monkeypatch, tmp_path)
    assert P.main(["--do", "2026-02-28", "--main", str(tmp_path)]) != 0
    assert not (tmp_path / "umolchanie.csv").exists()


def test_wf_s_out_pishet_tuda_a_ne_v_umolchanie(monkeypatch, tmp_path):
    _podgotovka(monkeypatch, tmp_path)
    out = tmp_path / "ferma" / "t.csv"
    out.parent.mkdir()
    assert P.main(["--do", "2026-02-28", "--main", str(tmp_path), "--out", str(out)]) == 0
    assert out.read_text().startswith("# s\n") and not (tmp_path / "umolchanie.csv").exists()


def test_2021_bez_out_pishet_v_umolchanie(monkeypatch, tmp_path):
    _podgotovka(monkeypatch, tmp_path)
    assert P.main(["--do", "2021-12-31", "--main", str(tmp_path)]) == 0
    assert (tmp_path / "umolchanie.csv").read_text().startswith("# s\n")
