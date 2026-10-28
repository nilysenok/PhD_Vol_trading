"""R-б: подмена E.run_e1 (запись сделок) не утекает из процесса — иначе следующие тесты движка получают KeyError 'ev'."""
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "revers"))
import r1_b  # noqa: E402
from fastbt import a_strogaya as AS, etap1, etap1_eval as E  # noqa: E402

RA = r1_b.RA


@dataclass
class Vhod:        # заглушка входа движка (RA.replace требует dataclass)
    div: dict


def test_posle_importa_run_e1_ishodnyj():
    assert E.run_e1 is RA._run_e1 and E.run_e1 is not RA.run_e1_zapis


def test_posle_rascheta_run_e1_ishodnyj(monkeypatch):
    videno = []
    monkeypatch.setattr(r1_b, "ustanovit_period", lambda p: None)
    monkeypatch.setattr(etap1, "build_inputs", lambda sid, period: Vhod(div={}))
    monkeypatch.setattr(etap1, "base_point", lambda sid: None)
    monkeypatch.setattr(RA, "klass", lambda cls: lambda *a, **k: SimpleNamespace(sdelki={}))

    def schitat(ev, strogo):
        videno.append(E.run_e1)
        return {}
    monkeypatch.setattr(AS, "schitat", schitat)
    r1_b.schitat(("S1", "train4"))
    assert videno and all(f is RA.run_e1_zapis for f in videno)   # внутри расчёта запись включена
    assert E.run_e1 is RA._run_e1                                  # после — исходная функция


def test_run_e1_vosstanavlivaetsya_pri_oshibke():
    with pytest.raises(RuntimeError):
        with RA.zapis_sdelok():
            assert E.run_e1 is RA.run_e1_zapis
            raise RuntimeError
    assert E.run_e1 is RA._run_e1
