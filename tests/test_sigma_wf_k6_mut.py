"""К-6, тестировщик #811: мутации, не пойманные tests/test_sigma_wf_k6.py.
Мутации: leg_raw хвоста шага 8 взят не из К-5 (×1,05); дыра старого ряда не в шаге 8 проходит молча; _sha читает только первый блок файла.
python3 -m pytest tests/test_sigma_wf_k6_mut.py -q"""
import hashlib
import importlib.util
from pathlib import Path

import numpy as np
import pytest

_s = importlib.util.spec_from_file_location("wk6", Path(__file__).resolve().parents[1] / "scripts" / "dannye" / "wf_sigma_910_sborka.py")
W = importlib.util.module_from_spec(_s)
_s.loader.exec_module(W)
_t = importlib.util.spec_from_file_location("tk6", Path(__file__).resolve().parent / "test_sigma_wf_k6.py")
_T = importlib.util.module_from_spec(_t)
_t.loader.exec_module(_T)
kit = _T.kit  # фикстура синтетики (хвост шага 8 удалён из прежнего ряда)


def test_hvost_shaga_8_leg_raw_iz_k5(kit):
    old, new = kit
    s = W.sklejka(old, new)
    t8 = s[(s.part == "test") & (s.step == 8)]
    ot = old[(old.part == "test") & (old.step == 8)]
    hv = t8[~t8.date.isin(ot.date)].merge(new[(new.part == "test") & (new.step == 8)], on=["date", "ticker"], suffixes=("", "_k5"))
    assert len(hv) == 2 and np.array_equal(hv.leg_raw, hv.leg_raw_k5)  # ровно хвост: значения К-5, не прежнего и не правленые


@pytest.mark.parametrize("step", [3, 5, 7])
def test_dyra_starogo_ryada_ne_v_shage_8_otvergaetsya(kit, step):
    old, new = kit
    t = old[(old.part == "test") & (old.step == step)]
    dyra = old.drop(t[t.date == t.date.max()].index)  # строка(и) теста шага < 8 потеряны: К-5 их не подставляет молча
    with pytest.raises(AssertionError):
        W.sklejka(dyra, new)


def test_sha_ves_fail(tmp_path):
    p = tmp_path / "f.bin"
    a = b"\x00" * ((1 << 20) + 5)
    p.write_bytes(a + b"x")
    q = tmp_path / "g.bin"
    q.write_bytes(a + b"y")  # различие после первого мегабайта
    assert W._sha(p) == hashlib.sha256(a + b"x").hexdigest()
    assert W._sha(p) != W._sha(q)
