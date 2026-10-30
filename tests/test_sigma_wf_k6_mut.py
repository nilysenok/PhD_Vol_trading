"""К-6, тестировщик #811: мутации, не пойманные tests/test_sigma_wf_k6.py.
Мутации: хвост шага 8 из К-5, а не из v1; дыра старого ряда не в шаге 8 проходит молча; _sha читает только первый блок;
шапка (shapka) берёт sha не того файла / не обновляется после смены файла.
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
kit, _v1 = _T.kit, _T._v1  # фикстура синтетики (хвост шага 8 удалён из прежнего ряда); v1 с хвостом


def test_hvost_shaga_8_znacheniya_iz_v1_a_ne_k5(kit):
    old, new = kit
    s = W.sklejka(old, new, _v1(old))
    t8 = s[(s.part == "test") & (s.step == 8)]
    ot = old[(old.part == "test") & (old.step == 8)]
    hv = t8[~t8.date.isin(ot.date)].merge(_v1(old), on=["date", "ticker"])
    assert len(hv) == 2
    assert np.array_equal(hv.rv_pred, hv.sigma_pred_h1.astype("float64") ** 2)  # rv_pred хвоста = σ̂²(v1) побитно
    k5 = new[(new.part == "test") & (new.step == 8)].merge(hv[["date", "ticker"]], on=["date", "ticker"])
    assert len(k5) == 2 and not np.isclose(k5.leg_raw.values, hv.leg_raw.values).any()  # значения К-5 в хвост не попали
    assert np.allclose(hv.leg_raw * hv.F_sm, hv.rv_pred, rtol=1e-12)


@pytest.mark.parametrize("step", [3, 5, 7])
def test_dyra_starogo_ryada_ne_v_shage_8_otvergaetsya(kit, step):
    old, new = kit
    t = old[(old.part == "test") & (old.step == step)]
    dyra = old.drop(t[t.date == t.date.max()].index)  # строка(и) теста шага < 8 потеряны: не заполняются молча
    with pytest.raises(AssertionError):
        W.sklejka(dyra, new, _v1(old))


def test_sha_ves_fail(tmp_path):
    p = tmp_path / "f.bin"
    a = b"\x00" * ((1 << 20) + 5)
    p.write_bytes(a + b"x")
    q = tmp_path / "g.bin"
    q.write_bytes(a + b"y")  # различие после первого мегабайта
    assert W._sha(p) == hashlib.sha256(a + b"x").hexdigest()
    assert W._sha(p) != W._sha(q)


def test_shapka_sha_svoego_faila_i_obnovlyaetsya(tmp_path):
    f = {n: tmp_path / n for n in ("vh1", "vh2", "vy1")}
    for i, p in enumerate(f.values()):
        p.write_bytes(b"data%d" % i)
    sh = W.shapka({"vh1": f["vh1"], "vh2": f["vh2"]}, {"vy1": f["vy1"]})
    assert set(sh) == {"vh1", "vh2", "vy1"}
    for k, p in f.items():  # sha каждого ключа - от своего файла, не соседнего
        assert sh[k] == hashlib.sha256(p.read_bytes()).hexdigest()
    assert len(set(sh.values())) == 3
    f["vy1"].write_bytes(b"izmeneno")  # файл выхода пересчитан -> шапка обязана смениться
    sh2 = W.shapka({"vh1": f["vh1"], "vh2": f["vh2"]}, {"vy1": f["vy1"]})
    assert sh2["vy1"] != sh["vy1"] and sh2["vh1"] == sh["vh1"]
