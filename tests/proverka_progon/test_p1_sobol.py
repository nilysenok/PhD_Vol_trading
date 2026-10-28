"""П-1, лист Л-6: куб, Соболь, продолжение прохода 2, резюм, диск, kod_slit. Красные до кода (Ж-081: без skip/xfail).
Допущение: драйвер имеет main(argv) и читает диск через shutil.disk_usage (задание Л-6, IG-6)."""
import copy
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from . import KOD, VHODY, manifest, pisat, row, ryady, stroki_jsonl

KUB = "fastbt/designs/progon/kub_S{k}.yaml"
NOVYE = {"ed_lim": ["sigma", "atr14", "diap", "pct"], "zhizn": ["den", "12", "36", "60"],
         "lestnica": ["net", "0.5", "1.0"], "gap": ["vhod", "propusk"],
         "t_vyhod": ["close", "1500", "1700", "open2"], "k_x": ["0.5", "1.0", "1.5", "net"],
         "storona": ["obe", "long", "short_shag1"]}


def kub(k):
    return yaml.safe_load(Path(KUB.format(k=k)).read_text(encoding="utf-8"))


def osi(node, out=None):
    """Все оси куба: узлы со списком 'setka', в порядке обхода."""
    out = {} if out is None else out
    if isinstance(node, dict):
        for kk, v in node.items():
            if isinstance(v, dict) and "setka" in v:
                out[kk] = [str(x) if isinstance(x, str) else x for x in v["setka"]]
            else:
                osi(v, out)
    return out


def _m():
    import fastbt.progon_sobol as m
    return m


@pytest.mark.parametrize("k", range(1, 7))
def test_kub_shapka_i_osi(k):
    c = kub(k)
    assert c["v"] == 2 and c["format"] == "progon-kub"
    assert c["sobol"] == {"seed": 20261028 + k, "scramble": True, "prohod1": 5000, "prohod2": 30000}
    assert c["komissiya"] == 0.0004 and c["fdr"] is None and c["period"] == "train4"
    o = osi(c)
    assert "sobytiya" not in o and "zaem" not in o
    for nm, vals in NOVYE.items():
        assert [str(x) for x in o[nm]] == vals, f"S{k}: ось {nm} или порядок значений"
    assert "filtr" in o and o["filtr"][0] == "net"


@pytest.mark.parametrize("k", [1, 5])
def test_tochki_vnutri_osej_i_kanon(k):
    import fastbt.ledger_v2 as lm
    c = kub(k)
    o = osi(c)
    pts = _m().tochki(c, 0, 300)
    assert [i for i, _ in pts] == list(range(300))
    for _, p in pts:
        for nm, v in p.items():
            assert str(v) in [str(x) for x in o[nm]], (nm, v)
        assert lm.canon_v2(f"S{k}", p) == p, "canon_v2 не применён (N завышен)"


def test_prohod2_prodolzhenie_prohoda1():
    m, c = _m(), kub(1)
    a = m.tochki(c, 0, 300)
    assert m.tochki(c, 0, 100) == a[:100]
    b = m.tochki(c, 100, 300)
    assert b == a[100:] and b[0][0] == 100
    assert m.tochki(c, 100, 300) != m.tochki(c, 0, 200)  # проход 2 не начинается с 0


def test_odin_seed_odni_tochki_drugoj_seed_drugie():
    m, c = _m(), kub(2)
    assert m.tochki(c, 0, 50) == m.tochki(copy.deepcopy(c), 0, 50)
    c2 = copy.deepcopy(c)
    c2["sobol"]["seed"] += 1
    assert m.tochki(c2, 0, 50) != m.tochki(c, 0, 50)


def test_seed_po_strategiyam_raznyj():
    m = _m()
    assert m.tochki(kub(1), 0, 20) != m.tochki(kub(2), 0, 20)


def test_rezyum_bez_dublej_i_poter(tmp_path):
    """Обрыв: закрыт один шард из двух; резюм считает только то, чего нет в done()."""
    import fastbt.ledger_v2 as lm
    m, c = _m(), kub(1)
    pts = m.tochki(c, 0, 12)
    pisat(lm, tmp_path, 0)
    w = lm.Pisatel(tmp_path, "S1", 0, manifest(), shard=4)
    for i, p in pts[:6]:  # 4 закрыты, 2 в буфере — «обрыв»
        w.add(row(lm, i, **p), *ryady(i))
    w2 = lm.Pisatel(tmp_path, "S1", 0, manifest(), shard=4)
    done = w2.done()
    ostalos = [(i, p) for i, p in pts if row(lm, i, **p)["klyuch_sha"] not in done]
    assert [i for i, _ in ostalos] == list(range(4, 12))
    for i, p in ostalos:
        w2.add(row(lm, i, **p), *ryady(i))
    w2.close()
    ch = lm.Chitatel(tmp_path, "S1")
    shas = [r["klyuch_sha"] for r in ch.rows()]
    assert len(shas) == len(set(shas)) == 12
    assert sorted(r["sobol_i"] for r in ch.rows()) == list(range(12))
    assert len(stroki_jsonl(tmp_path)) == 12


def test_cli_flagi():
    r = subprocess.run([sys.executable, "-m", "fastbt.progon_sobol", "--help"], capture_output=True, text=True)
    assert r.returncode == 0
    for fl in ("--prohod", "--tochek", "--processov", "--sverka", "--kod-slit"):
        assert fl in r.stdout, fl


def test_net_proverki_diska_otkaz(monkeypatch, tmp_path):
    """Нет места (IG-6): отказ до счёта, ничего не пишется."""
    import shutil
    m = _m()
    monkeypatch.setattr(shutil, "disk_usage", lambda p: shutil._ntuple_diskusage(10**12, 10**12 - 10**9, 10**9))
    with pytest.raises(BaseException) as e:
        m.main(["--prohod", "0", "--tochek", "1", "--processov", "1"])
    assert not (isinstance(e.value, SystemExit) and e.value.code in (0, None))


def test_kod_slit_flag_v_spravke():
    r = subprocess.run([sys.executable, "-m", "fastbt.progon_sobol", "--help"], capture_output=True, text=True)
    assert r.returncode == 0 and "--kod-slit" in r.stdout


def test_period_ne_train4_otkaz():
    m, c = _m(), kub(1)
    c["period"] = "validation"
    with pytest.raises((ValueError, SystemExit)):
        m.tochki(c, 0, 5)
