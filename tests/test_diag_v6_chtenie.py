"""Д-1 диагностики v6 train4: читатель ledger 2.2 без tik/. Синтетика через штатный Pisatel22 (сети нет)."""
import json
import shutil

import numpy as np
import pytest

from fastbt import diag_v6_chtenie as C
from fastbt import ledger_v2 as L

DATY = np.arange("2016-01-01", "2020-01-01", dtype="datetime64[D]")
T = len(DATY)
MAN = {"T": T, "kod": "k", "vhody": "v", "setka": "s", "period": "train4"}


def _row(i, rol="sobol"):
    kl = L.klyuch("S1", {"a": i}, "train4", 0.0004, None)
    return {"v": "2.2", "strategiya": "S1", "klyuch": kl, "klyuch_n": L.klyuch_n("S1", kl), "klyuch_sha": L.klyuch_sha(kl, "k", "v"),
            "kod": "k", "vhody": "v", "setka": "s", "rol": rol, "status": "ok", "sharpe": float(i)}


def _zapisat(root, idx, shard=2, potok=0, sid="S1", roli=None):
    w = L.Pisatel22(root, sid, potok, MAN, shard=shard, daty=DATY)
    for i in idx:
        rng = np.random.default_rng(i)
        w.add(_row(i, (roli or {}).get(i, "sobol")), rng.normal(0, 1e-3, T), rng.normal(0, 1e-3, (17, T)))
    w.close()
    return w


def _bez_sha_posl_shardov(root, potok="p00", ostavit=2):
    """Как забор в разгар счёта: у шардов с номером ≥ ostavit нет строк SHA (port и tik)."""
    p = root / "S1" / L.SUMS
    keep = [ln for ln in p.read_text().splitlines() if int(ln.split("/")[-1][:5]) < ostavit]
    p.write_text("".join(ln + "\n" for ln in keep))


def _legkiy(root):
    shutil.rmtree(root / "S1" / "p00" / "tik")
    (root / "S1" / L.SUMS).write_text("".join(ln + "\n" for ln in (root / "S1" / L.SUMS).read_text().splitlines() if "/port/" in ln))


def _g11(root):
    """Свойство G1.1: 3 шарда, третий без SHA → ровно строки двух шардов (0..3), хвост rows.jsonl не засчитан."""
    r = C.ChitatelPort(root, "S1")
    assert [x["sharpe"] for x in r.rows()] == [0.0, 1.0, 2.0, 3.0]
    assert r.summy().shape[:2] == (4, 16)


@pytest.fixture
def tri_shardov(tmp_path):
    _zapisat(tmp_path, range(6))
    _bez_sha_posl_shardov(tmp_path)
    shutil.rmtree(tmp_path / "S1" / "p00" / "tik")
    return tmp_path


def test_g11_tretij_shard_bez_sha(tri_shardov):
    assert len((tri_shardov / "S1" / "p00" / L.ROWS).read_text().splitlines()) == 6   # хвост в файле есть
    _g11(tri_shardov)


def test_g12_mutaciya_nezakrytyj_shard_zaschitan_lovitsya(tri_shardov, monkeypatch):
    def vse(sm, pd):
        return [(int(f.stem), int(np.load(f, mmap_mode="r").shape[0])) for f in sorted((pd / "port").glob("*.npy"))]
    monkeypatch.setattr(C, "_zakrytye", vse)
    with pytest.raises(AssertionError):
        _g11(tri_shardov)


def test_g13_sha_ne_sovpal_otkaz_i_mutaciya_lovitsya(tmp_path, monkeypatch):
    _zapisat(tmp_path, range(4))
    _legkiy(tmp_path)
    f = tmp_path / "S1" / "p00" / "port" / "00001.npy"
    b = bytearray(f.read_bytes())
    b[-1] ^= 1
    f.write_bytes(bytes(b))
    with pytest.raises(ValueError, match="sha256"):
        C.ChitatelPort(tmp_path, "S1")
    monkeypatch.setattr(C, "_sha_ok", lambda f, h: True)       # мутация: sha не сверяется
    C.ChitatelPort(tmp_path, "S1")                              # молча читает испорченный шард → тест выше упал бы
    monkeypatch.undo()
    with pytest.raises(ValueError, match="sha256"):
        C.ChitatelPort(tmp_path, "S1")


def test_g14_s_tik_ravno_shtatnomu_chitatelyu(tmp_path):
    _zapisat(tmp_path, range(5), roli={4: "kontrol"})
    shtat = L.Chitatel(tmp_path, "S1")
    mine = C.ChitatelPort(tmp_path, "S1")
    assert mine.rows(None) == shtat.rows(None) and len(mine.rows(None)) == 5
    assert np.array_equal(mine.summy(), shtat.summy(), equal_nan=True)
    _legkiy(tmp_path)
    leg = C.ChitatelPort(tmp_path, "S1")                        # без tik/ — то же
    assert leg.rows(None) == shtat.rows(None)
    assert np.array_equal(leg.summy(), shtat.summy(), equal_nan=True)
    with pytest.raises(Exception):
        assert len(L.Chitatel(tmp_path, "S1").rows()) == 5      # штатный без tik/ строк не видит


def test_g15_dubl_s_raznymi_znacheniyami_isklyuchenie(tmp_path):
    _zapisat(tmp_path, [0, 1], potok=0)
    w = L.Pisatel22(tmp_path, "S1", 1, MAN, shard=2, daty=DATY)
    rng = np.random.default_rng(99)                              # тот же klyuch_n (i = 0), другие суммы
    r = _row(0)
    r["klyuch_sha"] = "drugoj"
    w.add(r, rng.normal(0, 1e-3, T), rng.normal(0, 1e-3, (17, T)))
    w.add(_row(7), rng.normal(0, 1e-3, T), rng.normal(0, 1e-3, (17, T)))
    w.close()
    _legkiy(tmp_path)
    shutil.rmtree(tmp_path / "S1" / "p01" / "tik")
    (tmp_path / "S1" / L.SUMS).write_text("".join(ln + "\n" for ln in (tmp_path / "S1" / L.SUMS).read_text().splitlines() if "/port/" in ln))
    with pytest.raises(ValueError, match="дубль klyuch_n"):
        C.ChitatelPort(tmp_path, "S1")


def test_dubl_s_odinakovymi_znacheniyami_odin_raz(tmp_path):
    _zapisat(tmp_path, [0, 1], potok=0)
    _zapisat(tmp_path, [0, 2], potok=1)                          # i = 0 с тем же seed → те же суммы
    _legkiy(tmp_path)
    shutil.rmtree(tmp_path / "S1" / "p01" / "tik")
    r = C.ChitatelPort(tmp_path, "S1")
    assert sorted(x["sharpe"] for x in r.rows()) == [0.0, 1.0, 2.0]


def test_proverka_stroki_22_klyuch_n_podmenen(tmp_path):
    _zapisat(tmp_path, range(2))
    _legkiy(tmp_path)
    rp = tmp_path / "S1" / "p00" / L.ROWS
    ls = rp.read_text().splitlines()
    d = json.loads(ls[0])
    d["klyuch_n"] = "x"
    rp.write_text(json.dumps(d) + "\n" + ls[1] + "\n")
    with pytest.raises(ValueError, match="klyuch_n"):
        C.ChitatelPort(tmp_path, "S1")


def test_pustaya_lenta_i_shard_vne_ryada(tmp_path):
    _zapisat(tmp_path, range(2))
    _legkiy(tmp_path)
    (tmp_path / "S1" / L.SUMS).write_text("")                    # ни одного закрытого шарда
    assert C.ChitatelPort(tmp_path, "S1").rows(None) == []
    _zapisat(tmp_path / "b", range(6))
    _legkiy(tmp_path / "b")
    p = tmp_path / "b" / "S1" / L.SUMS
    p.write_text("".join(ln + "\n" for ln in p.read_text().splitlines() if "00001" not in ln))
    with pytest.raises(ValueError, match="подряд"):
        C.ChitatelPort(tmp_path / "b", "S1")
