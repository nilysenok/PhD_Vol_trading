"""Тестировщик #836: попытки сломать ChitatelPort и мутации. Синтетика через Pisatel22, сети нет."""
import json
import shutil

import numpy as np
import pytest

from fastbt import diag_v6_chtenie as C
from fastbt import ledger_v2 as L
from tests.test_diag_v6_chtenie import _legkiy, _zapisat


def _port(root, nom, p="p00"):
    return root / "S1" / p / "port" / f"{nom:05d}.npy"


def test_chuzhoj_shard_pod_chuzhoj_sha(tmp_path):
    _zapisat(tmp_path, range(6))
    _legkiy(tmp_path)
    _port(tmp_path, 1).write_bytes(_port(tmp_path, 2).read_bytes())     # содержимое шарда 2 под номером 1
    with pytest.raises(ValueError, match="sha256"):
        C.ChitatelPort(tmp_path, "S1")


def test_propusk_nomera_0_2(tmp_path, monkeypatch):
    _zapisat(tmp_path, range(6))
    _legkiy(tmp_path)
    p = tmp_path / "S1" / L.SUMS
    p.write_text("".join(x + "\n" for x in p.read_text().splitlines() if "00001" not in x))
    with pytest.raises(ValueError, match="подряд"):
        C.ChitatelPort(tmp_path, "S1")
    # мутация: проверка «подряд» снята -> строки шарда 2 читались бы с позиции шарда 1 (молча чужие строки)
    orig = C._zakrytye
    monkeypatch.setattr(C, "_zakrytye", lambda sm, pd: [(k, n) for k, n in _bez_proverki(sm, pd)])
    r = C.ChitatelPort(tmp_path, "S1")
    assert [x["sharpe"] for x in r.rows()] == [0.0, 1.0, 2.0, 3.0]       # прочитаны строки 2,3 вместо 4,5: мутация видна
    monkeypatch.setattr(C, "_zakrytye", orig)
    with pytest.raises(ValueError, match="подряд"):
        C.ChitatelPort(tmp_path, "S1")


def _bez_proverki(sm, pd):
    out = []
    for f in sorted((pd / "port").glob("*.npy")):
        if sm.get(f"{pd.name}/port/{f.name}") is not None:
            out.append((int(f.stem), int(np.load(f, mmap_mode="r").shape[0])))
    return out


def test_pustoj_ledger_bez_potokov(tmp_path):
    _zapisat(tmp_path, range(2))
    shutil.rmtree(tmp_path / "S1" / "p00")
    r = C.ChitatelPort(tmp_path, "S1")
    assert r.rows(None) == []


def test_oborvannaya_posledniya_stroka_hvosta(tmp_path):
    _zapisat(tmp_path, range(6))
    _legkiy(tmp_path)
    p = tmp_path / "S1" / L.SUMS
    p.write_text("".join(x + "\n" for x in p.read_text().splitlines() if int(x.split("/")[-1][:5]) < 2))
    rp = tmp_path / "S1" / "p00" / L.ROWS
    rp.write_text(rp.read_text()[:-15])                                  # обрыв записи в хвосте (шард 2 не закрыт)
    assert [x["sharpe"] for x in C.ChitatelPort(tmp_path, "S1").rows()] == [0.0, 1.0, 2.0, 3.0]


def test_oborvannaya_stroka_v_zakrytom_shardе_otkaz(tmp_path):
    _zapisat(tmp_path, range(4))
    _legkiy(tmp_path)
    rp = tmp_path / "S1" / "p00" / L.ROWS
    rp.write_text(rp.read_text()[:-15])
    with pytest.raises(ValueError):
        C.ChitatelPort(tmp_path, "S1")


def test_dubl_v_raznyh_shardah_odin_potok_odinakovye_summy(tmp_path):
    _zapisat(tmp_path, [0, 1, 2, 3], potok=0)
    _zapisat(tmp_path, [4, 3, 0, 5], potok=1)                            # 3 и 0 повторены с теми же seed
    _legkiy(tmp_path)
    shutil.rmtree(tmp_path / "S1" / "p01" / "tik")
    r = C.ChitatelPort(tmp_path, "S1")
    assert sorted(x["sharpe"] for x in r.rows()) == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    assert r.summy().shape[0] == 6


def test_chastichnyj_tik_ne_mesaet(tmp_path):
    _zapisat(tmp_path, range(6))
    shutil.rmtree(tmp_path / "S1" / "p00" / "tik")
    (tmp_path / "S1" / "p00" / "tik").mkdir()
    shtat = None
    r = C.ChitatelPort(tmp_path, "S1")
    assert len(r.rows(None)) == 6 and shtat is None


def test_gonka_shard_zapisan_a_sha_net(tmp_path):
    _zapisat(tmp_path, range(4))
    _legkiy(tmp_path)
    p = tmp_path / "S1" / L.SUMS
    p.write_text("".join(x + "\n" for x in p.read_text().splitlines() if "00001" not in x))   # .npy есть, SHA нет
    r = C.ChitatelPort(tmp_path, "S1")
    assert [x["sharpe"] for x in r.rows()] == [0.0, 1.0]
    (tmp_path / "S1" / "p00" / "port" / "00002.npy.part").write_bytes(b"x")                   # недописанный .part
    assert len(C.ChitatelPort(tmp_path, "S1").rows()) == 2


def _dubl_raznye(tmp_path):
    _zapisat(tmp_path, [0, 1], potok=0)
    from tests.test_diag_v6_chtenie import MAN, T, _row
    w = L.Pisatel22(tmp_path, "S1", 1, MAN, shard=2, daty=np.arange("2016-01-01", "2020-01-01", dtype="datetime64[D]"))
    rng = np.random.default_rng(99)
    w.add(_row(0), rng.normal(0, 1e-3, T), rng.normal(0, 1e-3, (17, T)))
    w.add(_row(7), rng.normal(0, 1e-3, T), rng.normal(0, 1e-3, (17, T)))
    w.close()
    _legkiy(tmp_path)
    shutil.rmtree(tmp_path / "S1" / "p01" / "tik")


def test_mutaciya_dubl_poslednee_pobezhdaet_lovitsya(tmp_path, monkeypatch):
    _dubl_raznye(tmp_path)
    with pytest.raises(ValueError, match="дубль"):
        C.ChitatelPort(tmp_path, "S1")

    def molcha(self, r, loc):
        key = (r["klyuch_n"], r["rol"])
        if key in self._seen:
            self._r = [(x, l) for x, l in self._r if (x["klyuch_n"], x["rol"]) != key]
        self._seen[key] = loc
        self._r.append((r, loc))
    monkeypatch.setattr(C.ChitatelPort, "_dobavit", molcha)
    C.ChitatelPort(tmp_path, "S1")                                       # мутация молчит -> тест выше упал бы


def test_mutaciya_proverit_stroku_22_ne_vyzyvaetsya_lovitsya(tmp_path, monkeypatch):
    _zapisat(tmp_path, range(2))
    _legkiy(tmp_path)
    rp = tmp_path / "S1" / "p00" / L.ROWS
    ls = rp.read_text().splitlines()
    d = json.loads(ls[0])
    d["kod"] = ""
    rp.write_text(json.dumps(d) + "\n" + ls[1] + "\n")
    with pytest.raises(ValueError):
        C.ChitatelPort(tmp_path, "S1")
    monkeypatch.setattr(C, "proverit_stroku_22", lambda r, s: None)
    C.ChitatelPort(tmp_path, "S1")                                       # мутация: подмена проходит молча


def test_mutaciya_shardy_ne_podryad_snyata_lovitsya(tmp_path, monkeypatch):
    test_propusk_nomera_0_2(tmp_path, monkeypatch)
