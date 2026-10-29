"""Б4 (Ж-131 п. 5, Ж-135): сводка ledger для каталогов v5 (train4 2.1, v2 2.0, k3) на синтетике — данных и сети нет.
Каталоги строит сам серверный драйвер (fastbt.progon_server) на подмене оценщика, числа сверены с драйвером точек."""
import importlib.util
import json
import types
from pathlib import Path

import numpy as np
import pytest
from test_progon_server import Fake5, fake, kuby5, rz5, sreda  # noqa: F401  (fake — фикстура)

import fastbt.ledger_v2 as lm
import fastbt.progon_server as srv
import fastbt.progon_sobol as ps

_sp = importlib.util.spec_from_file_location("svodka_ledger", Path(__file__).resolve().parents[1] / "scripts/server/svodka_ledger.py")
SV = importlib.util.module_from_spec(_sp)
_sp.loader.exec_module(SV)
N = {"S1": 600, "S3": 300}
PROHODY = {"train4_s1": rz5(), "train4_s2": rz5(seed="seed2"), "v2_s1": rz5("v2"), "v2_s2": rz5("v2", "seed2"), "k3": rz5(k3=True)}
METKI = {"train4_s1": "train4 s1", "train4_s2": "train4 s2", "v2_s1": "v2 s1", "v2_s2": "v2 s2", "k3": "k3"}


@pytest.fixture
def kat(fake, monkeypatch, tmp_path):                                    # noqa: F811
    monkeypatch.setattr(ps, "fabrika", lambda sid: sreda(sid, Fake5()))
    for nm, rz in PROHODY.items():
        srv.progon(kuby5(["S1", "S3"], rz), tmp_path / nm, 2, n=dict(N), rz=rz)
    return tmp_path


def ozhidaemo(nm, sid):
    rz = PROHODY[nm]
    return len(ps.tochki(ps.load_kub(sid, rz), 0, N[sid], rz))


def test_g1_svodka_po_vsem_prohodam_chisla_sverheny(kat):
    rows = SV.svodka_vse([kat], 3)
    assert [r["strategiya"] for r in rows] == ["S1", "S3"] * 5 + ["ИТОГ"], rows
    assert rows[-1]["sverka"] == "ok", [(r["prohod"], r["sverka"]) for r in rows]
    por = [n for n in sorted(PROHODY) for _ in (1, 2)]              # папки — по алфавиту
    kod = set()
    for r, nm in zip(rows, por):
        sid = r["strategiya"]
        assert r["prohod"] == METKI[nm] and r["sverka"] == "ok" and r["chekpoint_i"] == N[sid] - 1
        assert r["tochek"] == r["ok"] == r["unikalnyh"] == ozhidaemo(nm, sid) and r["oshibok"] == 0
        assert r["nesovm"] == N[sid] - r["tochek"]
        assert float(r["s_na_tochku_p50"]) >= 0 and float(r["obyom_mb"]) > 0 and len(r["top_sharpe_okna"].split()) == 3
        kod.add((r["kod"], r["vhody"]))
    assert kod == {("kod-fake", "vhody-fake")} and rows[-1]["tochek"] == sum(r["tochek"] for r in rows[:-1])
    assert any(r["nesovm"] > 0 for r in rows[:-1])                      # canon_v5 выбрасывает точки — «несовм.» не нулевой
    assert SV.main([str(kat)]) == 0 and SV.SHAPKA in SV.tablica(rows) and "ledger 2.0" in SV.SHAPKA


@pytest.mark.parametrize("arg", ["out", "sk"])
def test_g1_odin_katalog_out_i_sk(kat, arg):
    p = kat / "v2_s1" if arg == "out" else kat / "v2_s1" / "S1"
    rows = SV.svodka_vse([p], 2)
    assert [r["prohod"] for r in rows[:-1]] == ["v2 s1"] * (2 if arg == "out" else 1) and SV.main([str(p)]) == 0
    assert SV.svodka(kat / "v2_s1", 2)[0]["tochek"] == ozhidaemo("v2_s1", "S1")


def test_g1_sharpe_v2_ryad_portfelya_ne_okno21(kat):
    r = SV.svodka_vse([kat / "v2_s1"], 1)[0]
    ch = lm.Chitatel(kat / "v2_s1", "S1")
    assert ch.v == lm.V
    m = ch.matrica(ch.rows())
    sh = m.mean(0) / m.std(0, ddof=1) * np.sqrt(252)
    j = int(np.argmax(sh))
    assert r["top_sharpe_okna"] == f"{ch.rows()[j]['sobol_i']}:{sh[j]:.3f}"
    r21 = SV.svodka_vse([kat / "train4_s1"], 1)[0]
    assert lm.Chitatel(kat / "train4_s1", "S1").v == lm.V21 and r21["top_sharpe_okna"] != "—"


def test_sharpe_ryada_izvestnyj_otvet():
    m = np.array([[0.01, 0.02, 0.0], [0.03, -0.02, 0.0], [0.02, 0.0, 0.0]])
    sh = SV.sharpe_ryada(m)
    assert sh[0] == pytest.approx(0.02 / 0.01 * np.sqrt(252)) and sh[1] == pytest.approx(0.0) and sh[2] == 0.0
    assert SV.sharpe_ryada(-m)[0] == pytest.approx(-sh[0])                 # знак не перепутан


def podmena_sharda(kat):
    f = sorted((kat / "v2_s1" / "S1").glob("p*/port/*.npy"))[0]
    f.write_bytes(f.read_bytes() + b"\0")


def proverit_g2(kat):
    podmena_sharda(kat)
    rows = SV.svodka_vse([kat], 1)
    assert all(r["sverka"] == "ok" for r in rows[:-1] if not (r["prohod"] == "v2 s1" and r["strategiya"] == "S1"))
    bad = next(r for r in rows if r["prohod"] == "v2 s1" and r["strategiya"] == "S1")
    assert bad["sverka"].startswith("ОТКАЗ") and rows[-1]["sverka"] == "есть расхождения"
    assert SV.main([str(kat)]) == 1


def test_g2_podmenyonnyj_shard_kod_1(kat):
    proverit_g2(kat)


def test_g2_udalyonnyj_shard_kod_1(kat):
    nom = sorted((kat / "train4_s1" / "S1").glob("p*/port/*.npy"))[0]
    for sub in ("port", "tik"):
        (nom.parent.parent / sub / nom.name).unlink()
    r = SV.svodka_vse([kat / "train4_s1"], 1)[0]
    assert r["sverka"].startswith("ОТКАЗ: нет файла") and SV.main([str(kat / "train4_s1")]) == 1


def test_g3_raznye_kod_v_dvuh_shardah_kod_1(kat):
    d = kat / "train4_s1" / "S1"
    assert len(list(d.glob("p*/port/*.npy"))) >= 2
    for f, k in ((sorted(d.glob("p*/rows.jsonl"))[0], 0), (sorted(d.glob("p*/rows.jsonl"))[-1], -1)):
        ls = f.read_text(encoding="utf-8").splitlines()
        ls[k] = json.dumps({**json.loads(ls[k]), "kod": f"kod-{k}"}, sort_keys=True)
        f.write_text("\n".join(ls) + "\n", encoding="utf-8")
    r = SV.svodka_vse([kat / "train4_s1"], 1)[0]
    assert r["sverka"].startswith("ОТКАЗ") and "kod" in r["sverka"] and SV.main([str(kat / "train4_s1")]) == 1


def test_chekpoint_ne_ravnyj_manifest_i_pustoj_vhod(kat, tmp_path):
    c = kat / "k3" / "S3" / "checkpoint.json"
    c.write_text(json.dumps({**json.loads(c.read_text()), "kod": "chuzhoj"}), encoding="utf-8")
    assert SV.svodka_vse([kat / "k3"], 1)[1]["sverka"].startswith("чекпоинт ≠ manifest") and SV.main([str(kat / "k3")]) == 1
    (tmp_path / "pusto").mkdir()
    assert SV.main([str(tmp_path / "pusto")]) == 1                       # нечего сверять — не «ok»


def test_g5_mutaciya_sverka_shardov_vyklyuchena_lovitsya(kat, monkeypatch):
    class Lyuboj(str):
        __ne__ = lambda s, o: False  # noqa: E731
    monkeypatch.setattr(lm, "hashlib", types.SimpleNamespace(sha256=lambda b: types.SimpleNamespace(hexdigest=Lyuboj)))
    with pytest.raises(AssertionError):                                  # мутант хэш не сверяет — подмена проходит, код 0
        proverit_g2(kat)


def test_g5_mutaciya_polnota_shardov_vyklyuchena_lovitsya(kat, monkeypatch):
    monkeypatch.setattr(SV, "polnota_shardov", lambda d: [])
    with pytest.raises(AssertionError):                                  # без проверки полноты удаление шарда не видно
        test_g2_udalyonnyj_shard_kod_1(kat)
