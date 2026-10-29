"""Ж-131 п. 3–5: серверная обёртка Соболя (fastbt.progon_server) и сводка ledger на синтетике (данных нет).
«Побайтно» = канон каталога: строки по sobol_i без t_sek (время) и excess (адрес шарда — дело расписания), port/tik — tobytes()."""
import importlib.util
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pytest

import fastbt.ledger_v2 as lm
import fastbt.progon_server as srv
import fastbt.progon_sobol as ps

T, KUBY = 8, Path(ps.KUB).parent
_sp = importlib.util.spec_from_file_location("svodka_ledger", Path(__file__).resolve().parents[1] / "scripts/server/svodka_ledger.py")
SV = importlib.util.module_from_spec(_sp)
_sp.loader.exec_module(SV)


class Sboy(BaseException):          # не Exception: драйвер не превращает его в строку oshibka — кусок падает
    pass


class Fake:
    plohie: set = set()

    def evaluate_v2(self, pt, period):
        if json.dumps(pt, sort_keys=True) in self.plohie:
            raise Sboy("обрыв")
        rng = np.random.default_rng(int(lm.klyuch_sha(lm.klyuch("S1", pt, period, 0.0004, None), "a", "b")[:8], 16))
        tik = rng.normal(0, 0.01, (17, T)).astype(np.float32).astype(np.float64)
        return {"by_year": {str(y): 0.5 for y in range(2016, 2020)}, "T": T, "mediana_kv": 0.1}, np.nanmean(tik, axis=0), tik


def sreda(sid, ev=None, kod="kod-fake"):
    daty = np.arange(np.datetime64("2016-01-04"), np.datetime64("2016-01-04") + T)
    return ps.Sreda(ev or Fake(), kod, "vhody-fake", daty, [f"T{i:02d}" for i in range(17)])


@pytest.fixture
def fake(monkeypatch):
    monkeypatch.setattr(ps, "fabrika", sreda)
    monkeypatch.setattr(ps, "_W", {})
    monkeypatch.setattr(srv, "CHEKPOINT", 100)
    monkeypatch.setattr(shutil, "disk_usage", lambda p: shutil._ntuple_diskusage(10**13, 0, 10**13))   # IG-6 не от диска Мака


def kuby(sids, d=KUBY, seed=None):
    return {s: (d / f"kub_{s}.yaml", srv.zagruzit_kub(d / f"kub_{s}.yaml", "train4", seed)) for s in sids}


def kanon(root, sid):
    ch = lm.Chitatel(root, sid)
    rows = sorted(ch.rows(None), key=lambda r: r["sobol_i"])
    ok = [r for r in rows if r["status"] == "ok"]
    mf = {k: v for k, v in json.loads((Path(root) / sid / "manifest.json").read_text(encoding="utf-8")).items() if k != "sozdan"}
    sig = [json.dumps({k: v for k, v in r.items() if k not in ("t_sek", "excess")}, sort_keys=True) for r in rows]
    return sig, ch.summy(ok).tobytes(), ch.summy(ok, "tik").tobytes(), mf


syrye = lambda root, sid: [json.loads(x) for p in sorted((Path(root) / sid).glob("p[0-9]*/rows.jsonl"))  # noqa: E731
                            for x in p.read_text().splitlines()]
chp = lambda root, sid: json.loads((Path(root) / sid / srv.CHP).read_text(encoding="utf-8"))  # noqa: E731


class FakePoz(Fake):
    poz = None          # как EvaluatorV2 после Л7b: (|pos| порт. [T], |pos| тик. [17, T], сделок порт. [T], тик. [17, T])

    def evaluate_v2(self, pt, period):
        r, p, t = super().evaluate_v2(pt, period)
        self.poz = (np.abs(np.sign(p)), np.abs(np.sign(t)), (p > 0).astype(float), (t > 0).astype(float))
        return r, p, t


@pytest.mark.parametrize("rezhim,ev", [("obshchie", Fake), ("prezhniy", Fake), ("obshchie", FakePoz), ("prezhniy", FakePoz)])
def test_ravno_progon_sobol(fake, monkeypatch, tmp_path, rezhim, ev):
    monkeypatch.setattr(ps, "fabrika", lambda sid: sreda(sid, ev()))
    for s in ("S1", "S2"):          # гонка manifest.json в ledger_v2.Pisatel (см. test_manifest_iz_roditelya_do_fork) — обход
        lm.Pisatel21(tmp_path / "a", s, 0, ps.manifest(s, sreda(s), ps.load_kub(s), False), daty=sreda(s).daty, poz=ev is FakePoz)
    ps.schet(["S1", "S2"], 0, 250, "sobol1", tmp_path / "a", 2, False, False, lm.V21)
    it = srv.progon(kuby(["S1", "S2"]), tmp_path / "b", 2, n=250, rezhim=rezhim)
    for s in ("S1", "S2"):
        assert kanon(tmp_path / "a", s) == kanon(tmp_path / "b", s) and kanon(tmp_path / "b", s)[3]["poz_konec"] == (ev is FakePoz)
        assert it[s]["n"] == 250 and it[s]["sboy"] is None and chp(tmp_path / "b", s)["posledniy_i"] == 249
        assert {(x["kod"], x["vhody"]) for x in syrye(tmp_path / "b", s)} == {("kod-fake", "vhody-fake")}   # хэши в каждой строке


def test_obryv_posle_chekpointa_i_prodolzhenie(fake, monkeypatch, tmp_path):
    srv.progon(kuby(["S1"]), tmp_path / "celyj", 2, n=300)
    c = kuby(["S1"])["S1"][1]
    monkeypatch.setattr(Fake, "plohie", {json.dumps(p, sort_keys=True) for _, p in ps.tochki(c, 150, 200)})
    it = srv.progon(kuby(["S1"]), tmp_path / "b", 2, n=300)
    assert "Sboy" in it["S1"]["sboy"] and chp(tmp_path / "b", "S1")["posledniy_i"] == 99      # блок 100…199 не закрыт
    with pytest.raises(SystemExit, match="prodolzhit"):
        srv.progon(kuby(["S1"]), tmp_path / "b", 2, n=300)                                    # без флага — отказ
    monkeypatch.setattr(Fake, "plohie", set())
    it = srv.progon(kuby(["S1"]), tmp_path / "b", 2, n=300, prodolzhit=True)
    assert it["S1"]["sboy"] is None and chp(tmp_path / "b", "S1")["posledniy_i"] == 299 and it["S1"]["n"] <= 200
    assert kanon(tmp_path / "celyj", "S1") == kanon(tmp_path / "b", "S1")
    assert len(syrye(tmp_path / "b", "S1")) == 300                                              # без дублей


def test_mutaciya_chekpoint_bez_khesha_kuba_lovitsya(fake, monkeypatch, tmp_path):
    kd = Path(shutil.copytree(KUBY, tmp_path / "kub"))
    srv.progon(kuby(["S1"], kd), tmp_path / "o", 2, n=100)
    (kd / "kub_S1.yaml").write_text((KUBY / "kub_S1.yaml").read_text(encoding="utf-8") + "\n# другой куб\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="kub_sha"):
        srv.progon(kuby(["S1"], kd), tmp_path / "o", 2, n=200, prodolzhit=True)
    monkeypatch.setattr(srv, "POLYA", tuple(k for k in srv.POLYA if k != "kub_sha"))           # мутация
    it = srv.progon(kuby(["S1"], kd), tmp_path / "o", 2, n=200, prodolzhit=True)
    assert it["S1"]["sboy"] is None and it["S1"]["posledniy_i"] == 199                         # мутант продолжил молча


def test_otkaz_chuzhoj_seed_i_kod(fake, monkeypatch, tmp_path):
    srv.progon(kuby(["S1"]), tmp_path, 2, n=100)
    with pytest.raises(SystemExit, match="sobol_seed"):
        srv.progon(kuby(["S1"], seed=7), tmp_path, 2, n=200, prodolzhit=True)
    monkeypatch.setattr(ps, "fabrika", lambda sid: sreda(sid, kod="kod-drugoj"))
    with pytest.raises(SystemExit, match="kod"):
        srv.progon(kuby(["S1"]), tmp_path, 2, n=200, prodolzhit=True)
    assert chp(tmp_path, "S1")["posledniy_i"] == 99


def test_sboy_s1_ne_menyaet_katalog_s2(fake, monkeypatch, tmp_path):
    srv.progon(kuby(["S2"]), tmp_path / "a", 2, n=200)

    class Vsegda(Fake):
        def evaluate_v2(self, pt, period):
            raise Sboy("сбой S1")
    monkeypatch.setattr(ps, "fabrika", lambda sid: sreda(sid, Vsegda() if sid == "S1" else None))
    it = srv.progon(kuby(["S1", "S2"]), tmp_path / "b", 2, n=200)
    assert "Sboy" in it["S1"]["sboy"] and it["S1"]["posledniy_i"] is None and it["S2"]["sboy"] is None
    assert kanon(tmp_path / "a", "S2") == kanon(tmp_path / "b", "S2") and chp(tmp_path / "b", "S2")["posledniy_i"] == 199

    monkeypatch.setattr(ps, "fabrika", lambda sid: sreda(sid) if sid != "S1" else {}["нет баров"])   # сбой входов S1 в родителе
    it = srv.progon(kuby(["S1", "S2"]), tmp_path / "c", 2, n=200)
    assert "нет баров" in it["S1"]["sboy"] and kanon(tmp_path / "a", "S2") == kanon(tmp_path / "c", "S2")


def test_manifest_iz_roditelya_do_fork(fake, monkeypatch, tmp_path):
    """Гонка: Pisatel пишет manifest.json без tmp; два ребёнка на первых кусках одной стратегии читали полупустой файл."""
    orig = ps.schitat_kusok

    def proverka(zad):
        if not (Path(ps._W["root"]) / zad[0] / "manifest.json").exists():
            raise Sboy("manifest не из родителя")
        return orig(zad)
    monkeypatch.setattr(ps, "schitat_kusok", proverka)
    it = srv.progon(kuby(["S1", "S2"]), tmp_path, 2, n=200)
    assert it["S1"]["sboy"] is None and it["S2"]["sboy"] is None and it["S2"]["posledniy_i"] == 199


def _fabrika_s_metkoj(mk: Path, obj: dict):
    def fab(sid):
        (mk / f"{os.getpid()}_{sid}").touch()              # метка видна из любого процесса
        obj[sid] = sreda(sid)
        return obj[sid]
    return fab


def test_fork_obshchie_bez_kopii(fake, monkeypatch, tmp_path):
    """Метод: оценщик (все входы) строится только в родителе — меток fabrika с pid детей нет; дети считают на объекте
    с тем же id и тем же адресом буфера дат, что у родителя (fork, без копии). Контраст — режим prezhniy."""
    obj, mk = {}, tmp_path / "metki"
    mk.mkdir()
    monkeypatch.setattr(ps, "fabrika", _fabrika_s_metkoj(mk, obj))
    it = srv.progon(kuby(["S1", "S2"]), tmp_path / "o", 2, n=400, rezhim="obshchie")
    assert {p.name for p in mk.iterdir()} == {f"{os.getpid()}_S1", f"{os.getpid()}_S2"}
    for s in ("S1", "S2"):
        deti = it[s]["deti"]
        assert deti and all(pid != os.getpid() for pid, _, _ in deti)
        assert {(i, a) for _, i, a in deti} == {(id(obj[s].evr), obj[s].daty.__array_interface__["data"][0])}
    (mk2 := tmp_path / "metki2").mkdir()
    monkeypatch.setattr(ps, "fabrika", _fabrika_s_metkoj(mk2, {}))
    srv.progon(kuby(["S1", "S2"]), tmp_path / "p", 2, n=400, rezhim="prezhniy")
    assert {p.name.split("_")[0] for p in mk2.iterdir()} - {str(os.getpid())}                   # дети строили свои


def test_svodka_tablica_i_porcha_sharda(fake, tmp_path):
    srv.progon(kuby(["S1", "S2"]), tmp_path, 2, n=200)
    rows = SV.svodka(tmp_path, 3)
    assert [r["strategiya"] for r in rows] == ["S1", "S2", "ИТОГ"] and rows[-1]["tochek"] == 400
    assert all(r["sverka"] == "ok" for r in rows) and rows[0]["chekpoint_i"] == 199 and float(rows[-1]["obyom_mb"]) > 0
    assert len(rows[0]["top_sharpe_okna"].split()) == 3 and rows[0]["tochek"] == rows[0]["unikalnyh"] == 200
    assert SV.main([str(tmp_path)]) == 0 and "| ИТОГ |" in SV.tablica(rows)
    (f := sorted((tmp_path / "S2").glob("p*/port/*.npy"))[0]).write_bytes(f.read_bytes() + b"\0")
    rows = SV.svodka(tmp_path, 3)
    assert rows[1]["sverka"].startswith("ОТКАЗ") and rows[-1]["sverka"] == "есть расхождения" and rows[0]["sverka"] == "ok"
    assert SV.main([str(tmp_path)]) == 1


def test_v2_otkaz_i_procs_po_umolchaniyu(monkeypatch, tmp_path):
    with pytest.raises(SystemExit, match="v2"):
        srv.zagruzit_kub(KUBY / "kub_S1.yaml", "v2")
    with pytest.raises(SystemExit, match="v2"):
        srv.main(["--kub", str(KUBY / "kub_S1.yaml"), "--period", "v2", "--out", str(tmp_path), "--procs", "2"])
    monkeypatch.setattr(os, "cpu_count", lambda: 128)
    assert srv.procs_po_umolchaniyu() == 124
