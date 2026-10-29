"""Ж-131 п. 3–5: серверная обёртка Соболя (fastbt.progon_server) и сводка ledger на синтетике (данных нет).
«Побайтно» = канон каталога: строки по sobol_i без t_sek (время) и excess (адрес шарда — дело расписания), port/tik — tobytes()."""
import importlib.util
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pytest

import fastbt.canon_v5 as cv5
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
    if ch.v != lm.V21:          # ledger 2.0 (период v2): ряды портфеля и тиков, а не суммы сегментов
        return sig, ch.matrica(ok).tobytes(), ch.tik(ok).tobytes(), mf
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


# ---- Л10 v5: куб v5, период v2 (ledger 2.0), seed 2, k3, --n1 (Ж-131 п. 1, 3–4; Ж-130) ----
class Fake5(Fake):
    fdr = "t0d"            # fdr куба v5; оценщик обязан считать его же


def rz5(period="train4", seed="seed", k3=False):
    return ps.Rezhim("v5", period, seed, k3, "t0d")


def kuby5(sids, rz):
    return {s: (ps.kub_put(s, rz), srv.zagruzit_kub(ps.kub_put(s, rz), rz.period, rz=rz)) for s in sids}


def sravnit_s_drajverom(monkeypatch, tmp_path, rz, n, ist, rezhim="obshchie", sids=("S1", "S3")):
    """Драйвер (1 процесс) и обёртка на одних точках: канон каталога равен побайтно; чекпоинт — конец прохода."""
    monkeypatch.setattr(ps, "fabrika", lambda sid: sreda(sid, Fake5()))
    v = lm.V if rz.period == "v2" else lm.V21
    ps.schet(list(sids), 0, n, ist, tmp_path / "a", 1, False, False, v, rz)
    it = srv.progon(kuby5(sids, rz), tmp_path / "b", 2, n=n, rezhim=rezhim, rz=rz)
    for s in sids:
        assert kanon(tmp_path / "a", s) == kanon(tmp_path / "b", s) and kanon(tmp_path / "b", s)[0]
        assert it[s]["sboy"] is None and chp(tmp_path / "b", s)["posledniy_i"] == n[s] - 1
    return it


@pytest.mark.parametrize("rezhim", ["obshchie", "prezhniy"])
def test_v5_train4_seed1_ravno_drajveru(fake, monkeypatch, tmp_path, rezhim):           # G1
    it = sravnit_s_drajverom(monkeypatch, tmp_path, rz5(), {"S1": 256, "S3": 200}, "sobol1_seed1", rezhim)
    assert {x["istochnik"] for x in syrye(tmp_path / "b", "S1")} == {"sobol1_seed1"} and 0 < it["S1"]["n"] <= 256


def test_v5_period_v2_i_seed2_raznye_katalogi_period_i_seed_v_manifest(fake, monkeypatch, tmp_path):   # G2
    n = {"S1": 120, "S3": 120}
    sravnit_s_drajverom(monkeypatch, tmp_path / "v2", rz5("v2"), n, "sobol1_seed1")
    sravnit_s_drajverom(monkeypatch, tmp_path / "s2", rz5(seed="seed2"), n, "sobol1_seed2")
    mf = lambda d: json.loads((d / "b" / "S1" / "manifest.json").read_text())  # noqa: E731
    m2, ms2 = mf(tmp_path / "v2"), mf(tmp_path / "s2")
    assert (m2["period"], m2["okno_s"], m2["progrev_s"], m2["sobol_klyuch"]) == ("v2", "2015-01-01", "2014-08-26", "seed")
    assert (ms2["period"], ms2["sobol_klyuch"], ms2["sobol_seed"]) == ("train4", "seed2", 20261031) and m2["sobol_seed"] == 20261030
    assert chp(tmp_path / "v2" / "b", "S1")["period"] == "v2" and chp(tmp_path / "s2" / "b", "S1")["sobol_seed"] == 20261031
    assert {x["klyuch"]["period"] for x in syrye(tmp_path / "v2" / "b", "S1")} == {"v2"}
    k = lambda d: {x["klyuch_sha"] for x in syrye(d / "b", "S1")}  # noqa: E731
    assert k(tmp_path / "v2") and not k(tmp_path / "v2") & k(tmp_path / "s2")          # период и seed — разные ключи ledger


def test_v5_k3_ravno_drajveru(fake, monkeypatch, tmp_path):                              # G3
    it = sravnit_s_drajverom(monkeypatch, tmp_path, rz5(k3=True), {"S1": 150, "S3": 150}, "k3")
    assert {x["istochnik"] for x in syrye(tmp_path / "b", "S1")} == {"k3"} and it["S1"]["n"] > 0
    assert json.loads((tmp_path / "b" / "S1" / "manifest.json").read_text())["k3"] is True


def test_mutaciya_period_ne_v_otpechatke_chekpointa_lovitsya(fake, monkeypatch, tmp_path):   # G4
    monkeypatch.setattr(ps, "fabrika", lambda sid: sreda(sid, Fake5()))
    srv.progon(kuby5(["S1"], rz5()), tmp_path, 2, n={"S1": 100}, rz=rz5())
    with pytest.raises(SystemExit, match="period"):                                  # train4-каталог как v2 — отказ
        srv.progon(kuby5(["S1"], rz5("v2")), tmp_path, 2, n={"S1": 200}, prodolzhit=True, rz=rz5("v2"))
    monkeypatch.setattr(srv, "POLYA", tuple(k for k in srv.POLYA if k != "period"))   # мутация
    it = srv.progon(kuby5(["S1"], rz5("v2")), tmp_path, 2, n={"S1": 200}, prodolzhit=True, rz=rz5("v2"))
    assert it["S1"]["n"] >= 0                       # мутант отпечаток пропустил: SystemExit «period» не поднят, счёт пошёл


def test_mutaciya_vybros_canon_v5_propushchen_lovitsya(fake, monkeypatch, tmp_path):     # G5
    monkeypatch.setattr(ps, "fabrika", lambda sid: sreda(sid, Fake5()))
    ozhid = {i for i, _ in ps.tochki(ps.load_kub("S1", rz5()), 0, 256, rz5())}
    assert 0 < len(ozhid) < 256                                                       # в первых 256 точках S1 есть выброшенные
    srv.progon(kuby5(["S1"], rz5()), tmp_path / "ok", 2, n={"S1": 256}, rz=rz5())
    assert {x["sobol_i"] for x in syrye(tmp_path / "ok", "S1")} == ozhid              # обёртка = драйвер: индексы выброшенных пропущены
    orig = cv5.canon_v5
    monkeypatch.setattr(cv5, "canon_v5", lambda sid, pt: orig(sid, pt) or pt)        # мутация: несовместимая точка не выбрасывается
    srv.progon(kuby5(["S1"], rz5()), tmp_path / "mut", 2, n={"S1": 256}, rz=rz5())
    assert {x["sobol_i"] for x in syrye(tmp_path / "mut", "S1")} != ozhid


def test_main_v5_komanda_servera(fake, monkeypatch, tmp_path):                           # G7: CLI = команды из тела PR
    monkeypatch.setattr(ps, "fabrika", lambda sid: sreda(sid, Fake5()))
    base = ["--kub", "v5", "--strategii", "S1", "--procs", "2"]
    assert srv.main([*base, "--period", "v2", "--seed", "2", "--n1", "S1=60", "--out", str(tmp_path / "0")]) == 0
    assert srv.main([*base, "--period", "train4", "--k3", "--out", str(tmp_path / "1")]) == 0
    m0, m1 = (json.loads((tmp_path / i / "S1" / "manifest.json").read_text()) for i in "01")
    assert (m0["period"], m0["sobol_klyuch"], m1["k3"]) == ("v2", "seed2", True)
    for bad in (["--k3", "--seed", "2"], ["--seed", "3"]):
        with pytest.raises(SystemExit):
            srv.main([*base, "--period", "train4", *bad, "--out", str(tmp_path / "x")])
