"""С-v6 (Ж-137): серверный Соболь по кубу v6 на синтетике (данных, сети, счёта нет): куб designs/v6, заморозка Морриса, ledger 2.2.
Мутации: «заморозка не применена» (в точках замороженные оси); «v5-манифест в каталоге v6» (продолжение принято); «другая заморозка при
продолжении» (принято); «k3 берёт новые оси» (k3 ≠ решётка v5); «rol не по проходу» (sobol / kontrol / spravka)."""
import sys
import types
from pathlib import Path

import pytest
import yaml

import fastbt.canon_v6 as c6
import fastbt.ledger_v2 as lm
import fastbt.progon_server as srv
import fastbt.progon_sobol as ps
import fastbt.progon_v6 as v6
from fastbt.progon_morris_v6_polnyj import kub_bez, tochki_polnye
from test_progon_server import Fake, fake, sreda, syrye, chp  # noqa: F401 — fake: фикстура

SID, ZAM, N = "S1", ("dlit", "tihiy_othod"), {"S1": 300}


class Fake6(Fake):
    fdr = "t0d"            # fdr куба v6; оценщик обязан считать его же


def rz6(period="train4", seed="seed", k3=False, rol="sobol", zam=()):
    return ps.Rezhim("v6", period, seed, k3, "t0d", rol, None, zam)


def zam_fajl(katalog: Path, osi=ZAM, kub_sha=None) -> Path:
    """<katalog>/S1/zamorozka.yaml в формате morris_v6_zamorozka.zapisat_yaml."""
    d = katalog / SID
    d.mkdir(parents=True, exist_ok=True)
    sha = kub_sha or v6.sha_fajla(ps.kub_put(SID, rz6()))
    (d / "zamorozka.yaml").write_text(yaml.safe_dump({"v": 1, "data": "2026-10-29", "vhody": {"kub_sha": sha}, "osi": list(osi)}), "utf-8")
    return katalog


def rezhim_s_zam(tmp_path, osi=ZAM, **kw):
    return rz6(**kw)._replace(zam=v6.zamorozka(zam_fajl(tmp_path / "morris", osi), [SID], lambda s: ps.kub_put(s, rz6())))


def kuby6(rz, sids=(SID,)):
    return {s: (ps.kub_put(s, rz), srv.zagruzit_kub(ps.kub_put(s, rz), rz.period, rz=rz)) for s in sids}


def prohod(monkeypatch, root, rz, n=N, **kw):
    monkeypatch.setattr(ps, "fabrika", lambda sid: sreda(sid, Fake6()))
    return srv.progon(kuby6(rz), root, 2, n=n, rz=rz, **kw)


# ---- заморозка ----
def test_zamorozhennye_osi_vyneseny_iz_tochek_i_ledger(fake, monkeypatch, tmp_path):
    rz = rezhim_s_zam(tmp_path)
    it = prohod(monkeypatch, tmp_path / "s1", rz)
    assert it[SID]["sboy"] is None and it[SID]["n"] > 0
    rows = syrye(tmp_path / "s1", SID)
    assert rows and not any(set(r["klyuch"]) & set(ZAM) for r in rows)             # мутация «заморозка не применена» падает здесь
    kub = ps.load_kub(SID, rz)
    kontrol = tochki_polnye(SID, ZAM, 0, N[SID], "seed", kub)[0]                       # те же точки Соболя куба без замороженных осей
    assert {r["sobol_i"] for r in rows} == {i for i, _ in kontrol}
    assert any(set(c) & {"rel_signal", "sob_tip", "time_decay"} for _, c in kontrol)    # остальные новые оси остались в выборке
    svoboda = c6.tochki_v6(kub, 0, N[SID])[0]
    assert any(set(c) & set(ZAM) for _, c in svoboda)                              # без заморозки оси в точках были бы: тест что-то проверяет


def test_mutatsiya_zamorozka_ne_primenena_lovitsya(monkeypatch, tmp_path):
    rz = rezhim_s_zam(tmp_path)
    kub = ps.load_kub(SID, rz)
    chistye = lambda: [c for _, c in v6.tochki(kub, 0, N[SID], rz, ps.tochki_v5)]      # noqa: E731
    assert not any(set(c) & set(ZAM) for c in chistye())
    monkeypatch.setattr(v6, "tochki_polnye", lambda sid, zam, n0, n1, seed, kub: c6.tochki_v6(kub, n0, n1, seed))
    assert any(set(c) & set(ZAM) for c in chistye())                                # мутация даёт замороженные оси: проверка выше её ловит


def test_tochki_dobavlyayut_yakorya_v5_kak_v5(tmp_path):
    rz = rezhim_s_zam(tmp_path)
    kub = ps.load_kub(SID, rz)
    ya = kub["v5"]["yakorya_progon"]
    for _, c in v6.tochki(kub, 0, 50, rz, ps.tochki_v5):
        assert all(c[k] == v or k in c6.novye(kub) for k, v in ya.items())          # якорь v5 в точке (кроме осей v6, его заместивших)


def test_bez_zamorozki_ne_schitaetsya(fake, monkeypatch, tmp_path):
    with pytest.raises(ValueError):
        v6.zam_sid(rz6(), SID)
    with pytest.raises(SystemExit):
        prohod(monkeypatch, tmp_path / "x", rz6())                                  # rol sobol, zam пуст


def test_zamorozka_otkazy(tmp_path):
    put = lambda s: ps.kub_put(s, rz6())                                            # noqa: E731
    with pytest.raises(SystemExit, match="нет"):
        v6.zamorozka(tmp_path / "pusto", [SID], put)
    with pytest.raises(SystemExit, match="другом кубе"):
        v6.zamorozka(zam_fajl(tmp_path / "a", kub_sha="0" * 64), [SID], put)
    with pytest.raises(SystemExit, match="не новая"):
        v6.zamorozka(zam_fajl(tmp_path / "b", osi=("ma_window",)), [SID], put)      # ось v5 заморозке не подлежит
    z = v6.zamorozka(zam_fajl(tmp_path / "c"), [SID], put)
    assert z[0][0] == SID and z[0][2] == tuple(sorted(ZAM)) and len(z[0][1]) == 64


# ---- ledger 2.2 и роли ----
@pytest.mark.parametrize("period", ["train4", "v2"])
def test_ledger_22_setka_v6_rol_sobol(fake, monkeypatch, tmp_path, period):
    rz = rezhim_s_zam(tmp_path, period=period)
    prohod(monkeypatch, tmp_path / "s1", rz)
    mf = lm.Chitatel(tmp_path / "s1", SID).manifest
    assert (mf["v"], mf["setka"], mf["rol"], mf["period"], mf["k3"]) == (lm.V22, "v6", "sobol", period, False)
    assert mf["zamorozka_sha"] == rz.zam[0][1] and mf["kub_sha"] == v6.sha_fajla(ps.kub_put(SID, rz))
    rows = syrye(tmp_path / "s1", SID)
    assert {r["setka"] for r in rows} == {"v6"} and {r["rol"] for r in rows} == {"sobol"}      # мутация «v5-строка в v6» падает здесь
    assert all(r["klyuch_n"] == lm.klyuch_n(SID, r["klyuch"]) for r in rows)
    assert {r["istochnik"] for r in rows} == {"sobol1_seed1"}


def test_seed2_drugie_tochki_togo_zhe_kuba(fake, monkeypatch, tmp_path):
    prohod(monkeypatch, tmp_path / "s1", rezhim_s_zam(tmp_path))
    prohod(monkeypatch, tmp_path / "s2", rezhim_s_zam(tmp_path, seed="seed2"))
    a, b = ({r["klyuch_n"] for r in syrye(tmp_path / d, SID)} for d in ("s1", "s2"))
    assert a and b and a != b
    assert {r["istochnik"] for r in syrye(tmp_path / "s2", SID)} == {"sobol1_seed2"}
    assert lm.Chitatel(tmp_path / "s2", SID).manifest["sobol_seed"] == ps.load_kub(SID, rz6())["sobol"]["seed2"]


def test_k3_kontrol_reshetka_v5_novye_osi_vykl(fake, monkeypatch, tmp_path):
    rz = rz6(k3=True, rol="kontrol")
    it = prohod(monkeypatch, tmp_path / "k3", rz, n={SID: 150})
    assert it[SID]["sboy"] is None
    mf = lm.Chitatel(tmp_path / "k3", SID).manifest
    assert (mf["setka"], mf["rol"], mf["k3"], mf["zamorozka_sha"]) == ("v6", "kontrol", True, None)
    rows = syrye(tmp_path / "k3", SID)
    kub = ps.load_kub(SID, rz)
    nov = set(c6.novye(kub)) - set(kub["v5"]["yakorya_progon"])                     # якоря v5 с теми же именами (rezhim, ed_lim …) — не оси
    assert rows and not any(set(r["klyuch"]) & nov for r in rows)
    # мутация «k3 берёт новые оси» (tochki_v6 вместо кубa-основы) даёт в ключах оси nov и падает на этой проверке
    r5 = ps.Rezhim("v5", "train4", "seed", True, "t0d")
    v5 = {lm.klyuch_n(SID, lm.klyuch(SID, pt, "train4", ps.KOMISSIYA, "t0d")) for _, pt in ps.tochki(ps.load_kub(SID, r5), 0, 150, r5)}
    assert {r["klyuch_n"] for r in rows} == v5                                      # те же точки, что у k3 v5: v5 ⊂ v6 по klyuch_n
    assert {r["istochnik"] for r in rows} == {"k3"}


def test_spravka_v6_close_rol_spravka_vne_N(fake, monkeypatch, tmp_path):
    rz = rezhim_s_zam(tmp_path)
    prohod(monkeypatch, tmp_path / "s1", rz)
    prohod(monkeypatch, tmp_path / "sp", rz._replace(rol="spravka", t_vyhod="close"))
    sp = lm.Chitatel(tmp_path / "sp", SID)
    assert (sp.manifest["rol"], sp.manifest["t_vyhod"], sp.manifest["setka"]) == ("spravka", "close", "v6")
    rows = syrye(tmp_path / "sp", SID)
    assert {r["rol"] for r in rows} == {"spravka"} and {r["klyuch"]["t_vyhod"] for r in rows} == {"close"}
    s1 = lm.Chitatel(tmp_path / "s1", SID)
    assert lm.Obedinenie(sp).n_k() == 0 and lm.Obedinenie(s1, sp).n_k() == lm.Obedinenie(s1).n_k() > 0


def test_prodolzhit_chuzhoj_katalog_i_chuzhaya_zamorozka_otkaz(fake, monkeypatch, tmp_path):
    rz = rezhim_s_zam(tmp_path)
    monkeypatch.setattr(ps, "fabrika", lambda sid: sreda(sid, Fake6()))
    srv.progon({SID: (ps.kub_put(SID, ps.Rezhim("v5", "train4", "seed", False, "t0d")),
                      srv.zagruzit_kub(ps.kub_put(SID, ps.Rezhim("v5", "train4", "seed", False, "t0d")), "train4",
                                       rz=ps.Rezhim("v5", "train4", "seed", False, "t0d")))}, tmp_path / "x", 2, n={SID: 100},
               rz=ps.Rezhim("v5", "train4", "seed", False, "t0d"))
    with pytest.raises(SystemExit, match="не совпали"):
        prohod(monkeypatch, tmp_path / "x", rz, prodolzhit=True)                    # v5-каталог → v6: отказ (setka, rol, kub_sha)
    prohod(monkeypatch, tmp_path / "y", rz, n={SID: 100})
    drugaya = rezhim_s_zam(tmp_path, osi=("dlit",))
    assert drugaya.zam[0][1] != rz.zam[0][1]
    with pytest.raises(SystemExit, match="zamorozka_sha"):
        prohod(monkeypatch, tmp_path / "y", drugaya, prodolzhit=True)               # другая заморозка в тот же каталог: отказ
    assert chp(tmp_path / "y", SID)["zamorozka_sha"] == rz.zam[0][1]


def test_ledger_22_i_rol_obyazatelny_dlya_v6(fake, monkeypatch, tmp_path):
    rz = rezhim_s_zam(tmp_path)
    monkeypatch.setattr(ps, "fabrika", lambda sid: sreda(sid, Fake6()))
    with pytest.raises(SystemExit, match="ledger 2.2 с rol"):
        srv.progon(kuby6(rz), tmp_path / "a", 2, n=N, rz=rz._replace(rol=None))
    with pytest.raises(SystemExit):
        srv.progon(kuby6(rz), tmp_path / "b", 2, n=N, rz=rz, ledger=lm.V21)


# ---- N1, CLI, оценщик ----
def test_n1_po_umolchaniyu_200000_i_obem_k3_iz_kuba_v5():
    rz = rz6()
    kub = ps.load_kub(SID, rz)
    assert ps.n1_po_sid({SID: kub}, None, rz) == {SID: 200_000} and ps.n1_po_sid({SID: kub}, "1000", rz) == {SID: 1000}
    rk = rz6(k3=True, rol="kontrol")
    assert ps.n1_po_sid({SID: kub}, None, rk) == {SID: ps.obem_k3(kub["v5"], ps.Rezhim("v5", "train4", "seed", True, "t0d"))}


def test_cli_v6_zamorozka_nuzhna_krome_k3(tmp_path):
    base = ["--kub", "v6", "--strategii", SID, "--period", "train4", "--procs", "2", "--out", str(tmp_path / "o")]
    with pytest.raises(SystemExit):
        srv.main(base)                                                              # без заморозки
    with pytest.raises(SystemExit):
        srv.main(base + ["--k3", "--zamorozka", str(tmp_path)])                     # у k3 заморозки нет
    with pytest.raises(SystemExit):
        srv.main(["--kub", "v5", "--strategii", SID, "--period", "train4", "--zamorozka", str(tmp_path), "--out", str(tmp_path)])


def test_main_v6_s1_i_k3_peredayut_rol(fake, monkeypatch, tmp_path):
    monkeypatch.setattr(ps, "fabrika", lambda sid: sreda(sid, Fake6()))
    zam_fajl(tmp_path / "morris")
    base = ["--kub", "v6", "--strategii", SID, "--period", "train4", "--procs", "2", "--n1", "120"]
    assert srv.main(base + ["--zamorozka", str(tmp_path / "morris"), "--out", str(tmp_path / "s1")]) == 0
    assert srv.main(base[:-2] + ["--k3", "--out", str(tmp_path / "k3")]) == 0
    assert lm.Chitatel(tmp_path / "s1", SID).manifest["rol"] == "sobol" and lm.Chitatel(tmp_path / "k3", SID).manifest["rol"] == "kontrol"
    assert srv.main(base + ["--zamorozka", str(tmp_path / "morris"), "--rol", "spravka", "--t-vyhod", "close",
                            "--out", str(tmp_path / "sp")]) == 0
    assert lm.Chitatel(tmp_path / "sp", SID).manifest["rol"] == "spravka"


def test_ocenshchik_bez_istochnikov_otkaz_i_s_zagruzchikom(monkeypatch):
    rz, inp = rz6(), types.SimpleNamespace(sk=types.SimpleNamespace(tickers=("A", "B")))
    monkeypatch.delitem(sys.modules, v6.IST, raising=False)
    monkeypatch.setattr(v6.importlib, "import_module", lambda n: (_ for _ in ()).throw(ModuleNotFoundError(name=n)) if n == v6.IST else None)
    with pytest.raises(SystemExit, match="Ист-v6"):
        v6.ocenshchik(SID, inp, {}, 0.0004, rz)
    vidno = {}
    ist = object()
    stub = types.SimpleNamespace(zagruzit=lambda period, tikery: vidno.update(period=period, tikery=tikery) or ist)
    monkeypatch.setattr(v6.importlib, "import_module", lambda n: stub)
    import fastbt.eval_v6 as ev
    monkeypatch.setattr(ev, "EvaluatorV6", lambda sid, inp, bp, kom, fdr, ist: (sid, kom, fdr, ist))
    assert v6.ocenshchik(SID, inp, {}, 0.0004, rz) == (SID, 0.0004, "t0d", ist)
    assert vidno == {"period": "train4", "tikery": ["A", "B"]}


def test_vhody_uchityvaet_ryady_v6():
    s = lambda sha: types.SimpleNamespace(ist=types.SimpleNamespace(sha=sha))        # noqa: E731
    assert v6.vhody("d", types.SimpleNamespace(ist=types.SimpleNamespace())) == "d"
    assert v6.vhody("d", s("a")) != v6.vhody("d", s("b")) != "d"                     # подмена рядов v6 меняет ключ (Д5)


def test_zapusk_sobol_sh_v6():
    t = (Path(__file__).resolve().parents[1] / "scripts/server/perenos/zapusk_sobol.sh").read_text(encoding="utf-8")
    assert "PERENOS_MORRIS" in t and "--zamorozka" in t and "progon_morris_v6" in t
