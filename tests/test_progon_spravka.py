"""Ж-137 п. 6 (лист P-4): справочный проход `--rol spravka --t-vyhod close` — тот же куб, выход по close, ledger 2.2 с rol = spravka,
в N не входит. Синтетика (Fake), данных нет. Мутации: spravka попадает в N (VNE_N без неё — тест N падает); t_vyhod не подменён
(оценщик видит open2); флаг без пары принят; проход пишет 2.1 без rol."""
from pathlib import Path

import pytest

import fastbt.ledger_v2 as lm
import fastbt.progon_server as srv
import fastbt.progon_sobol as ps
from tests.test_progon_server import SV, Fake, chp, fake, kuby5, rz5, sreda, syrye  # noqa: F401 — fake: фикстура

SH = Path(__file__).resolve().parents[1] / "scripts/server/perenos/zapusk_sobol.sh"
N = {"S1": 120}


class Vidit(Fake):
    fdr = "t0d"            # как Fake5: fdr куба v5
    zhurnal: Path = Path()           # дети — fork: список в памяти родитель не увидит, пишем в файл

    def evaluate_v2(self, pt, period):
        with open(Vidit.zhurnal, "a") as f:
            f.write(f"{pt.get('t_vyhod')}\n")
        return super().evaluate_v2(pt, period)


def prohod(monkeypatch, root, rz):
    monkeypatch.setattr(ps, "fabrika", lambda sid: sreda(sid, Vidit()))
    return srv.progon(kuby5(["S1"], rz), root, 2, n=N, rz=rz)


def para(monkeypatch, tmp_path, period="train4"):
    rz, chto = rz5(period), lambda p: set(p.read_text().split()) if p.exists() else set()  # noqa: E731
    Vidit.zhurnal = tmp_path / "baza.txt"
    prohod(monkeypatch, tmp_path / "s1", rz)
    Vidit.zhurnal = tmp_path / "sprav.txt"
    it = prohod(monkeypatch, tmp_path / "sp", rz._replace(rol="spravka", t_vyhod="close"))
    return it, chto(tmp_path / "baza.txt"), chto(tmp_path / "sprav.txt")


@pytest.mark.parametrize("period", ["train4", "v2"])
def test_spravka_pishet_ledger_22_rol_i_close(fake, monkeypatch, tmp_path, period):
    it, baza_t, sp_t = para(monkeypatch, tmp_path, period)
    assert it["S1"]["sboy"] is None and it["S1"]["n"] > 0
    assert sp_t == {"close"} and "close" not in baza_t                    # выход по close только в справке; мутация «не подменён» падает здесь
    mf = lm.Chitatel(tmp_path / "sp", "S1").manifest
    assert (mf["v"], mf["rol"], mf["t_vyhod"], mf["period"]) == (lm.V22, "spravka", "close", period)
    rows = syrye(tmp_path / "sp", "S1")
    assert {r["rol"] for r in rows} == {"spravka"} and {r["klyuch"]["t_vyhod"] for r in rows} == {"close"}
    assert all(r["klyuch_n"] == lm.klyuch_n("S1", r["klyuch"]) for r in rows)
    assert chp(tmp_path / "sp", "S1")["rol"] == "spravka"                  # отпечаток чекпоинта: продолжение чужой роли — отказ


def test_spravka_ne_v_N_obedinenie(fake, monkeypatch, tmp_path):
    para(monkeypatch, tmp_path)
    s1, sp = lm.Chitatel(tmp_path / "s1", "S1"), lm.Chitatel(tmp_path / "sp", "S1")
    n_baza = lm.Obedinenie(s1).n_k()
    assert n_baza > 0 and lm.Obedinenie(sp).n_k() == 0                     # только spravka — N пуст
    assert lm.Obedinenie(s1, sp).n_k() == n_baza and lm.Obedinenie(s1, sp).klyuchi() == lm.Obedinenie(s1).klyuchi()
    monkeypatch.setattr(lm, "VNE_N", (lm.KONTROL,))                        # мутация: spravka попадает в N
    assert lm.Obedinenie(s1, sp).n_k() == 2 * n_baza != n_baza             # тест выше падает на ней


def test_spravka_ne_smeshivaetsya_s_baznym_kataloge_prodolzhenie_chuzhoj_roli(fake, monkeypatch, tmp_path):
    prohod(monkeypatch, tmp_path / "x", rz5())
    with pytest.raises(SystemExit):
        prohod(monkeypatch, tmp_path / "x", rz5()._replace(rol="spravka", t_vyhod="close"))   # без --prodolzhit: каталог начат


def test_svodka_pokazyvaet_spravku_otdelnoj_strokoj(fake, monkeypatch, tmp_path):
    para(monkeypatch, tmp_path)
    rows = SV.svodka_vse([tmp_path / "s1", tmp_path / "sp"], 1)
    pr = [r["prohod"] for r in rows[:-1]]
    assert pr == ["train4 s1", "spravka close train4"] and rows[-1]["sverka"] == "ok"


def test_flagi_tolko_paroj_i_tolko_v5(fake, tmp_path):
    base = ["--kub", "v5", "--strategii", "S1", "--procs", "2", "--out", str(tmp_path)]
    with pytest.raises(SystemExit):
        srv.main(base + ["--rol", "spravka"])
    with pytest.raises(SystemExit):
        srv.main(base + ["--t-vyhod", "close"])
    with pytest.raises(SystemExit):
        srv.main(["--kub", str(Path(ps.KUB).parent / "kub_S1.yaml"), "--period", "train4", "--rol", "spravka", "--t-vyhod", "close",
                  "--out", str(tmp_path)])


def test_zapusk_sobol_spravka_posle_polnogo_prohoda_v_svoj_katalog():
    s = SH.read_text(encoding="utf-8")
    assert "s1 s2 k3 spravka" in s                                         # порядок: полные проходы, затем spravka
    assert "--rol spravka --t-vyhod close" in s and '${per}_spravka_close' in s
