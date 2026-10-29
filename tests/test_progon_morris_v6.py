"""М-драйвер Морриса v6 на синтетике (данных нет, сети нет): известная функция с нулевой осью → ось заморожена, v5 нет в заморозке,
в точках полного прохода замороженных осей нет; окна Y без 2019; возобновление с чекпоинта; мутации «заморозка не применена» и «окно с 2019»."""
import argparse

import numpy as np
import pytest

from fastbt import canon_v6, progon_morris_v6 as pm, progon_morris_v6_polnyj as pn, progon_sobol as ps
from fastbt.metriki_e import sr_dnevnoy

SID = "S1"
REZ = ps.Rezhim("v6", "train4", "seed", False, "t0d")
DATY = np.arange(np.datetime64("2016-01-01"), np.datetime64("2019-01-01"))
SHUM = np.random.default_rng(7).normal(0, 0.01, DATY.size)
OS_REL = {"syroy": 0, "sektor": 1, "rynok": 2}


class Fake:
    """Excess = шум + 0,01·f(точки); f зависит ТОЛЬКО от rel_signal (новая ось) — остальные новые оси «нулевые»."""
    fdr = "t0d"

    def evaluate_v2(self, pt, period):
        return {"status": "ok"}, SHUM + 0.01 * OS_REL[pt.get("rel_signal", "syroy")], None


def sreda(sid, ev=None):
    return ps.Sreda(ev or Fake(), "kod", "vhody", DATY, ["T"])


@pytest.fixture
def fake(monkeypatch):
    monkeypatch.setattr(ps, "fabrika", sreda)


def progon(tmp_path, r=60, prodolzhit=False):
    a = argparse.Namespace(procs=2, prodolzhit=prodolzhit)
    return pm.strategiya(SID, a, tmp_path, REZ, r, mnozh=2)


def test_nulevaya_os_zamorozhena_a_rabochaya_net(fake, tmp_path):
    i = progon(tmp_path)
    zam = pn.prochitat(tmp_path / SID / "zamorozka.yaml")
    kub, ox = pm.setka(SID)
    nov, v5 = canon_v6.novye(kub), {n for n, _, _ in ox} - set(canon_v6.novye(kub))
    assert i["oshibok"] == 0 and "rel_signal" not in zam
    assert "tihiy_othod" in zam and "dlit" in zam              # корневые новые оси без эффекта
    assert zam <= set(nov) and not zam & v5                    # оси v5 не замораживаются никогда


def test_polnyj_prohod_bez_zamorozhennyh_osej(fake, tmp_path):
    progon(tmp_path)
    zam = pn.prochitat(tmp_path / SID / "zamorozka.yaml")
    pts, _ = pn.tochki_polnye(SID, zam, 0, 3000)
    assert pts and not any(set(c) & zam for _, c in pts)
    assert any(c.get("rel_signal", "syroy") != "syroy" for _, c in pts)        # рабочая ось в выборке осталась
    assert len(canon_v6.osi(pn.kub_bez(canon_v6.load_kub(SID), zam))) < len(canon_v6.osi(canon_v6.load_kub(SID)))


def test_mutatsiya_zamorozka_ne_primenena_lovitsya(fake, tmp_path, monkeypatch):
    progon(tmp_path)
    zam = pn.prochitat(tmp_path / SID / "zamorozka.yaml")
    monkeypatch.setattr(pn, "kub_bez", lambda kub, z: kub)                      # мутация: куб как есть
    pts, _ = pn.tochki_polnye(SID, zam, 0, 3000)
    assert any(set(c) & zam for _, c in pts)                                    # тест 2 на такой выборке упал бы


def test_prodolzhit_ne_schitaet_zanovo(fake, tmp_path):
    progon(tmp_path)
    with pytest.raises(SystemExit):                                             # каталог начат — нужен --prodolzhit
        progon(tmp_path)
    f = tmp_path / SID / pm.YF
    chto = f.read_text("utf-8")
    f.write_text(chto + '{"a": 0, "b": 5, "y": [[1', "utf-8")                   # обрыв записи
    progon(tmp_path, prodolzhit=True)
    assert f.read_text("utf-8") == chto


def test_okna_y_bez_2019_i_nol_sdelok():
    x = SHUM.copy()
    sr = ps.Sreda(type("E", (), {"evaluate_v2": lambda s, p, per: ({"status": "ok"}, x, None)})(), "k", "v", np.arange(
        np.datetime64("2016-01-01"), np.datetime64("2020-01-01"))[:x.size], ["T"])
    y = pm.y_okna(sr, {}, "train4")
    assert y == [sr_dnevnoy(x[sr.daty <= np.datetime64(d)]) for d in pm.DO.values()]
    x2 = x.copy()
    x2[sr.daty >= np.datetime64("2019-01-01")] = 5.0                            # утечка 2019 не должна менять Y (мутация: окно шире)
    sr2 = sr._replace(evr=type("E", (), {"evaluate_v2": lambda s, p, per: ({"status": "ok"}, x2, None)})())
    assert pm.y_okna(sr2, {}, "train4") == y
    nol = sr._replace(evr=type("E", (), {"evaluate_v2": lambda s, p, per: ({"status": "ok"}, None, None)})())
    assert pm.y_okna(nol, {}, "train4") == [0.0] * 3 and np.isnan(pm.y_okna(nol, None, "train4")).all()
