"""Эталоны v6 (Ж-137 п. 4): набор v6 пишется в npz, пересчитывается и сверяется; подложенные ошибки ловятся; v5-часть не меняется.
Оценщик — синтетика (детерминированная функция точки), данных и сети нет; кубы v6 — настоящие yaml (данных не читают)."""
import copy
import sys
import zlib
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/server"))
import ekviv  # noqa: E402
import ekviv_io  # noqa: E402
import ekviv_sravn as es  # noqa: E402
import ekviv_v6 as e6  # noqa: E402
from fastbt import progon_sobol as ps  # noqa: E402


class Fake:
    """Оценщик-синтетика: ряд — функция точки. Не-первый вызов с crc % 4 == 0 — исключение (первый вызов — якорь, он не падает)."""
    fdr = "t0d"

    def __init__(self):
        self.n = 0

    def full(self, d):
        return dict(d, a=1)

    def evaluate_v2(self, pt, period):
        self.n += 1
        h = zlib.crc32(repr(sorted((str(k), str(v)) for k, v in pt.items())).encode())
        if self.n > 1 and h % 4 == 0:
            raise ValueError("синтетическая ошибка точки")
        rng = np.random.default_rng(h)
        p = rng.normal(0, 0.01, 6)
        return {"status": "ok", "sharpe": float(p.mean() / p.std()), "god": {"2016": float(p[0])}}, p, rng.normal(0, 0.1, 3)


@pytest.fixture(autouse=True)
def sreda_fake(monkeypatch):
    ps._W.clear()
    daty = np.arange("2016-01-04", "2016-01-10", dtype="datetime64[D]")
    monkeypatch.setattr(ps, "fabrika", lambda sid: ps.Sreda(Fake(), "kod1", f"vh_{sid}", daty, ["AFLT"]))
    monkeypatch.setattr(ps, "fast_ok", lambda pt: False)
    yield
    ps._W.clear()


def _nabor(n=12):
    """-> (rez, meta) как у etalony_mak: v5 (6 точек) + v6 (n точек, набор по квотам)."""
    rez, meta = ekviv.poschitat(ekviv.vybor(6), 1)
    rez6, m6, kod6 = e6.poschitat(None, set(), 1, n)
    e6.dobavit(rez, meta, rez6, m6, kod6)
    return rez, meta


def _pereschet(eta):
    rez, meta = ekviv.poschitat(ekviv_io.tochki_iz(eta), 1)
    t6, b6 = e6.iz_etalona(eta)
    rez6, m6, kod6 = e6.poschitat(t6, b6, 1)
    e6.dobavit(rez, meta, rez6, m6, kod6)
    return rez, meta


def test_kvoty_dayut_200_i_34_33():
    k = e6.kvoty(200)
    assert sum(k.values()) == 200 and sorted(k.values()) == [33, 33, 33, 33, 34, 34]


def test_yakor_v6_sovmestim_na_vseh_sk():
    for sid in ps.SIDS:
        c = ps.load_kub(sid, e6.rezhim(sid))
        pt = e6.tochka_yakorya(c)
        assert pt is not None and set(c["v5"]["yakorya_progon"]) <= set(pt)


def test_v6_pishetsya_i_sveryaetsya_s_soboy(tmp_path):
    rez, meta = _nabor()
    assert sum(m.startswith("v6yakor__") for m in rez) == 6 and sum(m.startswith("v6tochka__") for m in rez) == 12
    f = str(tmp_path / "e.npz")
    ekviv_io.sohranit(f, rez, meta)
    eta, meta_eta = ekviv_io.zagruzit(f)
    moi, meta_moi = _pereschet(eta)
    r = es.sravnit(eta, meta_eta, moi, meta_moi, 1e-12)
    assert r["kod"] == 0 and r["max_tochki"] == 0.0 and r["max_yakor"] == 0.0 and set(eta) == set(moi)


def test_nabor_propuskaet_ne_ok_i_beryot_tolko_ok():
    rez, meta = _nabor()
    assert all('"status": "ok"' in r["sig"] for m, r in rez.items() if m.startswith("v6tochka__"))
    assert sum(meta["v6_propusheno"].values()) > 0          # синтетика падает на ~25 % точек: они пропущены, квота добрана


def _isporchennyj(eta, imya):
    moi = copy.deepcopy(eta)
    k = sorted(moi[imya]["hex"])[0]
    moi[imya]["hex"][k] = (float.fromhex(moi[imya]["hex"][k]) + 1e-9).hex()
    return moi


@pytest.mark.parametrize("pref", ["v6tochka__", "v6yakor__"])
def test_mutaciya_znachenie_na_1e9_lovitsya(tmp_path, pref):
    rez, meta = _nabor()
    imya = next(m for m in sorted(rez) if m.startswith(pref))
    r = es.sravnit(rez, meta, _isporchennyj(rez, imya), copy.deepcopy(meta), 1e-12)
    assert r["kod"] == 3 and r["nepreryvnoe_pervoe"][0] == imya
    assert es.sravnit(rez, meta, copy.deepcopy(rez), copy.deepcopy(meta), 1e-12)["kod"] == 0     # без порчи — ОК


def test_yakor_v6_dopusk_yakorya_a_ne_tochki():
    """Якорь v6 сверяется с допуском якоря 1e-12 даже при DOPUSK = 0 для точек; точка v6 при DOPUSK = 0 — побайтно."""
    rez, meta = _nabor()
    ya = next(m for m in sorted(rez) if m.startswith("v6yakor__"))
    moi = copy.deepcopy(rez)
    moi[ya]["p"] = moi[ya]["p"] + 1e-13
    assert es.sravnit(rez, meta, moi, copy.deepcopy(meta), 0.0)["kod"] == 0
    tc = next(m for m in sorted(rez) if m.startswith("v6tochka__"))
    moi[tc]["p"] = moi[tc]["p"] + 1e-13
    assert es.sravnit(rez, meta, moi, copy.deepcopy(meta), 0.0)["kod"] == 3


def test_mutaciya_perestanovka_dvuh_tochek_lovitsya():
    rez, meta = _nabor()
    a, b = [m for m in sorted(rez) if m.startswith("v6tochka__S1__")][:2]
    moi = copy.deepcopy(rez)
    moi[a], moi[b] = moi[b], moi[a]
    r = es.sravnit(rez, meta, moi, copy.deepcopy(meta), 1e-12)
    assert r["kod"] == 1 and r["diskretnoe"]                 # другой индекс Соболя / ключ точки — дискретное


def test_mutaciya_hash_vhodov_v6_lovitsya():
    rez, meta = _nabor()
    m2 = copy.deepcopy(meta)
    m2["v6_vhody"]["S1"] = "podmena"
    assert es.sravnit(rez, meta, copy.deepcopy(rez), m2, 1e-12)["kod"] == 1


def test_v5_chast_ne_izmenilas_ot_dobavleniya_v6(tmp_path):
    rez5, meta5 = ekviv.poschitat(ekviv.vybor(6), 1)
    ps._W.clear()
    rez, meta = _nabor()
    assert ekviv_io.tochki_iz(rez) == ekviv_io.tochki_iz(rez5)           # v6-имена не попадают в набор v5
    f5, f = tmp_path / "v5.npz", tmp_path / "v56.npz"
    ekviv_io.sohranit(str(f5), rez5, meta5)
    ekviv_io.sohranit(str(f), rez, meta)
    with np.load(f5) as z5, np.load(f) as z:
        for k in z5.files:
            if k != "__meta__":
                assert z[k].tobytes() == z5[k].tobytes(), k
        assert {k for k in z.files if not k.startswith(("v6", "__meta__"))} == set(z5.files) - {"__meta__"}


def test_staryj_etalon_bez_v6_sveryaetsya_po_staromu():
    rez5, meta5 = ekviv.poschitat(ekviv.vybor(6), 1)
    assert not e6.est_v6(rez5) and e6.iz_etalona(rez5) == ([], set())
    assert es.sravnit(rez5, meta5, copy.deepcopy(rez5), copy.deepcopy(meta5), 1e-12)["kod"] == 0
