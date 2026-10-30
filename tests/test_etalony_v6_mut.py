"""Тестировщик PR #722: мутации, которых нет в tests/test_etalony_v6.py (допуск по умолчанию, дискретные поля v6, v6 нет на сервере,
независимость набора от числа процессов, хэши v6_kub / v6_daty). Оценщик — синтетика, данных и сети нет."""
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
import proverka_ekviv as pe  # noqa: E402
from fastbt import progon_sobol as ps  # noqa: E402


class Fake:
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


def _nabor(n=12, procs=1):
    rez, meta = ekviv.poschitat(ekviv.vybor(6), 1)
    rez6, m6, kod6 = e6.poschitat(None, set(), procs, n)
    e6.dobavit(rez, meta, rez6, m6, kod6)
    return rez, meta


def _imya(rez, pref):
    return next(m for m in sorted(rez) if m.startswith(pref))


# (а) допуск по умолчанию в самой проверке, не только в sravnit
def test_tol_yakor_ravna_1e12():
    assert es.TOL_YAKOR == 1e-12


@pytest.mark.parametrize("pref", ["v6yakor__", "v6tochka__"])
def test_sdvig_1e9_lovitsya_pri_dopuske_1e12_cherez_main(tmp_path, monkeypatch, pref):
    """main без переменной DOPUSK: умолчание 1e-12 (не 1e-6) — сдвиг 1e-9 v6-записи даёт код 3, а не OK."""
    monkeypatch.delenv("DOPUSK", raising=False)
    rez, meta = _nabor()
    moi = copy.deepcopy(rez)
    imya = _imya(rez, pref)
    k = sorted(moi[imya]["hex"])[0]
    moi[imya]["hex"][k] = (float.fromhex(moi[imya]["hex"][k]) + 1e-9).hex()
    fe, fm = str(tmp_path / "e.npz"), str(tmp_path / "m.npz")
    ekviv_io.sohranit(fe, rez, meta)
    ekviv_io.sohranit(fm, moi, copy.deepcopy(meta))
    otm = tmp_path / "p" / "OK"
    rc = pe.main(["--etalony", fe, "--moi", fm, "--otmetka", str(otm), "--kod-sha", "k", "--dannye-sha", "d"])
    assert rc == 3 and not otm.exists()
    assert pe.main(["--etalony", fe, "--moi", fe, "--otmetka", str(otm), "--kod-sha", "k", "--dannye-sha", "d"]) == 0 and otm.exists()


# (б) дискретное поле v6 — строго, без допуска
@pytest.mark.parametrize("pole", ["sd", "vi"])
def test_diskretnoe_pole_v6_strogo(pole):
    rez, meta = _nabor()
    imya = _imya(rez, "v6tochka__")
    moi = copy.deepcopy(rez)
    forma = rez[imya][pole].shape[1:]           # у синтетики массивы пустые: даём непустые одной формы, значение большое
    rez[imya][pole] = np.full((3, *forma), 1_000_000, dtype=np.int64)
    moi[imya][pole] = rez[imya][pole].copy()
    assert es.sravnit(rez, meta, copy.deepcopy(rez), copy.deepcopy(meta), 0.0)["kod"] == 0
    moi[imya][pole][0] += 1                      # на 1 из 10^6: allclose/допуск это пропустит, array_equal — нет
    assert es.sravnit(rez, meta, moi, copy.deepcopy(meta), 1.0)["kod"] == 1       # даже при огромном допуске


def test_disk_stroka_v6_strogo():
    rez, meta = _nabor()
    imya = _imya(rez, "v6yakor__")
    moi = copy.deepcopy(rez)
    moi[imya]["disk"] = moi[imya]["disk"].replace("{", '{"lishnee": 1, ', 1)
    assert es.sravnit(rez, meta, moi, copy.deepcopy(meta), 1.0)["kod"] == 1


# (в) на сервере нет v6-записей, а в эталоне есть: ошибка, не OK
def test_net_v6_na_servere_fail_diskretnoe():
    rez, meta = _nabor()
    moi = {m: v for m, v in rez.items() if not m.startswith("v6")}
    m2 = {k: v for k, v in copy.deepcopy(meta).items() if not k.startswith("v6_")}
    r = es.sravnit(rez, meta, moi, m2, 1e-12)
    assert r["kod"] == 1 and any("набор записей" in s for s in r["diskretnoe"])


def test_net_v6_na_servere_fail_cherez_main(tmp_path):
    rez, meta = _nabor()
    moi = {m: v for m, v in rez.items() if not m.startswith("v6")}
    fe, fm = str(tmp_path / "e.npz"), str(tmp_path / "m.npz")
    ekviv_io.sohranit(fe, rez, meta)
    ekviv_io.sohranit(fm, moi, {k: v for k, v in meta.items() if not k.startswith("v6_")})
    otm = tmp_path / "p" / "OK"
    assert pe.main(["--etalony", fe, "--moi", fm, "--otmetka", str(otm), "--kod-sha", "k", "--dannye-sha", "d"]) == 1
    assert not otm.exists()


def test_main_pereschityvaet_v6_i_daet_ok_a_bez_v6_fail(tmp_path, monkeypatch):
    """Живой путь main (без --moi): эталон с v6 -> сервер обязан пересчитать v6; пропуск пересчёта = FAIL, не OK."""
    monkeypatch.delenv("DOPUSK", raising=False)
    rez, meta = _nabor()
    meta.update(ekviv.sreda_meta())
    fe = str(tmp_path / "e.npz")
    ekviv_io.sohranit(fe, rez, meta)
    otm = tmp_path / "p" / "OK"
    arg = ["--etalony", fe, "--otmetka", str(otm), "--kod-sha", "k", "--dannye-sha", "d", "--procs", "1"]
    ps._W.clear()
    assert pe.main(arg) == 0 and otm.exists()
    monkeypatch.setattr(pe.ekviv_v6, "est_v6", lambda r: False)       # сервер «не считает» v6
    ps._W.clear()
    assert pe.main(arg) == 1 and not otm.exists()


def test_otmetka_ok_snimaetsya_pri_fail(tmp_path):
    """Старая отметка OK не переживает FAIL (иначе счёт пойдёт по чужой отметке)."""
    rez, meta = _nabor()
    moi = {m: v for m, v in rez.items() if not m.startswith("v6")}
    fe, fm = str(tmp_path / "e.npz"), str(tmp_path / "m.npz")
    ekviv_io.sohranit(fe, rez, meta)
    ekviv_io.sohranit(fm, moi, meta)
    otm = tmp_path / "p" / "OK"
    otm.parent.mkdir()
    otm.write_text("staraya")
    pe.main(["--etalony", fe, "--moi", fm, "--otmetka", str(otm), "--kod-sha", "k", "--dannye-sha", "d"])
    assert not otm.exists()


# (д) набор не зависит от числа процессов
def test_nabor_ne_zavisit_ot_chisla_processov():
    r1, m1 = _nabor(procs=1)
    r2, m2 = _nabor(procs=2)
    assert sorted(r1) == sorted(r2)
    assert all(np.array_equal(r1[m]["p"], r2[m]["p"]) for m in r1)
    assert m1["v6_propusheno"] == m2["v6_propusheno"]


@pytest.mark.parametrize("n", [1, 7, 200])
def test_kvoty_summa_ravna_n(n):
    assert sum(e6.kvoty(n).values()) == n


# (е) хэши куба и дат v6 тоже сверяются
@pytest.mark.parametrize("pole", ["v6_kub", "v6_daty"])      # v6_vhody не сверяется (SVML ≠ libm): tests/test_sverka_bez_vhody.py
def test_hash_v6_lovitsya(pole):
    rez, meta = _nabor()
    m2 = copy.deepcopy(meta)
    m2[pole]["S1"] = "podmena"
    r = es.sravnit(rez, meta, copy.deepcopy(rez), m2, 1.0)
    assert r["kod"] == 1 and any(s.startswith(pole) for s in r["diskretnoe"])


def test_kod_v5_i_v6_dolzhen_sovpadat():
    rez, meta = _nabor()
    with pytest.raises(RuntimeError):
        e6.dobavit(rez, meta, {}, {}, "drugoj_kod")


def test_net_v6_v_etalone_ne_tyanet_v6_u_servera():
    """Эталон без v6, у сервера v6 есть: набор записей различается -> FAIL (лишние записи не молчат)."""
    rez, meta = _nabor()
    eta = {m: v for m, v in rez.items() if not m.startswith("v6")}
    me = {k: v for k, v in copy.deepcopy(meta).items() if not k.startswith("v6_")}
    assert es.sravnit(eta, me, rez, meta, 1e-12)["kod"] == 1
