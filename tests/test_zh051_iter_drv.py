"""К0-д Ж-051: ledger (sha256, только дополнение), общий PBO по матрице всех итераций, контроли вне N, шапка про дивиденды,
драйвер `--it 3` на подменённом оценщике (6 стратегий), правило остановки в конце. Без данных."""
import math
import zlib

import numpy as np
import pandas as pd
import pytest

from fastbt import etap1_poisk as P, etap1_poisk_vorota as V, zh051_iter as I, zh051_kriteriy as K


def fake_res(seed: int, drift: float = 0.0):
    rng = np.random.default_rng(seed)
    x = {y: rng.normal(drift, 0.01, 252) for y in (2018, 2019)}
    m = {y: float(v.mean() / v.std(ddof=1) * math.sqrt(252)) for y, v in x.items()}
    return {"v": "1.0", "excess_d": x, "msr_by_year": m, "by_year": m, "sdelok_y": {2018: 60.0, 2019: 60.0},
            "storony_y": {2018: 120.0, 2019: 120.0}, "vyrozhdena": False}


def fake_ev(p, per, calls=None):
    if calls is not None:
        calls.append(p)
    kp = {k: v for k, v in p.items() if k != "kontrol"}
    return fake_res(zlib.crc32(repr(sorted(kp.items(), key=str)).encode()) % 10 ** 6, 0.0004 * (kp.get("top_k", 0) == 3))


@pytest.fixture
def led_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(K, "LED", tmp_path)
    (tmp_path / "SHA256SUMS.txt").write_text("# sha256  файл  N\n", encoding="utf-8")
    return tmp_path


def test_ledger_sha_i_tolko_dopolnenie(led_dir):
    L = V.Ledger(fake_ev)
    for i in range(5):
        L({"x": i}, "train")
    K.save_ledger("S1", L, "ledger_it2.pkl")
    assert K.sums_read()["S1/ledger_it2.pkl"][1] == 5
    assert K.load_ledger("S1", "ledger_it2.pkl", fake_ev).n_unique == 5
    with pytest.raises(SystemExit, match="только дополняется"):
        K.save_ledger("S1", L, "ledger_it2.pkl")
    p = led_dir / "S1" / "ledger_it2.pkl"
    p.write_bytes(p.read_bytes() + b" ")                            # подмена файла
    with pytest.raises(SystemExit, match="sha256"):
        K.load_ledger("S1", "ledger_it2.pkl", fake_ev)
    with pytest.raises(SystemExit):
        K.load_ledger("S1", "ledger_it9.pkl", fake_ev)


def test_sverka_ledger_lovit_drift_i_merge_konflikt():
    L = V.Ledger(fake_ev)
    for i in range(8):
        L({"x": i}, "train")
    assert K.sverka_ledger(L, fake_ev, n=8) == []
    assert K.sverka_ledger(L, lambda p, per: fake_res(1), n=8) != []            # код дрейфует — ловится
    A = V.Ledger(fake_ev)
    A({"x": 0}, "train")
    B = V.Ledger(lambda p, per: fake_res(99))
    B({"x": 0}, "train")
    with pytest.raises(ValueError):
        K.merge(A, B)


def test_obshchiy_pbo_rastet_po_stolbcam_a_kontroli_ne_v_N():
    L = V.Ledger(lambda p, per: fake_res(p["i"], 0.0 if p["i"] >= 6 else 0.0005))
    for i in range(6):
        L({"i": i}, "train")
    n0, pb0, m0 = L.n_unique, V.pbo(V.matrix(L)), V.matrix(L).shape[1]
    kont = [fake_ev({"top_k": 5, "kontrol": ("sluchaynyy_rang", s)}, "train") for s in range(20)]    # вне Ledger
    assert len(kont) == 20 and L.n_unique == n0 and V.matrix(L).shape[1] == m0
    for i in range(6, 306):
        L({"i": i}, "train")
    assert V.matrix(L).shape[1] == m0 + 300 and L.n_unique == n0 + 300             # N и столбцы PBO растут вместе
    assert V.pbo(V.matrix(L)) > pb0                                                 # шум в матрице поднимает PBO
    cols = [V.series(L.cache[(V.pkey({"i": i}), "train")][1]) for i in range(6, 306)]
    only_new = np.column_stack([np.r_[s[2018], s[2019]] for s in cols])
    assert V.pbo(only_new) != V.pbo(V.matrix(L))                                    # PBO «по столбцам итерации» — другое число


def test_write_table_shapka_dividendy(tmp_path):
    p = tmp_path / "it3" / "S1_itog.csv"
    K.write_table(p, [{"a": 1}])
    K.write_table(p, [{"a": 2}])
    t = p.read_text("utf-8").splitlines()
    assert "PnL без дивидендов (находка #362)" in t[1] and t.count("a") == 1 and t[-1] == "2"


class FakeEvr:
    def __init__(self, sid, inp, base, comm=0.0004):
        self.sid, self.n = sid, []

    def evaluate(self, pt, period):
        return fake_ev(pt, period, self.n)


class SerialPool:
    def __init__(self, *a, **k): pass
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def map(self, fn, it): return [fn(x) for x in it]


@pytest.fixture
def drv(led_dir, tmp_path, monkeypatch):
    monkeypatch.setattr(I, "OUT", tmp_path / "tables")
    monkeypatch.setattr(I, "PlusEvaluator", FakeEvr)
    monkeypatch.setattr(I.etap1, "build_inputs", lambda sid: None)
    monkeypatch.setattr(I.etap1, "base_point", lambda sid: {})
    monkeypatch.setattr(I.etap1, "anchor_s9_diff", lambda sid, evr: {2018: 0.0, 2019: 0.0})
    monkeypatch.setattr(I, "ProcessPoolExecutor", SerialPool)
    for k in range(1, 7):                                          # ledger итерации 2: p1, якорь и 40 точек
        sid = f"S{k}"
        S = P.load_setka(sid)
        row = pd.read_csv(f"docs/tables/zh047/{sid}_itog_v2.csv", comment="#").iloc[-1]
        p1 = {a: P._coerce(str(row[f"p_{a}"]), S["axes"][a]) for a in S["axes"]}
        L = V.Ledger(fake_ev)
        L(p1, "train"), L(S["yakor"], "train")
        for i in range(40):
            L({**p1, "i": i}, "train")
        K.save_ledger(sid, L, "ledger_it2.pkl")
    return tmp_path


ALL = [f"S{k}" for k in range(1, 7)]


def test_it3_polnyy_progon_N_pbo_i_ostanovka(drv):
    st = I.main(["--it", "3", "--strategii", *ALL])
    for sid in ALL:
        f = drv / "tables" / "it3" / f"{sid}_itog.csv"
        r = pd.read_csv(f, comment="#").iloc[-1]
        assert r["N_obshchee"] == r["N_do"] + r["N_novyh"] and r["N_pbo"] == r["N_obshchee"]    # одна общая матрица
        assert r["N_do"] == 42 and r["N_novyh"] >= 4                       # ledger_it2 + пробы плана (+ G4), не контроли
        assert K.sums_read()[f"{sid}/ledger_it3.pkl"][1] == r["N_obshchee"]
        assert (drv / sid / "ledger_it3.pkl").exists()
        assert "находка #362" in f.read_text("utf-8").splitlines()[1]
    assert st["stop"] == (st["itog"] == "остановка, A′ = A")


def test_it3_kontroli_ne_popadayut_v_ledger_i_stop_pri_pbo_vyshe(drv, monkeypatch):
    seen = {}
    real = I.kontroli

    def spy(evr, pt, it, y):
        out = real(evr, pt, it, y)
        seen.update(n=len(out), pts=sum(1 for p in evr.n if "kontrol" in p))
        return out
    monkeypatch.setattr(I, "kontroli", spy)
    monkeypatch.setattr(I.V, "pbo", lambda M: 0.5)                  # PBO > 0,2 у всех
    st = I.main(["--it", "3", "--strategii", *ALL])
    assert st["stop"] and st["itog"] == "остановка, A′ = A"
    assert seen["n"] == 20 and seen["pts"] == 20
    L = K.load_ledger("S1", "ledger_it3.pkl", fake_ev)
    assert not any("kontrol" in dict(k[0]) for k in L.cache if not isinstance(k[0][0], str))
    assert (drv / "tables" / "it3" / "ostanovka.csv").exists()


def test_it3_ne_ostanovka_esli_pbo_nizkiy(drv, monkeypatch):
    monkeypatch.setattr(I.V, "pbo", lambda M: 0.1)
    assert not I.main(["--it", "3", "--strategii", *ALL])["stop"]   # мутация «остановка всегда» ловится


def test_it3_podmenennyy_ledger_stop(drv):
    p = drv / "S2" / "ledger_it2.pkl"
    p.write_bytes(p.read_bytes() + b"\0")
    with pytest.raises(SystemExit, match="sha256"):
        I.run_it("S2", 3)


def test_it3_mutatsiya_poroga_ostanovki(drv, monkeypatch):
    monkeypatch.setattr(K, "PBO_STOP", 0.9)                         # порог 0,2 поднят — остановка не должна срабатывать молча
    monkeypatch.setattr(I.V, "pbo", lambda M: 0.5)
    assert not I.main(["--it", "3", "--strategii", *ALL])["stop"]
