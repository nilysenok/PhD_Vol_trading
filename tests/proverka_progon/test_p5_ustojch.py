"""П-5 / Л-10: независимые тесты `fastbt/progon_ustojch.py` (PR #438) по контракту (§1.4, §5 вопрос 2) и заданию Л-10. Синтетика.
Требования не ослаблены. Изменения против первой версии (9b9b354) — под фактический API:
  loto_portfel -> summy_tik + portfel_bez (форма T x N);  loyo_trenirovka -> loyo(M, rows, daty) с подменой vybor_region;
  peresemple_kvartaly/bootstrap_kvartaly -> butstrap (ключи mediana_p5_50_95, dolya_p5_50_95);
  vybor_popravki -> vybrat;  PORIG_VYIGRYSH/MIN_GODOV -> POROG/MIN_GODOV;  dlya_kandidatov -> tiker_popravka (строки с istochnik='tiker').
python3 -m pytest tests/proverka_progon/test_p5_ustojch.py -q"""
import importlib
import inspect

import numpy as np
import pytest


def pu():
    return importlib.import_module("fastbt.progon_ustojch")


def _tik():
    rng = np.random.default_rng(7)
    x = rng.normal(size=(4, 17, 30)).astype(np.float64)
    x[0, 3, 5] = np.nan
    x[1, 0, :] = np.nan
    x[2, 16, 10] = np.nan
    return x


def _bez(x, i):
    s, c = pu().summy_tik(x)
    return pu().portfel_bez(x, s, c, i)


def test_loto_znamenatel_chislo_prisutstvuyushchih():
    x = _tik()
    for drop in (0, 3, 16):
        keep = [i for i in range(17) if i != drop]
        got = _bez(x, drop)
        assert got.shape == (30, 4)
        assert np.allclose(got, np.nanmean(x[:, keep, :], axis=1).T, atol=1e-15, equal_nan=True), drop


def test_loto_ne_delit_na_fiksirovannoe_chislo_pri_nan():
    x = np.ones((1, 17, 3))
    x[0, 5, 1] = np.nan
    assert np.allclose(_bez(x, 2), 1.0)                  # деление на 16 (или 17) дало бы 15/16 на дату 1


def test_loto_tiker_edinstvennyj_na_date_daet_nan():
    x = np.full((1, 17, 2), np.nan)
    x[0, 4, 0], x[0, 7, 1] = 0.5, 0.1
    got = _bez(x, 4)
    assert np.isnan(got[0, 0]) and got[1, 0] == 0.1


def test_loto_ne_menyaet_vhod_i_float32_schitaetsya_v_float64():
    x = _tik()
    c = x.copy()
    _bez(x, 5)
    assert np.array_equal(x, c, equal_nan=True)
    g32 = _bez(x.astype(np.float32), 5)
    assert g32.dtype == np.float64 and np.allclose(g32, _bez(x, 5), atol=1e-6, equal_nan=True)


def test_loyo_chetyre_folda_dvenadtsat_kvartalov_bez_goda(monkeypatch):
    calls = []

    def fake(M, rows, daty, kv=None, **kw):
        calls.append(None if kv is None else sorted(kv))
        return None
    monkeypatch.setattr(pu(), "vybor_region", fake)
    daty = np.arange(np.datetime64("2016-01-01"), np.datetime64("2020-01-01"))
    out = pu().loyo(np.zeros((len(daty), 3)), [], daty)
    assert calls[0] is None and len(calls) == 5 and len(out) == 4        # полный выбор + 4 фолда
    for i, kv in enumerate(calls[1:]):
        assert len(kv) == 12 and not set(kv) & set(range(4 * i, 4 * i + 4)) and set(kv) | set(range(4 * i, 4 * i + 4)) == set(range(16))
    assert [r["god"] for r in out] == [2016, 2017, 2018, 2019]


def _ryad_daty():
    daty = np.arange(np.datetime64("2016-01-01"), np.datetime64("2020-01-01"))
    x = 0.001 + 0.01 * np.random.default_rng(2).normal(size=len(daty))
    return x, daty


def test_butstrap_po_umolchaniyu_seed_i_n_po_zadaniyu():
    sg = inspect.signature(pu().butstrap)
    assert sg.parameters["seed"].default == 20261028 and sg.parameters["n"].default == 10000


def test_butstrap_vosproizvodim_i_porjadok_kvantilej():
    x, d = _ryad_daty()
    a, b, c = pu().butstrap(x, d, n=500), pu().butstrap(x, d, n=500), pu().butstrap(x, d, n=500, seed=1)
    assert a == b and a != c
    for k in ("mediana_p5_50_95", "dolya_p5_50_95"):
        p5, p50, p95 = a[k]
        assert p5 <= p50 <= p95
    assert 0.0 <= a["dolya_p5_50_95"][0] and a["dolya_p5_50_95"][2] <= 1.0


def test_butstrap_blokami_kvartalami_a_ne_dnyami(monkeypatch):
    x, d = _ryad_daty()
    seen = []
    real = np.random.default_rng

    class Spy:
        def __init__(self, g):
            self.g = g

        def integers(self, lo, hi, size=None, **kw):
            seen.append((lo, hi, size))
            return self.g.integers(lo, hi, size=size, **kw)
    monkeypatch.setattr(np.random, "default_rng", lambda *a, **k: Spy(real(*a, **k)))
    pu().butstrap(x, d, n=50)
    assert seen == [(0, 16, (50, 16))]                    # индексы кварталов 0…15, выборка 16 с возвращением; не дни (high = T)


def test_butstrap_c_vozvratom_kvartaly_povtoryayutsya():
    x, d = _ryad_daty()
    r = pu().butstrap(x, d, n=2000)
    assert r["n"] == 2000 and r["seed"] == 20261028
    assert r["dolya_p5_50_95"][2] - r["dolya_p5_50_95"][0] > 0     # разброс > 0: выборки различаются, не копия 16 кварталов


def test_vybrat_2_iz_4_ne_beretsya_3_iz_4_beretsya():
    assert pu().vybrat(0.25, [0.1, 0.1, -0.1, -0.1]) is False
    assert pu().vybrat(0.25, [0.1, 0.1, 0.1, -0.1]) is True
    assert pu().vybrat(0.25, [0.1, 0.1, 0.0, -0.1]) is False               # ноль — не выигрыш года
    assert pu().vybrat(-0.25, [-0.1, -0.1, -0.1, 0.1]) is False


def test_vybrat_porog_nestrogij():
    assert pu().POROG == 0.2 and pu().MIN_GODOV == 3
    assert pu().vybrat(0.2, [0.1] * 4) is True
    assert pu().vybrat(0.2 - 1e-9, [0.1] * 4) is False


def _ex(gody_s_vyigryshem):
    """ex-ряд тикера: база — шум; вариант = база + сдвиг +0,01/день в перечисленных годах (SR года заведомо растёт)."""
    daty = np.arange(np.datetime64("2016-01-01"), np.datetime64("2020-01-01"))
    base = np.random.default_rng(5).normal(0, 0.01, len(daty))
    yr = daty.astype("datetime64[Y]").astype(int) + 1970
    return daty, base, base + 0.01 * np.isin(yr, gody_s_vyigryshem)


@pytest.mark.parametrize("gody,vzyat", [((2016, 2017), False), ((2016, 2017, 2018), True), ((2016, 2017, 2018, 2019), True)])
def test_tiker_popravka_uslovie_3_iz_4_let_end_to_end(gody, vzyat):
    daty, base, var = _ex(gody)

    def ev(pt):
        return (var if pt["vhod"] == 0.5 else base)[None, :]
    rows, tochki = pu().tiker_popravka({"vhod": 1.0, "tau": 1.0}, ["SBER"], daty, ev, [0.5], [1.0])
    assert (rows[0]["vybor"] != "kandidat") is vzyat and rows[0]["gody"] == len(gody)
    assert tochki == [dict(pt={"vhod": 0.5, "tau": 1.0}, istochnik="tiker")]    # точки в N с istochnik=tiker; сам кандидат не пересчитан
    assert rows[0]["vyigrysh"] > 0.2 or not vzyat


def test_tiker_popravka_bez_vyigrysha_usadka_k_kandidatu():
    daty, base, _ = _ex(())
    rows, _ = pu().tiker_popravka({"vhod": 1.0, "tau": 1.0}, ["SBER"], daty, lambda pt: base[None, :], [0.5, 1.0], [1.0, 3.0])
    assert rows[0]["vybor"] == "kandidat"
