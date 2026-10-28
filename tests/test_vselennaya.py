"""К-а2: вселенная U_d. Синтетика; четыре мутации (mutations.md № 19 и др.) обязаны ловиться."""
import pandas as pd
import pytest

from fastbt import vselennaya as V

SOSTAV = (
    "date_eff,known_from,n,tickers\n"
    "2015-01-05,2015-01-05,2,A B\n"
    "2016-03-17,2016-03-02,3,A B C\n"
    "2017-06-16,2017-06-01,2,B C\n"
)
# оборот по годам: (2015, 2016, 2017)
OB = {"A": (9, 1, 5), "B": (5, 2, 4), "C": (1, 8, 3), "X": (100, 100, 100), "Z": (0, 0, 0)}


@pytest.fixture()
def env(tmp_path):
    s = tmp_path / "sostav.csv"
    s.write_text(SOSTAV)
    (tmp_path / "daily_value").mkdir()
    for tk, vals in OB.items():
        rows = [f"{2015 + i}-05-05,{v}" for i, v in enumerate(vals) if v > 0]
        (tmp_path / "daily_value" / f"{tk}.csv").write_text("date,value\n" + "\n".join(rows) + "\n")
    V._sostav_table.cache_clear()
    V.oborot_goda.cache_clear()
    return s, tmp_path


def u(d, env, k=40):
    return V.U(d, k, sostav_path=env[0], root=env[1])


def test_sostav_po_date_eff(env):
    s = env[0]
    assert V.sostav("2016-03-16", s) == {"A", "B"}
    assert V.sostav("2016-03-17", s) == {"A", "B", "C"}
    assert V.sostav("2014-12-31", s) == frozenset()


def test_u_rank_proshlogo_goda_i_topk(env):
    assert u("2016-06-01", env, k=1) == {"A"}  # 2015: A=9 > B=5
    assert u("2017-07-03", env, k=1) == {"C"}  # 2016: C=8
    assert u("2017-07-03", env) == {"B", "C"}


def test_tiker_vne_sostava_ne_popadaet(env):
    assert "X" not in u("2016-06-01", env)
    assert "X" not in u("2016-06-01", env, k=10)


def test_bez_oborota_ne_v_U_i_v_otchet(env):
    s, r = env
    s2 = r / "s2.csv"
    s2.write_text("date_eff,known_from,n,tickers\n2015-01-05,2015-01-05,3,A B Z\n")
    assert V.U("2016-06-01", 40, sostav_path=s2, root=r) == {"A", "B"}
    assert V.bez_oborota("2016-06-01", s2, r) == {"Z"}


def test_otkaz_ot_2022(env):
    with pytest.raises(ValueError):
        V.sostav("2022-01-03", env[0])
    with pytest.raises(ValueError):
        V.U("2022-01-03", sostav_path=env[0], root=env[1])
    with pytest.raises(ValueError):
        V.oborot_goda("A", 2022, str(env[1]))


def test_digest_menyaetsya_pri_podmene_vkhoda(env):
    s, r = env
    d0 = V.digest(s, r)
    (r / "daily_value" / "A.csv").write_text("date,value\n2015-05-05,10\n")
    assert V.digest(s, r) != d0


# --- мутации: неверная реализация обязана давать иной ответ на синтетике (т.е. тест её ловит) ---
def _u_mut(d, env, *, year_shift=0, col="date_eff", fixed_sostav=None, no_filter=False, k=40):
    s, r = env
    ts = pd.Timestamp(d)
    t = pd.read_csv(s)
    cur = frozenset()
    if fixed_sostav is not None:
        cur = fixed_sostav
    else:
        for row in t.sort_values(col).itertuples():
            if pd.Timestamp(getattr(row, col)) <= ts:
                cur = frozenset(row.tickers.split())
    pool = frozenset(OB) if no_filter else cur
    y = ts.year + year_shift
    rows = sorted(((V.oborot_goda(x, y, str(r)), x) for x in pool), key=lambda z: (-z[0], z[1]))
    return frozenset(x for v, x in rows[:k] if v > 0)


def test_mut19_oborot_tekushchego_goda_lovitsya(env):
    assert _u_mut("2016-06-01", env, year_shift=0, k=1) != u("2016-06-01", env, k=1)


def test_mut_known_from_vmesto_date_eff_lovitsya(env):
    d = "2016-03-10"  # между known_from (03-02) и date_eff (03-17)
    assert V.sostav(d, env[0]) == {"A", "B"}
    assert _u_mut(d, env, year_shift=-1, col="known_from") != u(d, env)


def test_mut_sostav_2021_v_2016_lovitsya(env):
    last = frozenset({"B", "C"})
    assert _u_mut("2016-06-01", env, year_shift=-1, fixed_sostav=last) != u("2016-06-01", env)


def test_mut_tiker_vne_sostava_lovitsya(env):
    assert _u_mut("2016-06-01", env, year_shift=-1, no_filter=True, k=1) != u("2016-06-01", env, k=1)
