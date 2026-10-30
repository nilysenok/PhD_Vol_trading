"""К-а2: вселенная U_d. Синтетика; четыре мутации (monkeypatch по fastbt.vselennaya) обязаны ронять штатные ассерты."""
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
    assert V.sostav("2016-03-10", s) == {"A", "B"}  # между known_from (03-02) и date_eff (03-17)
    assert V.sostav("2016-03-16", s) == {"A", "B"}
    assert V.sostav("2016-03-17", s) == {"A", "B", "C"}
    assert V.sostav("2014-12-31", s) == frozenset()


def test_u_rank_proshlogo_goda_i_topk(env):
    assert u("2016-06-01", env, k=1) == {"A"}  # 2015: A=9 > B=5
    assert u("2017-07-03", env, k=1) == {"C"}  # 2016: C=8
    assert u("2017-07-03", env) == {"B", "C"}


def test_sostav_2016_ne_2021(env):
    assert u("2016-06-01", env) == {"A", "B", "C"}  # состав 2016, не поздний {B, C}


def test_tiker_vne_sostava_ne_popadaet(env):
    assert "X" not in u("2016-06-01", env)
    assert "X" not in u("2016-06-01", env, k=10)


def test_bez_oborota_ne_v_U_i_v_otchet(env):
    s, r = env
    s2 = r / "s2.csv"
    s2.write_text("date_eff,known_from,n,tickers\n2015-01-05,2015-01-05,3,A B Z\n")
    assert V.U("2016-06-01", 40, sostav_path=s2, root=r) == {"A", "B"}
    assert V.bez_oborota("2016-06-01", s2, r) == {"Z"}


def test_otkaz_ot_2022(env, tmp_path, monkeypatch):
    """Без отметки снятого замка ≥ 2022 — отказ именно из-за замка (а не из-за устаревшего состава/пустого года)."""
    from fastbt import period_v2
    monkeypatch.setattr(period_v2, "OTMETKA", tmp_path / "net.json")
    for f in (lambda: V.sostav("2022-01-03", env[0]), lambda: V.U("2022-01-03", sostav_path=env[0], root=env[1]),
              lambda: V.oborot_goda("A", 2022, str(env[1]))):
        with pytest.raises(ValueError, match="нет отметки"):
            f()


def test_s_otmetkoy_granitsa_goda_wf(env, tmp_path, monkeypatch):
    """С отметкой: 2026 (последний год wf) читается, 2027 — отказ; год 2022 без строк оборота — отказ (Д4), не 0.0."""
    from fastbt import period_v2
    (tmp_path / "otm.json").write_text("{}")
    monkeypatch.setattr(period_v2, "OTMETKA", tmp_path / "otm.json")
    (env[1] / "daily_value" / "A.csv").write_text("date,value\n2026-02-27,7\n")
    V.oborot_goda.cache_clear()
    assert V.oborot_goda("A", 2026, str(env[1])) == 7.0
    with pytest.raises(ValueError, match="позже конца"):
        V.oborot_goda("A", 2027, str(env[1]))
    with pytest.raises(ValueError, match="нет строк года"):
        V.oborot_goda("A", 2022, str(env[1]))


def test_digest_menyaetsya_pri_podmene_vkhoda(env):
    s, r = env
    d0 = V.digest(s, r)
    (r / "daily_value" / "A.csv").write_text("date,value\n2015-05-05,10\n")
    assert V.digest(s, r) != d0


# --- мутации: подменяем функцию модуля; штатные тесты выше обязаны упасть ---
def _shtatnye(env):
    test_sostav_po_date_eff(env)
    test_u_rank_proshlogo_goda_i_topk(env)
    test_sostav_2016_ne_2021(env)
    test_tiker_vne_sostava_ne_popadaet(env)


def test_shtatnye_prokhodyat_bez_mutatsii(env):
    _shtatnye(env)


def _lovitsya(env):
    with pytest.raises(AssertionError):
        _shtatnye(env)


def test_mut19_oborot_tekushchego_goda_lovitsya(env, monkeypatch):
    monkeypatch.setattr(V, "rank_year", lambda d: V._year(d))
    _lovitsya(env)


def test_mut_known_from_vmesto_date_eff_lovitsya(env, monkeypatch):
    monkeypatch.setattr(V, "SOSTAV_COL", "known_from")
    _lovitsya(env)


def test_mut_sostav_2021_v_2016_lovitsya(env, monkeypatch):
    monkeypatch.setattr(V, "sostav", lambda d, sostav_path=None: frozenset({"B", "C"}))
    _lovitsya(env)


def test_mut_tiker_vne_sostava_lovitsya(env, monkeypatch):
    monkeypatch.setattr(V, "sostav", lambda d, sostav_path=None: frozenset(OB))
    _lovitsya(env)
