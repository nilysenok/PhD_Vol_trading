"""Э1-в, режим plato (fastbt/sweep_plato.py): пик против плато, соседи в N, предел активности, неактивные параметры.
Синтетика: подставной `FakeSw` (поверхность цели задана функцией), без данных и оценщика. `test_*` — на коде проходит,
`test_mut_*` — с подложенной ошибкой та же проверка обязана не пройти."""
import pytest

from fastbt import sweep_drivers as sd
from fastbt import sweep_plato as pl
from fastbt import sweep_pts as pts
from fastbt.sweep_spec import pkey

PER, C0 = ("setup", "control"), 0.04
G9 = list(range(1, 10))


class FakeSw:
    """Как Sweep.evaluate: проба = точка на настройке при комиссии отбора -> ledger[pkey] = sr_d."""

    def __init__(self, f, base):
        self.f, self.base, self.ledger = f, base, {}

    def evaluate(self, params, period="setup", comm=None):
        p = {**self.base, **params}
        r = self.f(p)
        self.ledger[pkey(p)] = r["sharpe"] / 16
        return {**p, "sr_d": r["sharpe"] / 16, "by_year": {2018: r["sharpe"]}, **r}

    def trials_sr(self):
        return list(self.ledger.values())


def surface(p):
    """Пик 3,0 в (7,7) с соседями −1; плато 1,0 на a,b ∈ {1,2,3}; прочее 0. Сделок в пике 1/год, на плато 10."""
    ab = (p["a"], p["b"])
    peak = ab == (7, 7)
    near = abs(p["a"] - 7) + abs(p["b"] - 7) == 1
    s = 3.0 if peak else (-1.0 if near else (1.0 if p["a"] <= 3 and p["b"] <= 3 else 0.0))
    return dict(sharpe=s, sdelok_y=1.0 if peak else 10.0)


def make(f=surface):
    exp = dict(base=dict(a=2, b=2), grid=dict(a=G9, b=G9), uslovie={})
    sw = FakeSw(f, exp["base"])
    for ab in [(7, 7), (2, 2), (3, 3), (2, 3)]:
        sw.evaluate(dict(a=ab[0], b=ab[1]))
    return sw, exp, pts.make_fix(exp)


def pick(sw, exp, fix, min_sd=0.0):
    c = pl.candidates(sw, exp, PER, C0, min_sd, fix)
    n = pl.add_plateau(sw, exp, PER, C0, ["a", "b"], fix, c)
    return c, n


# ---- 1. пик не выбирается, плато выбирается ----
def best_by_plateau(c):
    return max(c, key=lambda x: x["plateau"])["p"]


def test_odinochnyi_pik_ne_vybiraetsya_plato_da():
    sw, exp, fix = make()
    c, _ = pick(sw, exp, fix)
    assert max(c, key=lambda x: x["obj"])["p"] == dict(a=7, b=7)            # по пику выиграл бы шум
    assert best_by_plateau(c) in ({"a": 2, "b": 2}, {"a": 3, "b": 3}, {"a": 2, "b": 3})


def test_mut_plato_ravno_piku(monkeypatch):
    monkeypatch.setattr(pl, "plateau_value", lambda obj, nb: obj)
    sw, exp, fix = make()
    c, _ = pick(sw, exp, fix)
    assert best_by_plateau(c) == dict(a=7, b=7)                              # мутация возвращает пик — тест выше упал бы


# ---- 2. соседи достраиваются и идут в N ----
def test_sosedi_dostraivayutsya_i_idut_v_N():
    sw, exp, fix = make()
    n0 = len(sw.ledger)
    c, n = pick(sw, exp, fix)
    assert len(c) == 4 and n > 4
    assert len(sw.ledger) > n0 and len(sw.trials_sr()) == len(sw.ledger)
    assert all(x["n_nb"] >= 3 and not x["edge"] for x in c)


def test_mut_sosedi_ne_dostraivayutsya(monkeypatch):
    monkeypatch.setattr(pl, "neighbours", lambda *a: [])
    sw, exp, fix = make()
    n0 = len(sw.ledger)
    _, n = pick(sw, exp, fix)
    assert n == 0 and len(sw.ledger) == n0                                   # N не вырос — тест выше упал бы


def test_povtornyi_zapusk_N_ne_rastet():
    sw, exp, fix = make()
    c, _ = pick(sw, exp, fix)
    n1 = len(sw.ledger)
    pl.add_plateau(sw, exp, PER, C0, ["a", "b"], fix, c)                     # те же кандидаты и соседи: пробы уже в ledger
    assert len(sw.ledger) == n1


# ---- 3. предел активности ----
def test_predel_aktivnosti_otsekaet_tochku_s_odnoi_sdelkoi():
    sw, exp, fix = make()
    base_sd = sw.evaluate(exp["base"])["sdelok_y"]
    c = pl.candidates(sw, exp, PER, C0, 0.5 * base_sd, fix)
    assert dict(a=7, b=7) not in [x["p"] for x in c] and len(c) == 3


def test_mut_predel_aktivnosti_ne_primenyaetsya(monkeypatch):
    monkeypatch.setattr(pl, "admissible", lambda res, min_sd: True)
    sw, exp, fix = make()
    c = pl.candidates(sw, exp, PER, C0, 5.0, fix)
    assert dict(a=7, b=7) in [x["p"] for x in c]                             # мутация пропустила пик — тест выше упал бы


def test_net_sdelok_nan_ne_dopustimo():
    assert not pl.admissible(dict(sdelok_y=float("nan")), 1.0)


# ---- 4. неактивный параметр не раздувает N ----
def exp_uslovnaya(mode0="off"):
    return dict(base=dict(a=2, mode=mode0, q=1, sl_mult=1.0, rm_edinitsa="atr14", k=1),
                grid=dict(a=G9, mode=["on", "off"], q=[1, 2, 3], sl_mult=[0.5, 1.0, 2.0], rm_edinitsa=["atr14", "sqrt_rv"], k=[1, 2, 3]),
                uslovie={"q": {"mode": "on"}, "k": {"rm_edinitsa": "sqrt_rv"}})


def test_uslovnyi_parametr_pri_nevypolnennom_uslovii_dayet_odnogo_soseda():
    exp = exp_uslovnaya()
    fix = pts.make_fix(exp)
    nb = pl.neighbours(exp, exp["base"], ["q", "mode"], fix)
    assert [x["mode"] for x in nb] == ["on"]                                  # q ±1 при mode=off свёлся к базе


def test_mut_fix_ne_svodit_k_baze():
    exp = exp_uslovnaya()
    nb = pl.neighbours(exp, exp["base"], ["q", "mode"], lambda p: dict(p))
    assert len(nb) > 1                                                        # без fix q даёт дубли — тест выше упал бы


def test_atr_klyuchi_pri_sqrt_rv_svedeny_k_baze():
    exp = exp_uslovnaya()
    p = {**exp["base"], "rm_edinitsa": "sqrt_rv"}
    nb = pl.neighbours(exp, p, ["sl_mult", "rm_edinitsa"], pts.make_fix(exp))
    assert [x["rm_edinitsa"] for x in nb] == ["atr14"]                        # sl_mult под √RV не доходит до расчёта


def test_mut_sqrt_inert_ne_svoditsya(monkeypatch):
    monkeypatch.setattr(pts, "INERT_SQRT", ())
    exp = exp_uslovnaya()
    p = {**exp["base"], "rm_edinitsa": "sqrt_rv"}
    nb = pl.neighbours(exp, p, ["sl_mult", "rm_edinitsa"], pts.make_fix(exp))
    assert len(nb) == 3                                                       # мутация: sl_mult раздувает N


def test_kanon_sovpadaet_so_staroi_proboi_ledger():
    """Старая проба с неканоническим sl_mult под √RV: сосед по канону берётся из ledger, N не растёт."""
    exp = exp_uslovnaya()
    fix = pts.make_fix(exp)
    sw = FakeSw(lambda p: dict(sharpe=1.0, sdelok_y=9.0), exp["base"])
    old = {**exp["base"], "rm_edinitsa": "sqrt_rv", "sl_mult": 2.0}
    sw.evaluate(old)
    idx = pl.canon_index(sw, exp, fix)
    q = fix({**exp["base"], "rm_edinitsa": "sqrt_rv"})
    assert idx[pkey(q)] == old
    sw.evaluate(idx[pkey(q)])
    assert len(sw.ledger) == 1
