"""Тестировщик PR #891: подложенные ошибки прогона A′-расширенной (деньги). Синтетика, без данных.
Мутации: позиции центра без знака; хедж со сдвигом на день; паритет без нормировки; паритет не сошёлся — молчаливые веса.
Плюс характеристика рассогласования: IMOEX/MIX open→open против результата close→close (дефект хеджа, оговорка отчёта)."""
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fastbt import a_shtrih_v7 as A, a_shtrih_v7_kniga as K, a_shtrih_v7_vhody as V

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "analysis"))
import a_shtrih_v7 as drv  # noqa: E402

S = np.load(Path(__file__).with_name("fixtures_a_shtrih_v7_lw_stall.npy"))
T = 3000
BETA, SIG_IN, SIG_ON = 1.5, 0.007, 0.004


def mutirovat(mod, staroe: str, novoe: str) -> dict:
    src = Path(mod.__file__).read_text(encoding="utf-8")
    assert staroe in src
    nm = "mut_" + mod.__name__.split(".")[-1]
    m = types.ModuleType(nm)
    sys.modules[nm] = m                                                # dataclass ищет свой модуль в sys.modules
    try:
        exec(compile(src.replace(staroe, novoe), "mut", "exec"), m.__dict__)   # noqa: S102 — код репозитория с подложенной ошибкой
    finally:
        sys.modules.pop(nm, None)
    return m.__dict__


def rynok(seed=3):
    """cc[t] = close t−1 → close t; oo[t] = open t → open t+1. Книга держит позицию через ночь: X = β·cc."""
    rng = np.random.default_rng(seed)
    on, intra = rng.normal(0, SIG_ON, T + 1), rng.normal(0, SIG_IN, T)
    return on[:T] + intra, intra + on[1:]


def kniga(ns_or_mod, imoex, mix, cc):
    kr = ns_or_mod["kniga_ryada"] if isinstance(ns_or_mod, dict) else ns_or_mod.kniga_ryada
    X = (BETA * cc)[None, None, :]
    return kr(X, np.ones_like(X), np.array(["a"]), imoex, mix, None, vesa="vykl", sektor_netto=False, tsel_vol=False, vykl=False)


def ostatok(res, cc):
    """Доля дисперсии книги, оставшаяся после хеджа (прогрев β — 63 дня не считаются)."""
    return float(res.ryad[70:].var() / (BETA * cc)[70:].var())


def test_hedzh_vyrovnen_ostatok_nol():
    cc, _ = rynok()
    assert ostatok(kniga(K, cc, cc, cc), cc) < 1e-3


def test_mut_hedzh_so_sdvigom_na_den_lovitsya():
    cc, _ = rynok()
    ns = mutirovat(K, "h * np.where(mix_nan, 0.0, mix)", "h * np.roll(np.where(mix_nan, 0.0, mix), 1)")
    assert ostatok(kniga(ns, cc, cc, cc), cc) > 0.5               # check «остаток < 1e-3» мутацию ловит


def test_rassoglasovanie_open_open_protiv_close_close_hedzh_ne_hedzhiruet():
    """ДЕФЕКТ (оговорка отчёта #891): книга close→close, IMOEX/MIX open→open: β занижена, остаток ≈ 43 % дисперсии
    (аналитически 1 − cov²/(var·var)); при выровненных рядах — 0. Хедж «почти ничего» — следствие входа, не рынка."""
    cc, oo = rynok()
    ost = ostatok(kniga(K, oo, oo, cc), cc)
    teor = 1 - SIG_IN**4 / (SIG_IN**2 + SIG_ON**2) ** 2          # = 1 − cov²/(var·var) при равных дисперсиях cc и oo
    assert abs(ost - teor) < 0.08 and ost > 0.3, (ost, teor)
    assert ost > 100 * ostatok(kniga(K, cc, cc, cc), cc)


def dvh(znak_fn=None):
    n, N = 12, 3
    daty = pd.bdate_range("2016-03-01", periods=n)
    rng = np.random.default_rng(1)
    c = pd.DataFrame(100 * np.cumprod(1 + rng.normal(0, 0.01, (n, N)), axis=0), index=daty, columns=list("ABC"))
    D = dict(daty=daty, tk=list("ABC"), o=c.shift(1).bfill(), cr=c, div=pd.DataFrame(0.0, index=daty, columns=list("ABC")),
             stavka=np.full(n, 0.07))
    s = np.zeros((n, N), np.int8)
    s[1, 0], s[3, 1] = 1, -1
    X, P = V.pary(D, s, vhod="d1", tau=2, sost=False)
    return dict(X=X[None], poz=P[None], sektora=np.array(list("xyz")), imoex=np.zeros(n), mix=np.zeros(n), ks=np.zeros(n),
                daty=daty.values.astype("datetime64[D]"), znak="znakovyj", istochnik="ryady")


def test_poziciya_so_znakom_prohodit_vhod():
    drv.proverit_vhod(dvh())


def test_mut_poziciya_bez_znaka_lovitsya_vhodom(monkeypatch):
    orig = V._znak_poz
    monkeypatch.setattr(V, "_znak_poz", lambda rs, n: np.abs(orig(rs, n)))
    with pytest.raises(ValueError, match="шорт"):
        drv.proverit_vhod(dvh())


def test_mut_paritet_bez_normirovki_lovitsya():
    ns = mutirovat(A, "w = y / y.sum()", "w = y")
    w = ns["paritet"](S)
    assert abs(w.sum() - 1.0) > 1e-3                              # а ValueError невязки не срабатывает: равенство вкладов цело
    assert abs(A.paritet(S).sum() - 1.0) < 1e-12


def test_paritet_ne_soshyolsya_otkaz_a_ne_tihie_vesa():
    with pytest.raises(ValueError, match="не сошёлся"):
        A.paritet(S, maxit=1)
