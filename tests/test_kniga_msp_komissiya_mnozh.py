"""Комиссия книги МСП (#857, Ж-149, #861): акции дневные и оплачены в excess пары, не оплачен только хедж MIX. Синтетика."""
import numpy as np
import pytest

from fastbt import kniga_msp as km
from fastbt import tf_os_v5

C = 0.0004


def test_stavka_ta_zhe_chto_v_dvizhke():
    assert km.KOMISSIYA == tf_os_v5.KOMISSIYA == C


def test_dnevnye_stavki_excess_pary_uzhe_platit_polnyj_oborot():
    """Известный ответ: позиция 1 в день 1 и 2 в день 2 (множитель 1 -> 2), по два бара в день. Оборот бара:
    |Δpos| и закрытие на последнем баре -> [1, 1, 2, 2], издержки = C·6: каждый день вход и выход при СВОЁМ размере,
    ночью позиции нет (день 2 не зависит от дня 1). Значит, смена множителя отдельной сделки не порождает."""
    o = np.full(4, 100.0)
    den = np.array([0, 0, 1, 1])
    _, cost, turn = tf_os_v5.dohod(np.array([1, 1, 2, 2]), o, o, den)
    assert turn.tolist() == [1, 1, 2, 2] and cost.sum() == pytest.approx(C * 6)
    # линейность по размеру: excess·w — веса книги масштабируют и доходность, и издержки пары одинаково
    _, cost3, _ = tf_os_v5.dohod(np.array([3, 3, 3, 3]), o, o, den)
    _, cost1, _ = tf_os_v5.dohod(np.array([1, 1, 1, 1]), o, o, den)
    assert cost3.sum() == pytest.approx(3 * cost1.sum())


def test_komissiya_oborota_izvestnyj_otvet():
    """C·Σ|Δw| руками: |0,5−0|+|−1−(−0,5)|+|0,2−0,2| = 1,0."""
    assert km.komissiya_oborota([0.5, -1.0, 0.2], [0.0, -0.5, 0.2]) == pytest.approx(C * 1.0)
    assert km.komissiya_oborota([-0.4], [-0.1]) == pytest.approx(C * 0.3)


def test_mutaciya_modul_w_vmesto_delta_lovitsya():
    """Неизменная позиция оборота не даёт; мутант Σ|w| списал бы C·1,5."""
    w = [0.5, -1.0]
    assert km.komissiya_oborota(w, w) == 0.0
    assert km.komissiya_oborota(w, w) != pytest.approx(C * np.abs(w).sum())
    assert km.komissiya_oborota([1.0], [0.0]) == pytest.approx(C)      # вход с нуля платит |w|, а не 0


@pytest.mark.xfail(strict=True, reason="msp_v5: ex_kniga = u @ R линейно по знаку u; комиссия пары, зашитая в R "
                   "на единицу лонга, у шорта (u<0) даёт ВОЗВРАТ комиссии; значит «комиссия через excess·w» верна только "
                   "если вызывающий строит R с учётом знака — builder dohod для msp_v5 в репозитории не найден")
def test_msp_v5_den_tolko_mnozhitel_spisyvaet_komissiyu_pary_cherez_excess():
    """Тест владельца (Ж-149) через msp_v5: каждый день позиция пары открывается и закрывается (`tf_os_v5.dohod`,
    оборот 2·|pos|), excess пары net. Ожидание: netto-excess книги = брутто − C·оборот·|w| при новом весе w = u (m меняется
    между днями, сигналов не меняется): комиссия списана ровно один раз, через excess·w."""
    import pandas as pd

    from fastbt import msp_v5
    T, tik = 60, ["A", "B"]
    idx = pd.bdate_range("2016-01-04", periods=T)
    rng = np.random.default_rng(11)
    o = 100.0 + np.zeros((T, 2))
    c = o * (1 + 0.01 * rng.standard_normal((T, 2)))
    gross, net = np.zeros((T, 2)), np.zeros((T, 2))
    for j, p in enumerate((1, 1)):                     # R на единицу лонга (знак даёт u); один бар в день -> закрытие
        g, cost, turn = tf_os_v5.dohod(np.full(T, p), o[:, j], c[:, j], np.arange(T))
        assert (turn == 2).all()
        gross[:, j], net[:, j] = g, g - cost
    f = lambda v: pd.DataFrame(v, index=idx, columns=tik)      # noqa: E731
    sig = {"S1": {"storona": f(np.array([[1.0, -1.0]] * T)), "sila": f(rng.uniform(0, 1, (T, 2))),
                  "g": f(np.ones((T, 2)))}}
    sigma, sektor, im = f(np.full((T, 2), 0.02)), {"A": "X", "B": "Y"}, pd.Series(0.0, index=idx)
    r = msp_v5.progon(sig, sigma, sektor, f(net), im, 0.0, k=2, rho=0.001)
    u = r.u.to_numpy()[:, [0, 1]]
    razn = r.ex_kniga.to_numpy() - (u * gross).sum(axis=1)
    assert np.allclose(razn, -C * 2 * np.abs(u).sum(axis=1))
    m = np.abs(u).sum(axis=1)
    assert (np.abs(np.diff(m[30:])) > 1e-9).any()          # множитель действительно менялся между днями


def _ist(T=40):
    return {"ex_sk": np.zeros((T, 2)), "ex_port": np.zeros(T), "ex_kniga": np.zeros(T), "imoex": np.zeros(T),
            "kvartal": np.zeros(T, int)}


ARGS = (np.array([5, 4, 3, -2, -1, 0.5]), np.full(6, 0.02), np.arange(6), np.array([0, 0, 0, 1, 1, 1]))


def test_kniga_dnya_mnozhitel_akcij_ne_platit():
    """Множитель 1 -> >1 меняет веса акций, комиссии книги нет (она уже в excess·w)."""
    ist = _ist()
    a = km.kniga_dnya(20, *ARGS, ist, k=6, rho=0.001)
    ist["ex_port"][:21] = np.where(np.arange(21) % 2, 1, -1) * 0.05 / np.sqrt(252)
    b = km.kniga_dnya(21, *ARGS, ist, k=6, rho=0.001)
    assert b.m != a.m and not np.allclose(a.u, b.u)
    assert b.komissiya == 0.0


def test_kniga_dnya_hedzh_platit_po_izmeneniyu_bety():
    """β книги меняется между днями (бета по 63 строкам до i-1) -> хедж меняется -> C·|Δ хеджа|."""
    rng = np.random.default_rng(5)
    T = 90
    ist = _ist(T)
    ist["imoex"] = rng.normal(0, 0.01, T)
    ist["ex_kniga"] = 1.5 * ist["imoex"] + rng.normal(0, 0.002, T)
    ist["ex_kniga"][:64] = 0.5 * ist["imoex"][:64]
    a = km.kniga_dnya(70, *ARGS, ist, k=6)
    b = km.kniga_dnya(71, *ARGS, ist, k=6)
    assert a.hedzh != b.hedzh
    assert b.komissiya == pytest.approx(C * abs(b.hedzh - a.hedzh)) and b.komissiya > 0


def test_mutaciya_komissiya_akcij_po_delta_ves_lovitsya(monkeypatch):
    """Мутант: комиссия по |Δ итогового веса акций| (двойной учёт) — тест выше обязан упасть (комиссия ≠ 0)."""
    ist = _ist()
    a = km.kniga_dnya(20, *ARGS, ist, k=6, rho=0.001)
    ist["ex_port"][:21] = np.where(np.arange(21) % 2, 1, -1) * 0.05 / np.sqrt(252)
    b = km.kniga_dnya(21, *ARGS, ist, k=6, rho=0.001)
    mutant = C * np.abs(b.u - a.u).sum()
    assert mutant > 0 and b.komissiya == 0.0
