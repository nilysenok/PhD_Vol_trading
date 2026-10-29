"""Р-5 (29.10): σ̂ h22 по Д1 — максимум приращений на k = −1. Расследование: причина — отставание на день (гипотеза «а»), не утечка,
не ошибка скрипта проверки. Код и данные не менялись (pravki.md п. 12 — правка отдельным PR); сам ряд — xfail(strict).
Выводы (данные ≤ 2021, медиана по 17 тикерам, scripts/proverka_sdviga_wf.py):
- вес Δln RV_d / Δln RV_{d−1} в регрессии Δln σ̂²_d (3 лага): h1 0,227/0,096; h5 0,066/0,066; h22 0,017/0,042 — чем длиннее
  горизонт, тем позже информация (h22 впитывает RV_{d−1}, а RV_d почти нет); цель «трейлинг-22 среднее RV» даёт то же: k=−1 0,355, k=0 0,175;
- h22 к h1 по приращениям: k=−1 0,237, k=0 0,189 (h22 отстаёт от h1); k=+1 отрицателен у всех — заглядывания вперёд нет;
- fastbt_spec.md:59 — «прогноз в строке t, видимо, построен по данным до t−1; возможен лишний лаг в один день (18.10)»."""
import numpy as np
import pandas as pd
import pytest

from scripts import proverka_sdviga_wf as P

TAB = P.OUT


def _ryad(name):
    t = pd.read_csv(TAB, comment="#")
    return t[(t["ryad"] == name) & (t["tip"] == "prirashcheniya")].iloc[0]


def _slow(n=2000, seed=1, lag=0, lead=0, w=22):
    """Медленный ряд: логарифм скользящего среднего шумной RV; lag — информация только до d−lag, lead — заглядывание на lead."""
    rng = np.random.default_rng(seed)
    rv = pd.Series(np.exp(rng.normal(size=n)))
    x = np.log(rv.rolling(w).mean()).shift(lag - lead)
    return x.to_frame("T"), np.log(rv).to_frame("T")


def test_skript_pokazyvaet_otstavanie_v_minus_odin():
    """Гипотеза «в» отброшена: на синтетике со знанием до d−1 пик приращений на k=−1, с утечкой d+1 — на k=+1, без сдвига — на k=0."""
    for lag, lead, k in ((1, 0, -1), (0, 1, 1)):
        x, y = _slow(lag=lag, lead=lead)
        p = P.profil(x, y, diff=True)
        assert max(p, key=lambda j: abs(p[j])) == k
        assert not P.verdikt(p)[0]
    x, y = _slow()
    p = P.profil(x, y, diff=True)
    assert max(p, key=lambda j: abs(p[j])) == 0 and P.verdikt(p)[0]


def test_h22_pik_na_minus_odin_utechki_net():
    r = _ryad("sigma_h22")
    prof = {k: r[f"corr_k{k:+d}"] for k in P.KS}
    assert max(prof, key=lambda k: abs(prof[k])) == -1
    assert prof[1] < 0 and abs(prof[-1]) > 2 * abs(prof[0])      # на +1 отрицательная (возврат к среднему), не утечка


def test_h1_h5_pik_na_nule_h5_с_малым_запасом():
    assert _ryad("sigma_h1")["ok"]
    r = _ryad("sigma_h5")
    assert r["ok"] and r["corr_k+0"] - r["corr_k-1"] < 0.02       # запас h5 мал: тоже тянется к отставанию, потолок проверки


@pytest.mark.xfail(strict=True, reason="σ̂ h22 отстаёт на день (инфо до d−1) — Д1 не проходит; правка (сдвиг/пересборка) — отдельный PR")
def test_h22_proxodit_d1():
    assert bool(_ryad("sigma_h22")["ok"])
