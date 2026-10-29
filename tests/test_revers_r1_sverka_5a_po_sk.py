"""Реверс р.1, Ж-130: срез шага «утечка» по стратегиям S1–S6 (scripts/revers/r1_sverka_5a_po_sk.py). Синтетика, без данных."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R / "scripts" / "revers"))
import r1_sverka_5a_po_sk as K  # noqa: E402

DATES = pd.bdate_range("2022-01-03", "2024-12-30")
S2, S3 = K.SK[1], K.SK[2]
OSTALNYE = tuple(k for k in K.SK if k != S2)


def _pos(seed: int, strategii=K.SK) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    k = pd.MultiIndex.from_product([DATES, ["T1", "T2", "T3"], strategii], names=["date", "ticker", "strategy"]).to_frame(index=False)
    k["position"] = rng.choice([-1.0, 0.0, 1.0], len(k))
    k["daily_gross_return"] = rng.normal(0.0005, 0.01, len(k)) * np.abs(k["position"])
    return k


def _sloy():
    return K.sloy_msp(pd.Series(0.08, index=DATES))


def _izmenit(p: pd.DataFrame, strategii, seed: int) -> pd.DataFrame:
    """P1: те же ряды, но у перечисленных стратегий позиции и доходности другие (утечка меняет только их)."""
    q, m = p.copy(), p["strategy"].isin(strategii)
    n = _pos(seed)
    q.loc[m, ["position", "daily_gross_return"]] = n.loc[m, ["position", "daily_gross_return"]].to_numpy()
    return q


def test_smeshat_beryot_tolko_odnu_strategiyu_iz_p1():
    p0, p1 = _pos(1), _pos(2)
    m = K.smeshat(p0, p1, S3).sort_values(["date", "ticker", "strategy"]).reset_index(drop=True)
    for k in K.SK:
        ref = (p1 if k == S3 else p0)
        e = ref[ref["strategy"] == k].sort_values(["date", "ticker"])["position"].to_numpy()
        assert (m[m["strategy"] == k]["position"].to_numpy() == e).all()
    assert len(m) == len(p0)


def test_izmenena_odna_sk_vklad_tolko_eyo_i_raven_polnomu_shagu():
    p0 = _pos(1)
    p1 = _izmenit(p0, [S2], 7)
    t = K.vklady("A", p0, p1, _sloy())
    v = t[t["period"] == "ves"].set_index("strategiya")
    pol = v.loc["полный шаг P1−P0", "delta_sk"]
    assert abs(pol) > 1e-3
    assert abs(v.loc[S2, "vklad_pryamoj"] - pol) < 1e-12 and abs(v.loc[S2, "vklad_obratnyj"] - pol) < 1e-12
    for k in set(K.SK) - {S2}:
        assert v.loc[k, "vklad_pryamoj"] == 0 and v.loc[k, "vklad_obratnyj"] == 0 and v.loc[k, "delta_sk"] == 0
    assert abs(v.loc["Σ − полный (неаддитивность)", "vklad_pryamoj"]) < 1e-12


def test_podmena_vseh_sk_daet_p1_a_summa_vkladov_ne_ravna_polnomu():
    p0 = _pos(1)
    p1 = _izmenit(p0, list(K.SK), 9)
    s = _sloy()
    cur = p0
    for k in K.SK:
        cur = K.smeshat(cur, p1, k)
    assert all(abs(s(cur)[q] - s(p1)[q]) < 1e-12 for q in K.PER)
    v = K.vklady("C", p0, p1, s)
    ne = v[(v["strategiya"] == "Σ − полный (неаддитивность)") & (v["period"] == "ves")]["vklad_pryamoj"].iloc[0]
    assert abs(ne) > 1e-6                              # Шарп — отношение, EW по парам: вклады не складываются
    assert set(v["period"]) == {"ves", "2022", "2023", "2024"}


def test_mutaciya_smeshat_beret_vse_krome_k_lovitsya():
    p0 = _pos(1)
    p1 = _izmenit(p0, [S2], 7)
    plokho = lambda a, b, k: pd.concat([a[a["strategy"] == k], b[b["strategy"] != k]], ignore_index=True)    # noqa: E731
    v = K.vklady("A", p0, p1, _sloy(), smesh=plokho)
    v = v[v["period"] == "ves"].set_index("strategiya")
    pol = v.loc["полный шаг P1−P0", "delta_sk"]
    # на правильном коде вклады S1,S3–S6 нулевые, а S2 = полный шаг; мутация ломает хотя бы одно
    assert not (abs(v.loc[S2, "vklad_pryamoj"] - pol) < 1e-12 and all(v.loc[k, "vklad_pryamoj"] == 0 for k in OSTALNYE))


def test_mutaciya_znak_vklada_lovitsya():
    p0 = _pos(1)
    p1 = _izmenit(p0, [S2], 7)
    v = K.vklady("A", p0, p1, _sloy())
    v = v[v["period"] == "ves"].set_index("strategiya")
    assert np.sign(v.loc[S2, "vklad_pryamoj"]) == np.sign(v.loc["полный шаг P1−P0", "delta_sk"])        # перепутанный знак упал бы
