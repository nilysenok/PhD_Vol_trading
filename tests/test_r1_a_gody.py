"""R-а по годам: сумма по годам = полный период, годы ровно периода, период в воркере, мутация ddof=0 (синтетика, без данных проекта)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts" / "analysis"), str(ROOT / "scripts" / "revers")]
import r1_a_gody as G  # noqa: E402
from fastbt import metriki_e as me  # noqa: E402


def ryad(per: str, seed=0):
    dates = np.concatenate([np.arange(f"{y}-01-01", f"{y + 1}-01-01", dtype="datetime64[D]")[:120] for y in G.GODY[per]])
    rng = np.random.default_rng(seed)
    n = len(dates)
    return dict(x=rng.normal(0.0005, 0.01, n), baza=rng.uniform(0, 1, n), delo=rng.random(n) > 0.2, dates=dates)


def proverit_sharpe(df: pd.DataFrame, a: dict, per: str) -> None:
    """Эталон: Sharpe года = mean/std(ddof=1)·√252 независимо от кода модуля."""
    god = G.gody_ryada(a["dates"])
    for _, r in df[df.stroka.str.startswith("по году")].iterrows():
        v = (a["x"] - r.plata_zaem * a["baza"])[god == int(r.stroka[-4:])]
        assert r.sharpe == pytest.approx(v.mean() / v.std(ddof=1) * np.sqrt(252), rel=1e-12)


@pytest.mark.parametrize("per", ["train4", "test1"])
def test_summa_i_gody(per):
    a = ryad(per)
    df = G.tablica_iz({"yakor": a, "strogaya": ryad(per, 1)}, per)
    for imya in ("yakor", "strogaya"):
        for plata in (G.AS.PLATA, 0.0):
            s = df[(df.portfel == imya) & (df.plata_zaem == plata)]
            assert list(s.stroka) == ["(а) полные"] + [f"по году {y}" for y in G.GODY[per]]
            assert s.dni.iloc[1:].sum() == s.dni.iloc[0]
    proverit_sharpe(df[df.portfel == "yakor"], a, per)


def test_gody_ne_periodu_otkaz():
    a = ryad("test1")
    with pytest.raises(ValueError, match="годы"):
        G.kontrol_summy(a["x"], a["dates"], "train4")
    with pytest.raises(ValueError, match="годы"):
        G.kontrol_summy(a["x"][:120], a["dates"][:120], "test1")          # нет 2023, 2024


def test_summa_lovit_porchu():
    a = ryad("test1")
    G.kontrol_summy(a["x"], a["dates"], "test1")
    assert abs(sum(a["x"][G.gody_ryada(a["dates"]) == y].sum() for y in G.GODY["test1"]) - a["x"].sum()) < 1e-9


def test_period_v_voroner():
    z = G.zadaniya("test1")
    assert [p for _, p in z] == ["test1"] * len(G.AS.SIDS) and [s for s, _ in z] == list(G.AS.SIDS)


def test_mutaciya_ddof0_lovitsya(monkeypatch):
    a = ryad("test1")
    G.tablica_iz({"yakor": a, "strogaya": a}, "test1")
    monkeypatch.setattr(me, "sr_dnevnoy", lambda r: float(np.mean(r) / np.std(r, ddof=0)))
    df = G.tablica_iz({"yakor": a, "strogaya": a}, "test1")
    with pytest.raises(AssertionError):
        proverit_sharpe(df, a, "test1")


def test_sverka_pasport_lovit_rashozhdenie():
    a = ryad("train4")
    df = G.tablica_iz({"yakor": a, "strogaya": a}, "train4")
    pas = df[df.stroka.str.startswith("по году")].assign(ryad="A", variant=lambda d: d.portfel, god=lambda d: d.stroka.str[-4:])
    assert G.sverka_pasport(df, pas) == 0.0
    pas2 = pas.assign(sharpe=pas.sharpe + 0.01)
    assert G.sverka_pasport(df, pas2) == pytest.approx(0.01)
