"""Листы б и е: параметр --yakor A_zashch|A_fix (Ж-124 п. 1). Мутация «A_zashch молча берёт A_fix» — ловится сверкой с a_cd (без данных)
и сравнением с опубликованными таблицами. Модульных подмен нет (mutations.md № 43); состояние AS.PER/AS.GODY тест восстанавливает сам.
Подмножество (объявлено) для --yakor A_fix: лист б — все 6 Sk, все 7 вариантов (полная таблица A_fix против опубликованного b_razlozhenie.csv, 1e-9);
лист е — ряд A_fix по «po_fazam» против e_po_fazam_test1.csv (ряд A). Данные Z2 и IMOEX — из основной копии; нет среды — пропуск."""
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "revers"))
sys.path.insert(0, str(ROOT / "scripts" / "analysis"))
import r1_b  # noqa: E402
spec = importlib.util.spec_from_file_location("r1_e_yakor", ROOT / "scripts/revers/r1_e.py")
r1e = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r1e)

T = ROOT / "docs" / "tables" / "revers_r1"
A_CD = T / "a_cd_test1.csv"
T_FIX = T / "_do_a_zashch"      # опубликованная версия A_fix (29.10, перед пересчётом на A_zashch; sha256 — в MANIFEST.tsv): b_razlozhenie.csv и e_po_fazam_test1.csv в T уже A_zashch


def _opubl_b() -> pd.DataFrame:
    return pd.read_csv(T_FIX / "b_razlozhenie.csv", comment="#")


def _z2():
    if not (os.environ.get("FASTBT_Z2_R2FINAL") and os.environ.get("FASTBT_Z2_DAILY")):
        pytest.skip("нет среды Z2 (FASTBT_Z2_*)")


def test_a_zashch_otlichim_ot_a_fix_v_opublikovannyh_tablicah():
    o = _opubl_b()
    fix = o[(o.portfel == "yakor") & (o.kod == "baza") & (o.plata_zaem == 0)].sharpe.iloc[0]
    cd = pd.read_csv(A_CD, comment="#")
    zashch = cd[cd.portfel.str.startswith("A (s9") & (cd.stroka == "(а) полные")].sharpe.iloc[0]
    assert abs(fix - zashch) > 0.05                       # подмену одного другим сверка видит


def test_sverka_a_cd_prinimaet_a_zashch_i_lovit_podmenu_a_fix():
    cd = pd.read_csv(A_CD, comment="#")
    r = cd[cd.portfel.str.startswith("A (s9") & (cd.stroka == "(а) полные")].iloc[0]
    ok = pd.DataFrame([dict(portfel="A_zashch", kod="baza", plata_zaem=0.0, sharpe=r.sharpe, msr=r.msr)])
    assert r1_b.sverka_a_cd(ok, A_CD) == 0.0
    o = _opubl_b()
    mut = o[(o.portfel == "yakor") & (o.kod == "baza")].assign(portfel="A_zashch")          # мутация: A_zashch молча = A_fix
    with pytest.raises(SystemExit):
        r1_b.sverka_a_cd(mut, A_CD)
    sdvig = ok.assign(sharpe=ok.sharpe + 1e-9)
    with pytest.raises(SystemExit):                                                          # допуск 1e-12, не «примерно»
        r1_b.sverka_a_cd(sdvig, A_CD)


def test_tablica_podpisi_i_nedelnye_kolonki_v_konce():
    n = 120
    d = (np.datetime64("2022-01-03") + np.arange(n)).astype("datetime64[D]")
    rng = np.random.default_rng(1)
    port = {v: dict(x=rng.normal(0, 0.01, n), baza=np.abs(rng.normal(0, 0.1, n)), dates=d) for v, _ in r1_b.STROKI}
    df = r1_b.tablica({"A_zashch": port, "strogaya": port})
    assert set(df.portfel) == {"A_zashch", "strogaya"} and "yakor" not in set(df.portfel)
    assert list(df.columns[-7:]) == ["sharpe_ned", "sortino_ned", "msr_ned", "skew_ned", "kurt_ned", "n_nedel", "msr_dn_opredelen"]
    assert list(df.columns[:4]) == ["portfel", "stroka", "kod", "plata_zaem"]
    assert (df.n_nedel > 10).all()


def test_e_serii_s_raznymi_datami_ne_skleivayutsya():
    """A_zashch (s9) на 1 строку длиннее A_strogaya: у каждого ряда своя фаза и свои дни (мутация — общий список дат — ломает число дней)."""
    idx = pd.bdate_range("2021-06-01", periods=700)
    im = pd.Series(100 + np.cumsum(np.random.default_rng(0).normal(0, 0.5, len(idx))), index=idx)
    d1 = idx[200:500]
    d0 = idx[200:499]
    s = {"A_zashch": (d1, np.full(300, 0.001), np.zeros(300)), "A_strogaya": (d0, np.full(299, 0.001), np.zeros(299))}
    t = r1e.tablicy("test1", im, s)["po_fazam"]
    dni = t[(t.faza == "все") & (t.plata_zaem == 0)].set_index("ryad").dni
    assert dni["A_zashch"] == 300 and dni["A_strogaya"] == 299
    assert {"sharpe_ned", "n_nedel"} <= set(t.columns) and list(t.ryad.unique()) == ["A_zashch", "A_strogaya"]


def test_b_a_fix_vosproizvodit_opublikovannuyu_tablicu():
    _z2()
    from fastbt import a_strogaya as AS
    old = AS.PER, AS.GODY
    try:
        per = {s: r1_b.schitat((s, "test1", "A_fix")) for s in AS.SIDS}
        df = r1_b.tablica({"A_fix": r1_b.portfeli_pary(per, "A_fix")}).assign(portfel="yakor")
    finally:
        AS.PER, AS.GODY = old
    o = _opubl_b()
    m = df.merge(o, on=["portfel", "kod", "plata_zaem"], suffixes=("", "_o"))
    assert len(m) == 14
    for c in ("msr", "sharpe", "sum_pnl_pct", "d_sharpe", "d_sum_pnl_pct"):
        assert np.nanmax(np.abs(m[c] - m[c + "_o"])) <= 1e-9 and (m[c].isna() == m[c + "_o"].isna()).all(), c
    assert np.nanmax(np.abs(m.sharpe - m.sharpe.iloc[0])) > 0                                # не константа (пустая лента не пройдёт)


def test_e_a_fix_vosproizvodit_opublikovannye_ryady():
    _z2()
    imoex = Path(os.environ["FASTBT_Z2_IMOEX"])
    from fastbt import a_strogaya as AS
    old = AS.PER, AS.GODY
    try:
        ser = r1e.ryady("test1", "A_fix")
    finally:
        AS.PER, AS.GODY = old
    assert list(ser) == ["A_fix", "A_strogaya"]
    d, x, b = ser["A_fix"]
    from fastbt.cache import MAIN
    path_d = MAIN / "data" / "iss_check" / "index_d" / "IMOEX.parquet"
    faza = r1e.fazy_dnej(r1e.zagruzit_imoex(imoex, *r1e.PERIODY["test1"][:2], path_d), d)
    faza.index = d
    got = pd.DataFrame(r1e.po_fazam("A", x, b, faza, 0.05))
    o = pd.read_csv(T_FIX / "e_po_fazam_test1.csv", comment="#")
    o = o[(o.ryad == "A") & (o.plata_zaem == 0.05)].reset_index(drop=True)
    assert len(got) == len(o) > 5
    for c in ("dni", "sredn_den", "sharpe", "msr", "dohodnost"):
        assert np.nanmax(np.abs(got[c] - o[c])) <= 1e-9 and (got[c].isna() == o[c].isna()).all(), c
