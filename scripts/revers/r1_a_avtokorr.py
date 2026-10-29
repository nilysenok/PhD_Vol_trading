"""Реверс, раунд 1, лист R-а: автокорреляция дневных доходностей (Ж-126). Справочно, без выбора.
Объекты и ряды — теми же функциями, что строят таблицы листа а (копии формул нет): якорь A_fix и A_strogaya — r1_a.schitat + D.portfel
(плата за заём 0,05 и 0, как r1_a_gody); A (s9), TEST_C, D — r1_a_cd.schitat(pos_out) + r1_a_plata.baza_shorta/x_s_platoy (плата за шорт 0 и 5 %).
Периоды: полные + 2022/2023/2024. Метрики — fastbt.metriki_e.avtokorr_stroka (одна формула, Д7): ACF(1..5) = Σ(x_t−m)(x_{t−k}−m)/Σ(x_t−m)²,
VR(5) = Var(недельные суммы W-FRI, ddof=1)/(5·Var(дневной, ddof=1)) по тем же неделям, что Sharpe нед. (неполные недели входят как есть — выбрано до счёта;
контроль — vr5_polnye: только недели из 5 торговых дней), Льюнг–Бокс Q(5), серии убыточных дней (убыток = x<0).
Гейт: sharpe_dn и sharpe_ned = колонкам опубликованных a_dve_metriki / a_po_godam / a_cd (а_plata_short5 — только sharpe) с допуском 1e-12.
Запуск: те же FASTBT_* (Z2 + FASTBT_BARS1M), что у r1_a_cd.py и r1_a.py: python3 scripts/revers/r1_a_avtokorr.py [--out FILE]"""
from __future__ import annotations

import argparse
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "scripts" / "analysis", ROOT / "scripts" / "revers"):
    sys.path.insert(0, str(p))
from fastbt import a_strogaya as AS, metriki_e as me, zona  # noqa: E402
import dve_metriki_a as D  # noqa: E402
import r1_a  # noqa: E402
import r1_a_cd as CD  # noqa: E402
import r1_a_plata as PL  # noqa: E402

TOL = 1e-12
GODY = PL.GODY
TABL = ROOT / "docs" / "tables" / "revers_r1"
IMENA = {"yakor": "A_fix (якорь этапа 1)", "strogaya": "A_strogaya", "A": "A_zashch = A (s9)", "C": "TEST_C", "D": "D"}


def periody(dates):
    """('полные', все дни) и по годам 2022–2024."""
    god = np.asarray(dates).astype("datetime64[D]").astype("datetime64[Y]").astype(int) + 1970
    return [("полные", np.ones(len(god), bool)), *((str(y), god == y) for y in GODY)]


def stroki(kod: str, x, dates, plata: float) -> list:
    """Строки одного ряда (уже с платой): полные + по годам."""
    dates = np.asarray(dates).astype("datetime64[D]")
    return [dict(kod=kod, obekt=IMENA[kod], plata_zaem=plata, period=per, **me.avtokorr_stroka(np.asarray(x)[m], dates[m]))
            for per, m in periody(dates)]


def stroki_yakor_strogaya() -> list:
    with Pool(2) as p:
        res_s = dict(zip(AS.SIDS, p.map(r1_a.schitat, [(s, "test1") for s in AS.SIDS])))
    rows = []
    for imya in ("yakor", "strogaya"):
        a = D.portfel({s: res_s[s][imya] for s in AS.SIDS}, "baza")
        for plata in (AS.PLATA, 0.0):
            rows += stroki(imya, a["x"] - plata * a["baza"], a["dates"], plata)
    return rows


def stroki_cd() -> list:
    CD.ustanovit_period("test1")
    poz: dict = {}
    res = CD.schitat("test1", poz)
    cal_row, di, cnt = poz["_dni"]
    rows = []
    for kod, _ in PL.OBJEKTY:
        if "prichina" in res[kod]:
            rows.append(dict(kod=kod, obekt=IMENA[kod], plata_zaem=np.nan, period="все", primechanie=f"неприменимо: {res[kod]['prichina']}"))
            continue
        zona.check(res[kod]["dates"].max(), f"ряд {kod}")                       # ≥ 2025 — отказ
        baza = PL.baza_shorta(poz[kod], cal_row, di, cnt)
        for plata in (0.0, PL.PLATA):
            rows += stroki(kod, PL.x_s_platoy(res[kod]["x"], baza, plata), res[kod]["dates"], plata)
    return rows


def _chitat(f: str) -> pd.DataFrame:
    return pd.read_csv(TABL / f, comment="#")


def obraztsy() -> dict:
    """{(kod, плата, период): (sharpe, sharpe_ned | None)} из опубликованных таблиц листа а."""
    ob, nm = {}, lambda s: s.replace("(а) полные", "полные").replace("по году ", "")                   # noqa: E731
    for f in ("a_dve_metriki_test1.csv", "a_po_godam_test1.csv"):
        for _, r in _chitat(f).iterrows():
            if r.stroka.startswith(("(а)", "по году")):
                ob[(r.portfel, round(float(r.plata_zaem), 4), nm(r.stroka))] = (r.sharpe, r.sharpe_ned)
    pref = {"A (s9": "A", "TEST_C": "C", "D:": "D"}
    for _, r in _chitat("a_cd_test1.csv").iterrows():
        k = next((v for p, v in pref.items() if r.portfel.startswith(p)), None)
        if k and isinstance(r.stroka, str) and r.stroka.startswith(("(а)", "по году")) and pd.notna(r.sharpe):
            ob[(k, 0.0, nm(r.stroka))] = (r.sharpe, r.sharpe_ned)
    sok = {v[1]: k for k, v in zip(("A", "C", "D"), PL.OBJEKTY)}
    for _, r in _chitat("a_plata_short5_test1.csv").iterrows():
        if r.obekt in sok and r.plata_zaem == PL.PLATA and pd.notna(r.sharpe):
            ob[(sok[r.obekt], round(PL.PLATA, 4), str(r.period))] = (r.sharpe, None)
    return ob


def sverka(df: pd.DataFrame, ob: dict) -> tuple[float, int]:
    """Макс. |Δ| Sharpe дн./нед. против опубликованных; число сверенных значений. Ряд без образца — отказ (молчаливых пропусков нет)."""
    d, n = 0.0, 0
    for _, r in df[df.sharpe_dn.notna()].iterrows():
        o = ob.get((r.kod, round(float(r.plata_zaem), 4), r.period))
        if o is None:
            raise SystemExit(f"сверка: нет образца для {r.kod} плата {r.plata_zaem} период {r.period}")
        d, n = max(d, abs(r.sharpe_dn - o[0])), n + 1
        if o[1] is not None and pd.notna(o[1]):
            d, n = max(d, abs(r.sharpe_ned - o[1])), n + 1
    return d, n


def pervaya_stroka(df: pd.DataFrame) -> str:
    """VR(5) и P(убыток|убыток) по пяти главным объектам (полные; якорь и строгая — плата 0,05, A/C/D — плата 0)."""
    out = []
    for kod in IMENA:
        r = df[(df.kod == kod) & (df.period == "полные") & (df.plata_zaem == (AS.PLATA if kod in ("yakor", "strogaya") else 0.0))]
        if len(r):
            r = r.iloc[0]
            out.append(f"{IMENA[kod]}: VR(5)={r.vr5:.3f}, P(у|у)={r.p_ub_ub:.3f} (P(у)={r.p_ub:.3f})")
    return "; ".join(out)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=TABL / "a_avtokorr_test1.csv")
    a = ap.parse_args(argv)
    rows_cd = stroki_cd()                       # зона test1 — до всего остального (s9.set_period)
    df = pd.DataFrame(stroki_yakor_strogaya() + rows_cd)
    d, n = sverka(df, obraztsy())
    print(f"сверка Sharpe дн./нед. с опубликованными таблицами: {n} значений, max|Δ| = {d:.2e}")
    if n == 0 or not d <= TOL:
        raise SystemExit(f"гейт сверки не пройден (n={n}, |Δ|={d:.2e} > {TOL:.0e})")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    opr = ("# автокорреляция дневных доходностей (Ж-126): acf1..5 = Σ(x_t−m)(x_{t−k}−m)/Σ(x_t−m)²; summa_acf14 = Σ acf1..4; vr5 = Var(нед. суммы W-FRI)/(5·Var(дн.)), "
           "недели как в Sharpe нед. (неполные входят), vr5_polnye — только полные недели; q5, p_q5 — Льюнг–Бокс; p_ub — доля убыточных дней, p_ub_ub — P(убыток d+1 | убыток d), "
           "dlina — средняя длина серии убыточных дней, dlina_nezav = 1/(1−p_ub)\n")
    a.out.write_text(CD.shapka("test1") + opr + df.to_csv(index=False), encoding="utf-8")
    print(df.drop(columns=["kod"]).round(3).to_string())
    print("ПЕРВАЯ СТРОКА:", pervaya_stroka(df))


if __name__ == "__main__":
    main()
