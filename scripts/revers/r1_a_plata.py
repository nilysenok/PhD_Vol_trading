"""Реверс, раунд 1, плата за шорт 5 % для A_zashch (= A (s9)), TEST_C и D на test1 (Ж-124 п. 5). Справочно, без выбора.
Запуск: те же FASTBT_* (Z2 + FASTBT_Z2_DIV=<таблица>|NET), что у r1_a_cd.py: python3 scripts/revers/r1_a_plata.py [--out FILE]
Способ — как у якоря и A_strogaya (r1_a.py → D.stroka): x_с_платой = x − плата × база, плата 0,05 (AS.PLATA), x — дневной excess над КС
движка s9 (позиции — те же, что в r1_a_cd: A_row, A_row × затвор C, A_row × затвор D). База платы = доля капитала в шорте за ночь:
Σ_строк max(0, −поз) × календарных дней [d, d_next) / 365 / число пар на дату (формула fastbt/zaem_v2 и strogoe.port_strogiy). Лонг — 0.
Отличие от zaem_v2: в модели s9 close→close внутридневных сделок нет, поэтому правило «вход и выход в один день — 0» не действует.
Контроль: строки с платой 0 совпадают со строками r1_a_cd (a_cd_test1.csv; для NET — _net/a_cd_test1.csv) до 1e-12, иначе отказ.
mSR — в дневной шкале Корниш–Фишер (metriki_e.msr); вне области (mVaR ≤ 0) — NaN с пометкой в колонке `primechanie`."""
from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "scripts" / "analysis", ROOT / "scripts" / "revers"):
    sys.path.insert(0, str(p))
from fastbt import a_strogaya as AS, metriki_e as me, zona  # noqa: E402
import r1_a_cd as CD  # noqa: E402

PLATA, TOL, DNEI = AS.PLATA, 1e-12, 365.0
OBJEKTY = (("A", "A_zashch = A (s9)"), ("C", "TEST_C"), ("D", "D"))
GODY = (2022, 2023, 2024)
KOLONKI = ["obekt", "dividendy", "plata_zaem", "period", "sharpe", "sortino", "msr", "dni", "dolya_short_dnej", "primechanie"]
SOOTV = {"полные": "(а) полные", **{str(y): f"по году {y}" for y in GODY}}       # период → stroka в a_cd


def baza_shorta(pos, cal_row, date_idx, cnt) -> np.ndarray:
    """[даты]: доля капитала в шорте за ночь × календарные дни / 365; лонг и ноль дают 0."""
    sht = np.maximum(0.0, -np.asarray(pos, float)) * np.asarray(cal_row, float) / DNEI
    return np.bincount(date_idx, sht, len(cnt)) / cnt


def x_s_platoy(x, baza, plata: float) -> np.ndarray:
    """Дневной excess минус плата × база; плата 0 возвращает x без изменений."""
    return np.asarray(x, float) - plata * np.asarray(baza, float)


def metriki(x, baza) -> dict:
    """Sharpe (ddof=1, √252), Sortino (§6 metriki_e), mSR одного ряда; доля дней с шортом (база > 0)."""
    m = me.msr(x)
    return dict(sharpe=float(me.sr_dnevnoy(x) * np.sqrt(252)), sortino=me.sortino_e(x), msr=m, dni=len(x),
                dolya_short_dnej=float(np.mean(np.asarray(baza) > 0)),
                primechanie="mSR вне области (mVaR ≤ 0), NaN" if not np.isfinite(m) else "")


def stroki_obekta(imya: str, x, baza, dates, div: str) -> list:
    """Плата 0 и 5 %; период «полные» и по годам."""
    god = np.asarray(dates).astype("datetime64[D]").astype("datetime64[Y]").astype(int) + 1970
    rows = []
    for plata in (0.0, PLATA):
        xp = x_s_platoy(x, baza, plata)
        for per, m in (("полные", np.ones(len(x), bool)), *((str(y), god == y) for y in GODY)):
            rows.append(dict(obekt=imya, dividendy=div, plata_zaem=plata, period=per, **metriki(xp[m], baza[m])))
    return rows


def tablica(res: dict, poz: dict, div: str) -> pd.DataFrame:
    cal_row, di, cnt = poz["_dni"]
    rows = []
    for k, imya in OBJEKTY:
        if "prichina" in res[k]:
            rows.append(dict(obekt=imya, dividendy=div, plata_zaem=np.nan, period="все", primechanie=f"неприменимо: {res[k]['prichina']}"))
            continue
        zona.check(res[k]["dates"].max(), f"ряд {k}")                       # ≥ 2025 — отказ
        baza = baza_shorta(poz[k], cal_row, di, cnt)
        rows += stroki_obekta(imya, res[k]["x"], baza, res[k]["dates"], div)
    return pd.DataFrame(rows, columns=KOLONKI)


def sverka(df: pd.DataFrame, a_cd: Path) -> float:
    """Макс. |Δ| Sharpe и mSR строк с платой 0 против a_cd (полные и по годам); пропуск строки или лишняя — отказ."""
    ob = pd.read_csv(a_cd, comment="#")
    d, n = 0.0, 0
    for k, imya in OBJEKTY:
        for _, r in df[(df.obekt == imya) & (df.plata_zaem == 0)].iterrows():
            o = ob[ob.portfel.str.startswith(CD_PREFIKS[k]) & (ob.stroka == SOOTV[r.period])].iloc[0]
            for c in ("sharpe", "msr"):
                a, b = float(r[c]), float(o[c])
                if np.isnan(a) != np.isnan(b):
                    raise SystemExit(f"сверка: NaN не совпал {imya} {r.period} {c}")
                d, n = (d if np.isnan(a) else max(d, abs(a - b))), n + 1
    print(f"сверка с {a_cd.name}: {n} значений, max|Δ| = {d:.2e}")
    return d


CD_PREFIKS = {"A": "A (s9", "C": "TEST_C", "D": "D:"}


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def shapka(div: str) -> str:
    kom = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    s = f"# реверс раунд 1, 2022–2024, справочно, без выбора; период test1; плата за шорт 5 % (Ж-124 п. 5)\n# коммит: {kom}\n"
    s += f"# дивиденды: {div}\n# плата за шорт: x − 0,05 × база; база — доля капитала в шорте за ночь × календарные дни / 365 (A_row × затвор)\n"
    vh = [CD.sigma_z2.dev_path(h) for h in CD.sigma_z2.HORIZONS] + [CD.sigma_z2.z2_path(h) for h in CD.sigma_z2.HORIZONS]
    vh += [] if zona.div_net() else list(CD.dividendy._tables())
    return s + "".join(f"# вход: {p.name} sha256={sha(p)}\n" for p in vh) + zona.shapka_dividendy()


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)
    CD.ustanovit_period("test1")                      # зона test1 — до div_net()
    net = zona.div_net()
    div = "НЕТ (NET, справочно)" if net else "таблица 2022–2024 (основная)"
    poz: dict = {}
    res = CD.schitat("test1", poz)
    df = tablica(res, poz, div)
    base = ROOT / "docs" / "tables" / "revers_r1"
    out = a.out or (base / ("_net" if net else "") / "a_plata_short5_test1.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(shapka(div) + df.to_csv(index=False), encoding="utf-8")
    print(df.round(3).to_string())
    d = sverka(df, base / ("_net" if net else "") / "a_cd_test1.csv")
    if not d <= TOL:
        raise SystemExit(f"строки с платой 0 не совпали с a_cd (|Δ| = {d:.2e} > {TOL:.0e})")


if __name__ == "__main__":
    main()
