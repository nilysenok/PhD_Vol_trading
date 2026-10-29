"""Реверс р.1, Ж-124 п. 2: сверка «предзащита → s9 сейчас» для МСП C и D на 2022–2024 (весь период и годы).
Запуск: python3 scripts/revers/r1_sverka_CD.py [--out CSV]. Код выхода ≠ 0, если не прошли ворота (печать по годам, слой, итог).
Лестница — одно изменение за шаг; слой — fastbt.msp.portfolio(agg="pairs") (арифметика s9.Engine.port, тест test_sloy_raven_s9_port).
Шаги 1–4 — переключатели слоя на позициях предзащиты (daily_positions_pre2026.parquet); 5а — позиции fastbt после аудита
(final_test_main.parquet) на том же слое; 5б — s9 без дивидендов (_net/a_cd_test1.csv); 6 — дивиденды (a_cd_test1.csv).
Hold-out: CSV — построчно до первой даты ≥ 2025-01-01, parquet — фильтр date < 2025-01-01 до чтения. Ничего не подбирается."""
from __future__ import annotations

import argparse
import csv
import hashlib
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from fastbt import msp, zona  # noqa: E402
from fastbt.data import POS_PRE2026  # noqa: E402

GRAN, GODY = pd.Timestamp("2025-01-01"), (2022, 2023, 2024)
COMM = 0.0004                                  # AS.COMM: комиссия s9 на test1 (a_cd_test1.csv)
TAB = ROOT / "docs" / "tables" / "revers_r1"
S9_DIV, S9_NET = TAB / "a_cd_test1.csv", TAB / "_net" / "a_cd_test1.csv"
FINAL_TEST = Path("/Users/nilysenok/Desktop/Старое/PhD_Vol_trading/data/fastbt/final_test_main.parquet")   # тест 26.10 (уже открыт)
NAZV = {"C": "TEST_C: C1 W504 p70, hyb_h5", "D": "D: D6lag1 (выбор по валидации), hyb_h1"}
PECHAT = {"C": {2022: 2.86, 2023: 5.09, 2024: 2.98}, "D": {2022: 3.69, 2023: 3.89, 2024: 2.29}}
SRC_P = "moex_volatility/results/final/strategies/FINAL_RESULTS_CORRECTED_V2.md"
TOL_PECHAT, TOL_SLOY, TOL_ZAMK = 0.005, 1e-9, 1e-12
NE_MERITSYA = [("период калибровки", "утечка: A до 1.01.Y+1, валидация C/D [1.01.2021; 1.01.Y+1) против s9 (A до Y−1, валидация 2017…Y−1)"),
    ("сетка и выбор точек", "сетка v4 (RM 512/396/640 на стратегию, FINAL_GRIDS.txt:14-96) и выбор C/D по Шарпу на пару-год "
     "против фиксированной точки TEST_C и семьи D6lag1 s9; zh047 (Ж-047) — сетка A′ этапа 1, в s9 A не входит"),
    ("источник σ̂", "vpred_aligned.parquet против R2 final_d0_s2emb (rv_pred, σ = √rv_pred)"),
    ("горизонт h", "h ∈ {h1, h5, h22} по Шарпу на маске валидации против фиксированного hyb_h5 (C) / hyb_h1 (D)")]
NE_PRICHINA = ("не измеряется переключением: внутри шагов 5а/5б (позиции); разложение требует прогона fastbt.pipeline "
               "с одним переключателем на барах Z2 — pipeline читает ряды до 2026 (hold-out), режима зоны Z2 в нём нет")


def csv_do_gran(path: Path) -> pd.DataFrame:
    """Строки CSV печати с датой < 2025-01-01; чтение останавливается на первой дате ≥ границы (даты по возрастанию)."""
    rows, prev = [], ""
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["date"] < prev:
                raise ValueError(f"{path.name}: даты не по возрастанию ({prev} → {r['date']})")
            if pd.Timestamp(r["date"]) >= GRAN:
                break
            rows.append(r)
            prev = r["date"]
    d = pd.DataFrame(rows, columns=list(rows[0]) if rows else ["date"])
    return d.assign(date=pd.to_datetime(d["date"])).set_index("date").astype(float)


def stavka_vosst(a_csv: pd.DataFrame) -> pd.Series:
    """Ставка печати, восстановленная как msp.rate_restored, но только по строкам < 2025: rf = (total − excess)/cal·365."""
    cal = msp.cal_days(a_csv.index, forward=False)
    s = pd.Series((a_csv["total_ret"] - a_csv["excess_ret"]).to_numpy() / cal * 365.0, index=a_csv.index)
    s.attrs["cal"] = "back"
    return s


def s_dnyami(rate: pd.Series, nazad: bool) -> pd.Series:
    """Тот же ряд ставки с соглашением дней: назад (d_prev, d] — как печать; вперёд [d, d_next) — Н2 (и выход Н5)."""
    s = rate.copy()
    s.attrs = {"cal": "back"} if nazad else {}
    return s


def pozicii(ap: str, path: Path = POS_PRE2026) -> pd.DataFrame:
    """Позиции подхода ap, daily, метка года 2022–2024, даты < 2025 (фильтр до чтения в память)."""
    f = ((ds.field("tf") == "daily") & (ds.field("approach") == ap) & (ds.field("date") < GRAN)
         & ds.field("test_year").isin(list(GODY)))
    p = ds.dataset(path).to_table(filter=f).to_pandas()
    zona.check(p["date"].max(), f"позиции {ap}")
    return p


def zapolnit_a(pc: pd.DataFrame, pa: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """fix_c_fill: единицы (тикер, стратегия, год) без строк C берутся из A того же года; имеющиеся строки C не трогаются."""
    k = ["ticker", "strategy", "test_year"]
    est = set(map(tuple, pc[k].drop_duplicates().to_numpy()))
    dop = pa[[tuple(x) not in est for x in pa[k].to_numpy()]].assign(approach=pc["approach"].iloc[0])
    return pd.concat([pc, dop], ignore_index=True), len(dop)


def sharpy(ex: pd.Series) -> dict:
    return {"ves": msp.sharpe(ex.to_numpy()), **{y: msp.sharpe(ex[ex.index.year == y].to_numpy()) for y in GODY}}


def s9_iz(path: Path, ap: str) -> dict:
    t = pd.read_csv(path, comment="#")
    t = t[t["portfel"] == NAZV[ap]].set_index("stroka")["sharpe"]
    return {"ves": float(t["(а) полные"]), **{y: float(t[f"по году {y}"]) for y in GODY}}


def lestnica(ap: str, a_csv: pd.DataFrame, rate_cbr: pd.Series) -> tuple[list, dict]:
    """Шаги (шаг, изменение, Шарпы, источник) и ворота: печать по годам и слой на позициях предзащиты."""
    pech = csv_do_gran(msp.MSP_CSV / f"MSP_{ap}_inst.csv")
    rv = stavka_vosst(a_csv)
    pc, pa = pozicii(ap), pozicii("A")
    sloy = lambda pos, r, c: msp.portfolio(pos, r, c, agg="pairs")["excess"]        # noqa: E731
    e0 = sloy(pc, rv, 0.0)
    vor = {"sloy": float(np.max(np.abs(e0.to_numpy() - pech["excess_ret"].reindex(e0.index).to_numpy()))),
           "dni": (len(e0), len(pech)), "pechat": {y: sharpy(pech["excess_ret"])[y] - PECHAT[ap][y] for y in GODY}}
    pf, nfill = zapolnit_a(pc, pa)
    st = [("0", "как на предзащите: ряд MSP_%s_inst.csv (comm 0, ставка восстановленная, дни (d_prev, d], EW по парам)" % ap,
           sharpy(pech["excess_ret"]), f"{SRC_P}:23-25 (годы); MSP_{ap}_inst.csv < 2025"),
          ("1", "ставка: восстановленная из CSV → решения ЦБ (msp.rate_cbr), дни прежние", sharpy(sloy(pc, s_dnyami(rate_cbr, True), 0.0)),
           "fastbt/msp.py rate_cbr; docs/fastbt_spec.md п. 9"),
          ("2", "дни ставки (d_prev, d] → [d, d_next) (Н2); остановка торгов 25.02–24.03.2022: ставка за закрытие — "
           "на 24.03 → на 25.02", sharpy(sloy(pc, s_dnyami(rate_cbr, False), 0.0)), "fastbt/msp.py cal_days; аудит FA Н2"),
          ("3", f"комиссия ряда 0 → {COMM:.2%} на сторону (с выходом на конце окна, Н5)",
           sharpy(sloy(pc, s_dnyami(rate_cbr, False), COMM)), "fastbt/a_strogaya.py COMM; docs/fastbt_spec.md п. 8"),
          ("4", f"состав: пропущенные единицы C → C = A (fix_c_fill), добавлено строк: {nfill}",
           sharpy(sloy(pf, s_dnyami(rate_cbr, False), COMM)), "docs/fastbt_spec.md п. 18.3, 20"),
          ("5а", "позиции: предзащита (v4 + v4_1, маска [1], vpred_aligned, h по Шарпу) → fastbt после аудита (маска [0], "
           "шесть исправлений, main.yaml: σ final3 h1, отбор 0,05 %, фильтр оборота)",
           sharpy(sloy(pozicii(ap, FINAL_TEST), s_dnyami(rate_cbr, False), COMM)), f"{FINAL_TEST.name}; FINAL_DESIGN_strategies.md п. 3–4"),
          ("5б", "позиции: fastbt main.yaml (бары v4 с корп. событиями) → s9 (точка/семья s9, σ̂ R2 final_d0_s2emb, бары Z2), "
           "без дивидендов 2022–2024", s9_iz(S9_NET, ap), "docs/tables/revers_r1/_net/a_cd_test1.csv"),
          ("6", "дивиденды 2022–2024: нет → таблица 2026-10-29_dividendy_2022_2024.csv", s9_iz(S9_DIV, ap),
           "docs/tables/revers_r1/a_cd_test1.csv")]
    return st, vor


def zamknut(st: list) -> dict:
    """Δ шагов и остаток: остаток = (итог − строка 0) − ΣΔ по каждому столбцу; обязан быть ≈ 0 (лестница цепная)."""
    kl = ["ves", *GODY]
    d = [{k: s[2][k] - p[2][k] for k in kl} for p, s in zip(st[:-1], st[1:])]
    return {k: (st[-1][2][k] - st[0][2][k]) - sum(x[k] for x in d) for k in kl} | {"delty": d}


def tablica(ap: str, st: list) -> pd.DataFrame:
    z = zamknut(st)
    rows = []
    for i, (sh, izm, s, src) in enumerate(st):
        r = dict(obekt=ap, shag=sh, izmenenie=izm, sharpe=s["ves"], delta=np.nan if i == 0 else z["delty"][i - 1]["ves"])
        for y in GODY:
            r[f"sharpe_{y}"], r[f"delta_{y}"] = s[y], (np.nan if i == 0 else z["delty"][i - 1][y])
        rows.append(dict(r, istochnik=src))
    rows.append(dict(obekt=ap, shag="ост", izmenenie="остаток (итог − строка 0 − ΣΔ)", delta=z["ves"],
                     **{f"delta_{y}": z[y] for y in GODY}, istochnik="по построению цепной лестницы"))
    rows += [dict(obekt=ap, shag="—", izmenenie=f"{imya}: {chto}", istochnik=NE_PRICHINA) for imya, chto in NE_MERITSYA]
    rows += [dict(obekt=ap, shag="—", izmenenie=t, delta=0.0, **{f"delta_{y}": 0.0 for y in GODY}, istochnik=s) for t, s in (
        ("веса: EW по парам (стратегия, тикер) в обоих; s9.Engine.port делит на число строк даты", "fastbt/s9.py Engine.port"),
        ("ФДР и вычет ставки: одна формула fdr = max(0, 1 − экспозиция)·rf·дни, excess = total − rf·дни",
         "fastbt/msp.py portfolio; fastbt/s9.py Engine._finish"),
        ("шорты: плата за шорт 0 и в печати, и в a_cd_test1.csv (plata_zaem = 0)", "a_cd_test1.csv, столбец plata_zaem"))]
    return pd.DataFrame(rows)


def proverit(df: pd.DataFrame) -> list:
    """Ворота таблицы: Δ шага = Шарп шага − Шарп предыдущего; ΣΔ + остаток = итог − строка 0 (весь период и годы)."""
    osh = []
    for ob, g in df.groupby("obekt"):
        st, ost = g[g["sharpe"].notna()], g[g["shag"] == "ост"].iloc[0]
        for s, d in [("sharpe", "delta")] + [(f"sharpe_{y}", f"delta_{y}") for y in GODY]:
            raz = np.abs(st[d].to_numpy()[1:] - np.diff(st[s].to_numpy())).max()
            zam = abs(st[d].sum() + ost[d] - (st[s].iloc[-1] - st[s].iloc[0]))
            osh += [f"{ob} {d}: Δ≠разность {raz:.1e} / не замкнуто {zam:.1e}"] if max(raz, zam) >= TOL_ZAMK else []
    return osh


def shapka() -> str:
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()        # noqa: E731
    kom = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    vh = [msp.MSP_CSV / f"MSP_{x}_inst.csv" for x in "ACD"] + [POS_PRE2026, FINAL_TEST, msp.KEY_RATE, S9_NET, S9_DIV]
    s = ("# статус: реверс р.1, 2022–2024; сверка предзащита → s9 (Ж-124 п. 2), справочно, без выбора\n"
         f"# коммит: {kom}\n# хэш файла целиком (механически), в расчёт — только строки < 2025-01-01\n")
    nm = lambda p: p.relative_to(ROOT) if ROOT in Path(p).parents else p        # noqa: E731
    return s + "".join(f"# вход: {nm(Path(p))} sha256={sha(p)}\n" for p in vh)


def main(argv=None) -> int:
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--out", type=Path, default=TAB / "sverka_predzashchita_CD.csv")
    a = ap_.parse_args(argv)
    a_csv, rc = csv_do_gran(msp.MSP_CSV / "MSP_A_inst.csv"), msp.rate_cbr()
    parts, oshibki = [], []
    for ap in ("C", "D"):
        st, vor = lestnica(ap, a_csv, rc)
        print(f"{ap}: ворота слоя max|Δ excess| = {vor['sloy']:.2e}, дни {vor['dni']}; печать по годам Δ = "
              + ", ".join(f"{y}: {v:+.4f}" for y, v in vor["pechat"].items()))
        oshibki += [f"{ap}: слой {vor['sloy']:.1e}"] if not vor["sloy"] < TOL_SLOY or vor["dni"][0] != vor["dni"][1] else []
        oshibki += [f"{ap} {y}: печать {v:+.4f}" for y, v in vor["pechat"].items() if not abs(v) < TOL_PECHAT]
        parts.append(tablica(ap, st))
    df = pd.concat(parts, ignore_index=True)
    oshibki += proverit(df)
    a.out.write_text(shapka() + df.to_csv(index=False), encoding="utf-8")
    print(df[["obekt", "shag", "sharpe", "delta", "sharpe_2022", "sharpe_2023", "sharpe_2024"]].round(4).to_string())
    print("ворота: " + ("пройдены" if not oshibki else "НЕ ПРОЙДЕНЫ: " + "; ".join(oshibki)))
    return 1 if oshibki else 0


if __name__ == "__main__":
    raise SystemExit(main())
