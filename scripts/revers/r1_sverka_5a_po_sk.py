"""Реверс р.1, Ж-130: шаг «утечка окна» (P1: маска [1] → [0]) сверки 5а в разрезе стратегий S1–S6, МСП A, C, D, 2022–2024.
Позиции P0 и P1 — из прогона r1_sverka_5a (data/fastbt/sverka_5a/{P0,P1}.parquet); слой МСП тот же (шаги 1–4 сверки C/D).
P0 = с утечкой (маска [1]), P1 = без утечки (маска [0]); шаг «утечка» = убрать её. Вклад Sk — шаг применён только к Sk:
«прямой» = Шарп(P0, но Sk из P1) − Шарп(P0); «обратный» = Шарп(P1) − Шарп(P1, но Sk из P0).
Шарп самой Sk — портфель только из пар этой стратегии с утечкой (P0) и без (P1).
  FASTBT_MAIN=<своя копия> FASTBT_DATA_ROOT=<осн.>/data/fastbt/daily python3 scripts/revers/r1_sverka_5a_po_sk.py [--out CSV]"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "revers"))
import r1_sverka_5a as S5  # noqa: E402

CD = S5.CD
SK = ("S1_MeanRev", "S2_Bollinger", "S3_Donchian", "S4_Supertrend", "S5_PivotPoints", "S6_VWAP")      # имена в позициях fastbt
PER = ("ves", *CD.GODY)


def smeshat(p0: pd.DataFrame, p1: pd.DataFrame, k: str) -> pd.DataFrame:
    """Позиции P0, у которых стратегия k взята из P1 (утечка включена только у k)."""
    return pd.concat([p0[p0["strategy"] != k], p1[p1["strategy"] == k]], ignore_index=True)


def sloy_msp(rate: pd.Series):
    """positions -> {ves, 2022, 2023, 2024}: Шарп избыточной доходности портфеля (слой сверки C/D, agg=pairs)."""
    return lambda p: CD.sharpy(CD.msp.portfolio(p, CD.s_dnyami(rate, False), CD.COMM, agg="pairs")["excess"])


def n_par(p: pd.DataFrame, k: str) -> float:
    """Среднее за день число пар (тикер, стратегия) k — вес стратегии в EW по имеющимся парам."""
    return float(p[p["strategy"] == k].groupby("date").size().mean())


def vklady(ap: str, p0: pd.DataFrame, p1: pd.DataFrame, sloy, smesh=smeshat) -> pd.DataFrame:
    full0, full1 = sloy(p0), sloy(p1)
    rows = []
    for k in SK:
        s0, s1 = sloy(p0[p0["strategy"] == k]), sloy(p1[p1["strategy"] == k])
        pr, ob = sloy(smesh(p0, p1, k)), sloy(smesh(p1, p0, k))
        rows += [dict(obekt=ap, strategiya=k, period=str(q), sharpe_sk_s_utechkoj=s0[q], sharpe_sk_bez_utechki=s1[q], delta_sk=s1[q] - s0[q],
                      vklad_pryamoj=pr[q] - full0[q], vklad_obratnyj=full1[q] - ob[q], n_par_s_utechkoj=n_par(p0, k),
                      n_par_bez_utechki=n_par(p1, k))
                 for q in PER]
    t = pd.DataFrame(rows)
    s = t.groupby("period")[["vklad_pryamoj", "vklad_obratnyj"]].sum().reset_index().assign(obekt=ap, strategiya="Σ вкладов Sk")
    t = pd.concat([t, s], ignore_index=True)
    pol = pd.DataFrame([dict(obekt=ap, strategiya="полный шаг P1−P0", period=str(q), sharpe_sk_s_utechkoj=full0[q], sharpe_sk_bez_utechki=full1[q],
                             delta_sk=full1[q] - full0[q], vklad_pryamoj=full1[q] - full0[q], vklad_obratnyj=full1[q] - full0[q]) for q in PER])
    t = pd.concat([t, pol], ignore_index=True)
    sm, fl = t[t["strategiya"] == "Σ вкладов Sk"].set_index("period"), pol.set_index("period")
    ne = pd.DataFrame([dict(obekt=ap, strategiya="Σ − полный (неаддитивность)", period=str(q),
                            vklad_pryamoj=sm.loc[str(q), "vklad_pryamoj"] - fl.loc[str(q), "delta_sk"],
                            vklad_obratnyj=sm.loc[str(q), "vklad_obratnyj"] - fl.loc[str(q), "delta_sk"]) for q in PER])
    return pd.concat([t, ne], ignore_index=True)


def shapka() -> str:
    import hashlib
    import subprocess
    kom = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    h = "".join(f"# вход: {s}.parquet (позиции прогона r1_sverka_5a, в git нет) sha256={hashlib.sha256((S5.OUT / f'{s}.parquet').read_bytes()).hexdigest()}\n"
                for s in ("P0", "P1"))
    return ("# статус: реверс р.1, 2022–2024; утечка окна (шаг P1 сверки 5а) по стратегиям S1–S6, справочно, без выбора (Ж-130)\n"
            f"# коммит: {kom}\n# прочие входы (бары Z2, σ̂, ставка) — как в шапке sverka_5a.csv; 2024: P0 = предзащита (E0), см. sverka_5a.md\n" + h)


def main(argv=None) -> int:
    a = argparse.ArgumentParser()
    a.add_argument("--out", type=Path, default=CD.TAB / "sverka_5a_po_sk.csv")
    a = a.parse_args(argv)
    sloy, ref, osh, tabs = sloy_msp(CD.msp.rate_cbr()), pd.read_csv(CD.TAB / "sverka_5a.csv", comment="#"), [], []
    for ap in S5.APS:
        t = vklady(ap, S5.pozicii(ap, "P0"), S5.pozicii(ap, "P1"), sloy)
        r = ref[ref["obekt"] == ap].set_index("shag")                      # ворота: полный шаг = P1 основной сверки 5а
        d = t[t["strategiya"] == "полный шаг P1−P0"].set_index("period")["delta_sk"]
        osh += [f"{ap} {q}: P1−P0 ≠ sverka_5a ({d[q] - float(r.loc['P1', c]):.1e})" for q, c in
                zip(map(str, PER), ("delta", "delta_2022", "delta_2023", "delta_2024")) if not abs(d[q] - float(r.loc["P1", c])) < 1e-3]
        tabs.append(t)
    df = pd.concat(tabs, ignore_index=True)
    a.out.write_text(shapka() + df.to_csv(index=False), encoding="utf-8")
    print(df[df["period"] == "ves"][["obekt", "strategiya", "sharpe_sk_s_utechkoj", "sharpe_sk_bez_utechki", "vklad_pryamoj", "vklad_obratnyj"]].round(3).to_string())
    print("ворота: " + ("пройдены" if not osh else "НЕ ПРОЙДЕНЫ: " + "; ".join(osh)))
    return 1 if osh else 0


if __name__ == "__main__":
    raise SystemExit(main())
