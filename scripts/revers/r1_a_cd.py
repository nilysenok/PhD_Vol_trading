"""Реверс, раунд 1, лист R-а, дополнение: TEST_C и D поверх A (s9) — те же строки и определения, что у A в r1_a.py. Справочно, без выбора.
Запуск: FASTBT_Z2_R2FINAL=<R2/final> FASTBT_Z2_DAILY=… FASTBT_Z2_DIV=… python3 scripts/revers/r1_a_cd.py --period {train4,test1,train} [--out DIR]
Константы (ровно они, перебора нет): TEST_C = s9.TEST_C = ("C1:W504_p70", "hyb_h5"); D = замороженный выбор run_log_final3.json — семья
D6lag1 на hyb_h1 (s9.run_d, ключ ("D6lag1", "hyb_h1"), экран 0,10–0,50). σ̂ — семейство R2 final_d0_s2emb (dev ch4_dev + Z2) через s9.load_source.
Комиссия на test1 — 0,04 % (AS.COMM), выбор по валидации внутри семьи D — как в s9 (s9.COMM). Строки стоп 2σ / размер 1/σ — «неприменимо»:
позиции C/D = маска затвора × позиции A из кэша s9, движок s9 стопа и размера не пересчитывает. Плата за шорт в движке s9 не считается — только 0.
train4: σ-источников в s9 нет (s9.need_sigma) — отказ. Контроль: --period train (2018–2021) сверяет TEST_C и D с docs/tables/strat_dev/s9_C_rezhim.csv
и s9_D_zatvor.csv, допуск 1e-4 (в таблицах 4 знака)."""
from __future__ import annotations

import argparse
import hashlib
import os
import pickle
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "analysis"))
from fastbt import a_strogaya as AS, dividendy, s9, sigma_z2, zona  # noqa: E402
import dve_metriki_a as D  # noqa: E402

SHAPKA = "# реверс раунд 1, 2022–2024, справочно, без выбора"
PERIODY = ("train4", "test1", "train")
C_POINT, C_SRC = s9.TEST_C[0].split(":", 1)[1], s9.TEST_C[1]        # "W504_p70", "hyb_h5"
D_FAM, D_SRC, D_SCREEN = "D6lag1", "hyb_h1", (0.10, 0.50)
TOL_KONTROL, RVI_MIN = 1e-4, 0.5
NEPRIM = {"stop": "неприменимо: позиции C/D = маска × позиции A из кэша s9, стоп 2σ движок s9 не пересчитывает",
          "razmer": "неприменимо: размер 1/σ_d задаётся в оценщике A, у C/D позиция — маска × A из кэша s9"}


def ustanovit_period(per: str) -> None:
    """Период и зона в процессе расчёта; воркеры build_cache получают период аргументом задачи (s9._job, args[3])."""
    s9.set_period(per)


def kesh(per: str) -> dict:
    p = s9.OUT / f"cache_{s9.cache_key()}.pkl"        # Д5: ключ = код + входы + период
    if not p.exists():
        print("кэша s9 с таким ключом нет — собираю (2 процесса)", flush=True)
        p = s9.build_cache(2)
    return pickle.loads(p.read_bytes())


def portfeli(eng, S, comm: float) -> dict:
    """{'A','C','D'}: порт s9 (excess, exposure) для A, TEST_C и замороженной D; D6lag1 без RVI на днях периода — {'prichina'}."""
    out = {"A": eng.port(eng.A_row, comm=comm)}
    pts1, _ = S.c_points(C_SRC)
    G = dict(pts1)[C_POINT]
    out["C"] = eng.port(eng.A_row * eng.g_row(G), comm=comm)
    pk = pokrytie_rvi(eng)
    if pk < RVI_MIN:        # s9.Engine.rvi режет по CUT=2021-12-31: на test1 RVI нет, ранг NaN ⇒ затвор открыт ⇒ D ≡ A (константа в данных, Д4)
        out["D"] = dict(prichina=f"RVI_(d-1) определён на {pk:.1%} дней периода (< {RVI_MIN:.0%}): D6lag1 вырождается в A")
        return out
    fam = S.d_points(D_SRC)[D_FAM]
    keep = [(lab, G, k) for lab, G, k in fam if D_SCREEN[0] <= eng.calib_closed(G) <= D_SCREEN[1]]
    if not keep:
        out["D"] = dict(prichina="ни одна точка D6lag1 не прошла экран закрытых баров 0,10–0,50")
        return out
    g, _ = eng.select(np.stack([G for _, G, _ in keep]), "mask", D_SCREEN)
    out["D"] = eng.port(eng.A_row * g, comm=comm)
    return out


def pokrytie_rvi(eng) -> float:
    """Доля строк периода (бары пар тикер×Sk) с конечным RVI_{d−1}."""
    return float(np.isfinite(eng.rvi(1)[eng.r["tick"], eng.r["bar"]]).mean())


def schitat(per: str) -> dict:
    """Период берётся из состояния процесса (ustanovit_period); не совпал — отказ, а не молчаливый счёт на другом периоде."""
    if s9.PERIOD != per:
        raise RuntimeError(f"период в процессе {s9.PERIOD!r}, ожидался {per!r}: ustanovit_period не вызван")
    if per == "test1" and not os.environ.get("FASTBT_Z2_R2FINAL"):
        raise SystemExit("test1: не задан FASTBT_Z2_R2FINAL (σ̂ семейства R2 final_d0_s2emb)")
    eng = s9.Engine(kesh(per))
    S = types.SimpleNamespace(e=eng)
    S.c_points = lambda src: s9.S9.c_points(S, src)
    S.d_points = lambda src: s9.S9.d_points(S, src)
    comm = AS.COMM if per == "test1" else s9.COMM
    res = portfeli(eng, S, comm)
    out = {k: (v if isinstance(v, dict) else {"x": v["excess"].to_numpy(float), "delo": (v["exposure"].to_numpy(float) > 0),
                                              "dates": v.index.to_numpy().astype("datetime64[D]"), "sharpe": s9.metrics(v)})
           for k, v in res.items()}
    pk = pokrytie_rvi(eng)
    out["D"]["pokrytie_rvi"] = pk
    out["D"]["bez_rvi_dnej"] = int(round((1 - pk) * len(np.unique(eng.r["bar"]))))
    return out


def primechanie_rvi(pk: float, k: int) -> str:
    """Пустая строка при полном покрытии; иначе — сколько дней затвор D открыт из-за отсутствия ранга RVI."""
    return "" if pk >= 1 else f"на {k} днях без ранга RVI затвор открыт (D = A)"


def tablica(res: dict) -> pd.DataFrame:
    rows = []
    for imya, nazv in (("A", "A (s9, база сравнения)"), ("C", "TEST_C: C1 W504 p70, hyb_h5"), ("D", "D: D6lag1 (выбор по валидации), hyb_h1")):
        r = res[imya]
        if "prichina" in r:
            rows.append(dict(portfel=nazv, stroka="все строки", plata_zaem=0.0, primechanie=f"неприменимо: {r['prichina']}"))
            continue
        zona.check(r["dates"].max(), f"ряд {imya}")                       # ≥ 2025 — отказ
        x, delo = r["x"], r["delo"]
        for k, nm, hv in D.STROKI:
            if k in NEPRIM:
                rows.append(dict(portfel=nazv, stroka=nm, plata_zaem=0.0, primechanie=NEPRIM[k]))
                continue
            rows.append(dict(portfel=nazv, stroka=nm, plata_zaem=0.0, **D.stroka(x, delo, hv, 0.0, np.zeros(len(x)))))
        god = r["dates"].astype("datetime64[Y]").astype(int) + 1970
        for y in sorted(set(god)):
            m = god == y
            rows.append(dict(portfel=nazv, stroka=f"по году {y}", plata_zaem=0.0, **D.stroka(x[m], delo[m], None, 0.0, np.zeros(int(m.sum())))))
    df = pd.DataFrame(rows)
    pk = res["D"].get("pokrytie_rvi")
    if pk is not None:
        m = df.portfel.str.startswith("D:")
        df["pokrytie_rvi"] = np.where(m, pk, np.nan)
        pr = primechanie_rvi(pk, res["D"].get("bez_rvi_dnej", 0))
        if pr:
            if "primechanie" not in df:
                df["primechanie"] = ""
            old = df.loc[m, "primechanie"].fillna("").astype(str)
            df.loc[m, "primechanie"] = np.where(old == "", pr, old + "; " + pr)
    return df


def sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def shapka(per: str) -> str:
    kom = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    s = f"{SHAPKA}; период {per}\n# коммит: {kom}\n"
    if per == "test1":
        vh = [sigma_z2.dev_path(h) for h in sigma_z2.HORIZONS] + [sigma_z2.z2_path(h) for h in sigma_z2.HORIZONS]
        vh += [] if zona.div_net() else list(dividendy._tables())
        s += "".join(f"# вход: {p.name} sha256={sha(p)}\n" for p in vh)
    return s + zona.shapka_dividendy()


def kontrol_train(res: dict) -> float:
    """train: Шарп и годы TEST_C и D против таблиц разработки s9 (10.2026); возвращает макс. |Δ|."""
    rd = lambda n: pd.read_csv(ROOT / "docs" / "tables" / "strat_dev" / n, sep=";", decimal=",")        # noqa: E731
    c, d = rd("s9_C_rezhim.csv"), rd("s9_D_zatvor.csv")
    ob = {"C": c[(c.family == "C1") & (c.point == C_POINT) & (c.sigma == C_SRC)].iloc[0],
          "D": d[(d.family == f"{D_FAM} (выбор по валидации)") & (d.point == "семья") & (d.sigma == D_SRC)].iloc[0]}
    return max(abs(res[k]["sharpe"][m] - float(ob[k][m])) for k in ob for m in ("sharpe", *map(str, s9.YEARS)))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--period", choices=PERIODY, required=True)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)
    if a.period == "train4":
        raise SystemExit("train4: в s9 нет σ-источников (s9.need_sigma), TEST_C/D считаются на train (контроль) и test1")
    ustanovit_period(a.period)      # без внешнего пула: build_cache сам spawn'ит воркеров (daemonic-процесс детей иметь не может)
    res = schitat(a.period)
    df = tablica(res)
    out = a.out or ROOT / "docs" / "tables" / "revers_r1" / ("" if a.period == "test1" else f"_sverka_{a.period}")
    out.mkdir(parents=True, exist_ok=True)
    pk = res["D"].get("pokrytie_rvi")
    rvi_line = "" if pk is None else f"# покрытие RVI для D: {pk:.4f}\n"
    (out / f"a_cd_{a.period}.csv").write_text(shapka(a.period) + rvi_line + df.to_csv(index=False), encoding="utf-8")
    print(df.round(3).to_string())
    if a.period == "train":
        d = kontrol_train(res)
        print(f"контроль train: max|Δ| с таблицами s9 = {d:.2e} (допуск {TOL_KONTROL:.0e})")
        if not d < TOL_KONTROL:
            raise SystemExit("train не воспроизводит таблицы разработки s9")


if __name__ == "__main__":
    main()
