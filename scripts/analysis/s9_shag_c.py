"""Шаги 1–3 главы 4 (v3): сетка C на одной σ → docs/tables/strat_dev_v3/shag{N}_*.csv + каркас отчёта.

    python3 scripts/analysis/s9_shag_c.py --sigma naive|harj|hyb --shag 1|2|3 [--dry-run] [--reps 10000] [--workers 4]
Пары: naive — шаг 1, harj — шаг 2, hyb — шаг 3. --dry-run: число точек и план, данные не читаются, ничего не считается.
Без аргументов (и --smoke): дымовой прогон на СИНТЕТИКЕ (tests/test_s9_shag_c.py), в папку Python tempfile.
Реальный счёт — только по разрешению владельца на шаг; быстрый слой (fast_grid.patched), A/к3/к2 — fastbt.shag_c.
Выходы: shagN_setka / otbor / ustoychivost / razlozhenie / komissiya / sdvig / sverka_sloy .csv, shagN_zhurnal.txt
(первая строка — проверка сдвига σ), docs/otchety/2026-10-27_gl4_shag_N.md. Пометка: «разработка 2018–2021, после аудита».
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fastbt import shag0  # noqa: E402
from fastbt import shag_c as C  # noqa: E402
from fastbt import shag_c_tab as T  # noqa: E402

OUT = Path(os.environ.get("SHAG_C_OUT", ROOT / "docs" / "tables" / "strat_dev_v3"))
SDVIG = ROOT / "data" / "regen" / "ch4_dev" / "proverka_sdvig.csv"
NAMES = {"naive": "наивная √RV_w", "harj": "HAR-J (б)", "hyb": "гибрид"}


def write(rows, name: str, out: Path) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    df = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    p = out / name
    df.to_csv(p, sep=";", decimal=",", index=False, float_format="%.4f", encoding="utf-8-sig")
    return p


def dry_run(sigma: str, shag: int) -> None:
    grid = C.load_grid()
    pts = C.plan(sigma, grid)
    by = pd.DataFrame(pts).groupby(["src", "family"]).size()
    print(f"шаг {shag}, σ = {NAMES[sigma]}: {len(pts)} точек (yaml: {grid['sigmy'][sigma]['tochek']})")
    print(by.to_string())
    print("комиссии: основные 0,035 и 0,04; справочно для отбора 0,05; к2: 1000 сдвигов, seed", C.SEED_K2)
    print("таблицы:", ", ".join(f"shag{shag}_{n}.csv" for n in ("setka", "otbor", "ustoychivost", "razlozhenie",
                                                                "komissiya", "sdvig", "sverka_sloy")))
    print("данные σ не читались, ничего не считалось")


def sverka(eng, sel_pts: list, comm: float) -> list:
    """Быстрый слой (Engine.port) против медленного (msp.portfolio) на 5 точках: метрики до 4 знаков, max|Δ| excess."""
    from s9_shag0 import slow_port
    rows = []
    for lab, pos in sel_pts:
        a, b = eng.port(pos, comm=comm), slow_port(eng, pos, comm).reindex(eng.dates)
        f = lambda p: pd.DataFrame([shag0.stats(p, np.ones(len(p), bool))]).to_csv(sep=";", float_format="%.4f")  # noqa: E731
        rows.append(dict(pometka=shag0.POMETKA, row=lab, comm_pct=comm * 100, metriki_do_4_znaka=f(a) == f(b),
                         max_abs_excess="%.1e" % float(np.abs(a["excess"].values - b["excess"].values).max())))
    return rows


def tables(eng, sigma: str, grid: dict, reps: int, n_shift: int = 1000, with_sverka: bool = True) -> dict:
    df, gs, bases = C.run_setka(eng, sigma, grid, reps, n_shift=n_shift)
    otb, best = T.otbor(df)
    top = best if best is not None else df.loc[df[f"sharpe_{C.tag(C.COMM_SEL)}"].idxmax()]
    g = eng.g_row(gs[(top["h"], top["family"], top["label"])])
    raz = T.razlozhenie(eng, bases, g, reps)
    T.proverka_razlozheniya(raz, top)
    for r in raz:
        r["vybrana"] = best is not None
    res = dict(setka=df, otbor=otb, ustoychivost=T.ustoychivost(df, best), razlozhenie=raz,
               komissiya=T.komissiya(eng, g, reps, uslovie1=grid.get("uslovie1", C.USLOVIE1)), best=best, top=top)
    if with_sverka:
        idx = np.linspace(0, len(df) - 1, 5).round().astype(int)
        pts = [(df.loc[i, "point"] + f" h{df.loc[i, 'h']}", np.asarray(eng.A_row) * eng.g_row(
            gs[(df.loc[i, "h"], df.loc[i, "family"], df.loc[i, "label"])])) for i in idx]
        res["sverka_sloy"] = sverka(eng, pts, C.COMM_SEL)
    return res


def otchet(shag: int, sigma: str, res: dict, sd_line: str) -> str:
    df, top, best = res["setka"], res["top"], res["best"]
    t4, t35 = C.tag(0.0004), C.tag(0.00035)
    n_pass = int(df[f"proshla_c8v2_{t4}"].sum())
    pt = f"{top['point']} h={top['h']}" + ("" if best is not None else " (не отобрана: Шарп-лидер, справочно)")
    return f"""# Глава 4, шаг {shag} — C на σ: {NAMES[sigma]} (разработка 2018–2021)

Каркас: числа подставлены из `docs/tables/strat_dev_v3/shag{shag}_*.csv`; пометка «разработка 2018–2021, после аудита». Тест не открывался.

## 1. Проверка сдвига входов
{sd_line}. Таблица — `shag{shag}_sdvig.csv`.

## 2. Что считалось
Сетка `fastbt/designs/s9/setka_C_v3.yaml`, точек: {len(df)}; комиссия 0,035 и 0,04 % (отбор — при 0,04; справочно 0,05); к2 — 1 000 сдвигов, seed {C.SEED_K2}; ЛВ — блок 5, 10 000.

## 3. Таблица результатов
Прошли С8 v2 при 0,04: {n_pass} из {len(df)}. Точка: {pt}; Шарп 0,04: {top[f'sharpe_{t4}']:.3f}, 0,035: {top[f'sharpe_{t35}']:.3f};
Δ к A (0,04): {top[f'd_vs_A_{t4}']:.3f}; Δ к к3: {top[f'd_vs_k3_{t4}']:.3f} (p {top[f'p_vs_k3_{t4}']:.3f}); p к к2: {top[f'p_vs_k2_{t4}']:.3f}.
Файлы: `shag{shag}_setka.csv`, `shag{shag}_otbor.csv`, `shag{shag}_razlozhenie.csv`, `shag{shag}_komissiya.csv`.

## 4. Таблица устойчивости
`shag{shag}_ustoychivost.csv` (соседи ±1 по W и p, допуск 0,1 Шарпа; p50 — край по построению).

## 5. Что не проверено / оговорки
Проскальзывание не моделируется; 0,05 и 0,06 % — приложение. Дизайны с внутренним выбором (u02) не кандидаты.

## 6. Случайные просмотры
(заполняет агент шага)

## 7. Хэши входов и коммит кода
(заполняет агент шага: SHA256.txt входов, `git rev-parse HEAD`)
"""


def real(sigma: str, shag: int, reps: int, workers: int, out: Path, uslovie1: str | None = None) -> None:
    from fastbt import fast_grid, fast_signals, s9
    grid = C.load_grid()
    grid["uslovie1"] = uslovie1 or grid.get("uslovie1", C.USLOVIE1)
    sd = T.sdvig(SDVIG, sigma)
    line = T.sdvig_line(sd)
    if "НЕ ПРОЙДЕНА" in line:
        raise SystemExit(line)
    s9.set_scope("dev")
    s9.use_v2(C.COMM_SEL)
    s9.C_DOP = {"rvw": grid["rvw"]}
    s9.OUT.mkdir(parents=True, exist_ok=True)
    cache = fast_signals.build(workers)
    with fast_grid.patched():
        eng = s9.Engine(cache)
        C.check_sigma_distinct(eng, {pt["src"] for pt in C.plan(sigma, grid)})
        res = tables(eng, sigma, grid, reps)
    save(res, sigma, shag, out, sd, line)


def save(res: dict, sigma: str, shag: int, out: Path, sd: pd.DataFrame, line: str) -> None:
    for k in ("setka", "otbor", "ustoychivost", "razlozhenie", "komissiya", "sverka_sloy"):
        if k in res:
            write(res[k], f"shag{shag}_{k}.csv", out)
    write(sd, f"shag{shag}_sdvig.csv", out)
    sv = res.get("sverka_sloy", [])
    (out / f"shag{shag}_zhurnal.txt").write_text(
        line + "\n" + f"точек: {len(res['setka'])}; сверка слоёв (5 точек): "
        + ", ".join(f"{r['row']}: {r['metriki_do_4_znaka']}, max|Δ| {r['max_abs_excess']}" for r in sv) + "\n",
        encoding="utf-8")
    rep = out.parent.parent / "otchety" / f"2026-10-27_gl4_shag_{shag}.md" if out == OUT else out / "otchet.md"
    rep.parent.mkdir(parents=True, exist_ok=True)
    rep.write_text(otchet(shag, sigma, res, line), encoding="utf-8")
    print(line, "→", out)


def smoke() -> None:
    sys.path.insert(0, str(ROOT / "tests"))
    from test_s9_shag_c import mini_grid, world
    eng = world(seed=5)
    with tempfile.TemporaryDirectory() as d:
        res = tables(eng, "hyb", mini_grid(), 200, n_shift=50)
        line = "дымовой прогон на синтетике (реальных σ нет)"
        save(res, "hyb", 3, Path(d), pd.DataFrame(dict(ryad=["синтетика"], ok=[True])), line)
        print(sorted(p.name for p in Path(d).iterdir()))
    print("OK")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sigma", choices=sorted(C.SHAG))
    ap.add_argument("--shag", type=int, choices=(1, 2, 3))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--tolko-razlozhenie", action="store_true",
                    help="пересчитать только shagN_razlozhenie.csv (сетка и отбор — из готовых файлов)")
    ap.add_argument("--reps", type=int, default=10000)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--c8-uslovie1", choices=C.USLOVIYA1, default=None,
                    help="условие 1 С8 v2: strogo (4/4, умолчание yaml) | odno_isklyuchenie (3/4)")
    a = ap.parse_args()
    if a.smoke or len(sys.argv) == 1:
        smoke()
    elif a.sigma is None or a.shag is None or C.SHAG[a.sigma] != a.shag:
        ap.error("нужны --sigma и --shag в паре: naive-1, harj-2, hyb-3")
    elif a.dry_run:
        dry_run(a.sigma, a.shag)
    elif a.tolko_razlozhenie:
        import s9_shag_c_raz
        s9_shag_c_raz.run(a.sigma, a.shag, a.reps, a.workers, OUT)
    else:
        real(a.sigma, a.shag, a.reps, a.workers, OUT, a.c8_uslovie1)
