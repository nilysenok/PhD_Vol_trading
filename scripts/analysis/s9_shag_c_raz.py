"""Режим `--tolko-razlozhenie` для s9_shag_c.py: пересчитать только shag{N}_razlozhenie.csv.

Сетка и отбор НЕ пересчитываются: выбранная (или Шарп-лидер) точка берётся из готовых shag{N}_otbor.csv / _setka.csv,
затвор строится заново только для неё, A/к3 — те же `Base`. Проверка сходимости — к рядам сетки:
Δ к A и Δ к к3 пересчитываются тем же `evaluate` (то есть тем же кодом, что строит сетку), допуск 1e-6;
плюс сверка с напечатанной сеткой (4 знака, допуск 6e-5): отбор и точка совпадают с файлами."""
from __future__ import annotations

import pandas as pd

CSV = dict(sep=";", decimal=",", encoding="utf-8-sig")


def find_point(otbor: pd.DataFrame, setka: pd.DataFrame, t: str):
    """(строка сетки, выбрана ли) по готовым файлам: «ВЫБРАНА при 0,04», иначе Шарп-лидер при 0,04."""
    sel = otbor[otbor["rol"].astype(str).str.startswith("ВЫБРАНА")]
    if len(sel):
        s = sel.iloc[0]
        m = (setka["point"] == s["point"]) & (setka["h"] == s["h"])
        return setka[m].iloc[0], True
    return setka.loc[setka[f"sharpe_{t}"].idxmax()], False


def run(sigma: str, shag: int, reps: int, workers: int, out) -> None:
    from fastbt import fast_grid, fast_signals, s9
    from fastbt import shag_c as C
    from fastbt import shag_c_tab as T
    import s9_shag_c as M
    grid = C.load_grid()
    otb, setka = (pd.read_csv(out / f"shag{shag}_{n}.csv", **CSV) for n in ("otbor", "setka"))
    row, vybrana = find_point(otb, setka, C.tag(C.COMM_SEL))
    s9.set_scope("dev")
    s9.use_v2(C.COMM_SEL)
    s9.C_DOP = {"rvw": grid["rvw"]}
    s9.OUT.mkdir(parents=True, exist_ok=True)
    cache = fast_signals.build(workers)
    with fast_grid.patched():
        eng = s9.Engine(cache)
        bases = {c: C.Base(eng, c) for c in C.COMM_MAIN}
        src = next(p["src"] for p in C.plan(sigma, grid) if p["h"] == row["h"] and p["family"] == row["family"])
        g = eng.g_row(C.gates(eng, src, grid, sigma == "naive")[(row["family"], row["label"])])
        raz = T.razlozhenie(eng, bases, g, reps)
        fresh = {}                                    # те же числа, что строит сетка (evaluate)
        for c, b in bases.items():
            e = C.evaluate(b, g, reps, uslovie1=grid.get("uslovie1", C.USLOVIE1))
            fresh.update({f"d_vs_A_{C.tag(c)}": e["d_vs_A"], f"d_vs_k3_{C.tag(c)}": e["d_vs_k3"]})
    T.proverka_razlozheniya(raz, fresh)
    T.proverka_razlozheniya(raz, row, tol=6e-5)       # напечатанная сетка округлена до 4 знаков
    for r in raz:
        r["vybrana"] = vybrana
    M.write(raz, f"shag{shag}_razlozhenie.csv", out)
    print(f"шаг {shag}: shag{shag}_razlozhenie.csv пересчитан, точка {row['point']} h={row['h']}, сумма сходится с сеткой")
