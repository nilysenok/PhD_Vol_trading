"""Шаг 3′ разработки главы 4 (v3): D и B, 2018–2021 → docs/tables/strat_dev_v3/shag3D_*.csv, shag3B_*.csv.

    FASTBT_MAIN=<своя копия> python3 scripts/analysis/s9_shag_db.py --podhod D|B [--sigma naive|harj|hyb] [--reps 10000]
    python3 scripts/analysis/s9_shag_db.py --podhod B --dry-run     # план и число точек; σ и RVI не читаются
Сетки — fastbt/designs/s9/setka_{D,B}_v3.yaml (коммит до счёта). Отбор — С8 v2 при 0,04 %; те же точки при 0,035;
0,05/0,06 — в komissiya; справочно независимый выбор при 0,05 (vybor005, флаг sovpal). D: семьи С9 с
RVI_{d−1} (D6lag1; D6 с RVI_d не входит). Условие 3 (к2) — для всех, и для B, D7 (масштаб): круговой сдвиг
ряда экспозиции по тикерам (Ъ 27.10). Таблицы: setka, otbor, ustoychivost, komissiya, vybor005,
razlozhenie (+ shag3B_itog). Самопроверка без аргументов: python3 s9_shag_db.py
"""
from __future__ import annotations

import argparse
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import s9_shag_common as C  # noqa: E402

POMETKA = "разработка 2018–2021, после аудита, v3.2 (Н6, ряд до 18:50)"


def sdvig_zhurnal(podhod: str, sg: str, path=None) -> str:
    """Обязательная проверка сдвига σ перед счётом: тот же `sdvig_line`, то же правило #274 (блокирует только
    дневная цель), что в s9_shag_3b_dop.py. Строка — первой в журнале shag3{D|B}_zhurnal_{sg}.txt; «НЕ ПРОЙДЕНА» — стоп."""
    import os
    from fastbt import shag_c_tab as T
    p = path or Path(os.environ.get("FASTBT_MAIN", C.ROOT)) / "data" / "regen" / "ch4_dev" / "proverka_sdvig.csv"
    line = T.sdvig_line(T.sdvig(p, sg))
    C.OUT.mkdir(parents=True, exist_ok=True)
    (C.OUT / f"shag3{podhod}_zhurnal_{sg}.txt").write_text(line + "\n", encoding="utf-8")
    if "НЕ ПРОЙДЕНА" in line:
        raise SystemExit(line)
    return line


def write(rows, name: str) -> Path:
    C.OUT.mkdir(parents=True, exist_ok=True)
    p = C.OUT / name
    df = pd.DataFrame(rows)
    df.insert(0, "pometka", POMETKA)
    df.to_csv(p, sep=";", decimal=",", index=False, float_format="%.4f", encoding="utf-8-sig")
    return p


def points(podhod: str, eng, src: str, s9):
    """[(семья, метка, G, масштаб?, calib_closed)] для σ-ряда src; экрана по доле закрытых дней нет."""
    shim, out = types.SimpleNamespace(e=eng), []
    if podhod == "D":
        fams = s9.S9.d_points(shim, src)
        fams.pop("D6")                                    # RVI_d (lag 0) — не кандидат
        cand = [(f, lab, G, kind == "scale") for f, pts in fams.items() for lab, G, kind in pts]
    else:
        y = yaml.safe_load((C.DESIGNS / "setka_B_v3.yaml").read_text(encoding="utf-8"))
        cand = [("B", f"cap{cap:g}_band{int(band * 100)}", s9.S9.b_weights(shim, src, cap, band), True)
                for cap in y["cap"] for band in y["band"]]
    for f, lab, G, scale in cand:                         # экран закрытых дней снят (Ъ 27.10, 15:10): все точки идут в С8 v2
        out.append((f, lab, G, scale, eng.calib_closed(G)))
    return out


def run_grid(podhod, eng, s9, series, comm, reps):
    from fastbt import fast_kontrol as fk
    base, K = C.make_base(eng, comm, reps, C.uslovie1_yaml(podhod)), fk.Kontrol(eng, comm)
    C.check_exit_fee(eng, eng.A_row, comm)
    C.check_shift(K)                                  # лаги; сдвиг 0 — внутри eval_point на каждой точке
    rows, setka = [], []
    for src in series:
        for f, lab, G, scale, cc in points(podhod, eng, src, s9):
            setka.append(dict(family=f, point=lab, sigma=src, calib_closed=cc))
            g = eng.g_row(G)
            r = C.eval_point(eng, K, base, eng.A_row * g, g)
            r.update(family=f, point=lab, sigma=src, comm_pct=comm * 100, calib_closed=cc)
            rows.append(r)
    return pd.DataFrame(rows), setka, base


def run(podhod: str, sigma: str | None, reps: int, eng=None) -> None:
    """eng — движок (в тестах синтетический); иначе боевой из кэша."""
    from fastbt import s9
    if eng is None:
        s9.use_v2(C.COMM_MAIN)
        eng = s9.Engine(s9.load_cache())
    tag = f"shag3{podhod}"
    for sg in ([sigma] if sigma else list(C.SIGMAS)):
        print(sdvig_zhurnal(podhod, sg), flush=True)      # первой строкой журнала, до счёта
        series = C.SIGMAS[sg]
        d4, setka, base = run_grid(podhod, eng, s9, series, C.COMM_MAIN, reps)
        d35, _, _ = run_grid(podhod, eng, s9, series, C.COMM_ALT, reps)
        d5, _, _ = run_grid(podhod, eng, s9, series, C.COMM_REF, reps)
        write(setka, f"{tag}_setka_{sg}.csv")
        write(pd.concat([d4, d35]).to_dict("records"), f"{tag}_otbor_{sg}.csv")
        sel = C.select(d4)
        write([C.select_ref(d5, sel)], f"{tag}_vybor005_{sg}.csv")
        best = sel if sel is not None else d4.sort_values("sharpe", ascending=False).iloc[0]
        if sel is not None:
            write(C.ustoychivost(d4, sel).to_dict("records"), f"{tag}_ustoychivost_{sg}.csv")
        extra(podhod, eng, s9, best, sel is not None, series, base, reps, tag, sg)
        print(sg, "прошло С8 v2:", int(d4["passed_v2"].sum()), "из", len(d4), flush=True)
    if podhod == "B":
        itog_b(tag)


def extra(podhod, eng, s9, best, chosen, series, base, reps, tag, sg) -> None:
    """komissiya (4 комиссии), razlozhenie и доля закрытых дней по семьям S1–S6 — для выбранной точки
    (нет прошедших — для лучшей по Шарпу, пометка vybrana=False)."""
    from fastbt import fast_kontrol as fk
    pts = {(f, lab): (G, sc) for f, lab, G, sc, _ in points(podhod, eng, best["sigma"], s9)}
    G, _ = pts[(best["family"], best["point"])]
    g = eng.g_row(G)
    rows = []
    for comm in (C.COMM_MAIN, C.COMM_ALT, *C.COMM_APP):
        b = C.make_base(eng, comm, reps, C.uslovie1_yaml(podhod))
        r = C.eval_point(eng, fk.Kontrol(eng, comm), b, eng.A_row * g, g)
        rows.append(dict(family=best["family"], point=best["point"], sigma=best["sigma"], vybrana=chosen,
                         comm_pct=comm * 100, **{k: r[k] for k in r if k.startswith(("sharpe", "d_vs", "p_vs", "usl", "u1_", "passed"))}))
    write(rows, f"{tag}_komissiya_{sg}.csv")
    p = eng.port(eng.A_row * g, comm=C.COMM_MAIN)
    rz = C.razlozhenie(base, best, p, reps)
    write(rz + C.closed_by_sid(eng, g), f"{tag}_razlozhenie_{sg}.csv")


def itog_b(tag: str) -> None:
    passed = {}
    for sg in C.SIGMAS:
        f = C.OUT / f"{tag}_otbor_{sg}.csv"
        if f.exists():
            d = pd.read_csv(f, sep=";", decimal=",", encoding="utf-8-sig")
            passed[sg] = int(d[d["comm_pct"].round(4) == 0.04]["passed_v2"].astype(bool).sum())
    write(C.b_itog(passed).to_dict("records"), f"{tag}_itog.csv")


def dry_run(podhod: str, sigma: str | None) -> dict:
    pl = C.plan(podhod, sigma)
    print(f"[dry-run] шаг 3′ {podhod}: ряды σ {pl['ryady']}; точек на ряд {pl['na_ryad']}; всего {pl['vsego']}")
    print(f"[dry-run] комиссии: отбор {pl['komissii']['osnovnaya_otbor']}, пересчёт {pl['komissii']['pereschet']}, "
          f"приложение {pl['komissii']['prilozhenie']}; таблицы {C.OUT}/shag3{podhod}_*; σ и RVI не читаются")
    return pl


def selftest() -> None:
    assert C.plan("D")["vsego"] == 396 and C.plan("B")["vsego"] == 54
    assert C.plan("B", "hyb")["vsego"] == 18 and C.plan("D", "naive")["na_ryad"] == 66
    assert C.axes("ema3_W126_p70") == dict(ema=3, W=126, p=70) and C.axes("cap1.5_band10") == dict(cap=1.5, band=10)
    assert C.b_itog({"naive": 0, "harj": 0, "hyb": 0}).iloc[-1]["vyvod"] == C.B_NEGATIVE
    print("OK")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--podhod", choices=["D", "B"])
    ap.add_argument("--sigma", choices=list(C.SIGMAS))
    ap.add_argument("--reps", type=int, default=10000)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--tolko-zhurnal", action="store_true", help="только проверка сдвига → zhurnal, без счёта")
    a = ap.parse_args()
    if a.podhod is None:
        selftest()
    elif a.tolko_zhurnal:
        for s in ([a.sigma] if a.sigma else list(C.SIGMAS)):
            print(s, sdvig_zhurnal(a.podhod, s))
    elif a.dry_run:
        dry_run(a.podhod, a.sigma)
    else:
        run(a.podhod, a.sigma, a.reps)
