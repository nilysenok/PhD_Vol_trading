"""Драйвер итераций 16–20 Ж-051 (стратегии T16–T20; контракт #376 §2.7, К-T1). Только train 2018–2019, счёт не запускается здесь.

Запуск: FASTBT_MAIN=<копия> python -m fastbt.zh051_iter_t --it 16 [17 ...]
План — строки итерации из `setki_12_16_25.csv` (число = N_iteracii). События — `fastbt.t{NN}_*.sobytiya(inp, osi)`, вход
модуль отдаёт сам (`vhod()`), порог веса — `porog(osi)` (нет — 1,0). Оценка — `t_eval.TEvaluator`. Ledger T{it}: ключ
(("T", konfig), "train"), пишется один раз (`zapisat_ledger`). Центр — плато radius 1, ничья ≤ 0,05 → меньший σ126; досчёт k_s
в центре; контроли (б) сдвиг маски ≥ 63 дней и (а) случайные дни, 20 + 20, вне ledger. PBO — свой и по ledger 16–20.
"""
from __future__ import annotations

import argparse
import importlib
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from fastbt import etap1_poisk_vorota as V, metriki_e as me, sweep_drivers as sd, zh051_kriteriy as K

CSV, OUT = Path("docs/tables/zh051/setki_12_16_25.csv"), Path("docs/tables/zh051")
MODULI = {16: "t16_shok", 17: "t17_szhatie", 18: "t18_div_gap", 19: "t19_rv_povorot", 20: "t20_obyom"}
GODY, N_KONTROL, LAG_MIN = (2018, 2019), 20, 63


@dataclass
class Ctx:
    tev: object      # TEvaluator или двойник с `rows(events_by_t, tau, porog, stop_b, dnevnoy) -> (df, rs, acc)`
    years: dict      # {тикер: массив лет по дням 0..b}
    led: Path = K.LED


def plan(it: int, csv: Path | None = None) -> list:
    """Строки итерации: [{konfig, tip, osi{имя: значение}, k_s}]; число строк обязано равняться N_iteracii."""
    d = pd.read_csv(csv or CSV, comment="#", dtype=str)
    d["iteraciya"], d["N_iteracii"] = d["iteraciya"].astype(int), d["N_iteracii"].astype(int)
    d = d[d["iteraciya"] == it]
    if d.empty or len(d) != int(d["N_iteracii"].iloc[0]):
        raise SystemExit(f"итерация {it}: строк {len(d)} ≠ N_iteracii")
    out = []
    for r in d.itertuples():
        osi = {n: (float(v) if v != "centr" else v) for n, v in ((r.p1_imya, r.p1), (getattr(r, "p2_imya"), r.p2)) if isinstance(n, str) and n}
        osi["tau"] = int(r.tau) if r.tau != "centr" else "centr"
        out.append(dict(konfig=r.konfig, tip=r.tip, osi=osi, k_s=None if r.k_s == "net" else float(r.k_s)))
    return out


def sdelok_goda(rs, years: dict) -> dict:
    """Реальные сделки (`E1Rows.trades`) по годам входа — ворота по ним, а не по `prinyato` (e == b принимается, но не исполняется)."""
    return {y: sum(1 for t, r in rs.items() for tr in r.trades if int(years[t][tr.entry_day]) == y) for y in GODY}


def prinyato_goda(acc: dict, years: dict) -> dict:
    return {y: sum(1 for t, a in acc.items() for (_, e, _, _) in a if int(years[t][e]) == y) for y in GODY}


def ocenka(ctx: Ctx, events: dict, tau: int, porog: float, **kw) -> dict:
    """Строка ledger: excess по годам, mSR по годам, сделки по годам (реальные), принятые по годам (справочно)."""
    df, rs, acc = ctx.tev.rows(events, tau, porog, **kw)
    ex = df["excess"]
    yy = {y: ex.to_numpy()[ex.index.year == y] for y in GODY}
    rd = dict(zip(ctx.years, rs))
    return {"v": "T1.0", "msr_by_year": {y: me.msr(v) for y, v in yy.items()}, "excess_d": {y: list(v) for y, v in yy.items()},
            "sdelok_y": sdelok_goda(rd, ctx.years), "prinyato_y": prinyato_goda(acc, ctx.years)}


def vorota_sdelok(res: dict) -> bool:
    return all(res["sdelok_y"].get(y, 0) >= V.MIN_SDELOK for y in GODY)


def net_ledger(it: int, root: Path) -> None:
    p = root / f"T{it}" / f"ledger_it{it}.pkl"
    if p.exists():
        raise SystemExit(f"{p}: ledger уже есть, не перезаписывается")


def zapisat_ledger(led, it: int, root: Path) -> None:
    """Ledger T{it} не перезаписывается: файл или строка в SHA256SUMS уже есть — стоп."""
    net_ledger(it, root)
    K.save_ledger(f"T{it}", led, f"ledger_it{it}.pkl", root)


def pbo_obshchii(it: int, led, root: Path) -> dict:
    """PBO и N по объединённой матрице: свой ledger + уже записанные ledger других итераций 16–20."""
    cols = dict(led.cache)
    for j in MODULI:
        if j != it and (root / f"T{j}" / f"ledger_it{j}.pkl").exists():
            cols.update(K.load_ledger(f"T{j}", f"ledger_it{j}.pkl", None, root).cache)
    L = V.Ledger(None)
    L.cache = cols
    M = V.matrix(L)
    return dict(pbo_obsh=V.pbo(M), N_obsh=L.n_unique, M=M)


def kontroli(ctx: Ctx, events: dict, tau: int, porog: float, it: int) -> dict:
    """(б) циклический сдвиг дней событий ≥ 63 на тикер; (а) случайные дни того же года с тем же числом событий."""
    obj, out = lambda ev: V.obj(ocenka(ctx, ev, tau, porog)), {"b": [], "a": []}     # noqa: E731
    rng = np.random.default_rng(20261028 + it)
    for lag in rng.choice(np.arange(LAG_MIN, 252), size=N_KONTROL, replace=False):
        n = {t: len(y) for t, y in ctx.years.items()}
        out["b"].append(obj({t: [((d + int(lag)) % n[t], s, x) for d, s, x in ev] for t, ev in events.items()}))
    for i in range(N_KONTROL):
        r, ev2 = np.random.default_rng(1000 * it + i), {}
        for t, ev in events.items():
            yr = ctx.years[t]
            ev2[t] = [(int(r.choice(np.flatnonzero(yr == yr[d]))), s, x) for d, s, x in ev if d < len(yr)]
        out["a"].append(obj(ev2))
    return out


def p_kontrol(ctrl: list, obj_c: float) -> float:
    return (1 + sum(1 for x in ctrl if math.isfinite(x) and x >= obj_c)) / (len(ctrl) + 1)


def tsentr(rows: list, led, keys: list, grid: dict) -> dict | None:
    """Центр плато (radius 1) среди допустимых по сделкам; соседей ≥ 3; ничья ≤ 0,05 → меньший σ126."""
    ok = [dict(r, **r["osi"], objective=V.obj(led.cache[(("T", r["konfig"]), "train")][1])) for r in rows if r["tip"] == "setka"]
    sc = [s for s in sd.pick_by_plateau(dict(rows=ok, keys=keys, grid=grid), k=len(ok), radius=1, min_neighbors=1)
          if vorota_sdelok(led.cache[(("T", s["konfig"]), "train")][1]) and s["neighbors"] - 1 >= 3 and math.isfinite(s["plateau"])]
    if not sc:
        return None
    top = max(s["plateau"] for s in sc)
    return min((s for s in sc if s["plateau"] >= top - V.TIE), key=lambda s: V.sigma126(led.cache[(("T", s["konfig"]), "train")][1]))


def run_it(it: int, ctx: Ctx, mod, out: Path = OUT) -> dict:       # xtask: длинная — один сквозной проход итерации (план→ledger→центр→контроли→PBO)
    rows = plan(it)
    inp = mod.vhod()
    ev = lambda osi: _by_t(mod.sobytiya(inp, osi))                                   # noqa: E731
    por = lambda osi: float(mod.porog(osi)) if hasattr(mod, "porog") else 1.0        # noqa: E731
    led, sid = V.Ledger(None), f"T{it}"
    net_ledger(it, ctx.led)
    for r in rows:
        if r["tip"] == "setka":
            o = r["osi"]
            K.add_cfg(led, ("T", r["konfig"]), o, ocenka(ctx, ev(o), o["tau"], por(o)))
    keys = [k for k in rows[0]["osi"]]
    grid = {k: sorted({r["osi"][k] for r in rows if r["tip"] == "setka"}) for k in keys}
    c = tsentr(rows, led, keys, grid)
    itog = dict(strategiya=sid, iteraciya=it, N_plan=len(rows), centr=c["konfig"] if c else "", proshla=False)
    if c:
        o = {k: c[k] for k in keys}
        for r in (r for r in rows if r["tip"] == "doschet"):
            K.add_cfg(led, ("T", r["konfig"]), dict(o, k_s=r["k_s"]), ocenka(ctx, ev(o), o["tau"], por(o), stop_b=r["k_s"]))
        res, e = led.cache[(("T", c["konfig"]), "train")][1], ev(o)
        k = kontroli(ctx, e, o["tau"], por(o), it)
        d = ocenka(ctx, e, o["tau"], por(o), dnevnoy=True)
        s_ok = all(np.max(np.abs(np.asarray(res["excess_d"][y]) - np.asarray(d["excess_d"][y]))) <= 1e-12 for y in GODY)
        itog.update(obj=V.obj(res), p_kontrol_b=p_kontrol(k["b"], V.obj(res)), p_kontrol_a=p_kontrol(k["a"], V.obj(res)),
                    G2_sdelki=vorota_sdelok(res), G3_plato=True, G5_dnevnoy=bool(s_ok), **{f"sdelok_{y}": res["sdelok_y"][y] for y in GODY})
    po = pbo_obshchii(it, led, ctx.led)
    M = V.matrix(led)
    itog.update(pbo_svoi=V.pbo(M), N_svoi=led.n_unique, pbo_obsh=po["pbo_obsh"], N_obsh=po["N_obsh"])
    itog["G1_pbo"] = bool(itog["pbo_svoi"] < V.PBO_MAX)
    if c:
        itog["proshla"] = all(bool(itog[g]) for g in ("G1_pbo", "G2_sdelki", "G3_plato", "G5_dnevnoy")) and max(
            itog["p_kontrol_b"], itog["p_kontrol_a"]) <= 0.1
    zapisat_ledger(led, it, ctx.led)
    K.write_table(out / f"it{it}" / f"T{it}_itog.csv", [itog])
    return itog


def _by_t(evs) -> dict:
    out = {}
    for t, d, side, sila in evs:
        out.setdefault(t, []).append((int(d), int(side), float(sila)))
    return out


def build_ctx() -> Ctx:
    """Настоящий запуск: входы этапа 1 (до 2019), движок train; порядок `years` = порядок тикеров `TEvaluator`."""
    from fastbt import etap1, etap1_eval as EE                    # noqa: PLC0415  (данные — только при настоящем запуске)
    from fastbt.etap1_data import _one_bar                        # noqa: PLC0415
    from fastbt.t_eval import TEvaluator, TInput                  # noqa: PLC0415
    inp = etap1.build_inputs("S1")
    ev = EE.Evaluator("S1", inp, etap1.base_point("S1"))
    xs, years = {}, {}
    for t in inp.sk.tickers:
        b = ev._b(t, "train")
        dates = inp.meta[t]["dates"][:b + 1]
        days = inp.bars[t][:b] + [_one_bar(float(inp.close_n6[t][b]))]
        a = int(np.searchsorted(dates, np.datetime64("2018-01-01")))
        xs[t] = TInput(days, inp.close_n6[t][:b + 1], inp.sigma[EE.YAKOR["sigma"]][t][:b + 1], inp.tick[t], a, b)
        years[t] = pd.DatetimeIndex(dates).year.to_numpy()
    return Ctx(TEvaluator(xs, ev._engine("train")[0]), years)


def main(argv=None) -> list:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--it", type=int, nargs="+", required=True, choices=sorted(MODULI))
    a = ap.parse_args(argv)
    ctx = build_ctx()
    return [run_it(i, ctx, importlib.import_module(f"fastbt.{MODULI[i]}")) for i in a.it]


if __name__ == "__main__":
    main()
