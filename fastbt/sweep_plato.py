"""Режим `plato` Э1-в (Ж-031, толкование 27.10 19:35): топ-3 по плато, а не по пику, на достроенных соседях.

1. Из всех проб ledger стратегии (настройка 2018, комиссия отбора) берутся топ-10 по цели, допустимые по активности:
   сделок/год (`sdelok_y`: стороны сделок) ≥ `min_frac` × база tune 2018. Порядок просмотра — по дневному SR пробы: цель на одном
   годе настройки = Шарп 2018, порядок тот же; цель кандидата пересчитывается оценщиком.
2. Для каждого достраиваются соседи ±1 узел сетки по каждому параметру с эффектом (размах цели в торнадо ≥ 0,05) — OAT вокруг
   кандидата. Соседи считаются тем же `Sweep.evaluate`, пробы идут в ledger и в N. Сосед, равный по канону уже посчитанной
   точке (неактивные параметры сведены к базе, `make_fix`), берётся из ledger без новой пробы.
3. Плато = среднее цели точки и посчитанных соседей; соседей < 3 — край (`edge`), в топ-3 не берётся.
Выход: `S{k}_plato_{c0}.csv` (все кандидаты), строки `mode=plato` в `S{k}_top3.csv` (строки других режимов не стираются).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd

from fastbt import sweep_drivers as sd
from fastbt.sweep import Sweep, make_evaluator
from fastbt.sweep_pts import MIN_NB, TOR_MIN, effect_params, make_fix
from fastbt.sweep_pts import point_cols
from fastbt.sweep_run import header, ledger_dir, load, periods
from fastbt.sweep_spec import Spec, pkey

TOP_N, TOP_K = 10, 3


def admissible(res: dict, min_sd: float) -> bool:
    """Допустима ли проба по активности: нет оценки сделок (NaN) — недопустима."""
    return res["sdelok_y"] >= min_sd


def plateau_value(obj: float, nb: list) -> float:
    """Плато: среднее цели точки и её соседей."""
    return sum([obj, *nb]) / (1 + len(nb))


def neighbours(exp: dict, p: dict, keys: list, fix) -> list:
    """Канонические точки ±1 узел по каждому параметру `keys`; совпавшие с p или между собой по канону (неактивные) пропускаются."""
    seen, out = {pkey(p)}, []
    for k in keys:
        vals = list(exp["grid"][k])
        if p.get(k) not in vals:
            continue
        i = vals.index(p[k])
        for j in (i - 1, i + 1):
            q = fix({**p, k: vals[j]}) if 0 <= j < len(vals) else None
            if q is not None and pkey(q) not in seen:
                seen.add(pkey(q))
                out.append(q)
    return out


def canon_index(sw: Sweep, exp: dict, fix) -> dict:
    """{ключ канонической точки: точка в том виде, как лежит в ledger} — повтор по канону берётся из ledger, а не пишется заново."""
    idx = {}
    for k in sw.ledger:
        p = dict(json.loads(k))
        idx.setdefault(pkey(fix({**exp["base"], **p})), p)
    return idx


def candidates(sw: Sweep, exp: dict, per: tuple, c0: float, min_sd: float, fix, n: int = TOP_N) -> list:
    """Топ-n допустимых проб ledger: dict(p канон., p_led как в ledger, res, obj)."""
    out, seen = [], set()
    order = sorted(sw.ledger.items(), key=lambda kv: -(kv[1] if kv[1] == kv[1] else float("-inf")))
    for k, _ in order:
        p_led = dict(json.loads(k))
        p = fix({**exp["base"], **p_led})
        if pkey(p) in seen:
            continue
        seen.add(pkey(p))
        res = sw.evaluate(p_led, per[0], c0)
        if admissible(res, min_sd):
            out.append(dict(p=p, p_led=p_led, res=res, obj=sd.objective_stable(res)))
        if len(out) == n:
            break
    return out


def add_plateau(sw: Sweep, exp: dict, per: tuple, c0: float, keys: list, fix, cands: list) -> int:
    """Считает соседей каждого кандидата; кладёт plateau, n_nb, nb_min, edge. Возвращает число посчитанных соседних точек."""
    idx, n_pts = canon_index(sw, exp, fix), 0
    for c in cands:
        nb = []
        for q in neighbours(exp, c["p"], keys, fix):
            o = sd.objective_stable(sw.evaluate(idx.get(pkey(q), q), per[0], c0))
            n_pts += 1
            if o == o:
                nb.append(o)
        c.update(plateau=plateau_value(c["obj"], nb), n_nb=len(nb), nb_min=min(nb) if nb else float("nan"), edge=len(nb) < MIN_NB)
    return n_pts


def table(sw: Sweep, cands: list, per: tuple, comms: tuple, trials: list) -> pd.DataFrame:
    """Строки кандидатов по плато (убывание): метрики при комиссии отбора, tune/ctrl при остальных, DSR(N = len(trials))."""
    rows = []
    for c in sorted(cands, key=lambda c: -c["plateau"]):
        m = point_cols(sw, c["p_led"], per, comms[0])
        r = dict(mode="plato", plateau=c["plateau"], neighbors=c["n_nb"], nb_min_objective=c["nb_min"], edge=c["edge"],
                 objective=m["objective"], tune_sharpe=m["tune_sharpe"], ctrl_sharpe=m["ctrl_sharpe"],
                 sdelok_y=m["tune_sdelok_y"], N=len(trials), comm_pct=comms[0],
                 dsr=sd.dsr(m["sr_d"], trials, m["T"], m["skew"], m["kurt"])["dsr"])
        for cc in comms[1:]:
            x = point_cols(sw, c["p_led"], per, cc)
            r.update({f"tune_sharpe_c{cc:g}": x["tune_sharpe"], f"ctrl_sharpe_c{cc:g}": x["ctrl_sharpe"]})
        rows.append(dict(r, **{f"p.{k}": v for k, v in c["p"].items()}))
    return pd.DataFrame(rows)


def write_top3(out: Path, name: str, h: str, t: pd.DataFrame):
    f = out / f"{name}_top3.csv"
    new = t[~t["edge"]].head(TOP_K).copy()
    new.insert(0, "rank", range(1, len(new) + 1))
    old = pd.read_csv(f, comment="#") if f.exists() else pd.DataFrame()
    old = old[old["mode"] != "plato"] if len(old) else old
    f.write_text(h + pd.concat([x for x in (old, new.drop(columns="edge")) if len(x)], ignore_index=True).to_csv(index=False), "utf-8")


def run_plato(name: str, setka, o: dict, out_dir, evaluator=None) -> dict:
    spec_d, exp, raw = load(setka)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    spec = Spec(**spec_d, out=ledger_dir(out, name))
    sw = Sweep(spec, evaluator or make_evaluator(spec))
    per, comms, fix, t0 = periods(raw), spec.comm_pct, make_fix(exp), time.time()
    c0, n0 = comms[0], len(sw.ledger)
    thr = TOR_MIN if o.get("tornado_min") is None else o["tornado_min"]
    keys = effect_params(exp, out, name, c0, thr)
    min_sd = (o.get("min_frac") or 0.0) * sw.evaluate(exp["base"], per[0], c0)["sdelok_y"]
    cands = candidates(sw, exp, per, c0, min_sd, fix)
    n_pts = add_plateau(sw, exp, per, c0, keys, fix, cands)
    sw.save()
    t = table(sw, cands, per, comms, sw.trials_sr())
    h = (header(name, raw, sw.ev, sw) + f"# режим plato; кандидатов {len(cands)}; параметров с эффектом {len(keys)}; соседних точек {n_pts}; "
         f"предел активности {min_sd:.3f} сделок/год; проб в ledger до {n0}, после {len(sw.ledger)}\n")
    if len(t):
        (out / f"{name}_plato_{c0:g}.csv").write_text(h + t.to_csv(index=False), "utf-8")
        write_top3(out, name, h, t)
    return dict(table=t, ledger=sw.ledger, n_before=n0, n_neighbors=n_pts, min_sd=min_sd, secs=time.time() - t0)
