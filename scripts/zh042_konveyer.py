"""Конвейер Ж-042 по одной стратегии: этапы diag | tornado | dovodka | itog. Только train 2018–2019, при 0,04 %.

Целевая — min(Шарп 2018, Шарп 2019) после издержек (толкование Ж-042 п. 4). Только точки с боевым эталоном
(`ekviv_filtr.prichina_net`). Плато и активность — правила 19:35 (соседей ≥ 3, сделок ≥ 0,5 базы на tune 2018).
Выход: docs/tables/zh042/{sid}_{diagnostika,tornado_min,dovodka,itog}.csv. Состояние между этапами — только в этих файлах.
"""
from __future__ import annotations

import sys
from itertools import product

import numpy as np
import pandas as pd

sys.path[:0] = [".", "scripts"]
import zh042_lib as L                                   # noqa: E402
from fastbt.ekviv_filtr import prichina_net            # noqa: E402
from fastbt.sweep_pts import TOR_MIN, levels, make_fix  # noqa: E402
from fastbt.sweep_spec import pkey                      # noqa: E402

N_TOP, MIN_NB, MIN_ACT = 10, 3, 0.5


def diag(sid: str):
    sw, exp = L.open_sweep(sid)
    rows = L.main_rows(sid) + L.ref_rows(sid, sw.ev, dict(exp["base"]), "tsentr_setki_ekviv_s9")
    L.write_csv(pd.DataFrame(rows), f"{sid}_diagnostika.csv")
    print(pd.DataFrame(rows).T.to_string())


def tornado(sid: str):
    """Торнадо по min(Шарп 2018, Шарп 2019) из OAT 0,04 (`sweep_e1v2`); размах — по всем значениям и только по значениям с эталоном."""
    sw, exp = L.open_sweep(sid)
    t = pd.read_csv(L.E1 / f"{sid}_oat_0.04.csv", comment="#")
    t["obj_min"] = t[["tune_sharpe_2018", "ctrl_sharpe_2019"]].min(axis=1)
    t["net"] = [prichina_net(sid, L.point_from_row(exp, r)) or "" for _, r in t.iterrows()]
    out = []
    for prm, g in t[t.param != "base"].groupby("param", sort=False):
        b = t[(t.param == "base") & (t.group == g.group.iloc[0])].iloc[:1]
        a, e = pd.concat([g, b]), pd.concat([g, b])
        e = e[e.net == ""]
        ge = g[g.net == ""]
        out.append(dict(param=prm, n_znach=len(g), n_s_etalonom=len(ge), bez_etalona=int(len(ge) == 0),
                        razmakh_min_vse=a.obj_min.max() - a.obj_min.min(),
                        razmakh_min_etalon=(e.obj_min.max() - e.obj_min.min()) if len(e) else 0.0,
                        razmakh_staryy_tune2018=a.objective.max() - a.objective.min(),
                        luchshee_znachenie_etalon=ge.loc[ge.obj_min.idxmax(), "value"] if len(ge) else "",
                        luchshiy_min_etalon=ge.obj_min.max() if len(ge) else np.nan,
                        prichina_net="; ".join(sorted(set(g.net[g.net != ""])))[:160]))
    tor = pd.DataFrame(out).sort_values("razmakh_min_etalon", ascending=False)
    base = t[t.param == "base"].iloc[0]
    L.write_csv(tor, f"{sid}_tornado_min.csv", f"# центр сетки: min = {base.obj_min:.4f} (2018 {base.tune_sharpe_2018:.4f}, "
                                                f"2019 {base.ctrl_sharpe_2019:.4f}); 0,04 %; значения — потолок (подбор на train)\n")
    print(tor.drop(columns=["prichina_net"]).to_string(index=False))


def etalon_values(sid: str, exp: dict) -> dict:
    """{параметр: значения сетки, для которых OAT-точка имеет боевой эталон}; из OAT 0,04."""
    t = pd.read_csv(L.E1 / f"{sid}_oat_0.04.csv", comment="#")
    ok = {}
    for k, vals in exp["grid"].items():
        g = t[t.param == k]
        good = [L.coerce(exp, k, r["value"]) for _, r in g.iterrows() if not (prichina_net(sid, L.point_from_row(exp, r)))]
        ok[k] = [v for v in vals if v in good or v == exp["base"][k] and prichina_net(sid, exp["base"]) is None]
    return ok


def evalpt(sw, p: dict) -> dict:
    a, b = sw.evaluate(p, "setup", 0.04), sw.evaluate(p, "control", 0.04)
    a5, b5 = sw.evaluate(p, "setup", 0.035), sw.evaluate(p, "control", 0.035)
    s18, s19 = a["by_year"][2018], b["by_year"][2019]
    obj = min(s18, s19) if np.isfinite(s18) and np.isfinite(s19) else float("nan")
    return dict(s2018=s18, s2019=s19, obj=obj, s2018_0035=a5["by_year"][2018], s2019_0035=b5["by_year"][2019],
                storon_god_2018=a["sdelok_y"], storon_god_2019=b["sdelok_y"])


# параметры, не действующие в режиме (проверено 27.10: Шарп 2018 и 2019 совпадают при трёх значениях каждого)
INERT = {"S3": {("trail_type", "parabolic_step"): ("trail_atr_mult", "trail_n")}}


def fix_inert(sid: str, exp: dict):
    f0 = make_fix(exp)

    def fix(p: dict) -> dict:
        q = f0(p)
        for (ck, cv), ks in INERT.get(sid, {}).items():
            if q.get(ck) == cv:
                q.update({k: exp["base"][k] for k in ks})
        return q
    return fix


def _other(exp: dict, k: str, p: dict):
    """Значение сетки, отличное от текущего: проверка, что параметр k действует (fix не возвращает точку в p)."""
    return next(v for v in exp["grid"][k] if v != p[k])


def neighbors(sid, exp, p, keys, allowed, fix) -> list:
    out = {}
    for k in keys:
        vals = exp["grid"][k]
        i = vals.index(p[k])
        for j in (i - 1, i + 1):
            if 0 <= j < len(vals) and vals[j] in allowed[k]:
                q = fix({**p, k: vals[j]})
                if pkey(q) != pkey(p) and prichina_net(sid, q) is None:
                    out[pkey(q)] = q
    return list(out.values())


def dovodka(sid: str):
    sw, exp = L.open_sweep(sid)
    tor, fix, allowed = pd.read_csv(L.OUT / f"{sid}_tornado_min.csv", comment="#"), fix_inert(sid, exp), etalon_values(sid, exp)
    eff = tor[(tor.razmakh_min_etalon >= TOR_MIN) & (tor.n_s_etalonom >= 2)]
    keys = list(eff.param)
    mode = {}
    for k in keys[:3]:                                          # условный параметр из тройки задаёт режим (условие его блока)
        mode.update(exp["uslovie"].get(k, {}))
    base_m = fix({**exp["base"], **mode})
    pick = [k for k in keys if fix({**base_m, k: _other(exp, k, base_m)}) != base_m][:3]   # без параметров, инертных в режиме
    best = {k: L.coerce(exp, k, tor.set_index("param").loc[k, "luchshee_znachenie_etalon"]) for k in pick}
    lv = {k: levels(allowed[k], best[k]) for k in pick}       # центр уровней — лучшее значение OAT, не центр сетки
    base_sd = sw.evaluate(exp["base"], "setup", 0.04)["sdelok_y"]
    rows = []
    for vs in product(*[lv[k] for k in pick]):
        p = fix({**base_m, **dict(zip(pick, vs))})
        if prichina_net(sid, p) is None:
            rows.append(dict(rol="faktor", p=p, **evalpt(sw, p)))
    seen = {pkey(r["p"]) for r in rows}
    oat = pd.read_csv(L.E1 / f"{sid}_oat_0.04.csv", comment="#")      # в пул — и точки OAT с эталоном (иначе центр уровней их теряет)
    for _, o in oat.iterrows():
        q = fix(L.point_from_row(exp, o))
        if prichina_net(sid, q) is None and pkey(q) not in seen:
            seen.add(pkey(q))
            rows.append(dict(rol="oat", p=q, **evalpt(sw, q)))
    for r in rows:
        r["aktivnost_ok"] = int(r["storon_god_2018"] >= MIN_ACT * base_sd)
    pool = sorted([r for r in rows if r["aktivnost_ok"] and np.isfinite(r["obj"])], key=lambda r: -r["obj"])[:N_TOP]
    for r in pool:
        nb = [dict(rol="sosed", p=q, **evalpt(sw, q)) for q in neighbors(sid, exp, r["p"], keys, allowed, fix)]
        vals = [x["obj"] for x in [r] + nb if np.isfinite(x["obj"])]
        r.update(plato=float(np.mean(vals)), sosedey=len(vals) - 1, kray=int(len(vals) - 1 < MIN_NB), top10=1)
        rows += nb
    pr = sorted([r for r in pool], key=lambda r: (r["kray"], -r["plato"]))
    for i, r in enumerate(pr, 1):
        r["rang_plato"] = i
    flat = [{**{f"p.{k}": v for k, v in r["p"].items()}, **{k: v for k, v in r.items() if k != "p"}} for r in rows]
    df = pd.DataFrame(flat)
    L.write_csv(df, f"{sid}_dovodka.csv", f"# параметры доводки (3³): {pick}; направления плато: {keys}; база активности: "
                                          f"{base_sd:.3f} сторон/год на тикер; N проб (2018, 0,04) этого этапа: {len(sw.ledger)}\n")
    print(df[df.top10 == 1].sort_values("rang_plato")[["rang_plato", "obj", "s2018", "s2019", "plato", "sosedey", "kray"]].to_string(index=False))


if __name__ == "__main__":
    if sys.argv[2] == "itog":
        import zh042_itog
        zh042_itog.itog(sys.argv[1])
    else:
        {"diag": diag, "tornado": tornado, "dovodka": dovodka}[sys.argv[2]](sys.argv[1])
