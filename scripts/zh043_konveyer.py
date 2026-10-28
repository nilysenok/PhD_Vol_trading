"""Конвейер Ж-043 по одной стратегии: diag | tornado | dovodka | itog | all. Только train 2018–2019, 0,04 %.

Целевая — min(mSR 2018, mSR 2019) (`metriki_e.msr`); mSR не определён -> точка недопустима. Только точки с боевым эталоном.
Пул плато — все посчитанные допустимые точки, включая соседей, с достройкой соседей новых лидеров (`zh043_pul`).
Выход: docs/tables/zh043/{sid}_{diagnostika,tornado,dovodka,itog}.csv. Пробы OAT берутся из `sweep_e1v2` (mSR в таблице есть).
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

sys.path[:0] = [".", "scripts"]
import zh042_konveyer as K                              # noqa: E402
import zh043_lib as L                                   # noqa: E402
import zh043_pul as P                                   # noqa: E402
from fastbt.ekviv_filtr import prichina_net            # noqa: E402
from fastbt.sweep_pts import TOR_MIN, levels            # noqa: E402
from fastbt.sweep_spec import pkey                      # noqa: E402

MIN_ACT = 0.5
OAT = lambda sid: pd.read_csv(L0_E1 / f"{sid}_oat_0.04.csv", comment="#")   # noqa: E731
L0_E1 = L.L0.E1


def diag(sid: str):
    sw, exp = L.open_sweep(sid)
    rows = L.main_rows(sid) + L.ref_rows(sid, sw.ev, dict(exp["base"]), "tsentr_setki_ekviv_s9", "zh043")
    c = L.evalpt(sw, dict(exp["base"]))
    for r in rows[2:]:
        g = r["god"]
        r.update({f"{k}_posle_004": c[f"{k}_{g}"] for k in ("msr", "sortino", "kalmar", "mdd")}, msr_ge1=int(L.fin(c[f"msr_{g}"]) and c[f"msr_{g}"] >= 1))
    df = pd.DataFrame(rows)
    L.write_csv(df, f"{sid}_diagnostika.csv", "# строка A_tolko_Sk_s9 — полный движок s9 как S9Baza (сверка с bazy_sverka); центр сетки — окно по end (ekviv_s9)\n")
    print(df.T.to_string())


def tornado(sid: str):
    """Торнадо по min(mSR 2018, mSR 2019) из OAT 0,04 (`sweep_e1v2`); nan -> точка недопустима и в размах не входит."""
    sw, exp = L.open_sweep(sid)
    t = OAT(sid)
    t["obj"] = [L.obj_msr(a, b) for a, b in zip(t.tune_msr, t.ctrl_msr)]
    t["net"] = [prichina_net(sid, L.point_from_row(exp, r)) or "" for _, r in t.iterrows()]
    out = []
    for prm, g in t[t.param != "base"].groupby("param", sort=False):
        b = t[(t.param == "base") & (t.group == g.group.iloc[0])].iloc[:1]
        a, e, ge = pd.concat([g, b]), pd.concat([g, b]), g[(g.net == "") & g.obj.notna()]
        e = e[(e.net == "") & e.obj.notna()]
        out.append(dict(param=prm, n_znach=len(g), n_s_etalonom=int((g.net == "").sum()), n_nan_msr=int(g.obj.isna().sum()),
                        razmakh_vse=np.ptp(a.obj.dropna()) if a.obj.notna().any() else 0.0,
                        razmakh_etalon=np.ptp(e.obj) if len(e) else 0.0,
                        luchshee_znachenie_etalon=ge.loc[ge.obj.idxmax(), "value"] if len(ge) else "",
                        luchshiy_etalon=ge.obj.max() if len(ge) else np.nan,
                        prichina_net="; ".join(sorted(set(g.net[g.net != ""])))[:160]))
    tor = pd.DataFrame(out).sort_values("razmakh_etalon", ascending=False)
    base = t[t.param == "base"].iloc[0]
    L.write_csv(tor, f"{sid}_tornado.csv", f"# центр сетки: min mSR = {base.obj:.4f} (2018 {base.tune_msr:.4f}, 2019 {base.ctrl_msr:.4f}); "
                                           "0,04 %; значения — потолок (подбор на train); nan -> недопустимо\n")
    print(tor.drop(columns=["prichina_net"]).to_string(index=False))


def dovodka(sid: str):
    sw, exp = L.open_sweep(sid)
    tor = pd.read_csv(L.OUT / f"{sid}_tornado.csv", comment="#")
    fix, allowed = K.fix_inert(sid, exp), K.etalon_values(sid, exp)
    eff = tor[(tor.razmakh_etalon >= TOR_MIN) & (tor.n_s_etalonom >= 2)]
    keys = list(eff.param)
    mode = {}
    for k in keys[:3]:
        mode.update(exp["uslovie"].get(k, {}))
    base_m = fix({**exp["base"], **mode})
    pick = [k for k in keys if fix({**base_m, k: K._other(exp, k, base_m)}) != base_m][:3]
    best = {k: L.coerce(exp, k, tor.set_index("param").loc[k, "luchshee_znachenie_etalon"]) for k in pick}
    lv = {k: levels(allowed[k], best[k]) for k in pick}
    base_sd = sw.evaluate(exp["base"], "setup", 0.04)["sdelok_y"]
    rows = {}

    def ev(p, rol):
        d = L.evalpt(sw, p)
        sig = tuple(round(d[k], 10) for k in ("s2018", "s2019", "storon_god_2018", "storon_god_2019"))
        return dict(p=p, rol=rol, aktivnost_ok=int(d["storon_god_2018"] >= MIN_ACT * base_sd), sig=sig, **d)

    def add(p, rol):
        if prichina_net(sid, p) is None and pkey(p) not in rows:
            rows[pkey(p)] = ev(p, rol)
    from itertools import product
    for vs in product(*[lv[k] for k in pick]):
        add(fix({**base_m, **dict(zip(pick, vs))}), "faktor")
    for _, o in OAT(sid).iterrows():
        add(fix(L.point_from_row(exp, o)), "oat")
    n0 = len(rows)
    top = P.build_pool(rows, ev, lambda p: K.neighbors(sid, exp, p, keys, allowed, fix), pkey)
    flat = [{**{f"p.{k}": v for k, v in r["p"].items()}, **{k: v for k, v in r.items() if k not in ("p", "nb", "sig")},
             "top10": int(any(r is t for t in top))} for r in rows.values()]
    df = pd.DataFrame(flat)
    L.write_csv(df, f"{sid}_dovodka.csv", f"# параметры доводки (3³): {pick}; направления плато: {keys}; база активности: {base_sd:.3f} сторон/год на тикер; "
                                          f"проб до достройки соседей {n0}, после {len(rows)}; топ-10 по цели из всего пула (факторные, OAT, соседи); "
                                          f"в топе без достроенных соседей (топ не стабилизировался за {P.MAX_REP} повтора): {sum(r['kray'] and 'nb' not in r for r in top)}\n")
    print(df[df.top10 == 1].sort_values("obj", ascending=False)[["rol", "obj", "msr_2018", "msr_2019", "s2018", "s2019", "plato", "sosedey", "sosedey_pos", "kray"]].to_string(index=False))


if __name__ == "__main__":
    import zh043_itog
    sid, st = sys.argv[1], sys.argv[2:] or ["all"]
    for s in (["diag", "tornado", "dovodka", "itog"] if st == ["all"] else st):
        print(f"--- {sid} {s}")
        if s == "ispravlenie":                          # однократно: строка-исправление S3 по Ж-042
            zh043_itog.ispravlenie_s3_zh042()
        else:
            zh043_itog.itog(sid) if s == "itog" else {"diag": diag, "tornado": tornado, "dovodka": dovodka}[s](sid)
