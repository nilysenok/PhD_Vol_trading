"""V6-2: строки для EVIDENCE G24–G29 и первой строки отчёта (числа из пересчёта в памяти, `stage 2 --check`)."""
from __future__ import annotations

from models_v2.s13 import v6_1b_ev as E1B


def first_line(r: dict) -> str:
    i = r["info"]
    i = i[(i["window"] == "2017_2021") & (i["y"] == "A0")].set_index("pair")
    s = ["%s: Δ 2017–2021 %+.2f %% уровня K1 [%+.2f; %+.2f], DM p %.3f, лет с Δ<0 %d из 5, вывод: %s" % (p, x["delta_pct"], x["lo90"], x["hi90"], x["dm_p_one"], x["years_neg"], "несёт новую информацию" if x["new_information"] else "новой информации нет")
         for p, x in ((p, i.loc[p]) for p in i.index if p in ("C1 vs A0", "C2 vs A0", "C3 vs A0"))]
    return " | ".join(s) + " | " + ("C4 посчитана по правилу 2.2" if r["dec"]["compute"] else "C4 не считалась (правило 2.2 не выполнено)")


def unstable_lines(r: dict) -> list:
    u = r["unstable"]
    u = u[(u["window"] == "2017_2021") & (u["base"] == "A0")]
    return ["G26: %s | %s: n = %d, Δ к A0 %+.2f %% уровня K1 группы [%+.2f; %+.2f]" % (x.config, x.group, x.n_days, x.delta_pct, x.lo90, x.hi90) for x in u.itertuples()]


def evidence(r: dict) -> list:
    cp = r["checkpoint"]
    qline = lambda c: E1B.qline({**r, "cfg": r["cfg"], "r0": r["r0"]}, c)
    out = [first_line(r)] + [qline(c) for c in r["cids"]] + [E1B.go_line(r, c) for c in r["cids"]] + unstable_lines(r)
    out.append("G24: Q %d × %d без NaN; групп u: %s" % (len(r["Qd"]), 8, cp["u_groups"]))
    out.append("G27: C4 -> %s" % cp["C4"])
    out.append("G28: конфигураций %d из %d; уровни: %s" % (cp["used"], cp["budget"], {c: "".join(str(int(x["passed"] == "да")) for x in g.to_dict("records")) for c, g in r["levels"].groupby("config")}))
    for c, st in cp["coef_stability"].items():
        out.append("%s: устойчивый вес (знак ≥ 90 %% переоценок и |среднее| > СКО): %s" % (c, [k for k, v2 in st.items() if v2["stable"]]))
    return out
