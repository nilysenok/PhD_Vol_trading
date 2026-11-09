"""V6-5: строки для EVIDENCE G41–G45 и первой строки отчёта (числа из пересчёта в памяти, `stage 5 --check`)."""
from __future__ import annotations

from models_v2.s13 import v6_1b_ev as E1B


def first_line(r: dict) -> str:
    i = r["info"]
    i = i[(i["window"] == "2017_2021") & (i["role"] == "правило п. 2.2")].set_index("pair")
    s = ["%s: Δ 2017–2021 %+.2f %% уровня K1 [%+.2f; %+.2f], DM p %.3f, лет с Δ<0 %d из 5, вывод: %s" % (p, x["delta_pct"], x["lo90"], x["hi90"], x["dm_p_one"], x["years_neg"], "несёт новую информацию" if x["new_information"] else "новой информации нет")
         for p, x in i.iterrows()]
    return " | ".join(s)


def years_lines(r: dict) -> list:
    y = r["years"]
    s = y[y["section"] == "итог"]
    out = ["G43: %s, %s: СКО годовых Δ %.4f QLIKE (%.2f %% уровня K1), средняя Δ %+.2f %% уровня K1, лет с Δ<0 %d из %d" % (x.pair, x.window, x.sd_delta_qlike, x.sd_delta_pct_K1, x.mean_delta_pct_K1, x.years_neg, x.years_n) for x in s.itertuples()]
    return out + ["G43: %s, %s: Δ QLIKE %+.4f (%+.2f %% K1)" % (x.pair, x.window, x.delta_qlike, x.delta_pct_K1) for x in y[y["section"] == "год"].itertuples()]


def cover_lines(r: dict) -> list:
    out = []
    for c in r["cids"]:
        t = r["cfg"][c]["t"]["tests"]
        t = t[(t["section"] == "coverage") & (t["window"] == "2017_2021")]
        out += ["G44: охват %s к %s, 2017–2021: %s" % (c, x["base"], {"w": round(x["w"], 4), "lo90": round(x["lo90"], 4), "hi90": round(x["hi90"], 4)}) for x in t.to_dict("records")]
    return out


def evidence(r: dict) -> list:
    cp, cids = r["checkpoint"], r["cids"]
    out = [first_line(r)] + [E1B.qline(r, c) for c in cids] + [E1B.go_line(r, c) for c in cids] + years_lines(r) + cover_lines(r)
    out.append("G42: сходимость L-BFGS-B: %s" % cp["convergence"])
    out.append("G44: конфигураций %d из %d; уровни: %s" % (cp["used"], cp["budget"], {c: "".join(str(int(x["passed"] == "да")) for x in g.to_dict("records")) for c, g in r["levels"].groupby("config")}))
    for c, st in cp["coef_stability"].items():
        out.append("%s: устойчивый вес (знак ≥ 90 %% переоценок и |среднее| > СКО): %s" % (c, [k for k, v2 in st.items() if v2["stable"]]))
    return out
