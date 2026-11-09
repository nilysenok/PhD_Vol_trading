"""V6-3: строки для EVIDENCE G30–G35 и первой строки отчёта (числа из пересчёта в памяти, `stage 3 --check`)."""
from __future__ import annotations

from models_v2.s13 import v6_1b_ev as E1B


def first_line(r: dict) -> str:
    i = r["info"]
    i = i[(i["window"] == "2017_2021") & (i["y"] == "A0")].set_index("pair")
    s = ["%s: Δ 2017–2021 %+.2f %% уровня K1 [%+.2f; %+.2f], DM p %.3f, лет с Δ<0 %d из 5, вывод: %s" % (p, x["delta_pct"], x["lo90"], x["hi90"], x["dm_p_one"], x["years_neg"], "несёт новую информацию" if x["new_information"] else "новой информации нет")
         for p, x in ((p, i.loc[p]) for p in ("E1 vs A0", "E2 vs A0"))]
    cp = r["checkpoint"]
    return " | ".join(s) + " | " + ("E3 посчитана по правилу" if r["dec"]["compute"] else "E3 не считалась (правило не выполнено)") + " | слиты с Σ (|корр.| > 0,8): %s; оставлены в R: %s" % (cp["r_merged_with_sigma"] or "нет", cp["r_kept"])


def vs_lines(r: dict) -> list:
    v = r["vs_ac"]
    v = v[v["window"].isin(["2017_2021", "all"])]
    return ["%s, %s: Δ %+.2f %% уровня K1 [%+.2f; %+.2f], DM p %.3f" % (x.pair, x.window, x.delta_pct, x.lo90, x.hi90, x.dm_p_one) for x in v.itertuples() if x.y in ("A2", "C2") and x.x == "E2"]


def evidence(r: dict) -> list:
    cp = r["checkpoint"]
    cids = r["cids"]
    out = [first_line(r)] + [E1B.qline(r, c) for c in cids] + [E1B.go_line(r, c) for c in cids] + vs_lines(r)
    out.append("G30: R %d дней без NaN; крайние случаи (дни): %s; макс. |корр.| ρ с Σ: %s" % (cp["days"], cp["r_edge_days"], {k: round(v, 3) for k, v in cp["r_max_abs_corr_sigma"].items()}))
    out.append("G31: обращение (число признаков по выводам): %s" % cp["tr_summary"])
    out.append("G33: E3 -> %s" % cp["E3"])
    out.append("G34: конфигураций %d из %d; уровни: %s" % (cp["used"], cp["budget"], {c: "".join(str(int(x["passed"] == "да")) for x in g.to_dict("records")) for c, g in r["levels"].groupby("config")}))
    for c, st in cp["coef_stability"].items():
        out.append("%s: устойчивый вес (знак ≥ 90 %% переоценок и |среднее| > СКО): %s" % (c, [k for k, v2 in st.items() if v2["stable"]]))
    return out
