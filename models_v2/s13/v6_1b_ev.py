"""V6-1b: строки для EVIDENCE G17–G23 и первой строки отчёта (числа из пересчёта в памяти, `stage 1b --check`)."""
from __future__ import annotations

from models_v2.s13 import v6_0_q as Q


def first_line(r: dict) -> str:
    i = r["info"]
    i = i[i["window"] == "2017_2021"].set_index("pair")
    s = []
    for p in i.index:
        x = i.loc[p]
        s.append("%s: Δ 2017–2021 %+.2f %% уровня K1 [%+.2f; %+.2f], DM p %.3f, лет с Δ<0 %d из 5, вывод: %s" % (p, x["delta_pct"], x["lo90"], x["hi90"], x["dm_p_one"], x["years_neg"], "несёт новую информацию" if x["new_information"] else "новой информации нет"))
    a3 = "A3 посчитана по правилу 2.2" if r["dec"]["compute"] else "A3 не считалась (правило 2.2 не выполнено)"
    return " | ".join(s) + " | " + a3


def qline(r: dict, c: str) -> str:
    w, f = r["w"], r["fs"][c]
    q = {win: float(Q.ql(w["rv"].to_numpy()[Q.wmask(w["t_plus"].to_numpy(), win)], f[Q.wmask(w["t_plus"].to_numpy(), win)]).mean()) for win in ("all", "2017_2021", "2022_2025")}
    t = r["cfg"][c]["t"]["tests"] if c in r["cfg"] else r["r0"]["tests"]
    d = t[t["section"] == "delta_dm"]
    iv = lambda win, b: "%+.2f%% [%+.2f; %+.2f]" % tuple(d[(d["window"] == win) & (d["base"] == b)][["delta_pct", "lo90", "hi90"]].iloc[0])
    return "%s: QLIKE вся %.4f / 2017–2021 %.4f / 2022–2025 %.4f; Δ вся|2017–2021 к K1 %s | %s, K3 %s | %s, K8 %s | %s, V4:NTP2 %s | %s" % (
        c, q["all"], q["2017_2021"], q["2022_2025"], iv("all", "K1"), iv("2017_2021", "K1"), iv("all", "K3"), iv("2017_2021", "K3"), iv("all", "K8"), iv("2017_2021", "K8"), iv("all", "V4:NTP2"), iv("2017_2021", "V4:NTP2"))


def go_line(r: dict, c: str) -> str:
    return "%s go 1–5: " % c + "; ".join("%d %s -> %s" % (x["cond"], "%.5f vs %.5f" % (x["value"], x["threshold"]), "да" if x["passed"] else "нет") for x in r["cfg"][c]["t"]["go"].to_dict("records"))


def dx_line(dx) -> list:
    w = dx[dx["test"].str.startswith("Wald:")]
    return ["DX-A: %s | тест %s: df %d, W %.2f, p %.4f (n = %d)" % (x.model, x.test, x.df, x.stat, x.p, x.n) for x in w.itertuples()]


def evidence(r: dict, dx) -> list:
    cp = r["checkpoint"]
    out = [first_line(r)] + [qline(r, c) for c in r["ids"]] + [go_line(r, c) for c in r["cfg"]] + dx_line(dx)
    v = r["vs"]
    out.append("G20: A_VS_SP строк %d (Δ к SP1–SP4, DM, охват); флаги дней: %s" % (len(v), cp["flags_days"]))
    out.append("G21: A3 -> %s" % cp["A3"])
    out.append("G22: конфигураций %d из %d; уровни: %s" % (cp["used"], cp["budget"], {c: "".join(str(int(x["passed"] == "да")) for x in g.to_dict("records")) for c, g in r["levels"].groupby("config")}))
    for c, st in cp["coef_stability"].items():
        out.append("%s: устойчивый вес (знак ≥ 90 %% переоценок и |среднее| > СКО): %s" % (c, [k for k, v2 in st.items() if v2["stable"]]))
    return out
