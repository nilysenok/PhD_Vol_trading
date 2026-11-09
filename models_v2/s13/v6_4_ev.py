"""V6-4: строки для EVIDENCE G36–G39 и первой строки отчёта (числа из пересчёта в памяти, `stage 4 --check`)."""
from __future__ import annotations

from models_v2.s13 import v6_0_q as Q, v6_1b_ev as E1B


def first_line(r: dict) -> str:
    i = r["info"]
    i = i[(i["window"] == "2017_2021") & (i["role"] == "правило п. 2.2")].set_index("pair")
    s = ["%s: Δ 2017–2021 %+.2f %% уровня K1 [%+.2f; %+.2f], DM p %.3f, лет с Δ<0 %d из 5, вывод: %s" % (p, x["delta_pct"], x["lo90"], x["hi90"], x["dm_p_one"], x["years_neg"], "несёт новую информацию" if x["new_information"] else "новой информации нет")
         for p, x in ((p, i.loc[p]) for p in i.index)]
    k = r["info"]
    k = k[(k["pair"] == "K9 vs K1") & (k["window"] == "2017_2021")].iloc[0]
    return " | ".join(s) + " | K9 против K1: Δ 2017–2021 %+.2f %% [%+.2f; %+.2f], DM p %.3f, лет с Δ<0 %d из 5" % (k["delta_pct"], k["lo90"], k["hi90"], k["dm_p_one"], k["years_neg"])


def qline_k9(r: dict) -> str:
    w, f = r["w"], r["f_k9"]
    q = {win: float(Q.ql(w["rv"].to_numpy()[Q.wmask(w["t_plus"].to_numpy(), win)], f[Q.wmask(w["t_plus"].to_numpy(), win)]).mean()) for win in ("all", "2017_2021", "2022_2025")}
    return "K9: QLIKE вся %.4f / 2017–2021 %.4f / 2022–2025 %.4f" % (q["all"], q["2017_2021"], q["2022_2025"])


def vs_lines(r: dict) -> list:
    v = r["info"]
    v = v[(v["role"] == "описательно") & v["window"].isin(["2017_2021", "all"]) & (v["pair"] != "K9 vs K1")]
    return ["%s, %s: Δ %+.2f %% уровня K1 [%+.2f; %+.2f], DM p %.3f" % (x.pair, x.window, x.delta_pct, x.lo90, x.hi90, x.dm_p_one) for x in v.itertuples() if x.x == "D2"]


def evidence(r: dict) -> list:
    cp = r["checkpoint"]
    cids = r["cids"]
    q = lambda c: E1B.qline({"w": r["w"], "fs": r["fs"], "cfg": r["cfg"], "r0": None}, c)
    out = [first_line(r), qline_k9(r)] + [q(c) for c in cids] + [E1B.go_line(r, c) for c in cids] + vs_lines(r)
    out.append("G36: пул дня d: %d … %d закрытых пар на %d датах; k_D1 %d, k_D2 %d, k_h %d, κ %d, Z_D %s" % (r["st"]["n_pool"].min(), r["st"]["n_pool"].max(), len(r["w"]), cp["params"]["k_d1"], cp["params"]["k_d2"], cp["params"]["k_h"], cp["params"]["kappa"], cp["params"]["z"]))
    out.append("G38: D_REGIME: %s; корреляция ln M_D2 и ln H_t: %s" % (cp["regime_verdict"], {k: round(v, 3) for k, v in cp["corr_lnM_lnH"].items()}))
    out.append("G39: конфигураций %d из %d; уровни: %s" % (cp["used"], cp["budget"], {c: "".join(str(int(x["passed"] == "да")) for x in g.to_dict("records")) for c, g in r["levels"].groupby("config")}))
    return out
