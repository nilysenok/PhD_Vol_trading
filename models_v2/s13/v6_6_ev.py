"""V6-6: строки для EVIDENCE G46–G49 и первой строки отчёта (числа из пересчёта в памяти, `stage 6 --check`)."""
from __future__ import annotations

from models_v2.s13 import v6_1b_ev as E1B, v6_6_stage as ST


def _fmt(p: str, x) -> str:
    return "%s: Δ 2017–2021 %+.2f %% уровня K1 [%+.2f; %+.2f], DM p %.3f, лет с Δ<0 %d из 5, вывод: %s" % (p, x["delta_pct"], x["lo90"], x["hi90"], x["dm_p_one"], x["years_neg"], "несёт новую информацию" if x["new_information"] else "новой информации нет")


def rows(r: dict, role: str) -> dict:
    i = r["info"]
    return i[(i["window"] == "2017_2021") & (i["role"] == role)].set_index("pair")


def first_line(r: dict) -> str:
    d = rows(r, ST.ROLE_DESC).loc["A0g vs A0"]
    return "; ".join([_fmt("A0g против A0", d)] + [_fmt(p, x) for p, x in rows(r, ST.ROLE_RULE).iterrows()])


def evidence(r: dict) -> list:
    cp, c = r["checkpoint"], ST.CID
    out = [first_line(r), E1B.qline(r, c), E1B.go_line(r, c)]
    out += ["G47 (%s): %s" % (role, _fmt(p, x)) for role in (ST.ROLE_RULE, ST.ROLE_OLD, ST.ROLE_DESC) for p, x in rows(r, role).iterrows()]
    out.append("G46: признаки %d (S), обучатель %s; 2271 общих дат; SHA прогноза %s" % (len(cp["features"]), cp["learner"], cp["f_sha"][c]))
    out.append("G48: конфигураций %d из %d; уровни: %s" % (cp["used"], cp["budget"], {k: "".join(str(int(x["passed"] == "да")) for x in g.to_dict("records")) for k, g in r["levels"].groupby("config")}))
    return out
