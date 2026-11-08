"""V5 лист FIN, G24: блоки итогового документа, часть 2 — совместные ошибки, значимость, MCS, стабильность, стоимость, абляция, охват, реестр, критерий выбора, основания закрытия направлений. Все числа — из таблиц на диске (см. `v5_fin_doc`)."""
from __future__ import annotations

import json

import pandas as pd

from models_v2.s12 import v5_fin_data as D, v5_fin_doc as W
from models_v2.s12.v5_fin_doc import f, hdr, nm, s

WINS = ("2017_2021", "2022_2025", "all")


def joint_pairs_block(t: dict) -> str:
    j, rows = t["joint"], []
    c, g = j[j["section"] == "corr"].set_index(["a", "b", "window"]), j[j["section"] == "same_sign"].set_index(["a", "b", "window"])
    for a, b in dict.fromkeys(zip(c.index.get_level_values("a"), c.index.get_level_values("b"))):
        rows.append("| %s – %s | %s | %s |" % (nm(a), nm(b), " / ".join(f(c.loc[(a, b, w), "value"], 3) for w in WINS), " / ".join(f(g.loc[(a, b, w), "value"], 3) for w in WINS)))
    return hdr(["пара", "корреляция ln(RV/F), 2017–2021 / 2022–2025 / вся", "доля дней с ошибкой одного знака, те же окна"]) + "\n".join(rows) + "\n"


def joint_all_block(t: dict) -> str:
    j = t["joint"]
    lab = {"all_same_sign": "все шесть ошибок одного знака", "rv_below_min": "RV ниже min прогнозов (все переоценивают)", "rv_above_max": "RV выше max прогнозов (все недооценивают)", "rv_outside_range": "RV вне [min, max] шести прогнозов"}
    v = lambda n, w: float(j[(j["section"] == n) & (j["window"] == w)]["value"].iloc[0])
    rows = ["| %s | %s |" % (txt, " / ".join(f(v(n, w), 3) for w in WINS)) for n, txt in lab.items()]
    cc = j[j["section"] == "cross_ticker_corr"].set_index(["a", "window"])["value"]
    rows += ["| средняя межбумажная корреляция ошибок, %s | %s |" % (nm(a), " / ".join(f(cc[(a, w)], 3) for w in WINS)) for a in dict.fromkeys(cc.index.get_level_values("a"))]
    return hdr(["показатель (M16)", "2017–2021 / 2022–2025 / вся"]) + "\n".join(rows) + "\n"


def tests_block(t: dict) -> str:
    d, cp = t["tests"], t["cp"]
    return hdr(["конфигурация", "база", "окно", "вид", "Δ QLIKE", "DM p", "бутстрап 95 % ДИ", "p_holm (N=%d)" % cp["N_declared_before_stats"], "p_holm_wide (N=%d)" % cp["N_wide"]]) + "".join(
        "| %s | %s | %s | %s | %s | %s | [%s; %s] | %s | %s |\n" % (r.config, nm(r.comparator), W.WN[r.window], "основное" if r.kind == "main" else "описательно", s(r.delta_mean, 5), f(r.dm_p, 3), s(r.boot_lo, 5), s(r.boot_hi, 5),
                                                                  "—" if pd.isna(r.p_holm) else f(r.p_holm, 3), "—" if pd.isna(r.p_holm_wide) else f(r.p_holm_wide, 3)) for r in d.itertuples())


def mcs_block(t: dict) -> str:
    mc, rows = t["mcs"], []
    for w in WINS:
        d = mc[(mc["window"] == w) & mc["in_mcs90"].astype(str).str.lower().eq("true")].sort_values("mean_qlike")
        rows.append("| %s | %d из %d | %s |" % (W.WN[w], len(d), (mc["window"] == w).sum(), ", ".join("%s (%s)" % (nm(r.model), f(r.mean_qlike, 4)) for r in d.itertuples())))
    return hdr(["окно", "в MCS 90 %", "ряды (средний панельный QLIKE)"]) + "\n".join(rows) + "\n"


def stress_block(t: dict) -> str:
    lab = {"full_crit_panel": "crit_panel", "no_2017_crit_panel": "без 2017", "no_1pct_extreme_crit_panel": "без 1 % крайних дат", "only_after_2021_mean_qlike_2022_2025": "только 2022–2025 (средний QLIKE)", "without_top_ticker_crit_panel": "без самой влиятельной бумаги"}
    rows = []
    for key, best in (("gs_stress", t["gs"]), ("sp_stress", t["sp"])):
        d = t[key].set_index(["stress", "series"])
        for st, name in lab.items():
            top = t[key].loc[(t[key]["stress"] == st) & t[key]["top_ticker"].notna(), "top_ticker"]
            rows.append("| %s | %s%s | %s | %s | %s |" % (best, name, " (%s)" % top.iloc[0] if len(top) else "", f(d.loc[(st, "K8"), "value"]), f(d.loc[(st, best), "value"]), s(d.loc[(st, best), "value"] - d.loc[(st, "K8"), "value"], 5)))
    return hdr(["лучшая", "проверка", "K8", "ряд", "Δ к K8 (минус = лучше)"]) + "\n".join(rows) + "\n"


def loto_summary(t: dict) -> pd.DataFrame:
    rows = []
    for key in ("gs_loto", "sp_loto"):
        d = t[key]
        for c, g in d.groupby("config", sort=False):
            rows.append(dict(config=c, loto=g["crit_wf_loto"].mean(), main=g["crit_wf_main"].mean(), k8=g["crit_wf_K8"].mean(), n_better=int((g["loto_minus_K8"] < 0).sum()), n=len(g)))
    return pd.DataFrame(rows)


def loto_block(t: dict) -> str:
    d = loto_summary(t)
    return hdr(["конфигурация", "crit_panel (LOTO)", "crit_panel (основной)", "K8", "LOTO − K8", "бумаг, где LOTO лучше K8"]) + "".join(
        "| %s | %s | %s | %s | %s | %d из %d |\n" % (r.config, f(r.loto), f(r.main), f(r.k8), s(r.loto - r.k8, 5), r.n_better, r.n) for r in d.itertuples())


def cost_block(t: dict) -> str:
    c = t["cost"]
    tm, pr = c[c["section"] == "time"], c[c["section"] == "params"]
    a = hdr(["стадия", "команда `--check`", "секунды", "rc"]) + "".join("| %s | `%s` | %s | %d |\n" % (r.stage, r.command, f(r.seconds, 1), r.rc) for r in tm.itertuples())
    b = hdr(["конфигурация", "оцениваемых параметров", "состав"]) + "".join("| %s | %d | %s |\n" % (r.stage, r.n_params, r.note) for r in pr.itertuples())
    return a + "\n" + b


def ablation_block(t: dict) -> str:
    a = t["abl"]
    pr, q = a[a["section"] == "pair"].set_index("chain"), a[a["section"] == "qlike"].set_index(["chain", "window"])
    rows = ["| %s | %s | %s | %s | %s | %s (p %s) | %s (p %s) |" % (ch, s(r.d_crit_panel, 5), s(r.d_crit_no2017, 5), s(r.d_crit_trim1, 5), s(r.d_EqY, 5), s(q.loc[(ch, "2022_2025"), "delta_mean"], 5), f(q.loc[(ch, "2022_2025"), "dm_p"], 3),
                                                                  s(q.loc[(ch, "all"), "delta_mean"], 5), f(q.loc[(ch, "all"), "dm_p"], 3)) for ch, r in pr.iterrows()]
    return hdr(["звено (a − b)", "Δ crit_panel", "Δ без 2017", "Δ без 1 %", "Δ EqY", "Δ QLIKE 2022–2025", "Δ QLIKE вся"]) + "\n".join(rows) + "\n"


def enc_block(t: dict) -> str:
    return hdr(["лучшая", "база", "окно", "вес w", "90 % интервал", "включает 0", "корреляция ln(RV/F) и ln(RV/M)", "ошибка одного знака с M"]) + "".join(
        "| %s | %s | %s | %s | [%s; %s] | %s | %s | %s |\n" % (r.best, r.base, W.WN[r.window], s(r.w_hat, 3), s(r.w_lo90, 3), s(r.w_hi90, 3), "да" if r.w_lo90 <= 0 <= r.w_hi90 else "нет", f(r.corr_ln_rv_over_F_ln_rv_over_M, 3), f(r.same_sign_share, 3))
        for r in t["enc"].itertuples())


def registry_block(t: dict) -> str:
    cp, fin = t["cp"], t["cp"]["FIN"]
    rows = "".join("| %s | %s | %s |\n" % (r.key, r.direction, json.loads(r.config)["variant"]) for r in t["trials"].itertuples())
    return (hdr(["ключ реестра", "направление", "вариант"]) + rows + "\nN_prior %d (объединение ключей реестров: 4 реестра до V4 и V4), + бюджет V5 %d = N_declared %d; N_wide %d (с %d пробами Optuna h = 1). Израсходовано %d из %d (GS %d, SP %d); FIN расходует %d. Состав N по реестрам — `SEARCH_HISTORY.csv`; закрытые направления и ключи кэшей прогнозов — `CHECKPOINT.json`, ключ `FIN`."
            % (cp["N_prior"], cp["budget"], cp["N_declared_before_stats"], cp["N_wide"], cp["N_wide_optuna_trials"], fin["used_total"], fin["budget"], fin["by_direction"]["GS"], fin["by_direction"]["SP"], fin["configs_spent_by_FIN"]))


def criterion_block(t: dict) -> str:
    e, au = t["eqy"].set_index("series"), t["audit"]
    wt = au[au["section"] == "weight"].set_index("year")["weight"]
    rho = e["rank_EqY"].corr(e["rank_crit"], method="spearman")
    rows = "".join("| %s | %s | %s | %d | %d |\n" % (nm(x), f(e.loc[x, "EqY"]), f(e.loc[x, "crit_panel"]), e.loc[x, "rank_EqY"], e.loc[x, "rank_crit"]) for x in ("k1", "k2", "k3", "d2", "M", "K8"))
    return ("Веса годов целей в `crit_panel` (следствие формулы `crit_wf`): %s. Год 2021 не входит ни в один штраф, 2017 входит во все. Критерий не менялся (определение, не ошибка реализации).\n\n" % ", ".join("%d — %s" % (y, f(v, 4)) for y, v in wt.items())
            + hdr(["ряд", "EqY", "crit_panel", "ранг EqY (из %d)" % len(e), "ранг crit_panel"]) + rows + "\nСпирмен между рангами по EqY и по crit_panel %s (%d рядов). EqY — только диагностика, выбор по ней не делался." % (f(rho, 4), len(e)))


def closure(t: dict) -> dict:
    """Числа оснований закрытия направлений 3 и 5 — из таблиц листа 1b."""
    mo, jm = t["mom"], t["jump"]
    w = mo[(mo["base"] == "K8") & (mo["window"] == "2017_2021") & (mo["item"] == "wald")].iloc[0]
    jr = jm[(jm["section"] == "conc") & (jm["base"] == "K8") & (jm["window"] == "2017_2021")].iloc[0]
    return dict(p_wald=float(w["p_wald"]), share_rows=float(jr["share_rows"]), share_qlike=float(jr["share_qlike"]))
