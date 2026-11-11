"""S-Z-EVAL: разделы 1–3 отчёта `S_Z_REPORT.md` (первая таблица, вывод о механизме, нулевое распределение, решение Z-DEC) — строки, собранные из таблиц; чисел, вписанных вручную, нет."""
from __future__ import annotations

import pandas as pd

from models_v2.s18 import z_eval_cfg as C
from models_v2.s18 import z_eval_null as ZN
from models_v2.s18.z_doc_fmt import ci, fv, md, pick, pv, sgn

COLS = ("ряд", "QLIKE", "первая t⁺", "последняя t⁺", "n", "Δ к S-RCK, % [90 %]", "Δ к W5-G, % [90 %]", "Δ к X1, %", "Δ к K2, %", "Δ к K3, %")


def first_table(tb: pd.DataFrame, series=C.SHOWN) -> str:
    rows = []
    for s in series:
        r = pick(tb, series=s)
        d = {"ряд": s, "QLIKE": "%.6f" % r["qlike"], "первая t⁺": r["first_tplus"], "последняя t⁺": r["last_tplus"], "n": str(int(r["n"]))}
        for lab, b in zip(COLS[5:], C.BASES):
            k = "d_%s_pct" % b
            d[lab] = "—" if s == b else ci(r[k], r["lo90_" + b], r["hi90_" + b]) if b in ("S-RCK", "W5-G") else sgn(r[k])
        rows.append(d)
    return md(pd.DataFrame(rows), {c: "%s" for c in COLS})


def mv(me: pd.DataFrame, section: str, item: str, key=None) -> pd.Series:
    m = (me["section"] == section) & (me["item"] == item)
    if key is not None:
        m &= me["key"] == key
    return me[m].iloc[0]


def mech_lines(t: dict) -> list:
    st, me, dec = t["stats"], t["mech"], t["dec"].set_index("id")
    h1, h2 = pick(st, kind="main", hypothesis="S-RCKZ|S-RCK"), pick(st, kind="main", hypothesis="S-RCKZ|S-RCKZs")
    sh = mv(me, "hz2", "share_control")
    dz, dzs = mv(me, "hz2", "delta_Z_vs_RCK_pct")["value"], mv(me, "hz2", "delta_Zs_vs_RCK_pct")["value"]
    tot, fr, lay = (mv(me, "decomp", i)["value"] for i in ("total_vs_RCK", "frozen_pi", "layer_reaction"))
    hz1, hz2 = bool(dec.loc["H-Z1", "passed"]), bool(dec.loc["H-Z2", "passed"])
    return ["- **H-Z1** (S-RCKZ против S-RCK, все даты D_M): Δ = %s %%, 90 %% интервал [%s; %s], p_boot %s, DM NW-8 %s, p Холма (N = %d) %s, p SPA (а) %s — **%s**; условие H-Z1 %s." % (
                sgn(h1["delta_pct"]), sgn(h1["lo90"]), sgn(h1["hi90"]), pv(h1["p_boot"]), pv(h1["p_dm"]), int(h1["n_holm"]), pv(h1["p_holm_N"]), pv(h1["p_spa_c"]), h1["verdict"], "выполнено" if hz1 else "не выполнено"),
            "- **H-Z2** (S-RCKZ против S-RCKZs, механизм): Δ = %s %%, 90 %% интервал [%s; %s], p Холма (N) %s, p SPA %s — **%s**; условие H-Z2 %s." % (
                sgn(h2["delta_pct"]), sgn(h2["lo90"]), sgn(h2["hi90"]), pv(h2["p_holm_N"]), pv(h2["p_spa_c"]), h2["verdict"], "выполнено" if hz2 else "не выполнено"),
            "- **Доля выигрыша, которую даёт контроль:** Δ(S-RCKZs к S-RCK) = %s %%, Δ(S-RCKZ к S-RCK) = %s %%; доля = Δ(S-RCKZs, S-RCK)/Δ(S-RCKZ, S-RCK) = %s; %s%s." % (sgn(dzs), sgn(dz), sgn(sh["value"]), "контроль S-RCKZs хуже S-RCK (%s %%), доля отрицательна; " % sgn(dzs) if dzs > 0 and sh["value"] < 0 else "", sh["key"]),
            "- **Разложение Δ(S-RCKZ к S-RCK)** на «эффект при замороженном π» (F^fr) и «реакцию слоя»: всего %s %% = заморожено %s %% + реакция слоя %s %%." % (sgn(tot), sgn(fr), sgn(lay)), ""]


def null_section(t: dict) -> list:
    me, (s, m) = t["mech"], ZN.read()
    q = lambda i, st: float(m[(m["item"] == i) & (m["stat"] == st)]["value"].iloc[0])
    lo, hi = ZN.abs_bounds(s)
    rows = [{"величина": lab, "реальный Δ по модулю, %": "%.4f" % abs(mv(me, "noise_real", k)["value"]), "5 % модуля нуля": "%.4f" % lo, "95 % модуля нуля": "%.4f" % hi, "правило": ZN.rule_abs(mv(me, "noise_real", k)["value"], lo, hi)}
            for lab, k in (("Δ(S-RCKZ к S-RCK), D_M", "Z_vs_RCK|D_M"), ("Δ(S-RCKZ к S-RCK), активные даты", "Z_vs_RCK|active"), ("Δ(S-RCKZ к S-RCKZs), D_M", "Z_vs_Zs|D_M"), ("Δ(S-RCKZ к S-RCKZs), активные даты", "Z_vs_Zs|active"))]
    pm = pick(t["params"], variant="S-RCKZ", kind="months", item="share_theta_nonzero")
    return ["Объявленная диагностика (журнал 12.11 01:34, до QLIKE; не условие Z-DEC): синтетика `z_walk_syn` (цель ln RV = ln F⁰ + шум без связи с z), процедура S-RCKZ, реальные z̃, seeds шума %d–%d, n = %d шумов." % (ZN.SEEDS[0], ZN.SEEDS[-1], len(s)),
            "QLIKE(R) − QLIKE(θ = 0), %% QLIKE(θ = 0) на датах активных месяцев: среднее %s, СКО %s, 5 %% %s, 95 %% %s; доля месяцев θ ≠ 0 — среднее %.3f (реальный S-RCKZ: %.3f)." % (
                sgn(q("qlike_rel_pct", "mean")), sgn(q("qlike_rel_pct", "std")), sgn(q("qlike_rel_pct", "q05")), sgn(q("qlike_rel_pct", "q95")), q("share_theta_nonzero", "mean"), pm["value"]), "",
            md(pd.DataFrame(rows), {c: "%s" for c in rows[0]}), "",
            "Правило (журнал 01:34, буквальное чтение ↺): |Δ| внутри [5 %%; 95 %%] распределения |Δ_null| по %d шумам ⇒ «не отличим от шума процедуры»; Δ_null — QLIKE(R) − QLIKE(θ = 0) на активных датах." % len(s),
            "Оговорка о соразмерности: нуль — разность одной процедуры с θ = 0, а реальные Δ к S-RCK и S-RCKZs — разности двух обученных процедур, их шум примерно в √2 шире; при %d шумах квантили 5 и 95 %% — это фактически минимум и максимум |Δ_null|. Поэтому вывод — справочно, грубая граница шума процедуры, а не тест." % len(s),
            "", "Нулевое распределение по seeds шума:", md(s, {"seed": "%d", "n_active": "%d", "qlike_rel_pct": "%+.4f", "share_theta_nonzero": "%.3f"}), ""]


def dec_lines(t: dict) -> list:
    d = t["dec"].set_index("id")
    rows = t["dec"].iloc[:-1].copy()
    return [md(rows.assign(value=rows["value"].map(fv)), {"id": "%s", "condition": "%s", "value": "%s", "passed": "%s"}), "",
            "**Z-DEC (код %s):** %s" % (d.loc["Z-DEC", "value"], d.loc["Z-DEC", "text"]), "",
            "Замена кандидата в любом случае — решение владельца; «превзошёл» (код 1) только при подтверждении: p Холма (N = %d) < 0,05 и p SPA (а) < 0,05." % C.N_HOLM, ""]
