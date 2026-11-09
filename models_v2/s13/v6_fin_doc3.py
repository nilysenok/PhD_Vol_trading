"""V6 лист FIN, G52: блоки итогового документа 10–12 (измеренные причины, реестр и CHECKPOINT, перспективы) и условия `claims_fail`, на которых стоят словесные выводы. Только измеренные свойства, без догадок и обещаний."""
from __future__ import annotations

from models_v2.s13 import v6_fin_abl as AB, v6_fin_data as D
from models_v2.s13.v6_fin_doc import best, clear_vs_boosts, f, hdr, iv, level5, m, s, tt

GBM_X = AB.GBM_X
CK = ["0", "1a", "1b", "2", "3", "4", "5", "6", "FIN"]


def pair(t: dict, p: str, win: str = "2017_2021"):
    d = t["info"]
    return d[(d["pair"] == p) & (d["window"] == win)].iloc[0]


def dec(t: dict, part: str, x: str, y: str, win: str = "2017_2021"):
    d = t["abl"]
    return d[(d["part"] == part) & (d["x"] == x) & (d["y"] == y) & (d["window"] == win)].iloc[0]


def lin_rows(t: dict) -> list:
    return [dec(t, "features_lin", x, "A0") for x in AB.LIN_X]


def year_counts(t: dict, c: str) -> dict:
    y = t["years"].set_index("year")
    ys = [i for i in y.index if i <= 2025]
    return {b: int(sum(y.loc[i, c] < y.loc[i, b] for i in ys)) for b in ("K2", "K3", "K8", "V4:NTP2")} | {"n": len(ys)}


def claims_fail(t: dict) -> list:
    """Условия, на которых стоят словесные выводы документа (не выполнены — сборка отказывает)."""
    b17, ba = best(t, "2017_2021"), best(t, "all")
    lr, a2, g = pair(t, "A0g vs A0"), pair(t, "A2 vs A0g"), t["go"]
    npass = g.assign(ok=g["passed"].astype(str).str.lower().eq("true")).groupby("config")["ok"].sum()
    ok = {"обучатель: A0g лучше A0 5 из 5, интервал ниже 0": lr.hi90 < 0 and lr.years_neg == 5 and lr.dm_p_one < 0.05,
          "A2 против A0g: правило выполнено, интервал задевает 0, 3 из 5, Холм не значим": bool(a2.new_information) and a2.hi90 > 0 and a2.years_neg == 3 and a2.p_holm_N > 0.05,
          "C2, C4, E2 против A0g: новой информации нет": all(not bool(pair(t, "%s vs A0g" % x).new_information) for x in ("C2", "C4", "E2")),
          "LIN с новыми признаками: ни один интервал не ниже 0": all(r.hi90 >= 0 for r in lin_rows(t)),
          "ни одна конфигурация не выполнила все пять условий «go»": int(npass.max()) < 5,
          "ни одна проверка не значима после Холма (N = 722)": float(t["tests"].query("section == 'delta_dm'")["p_holm_N"].min()) >= 0.05,
          "D против K9 и F против CB1: новой информации нет": all(not bool(pair(t, p).new_information) for p in ("D1 vs K9", "D2 vs K9", "D3 vs K9", "F1 vs CB1", "F2 vs CB1")),
          "A0g: интервалы к K2, K3, K8, NTP2 в 2017–2021 включают 0": all(tt(t, "A0g", b, "2017_2021").lo90 <= 0 <= tt(t, "A0g", b, "2017_2021").hi90 for b in ("K2", "K3", "K8", "V4:NTP2")),
          "уровень 5 «нет» у всех, слова блока у лучших двух": (t["levels"].query("level == 5")["passed"] == "нет").all() and all(level5(t, c).endswith("статистически неподтверждённое историческое улучшение") for c in (b17, ba)),
          "лучший прежний V4:NTP2 лучше всех конфигураций на всей истории": all(m(t).loc[("V4:NTP2", "all"), "qlike"] < m(t).loc[(c, "all"), "qlike"] for c in D.CFG),
          "лучшая 2017–2021 хуже K2 и K3 в 2022–2025 (точечно)": all(tt(t, b17, b, "2022_2025").delta_pct > 0 for b in ("K2", "K3")),
          "лучшая 2017–2021 лучше NTP2 по QLIKE в 2017–2021": m(t).loc[(b17, "2017_2021"), "qlike"] < m(t).loc[("V4:NTP2", "2017_2021"), "qlike"],
          "лучшая вся история (%s) не имеет интервала ниже 0 против K2, K3" % ba: not clear_vs_boosts(t, ba, "all")}
    return [k for k, v in ok.items() if not v]


def causes(t: dict) -> str:
    b17, ba = best(t, "2017_2021"), best(t, "all")
    lr, a2 = pair(t, "A0g vs A0"), pair(t, "A2 vs A0g")
    tot, fe = dec(t, "total_gbm", "A2", "A0"), dec(t, "features_gbm", "A2", "A0g")
    ln = lin_rows(t)
    yc, e = year_counts(t, b17), t["enc"].set_index(["config", "window", "base"])
    g = t["go"].assign(ok=lambda x: x["passed"].astype(str).str.lower().eq("true")).groupby("config")["ok"].sum()
    st = t["stress"]
    st = st[(st["kind"] == "stress") & (st["config"] == b17)]
    mc = t["mcs"]
    n_in = lambda w: int(mc[(mc["window"] == w) & mc["in_mcs90"].astype(str).str.lower().eq("true")].shape[0])
    v6_in = int(mc[(mc["window"] == "2017_2021") & mc["in_mcs90"].astype(str).str.lower().eq("true") & mc["model"].isin(D.CFG)].shape[0])
    r = lambda c, w, b: e.loc[(c, w, b)]
    return "\n".join([
        "- **Обучатель.** Тот же набор S, GBM вместо LIN (A0g против A0): Δ %s, DM p %s, лет с Δ < 0 — %d из 5." % (iv(lr), f(lr.dm_p_one, 3), lr.years_neg),
        "- **Признаки при GBM.** A2 против A0: суммарно %s; из них на обучатель приходится %s, на признаки Σ — %s (A2 против A0g: DM p %s, %d из 5 лет, p Холма N = 722: %s). C2, C4, E2 против A0g: Δ %s, %s, %s — новой информации нет."
        % (iv(tot), iv(dec(t, "learner", "A0g", "A0")), iv(fe), f(a2.dm_p_one, 3), a2.years_neg, f(a2.p_holm_N, 3), *(iv(pair(t, "%s vs A0g" % x)) for x in ("C2", "C4", "E2"))),
        "- **Признаки при LIN.** Новые признаки (A1, A3, C1, C3, E1, E3, CB1, F1, F2) против A0: Δ 2017–2021 от %s до %s; интервал ниже 0 — у %d из %d, целиком выше 0 — у %d." % (s(min(x.delta_pct for x in ln)), s(max(x.delta_pct for x in ln)), sum(x.hi90 < 0 for x in ln), len(ln), sum(x.lo90 > 0 for x in ln)),
        "- **Годы.** %s (лучшая 2017–2021) ниже по QLIKE, чем K2 — в %d, K3 — в %d, K8 — в %d, V4:NTP2 — в %d из %d лет 2017–2025 (FIN_YEARS)." % (b17, yc["K2"], yc["K3"], yc["K8"], yc["V4:NTP2"], yc["n"]),
        "- **Окна.** У %s Δ к K8: 2017–2021 %s, 2022–2025 %s, вся история %s; к K3: %s, %s, %s." % (b17, *(iv(tt(t, b17, b, w)) for b in ("K8", "K3") for w in ("2017_2021", "2022_2025", "all"))),
        "- **Охват.** Вес %s в K8^(1−w)·F^w: 2017–2021 %s [%s; %s], вся история %s [%s; %s]; в K3^(1−w)·F^w: 2017–2021 %s [%s; %s], вся история %s [%s; %s]." % (
            b17, *(f(x, 2) for x in (r(b17, "2017_2021", "K8").w, r(b17, "2017_2021", "K8").lo90, r(b17, "2017_2021", "K8").hi90, r(b17, "all", "K8").w, r(b17, "all", "K8").lo90, r(b17, "all", "K8").hi90,
                                      r(b17, "2017_2021", "K3").w, r(b17, "2017_2021", "K3").lo90, r(b17, "2017_2021", "K3").hi90, r(b17, "all", "K3").w, r(b17, "all", "K3").lo90, r(b17, "all", "K3").hi90))),
        "- **«Go».** Наибольшее число выполненных условий из пяти — %d (конфигурации: %s); все пять — ни у одной." % (int(g.max()), ", ".join(c for c in D.CFG if g[c] == g.max())),
        "- **Стресс.** У %s Δ к K8 ниже 0 в %d из %d стресс-окон (таблица раздела 9)." % (b17, int((st["d_pct_K8"] < 0).sum()), len(st)),
        "- **MCS 90 %%.** В множество входят %d (2017–2021), %d (2022–2025), %d (вся история) из %d рядов; конфигураций V6 в MCS 2017–2021 — %d из %d." % (n_in("2017_2021"), n_in("2022_2025"), n_in("all"), len(D.SER), v6_in, len(D.CFG)),
        "- **Лучшая вся история.** %s: QLIKE %s против V4:NTP2 %s." % (best(t, "all"), f(m(t).loc[(ba, "all"), "qlike"]), f(m(t).loc[("V4:NTP2", "all"), "qlike"])),
    ]) + "\n"


def registry_block(t: dict) -> str:
    tr, cp = t["trials"], t["cp"]
    rows = "\n".join("| %s | %s | %s |" % (r.key, r.direction, r.source) for r in tr.itertuples())
    cks = []
    for k in CK:
        if k not in cp:
            continue
        e = cp[k]
        cks.append("| %s | %s | %s | %s |" % (k, e.get("used", "—"), ", ".join("%s %s" % (a.replace("code_hash", "hash"), b) for a, b in e.items() if a.startswith("code_hash")) or "—", e.get("inputs_hash", "—")))
    return hdr(["ключ реестра", "направление", "источник (объявлено)"]) + rows + "\n\nCHECKPOINT.json:\n\n" + hdr(["ключ", "израсходовано", "хэши кода", "хэш входов"]) + "\n".join(cks) + "\n"


def perspectives(t: dict) -> str:
    a2, lr = pair(t, "A2 vs A0g"), pair(t, "A0g vs A0")
    dx = t["abl"][(t["abl"]["part"] == "DX-A") & t["abl"]["test"].astype(str).str.startswith("Wald:Σ")]
    wald = "; ".join("%s: p %s" % (r.model, f(r.p, 4)) for r in dx.itertuples())
    return ("- **Измерено (без поправки на множественность).** Нелинейный обучатель GBM на сводке S точечно и по 90 %% интервалу лучше линейного на 2017–2021 (A0g против A0, %s, %d из 5 лет); после поправки Холма (N = 722) не значимо: p Холма %s. Это свойство обучателя на тех же признаках; против K2, K3, K8 и V4:NTP2 интервалы Δ A0g в 2017–2021 включают 0.\n"
            "- **Не подтверждено.** Признаки Q (качество измерения) и R (шок и восстановление) при GBM против A0g новой информации не несут; при LIN ни один интервал Δ новых признаков против A0 не лежит ниже 0. Аналоги D и робастные цели F правило п. 2.2 не выполнили. Ни одна конфигурация не имеет значимости после поправки Холма.\n"
            "- **Единственный источник сверх обучателя, не отброшенный правилом против A0g.** Площади Леви Σ при GBM (A2 против A0g: Δ %s, DM p %s, %d из 5 лет, интервал задевает 0, p Холма N = 722 — %s). Диагностика DX-A задним числом (не выбор), тест Вальда для Σ при наличии спектра: %s.\n"
            "- **Какая проверка отделит его от шума.** Одна предрегистрированная проверка A2 против A0g без подбора параметров: либо честная половина (параметры и порог фиксируются на 2017–2021, оценка на непросмотренных датах после 2026-02-26 по мере накопления), либо новые данные (другая бумага или горизонт при тех же формулах). Порог и число сравнений объявляются до просмотра.\n"
            "- **Чего не утверждается.** Результат не обещается; окна 2022–2025 и 2026 просмотрены в прошлых программах и не являются независимым OOS." % (iv(lr), lr.years_neg, f(lr.p_holm_N, 3), iv(a2), f(a2.dm_p_one, 3), a2.years_neg, f(a2.p_holm_N, 3), wald))
