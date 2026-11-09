"""V6 лист Z1 (Ж-230, G57–G59): статус A0g в итоговом документе — подпись «диагностический контроль (Ж-230)» во всех таблицах, подраздел «A0g: статус, причина, моменты»,
учёт A0g в реестре и в N. Времена берутся из журнала `docs/resheniya/resheniya.md` и `state/harj_hybrid_v6.md` (в тексте генератора времён нет), числа — из таблиц на диске.
Ничего не обучает и не пересчитывает; пороги и правила не меняются (сверка с планом — тест `tests/test_v2_v6_z1_n.py`)."""
from __future__ import annotations

import inspect
import re

import numpy as np

from models_v2.s3 import dm as dmod
from models_v2.s13 import v6_0_crit as CR, v6_0_io as IO, v6_0_q as Q, v6_1_eval as EV, v6_fin_data as D, v6_fin_doc as W, v6_fin_doc2 as W2, v6_fin_doc3 as W3

JOURNAL = IO.ROOT / "docs/resheniya/resheniya.md"
STATE = IO.ROOT / "state/harj_hybrid_v6.md"
A0G = re.compile(r"\bA0g\b")
HM = r"(\d\d\.\d\d)(?:\.\d{4})?,? \(?(\d\d:\d\d)"
RX = {"review_1b": (r"^≡ \[V6\] Лист 1b \(перепроверка дирижёра (\d\d\.\d\d) (\d\d:\d\d)–(\d\d:\d\d)\)", "j"),
      "question": (r"откат: вопрос владельцу 1 ниже \((\d\d\.\d\d) (\d\d:\d\d)\)", "j"),
      "deadline": (r"Умолчание, если нет ответа до (\d\d:\d\d) (\d\d\.\d\d)", "j"),
      "default": (r"^≡ \[V6\] (\d\d\.\d\d) (\d\d:\d\d) \(по `date`\): ответа владельца на вопрос 1 к сроку", "j"),
      "leaf6": (r"^- Лист 6 \(абляционная база A0g.*?выполнен (\d\d\.\d\d)\.\d{4} \(до (\d\d:\d\d) MSK", "s"),
      "owner": (r"^### Ж-230 — (\d\d\.\d\d)\.\d{4}, (\d\d:\d\d) MSK", "j"),
      "quote": (r"Ъ «(Утверждаю сохранение A0g в итоговых материалах V6)»", "j")}


def minutes(hhmm: str) -> int:
    h, m = hhmm.split(":")
    return 60 * int(h) + int(m)


def moments(journal: str, state: str) -> dict:
    """Моменты решения по A0g из журнала и state. Нет записи или нарушен порядок времён — ValueError."""
    got = {}
    for k, (rx, src) in RX.items():
        m = re.search(rx, journal if src == "j" else state, re.M | re.S if src == "s" else re.M)
        if not m:
            raise ValueError("Z1: в %s нет записи «%s»" % ("журнале" if src == "j" else "state", k))
        got[k] = m.groups()
    t = {"review_end": got["review_1b"][2], "question": got["question"][1], "default": got["default"][1], "leaf6": got["leaf6"][1], "owner": got["owner"][1]}
    seq = [minutes(t[k]) for k in ("review_end", "question", "default", "leaf6", "owner")]
    if seq != sorted(seq) or {got[k][0] for k in ("question", "default", "leaf6", "owner")} != {"09.11"}:
        raise ValueError("Z1: времена журнала не по порядку или не 09.11: %s" % t)
    return {**t, "review_start": got["review_1b"][1], "deadline": got["deadline"][0], "quote": got["quote"][0]}


def selection_fail(t: dict) -> list:
    return ["лучшая конструкция (%s) выбирается без A0g: с A0g выбор %s ≠ %s" % (w, W.best_with_diag(t, w), W.best(t, w)) for w in ("2017_2021", "all") if W.best_with_diag(t, w) != W.best(t, w)]


def registry_fail(t: dict) -> list:
    """A0g в ALL_TRIALS.csv и в бюджете, вошедшем в N и N_wide: N − N_V5 = N_wide − N_wide_V5 = строк реестра = использовано."""
    tr, c0, c6, bad = t["trials"], t["cp"]["0"], t["cp"]["6"], []
    if "Ж-227:A0g" not in set(tr["key"]):
        bad.append("нет строки Ж-227:A0g в ALL_TRIALS.csv")
    n = len(tr)
    if not (n == c6["used"] == c0["budget_V6"] == c0["N"] - c0["N_V5_declared"] == c0["N_wide"] - c0["N_wide_V5"]):
        bad.append("испытания реестра (%d) не равны бюджету, использованному и приросту N (%d, %d)" % (n, c0["N"] - c0["N_V5_declared"], c0["N_wide"] - c0["N_wide_V5"]))
    if "A0g" not in c6["configs"] or (c0["N"], c0["N_wide"]) != (722, 3122):
        bad.append("A0g не в ключе 6 или N, N_wide ≠ 722, 3122")
    return bad


def holm_fail(t: dict) -> list:
    """p Холма в G_INFO (пары с A0g) = Холм с N, N_wide ключа 0 по парам окна (пересчёт p Холма из сырых p; ничего не обучается)."""
    g, bad = t["info"], []
    g = g[g["src"] == "G_INFO.csv"]
    for role in dict.fromkeys(g["role"]):
        for win in dict.fromkeys(g["window"]):
            s = g[(g["role"] == role) & (g["window"] == win) & g["dm_p_one"].notna()]
            a, b = EV.holm_both(s["dm_p_one"].to_numpy(float), t["cp"]["0"])
            if not (np.allclose(a, s["p_holm_N"], atol=1e-9) and np.allclose(b, s["p_holm_Nwide"], atol=1e-9)):
                bad.append("p Холма G_INFO (%s, %s) не равны Холму с N = %s, N_wide = %s" % (role, win, t["cp"]["0"]["N"], t["cp"]["0"]["N_wide"]))
    return bad


PLAN = IO.ROOT / "docs/otchety/HARJ_HYBRID_SBER_V6_PLAN.md"
PLAN_RX = {"p < 0,05": r"p < 0,05 без поправки", "3 года из 5": r"не менее чем в 3 годах из 5", "лаг 5": r"НУ, лаг 5", "N, N_wide": r"N = 722, N_wide = 3122", "бутстрап": r"блок 20, 2000, seed 0"}


def thresholds_fail(plan: str | None = None, src: str | None = None) -> list:
    """Пороги в коде листов = объявленным в плане (коммит 99bd6681) и в GATES_V6/Ж-229 (δ): δ 0,0005, p 0,05, 3 из 5, пять условий «go», Холм N, бутстрап 20 · 2000 · seed 0, лаг DM 5."""
    plan = PLAN.read_text(encoding="utf-8") if plan is None else plan
    src = inspect.getsource(EV.new_information) if src is None else src
    bad = ["в плане нет «%s»" % k for k, rx in PLAN_RX.items() if not re.search(rx, plan)]
    code = {"δ (v6_0_q.MARGIN) = 0,0005": Q.MARGIN == CR.DELTA == 0.0005, "бутстрап EV = 2000 · 20 · 0": (EV.B, EV.BLOCK, EV.SEED) == (2000, 20, 0), "бутстрап листа 0 = 2000 · 20 · 0": (CR.B, CR.BLOCK, CR.SEED) == (2000, 20, 0),
            "p < 0,05 в правиле п. 2.2": "p1 < 0.05" in src, "3 из 5 в правиле п. 2.2": "neg >= 3" in src, "лаг DM = 5": dmod.LAG == 5}
    return bad + ["в коде отличается от плана: " + k for k, ok in code.items() if not ok]


def claims_fail(t: dict) -> list:
    return selection_fail(t) + registry_fail(t) + holm_fail(t) + thresholds_fail()


def first_prose(part: str) -> int | None:
    """Позиция первого упоминания A0g вне строк таблиц раздела, если рядом нет подписи; иначе None."""
    pos = 0
    for ln in part.split("\n"):
        m = A0G.search(ln)
        if m and not ln.startswith("|"):
            i = pos + m.start()
            return None if "диагностический контроль" in part[i:i + 90] else i
        pos += len(ln) + 1
    return None


def label_prose(doc: str) -> str:
    """В прозе каждого раздела «## n.» первое упоминание A0g получает подпись (если её рядом нет)."""
    out = []
    for part in re.split(r"(?m)^(?=## )", doc):
        i = first_prose(part) if part.startswith("## ") else None
        out.append(part if i is None else part[:i] + A0G.sub(W.LBL, part[i:], count=1))
    return "".join(out)


def label_tables(doc: str) -> str:
    """Таблицы: в строке с A0g первое упоминание получает подпись «диагностический контроль (Ж-230)»; проза: см. `label_prose`."""
    t = "\n".join(A0G.sub(W.LBL, ln, count=1) if ln.startswith("|") and A0G.search(ln) and W.LBL not in ln else ln for ln in doc.split("\n"))
    return label_prose(t)


def label_bad(doc: str) -> list:
    n = [ln for ln in doc.splitlines() if ln.startswith("|") and A0G.search(ln) and W.LBL not in ln]
    bad = ["Z1: строка таблицы с A0g без подписи «диагностический контроль (Ж-230)»: %s" % n[0][:60]] if n else []
    for part in re.split(r"(?m)^(?=## )", doc):
        if part.startswith("## ") and first_prose(part) is not None:
            bad.append("Z1: проза раздела «%s» упоминает A0g без подписи" % part.split("\n")[0][:40])
    return bad


def budget_line(t: dict) -> str:
    c6 = t["cp"]["6"]
    return ("%d предрегистрированных конфигураций + A0g, диагностический контроль на месте не посчитанной по правилу CB2; всего %d из %d." % (len(W.PRE), c6["used"], c6["budget"]))


def section(t: dict, journal: str | None = None, state: str | None = None) -> str:
    mo = moments(JOURNAL.read_text(encoding="utf-8") if journal is None else journal, STATE.read_text(encoding="utf-8") if state is None else state)
    c0, c6, a2 = t["cp"]["0"], t["cp"]["6"], W3.pair(t, "A2 vs A0")
    sel = "; ".join("%s: из 17 — %s (QLIKE %s), из 18 мест — %s, QLIKE A0g %s" % (W.WN[w], W.best(t, w), W.f(W.q(t, W.best(t, w), w)), W.best_with_diag(t, w), W.f(W.q(t, "A0g", w))) for w in ("2017_2021", "all"))
    return "\n".join([
        "### 5.1. A0g: статус, причина, моменты (Ж-230)", "",
        "- **Статус.** A0g (S + GBM) — дополнительный диагностический контроль (Ж-230), а не заранее утверждённая основная конфигурация. Бюджет: %s Лучшая конструкция V6 выбирается только из %d предрегистрированных; выбор с A0g и без неё (%s) — не меняется." % (budget_line(t), len(W.PRE), sel),
        "- **Причина.** В плане (коммит 99bd6681) GBM-конфигурации (A2, C2, C4, E2) сравнивались по правилу п. 2.2 с линейной базой A0, и обучатель не отделялся от признаков: выигрыш A2 над A0 (Δ %s) мог быть вкладом нелинейного обучателя, а не порядка событий." % W.iv(a2),
        "- **Моменты (журнал `resheniya.md`, state).** Лист 1b перепроверен %s–%s (09.11); вопрос 1 открыт %s (09.11), срок ответа %s, умолчание «A0g вместо CB2»; умолчание применено %s (09.11); лист 6 (A0g) посчитан к %s (09.11, state); утверждено владельцем Ж-230 %s (09.11): «%s»." % (mo["review_start"], mo["review_end"], mo["question"], mo["deadline"], mo["default"], mo["leaf6"], mo["owner"], mo["quote"]),
        "- **Прямо.** Решение о добавлении A0g принято после просмотра результатов листа 1b — это отступление от предрегистрации, а не предрегистрация. Критерии отбора и статистические пороги (δ 0,0005, p 0,05 без поправки, не менее 3 лет из 5, пять условий «go», Холм N = %d и N_wide = %d, бутстрап блок 20 · 2000 · seed 0) не менялись. Выводы против A0 сохранены как были, против A0g — рядом:" % (c0["N"], c0["N_wide"]), "",
        W2.versions_block(t),
        "- **Реестр и множественность.** Строка `Ж-227:A0g` есть в `ALL_TRIALS.csv`; строк реестра %d, использовано %d из %d. N = N_V5 %d + бюджет V6 %d = %d; N_wide = %d + %d = %d. A0g занимает одно из %d мест бюджета, уже учтённых в N и N_wide (CB2 не считалась, сверх бюджета испытаний не добавлено), поэтому её испытание входит в N и N_wide, а не добавляется к ним. p Холма в `G_INFO.csv` (все пары с A0g) посчитаны с этими N и N_wide и равны пересчёту (проверка генератора)." % (len(t["trials"]), c6["used"], c6["budget"], c0["N_V5_declared"], c0["budget_V6"], c0["N"], c0["N_wide_V5"], c0["budget_V6"], c0["N_wide"], c0["budget_V6"])]) + "\n"
