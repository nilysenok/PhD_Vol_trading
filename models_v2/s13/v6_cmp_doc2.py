"""V6 лист Z2, G64: разделы 4–9 документа (точный лучший результат, MCS, воспроизведение, что не сопоставимо, отступления, файлы) и сборка `build(t)`. Статические тексты — без чисел с запятой."""
from __future__ import annotations

import re

from models_v2.s13 import v6_cmp_doc as D1, v6_cmp_fmt as F, v6_cmp_io as IOC, v6_cmp_reg as R, v6_cmp_tab as T

PARTS = ["Главная таблица V1–V6", "QLIKE по годам", "Статистика", "Точный лучший результат SBER", "MCS 90 %", "Воспроизведение чисел программ", "Что не сопоставимо и почему", "Отступления", "Источники и файлы"]
NOT_COMP = [
    "V3 как номер программы Ж-221 не исполнялся (CHECKPOINT V3 не создавался): строка V3 — панельный гибрид Ж-217–Ж-220 (толкование дирижёра, журнал Ж-230).",
    "Все ряды V1 и V2 — класс «потолок» (набор, окно или финалист выбраны по crit_wf на тех же датах SBER, лист 0): в выбор лучшего не входят, показаны отдельным столбцом.",
    "V3, V4, V5 выбирались по панели из 16 бумаг (M16), SBER в выбор не входит; V6 — только SBER, конфигурации последовательно зависели от результатов SBER (условные конфигурации по правилу «новой информации» плана). Сравнение по SBER не равно сравнению программ по их собственному критерию.",
    "K2 и K3 (дизайн Ж-204/206/210) выбирались на SBER 2017–2021 — оговорка для всех сравнений с ними.",
    "Колонка (б) — выбор после просмотра (потолок); её p Холма с N программы и с N = 722 и N_wide = 3122 — по перечню покрытия ниже, не гарантия отсутствия подгонки.",
    "Год 2026 неполный; в выбор лучшего и в «годы лучше» не входит.",
    "V1 разделена на две строки (Ж-211 и Ж-212), потому что у программ разные финалисты и разные N; V6 имеет два объявленных лучших (A2 по 2017–2021, E2 по всей истории).",
    "A0g (S + GBM) — диагностический контроль, а не гибрид HAR-J: в пулы (б) и в выбор лучшего не входит.",
]
DEVIATIONS = [
    "Отступление: V1 — две строки таблицы (Ж-211, Ж-212), V6 — два ряда (а); причина — у программ несколько объявленных итоговых или лучших гибридов.",
    "Отступление: конфигурации V6 отнесены к классу «PIT-сопоставим» (толкование; класс V6 в листе 0 не присваивался). Чувствительность — множество «основное без V6» в разделе 4.",
    "Отступление: семейство Холма — все сравнения окна таблицы CMP_TESTS (гибриды × K1, K2, K3, K8); N программы взят из её CHECKPOINT/FINAL (колонка N_prog), N = 722 и N_wide = 3122 — из V6.",
    "Отступление: у двух рядов (J211feat:lgbm_S345, PANxgb:A2) в таблицах своей программы числа нет — воспроизведение невозможно, числа даны только пересчётом.",
    "Отступление: K1, K2, K3 в V3 берутся из общих файлов листа 0 (отдельного источника у V3 нет); побайтно сверены K1/K2/K3 с components.parquet (V1, V2) и контролями V6.",
]


def best_section(t: dict) -> str:
    b = t["BEST"]
    out = []
    for w in D1.WIN3:
        x = b[(b["set"].str.startswith("основное (")) & (b["window"] == w)].set_index("ref_kind")
        s2, k2, k3 = x.loc["второй"], x.loc["K2"], x.loc["K3"]
        fmt = lambda r: "не применимо (лучший ряд сам %s)" % r["ref"] if r["ref"] == r["best"] else "%s (QLIKE %s)" % (F.dci(r), F.n4(r["ref_qlike"]))
        out.append("**%s.** Лучший: **%s** (%s), QLIKE %s. Отрыв (Δ в %% уровня K1, 90 %% интервал, DM p): от второго ряда (%s) — %s; от K2 — %s; от K3 — %s. MCS 90 %%: %d из %d рядов (%s). %s." % (
            F.win_name(w), s2["best"], s2["best_cls"], F.n4(s2["best_qlike"]), s2["ref"], fmt(s2), fmt(k2), fmt(k3), int(s2["mcs_size"]), int(s2["n_series"]), s2["mcs_members"].replace(";", ", "), s2["verdict"][0].upper() + s2["verdict"][1:]))
    rows = [[r["set"], F.win_name(r["window"]), r["best"], F.n4(r["best_qlike"]), r["ref"], F.dci(r)] for _, r in b[b["ref_kind"] == "второй"].iterrows()]
    return "\n\n".join(out) + "\n\nЧувствительность к составу множества (лучший ряд и отрыв от второго):\n\n" + F.md_table(["Множество", "Окно", "Лучший", "QLIKE", "Второй", "Δ к второму, %"], rows)


def mcs_section(t: dict) -> str:
    m = t["MCS"]
    rows = []
    for (name, w), g in m.groupby(["set", "window"], sort=False):
        mem = g[g["in_mcs90"]]
        rows.append([name, F.win_name(w), "%d из %d" % (len(mem), len(g)), ", ".join(mem["model"]) if len(mem) < len(g) else "все ряды множества", F.n3(g["mcs_p"].min())])
    return F.md_table(["Множество", "Окно", "в MCS 90 %", "ряды", "минимальный MCS p"], rows)


def repro_section(t: dict) -> str:
    r = t["REPRO"]
    q = r[r["kind"] == "qlike"]
    ok, no = q[q["status"] == "совпало"], q[q["status"] == "нет числа в таблицах"]
    idn = r[r["kind"] == "identity"]
    return ("Пересчёт QLIKE из сохранённых прогнозов против чисел таблиц своих программ (допуск 1e-9): сверено %d чисел, совпало %d, расхождений %d, максимальная разность %.1e. Без числа в таблицах программы: %s. "
            "Контроли побайтно (sha256 массивов): %d проверок, совпало %d." % (len(q) - len(no), len(ok), len(q) - len(no) - len(ok), ok["abs_diff"].max(), ", ".join(no["series"]) or "нет", len(idn), int((idn["status"] == "совпало").sum())))


def files_section(t: dict) -> str:
    return ("Таблицы: docs/otchety/harj_hybrid_v6/cmp/ (%s), CHECKPOINT_CMP.json. Код: models_v2/s13/v6_cmp_*.py, run_v6_cmp.py. Реестр: %d строк, %d рядов." % (", ".join(IOC.TABLES), len(t["SERIES"]), t["SERIES"]["series"].nunique()))


NUM = re.compile(r"[−+-]?\d+[,.]\d+(?:e[-+]\d+)?")          # со знаком: перевёрнутый знак Δ — расхождение


def compare(doc: str, fresh: str) -> tuple:
    """-> (сверено десятичных чисел, расхождений): числа документа и пересборки из таблиц попарно по порядку; лишнее или недостающее число — расхождение."""
    a, b = NUM.findall(doc), NUM.findall(fresh)
    return len(b), sum(x != y for x, y in zip(a, b)) + abs(len(a) - len(b))


def build(t: dict) -> str:
    n = t["SERIES"]
    u_dates = int(n["n_dates"].min())
    head = ["# Сравнение SBER (h = 1) по программам V1–V6: HAR-J (K1), XGBoost (K2), LightGBM (K3) и гибриды", "", D1.first_line(t), "",
            "DEVELOPMENT / HISTORICAL WALK-FORWARD, не независимый OOS; первая цифра — потолок. Общие даты: %d (d = %s … %s). Δ — в процентах уровня QLIKE K1 окна, 90 %% интервал — блоковый бутстрап (блок 20, 2000 повторов, seed 0), DM — Ньюи–Уэст лаг 5, односторонний «первый ряд лучше», Холм — N = 722 / N_wide = 3122 / N программы. Минус — лучше. Правила объявлены до счёта." % (u_dates, n["first_d"].iloc[0], n["last_d"].iloc[0]), ""]
    body = [D1.main_table(t), D1.years_table(t), D1.stats_table(t) + "\n\nПокрытие N реестрами:\n\n" + D1.cov_table(t), best_section(t), mcs_section(t), repro_section(t),
            "\n".join("- " + s for s in NOT_COMP), "\n".join("- " + s for s in DEVIATIONS), files_section(t)]
    return "\n".join(head) + "\n" + "\n\n".join("## %d. %s\n\n%s" % (i + 1, p, b) for i, (p, b) in enumerate(zip(PARTS, body))) + "\n"
