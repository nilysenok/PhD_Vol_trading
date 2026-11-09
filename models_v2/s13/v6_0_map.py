"""V6-0, G4: MAP_DONE_V6.md — реестр проверенных механизмов (этап 3 блока Ж-227) и карта направлений A–F V6 к ближайшим проверенным конструкциям. Текст строится из таблицы метрик G3 (QLIKE вся история на общих датах, 2271 дата),
поэтому числа не расходятся с BEST_SBER.csv. Статусы и «главная проблема» — по итоговым документам программ (ссылки в ячейках); код целиком не перечитывался."""
from __future__ import annotations

import pandas as pd

V4 = "docs/otchety/harj_hybrid_v4/DONE_EXPERIMENTS.md"
V5 = "docs/otchety/harj_hybrid_v5/MAP_DONE.md"
HF = "docs/otchety/hybrid_final_v1/HYBRID_HAR_BOOSTING_FULL_RESEARCH_FINAL.md"
HS = "docs/otchety/hybrid_struct_v1/HAR_BOOSTING_STRUCTURAL_REDESIGN_FINAL.md"
V2 = "docs/otchety/HARJ_HYBRID_V2_FINAL_RESEARCH.md"
V4F = "docs/otchety/HARJ_HYBRID_V4_FINAL.md"
V5F = "docs/otchety/HARJ_HYBRID_V5_FINAL.md"


def ids(prefix: str, names: str) -> list:
    return [prefix + n for n in names.split()]


A = ["J210a:A%d_%s" % (i, s) for i in range(1, 5) for s in ("fixed_loyo", "inv_qlike_252", "inv_rmse_252", "roll_252", "shrink_252")] + ["J210a:A1_roll_126"] + ["J212F:F1", "J212F:F2", "M", "K8"]
ROWS = [
    ("Обычное смешивание", "реализован и посчитан", "Ж-210/211 (A1–A4), Ж-212 (F1, F2), контроли M, K8", A, "второй бустинг не добавляет (H7 против H1: Δ +0,0007, p 0,509); финалист семейства — по crit_wf на SBER (потолок)", "models_v2/s6/e/rules.py", HF),
    ("Динамические веса", "реализован и посчитан", "Ж-211 (HF7: EMA, ограничение скорости), V4 B1–B4, B6", [s for s in ids("J211w:", "H1_roll_126_ema21 H1_roll_126_ema63 H1_roll_126_rate0.005 H1_roll_126_rate0.02 H2_roll_252_ema21 H2_roll_252_ema63 H2_roll_252_rate0.005 H2_roll_252_rate0.02 H7_roll_252_ema63 H7_roll_252_rate0.02")] + ids("V4:", "B1 B2 B3 B4 B6"),
     "сглаживание веса не улучшает перенос (A1_roll_126 лучше всех H1_*, GATES_HF G6)", "models_v2/s7/hf_rules.py", V4),
    ("Online QLIKE", "реализован частично: калибровка V4 E и K8-RQC; Hedge F3/F4 (Ж-212) только запланирован", "V4 E1–E4, E6, RQC1–RQC3", ids("V4:", "E1 E2 E3 E4 E6 RQC1 RQC2 RQC3"), "подтверждённого преимущества над лучшим бустингом нет (V4 FINAL); строки SBER — `E_SBER.csv`, `RQC_SBER.csv` в `docs/otchety/harj_hybrid_v4/`", "models_v2/s11/v4_e_calib.py", V4F),
    ("Graph-Score", "реализован и посчитан (SBER — диагностика, не в M16)", "V5 GS1–GS5", ids("V5:", "GS1 GS2 GS3 GS4 GS5"), "передача между бумагами (граф W) в V6 невозможна: один тикер; остаётся собственный score (E1/E5-подобный)", "models_v2/s12/v5_gs_run.py", V5F),
    ("Spectral Energy", "реализован и посчитан", "V5 SP1–SP4", ids("V5:", "SP1 SP2 SP3 SP4"), "частные корреляции high-полосы −0,023…−0,041, ΔR² ≤ 0,17 %, mid ≈ 0 (V5 лист 0, `PASS_SPECTRAL.csv`)", "models_v2/s12/v5_sp_feat.py", V5F),
    ("Conditional Moments", "только диагностика (счёта конфигураций нет)", "V5 DX1 (`DX1_MOMENTS.csv`)", [], "нет результата QLIKE; моментная регуляризация эквивалентна первому порядку условий QLIKE-корректора", "models_v2/s12/v5_dx1_mom.py", V5F),
    ("Jump-aware", "реализован и посчитан (V4 I); диагностика V5 DX1 (`DX1_JUMPS.csv`)", "V4 I1–I3, Ж-214 S1, S3, J1", ids("V4:", "I1 I2 I3") + ids("V2:", "S1 S3 J1"), "J > 0 в ~82 % дней: событие не редкое и не концентрирует потери (V5 лист 0, PASS_JUMP.csv)", "models_v2/s11/v4_i_comp.py", V4F),
    ("Intraday Shape", "реализован и посчитан", "V4 C1–C4 (C-a), Ж-214 T1–T4, P1, P2", ids("V4:", "C1 C2 C3 C4") + ids("V2:", "T1 T2 T3 T4 P1 P2"), "доли open/mid/close, herf, lasth — функции агрегатов q = r² по времени; порядок событий внутри окна не кодируют", "models_v2/s11/v4_z_shape.py", V4F),
    ("State-Space Calibration", "реализован и посчитан", "V4 E5 (Калман), E1–E4, E6", ids("V4:", "E1 E2 E3 E4 E5 E6"), "подтверждённого преимущества над лучшим бустингом нет (V4 FINAL); Калман E5 — `E_SBER.csv`", "models_v2/s11/v4_e_kalman.py", V4F),
    ("Семивариации", "реализован и посчитан", "Ж-214 S1, S3 (HAR-J + RS±), Ж-212 D0_*", ids("V2:", "S1 S3") + ["J212D:D0_full", "J212D:D0_drop_a"], "финалист ступени Л5 (J1) crit_wf 0,2339 хуже финалиста Л2 (H2) 0,2199 (`statistical_validation/FINALISTS.csv`)", "models_v2/s9/hv2_m10.py", V2),
    ("Режимы", "реализован: V4 C8 (два режима) и C5–C10; режимы Ж-210/211 — ABANDON (0 trials)", "V4 C5–C10", ids("V4:", "C5 C6 C7 C8 C9 C10"), "режимная конструкция отклонена (V4-0, этап F)", "models_v2/s11/v4_cb_fit.py", V4),
    ("Компоненты C + J", "реализован и посчитан", "V4 I1–I3, Ж-214 T3, R2T, Ж-212 Dx_J", ids("V4:", "I1 I2 I3") + ["V2:T3", "V2:R2T", "J212D:Dx_J"], "R2T исключена правилом V11; Dx_J значимо хуже XJ (p 0,029 сырой), V4-0 этап I", "models_v2/s11/v4_i_fit.py", V4),
    ("Ортогональная поправка", "реализован и посчитан", "V4 H1–H8", ids("V4:", "H1 H2 H3 H4 H5 H6 H7 H8"), "подтверждённого преимущества над лучшим бустингом нет (V4 FINAL); строки SBER — `H_SBER.csv`", "models_v2/s11/v4_h_fit.py", V4F),
    ("Защита от отрицательного переноса", "реализован и посчитан", "V4 NTP1–NTP4, CS1–CS2 (холодный старт)", ids("V4:", "NTP1 NTP2 NTP3 NTP4 CS1 CS2"), "лучший PIT-сопоставимый ряд SBER; его выигрыш за 2017–2021 к лучшему из K2/K3 меньше различимого эффекта G7 (`CRIT_SBER.csv`, section noise)", "models_v2/s11/v4_ntp_state.py", V4F),
]
DIR = [
    ("A Path Signature", "Ж-214 T1–T4, P1, P2 (части дня), V4 C1–C4 (lasth, herf), V5 SP1–SP4 (полосы DCT)", "все — симметричные агрегаты или локализованные окна; iterated integrals порядка 2 (площадь Леви между ценой, накопленной RV, объёмом и временем) порядок событий кодируют и не сводятся к ним", "новое"),
    ("B Time-Reversal", "Ж-214 rv_open/rv_close (утро/вечер), rs_neg; V4 C-a lasth", "«ранняя против поздней концентрации отрицательных доходностей» и «утро против вечера» ≈ rv_open/rv_close и rs_neg × часть дня → эквивалентно (исключить); «RV после экстремального движения» и «асимметрия восстановления» не вычислялись", "эквивалентно Ж-214 T/P и V4 C1–C4 для части признаков; остальные — новое"),
    ("C Measurement Uncertainty", "Ж-214 `seasonality/DIAGNOSTICS_V.md`, `SIGNATURE_NOISE.csv`: нулевые доходности 1,1–3,1 %, signature plot — диагностика", "диагностика описательная; признаки качества измерения в коррекцию (g_ML(Z, Q)) не подавались; RV на сетках M20/M30/M60 подвыборкой ни в одном ряду не использовалась", "новое (диагностика уже есть, гибрида нет)"),
    ("D Historical Analog", "V4 E1–E4 (веса по времени ω, полураспад), CS, `models_v2/s5/b2/loyo.py::nearest` (окно годов для HP)", "E1–E4 — локальная поправка с ядром только по времени: при ядре без состояния Z аналоги ≡ E1/E2 (исключить); аналоги по вектору Z (RV_history, asymmetry, shape, volume) не проверялись", "новое при ядре по Z; ядро только по времени — эквивалентно V4 E1/E2"),
    ("E Shock-Recovery", "Ж-214 DEC (rs_pos, rs_neg, J), V4 E (шоки описательно, E_BIAS.csv), I1–I3", "время от крупнейшего отрицательного движения, скорость восстановления, RV до/после шока не вычислялись; частично дублируют признаки path signature", "новое; при дублировании A — объединить с A"),
    ("F Distributionally Robust", "Ж-211 R2/R3 (клип и сжатие поправки), Ж-212 Cb/Ce (штраф μ‖g‖²), V4 CS (штраф холодного старта)", "штраф на величину/изменение поправки ≡ Cb/Ce/R3 (исключить); разброс потерь по прошлым годам и worst-environment risk как цель обучения не проверялись", "новое для критерия «разброс по средам»; штраф нормы g эквивалентен Ж-212 Cb/Ce"),
]


def best_of(met: pd.DataFrame, cols: list) -> str:
    x = met[(met["window"] == "all") & met["series"].isin(cols)].sort_values("qlike")
    if x.empty:
        return "нет результата"
    b = x.iloc[0]
    return "%s %.4f (%s)" % (b["series"], b["qlike"], "PIT" if b["class_"] in ("PIT-сопоставим", "контроль") else "потолок")


def build_text(met: pd.DataFrame, prior: pd.DataFrame) -> str:
    q = met[met["window"] == "all"].set_index("series")["qlike"]
    k23 = min(q["K2"], q["K3"])
    head = ["# MAP_DONE_V6 — реестр проверенных механизмов SBER h = 1 и карта направлений V6 (G4, этап 3 блока Ж-227)", "",
            "Статус: DEVELOPMENT / HISTORICAL WALK-FORWARD, не независимый OOS, первая цифра — потолок; Ж-227, лист 0. Лучший QLIKE — на общих 2271 датах (d = 2016-12-30 … 2026-02-26), вся история; «PIT» — класс «PIT-сопоставим» (`PRIOR_SBER.csv`), «потолок» — "
            "выбор на SBER. Лучший из K2/K3 = %.4f. Источник чисел — `BEST_SBER.csv`. Вариант без достоверных результатов SBER помечен «нет результата», не «неудача»." % k23, "",
            "## 1. Реестр механизмов (таблица этапа 3 блока)", "", "| Механизм | Реализован и посчитан? | Программа, id | Лучший QLIKE SBER (ряд, вся история) | Главная проблема | Код | Отчёт |", "|---|---|---|---|---|---|---|"]
    body = ["| %s | %s | %s | %s | %s | `%s` | `%s` |" % (m, st, pr, best_of(met, cols), prob, code, rep) for m, st, pr, cols, prob, code, rep in ROWS]
    d = ["", "## 2. Направления A–F V6: ближайшая проверенная конструкция и отличие", "", "| Направление | Ближайшая проверенная конструкция | Чем V6 отличается математически | Вердикт |", "|---|---|---|---|"]
    d += ["| %s | %s | %s | %s |" % x for x in DIR]
    tail = ["", "Ссылки: `docs/otchety/harj_hybrid_v5/MAP_DONE.md` (карта V5), `docs/otchety/harj_hybrid_v4/DONE_EXPERIMENTS.md` (V4-0), `docs/otchety/harj_hybrid_v2/seasonality/DIAGNOSTICS_V.md`.", ""]
    return "\n".join(head + body + d + tail)
