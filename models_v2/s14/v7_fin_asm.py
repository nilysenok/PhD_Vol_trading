"""V7 лист FIN, G50: сборка `HARJ_HYBRID_SBER_V7_M10_MOE_FINAL.md`, `REPRODUCTION_V7.md` и `FIN_WEIGHTS.png` из таблиц на диске. Сборка дважды даёт тот же текст. Утверждения текста, зависящие от данных, проверяет `claims_fail`:
не выполнены — сборка отказывает (конфигурация «не считается» не может стоять как посчитанная; версия 1 не может быть основной)."""
from __future__ import annotations

from models_v2.s14 import v7_0_io as IO, v7_fin_data as D, v7_fin_doc as W, v7_fin_doc2 as W2, v7_fin_text as TX

FINAL = IO.ROOT / "docs/otchety/HARJ_HYBRID_SBER_V7_M10_MOE_FINAL.md"
REPRO = IO.REP / "REPRODUCTION_V7.md"
PNG = IO.REP / "FIN_WEIGHTS.png"
STAGES = ["0", "1a", "1b", "2", "DX", "3", "4", "FIN"]
SECTIONS = ["Результат", "Математическая архитектура", "Данные", "Реестр 11 мест", "bias по окнам", "QLIKE по годам", "Сравнение с XGBoost", "Абляции", "Веса экспертов", "Устойчивость",
            "После 2021", "Причины выигрыша и проигрыша", "Отступления программы", "Решение о перспективах", "Реестр испытаний"]


def load() -> dict:
    """Таблицы FIN_* с диска + CHECKPOINT, число параметров энкодеров и хэши (все из файлов)."""
    from models_v2.s14 import v7_1e_net as NET
    t = {n: IO.read_csv(n + ".csv") for n in D.TABLES}
    t["FIN_PAIRS"]["hyp"] = t["FIN_PAIRS"]["hyp"].astype(str)
    t["cp"], t["enc"] = D.read_cp(), dict(NET.PARAMS)
    t["shas"] = dict(inputs=IO.inputs_hash(), trials=IO.sha_file(IO.REP / "ALL_TRIALS.csv")[:16])
    return t


def claims_fail(t: dict) -> list:
    bad, r = [], t["FIN_REGISTRY"].set_index("place")["status"]
    if len(r) != 11 or any(r[i] != "посчитана" for i in (1, 2, 3, 4, 5, 6, 7, 8, 10)) or any(not str(r[i]).startswith("не считается") for i in (9, 11)):
        bad.append("реестр: места 1–8, 10 — «посчитана», 9 и 11 — «не считается»")
    if set(t["FIN_ALL"]["series"]) != set(D.SER):
        bad.append("FIN_ALL: набор рядов ≠ 9 контролей и 9 конфигураций (9 и 11 не считались)")
    v1 = t["FIN_V1"]
    for c in D.CFG[:7]:
        x = v1[(v1["config"] == c) & (v1["window"] == "2018_2021")].iloc[0]
        a = W.q(t, c, "2018_2021")
        if abs(a - x["qlike_v2"]) > 1e-9 or abs(a - x["qlike_v1"]) < 1e-12:
            bad.append("%s: основная таблица не равна версии 2 (или равна версии 1)" % c)
    if not (W.q(t, "V6:A2", "2018_2021") < W.q(t, "K3", "2018_2021") and W.q(t, "V6:A2", "2019_2021") > W.q(t, "K3", "2019_2021")):
        bad.append("оговорка про 2018 год: A2 не лучше K3 на 2018–2021 или не хуже на 2019–2021")
    w = t["FIN_WEIGHTS"]
    if int(w[(w["config"] == "cfg10") & (w["quantity"] == "pi0") & (w["period"] == "2018")]["n"].iloc[0]) != 0:
        bad.append("веса 10: 2018 должен быть без Stage B")
    return bad


def sec(n: int, title: str, body: str) -> str:
    return "## %d. %s\n\n%s\n" % (n, title, body.rstrip("\n"))


def build(t: dict) -> str:
    bad = claims_fail(t)
    if bad:
        raise ValueError("утверждения документа не выполнены на данных: %s" % bad)
    b = [W.sec1(t), W.sec2(t, t["cp"], t["enc"]), W.sec3(t), W.sec4(t), W.sec5(t), W.sec6(t), W.sec7(t), W2.sec8(t), W2.sec9(t), W2.sec10(t), W2.sec11(t), W2.sec12(t), W2.sec13(t), W2.sec14(t), W2.sec15(t, t["cp"], t["shas"])]
    titles = ["Результат: преимущество V7 над XGBoost, LightGBM, NTP2 и A2", "Математическая архитектура", "Данные", "Реестр 11 мест", "QLIKE, RMSE, MAE, R², bias по окнам", "QLIKE по годам 2018…2026", "Сравнение с XGBoost, LightGBM, NTP2, A2",
              "Абляции", "Веса экспертов по годам, якорь, специализация", "Устойчивость", "После 2021", "Причины выигрыша и проигрыша", "Отступления программы и ошибки дирижёра", "Решение о перспективах", "Реестр испытаний, контрольные хэши, CHECKPOINT"]
    head = [W.first_line(t), "", TX.STATUS, "", "# HARJ_HYBRID_SBER_V7_M10_MOE_FINAL — итог программы HAR-J Hybrid V7 (Ж-232, Ж-233, Ж-235, Ж-236), SBER, h = 1", ""]
    return "\n".join(head + [sec(i + 1, titles[i], b[i]) for i in range(15)]) + "\n"


def reproduction() -> str:
    cmds = "\n".join("python3 -W ignore -m models_v2.s14.run_v7 --stage %s\npython3 -W ignore -m models_v2.s14.run_v7 --stage %s --check" % (s, s) for s in STAGES)
    return ("# REPRODUCTION_V7 — воспроизведение программы HAR-J Hybrid V7 (Ж-232), SBER, h = 1\n\nСтатус: DEVELOPMENT / HISTORICAL WALK-FORWARD, не независимый OOS. Запуск из корня репозитория, по одной стадии. Порядок: 0 (аудит, паспорт M10, последовательности) → 1a (инфраструктура, реестр) → 1b (энкодеры, конфигурации 1–3) → 2 (конфигурации 4–7) → DX (диагностика) → 3 (конфигурация 8) → 4 (конфигурация 10; 11 не считается) → FIN (итог). "
            "Стадия без `--check` ПИШЕТ выходы, `--check` только читает (пересчёт в памяти, сверка с диском, хэш выходов до и после). Прогнозы — `data/regen/harj_hybrid_v7/`; FIN прогнозов не считает, конфигурации не расходует.\n\n```\n%s\n```\n\n"
            "Тесты: `python3 -W ignore -m pytest tests/test_v2_v7_*.py -q -p no:cacheprovider`.\n\n"
            "**Протокол регрессии (G54).** Причина риска: lightgbm и xgboost (V5–V6) и torch (V7) приносят каждый свою библиотеку OpenMP; многопоточный torch после загрузки lightgbm/xgboost роняет процесс сегфолтом (rc 139) на первом softmax. "
            "Поэтому `models_v2/s14/__init__.py` закрепляет torch на одном потоке (`set_num_threads(1)`, `set_num_interop_threads(1)`) при импорте пакета; V7 считает в один поток, числа не меняются. С закреплением единый прогон одним процессом проходит: "
            "`python3 -m pytest -q -p no:cacheprovider tests/test_v2_v5_*.py tests/test_v2_v6_*.py tests/test_v2_v7_*.py` (860 passed). Тесты V7 можно запускать и отдельным процессом.\n\nВыходы: `docs/otchety/harj_hybrid_v7/` (таблицы, отчёты листов, `FIN_*.csv`, `FIN_WEIGHTS.png`, `CHECKPOINT.json`, `ALL_TRIALS.csv`), `docs/otchety/HARJ_HYBRID_SBER_V7_M10_MOE_FINAL.md`. Итоговый документ собирается из таблиц на диске; `--stage FIN --check` пересобирает его и сверяет с файлом.\n" % cmds)


def plot_weights(t: dict, path=PNG) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    w = t["FIN_WEIGHTS"].drop_duplicates(["config", "quantity", "period"])
    yrs = [str(y) for y in range(2018, 2027)]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for c, k, lab in (("cfg6", "pi1", "6: π₁"), ("cfg6", "pi2", "6: π₂"), ("cfg8", "w", "8: w (якорь)"), ("cfg8", "pi1", "8: π₁"), ("cfg10", "pi1", "10: π₁"), ("cfg7", "pi1", "7: π₁")):
        g = w[(w["config"] == c) & (w["quantity"] == k) & w["period"].isin(yrs)].set_index("period")["mean"].reindex(yrs)
        ax.plot([int(y) for y in yrs], g.to_numpy(float), marker="o", label=lab)
    ax.axvline(2021.5, color="grey", ls=":")
    ax.set_xlabel("год цели"); ax.set_ylabel("средний вес"); ax.legend(ncol=3, fontsize=8); ax.set_title("V7: веса экспертов и якоря по годам (SBER, h = 1)")
    fig.savefig(path, dpi=110, metadata={"Software": None})
    plt.close(fig)


def write_docs(t: dict) -> None:
    FINAL.write_text(build(t), encoding="utf-8")
    REPRO.write_text(reproduction(), encoding="utf-8")
    plot_weights(t)
