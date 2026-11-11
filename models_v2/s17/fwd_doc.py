"""Л6: отчёт `FWD_REPORT.md` только из таблиц (протокол HARJ_HYBRID_SBER_FWD, разделы 5 и 10).

Все числа и даты берутся из таблиц семейства и описания (CSV `FWD_*.csv`); в тексте генератора чисел нет (AST-тест сканирует строковые
константы), параметры метода — из констант `fwd_eval`. Первая строка — статус; обязательный абзац раздела 10. `verify` перерисовывает отчёт
из таблиц и сверяет побайтно: подменённое число, слово «лучше/хуже» или «подтверждено» не по таблице — замечание. Читает только переданные таблицы.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_v2.s17 import fwd_eval as E

ND = 2
BAD_PHRASE = "лучший гибрид"
NA = "н/д"
VIOL, LOCK_VIOL = "НАРУШЕНИЕ ЗАМОРОЗКИ", "нарушение заморозки"


def _f(v, nd: int = ND, sign: bool = False) -> str:
    if v is None or not np.isfinite(float(v)):
        return NA
    s = f"{float(v):{'+' if sign else ''}.{nd}f}"
    return s.replace(".", ",")


def _s(v) -> str:
    return NA if v is None or v == "" or (isinstance(v, float) and np.isnan(v)) else str(v)


def _p(v) -> str:
    return _f(v, ND + 2)


def _tbl(head: list, rows: list) -> list:
    return ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)] + ["| " + " | ".join(map(str, r)) + " |" for r in rows]


def _word(delta: float) -> str:
    if not np.isfinite(delta):
        return NA
    return "лучше" if delta < 0 else ("хуже" if delta > 0 else "не отличается от")


def power_text(power: pd.DataFrame | None, label: str, n: int) -> str:
    """Мощность по POWER.csv для фактического n: исторические эффекты, блок как у семейства; нет строк — прямо так и сказано."""
    if power is None or len(power) == 0:
        return NA
    p = power[(power["kind"] == "istoricheskij") & (power["hyp"] == label) & (power["n_new"] == n) & (power["block"] == E.BLOCK)]
    if len(p) == 0:
        return f"{NA}: в POWER.csv нет строки для n = {n}"
    return "; ".join(f"{r['window']}: {_f(r['power_holm7'], ND + 1)}" for _, r in p.iterrows())


def render(t: dict, power: pd.DataFrame | None = None, violation: list | None = None) -> str:
    """violation — модули, загруженные бегуном вне списка заморозки: первой строкой «НАРУШЕНИЕ ЗАМОРОЗКИ: …» (допустимо только вместе со статусом замка)."""
    fam, ser = t["family"], t["series"]
    tp0, tp1 = (_s(fam["tp_first"].iloc[0]), _s(fam["tp_last"].iloc[0])) if len(fam) else (NA, NA)
    n = int(fam["n"].iloc[0]) if len(fam) else 0
    head = [f"{VIOL}: после прогона загружены модули вне списка заморозки: {', '.join(violation)}; результат не имеет статуса замороженной проверки.", ""] if violation else []
    out = head + [f"Статус: проверка на новом периоде {tp0}…{tp1}, одно открытие по протоколу; X1, ZB заморожены до открытия.", "",
           "Кандидаты X1 и ZB, контроли K1, K2, K3; ZE не допущена. Метрика — QLIKE на общих парах всех пяти рядов.",
           f"Δ — в % среднего QLIKE базы (второго ряда пары); p — DM Ньюи–Уэст, лаг {E.LAG}, односторонний «первый лучше»; "
           f"интервал {E.CI} % — блочный бутстрап (блок {E.BLOCK}, {E.B} повторов, seed {E.SEED}); поправка Холма по {E.M_FAMILY}, α = {_f(E.ALPHA)}.", "",
           "## Ряды (абсолютный QLIKE)", ""]
    out += _tbl(["ряд", "роль", "QLIKE", "n", "d первая", "d последняя", "t⁺ первая", "t⁺ последняя"],
                [(r.series, r.role, _f(r.qlike, ND + 2), r.n, r.d_first, r.d_last, r.tp_first, r.tp_last) for r in ser.itertuples()])
    out += ["", "## Семейство гипотез", ""]
    out += _tbl(["№", "гипотеза", "n", "QLIKE A", "QLIKE B", "Δ, %", f"интервал {E.CI} %", "DM", "p", "p Холма", "вывод", "мощность по POWER.csv"],
                [(r.hyp, f"{r.A} лучше {r.B}", r.n, _f(r.qlike_A, ND + 2), _f(r.qlike_B, ND + 2), _f(r.delta_pct, sign=True),
                  f"[{_f(r.lo90_pct, sign=True)}; {_f(r.hi90_pct, sign=True)}]", _f(r.dm_stat), _p(r.p_one), _p(r.p_holm), r.verdict,
                  power_text(power, f"{r.A}<{r.B}", int(r.n))) for r in fam.itertuples()])
    out += [""] + [f"- {r.hyp}: {r.A} {_word(r.delta_pct)} {r.B} на {_f(abs(r.delta_pct)) if np.isfinite(r.delta_pct) else NA} % QLIKE; {r.verdict}." for r in fam.itertuples()]
    out += ["", "«Не подтверждено» не означает «опровергнуто»: рядом стоят Δ, интервал и мощность. Выбор лучшего кандидата не делается; вывод — по строкам кандидата.", "",
            "## Описание (без вывода)", "", "MCS по пяти рядам:", ""]
    out += _tbl(["ряд", "QLIKE", "p MCS", "в наборе"], [(r.series, _f(r.qlike, ND + 2), _p(r.p_mcs), "да" if r.in_mcs else "нет") for r in t["mcs"].itertuples()])
    for key, lab in (("months", "Месяцы t⁺"), ("states", "Состояния e_d (E — экстремальный день, N — прочие)")):
        g = t[key]
        name = g.columns[0] if len(g) else key
        out += ["", f"{lab}:", ""] + _tbl([name, "n"] + [f"QLIKE {s}" for s in E.SERIES] + [f"Δ {h}, %" for h, _, _ in E.HYP],
            [[r[0], r[1]] + [_f(v, ND + 2) for v in r[2 + 4:2 + 4 + len(E.SERIES)]] + [_f(v, sign=True) for v in r[2 + 4 + len(E.SERIES):]] for r in g.itertuples(index=False)])
    out += ["", "ZB против ZB0, X1 против M и cfg8:", ""]
    out += _tbl(["пара", "n", "QLIKE A", "QLIKE B", "Δ, %", f"интервал {E.CI} %", "p"],
                [(f"{r.A} против {r.B}", r.n, _f(r.qlike_A, ND + 2), _f(r.qlike_B, ND + 2), _f(r.delta_pct, sign=True), f"[{_f(r.lo90_pct, sign=True)}; {_f(r.hi90_pct, sign=True)}]", _p(r.p_one)) for r in t["descr"].itertuples()])
    if "checks" in t and len(t["checks"]):
        out += ["", "## Сверки и отступления", ""] + _tbl(["сверка", "статус", "подробности"], [(r.check, r.status, r.detail) for r in t["checks"].itertuples()])
    per = f"{tp0[:7]}…{tp1[5:7]}" if tp0 != NA else NA
    out += ["", "## После открытия", "",
            f"Любые изменения X1, ZB или новых гибридов, сделанные по результатам этого периода, не имеют на нём независимого подтверждения; в отчётах они помечаются «после просмотра {per}». "
            f"Следующая подтверждающая проверка — новый заранее зафиксированный протокол на данных после {tp1}. Повторных открытий этого периода нет. Число пар: {n}.", ""]
    return "\n".join(out)


def verify(text: str, t: dict, power: pd.DataFrame | None = None, violation: list | None = None, lock_status: str | None = None) -> list:
    """Замечания: пустой список — отчёт равен перерисовке из таблиц; плюс запрет фразы о «лучшем» и согласованность таблицы семейства.
    «НАРУШЕНИЕ ЗАМОРОЗКИ» в отчёте допустимо только вместе со статусом замка «нарушение заморозки: …», и наоборот."""
    bad = list(E.check_family(t["family"]))
    flagged = text.startswith(VIOL)
    if flagged != bool(violation) or flagged != str(lock_status or "").startswith(LOCK_VIOL):
        bad.append("отчёт и статус замка не согласованы по нарушению заморозки")
    if text != render(t, power, violation):
        bad.append("отчёт не равен перерисовке из таблиц (подменено число, слово или абзац)")
    if BAD_PHRASE in text.lower():
        bad.append("запрещённая фраза о лучшем кандидате")
    return bad
