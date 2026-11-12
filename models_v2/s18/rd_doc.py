"""S-RCD-EVAL: отчёт `S_RCD_REPORT.md` — собирается ТОЛЬКО из таблиц `S_RCD_*.csv`, ключей CHECKPOINT и файла гейтов (ни одно число не вписано вручную). `--write` пишет, `--check` пересобирает в памяти и сравнивает (только чтение; sha256 отчёта до и после).
Порядок (Ж-259: «сначала числа и вклад механизма, затем проверки и следующий шаг»): первая строка — пометка; 1) первая таблица (K1, K2, K3, B_mix, X1, W5-G, S-CD, S-CK, S-CDF, S-CKF, S-RCK, S-RCD, S-RCKZ, S-RL1: QLIKE, даты, n, Δ к S-CDF, S-RCK, W5-G, X1, K2, K3);
2) разложение вклада (разд. 6); 3) H-RCD1, защита (два набора), годы, половины, срезы; 4) статистика; 5) нулевое распределение; 6) параметры по месяцам; 7) RCD-DEC; 8) красный гейт M.G4 (а); 9) порядок, оговорки, следующий шаг.
Формулировки «подтверждено» и «лучше» — только из столбца verdict (разд. 7.5) и текста RCD-DEC; слова «эквивалентно» в отчёте нет.
Запуск: `python3 -W ignore -m models_v2.s18.rd_doc [--write | --check]`."""
from __future__ import annotations

import json
import re
import sys

import pandas as pd

from models_v2.s18 import rd_eval as E
from models_v2.s18.rd_doc_s1 import first_table, layer_table, section2
from models_v2.s18.rd_doc_s2 import dec_lines, guard_lines, h1_lines, null_lines, params_lines, slices_lines, stats_lines, years_lines
from models_v2.s18.z_doc_fmt import cap, tm

FIRST = E.MARK
GATES = E.ROOT / "docs/otchety/harj_struct/GATES_SRCD_IMPL.md"
ORDER = ("K1", "K2", "K3", "B_mix", "X1", "W5-G", "S-CD", "S-CK", "S-CDF", "S-CKF", "S-RCK", "S-RCD", "S-RCKZ", "S-RL1")


def tabs(rep=E.REP) -> dict:
    return {k: pd.read_csv(rep / n, comment="#") for k, n in zip(E.KEYS, E.OUTS)}


def gate_m4(p=GATES) -> dict:
    """Числа красного гейта M.G4 (а) — из строки гейтов `GATES_SRCD_IMPL.md`, не вписанные."""
    line = [l for l in p.read_text(encoding="utf-8").splitlines() if l.startswith("| M.G4")]
    m = re.search(r"НЕ ЗАКРЫТ \(а\): λ = 0,1: (\+[0-9,]+) % при пороге (\+[0-9,]+) %.*?C-K на тех же данных (\+[0-9,]+) %", line[0] if line else "")
    if not m:
        raise SystemExit("в GATES_SRCD_IMPL.md нет строки M.G4 (а) в ожидаемом виде")
    return dict(cd=m.group(1), thr=m.group(2), ck=m.group(3))


def gate_lines(t: dict) -> list:
    g, s = gate_m4(), t["decomp"]
    nl = lambda k: float(s[(s["section"] == "noise") & (s["item"] == k)]["value"].iloc[0])
    return ["Гейт M.G4 (а) листа IMPL-1 (`GATES_SRCD_IMPL.md`) — **красный, не закрыт**: нулевой сигнал, одна подгонка при λ = 0,1: C-D %s %%, C-K на тех же данных %s %% при пороге %s %% (числа — из файла гейтов)." % (g["cd"], g["ck"], g["thr"]),
            "Решение: **порог не менялся**; гейт не переписан и не помечен закрытым. Причина превышения в этом листе не исследовалась. Полная процедура S-RCD (блоки, θ = 0 в сетке, правило 2 из 3) измерена отдельно в разделе 5: QLIKE(R) − QLIKE(θ = 0) по 10 шумам — среднее %+.4f %%, 5 %% %+.4f %%, 95 %% %+.4f %%." % (nl("mean"), nl("q05"), nl("q95")),
            "Эта диагностика объявлена до QLIKE и порог гейта не заменяет; она показывает, как ведёт себя процедура в целом, а не закрывает M.G4 (а).", ""]


def order_lines(cp: dict, t: dict) -> list:
    k, r = cp[E.KEY], cp["S.RCD.registered"]
    RD = E.ROOT / "data/regen/harj_struct/RD"
    ts = ", ".join("%s %s" % (n, tm((RD / n).stat().st_mtime)) for n in ("rcd.parquet", "cdf.parquet", "ckf.parquet"))
    nxt = "H-RCD1 выполнено численно: разложение элементов R на C-D (обрезка, блоки, m) — отдельная регистрация (разд. 3.3 спецификации); решение — за владельцем (класс A)." if t["dec"].set_index("id").loc["H-RCD1", "passed"] else "H-RCD1 не выполнено: следующий шаг по этой линии определяет владелец (класс A)."
    return ["- Регистрация S-RCD: %s; файлы: %s; нулевая диагностика `S_RCD_NULL.csv`: %s; ключ `S.RCD.eval`: %s (до таблиц с p)." % (r["registered_at"], ts, tm(E.RN.OUT.stat().st_mtime), k["written_at"]),
            "- В-1 (спецификация): S-RCD включается с 2020-08-03, S-CDF — с 2018-07-02; разница включений входит в эффект R, общее активное окно показано отдельными строками разд. 2.",
            "- Оценка читает сохранённые ряды; гейты воспроизводимости проходов (W.G2…W.G5 листа IMPL-2) в этом листе не повторялись.",
            "- Отсутствие значимой разницы не доказывает отсутствия эффекта; выбор по минимуму QLIKE и по победам по годам не производился. Все числа — разработка на раскрытой истории: исследовательский результат, не OOS.",
            "- Следующий шаг: " + nxt]


def build(t: dict, cp: dict) -> str:
    tb = t["table"]
    cp_ = cap(tb["first_tplus"].iloc[0], tb["last_tplus"].iloc[0], tb["n"].iloc[0])
    L = ["# " + FIRST, "", "Выведено из таблиц `S_RCD_*.csv` (`rd_doc`); D_M: %s. Обычная QLIKE (среднее дневных потерь на всех датах, без обрезки); Δ в %% QLIKE базы, минус — лучше; 90 %% интервал — стационарный бутстрап. N программы %d / %d." % (cp_, cp[E.KEY]["n_holm"], cp[E.KEY]["n_holm_wide"]), "",
         "## 1. Первая таблица: QLIKE и Δ (%s)" % cp_, first_table(tb), "", "Слои доверия X1 над C рядов (Δ к X1; %s):" % cp_, layer_table(tb), "",
         "## 2. Разложение вклада механизма (разд. 6)"] + section2(t) + ["## 3. H-RCD1, защита (два набора), годы, половины, срезы"] + h1_lines(t, tb) + guard_lines(t, tb) + years_lines(t) + slices_lines(t, tb) + [
         "## 4. Статистика: DM, Холм, SPA, MCS, мощность"] + stats_lines(t, cp) + ["## 5. Нулевое распределение полной процедуры"] + null_lines(t) + ["## 6. Параметры по месяцам"] + params_lines(t) + [
         "## 7. Решение RCD-DEC"] + dec_lines(t) + ["", "## 8. Красный гейт M.G4 (а)"] + gate_lines(t) + ["## 9. Порядок, оговорки, следующий шаг"] + order_lines(cp, t)
    return "\n".join(L) + "\n"


def current(rep=E.REP, ckpt=E.CKPT) -> str:
    return build(tabs(rep), json.loads(ckpt.read_text(encoding="utf-8")))


def words(txt: str, t: dict) -> list:
    """«подтверждено» и «лучше» — только внутри формулировок разд. 7.5 (столбец verdict) и текста RCD-DEC; «эквивалент» — нигде."""
    rest = txt
    for s in list(t["stats"]["verdict"].dropna()) + [str(t["dec"].iloc[-1]["text"])]:
        rest = rest.replace(s, "")
    rest = rest.replace("минус — лучше", "")                  # правило знака Δ (№ 125), не вердикт
    return ["слово «подтверждено» / «лучше» вне формулировок разд. 7.5"] * bool(re.search(r"(?<![а-яё])(подтверждено|лучше)(?![а-яё])", rest)) + ["слово «эквивалент» в отчёте"] * ("эквивалент" in txt)


def check(rep=E.REP, ckpt=E.CKPT) -> list:
    p = rep / E.REPORT
    txt = p.read_text(encoding="utf-8") if p.exists() else ""
    t = tabs(rep)
    bad = [] if p.exists() and txt == current(rep, ckpt) else ["отчёт ≠ пересборке из таблиц"]
    bad += ["первая строка не пометка"] * (not txt.startswith("# " + FIRST + "\n"))
    bad += ["нет границ дат и n"] * ("t⁺ 2018-01-04…2026-11-10, n = 2198" not in txt)
    pos = [txt.find("| %s |" % s) for s in ORDER]
    bad += ["первая таблица: порядок рядов ≠ заданию"] * (min(pos) < 0 or pos != sorted(pos))
    heads = [txt.find("## %d." % i) for i in range(1, 10)]
    bad += ["порядок разделов ≠ Ж-259 (числа и вклад механизма → проверки → следующий шаг)"] * (min(heads) < 0 or heads != sorted(heads))
    bad += ["первая таблица не раньше разложения"] * (not (txt.find("## 1.") < txt.find("| K1 |") < txt.find("## 2.")))
    bad += ["нет раскрытия красного гейта M.G4 (а) с порогом"] * ("порог не менялся" not in txt)
    return bad + words(txt, t)


if __name__ == "__main__":
    if "--write" in sys.argv:
        (E.REP / E.REPORT).write_text(current(), encoding="utf-8")
        print("записано", E.REPORT)
    elif "--check" in sys.argv:
        h0 = (E.REP / E.REPORT).read_bytes() if (E.REP / E.REPORT).exists() else b""
        b = check()
        b += ["--check изменил отчёт"] * ((E.REP / E.REPORT).exists() and (E.REP / E.REPORT).read_bytes() != h0)
        print("S-RCD doc --check: %s" % ("ok" if not b else "FAIL " + "; ".join(b)))
        sys.exit(1 if b else 0)
    else:
        sys.exit("укажите --write или --check")
