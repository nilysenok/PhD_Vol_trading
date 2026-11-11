"""S-Z-EVAL: отчёт `S_Z_REPORT.md` — собирается ТОЛЬКО из таблиц `S_Z_*.csv` и ключей CHECKPOINT (ни одно число не вписано вручную). `--write` пишет, `--check` пересобирает в памяти и сравнивает (только чтение; sha256 отчёта до и после).
Порядок (Ж-258: «начни с таблицы и вывода о механизме»): первая строка — пометка; первая таблица (K1, K2, K3, B_mix, X1, W5-G, S-RCK, S-RL1, S-RCKZ, S-RZ, S-RCKZs: QLIKE, даты, n, Δ к S-RCK, W5-G, X1, K2, K3); вывод о механизме и нулевое распределение; защитные условия; годы; срезы; статистика; параметры; Z-DEC; порядок и оговорки.
Формулировки «подтверждено» и «лучше» — только из столбца verdict (разд. 7.5) и текста Z-DEC; слов «эквивалентно» в отчёте нет.
Запуск: `python3 -W ignore -m models_v2.s18.z_doc [--write | --check]`."""
from __future__ import annotations

import json
import re
import sys

import pandas as pd

from models_v2.s18 import z_eval as E
from models_v2.s18.z_doc_fmt import cap, md, tm
from models_v2.s18.z_doc_s1 import dec_lines, first_table, mech_lines, mv, null_section
from models_v2.s18.z_doc_s2 import guard_lines, params_lines, slices_lines, stats_lines, years_lines

FIRST = E.MARK


def tabs(rep=E.REP) -> dict:
    return {k: pd.read_csv(rep / n, comment="#") for k, n in zip(E.KEYS, E.OUTS)}


def order_lines(cp: dict) -> list:
    k, r = cp[E.KEY], cp["S.Z.registered"]
    Z = E.ROOT / "data/regen/harj_struct/Z"
    ts = ", ".join("%s %s" % (n, tm((Z / n).stat().st_mtime)) for n in ("zmaps.npz", "rckzs.parquet", "rckz.parquet", "rz.parquet"))
    return ["- Регистрация S-Z: %s; файлы: %s; нулевая диагностика `S_Z_NULL.csv`: %s; ключ `S.Z.eval`: %s (до таблиц с p)." % (r["registered_at"], ts, tm(E.ZN.OUT.stat().st_mtime), k["written_at"]),
            "- Реальные z̃ вне выборки только на днях прогноза; z пула внутривыборочны для энкодера года (З-5, оговорка спецификации): смещение не оценивалось, показана std z̃.",
            "- θ_Z разных лет несравнимы (координаты энкодеров не согласованы между годами); `thz_*` — среднее по пяти головам.",
            "- «η_Z» отдельно не сохранён: в разд. 3 показан ln(C_Z/C_S-RCK) (включает разницу θ_CK двух рядов), корреляция — ex post.",
            "- p SPA строк S-RCKZs, K2, K3 — SPA с одним кандидатом против своей базы (в спецификации заданы только (а), (б), (в)).",
            "- Отсутствие значимой разницы не доказывает отсутствия эффекта; выбор по минимуму QLIKE не производился. Все числа — разработка на раскрытой истории: исследовательский результат, не OOS."]


def layer_table(tb: pd.DataFrame) -> str:
    t = tb[tb["role"] == "layer X1"]
    r = pd.DataFrame({"слой": t["series"], "QLIKE": t["qlike"], "первая t⁺": t["first_tplus"], "последняя t⁺": t["last_tplus"], "n": t["n"], "Δ к X1, %": t["d_X1_pct"], "90 % снизу": t["lo90_X1"], "90 % сверху": t["hi90_X1"]})
    return md(r, {"слой": "%s", "QLIKE": "%.6f", "первая t⁺": "%s", "последняя t⁺": "%s", "n": "%d", "Δ к X1, %": "%+.4f", "90 % снизу": "%+.4f", "90 % сверху": "%+.4f"})


def build(t: dict, cp: dict) -> str:
    tb, d = t["table"], t["dec"].set_index("id")
    cp_ = cap(tb["first_tplus"].iloc[0], tb["last_tplus"].iloc[0], tb["n"].iloc[0])
    lo, hi = E.ZN.abs_bounds(E.ZN.read()[0])
    nr = lambda k: E.ZN.rule_abs(mv(t["mech"], "noise_real", k)["value"], lo, hi)
    hz = lambda i: "выполнено численно" if d.loc[i, "passed"] else "не выполнено"
    rule = "H-Z1 %s; H-Z2 %s (статистика — по вердиктам выше, разд. 7.5); по нулевому распределению: Δ(S-RCKZ к S-RCK) на D_M — %s, на активных датах — %s; Δ(S-RCKZ к S-RCKZs) на D_M — %s, на активных датах — %s" % (
        hz("H-Z1"), hz("H-Z2"), nr("Z_vs_RCK|D_M"), nr("Z_vs_RCK|active"), nr("Z_vs_Zs|D_M"), nr("Z_vs_Zs|active"))
    L =["# " + FIRST, "", "Выведено из таблиц `S_Z_*.csv` (`z_doc`); D_M: %s. Обычная QLIKE (среднее дневных потерь на всех датах, без обрезки); Δ в %% QLIKE базы, минус — лучше; 90 %% интервал — стационарный бутстрап. N программы %d / %d." % (cp_, cp[E.KEY]["n_holm"], cp[E.KEY]["n_holm_wide"]), "",
         "## 1. Первая таблица: QLIKE и Δ (%s)" % cp_, first_table(tb), "",
         "Слои доверия X1_R над C рядов (Δ к X1; %s):" % cp_, layer_table(tb), "",
         "## 2. Вывод о механизме"] + mech_lines(t) + ["- **Итог механизма** (выведен кодом): %s." % rule, ""] + [
         "## 3. Нулевое распределение процедуры"] + null_section(t) + ["## 4. Защитные условия (два набора)"] + guard_lines(t, tb) + ["## 5. Годы"] + years_lines(t) + ["## 6. Срезы"] + slices_lines(t, tb) + [
         "## 7. Статистика"] + stats_lines(t, cp) + ["## 8. Параметры и диагностика"] + params_lines(t) + ["## 9. Решение Z-DEC"] + dec_lines(t) + ["## 10. Порядок и оговорки"] + order_lines(cp)
    return "\n".join(L) + "\n"


def current(rep=E.REP, ckpt=E.CKPT) -> str:
    return build(tabs(rep), json.loads(ckpt.read_text(encoding="utf-8")))


def words(txt: str, t: dict) -> list:
    """«подтверждено» и «лучше» — только внутри формулировок разд. 7.5 (столбец verdict) и текста Z-DEC; «эквивалент» — нигде."""
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
    bad += ["первая таблица: порядок рядов ≠ заданию"] * (txt.find("| K1 |") < 0 or not (txt.find("| K1 |") < txt.find("| K2 |") < txt.find("| K3 |") < txt.find("| B_mix |") < txt.find("| X1 |") < txt.find("| W5-G |") < txt.find("| S-RCK |") < txt.find("| S-RL1 |") < txt.find("| S-RCKZ |") < txt.find("| S-RZ |") < txt.find("| S-RCKZs |")))
    bad += ["первая таблица не раньше вывода о механизме"] * (not (txt.find("## 1.") < txt.find("## 2.")))
    return bad + words(txt, t)


if __name__ == "__main__":
    if "--write" in sys.argv:
        (E.REP / E.REPORT).write_text(current(), encoding="utf-8")
        print("записано", E.REPORT)
    elif "--check" in sys.argv:
        h0 = (E.REP / E.REPORT).read_bytes() if (E.REP / E.REPORT).exists() else b""
        b = check()
        b += ["--check изменил отчёт"] * ((E.REP / E.REPORT).exists() and (E.REP / E.REPORT).read_bytes() != h0)
        print("S-Z doc --check: %s" % ("ok" if not b else "FAIL " + "; ".join(b)))
        sys.exit(1 if b else 0)
    else:
        sys.exit("укажите --write или --check")
