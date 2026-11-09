"""V6 лист Z2, G64: документ `HARJ_SBER_V1_V6_SRAVNENIE.md` собирается ГЕНЕРАТОРОМ из таблиц `cmp/` (разделы 1–3: главная таблица, годы, статистика). Статические тексты чисел с запятой не содержат (тест);
все числа — из таблиц через `v6_cmp_fmt`."""
from __future__ import annotations

from models_v2.s13 import v6_cmp_fmt as F, v6_cmp_reg as R

WIN3 = ("all", "2017_2021", "2022_2025")


def first_line(t: dict) -> str:
    b = t["BEST"]
    b = b[(b["set"].str.startswith("основное (")) & (b["ref_kind"] == "второй")].set_index("window")
    parts = ["%s: %s (QLIKE %s)" % (F.win_name(w), b.loc[w, "best"], F.n4(b.loc[w, "best_qlike"])) for w in WIN3]
    sz = b.loc["all", "mcs_size"]
    te = t["TESTS"]
    hy = te[(te["cls"] == R.PIT) & te["base"].isin(["K2", "K3"])]
    sig = int((hy["p_holm_N722"] < 0.05).sum())
    return ("Результат: точный лучший результат SBER (h = 1) по точке — %s; MCS 90 %% содержит %d из %d рядов основного множества, то есть лучший по точке статистически не отделим от остальных; "
            "значимых после поправки Холма (N = 722) сравнений PIT-гибридов с K2 или K3: %d из %d. Все цифры — потолок, development walk-forward, не независимый OOS." % ("; ".join(parts), int(sz), int(b.loc["all", "n_series"]), sig, len(hy)))


def main_table(t: dict) -> str:
    q = F.ix(t["WINDOWS"], ["series", "window"])
    reg = t["SERIES"]
    out = []
    for w in WIN3:
        rows = []
        for sp in R.SPEC:
            g = reg[reg["row"] == sp["row"]]
            cell = lambda role: "; ".join(F.cell_series(r["series"], q[(r["series"], w)]["qlike"], " (потолок)" if r["cls"] == R.CEIL else "") for _, r in g[g["role"] == role].iterrows()) or "—"
            rows.append([sp["row"]] + [F.n4(q[(k, w)]["qlike"]) for k in ("K1", "K2", "K3")] + [cell("а"), cell("б") if (g["role"] == "б").any() else "— (нет PIT-гибридов в пуле)", cell("потолок_мин")])
        out.append("### %s\n\n%s" % (F.win_name(w), F.md_table(["Строка", "K1", "K2", "K3", "гибрид (а): финалист программы", "гибрид (б): лучший PIT после просмотра, потолок", "потолок-минимум пула (в выбор не входит)"], rows)))
    return "\n\n".join(out)


def years_table(t: dict) -> str:
    y = F.ix(t["YEARS"], ["series", "year"])
    reg = t["SERIES"]
    labs = list(dict.fromkeys(list(reg["series"])))
    yrs = sorted(t["YEARS"]["year"].unique())
    n26 = int(y[(labs[0], yrs[-1])]["n"])
    rows = [[s + (" (потолок)" if R.klass(s)[0] == R.CEIL else "")] + [F.n4(y[(s, yy)]["qlike"]) for yy in yrs] for s in labs]
    return "QLIKE по году цели t⁺ (на общих датах). %d: неполный год, %d дат (до 26.02).\n\n%s" % (yrs[-1], n26, F.md_table(["Ряд"] + [str(yy) for yy in yrs], rows))


def stats_table(t: dict) -> str:
    te, wi = t["TESTS"], F.ix(t["WINDOWS"], ["series", "window"])
    out = []
    for w in WIN3:
        x = te[te["window"] == w]
        ti = F.ix(x, ["row", "series", "base"])
        rows = []
        for (row, s), g in x.groupby(["row", "series"], sort=False):
            r2, r3 = ti[(row, s, "K2")], ti[(row, s, "K3")]
            yrs = "%d из %d / %d из %d" % (wi[(s, w)]["yrs_neg_K2"], wi[(s, w)]["yrs_n"], wi[(s, w)]["yrs_neg_K3"], wi[(s, w)]["yrs_n"]) if w != "all" else "—"
            rows.append([row, "%s (%s; %s)" % (s, g.iloc[0]["role"], g.iloc[0]["cls"]), F.dci(ti[(row, s, "K1")]), F.dci(r2), F.dci(r3), F.dci(ti[(row, s, "K8")]), yrs, F.holm3(r2), F.holm3(r3)])
        out.append("### %s\n\n%s" % (F.win_name(w), F.md_table(["Строка", "Ряд (роль; класс)", "Δ к K1, %", "Δ к K2, %", "Δ к K3, %", "Δ к K8, %", "лет Δ < 0 к K2 / K3", "p Холма к K2 (N 722 / 3122 / прогр.)", "p Холма к K3 (N 722 / 3122 / прогр.)"], rows)))
    return "\n\n".join(out)


def cov_table(t: dict) -> str:
    c = t["COV"]
    return F.md_table(["Строка", "Реестр", "ключей", "в N = 722", "в N_wide = 3122", "примечание"], [[r["row"], r["registry"], int(r["unique_keys"]), "да" if r["in_N722"] else "нет", "да" if r["in_N3122"] else "нет", r["note"]] for _, r in c.iterrows()])
