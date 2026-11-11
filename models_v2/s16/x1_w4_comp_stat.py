"""W4-B2: статистика ступени 1 и решение по 2.5 (`W4_SPEC.md`). Всё — из рядов `data/regen/harj_x1/W4/*.parquet`; решение `decide(csv)` читает ТОЛЬКО `W4_STAGE1.csv` (пороги — константы ниже, объявлены в 2.5 до счёта).
API: `table(ser, perms, state) -> DataFrame` (CSV), `csv_text(df)`, `decide(df) -> dict`, `render(df, dec) -> str` (отчёт целиком из CSV), `write_stage1(ckpt, dec, csv_sha)`."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from models_v2.s16 import x1_w3_core as W3

HALF_TOL = 0.005
SETS = ("S1", "S2")
KB = {"S1": "Kb1", "S2": "Kb2"}
COLS = ["kind", "series", "key", "n", "qlike", "d_M_pct", "d_Ka_pct", "d_Kb_pct", "c_y_pct", "p_perm", "value"]
HALVES = {"all": lambda y: y >= 2018, "h1": lambda y: y <= 2021, "h2": lambda y: y >= 2022}


def _row(kind, series, key, **kw) -> dict:
    return {**{c: np.nan for c in COLS}, "kind": kind, "series": series, "key": key, **kw}


def losses(ser: dict) -> dict:
    return {k: W3.qlike(v["rv"].to_numpy(float), v["F"].to_numpy(float)) for k, v in ser.items()}


def table(ser: dict, perms: dict, state: np.ndarray) -> pd.DataFrame:
    """ser: M, S1, S2, Ka, Kb1, Kb2 -> кадры окна (d, t_plus, rv, M, F, …; M — как ряд F = M); perms[set] = {j: ряд QLIKE_j на окне M} 19 значений."""
    L, y = losses(ser), ser["S1"]["t_plus"].dt.year.to_numpy()
    rows = []
    for k, l in L.items():
        for h, f in HALVES.items():
            m = f(y)
            q, qm = float(l[m].mean()), float(L["M"][m].mean())
            qk = float(L["Ka"][m].mean()) if k not in ("M", "Ka") else np.nan
            qb = float(L[KB[k]][m].mean()) if k in KB else np.nan
            p = np.nan
            if k in SETS and h == "all":
                p = (1 + sum(v <= q for v in perms[k].values())) / (1 + len(perms[k]))
            rows.append(_row("window", k, h, n=int(m.sum()), qlike=q, d_M_pct=(q / qm - 1) * 100 if k != "M" else np.nan,
                             d_Ka_pct=(q / qk - 1) * 100 if qk == qk else np.nan, d_Kb_pct=(q / qb - 1) * 100 if qb == qb else np.nan, p_perm=p))
    for k in SETS:
        qm_all, n = float(L["M"].mean()), len(y)
        for yr in sorted(set(y)):
            m = y == yr
            rows.append(_row("year", k, str(yr), n=int(m.sum()), qlike=float(L[k][m].mean()), d_M_pct=(float(L[k][m].mean()) / float(L["M"][m].mean()) - 1) * 100,
                             c_y_pct=float((L[k][m] - L["M"][m]).sum() / (n * qm_all) * 100)))
        for sk, name in ((0, "N"), (1, "E")):
            m = state == sk
            rows.append(_row("state", k, name, n=int(m.sum()), qlike=float(L[k][m].mean()), d_M_pct=(float(L[k][m].mean()) / float(L["M"][m].mean()) - 1) * 100))
        d = ser[k][ser[k]["refit_date"].notna()].drop_duplicates("refit_date")
        rows += [_row("refit", k, "months_fitted", n=len(d), value=float(len(d))), _row("refit", k, "share_months_rho0", n=len(d), value=float((d["rho"] == 0).mean()))]
        rows += [_row("refit", k, "months_rho_%g" % r, n=len(d), value=float((d["rho"] == r).sum())) for r in (0.0, 0.25, 0.5, 0.75, 1.0)]
        rows += [_row("refit", k, "trees_" + nm, n=len(d), value=float(getattr(np, nm)(d["n_trees"]))) for nm in ("min", "median", "max")]
        rows += [_row("perm", k, "j%02d" % j, n=252 + 100 * (j - 1), qlike=float(v)) for j, v in sorted(perms[k].items())]
    return pd.DataFrame(rows)[COLS]


def csv_text(df: pd.DataFrame) -> str:
    return df.to_csv(index=False, lineterminator="\n")


def decide(df: pd.DataFrame) -> dict:
    """2.5: набор проходит, если на окне M: Q(M⁺) < Q(M); Q(M⁺)/Q(M) − 1 ≤ +0,5 % в обеих половинах; Q(M⁺) < Q(K_a). Выбор для ступени 2 — раздел 3 (оба прошли — S2)."""
    w = df[df["kind"] == "window"].set_index(["series", "key"])["qlike"]
    out = {}
    for k in SETS:
        c1 = bool(w[k, "all"] < w["M", "all"])
        c2 = {h: float(w[k, h] / w["M", h] - 1) for h in ("h1", "h2")}
        c3 = bool(w[k, "all"] < w["Ka", "all"])
        out[k] = dict(c1_better_than_M=c1, c2_half_excess=c2, c2_ok=all(v <= HALF_TOL for v in c2.values()), c3_better_than_Ka=c3, passed=bool(c1 and c3 and all(v <= HALF_TOL for v in c2.values())))
    passed = [k for k in SETS if out[k]["passed"]]
    return dict(sets=out, passed=passed, chosen=("S2" if "S2" in passed else (passed[0] if passed else None)),
                stage2="считается" if passed else "не считается, условие ложно", half_tol=HALF_TOL)


def render(df: pd.DataFrame, dec: dict) -> str:
    w = df[df["kind"] == "window"].set_index(["series", "key"])
    f = lambda v: "—" if v != v else "%.4f" % v  # noqa: E731
    p = lambda v: "—" if v != v else "%+.2f %%" % v  # noqa: E731
    L = ["# W4 ступень 1: компонент M⁺ над M (Ж-250)", "", "Статус: DEVELOPMENT / HISTORICAL WALK-FORWARD, не независимый OOS. Окно M: t⁺ 2018-01-04…2026-11-10; QLIKE — среднее дневных потерь. Числа — из `W4_STAGE1.csv`.", "",
         "## QLIKE по окнам", "", "| ряд | окно | n | QLIKE | Δ к M | Δ к K_a | Δ к K_b | p (перестановки) |", "|---|---|---|---|---|---|---|---|"]
    for (s, h), r in w.iterrows():
        L.append("| %s | %s | %d | %s | %s | %s | %s | %s |" % (s, h, r["n"], f(r["qlike"]), p(r["d_M_pct"]), p(r["d_Ka_pct"]), p(r["d_Kb_pct"]), f(r["p_perm"])))
    L += ["", "## Вклады лет (Σ c_y = Δ к M, в % QLIKE(M))", "", "| набор | год t⁺ | n | QLIKE | Δ к M | c_y |", "|---|---|---|---|---|---|"]
    L += ["| %s | %s | %d | %s | %s | %s |" % (r.series, r.key, r.n, f(r.qlike), p(r.d_M_pct), p(r.c_y_pct)) for r in df[df["kind"] == "year"].itertuples()]
    L += ["", "## Состояния X1 (N/E) и переоценки", "", "| вид | набор | ключ | n | значение |", "|---|---|---|---|---|"]
    L += ["| %s | %s | %s | %d | %s |" % (r.kind, r.series, r.key, r.n, f(r.qlike if r.kind == "state" else r.value) + ("" if r.kind != "state" else " (Δ к M " + p(r.d_M_pct) + ")")) for r in df[df["kind"].isin(["state", "refit"])].itertuples()]
    L += ["", "## Решение по 2.5", ""]
    for k, v in dec["sets"].items():
        L.append("- %s: Q < Q(M) на окне — %s; превышение к M в половинах: 2018–2021 %s, 2022–2026 %s (порог +0,50 %%) — %s; Q < Q(K_a) — %s; **%s**." % (
            k, v["c1_better_than_M"], p(v["c2_half_excess"]["h1"] * 100), p(v["c2_half_excess"]["h2"] * 100), v["c2_ok"], v["c3_better_than_Ka"], "прошёл" if v["passed"] else "не прошёл"))
    L += ["", "Прошли: %s. Набор для ступени 2: %s. Ступень 2 (W4-X, W4-Xa, W4-Xb): %s." % (", ".join(dec["passed"]) or "ни один", dec["chosen"] or "—", dec["stage2"]),
          "", "Отступление W4-B2: обучатель `fit_gbm` не строит дерево, если обучающих строк (закрытые пары без последних 252) меньше 100: переоценки с d₀ 2018-04-02 и 2018-05-02 дали g = 0 (M⁺ = M).",
          "Результат — историческая разработка с PIT-обучением; независимым подтверждением не является. Значимость Холма на ступени 1 не считается (N программы 770 — для ступени 2)."]
    return "\n".join(L) + "\n"


def write_stage1(ckpt: Path, dec: dict, csv_sha: str, extra: dict) -> None:
    """Слияние ключа `W4.stage1`; прочие ключи — побайтно (формат файла: indent 1, sorted keys). Если файл не в этом формате — стоп."""
    raw = ckpt.read_text(encoding="utf-8")
    cp = json.loads(raw)
    if json.dumps(cp, indent=1, sort_keys=True, ensure_ascii=False) != raw:
        raise SystemExit("CHECKPOINT: формат файла не совпадает с записью слиянием")
    before = {k: json.dumps(v, sort_keys=True) for k, v in cp.items() if k != "W4.stage1"}
    cp["W4.stage1"] = dict(decision=dec, csv_sha256=csv_sha, **extra)
    if {k: json.dumps(v, sort_keys=True) for k, v in cp.items() if k != "W4.stage1"} != before:
        raise SystemExit("CHECKPOINT: прочие ключи изменились")
    ckpt.write_text(json.dumps(cp, indent=1, sort_keys=True, ensure_ascii=False), encoding="utf-8")
