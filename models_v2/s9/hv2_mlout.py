"""Ж-214, лист 4 (V9): сверка баз с хранимыми рядами, контроли-ссылки, строка таблицы, проверки на боевом пути (производные QLIKE на реальных парах, λ = 0 ≡ база и λ = 1 ≡ полная g побитно,
положительность), таблицы `xgboost_structural/`, `lightgbm_structural/` (раздельно), абляции, корреляция ошибок XGB и LGBM (диагностика, не комбинация)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from models_v2.s8 import hs_eval, hs_paths as HP, hs_store
from models_v2.s9 import hv2_decstage as DS, hv2_eval as E, hv2_nfit as NF, hv2_paths as P

CTRL = ("A0", "XJ", "LF", "Bstar_H2", "T3", "Ca_x", "Cc_x", "Ca_l", "Cc_l", "A1_roll_126", "parent")
SUB = {"XGB": "xgboost_structural", "LGBM": "lightgbm_structural"}
OWN = {"XGB": "XJ", "LGBM": "LF"}


def rel(a, b) -> float:
    return float(np.max(np.abs(np.asarray(a, float) / np.asarray(b, float) - 1)))


def base_check(ctx: dict, dw, B: dict) -> dict:
    """Базы, пересчитанные на расширенных строках, на сетке = хранимые ряды A0 (components.HAR), H2 (H2_series), T3 (DEC_series); иначе стоп."""
    el = dw.el
    r = {"A0": rel(B["A0"][el], ctx["refs"]["A0"]), "H2": rel(B["H2"][el], pd.read_parquet(P.OUT / "H2_series.parquet")["H2"]),
         "T3": rel(B["T3"]["s2"][el], pd.read_parquet(P.OUT / "DEC_series.parquet")["T3"])}
    if max(r.values()) > 1e-12:
        raise SystemExit("стоп: база на расширенных строках не совпала с хранимой на сетке: %s" % r)
    return r


def add_refs(ctx: dict, dw, B: dict) -> None:
    """Контроли-ссылки (не trials): B* = H2, T3, Ж-212 Ca/Cc обеих библиотек, A1_roll_126 (Ж-210)."""
    el, refs = dw.el, ctx["refs"]
    refs["Bstar_H2"], refs["T3"] = B["H2"][el], B["T3"]["s2"][el]
    for lib, s in (("xgb", "x"), ("lgbm", "l")):
        for c in ("Ca", "Cc"):
            v = hs_store.get("C", "C_%s_%s" % (lib, c))
            if v is not None:
                refs["%s_%s" % (c, s)] = v
    a = pd.read_parquet(HP.SERIES_A)
    if (a["d"].to_numpy() == ctx["df"]["d"].to_numpy()).all():
        refs["A1_roll_126"] = a["A1_roll_126"].to_numpy(float)


def lam_checks(dw, B: dict, sp: dict, name: str, ser: dict) -> dict:
    """λ = 0 ≡ база побитно и λ = 1 ≡ полная g (кандидаты, пересчитанные из СОХРАНЁННЫХ g), положительность σ̂²."""
    el = dw.el
    if sp["kind"] == "N2":
        ks = list(B["T3"]["comps"])
        S = NF.n2_candidates([B["T3"]["comps"][k][el] for k in ks], [ser["%s__g_%s" % (name, k)] for k in ks], B["T3"]["fb"][el] > 0, B["T3"]["s2"][el])
        base = B["T3"]["s2"][el]
    else:
        base = B[sp["base"]][el]
        S = NF.n1_candidates(base, ser[name + "__g"])
    return {"lam0_bitwise": bool(np.array_equal(S[:, 0], base)), "lam1_equals_full": bool(np.array_equal(S[:, -1], ser[name + "__full"])),
            "min_sigma2": float(ser[name].min()), "positive": bool((ser[name] > 0).all() and (S > 0).all())}


def row(ctx: dict, dw, B: dict, name: str, sp: dict, key: str, ser: dict, trunc: float, parent: str | None) -> dict:
    lam, fb = ser[name + "__lam"], ser[name + "__fb"] > 0
    ex = {"config": name, "kind": sp["kind"], "lib": sp["lib"], "base": sp["base"], "groups": "".join(sp["groups"]), "mu": sp["mu"], "key": key, "trunc_max_diff": trunc,
          "n_fallback": int(fb.sum()), "lam_mean": float(lam.mean()), "share_lam_pos": float((lam > 0).mean()), "parent": parent or "", **lam_checks(dw, B, sp, name, ser)}
    return E.row(ctx, name, ser[name], ex, CTRL)


def per_library(rows: list) -> dict:
    """Информативно: каждая библиотека против СВОЕГО самостоятельного бустинга (XGB — XJ, LGBM — LF); V13 решает лучшая строка против A0 и S* = XJ."""
    out = {}
    for r in rows:
        k = OWN[r["lib"]]
        out[r["config"]] = {"own_control": k, "d_crit": r["d_crit_" + k], "years_better": r["years_better_" + k], "d_crit_A0": r["d_crit_A0"], "d_crit_XJ": r["d_crit_XJ"]}
    return out


def deriv_real(ctx: dict, dw, B: dict) -> pd.DataFrame:
    """Производные QLIKE численно на реальных парах (y = RV_{t+1}, offset = ln A0_t) со случайным сдвигом m, μ ∈ {0; 0,05}."""
    t = np.flatnonzero(np.isfinite(B["A0"][:-1]) & (dw.p.seg[:-1] == dw.p.seg[1:]))
    lb, y = np.log(B["A0"][t]), dw.p.rv[t + 1]
    m = lb + np.random.default_rng(42).normal(0, 0.5, len(t))
    return pd.DataFrame([{"mu": mu, "n_pairs": len(t), **NF.deriv_check(y, m, lb, mu)} for mu in (0.0, 0.05)])


def err_corr(ctx: dict, rows: list, ser: dict) -> pd.DataFrame:
    """Корреляция ошибок (RV − σ̂²) и разница XGB и LGBM одной конструкции — диагностика условной информации LGBM к XGB, не комбинация."""
    df, rv, out = ctx["df"], ctx["df"]["rv"].to_numpy(float), []
    pairs = [("XJ", ctx["refs"]["XJ"], "LF", ctx["refs"]["LF"])]
    names = {r["config"] for r in rows}
    for a, b in (("N1x_A0", "N1l_A0"), ("N1x_B", "N1l_B"), ("N2x", "N2l")):
        if a in names and b in names:
            pairs.append((a, ser[a][a], b, ser[b][b]))
    for na, sa, nb, sb in pairs:
        c = hs_eval.row(df, nb, sb, sa)
        out.append({"xgb": na, "lgbm": nb, "corr_err": float(np.corrcoef(rv - sa, rv - sb)[0, 1]), "corr_ln_err": float(np.corrcoef(np.log(rv / sa), np.log(rv / sb))[0, 1]),
                    "d_qlike_lgbm_minus_xgb": c["d_qlike"], "d_crit_lgbm_minus_xgb": c["d_crit"], "p_dm": c["p_dm"], "years_lgbm_better": c["years_better"]})
    return pd.DataFrame(out)


def write_all(ctx: dict, dw, B: dict, rows: list, ser: dict, sps: dict, chk: dict, v13: dict) -> None:
    refs, df = ctx["refs"], ctx["df"]
    ctl = [E.row(ctx, k, refs[k], {"kind": "контроль"}, CTRL) for k in ("A0", "XJ", "LF", "Ca_x", "Cc_x", "Ca_l", "Cc_l", "A1_roll_126", "Bstar_H2", "T3") if k in refs]
    note = "V9, структурная коррекция; Δ — строка минус контроль (минус — строка лучше); parent — родительская конфигурация (абляции, Cb); потолок: выбор по crit_wf на всей истории; V13: %s" % v13["best"]
    var = [E.row(ctx, r["config"] + "|lam1", ser[r["config"]][r["config"] + "__full"], {"config": r["config"] + "|lam1", "variant": "полная g (λ = 1), диагностика, не trial", "lib": r["lib"]}, CTRL) for r in rows]
    for lib in ("XGB", "LGBM"):
        P.write_csv(pd.DataFrame(ctl + [r for r in rows + var if r["lib"] == lib]), "STRUCT_%s.csv" % lib, note, SUB[lib])
        sel = {"A0": refs["A0"], "XJ": refs["XJ"], **{r["config"]: ser[r["config"]][r["config"]] for r in rows if r["lib"] == lib}}
        P.write_csv(DS.by_year(df, sel), "STRUCT_BY_YEAR_%s.csv" % lib, "QLIKE по годам цели; Δ к A0 и XJ", SUB[lib])
        ab = [r for r in rows if r["lib"] == lib and r["parent"]]
        if ab:
            cols = ["config", "parent", "groups", "mu", "qlike_all", "crit_wf", "d_crit_parent", "d_qlike_parent", "d_qlike_no_240222_parent", "years_better_parent", "p_dm_parent"]
            P.write_csv(pd.DataFrame(ab)[cols], "ABLATION_GROUPS_%s.csv" % lib, "абляции групп (без группы) и Cb-штраф против родителя; плюс Δ — группа/штраф УХУДШАЕТ, если родитель лучше", SUB[lib])
    P.write_csv(pd.DataFrame([{"check": "base_vs_stored_" + k, "max_rel": v} for k, v in chk.items()] + [{"check": "deriv_real", **r} for r in deriv_real(ctx, dw, B).to_dict("records")]
                             + [{"check": "lam_" + r["config"], "lam0_bitwise": r["lam0_bitwise"], "lam1_equals_full": r["lam1_equals_full"], "positive": r["positive"], "min_sigma2": r["min_sigma2"]} for r in rows]),
                "ML_CHECKS.csv", "боевой путь: базы на расширенных строках = хранимые на сетке; производные QLIKE на реальных парах; λ = 0 ≡ база и λ = 1 ≡ полная g побитно; положительность", "xgboost_structural")
    P.write_csv(err_corr(ctx, rows, ser), "ML_ERRCORR.csv", "XGB против LGBM: корреляция ошибок и разница (диагностика; LGBM не включается компонентом без условной информации к XGB)", "lightgbm_structural")
