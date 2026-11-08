"""V6-1a: строки для EVIDENCE G10–G15 и первая строка отчёта (числа из пересчёта в памяти, `stage 1a --check`)."""
from __future__ import annotations

from models_v2.s13 import v6_1_feat as FE


def first_line(r: dict) -> str:
    t, cp = r["tests"], r["checkpoint"]
    d = t[(t["section"] == "delta_dm") & (t["window"] == "all")].set_index("base")
    d2 = t[(t["section"] == "delta_dm") & (t["window"] == "2017_2021")].set_index("base")
    iv = lambda x, b: "%+.2f%% [%+.2f; %+.2f]" % (x.loc[b, "delta_pct"], x.loc[b, "lo90"], x.loc[b, "hi90"])
    go = " ".join("%d:%s" % (c["cond"], "да" if c["passed"] else "нет") for c in cp["go"])
    return "A0: QLIKE вся история %.4f, 2017–2021 %.4f; Δ к K1 %s / %s, к K3 %s / %s (вся / 2017–2021, %% уровня K1, 90 %%); go %s; go_all %s" % (
        cp["qlike_all"], cp["qlike_2017_2021"], iv(d, "K1"), iv(d2, "K1"), iv(d, "K3"), iv(d2, "K3"), go, "да" if cp["go_all"] else "нет")


def evidence(r: dict) -> list:
    hc, cp, s, w = r["hc"], r["checkpoint"], r["sens"], r["w"]
    ni = cp["new_information_vs_K1"]
    out = [first_line(r),
           "G10: дат прогрева %d (%s … %s); пересчёт = K1 на %d датах до 2017-03-31: max|diff| = %.1e; на всех %d общих датах K1: %.1e; SHA K1 %s" % (hc["n_warm"], hc["first_warm"], hc["last_warm"], hc["n_early"], hc["max_diff_early"], hc["n_all"], hc["max_diff_all"], r["k1_sha"][:12]),
           "G11: файлы данных признаков: %s; Σr² = RV: max отн. разн. %.1e; S: %d × %d, NaN %d" % ([p.name for p in FE.allowed_files()], r["ident"], len(r["S"]), r["S"].shape[1] - 1, int(r["S"].drop(columns="d").isna().sum().sum())),
           "G13: переоценок %d (первая %s, последняя %s); закрытых пар на первой %d, на последней %d; λ по сетке %s; F>0 на %d датах" % (len(r["run"]["fits"]), r["run"]["fits"][0]["d0"].date(), r["run"]["fits"][-1]["d0"].date(), r["run"]["fits"][0]["n_train"], r["run"]["fits"][-1]["n_train"], cp["lam_counts"], int((r["a0"]["F"] > 0).sum())),
           "G15: дат A0 %d; конфигураций %d из %d; новая информация к K1 (справочно): %s" % (cp["n_dates"], cp["used"], cp["budget"], {k: (round(v, 4) if isinstance(v, float) else v) for k, v in ni.items()})]
    out += ["G15 чувствительность λ: %s: QLIKE вся %.4f, 2017–2021 %.4f, Δ к K1 вся %+.2f%%" % (x.variant, x.qlike_all, x.qlike_2017_2021, x.d_pct_K1_all) for x in s.itertuples()]
    out.append("G15 условия go: " + "; ".join("%d %s: %.5f vs %.5f -> %s" % (c["cond"], c["name"], c["value"], c["threshold"], "да" if c["passed"] else "нет") for c in cp["go"]))
    return out
