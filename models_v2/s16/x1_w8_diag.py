"""W8-A: перенос `w8/build.py` БЕЗ изменения логики — строит `docs/otchety/harj_x1/W8_DECOMP.csv` (объявление — `W8_SPEC.md` разд. 1.0, 11.11 15:08). Новых прогнозов нет.
Блоки: A_cell (год/половина/окно × тип перерыва; базы X1, K2, K3), B_claim (утверждение Ж-253 п. 3), C_group / D_transfer / E_verdict (`x1_w8_diag_cde`), F_noise, G_null (`x1_w8_diag_null`).
Публичное API: `OUT`, `HEAD`, `render() -> (текст CSV, DataFrame, x)`, `check(out=OUT) -> list` замечаний (только чтение: пересчёт побайтно равен файлу; файл до и после не изменён).
Запуск: `python3 -W ignore -m models_v2.s16.x1_w8_diag [--check | --check-verdict | --write]`; без флага — сухой прогон (ничего не пишет).
`--check-verdict` — `x1_w8_diag_v`: П1–П5 заново из блоков D_transfer и C_group CSV; печатает `прошли: <признаки>; проверено N`."""
from __future__ import annotations

import hashlib
import io
import sys

import numpy as np
import pandas as pd

from models_v2.s16 import x1_w8_diag_cde as CDE
from models_v2.s16 import x1_w8_diag_core as K
from models_v2.s16 import x1_w8_diag_feat as FE

OUT = K.ROOT / "docs/otchety/harj_x1/W8_DECOMP.csv"
TYPES = ["weekend", "hol1", "long", "worksat", "other", "none"]
YEARS = K.YEARS
HEAD = ("# W8_DECOMP — разложение W5-G по готовым прогнозам (Ж-253 п. 4). Объявление — W8_SPEC.md разд. 1.0 (11.11.2026 15:08). Новых прогнозов нет.\n"
        "# после просмотра окна M (W1–W6); диагностика; исследовательский результат на раскрытой истории, не независимое подтверждение (Ж-250 п. 6)\n"
        "# состояние gap W5-G выбрано из 24 после анализа истории окна M; способ оценки β выбран после просмотра 24.02.2022 (Д-2 W5)\n"
        "# окно M: t⁺ 2018-01-04…2026-11-10, n = 2198; h1 = t⁺ 2018–2021, h2 = t⁺ 2022–2026; l = RV/F − ln(RV/F) − 1; u = ln(RV/X1)\n"
        "# A_cell: beta_app = mean ln(F_W5G/X1) клетки; beta_expost = ln mean e^u клетки (QLIKE-оптимальная постоянная над X1); delta_pct = (Q_W5G − Q_base)/Q_base·100 клетки;\n"
        "#   contrib_pp_M = Σ(l_W5G − l_base)/(n_M·Q_base,M)·100; contrib_pp_period = то же с n и Q периода. Тип — плановая длина L_d на d 18:40 (W5_CALENDAR_PIT).\n"
        "# B_claim: неокруглённые Δ W5-G к K2, K3, X1 по половинам и окну; срезы no_20220224, no_2022 — справочно\n"
        "# C_group / D_transfer / E_verdict: признаки W8D1…W8D10, оценщик β_G = clip(Δū + ½Δs², 0, 0,5) на половине A, группы < 15 дней gap → общий β; критерий П1–П5 — W8_SPEC.md 1.0\n"
        "# F_noise: SE блокового бутстрапа (блок 20, B 2000, seed 0) разностей W5G − X1, X1 − K2, X1 − K3, в % Q базы — калибровка порогов W8 разд. 2\n"
        "# G_null — ПОСЛЕ ПРОСМОТРА E_verdict: W8D8 при перестановке меток (500, seed 0) и 19 циклических сдвигах ряда IMOEX; delta_pct здесь = p = (1 + прошли)/(1 + число); year — c* групп по годам\n")


def cell(x, m, base, nM, QM):
    xs = x[m]
    n = int(m.sum())
    if n == 0:
        return dict(n=0, beta_app=np.nan, beta_expost=np.nan, Q_base=np.nan, Q_W5G=np.nan, delta_pct=np.nan, sum_diff=0.0)
    lb, lw = xs["l_" + base].to_numpy(), xs["l_W5G"].to_numpy()
    return dict(n=n, beta_app=float(xs["c_app"].mean()), beta_expost=float(np.log(np.mean(np.exp(xs["u"].to_numpy())))),
                Q_base=float(lb.mean()), Q_W5G=float(lw.mean()), delta_pct=float((lw.mean() - lb.mean()) / lb.mean() * 100), sum_diff=float((lw - lb).sum()))


def block_a(x):
    rows = []
    nM = len(x)
    pers = [(y, (x["yr"] == y).to_numpy()) for y in YEARS] + [("h1", (x["half"] == "h1").to_numpy()), ("h2", (x["half"] == "h2").to_numpy()),
                                                            ("M", np.ones(nM, bool))]
    checks = []
    for base in ("X1", "K2", "K3"):
        QM = float(x["l_" + base].mean())
        for p, pm in pers:
            Qp, n_p = float(x.loc[pm, "l_" + base].mean()), int(pm.sum())
            tot = 0.0
            for t in TYPES + ["all_gap", "all"]:
                tm = pm & ((x["type"] == t).to_numpy() if t in TYPES else ((x["gap"] == 1).to_numpy() if t == "all_gap" else np.ones(nM, bool)))
                c = cell(x, tm, base, nM, QM)
                sd = c.pop("sum_diff")
                r = dict(block="A_cell", base=base, period=p, type=t, **c, contrib_pp_M=sd / (nM * QM) * 100, contrib_pp_period=sd / (n_p * Qp) * 100)
                rows.append(r)
                if t in TYPES:
                    tot += r["contrib_pp_period"]
                if t == "all":
                    checks.append(abs(tot - r["delta_pct"]))
        a = pd.DataFrame([r for r in rows if r["base"] == base and r["period"] in YEARS and r["type"] == "all"])
        m = [r for r in rows if r["base"] == base and r["period"] == "M" and r["type"] == "all"][0]
        checks.append(abs(a["contrib_pp_M"].sum() - m["delta_pct"]))
    return rows, max(checks)


def block_b(x):
    rows = []
    variants = [("full", np.ones(len(x), bool)), ("no_20220224", (x["t_plus"] != pd.Timestamp("2022-02-24")).to_numpy()),
                ("no_2022", (x["yr"] != "2022").to_numpy())]
    for base in ("K2", "K3", "X1"):
        for vname, vm in variants:
            for p, pm in (("h1", (x["half"] == "h1").to_numpy()), ("h2", (x["half"] == "h2").to_numpy()), ("M", np.ones(len(x), bool))):
                m = vm & pm
                if vname != "full" and p == "h1":
                    continue
                lb, lw = x.loc[m, "l_" + base].to_numpy(), x.loc[m, "l_W5G"].to_numpy()
                rows.append(dict(block="B_claim", base=base, period=p, type=vname, n=int(m.sum()), Q_base=float(lb.mean()), Q_W5G=float(lw.mean()),
                                 delta_pct=float((lw.mean() - lb.mean()) / lb.mean() * 100)))
    return rows


def block_f(x):
    """F_noise: шум разреженной поправки над X1 в днях gap — блоковый бутстрап W5-G − X1 (блок 20, B 2000, seed 0)."""
    from models_v2.s5.d import stats_d as SD
    rows = []
    yr = x["yr"].to_numpy()
    wins = {"M": np.ones(len(x), bool), "h1": (x["half"] == "h1").to_numpy(), "h2": (x["half"] == "h2").to_numpy(), "2018": yr == "2018",
            "M_no2022": yr != "2022", "h2_no2022": (x["half"] == "h2").to_numpy() & (yr != "2022")}
    for a, b in (("W5G", "X1"), ("X1", "K2"), ("X1", "K3")):
        la, lb = x["l_" + a].to_numpy(), x["l_" + b].to_numpy()
        for wn, m in wins.items():
            dd = la[m] - lb[m]
            bm = SD.boot_means(dd, SD.block_idx(len(dd), B_=2000, block=20, seed=0))
            q = lb[m].mean()
            rows.append(dict(block="F_noise", base=b, period=wn, type="%s-%s" % (a, b), n=int(m.sum()), Q_base=q, delta_pct=100 * dd.mean() / q,
                             se_pct=100 * bm.std(ddof=1) / q, lo90_pct=100 * np.percentile(bm, 5) / q, hi90_pct=100 * np.percentile(bm, 95) / q))
    return rows


def block_g(x, feats):
    """G_null (ПОСЛЕ ПРОСМОТРА E_verdict): вероятность пройти П1–П5 случайно для W8D8 — перестановки (500, seed 0) и 19 циклических сдвигов."""
    from models_v2.s16 import x1_w8_diag_null as NU
    pp, n, sp, yrs = NU.run(x, feats)
    rows = [dict(block="G_null", feature="W8D8_imoex_rv", type="perm", n=n, passed=pp, delta_pct=(1 + pp) / (1 + n)),
            dict(block="G_null", feature="W8D8_imoex_rv", type="shift", n=19, passed=int(sum(sp)), delta_pct=(1 + sum(sp)) / 20,
                 betas=";".join(str(v) for v in sp))]
    for y, v in yrs.items():
        for G, (nn, cs) in v.items():
            rows.append(dict(block="G_null", feature="W8D8_imoex_rv", type="year", period=y, group=G, n=nn, beta_expost=cs))
    return rows


def render():
    x = K.load()
    feats, info = FE.features(x)
    ra, ident = block_a(x)
    if ident > 1e-12:
        raise SystemExit("тождество вкладов нарушено: %.3g" % ident)
    rb = block_b(x)
    rc, rd, re_ = CDE.blocks_cde(x, feats)
    rf = block_f(x)
    rg = block_g(x, feats)
    cols = ["block", "base", "period", "type", "feature", "half", "direction", "group", "n", "beta_app", "beta_expost", "beta_est", "Q_base", "Q_W5G", "delta_pct",
            "contrib_pp_M", "contrib_pp_period", "mean_u", "delta_W5G_pct_in_group", "contrib_pp_half", "share_help", "Q_X1", "Q_ctrl", "Q_feat",
            "d_feat_vs_ctrl_pct", "d_feat_vs_X1_pct", "d_ctrl_vs_X1_pct", "beta_pooled", "betas", "n_groups", "P1", "P2", "P3", "P4", "P5", "passed",
            "order_h1", "order_h2no22", "se_pct", "lo90_pct", "hi90_pct"]
    df = pd.DataFrame(ra + rb + rc + rd + re_ + rf + rg)[cols]
    n_type = x.loc[x["gap"] == 1, "type"].value_counts().to_dict()
    disagree = int(((x["delta_real"] > 0).astype(int) != x["gap"]).sum())
    lmis = int(((x["gap"] == 1) & (x["delta_real"] + 0 != x["L"]) & (x["delta_real"] > 0)).sum())
    meta = ("# сверка: L ≥ 1 ⇔ gap^PIT во всех 2198 парах; типы дней gap: %s; gap^PIT ≠ gap^real: %d пар; плановая L ≠ реализованной Δ_d (при Δ_d > 0): %d пар;"
            " тождество вкладов max %.2e; RV дня d нет: %d; IMOEX дня d нет: %d (из них дни gap: %d)\n"
            % (";".join("%s=%d" % kv for kv in sorted(n_type.items())), disagree, lmis, ident, info["rv_d_na"], info["imoex_na"], info["imoex_na_gap"]))
    buf = io.StringIO()
    df.to_csv(buf, index=False, lineterminator="\n", float_format="%.10g")
    return HEAD + meta + buf.getvalue(), df, x


def check(out=OUT) -> list:
    """Только чтение: пересчёт побайтно равен файлу; файл до и после не изменён."""
    before = out.read_bytes()
    text, _, _ = render()
    bad = [] if before == text.encode("utf-8") else ["W8_DECOMP.csv ≠ пересчёту"]
    return bad + ([] if out.read_bytes() == before else ["--check изменил файл"])


def main():
    if "--check-verdict" in sys.argv:
        from models_v2.s16 import x1_w8_diag_v as V
        sys.exit(V.main(OUT))
    if "--check" in sys.argv:
        b = check()
        print("W8_DECOMP --check:", "ok" if not b else "РАСХОЖДЕНИЕ " + "; ".join(b), hashlib.sha256(OUT.read_bytes()).hexdigest()[:16])
        sys.exit(1 if b else 0)
    text, df, _ = render()
    if "--write" in sys.argv:
        OUT.write_text(text, encoding="utf-8")
    print("записано" if "--write" in sys.argv else "сухой прогон", OUT, len(df), "строк; sha256", hashlib.sha256(text.encode("utf-8")).hexdigest()[:16])


if __name__ == "__main__":
    main()
