#!/usr/bin/env python3
"""etap2: таблицы 1-3, Б0/Б1, рычаги Г1/Г3, проверка V1 h=5. Вход: results/etap2/har_windows.parquet
(python3 scripts/analysis/etap2_okno.py). Выход: results/etap2/tables.md и results/etap2/summary.json.
Запуск: python3 scripts/analysis/etap2_analiz.py"""
import json
from etap2_common import *

CAND = ["E", "R1", "R2", "R3", "D2", "AveW"]
f4 = lambda v: f"{v:.4f}".replace(".", ",")
f2 = lambda v: f"{v:+.2f}".replace(".", ",")
fp = lambda p: "<0,0001" if p < 1e-4 else f"{p:.4f}".replace(".", ",")
OUTD = ROOT / "results/etap2"


def prep(w, h, col="p"):
    xt, xv = wide(w, h, "test", col), wide(w, h, "val", col)
    for x, k in ((xt, 0), (xv, 1)):                     # AveW = среднее прогнозов четырёх окон
        x["AveW"] = x[WIN].mean(axis=1)
    st = xt.index.get_level_values(0); sv = xv.index.get_level_values(0)
    dt = xt.index.get_level_values(1).year; dv = xv.index.get_level_values(1).year
    assert (dt == st).all() and (dv == sv - 1).all(), "val/test годы перепутаны"
    return xt, xv


def val_q(xv, cols):
    """валидационный QLIKE (пул) по шагам: DataFrame step x spec"""
    g = {}
    for s, d in xv.groupby(level=0):
        d = d[d.y > 0]; g[s] = {c: qmean(d.y, d[c]) for c in cols}
    return pd.DataFrame(g).T


def honest(xt, vq, cols):
    """прогноз и выбранная спецификация: минимум валидационного QLIKE шага (тест не используется)"""
    pick = vq[cols].idxmin(axis=1)
    st = xt.index.get_level_values(0)
    p = pd.Series([xt[pick[s]].iloc[i] for i, s in enumerate(st)], index=xt.index)
    return p, pick


def choose(yv, har, xg, lg):
    """(ML, w) по сетке w in [0;0,6] шаг 0,01 на валидации (walk_forward_full.py:258-279, ML выбирается по val)"""
    best = None
    for name, ml in (("XGB", xg), ("LGB", lg)):
        bq, bw = qmean(yv, ml), 0.0
        for w in np.arange(0, 0.61, 0.01):
            q = qmean(yv, w * har + (1 - w) * ml)
            if q < bq: bq, bw = q, w
        if bw == 0:
            a, b = 1 / (qmean(yv, har) + 1e-10), 1 / (qmean(yv, ml) + 1e-10); bw = a / (a + b)
        if best is None or bq < best[0]: best = (bq, name, bw)
    return best[1], best[2]


def proxy_val(h, step, har_v):
    """валидационный год шага t = тестовый год шага t-1: сохранённые OOS-прогнозы бустинга (бустинг не переобучается)"""
    s = saved(h); s = s[s.year == step - 1].set_index(["date", "ticker"])
    return s


def blends(xt, xv, h, hv2):
    """V1 из компонент (har=HAR-J из файла) и Г1/Г3 (har=честный v2); возвращает dict name->Series прогноза на test"""
    sv_all = saved(h).set_index(["date", "ticker"]); out = {k: [] for k in ("V1re", "G1", "G3")}
    idx = []
    for step in range(2017, 2027):
        t = xt[xt.index.get_level_values(0) == step]; v = xv[xv.index.get_level_values(0) == step]
        s = sv_all[sv_all.year == step]; tk = t.droplevel(0).index
        tj = s.reindex(tk); idx += list(t.index)
        if step == 2017:                                  # нет OOS-прогнозов бустинга для валидации 2016: как в сохранённом V1 (w=0,2, XGB)
            for k, har in (("V1re", tj["pred_HAR-J"]), ("G1", hv2[t.index].values), ("G3", hv2[t.index].values)):
                out[k] += list(0.2 * np.asarray(har) + 0.8 * tj["pred_XGBoost"].values)
            continue
        pv = proxy_val(h, step, None).reset_index(); pv = pv[pv.rv_actual > 0].set_index(["date", "ticker"])
        vv = v.droplevel(0).reindex(pv.index)
        for k in out:
            hv = pv["pred_HAR-J"] if k == "V1re" else vv["_v2"]
            ht = tj["pred_HAR-J"].values if k == "V1re" else hv2[t.index].values
            if k == "G3":
                qs = [1 / qmean(pv.rv_actual, c) for c in (hv, pv.pred_XGBoost, pv.pred_LightGBM)]
                ws = np.array(qs) / sum(qs)
                out[k] += list(ws[0] * ht + ws[1] * tj.pred_XGBoost.values + ws[2] * tj.pred_LightGBM.values)
            else:
                ml, w = choose(pv.rv_actual.values, hv.values, pv.pred_XGBoost.values, pv.pred_LightGBM.values)
                mt = tj.pred_XGBoost.values if ml == "XGB" else tj.pred_LightGBM.values
                out[k] += list(w * ht + (1 - w) * mt)
    return {k: pd.Series(v, index=pd.MultiIndex.from_tuples(idx)) for k, v in out.items()}


def main():
    w = load_windows(); md = []; S = {}
    T1, T2, T3, B0, B1, G, V = [], [], [], [], [], [], []
    b0pass = True; dat = {}
    for h in (1, 5, 22):
        xt, xv = prep(w, h); xtc, xvc = prep(w, h, "pc")
        vq = val_q(xv, CAND); vqc = val_q(xvc, CAND)
        ok = bool((vqc[CAND[:5]].mean() <= vq[CAND[:5]].mean()).all()); b0pass &= ok
        B0.append(f"| {h} | " + " | ".join(f"{f4(vq[c].mean())} / {f4(vqc[c].mean())}" for c in CAND[:5]) + f" | {'да' if ok else 'нет'} |")
        hp, pick = honest(xt, vq, WIN)
        v2p, pick2 = honest(xt, vq, CAND)
        xv["_v2"] = xv[CAND].iloc[:, 0]; vv2 = xv.apply(lambda r: r[pick2[r.name[0]]], axis=1); xv["_v2"] = vv2
        sv = saved(h).set_index(["date", "ticker"]); hyb = sv["pred_V1_Adaptive"]
        keys = [(d, t) for _, d, t in xt.index]
        yS = pd.Series(xt["y"].values, index=xt.index)
        L = {c: pooled_loss(yS, xt[c]) for c in WIN + ["AveW", "D2"]}
        L["honest"] = pooled_loss(yS, hp); L["v2"] = pooled_loss(yS, v2p)
        hb = pd.Series(hyb.reindex(keys).values, index=xt.index); mk = hb.notna().values
        Lh = pooled_loss(yS[mk], hb[mk]); L = {k: v[v.index.isin(Lh.index)] for k, v in L.items()}
        dat[h] = (xt, xv, L, Lh, hp, pick, pick2, v2p, yS, hb)
        r = {k: summary(v) for k, v in L.items()}; rh = summary(Lh)
        T1.append(f"| {h} | " + " | ".join(f4(r[k][0]) for k in WIN + ["AveW", "honest"]) + f" | {f4(rh[0])} |")
        T1.append(f"| {h}, 2026 | " + " | ".join(f4(r[k][1]) for k in WIN + ["AveW", "honest"]) + f" | {f4(rh[1])} |")
        for k in WIN + ["AveW", "honest"]:
            t0, th, p0, ph = dm_row(L[k], Lh, h); V.append(f"| {h} | гибрид vs {k} | {f2(t0)} | {f2(th)} | {fp(p0)} | {fp(ph)} |")
        B1.append((h, r["v2"][0], r["AveW"][0], rh[0], dm_row(L["v2"], Lh, h), dm_row(L["AveW"], Lh, h)))
        T3.append(f"| {h} | " + " | ".join(f"{pick[s]}" for s in sorted(pick.index)) + " |")
        T3.append(f"| {h}, v2 | " + " | ".join(f"{pick2[s]}" for s in sorted(pick2.index)) + " |")
        S[h] = {"E": r["E"][0], "honest": r["honest"][0], "v2": r["v2"][0], "hyb": rh[0], "AveW": r["AveW"][0]}
    for h in (1, 22):
        _, _, L, Lh, *_ = dat[h]
        for y in range(2017, 2027):
            f = lambda l: l[l.index.get_level_values("date").year == y].mean()
            T2.append(f"| {h} | {y} | " + " | ".join(f4(f(L[k])) for k in WIN + ["AveW", "honest"]) + f" | {f4(f(Lh))} |")
    ph = holm([b[4][3] for b in B1]); pa = holm([b[5][3] for b in B1])
    b1ok = all(b[4][1] > 0 and ph[i] < .01 and pa[i] < .01 for i, b in enumerate(B1))
    # --- V1 h=5 проверка и рычаги ---
    gl = {}
    for h in (1, 5, 22):
        xt, xv, L, Lh, hp, pick, pick2, v2p, yS, hb = dat[h]
        bl = blends(xt, xv, h, v2p); sv = saved(h).set_index(["date", "ticker"])
        keys = [(d, t) for _, d, t in xt.index]
        bs = {c: pd.Series(sv[c].reindex(keys).values, index=xt.index) for c in ("pred_XGBoost", "pred_LightGBM")}
        for nm, s in {**bl, **{"XGB": bs["pred_XGBoost"], "LGB": bs["pred_LightGBM"]}}.items():
            m = s.reindex(yS.index).notna().values
            l = pooled_loss(yS[m], s.reindex(yS.index)[m]); gl[(h, nm)] = l
    for h in (1, 5, 22):
        Lh = dat[h][3]; v2l = dat[h][2]["v2"]
        row = {nm: gl[(h, nm)] for nm in ("V1re", "G1", "G3", "XGB", "LGB")}
        V1 = summary(Lh)[0]
        G.append(f"| {h} | {f4(V1)} | " + " | ".join(f4(summary(row[n])[0]) for n in ("V1re", "G1", "G3", "XGB", "LGB")) + " |")
        for n in ("G1", "G3"):
            t0, th, p0, ph_ = dm_row(v2l, row[n], h)
            S[h][n] = {"q": summary(row[n])[0], "t_hac": th, "p_hac": ph_}
    for n in ("G1", "G3"):
        pv_ = holm([S[h][n]["p_hac"] for h in (1, 5, 22)])
        S[n] = {"holm": list(pv_), "ok": bool(all(S[h][n]["t_hac"] > 0 for h in (1, 5, 22)) and (pv_ < .01).all()
                and S[22][n]["q"] <= min(summary(gl[(22, "XGB")])[0], summary(gl[(22, "LGB")])[0]))}
    S["b0pass"], S["b1ok"] = b0pass, b1ok
    S["b1"] = [(b[0], b[1], b[2], b[3], b[4][1], b[4][3], b[5][1], b[5][3]) for b in B1]
    S["holm_v2"], S["holm_ave"] = list(ph), list(pa)
    hd = "| h | " + " | ".join(WIN + ["AveW", "честный"]) + " | гибрид |\n|" + "---|" * 8
    md += ["### Таблица 1 (QLIKE, среднее годовых 2017–2025; строка «h, 2026» — янв–фев 2026 отдельно)", hd] + T1
    md += ["", "### DM гибрид против каждого (t>0: гибрид лучше)", "| h | пара | t номинальный | t HAC | p номин. | p HAC |", "|---|---|---|---|---|---|"] + V
    md += ["", "### Таблица 2 (по годам)", "| h | год | " + " | ".join(WIN + ["AveW", "честный"]) + " | гибрид |", "|" + "---|" * 9] + T2
    md += ["", "### Таблица 3 (окно по шагам 2017…2026)", "| h | " + " | ".join(str(y) for y in range(2017, 2027)) + " |", "|" + "---|" * 11] + T3
    md += ["", "### Б0 (валидационный QLIKE без поправки / с exp(ŷ+s²/2), среднее по шагам)", "| h | " + " | ".join(CAND[:5]) + " | с поправкой ≤ |", "|" + "---|" * 7] + B0
    md += ["", "### Б1 (честный v2 = min val по E,R1,R2,R3,D2,AveW)", "| h | v2 | AveW | гибрид | t HAC гибрид–v2 | p Холм | t HAC гибрид–AveW | p Холм |", "|---|---|---|---|---|---|---|---|"]
    for i, b in enumerate(B1): md.append(f"| {b[0]} | {f4(b[1])} | {f4(b[2])} | {f4(b[3])} | {f2(b[4][1])} | {fp(ph[i])} | {f2(b[5][1])} | {fp(pa[i])} |")
    md += ["", "### Рычаги и проверка V1 (QLIKE 2017–2025)", "| h | V1 сохранён | V1 пересбор (proxy) | Г1 | Г3 | XGB | LGB |", "|---|---|---|---|---|---|---|"] + G
    (OUTD / "tables.md").write_text("\n".join(md) + "\n")
    json.dump(S, open(OUTD / "summary.json", "w"), default=float, ensure_ascii=False, indent=1)
    print("\n".join(md)); print(json.dumps({k: S[k] for k in ("b0pass", "b1ok", "G1", "G3")}, default=float))


if __name__ == "__main__":
    main()
