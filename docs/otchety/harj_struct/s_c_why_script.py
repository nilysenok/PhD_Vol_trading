"""S-C-WHY: все числа разбора в одну таблицу docs/otchety/harj_struct/S_C_WHY.csv (длинный формат). Только чтение сохранённых рядов;
слои E1/W5 пересчитываются над сохранёнными C_V тем же кодом (cm_layers.layers), 𝓜 — cm_model.forward с сохранёнными θ. Обучения нет.
Запуск из корня: PYTHONPATH=. python3 -W ignore <scratchpad>/s_c_why/why_csv.py"""
import numpy as np
import pandas as pd
import torch

from models_v2.s11.v4_ntp_state import combine
from models_v2.s16 import x1_w3_core as C
from models_v2.s16 import x1_ya_walk as YA
from models_v2.s18 import cm_feat as CF
from models_v2.s18 import cm_layers as L
from models_v2.s18 import cm_model as M

R = C.ROOT
OUT = R / "docs/otchety/harj_struct/S_C_WHY.csv"
VARS = ("cd", "ck", "cka", "c7", "ckl")
ID = {"cd": "S-CD", "ck": "S-CK", "cka": "S-CKA", "c7": "S-C7", "ckl": "S-CKL", "a2": "S-A2"}
DM = "D_M t+ 2018-01-04..2026-11-10"
ACT = "активные d >= 2018-07-02"


def q(rv, f):
    r = rv / f
    return r - np.log(r) - 1.0


rows = []


def put(sec, hyp, var, level, metric, value, window, n, period=""):
    rows.append(dict(section=sec, hypothesis=hyp, variant=ID.get(var, var), level=level, metric=metric, value=float(value), window=window, n=int(n), period=str(period)))


x, gap = L.inputs()
w5 = pd.read_parquet(R / "data/regen/harj_x1/W5/w5g.parquet")
a2 = pd.read_parquet(R / "data/regen/harj_struct/A/a2.parquet")
cm = {v: pd.read_parquet(R / ("data/regen/harj_struct/CM/%s.parquet" % v)) for v in VARS}
rv, c8, W, Mx = (np.asarray(a, float) for a in (x["rv"], x["C"], w5["F"], x["M"]))
xd, tp = pd.DatetimeIndex(x["d"]), pd.DatetimeIndex(x["t_plus"])
act = xd >= YA.ACTIVATION
n, na = len(x), int(act.sum())

# ---------- H4: даты и строки
for v, df in cm.items():
    same = all([(df["d"].to_numpy() == x["d"].to_numpy()).all(), (df["t_plus"].to_numpy() == x["t_plus"].to_numpy()).all(), (df["rv"].to_numpy() == rv).all(),
                (df["cfg8"].to_numpy() == c8).all(), (w5["d"].to_numpy() == x["d"].to_numpy()).all(), (w5["t_plus"].to_numpy() == x["t_plus"].to_numpy()).all()])
    put("H4", "4", v, "строки", "d, t+, rv, cfg8 = x1/w5g побитно (1 = да)", same, DM, n)
    put("H4", "4", v, "C", "max |C/(cfg8·M) − 1|", np.max(np.abs(df["C"] / (df["cfg8"] * df["M"]) - 1)), DM, n)
    put("H4", "4", v, "F", "до включения C = cfg8 и F = W5-G побитно (1 = да)", (df["C"].to_numpy()[~act] == c8[~act]).all() and (df["F"].to_numpy()[~act] == W[~act]).all(), "d < 2018-07-02", n - na)

# ---------- H5: слой доверия
b = L.layers(x, x["C"], gap)
put("H5", "5", "cfg8", "X1", "max отн. |run(B_mix, cfg8) / x1.F − 1|", np.max(np.abs(b["X1_V"] / x["F"] - 1)), DM, n)
put("H5", "5", "cfg8", "F", "max отн. |W5(run(B_mix, cfg8)) / w5g.F − 1|", np.max(np.abs(b["F"] / W - 1)), DM, n)
for v, df in cm.items():
    lay = L.layers(x, df["C"].to_numpy(float), gap)
    put("H5", "5", v, "X1", "max отн. |run(B_mix, C_V) / X1_V сохр. − 1|", np.max(np.abs(lay["X1_V"] / df["X1_V"] - 1)), DM, n)
    put("H5", "5", v, "X1", "max |π пересчёт − π сохр.|", np.max(np.abs(lay["pi"] - df["pi"])), DM, n)
    put("H5", "5", v, "X1", "max отн. |combine(B_mix, C_V, π) / X1_V − 1|", np.max(np.abs(combine(Mx, df["C"], df["pi"]) / df["X1_V"] - 1)), DM, n)
    put("H5", "5", v, "X1", "дней с α₀ ≠ α₀ X1", int((lay["a0"] != x["a0"]).sum()), DM, n)
    put("H5", "5", v, "X1", "дней с α₁ ≠ α₁ X1", int((lay["a1"] != x["a1"]).sum()), DM, n)
    put("H5", "5", v, "X1", "среднее π (X1: %.4f)" % x["pi"].mean(), df["pi"].mean(), DM, n)

# ---------- H1: уровень C против уровня X1 и F
Q = lambda f, m=slice(None): float(np.mean(q(rv[m], np.asarray(f, float)[m])))
qc8, qX1, qW = Q(c8), Q(x["F"]), Q(W)
put("H1", "1", "cfg8", "C", "QLIKE", qc8, DM, n)
put("H1", "1", "B_mix", "M", "QLIKE", Q(Mx), DM, n)
put("H1", "1", "X1", "X1", "QLIKE", qX1, DM, n)
put("H1", "1", "W5-G", "F", "QLIKE", qW, DM, n)
put("H1", "1", "a2", "C", "Δ QLIKE C_A к cfg8, %", 100 * (Q(a2["C_A"]) / qc8 - 1), DM, n)
for v, df in cm.items():
    qcf = Q(combine(Mx, df["C"], x["pi"]))
    put("H1", "1", v, "C", "Δ QLIKE C_V к cfg8, %", 100 * (Q(df["C"]) / qc8 - 1), DM, n)
    put("H1", "1", v, "X1", "Δ QLIKE X1_V к X1, %", 100 * (Q(df["X1_V"]) / qX1 - 1), DM, n)
    put("H1", "1", v, "X1", "из них: C_V при прежнем π X1, %", 100 * (qcf / qX1 - 1), DM, n)
    put("H1", "1", v, "X1", "из них: реакция π на новый C, %", 100 * (Q(df["X1_V"]) - qcf) / qX1, DM, n)
    put("H1", "1", v, "F", "Δ QLIKE F к W5-G, %", 100 * (Q(df["F"]) / qW - 1), DM, n)

# ---------- H2: уровень против разброса
for nm, s in (("cfg8", c8), ("B_mix", Mx), ("X1", x["F"]), ("W5-G", W)):
    put("H2", "2", nm, "-", "ln mean(RV/F) — оптимальный сдвиг уровня (исход)", np.log(np.mean(rv / np.asarray(s, float))), DM, n)
for y in sorted(set(tp.year)):
    m = tp.year == y
    put("H2", "2", "cfg8", "C", "ln mean(RV/cfg8) по году t+ (исход)", np.log(np.mean(rv[m] / c8[m])), "год t+", m.sum(), y)
ks = np.round(np.arange(-1.0, 2.0001, 0.05), 2)
for v, df in cm.items():
    for lev, base, new in (("C", c8, df["C"].to_numpy(float)), ("F", W, df["F"].to_numpy(float))):
        dl, bb, r_ = np.log(new / base)[act], base[act], rv[act]
        Qb = np.mean(q(r_, bb))
        u = np.log(r_ / bb)
        qk = np.array([np.mean(q(r_, bb * np.exp(k * dl))) for k in ks])
        put("H2", "2", v, lev, "Δ QLIKE, %", 100 * (np.mean(q(r_, bb * np.exp(dl))) / Qb - 1), ACT, na)
        put("H2", "2", v, lev, "среднее δ = ln(V/база)", dl.mean(), ACT, na)
        put("H2", "2", v, lev, "Δ только уровень (δ := среднее δ), %", 100 * (np.mean(q(r_, bb * np.exp(dl.mean()))) / Qb - 1), ACT, na)
        put("H2", "2", v, lev, "σ δ", dl.std(ddof=1), ACT, na)
        put("H2", "2", v, lev, "Δ только разброс (δ − среднее), %", 100 * (np.mean(q(r_, bb * np.exp(dl - dl.mean()))) / Qb - 1), ACT, na)
        put("H2", "2", v, lev, "corr(δ, ln(RV/база))", np.corrcoef(u, dl)[0, 1], ACT, na)
        put("H2", "2", v, lev, "κ* = argmin QLIKE(база·exp(κδ)), сетка −1..2 шаг 0,05 (исход, потолок)", ks[int(np.argmin(qk))], ACT, na)

# ---------- H3 и H6: λ по месяцам, внутри выборки против вне, концентрация
feat = CF.load_feat()
pos = YA.month_starts(xd)
j2224 = int(np.flatnonzero(tp == pd.Timestamp("2022-02-24"))[0])
put("H6", "6", "cfg8", "C", "QLIKE пары t+ 2022-02-24 (RV/cfg8 = %.1f)" % (rv[j2224] / c8[j2224]), q(rv[j2224], c8[j2224]), "одна пара", 1, "2022-02-24")
for v in ("cd", "ck", "cka", "ckl"):
    var, df = v.upper(), cm[v]
    D, names, mon = M.build_data(x, feat, lag=var == "CKL"), list(M.NAMES[var]), []
    for a, b_ in zip(pos, pos[1:] + [n]):
        if xd[a] < YA.ACTIVATION:
            continue
        th = torch.as_tensor(df.loc[a, names].to_numpy(float))
        lam = float(df.loc[a, "lam"])
        pool = YA.closed_pool(tp, xd, xd[a])
        V = D.view(pool)
        Cp, rp, cp = (V.c8 * M.forward(th, V, var)["M"]).numpy(), V.rv.numpy(), V.c8.numpy()
        g = q(rp, Cp) - q(rp, cp)
        k1 = max(1, int(round(0.01 * len(g))))
        srt = np.sort(g)
        days = np.arange(a, b_)
        r = dict(lam=lam, n_train=len(pool), pen=lam * float((th ** 2).sum()), d_ins=g.mean(), d_ins_rest99=srt[k1:].mean(), top1=srt[:k1].sum() / g.sum(),
                 d_oos_C=np.mean(q(rv[days], df["C"].to_numpy(float)[days])) - np.mean(q(rv[days], c8[days])),
                 d_oos_F=np.mean(q(rv[days], df["F"].to_numpy(float)[days])) - np.mean(q(rv[days], W[days])),
                 lnk_pool=np.log(np.mean(rp / cp)), lnk_month=np.log(np.mean(rv[days] / c8[days])), lnM=float(np.mean(np.log(df["M"].to_numpy(float)[days]))), pre=xd[a] < pd.Timestamp("2022-02-24"))
        jj = np.flatnonzero(tp[pool] == pd.Timestamp("2022-02-24"))
        r["sh2224"] = g[jj].sum() / g.sum() if len(jj) else np.nan
        mon.append(r)
        for k in ("lam", "n_train", "d_ins", "d_ins_rest99", "top1", "d_oos_C", "d_oos_F"):
            put("H3_month", "3", v, "C" if k != "d_oos_F" else "F", k, r[k], "месяц D (выигрыш — QLIKE абс., минус лучше cfg8/W5-G)", len(pool) if k.startswith("d_ins") or k == "top1" else len(days), xd[a].date())
    t = pd.DataFrame(mon)
    for lv in M.LAMS:
        put("H3", "3", v, "C", "месяцев с λ = %g" % lv, (t.lam == lv).sum(), "100 активных месяцев", len(t))
    put("H3", "3", v, "C", "медиана штраф / выигрыш внутри выборки", (t.pen / -t.d_ins).median(), "100 активных месяцев", len(t))
    put("H3", "3", v, "C", "ср. выигрыш внутри выборки (QLIKE абс.)", t.d_ins.mean(), "пул месяца", len(t))
    put("H3", "3", v, "C", "ср. выигрыш вне выборки (QLIKE абс.)", t.d_oos_C.mean(), "месяц D", len(t))
    put("H3", "3", v, "C", "месяцев, где C_V хуже cfg8 вне выборки", (t.d_oos_C > 0).sum(), "100 активных месяцев", len(t))
    put("H2", "2", v, "C", "corr(ln mean(RV/cfg8) пула, ср. ln 𝓜 месяца)", np.corrcoef(t.lnk_pool, t.lnM)[0, 1], "100 активных месяцев", len(t))
    put("H2", "2", v, "C", "corr(ср. ln 𝓜 месяца, ln mean(RV/cfg8) месяца)", np.corrcoef(t.lnM, t.lnk_month)[0, 1], "100 активных месяцев", len(t))
    put("H2", "2", v, "C", "ср. ln mean(RV/cfg8) пула", t.lnk_pool.mean(), "100 активных месяцев", len(t))
    put("H2", "2", v, "C", "ср. ln mean(RV/cfg8) месяца прогноза", t.lnk_month.mean(), "100 активных месяцев", len(t))
    put("H6", "6", v, "C", "медиана доли выигрыша внутри выборки от 1 % лучших пар", t.top1.median(), "100 активных месяцев", len(t))
    put("H6", "6", v, "C", "медиана доли от пары t+ 2022-02-24 (месяцы после неё)", t.sh2224.median(), "месяцы D > 2022-02-24", t.sh2224.notna().sum())
    put("H6", "6", v, "C", "ср. выигрыш остальных 99 % пар внутри выборки (QLIKE абс.)", t.d_ins_rest99.mean(), "100 активных месяцев", len(t))
    put("H6", "6", v, "C", "месяцев, где остальные 99 % пар хуже cfg8 внутри выборки", (t.d_ins_rest99 > 0).sum(), "100 активных месяцев", len(t))

out = pd.DataFrame(rows)
out.to_csv(OUT, index=False)
s = out[out.section != "H3_month"]
with pd.option_context("display.max_rows", 500, "display.width", 250, "display.max_colwidth", 90):
    print(s[["section", "variant", "level", "metric", "value", "n", "period"]].to_string(index=False))
print("записано строк:", len(out))
