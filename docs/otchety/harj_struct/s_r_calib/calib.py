"""S-R-SPEC, калибровка до счёта (только чтение сохранённых рядов; ни одна новая модель не обучается, потери новых вариантов не считаются).
1) вложение потери уровня F: B_mix^(1-pi)*cfg8^pi*exp(beta*gap) против w5g.F;
2) PIT-квантиль q = 0,99 отношения x0 = RV/F0 по закрытым парам (t+ <= D), m = mean(min(x0, c)), смещение обрезки;
3) первый месяц, когда закрытых пар >= 630 (3 блока по 126 + 252 обучения), для пулов C (пары D_M) и A (пары K1 с 2017);
4) доля выигрыша от 1 % лучших дат для сохранённых пар рядов (калибровка порога концентрации 0,5, объявленного в S_C_WHY разд. 5.4).
Запуск из корня: PYTHONPATH=. python3 -W ignore <sp>/s_r/calib.py"""
import numpy as np
import pandas as pd

from models_v2.s16 import x1_w3_core as C
from models_v2.s11.v4_ntp_state import combine, qlike
from models_v2.s18 import a_feat as A

Q, VAL, NBLK, MIN_TRAIN = 0.99, 126, 3, 252
ROOT = C.ROOT
x = C.load()
w = pd.read_parquet(ROOT / "data/regen/harj_x1/W5/w5g.parquet")
assert (w["d"].to_numpy() == x["d"].to_numpy()).all()
on = (w["gap"].to_numpy() == 1) & ~pd.isna(w["refit_date"]).to_numpy()
f0 = combine(x["M"], x["C"], x["pi"]) * np.exp(np.where(on, w["beta"].to_numpy(), 0.0))
print("1) max отн. |F0 - w5g.F| =", "%.2e" % np.max(np.abs(f0 / w["F"].to_numpy() - 1)))
print("   max отн. |combine(M,C,pi) - x1.F| =", "%.2e" % np.max(np.abs(combine(x["M"], x["C"], x["pi"]) / x["F"].to_numpy() - 1)))

rv, tp, d = x["rv"].to_numpy(float), pd.DatetimeIndex(x["t_plus"]), pd.DatetimeIndex(x["d"])
x0 = rv / w["F"].to_numpy()
i24 = int(np.flatnonzero(tp == pd.Timestamp("2022-02-24"))[0])
print("   x0(2022-02-24) = RV/W5-G = %.2f; RV/cfg8 = %.2f" % (x0[i24], rv[i24] / x["C"].iloc[i24]))

k = A.load_k1()
k = k[~k["cross_seg"]].reset_index(drop=True)
kd, ktp = pd.DatetimeIndex(k["d"]), pd.DatetimeIndex(k["target_date"])
xa = k["rv"].to_numpy(float) / k["sigma2"].to_numpy(float)
pos = pd.Series(np.arange(len(x)), index=d)
in_m = kd.isin(d)
xa[in_m] = x0[pos.reindex(kd[in_m]).to_numpy().astype(int)]
print("   пул A: пар K1 без cross_seg %d, из них на D_M %d; первая t+ %s" % (len(k), in_m.sum(), ktp.min().date()))

ms = [int(i) for i in np.flatnonzero(~pd.Series(list(zip(d.year, d.month))).duplicated().to_numpy())]
rows = []
for a in ms:
    D = d[a]
    for name, xs, t in (("C", x0, tp), ("A", xa, ktp)):
        p = xs[np.asarray(t <= D)]
        if len(p) < 50:
            rows.append((D.date(), name, len(p), np.nan, np.nan, np.nan, np.nan, np.nan))
            continue
        c = float(np.quantile(p, Q))
        m = float(np.mean(np.minimum(p, c)))
        rows.append((D.date(), name, len(p), c, m, float(np.mean(p)), float(np.mean(p > c)), float(np.max(p))))
R = pd.DataFrame(rows, columns=["D", "pool", "n", "c_q99", "m_c", "mean_x0", "share_gt_c", "max_x0"])
R.to_csv(__file__.replace("calib.py", "calib_months.csv"), index=False)
show = R[pd.to_datetime(R["D"]).dt.month.isin([1]) | (pd.to_datetime(R["D"]) == pd.Timestamp("2022-04-01")) | (pd.to_datetime(R["D"]) == pd.Timestamp("2018-07-02"))]
print("2) PIT q99 и m_c по закрытым парам (выборка месяцев):")
print(show.to_string(index=False, float_format=lambda v: "%.4f" % v))
for name in ("C", "A"):
    s = R[(R["pool"] == name) & R["c_q99"].notna()]
    print("   %s: c_q99 мин/медиана/макс %.3f / %.3f / %.3f; 1 - m_c мин/медиана/макс %.4f / %.4f / %.4f" % (name, s.c_q99.min(), s.c_q99.median(), s.c_q99.max(), (1 - s.m_c).min(), (1 - s.m_c).median(), (1 - s.m_c).max()))
need = MIN_TRAIN + NBLK * VAL
for name in ("C", "A"):
    s = R[(R["pool"] == name) & (R["n"] >= need)]
    kap = s.c_q99 / s.m_c
    print("   %s, активные месяцы (n >= %d, %d мес.): c_q99 %.3f…%.3f (медиана %.3f); 1 - m_c %.4f…%.4f (медиана %.4f); κ = c/m %.3f…%.3f; доля пар выше c %.4f…%.4f"
          % (name, need, len(s), s.c_q99.min(), s.c_q99.max(), s.c_q99.median(), (1 - s.m_c).min(), (1 - s.m_c).max(), (1 - s.m_c).median(), kap.min(), kap.max(), s.share_gt_c.min(), s.share_gt_c.max()))
    last = s.iloc[-1]
    print("   %s, последний месяц %s: n %d, c %.3f, m %.4f; влияние 2022-02-24 на балл уровня: без обрезки (x0/1 - 1)/n = %.5f, с обрезкой (κ - 1)/n = %.5f"
          % (name, last.D, last.n, last.c_q99, last.m_c, (x0[i24] - 1) / last.n, (last.c_q99 / last.m_c - 1) / last.n))
for name in ("C", "A"):
    s = R[(R["pool"] == name) & (R["n"] >= need)]
    print("3) пул %s: первый месяц с n >= %d — %s (n = %d)" % (name, need, s.iloc[0]["D"], s.iloc[0]["n"]))
s = R[R["pool"] == "C"]
print("   пул C: n на 2018-07-02 = %d; на 2019-01 = %d" % (s.iloc[[i for i, v in enumerate(s["D"]) if str(v) == "2018-07-02"][0]]["n"], s[pd.to_datetime(s["D"]).dt.strftime("%Y-%m") == "2019-01"].iloc[0]["n"]))

# 4) доля выигрыша от 1 % лучших дат: s = sum(-d_t по 22 датам с наименьшим d_t) / sum(-d_t), d_t = l_cand - l_base; определена при sum(-d_t) > 0
def share(cand, base):
    dd = qlike(rv, cand) - qlike(rv, base)
    g = -dd.sum()
    top = np.sort(dd)[: int(np.ceil(0.01 * len(dd)))]
    return 100 * dd.mean() / qlike(rv, base).mean(), (-top.sum() / g) if g > 0 else np.nan, np.mean(dd < 0)

ser = dict(K1=x["K1"].to_numpy(), K2=x["K2"].to_numpy(), K3=x["K3"].to_numpy(), B_mix=x["M"].to_numpy(), X1=x["F"].to_numpy(), W5G=w["F"].to_numpy(),
           W8G=pd.read_parquet(ROOT / "data/regen/harj_x1/W8/w8g.parquet")["F"].to_numpy(),
           SA2=pd.read_parquet(ROOT / "data/regen/harj_struct/A/a2.parquet")["F"].to_numpy(),
           SCK=pd.read_parquet(ROOT / "data/regen/harj_struct/CM/ck.parquet")["F"].to_numpy())
print("4) доля выигрыша от 1 %% лучших дат (22 из %d), D_M:" % len(rv))
print("   крупнейшие x0 в D_M:", ", ".join("%s %.2f" % (tp[i].date(), x0[i]) for i in np.argsort(-x0)[:5]))
for a, b in (("K2", "K1"), ("B_mix", "K1"), ("X1", "K2"), ("X1", "K3"), ("X1", "B_mix"), ("W5G", "X1"), ("W8G", "W5G"), ("W8G", "X1"), ("SA2", "W5G"), ("SA2", "X1"), ("SCK", "W5G")):
    dl, sh, fr = share(ser[a], ser[b])
    print("   %-6s против %-6s: Δ %+.4f %%, доля 1 %% лучших %s, доля дат с выигрышем %.3f" % (a, b, dl, "—" if np.isnan(sh) else "%.3f" % sh, fr))
