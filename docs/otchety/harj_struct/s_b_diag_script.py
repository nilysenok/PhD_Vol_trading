"""S-B-DIAG (Ж-254 п. 7, 10): есть ли основание ждать пользы модуля B. Только чтение готовых рядов A0/A1/A2 и баров SBER; новых обучений гибрида нет.
Критерий — `KRITERIJ.md` рядом (объявлен до счёта). Запуск из корня репозитория: PYTHONPATH=. python3 <этот файл> <путь S_B_DIAG.csv>."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from models_v2.s13 import v6_1_fit as FT, v6_2_q as QF
from models_v2.s17.fwd_src import LAST_SESSION, Src
from models_v2.s18 import a_feat as A

ROOT = Path.cwd()
OUT = Path(sys.argv[1])
NORM_END = pd.Timestamp("2017-12-29")
SHOCK = pd.Timestamp("2022-02-24")
N_NULL, MIN_SHIFT, SEED = 500, 60, 0
THR = 0.10
rows_out = []


def put(block, item, value, base="", measure="", window="D_M", n="", note=""):
    rows_out.append(dict(block=block, item=item, base=base, measure=measure, window=window, n=n, value=value, note=note))


# ---------- данные ----------
src = Src()
bars = src.bars("SBER", LAST_SESSION)
F = A.build(bars, src.index_days(LAST_SESSION))
qt = QF.day_table(bars).reindex(F.dates)
qt["q8"] = QF.mode_share(qt["n"].dropna()).reindex(qt.index)
Q = qt[QF.Q_NAMES].to_numpy(float)
if not np.isfinite(Q).all():
    raise SystemExit("NaN в q на днях дневного ряда: %d" % (~np.isfinite(Q)).sum())

# сверка с сохранённым Q V6
v6 = pd.read_parquet(ROOT / "data/regen/harj_hybrid_v6/Q_features.parquet").set_index("d")
common = v6.index.intersection(F.dates)
diff = np.abs(qt.loc[common, QF.Q_NAMES].to_numpy() - v6.loc[common, QF.Q_NAMES].to_numpy())
put("sverka", "max |q − Q_features V6| по общим датам", float(diff.max()), window="общие даты V6 %s…%s" % (common.min().date(), common.max().date()), n=len(common), note="q1–q8 кодом v6_2_q без правок; ненулевое — порядок суммирования или бары")

z5 = np.log(Q[:, QF.Q_NAMES.index("q5")])
C = QF.components(Q)
tr = np.asarray(F.dates <= NORM_END)
mu, sd = FT.norm_stats(C[tr])
zu = ((C - mu) / sd).mean(axis=1)
Z = {"ln_q5": z5, "u": zu}

base = {v: pd.read_parquet(ROOT / ("data/regen/harj_struct/A/%s.parquet" % v.lower())) for v in ("A0", "A1", "A2")}
d = base["A1"]["d"]
for v in base:
    if not (base[v]["d"].to_numpy() == d.to_numpy()).all() or len(d) != 2198:
        raise SystemExit("даты %s ≠ D_M" % v)
r = F.dates.get_indexer(d)
if (r < 0).any():
    raise SystemExit("дата D_M вне дневного ряда")
rv_next = base["A1"]["rv"].to_numpy(float)
tplus = pd.DatetimeIndex(base["A1"]["t_plus"])
n = len(r)


def ql(y, g=0.0):
    return y * np.exp(-g) + g - np.log(y) - 1.0


# ---------- веса a базы, направления B ----------
lag = A.lag_idx(F, r)


def weights(dfb):
    """a[k] (n, L_k) по θ строки (среднее по seeds), группами по месяцу переоценки."""
    out = [np.zeros(ok.shape) for _, ok in lag]
    for _, ix in dfb.groupby("refit_date").indices.items():
        row = dfb.iloc[ix[0]]
        for k, ks in enumerate(A.KS):
            om = torch.tensor(float(row["om_" + ks]), dtype=torch.float64)
            ps = torch.tensor([float(row["psi_%s1" % ks]), float(row["psi_%s2" % ks])], dtype=torch.float64)
            out[k][ix] = A.lag_weights(lag[k][1][ix], k, om, ps).numpy()
    return out


def parts(dfb):
    om = dfb[["om_s", "om_m", "om_l"]].to_numpy(float)
    W = weights(dfb)
    lnS, DB, DM, lnS_har = [], {m: [] for m in Z}, {m: [] for m in Z}, []
    for k, (ix, ok) in enumerate(lag):
        a = W[k]
        RVl = np.where(ok, F.rv[ix], 0.0)
        S = (a * RVl).sum(1)
        lnS.append(np.log(S))
        L = ok.shape[1]
        R = ok & (np.arange(L) < A.RECT[k])[None, :]
        R = R / R.sum(1, keepdims=True)
        lnS_har.append(np.log((R * RVl).sum(1)))
        G = (a - (1 - om[:, k])[:, None] * R)
        Gs = G.sum(1, keepdims=True)
        Gn = np.where(Gs > 1e-12, G / np.where(Gs > 1e-12, Gs, 1.0), 0.0)
        SG, SR = (Gn * RVl).sum(1), (R * RVl).sum(1)
        for m, z in Z.items():
            zl = np.where(ok, z[ix], 0.0)
            DB[m].append(-((a * zl * RVl).sum(1) / S - (a * zl).sum(1)))
            DM[m].append(z[r] * (SG - SR) / S)
    return np.column_stack(lnS), np.column_stack(lnS_har), {m: np.column_stack(v) for m, v in DB.items()}, {m: np.column_stack(v) for m, v in DM.items()}


# ---------- восстановление H_A A1 ----------
k1 = A.load_k1().set_index("d").reindex(d)
lnS1, lnS0, DB1, DM1 = parts(base["A1"])
b1 = base["A1"]
beta = k1[["b_d", "b_w", "b_m"]].to_numpy(float)
dl = b1[["d_s", "d_m", "d_l"]].to_numpy(float)
lnH = np.log(k1["sigma2"].to_numpy(float)) + b1["d0"].to_numpy() + ((beta + dl) * lnS1).sum(1) - (beta * lnS0).sum(1) + b1["d_J"].to_numpy() * np.log1p(F.j[r])
relH = np.abs(np.exp(lnH) / b1["H_A"].to_numpy() - 1)
put("sverka", "max отн. |H_A(θ строки) / H_A A1 − 1|", float(relH.max()), base="A1", n=n, note="θ — среднее по 5 seeds, поэтому сверка приближённая; медиана %.2e" % np.median(relH))
put("sverka", "доля дат с отн. расхождением H_A > 1e-3", float((relH > 1e-3).mean()), base="A1", n=n, note="месяцы, где seeds разошлись; направление B на них приближённое")
lnS0_chk = np.abs(lnS0 - np.log(np.column_stack([F.rv[r], F.rv_w[r], F.rv_m[r]]))).max()
put("sverka", "max |ln S(HAR) − ln(rv, rv_w, rv_m)|", float(lnS0_chk), n=n)

# ---------- оракул ----------


def fit(y, X):
    """min mean QLIKE(y·e^{−g}), g = Xβ; X — с константой первым столбцом. Ньютон с поиском шага."""
    Xs = X.copy()
    s = Xs[:, 1:].std(0)
    Xs[:, 1:] /= np.where(s > 0, s, 1.0)
    b = np.zeros(X.shape[1])
    b[0] = np.log(y.mean())
    f = ql(y, Xs @ b).mean()
    for _ in range(200):
        g = Xs @ b
        w = y * np.exp(-g)
        gr = Xs.T @ (1 - w) / len(y)
        H = (Xs * w[:, None]).T @ Xs / len(y) + 1e-12 * np.eye(len(b))
        st = np.linalg.solve(H, gr)
        t, dec = 1.0, gr @ st
        while True:
            nb = b - t * st
            nf = ql(y, Xs @ nb).mean()
            if nf <= f - 1e-4 * t * dec or t < 1e-10:
                break
            t /= 2
        b, fo, f = nb, f, nf
        if abs(fo - f) < 1e-15:
            break
    return f


def gains(y, K, B, base_q):
    one = np.ones((len(y), 1))
    q_c = fit(y, one)
    q_k = fit(y, np.hstack([one, K]))
    q_b = fit(y, np.hstack([one, B]))
    q_kb = fit(y, np.hstack([one, K, B]))
    g = lambda a, b: 100 * (a - b) / base_q
    return dict(G_K=g(q_c, q_k), G_B=g(q_c, q_b), G_BK=g(q_k, q_kb), q_c=q_c)


rng = np.random.default_rng(SEED)
shifts = rng.integers(MIN_SHIFT, n - MIN_SHIFT + 1, N_NULL)
keep = np.asarray(tplus != SHOCK)
hyp = []
for v in ("A1", "A2"):
    dfb = base[v]
    lnF = np.log(dfb["F"].to_numpy(float))
    y = rv_next / np.exp(lnF)
    base_q = ql(y).mean()
    put("baza", "QLIKE F", float(base_q), base=v, n=n)
    if v == "A1":
        DB, DMX = DB1, DM1
    else:
        _, _, DB, DMX = parts(dfb)
    s = np.log(np.column_stack([F.rv[r], F.rv_w[r], F.rv_m[r]])) - lnF[:, None]
    for m, z in Z.items():
        zd = z[r]
        K = np.column_stack([zd, zd[:, None] * s])
        B = DB[m]
        gg = gains(y, K, B, base_q)
        one = np.ones((n, 1))
        q_k = fit(y, np.hstack([one, K]))
        null = np.array([100 * (q_k - fit(y, np.hstack([one, K, np.roll(B, sh, axis=0)]))) / base_q for sh in shifts])
        med = float(np.median(null))
        p = (1 + (null >= gg["G_BK"]).sum()) / (N_NULL + 1)
        yk, Kk, Bk = y[keep], K[keep], B[keep]
        bq_k = ql(yk).mean()
        g_ns = gains(yk, Kk, Bk, bq_k)["G_BK"]
        hyp.append(dict(base=v, measure=m, G=gg["G_BK"], med=med, p=p, E=gg["G_BK"] - med, E_ns=g_ns - med))
        put("oracul", "G(K): обычный корректор z_d, z_d·s1..s3", gg["G_K"], v, m, n=n, note="% QLIKE базы, в выборке (потолок)")
        put("oracul", "G(B): направление B D_s, D_m, D_l", gg["G_B"], v, m, n=n, note="% QLIKE базы, в выборке (потолок)")
        put("oracul", "G(B|K)", gg["G_BK"], v, m, n=n, note="прирост B сверх K, % QLIKE базы")
        put("oracul", "нуль G(B|K): медиана", med, v, m, n=n, note="%d циклических сдвигов B, seed 0" % N_NULL)
        put("oracul", "нуль G(B|K): 95-й процентиль", float(np.quantile(null, 0.95)), v, m, n=n)
        put("oracul", "E = G(B|K) − медиана нуля", gg["G_BK"] - med, v, m, n=n, note="условие 1: ≥ %.2f" % THR)
        put("oracul", "p перестановочный G(B|K)", p, v, m, n=n)
        put("oracul", "E без t⁺ 2022-02-24", g_ns - med, v, m, window="D_M без 2022-02-24", n=int(keep.sum()), note="условие 3: ≥ %.2f" % THR)
        # справочно, добавлено после первого счёта, в решение не входит: нуль на срезе без 24.02.2022 и нуль для K
        q_kk = fit(yk, np.hstack([one[keep], Kk]))
        null_ns = np.array([100 * (q_kk - fit(yk, np.hstack([one[keep], Kk, np.roll(Bk, sh, axis=0)]))) / bq_k for sh in shifts])
        put("sprav", "G(B|K) без 2022-02-24", g_ns, v, m, window="D_M без 2022-02-24", n=int(keep.sum()), note="справочно, после счёта; в решение не входит")
        put("sprav", "нуль G(B|K) без 2022-02-24: медиана / 95-й процентиль", "%.4f / %.4f" % (np.median(null_ns), np.quantile(null_ns, 0.95)), v, m, window="D_M без 2022-02-24", n=int(keep.sum()), note="справочно")
        put("sprav", "p перестановочный G(B|K) без 2022-02-24", float((1 + (null_ns >= g_ns).sum()) / (N_NULL + 1)), v, m, window="D_M без 2022-02-24", n=int(keep.sum()), note="справочно")
        q_c = fit(y, one)
        null_k = np.array([100 * (q_c - fit(y, np.hstack([one, np.roll(K, sh, axis=0)]))) / base_q for sh in shifts])
        put("sprav", "нуль G(K): медиана / 95-й процентиль", "%.4f / %.4f" % (np.median(null_k), np.quantile(null_k, 0.95)), v, m, n=n, note="справочно: обычный корректор в выборке против сдвигов K")
        put("sprav", "p перестановочный G(K)", float((1 + (null_k >= gg["G_K"]).sum()) / (N_NULL + 1)), v, m, n=n, note="справочно")
        # B-mix: перенос веса с лага 0 на ядро по качеству дня d
        gm = gains(y, K, DMX[m], base_q)
        put("oracul", "G(Bmix): z_d·(S_G − S_R)/S", gm["G_B"], v, m, n=n, note="справочно: это HARQ-подобный канал")
        put("oracul", "G(Bmix|K)", gm["G_BK"], v, m, n=n, note="справочно")
        if v == "A1":
            harq = zd * (np.log(F.rv_w[r]) - np.log(F.rv[r]))
            c = np.corrcoef(DMX[m][:, 0], harq)[0, 1] if DMX[m][:, 0].std() > 0 else np.nan
            put("oracul", "corr(Bmix_s, z_d·ln(RV_w/RV_d))", float(c), v, m, n=n, note="справочно")
            put("oracul", "доля дат с ω_s = 0 (B на короткой памяти не действует)", float((dfb["om_s"] <= 1e-8).mean()), v, n=n)
            put("oracul", "ω_s, ω_m, ω_l A1 — среднее по датам", "%.4f / %.4f / %.4f" % tuple(dfb[["om_s", "om_m", "om_l"]].mean()), v, n=n)

# Холм m = 4
ps = np.array([h["p"] for h in hyp])
order = np.argsort(ps)
ph = np.empty(4)
run = 0.0
for i, j in enumerate(order):
    run = max(run, min(1.0, (4 - i) * ps[j]))
    ph[j] = run
any_ok = False
for h, p_h in zip(hyp, ph):
    ok = h["E"] >= THR and p_h < 0.05 and h["E_ns"] >= THR
    any_ok |= ok
    put("reshenie", "p Холма (m = 4)", float(p_h), h["base"], h["measure"], n=n)
    put("reshenie", "условия 1/2/3 выполнены", "%s/%s/%s" % (h["E"] >= THR, p_h < 0.05, h["E_ns"] >= THR), h["base"], h["measure"], n=n)
put("reshenie", "основание есть", bool(any_ok), n=n, note="по KRITERIJ.md: хотя бы одна гипотеза с условиями 1, 2, 3")

# ---------- группы по PIT-терцилям ----------
q_a1 = ql(rv_next / base["A1"]["F"].to_numpy(float))
q_a0 = ql(rv_next / base["A0"]["F"].to_numpy(float))
res1 = np.log(rv_next / base["A1"]["F"].to_numpy(float))


def lag_mean(z):
    out = np.full(len(F.dates), np.nan)
    for i in range(len(F.dates)):
        L = min(21, F.pos[i])
        if L >= 5:
            out[i] = z[i - L:i].mean()
    return out


def terc(series_all, rows):
    lab = np.full(len(rows), -1)
    for i, ri in enumerate(rows):
        h = series_all[:ri]
        h = h[np.isfinite(h)]
        if len(h) >= 250 and np.isfinite(series_all[ri]):
            t1, t2 = np.quantile(h, [1 / 3, 2 / 3])
            lab[i] = 0 if series_all[ri] <= t1 else (1 if series_all[ri] <= t2 else 2)
    return lab


for m, z in Z.items():
    for nm, ser in (("качество дня d", z), ("среднее качество лагов 1–21", lag_mean(z))):
        lab = terc(ser, r)
        for g, gname in enumerate(("лучшая треть", "средняя", "худшая треть")):
            sel = lab == g
            put("gruppy", "%s: %s — QLIKE A1" % (nm, gname), float(q_a1[sel].mean()), "A1", m, n=int(sel.sum()))
            put("gruppy", "%s: %s — Δ A1 к A0, %%" % (nm, gname), float(100 * (q_a1[sel].mean() - q_a0[sel].mean()) / q_a0[sel].mean()), "A1", m, n=int(sel.sum()))
            put("gruppy", "%s: %s — средний ln(RV/F_A1)" % (nm, gname), float(res1[sel].mean()), "A1", m, n=int(sel.sum()))
        put("gruppy", "%s: дат без терциля (история < 250)" % nm, int((lab < 0).sum()), "A1", m, n=n)

# ---------- применимость к M10 ----------
nb = F.n
yrs = pd.DatetimeIndex(F.dates).year
put("primenimost", "сессий в дневном ряду", len(F.dates), window="2014-06…2026-11-09")
for v_, c_ in pd.Series(nb[r]).value_counts().sort_index().items():
    put("primenimost", "дней D_M с %d барами" % v_, int(c_), n=n)
for y_ in sorted(set(yrs)):
    sel = yrs == y_
    put("primenimost", "доля сессий с 53 барами, год %d" % y_, float((nb[sel] == 53).mean()), window=str(y_), n=int(sel.sum()))
q8 = Q[:, 7]
put("primenimost", "доля дат D_M с q8 ≠ 1", float((np.abs(q8[r] - 1) > 1e-12).mean()), measure="q8", n=n, note="мера доступности почти не меняется")
q5 = Q[:, 4][r]
relse = np.sqrt(2 / 3 * q5)
for nm, v_ in (("медиана", 0.5), ("90-й процентиль", 0.9), ("99-й процентиль", 0.99)):
    put("primenimost", "q5, %s" % nm, float(np.quantile(q5, v_)), measure="q5", n=n)
    put("primenimost", "отн. ошибка RV √((2/3)·q5), %s" % nm, float(np.quantile(relse, v_)), measure="q5", n=n, note="BPQ: Var(RV − IV)/IV² ≈ (2/3)·Σr⁴/(Σr²)²")
put("primenimost", "n·q5, медиана (гаусс., равная дисперсия ⇒ 3)", float(np.median(nb[r] * q5)), measure="q5", n=n, note="больше 3 — внутридневная неоднородность и толстые хвосты")
jr = F.j[r] / F.rv[r]
lrv = np.log(F.rv[r])
sp = lambda a, b: float(pd.Series(a).rank().corr(pd.Series(b).rank()))
for m, z in Z.items():
    zd = z[r]
    put("primenimost", "Spearman(z, ln RV)", sp(zd, lrv), measure=m, n=n, note="связь с уровнем волатильности")
    put("primenimost", "Spearman(z, J/RV)", sp(zd, jr), measure=m, n=n, note="связь со скачком")
    top = zd >= np.quantile(zd, 0.8)
    put("primenimost", "средняя J/RV: верхний квинтиль z / остальные", "%.4f / %.4f" % (jr[top].mean(), jr[~top].mean()), measure=m, n=n, note="в выборке, только описание")
    put("primenimost", "средний ln RV: верхний квинтиль z / остальные", "%.4f / %.4f" % (lrv[top].mean(), lrv[~top].mean()), measure=m, n=n)
for i, nm in ((0, "q1 = ln(RV20/RV10)"), (1, "q2 = ln(RV30/RV10)"), (5, "q6 (первая автокорреляция)"), (3, "q4 (доля нулевых r)"), (2, "q3 (разброс RV30 по сдвигам)")):
    x = Q[:, i][r]
    put("primenimost", "%s: среднее" % nm, float(x.mean()), measure=QF.Q_NAMES[i], n=n)
    put("primenimost", "%s: СКО" % nm, float(x.std()), measure=QF.Q_NAMES[i], n=n)
    put("primenimost", "%s: Spearman с ln RV" % nm, sp(x, lrv), measure=QF.Q_NAMES[i], n=n)


def dropped_share(row_r2, m):
    """Средняя по сдвигам доля RV в барах, которые сетка m·10 минут отбрасывает (неполный хвост и начало при сдвиге) — как grid_rv V6."""
    x2 = row_r2[np.isfinite(row_r2)]
    tot, out = x2.sum(), []
    for s in range(m):
        k = (len(x2) - s) // m
        out.append((tot - x2[s:s + k * m].sum()) / tot)
    return float(np.mean(out))


for i, m_ in ((0, 2), (1, 3)):
    e = np.log(1 - np.array([dropped_share(F.r2[ri], m_) for ri in r]))
    x = Q[:, i][r]
    put("primenimost", "%s: ln(1 − доля RV в отброшенных краевых барах), среднее" % QF.Q_NAMES[i], float(e.mean()), measure=QF.Q_NAMES[i], n=n, note="при n = 53 сетки отбрасывают бар 10:00 (r = ln c/o) и/или 18:40")
    put("primenimost", "%s: корреляция с краевым членом" % QF.Q_NAMES[i], float(np.corrcoef(x, e)[0, 1]), measure=QF.Q_NAMES[i], n=n)
    put("primenimost", "%s: среднее за вычетом краевого члена" % QF.Q_NAMES[i], float((x - e).mean()), measure=QF.Q_NAMES[i], n=n, note="остаток ≈ перекрёстные члены (автокорреляция q6)")
put("primenimost", "Spearman(q1, q6)", sp(Q[:, 0][r], Q[:, 5][r]), measure="q1", n=n)
put("primenimost", "1/√n при n = 53 (сэмплинговое СКО q6 без шума, равная дисперсия)", float(1 / np.sqrt(53)), measure="q6", n=n, note="ориентир; фактическое СКО q6 — строка выше")
for lab_, sel in (("52 бара", nb == 52), ("53 бара", nb == 53), ("< 50 баров", nb < 50)):
    if sel.any():
        put("primenimost", "медиана n·q5, %s (все сессии)" % lab_, float(np.median(nb[sel] * Q[sel, 4])), measure="q5", window="все сессии", n=int(sel.sum()))

hdr = ["# S-B-DIAG (Ж-254 п. 7, 10): основание для модуля B. Критерий объявлен до счёта: scratchpad s_b/KRITERIJ.md, sha256 e2953ec7c1822823…, 11.11.2026 18:02:19",
       "# D_M: t⁺ 2018-01-04…2026-11-10, n = 2198; QLIKE = среднее дневных потерь; G — % QLIKE базы; оракул в выборке — потолок, только для решения «нет основания»; новых обучений гибрида нет",
       "# DEVELOPMENT / HISTORICAL; исследовательский результат на раскрытой истории, не OOS"]
with open(OUT, "w", encoding="utf-8") as f:
    f.write("\n".join(hdr) + "\n")
    pd.DataFrame(rows_out).to_csv(f, index=False, lineterminator="\n")
pd.set_option("display.width", 250, "display.max_colwidth", 70, "display.max_rows", 500)
print(pd.DataFrame(rows_out)[["block", "item", "base", "measure", "window", "n", "value"]].to_string())
