"""М0 п.2: эмпирическая проверка утечки по времени.
T1: доходность ряда дня t (как в матрице) против доходности IMOEX t+1 (закрытие 18:50), лаг 0 и 1.
T2: прирост R^2 в регрессии ln RV(t+1) (среднее по 17 акциям) на HAR(ln rv_t, ln rv_w, ln rv_m) + |ret ряда|, лаг 0 и 1.
T3: скачки S&P 500 (верхний 1% |ret|): реакция IMOEX в день t и t+1.
T4: CNYRUB_TOM и RVI: признак с вечерней сессией (как в матрице) против среза 18:50.
Выход: docs/otchety/m0/utechka_emp.csv, stdout."""
import numpy as np
import pandas as pd
from m0_common import DF, OUT, calendar, load_features_all, rv_moex_pit, md_table

cal = calendar()
fa = load_features_all()
fa = fa[fa.date.isin(cal)]
lr = np.log(fa["rv"].clip(lower=1e-12))
panel = fa.assign(lrv=lr, lw=np.log(fa.rv_w.clip(lower=1e-12)), lm=np.log(fa.rv_m.clip(lower=1e-12)))
day = panel.groupby("date")[["lrv", "lw", "lm"]].mean().reindex(cal)
day["y1"] = day["lrv"].shift(-1)  # ln RV следующего торгового дня (среднее по акциям)
sb = fa[fa.ticker == "SBER"].set_index("date").reindex(cal)

rvm = pd.read_parquet(DF / "rv_moex.parquet")
rvm["date"] = pd.to_datetime(rvm.date)
im = rvm[rvm.ticker == "IMOEX"].set_index("date").reindex(cal)
r_im = np.log(im.close).diff()
good = (im.n_bars >= 30) & (im.n_bars.shift(1) >= 30)  # без неполных дней IMOEX
r_im1 = r_im.shift(-1).where(good.shift(-1, fill_value=False))
from m0_common import index_bars
_b = index_bars("IMOEX")
_first = _b[_b.begin.dt.time == pd.Timestamp("10:00").time()].set_index("date")["close"].reindex(cal)
gap1 = (np.log(_first) - np.log(im.close).shift(1)).shift(-1).where(good.shift(-1, fill_value=False))  # гэп 18:50 t -> 10:10 t+1


def ols_r2(X, y, nbase=3):
    """R^2 полной модели, t последнего коэффициента, n; dR2 считается против базы на той же выборке."""
    m = np.isfinite(X).all(1) & np.isfinite(y)
    rb = _r2(X[m][:, :nbase], y[m])
    X, y = np.column_stack([np.ones(m.sum()), X[m]]), y[m]
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    e = y - X @ b
    s2 = e @ e / (len(y) - X.shape[1])
    se = np.sqrt(np.diag(s2 * np.linalg.inv(X.T @ X)))
    return 1 - e.var() / y.var() - rb, b[-1] / se[-1], m.sum()


def _r2(X, y):
    X = np.column_stack([np.ones(len(y)), X])
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    return 1 - (y - X @ b).var() / y.var()


base = day[["lrv", "lw", "lm"]].values
m0 = np.isfinite(base).all(1) & np.isfinite(day.y1.values)
r2_base = _r2(base[m0], day.y1.values[m0])
rows = []
series = {f"yahoo_{c}_ret1": c for c in ["sp500", "vix", "eem", "dxy", "brent", "wti", "gold", "treasury_10y"]}
series.update({"si_fut_diff1": "si_fut", "idx_RVI_rv_annualized_ret1": "RVI", "idx_CNYRUB_TOM_rv_annualized_ret1": "CNYRUB_TOM",
               "idx_IMOEX_rv_annualized_ret1": "IMOEX (контроль, 18:50)"})
for col, name in series.items():
    x = sb[col]
    if col == "si_fut_diff1":
        x = np.log(sb["si_fut"]).diff()
    out = {"ряд": name, "признак": col}
    for k in (0, 1):
        xk = x.shift(k)
        out[f"corr_ret_IMOEX_t+1_лаг{k}"] = round(pd.concat([xk, r_im1], axis=1).dropna().corr().iloc[0, 1], 3)
        out[f"corr_гэп_IMOEX_t+1_лаг{k}"] = round(pd.concat([xk, gap1], axis=1).dropna().corr().iloc[0, 1], 3)
        r2, t, n = ols_r2(np.column_stack([base, xk.abs().values]), day.y1.values)
        out[f"dR2_лаг{k}_%"] = round(100 * r2, 3)
        out[f"t_лаг{k}"] = round(t, 1)
    rows.append(out)

# T4: CNYRUB_TOM / RVI — вечер против среза 18:50 (лаг 0)
pit = rv_moex_pit()
for t in ["RVI", "CNYRUB_TOM"]:
    a = rvm[rvm.ticker == t].set_index("date").reindex(cal)
    c = pit[pit.ticker == t].set_index("date").reindex(cal)
    for lab, s in (("как в матрице (до 23:50)", a), ("срез 18:50", c)):
        ret = np.log(s.close).diff()
        lrvx = np.log(s.rv_annualized.clip(lower=1e-6))
        out = {"ряд": f"{t} {lab}", "признак": f"ret close_{t}; ln rv_{t}"}
        out["corr_ret_IMOEX_t+1_лаг0"] = round(pd.concat([ret, r_im1], axis=1).dropna().corr().iloc[0, 1], 3)
        r2, tt, n = ols_r2(np.column_stack([base, lrvx.values]), day.y1.values)
        out["dR2_лаг0_%"] = round(100 * r2, 3)
        out["t_лаг0"] = round(tt, 1)
        rows.append(out)
    ev = np.log(a.close) - np.log(c.close)  # вечернее движение 18:50 -> 23:50 дня t
    m = (a.close != c.close)
    print(f"{t}: дней с вечерними барами {int(m.sum())}; corr(вечернее движение t, гэп IMOEX t+1) = "
          f"{pd.concat([ev[m], gap1[m]], axis=1).dropna().corr().iloc[0, 1]:.3f}")

res = pd.DataFrame(rows)
res.to_csv(OUT / "utechka_emp.csv", index=False)
print(f"R^2 базовой HAR (ln RV t+1, среднее по акциям) = {r2_base:.4f}; дней {int(day.y1.notna().sum())}")
print(md_table(res.fillna("")))

# T3: скачки S&P 500
sp = sb["yahoo_sp500_ret1"]
us_days = sb["sp500"].diff() != 0  # в этот день MOEX есть новое закрытие США (не протянуто)
thr = sp[us_days].abs().quantile(0.99)
big = sp[us_days & (sp.abs() >= thr)].index
same = pd.concat([sp[big], r_im[big]], axis=1).dropna()
nxt = pd.concat([sp[big], gap1[big]], axis=1).dropna()
print(f"\nT3: скачков S&P (|ret|>={thr:.3%}) {len(big)}. Совпадение знака с IMOEX: день t "
      f"{(np.sign(same.iloc[:, 0]) == np.sign(same.iloc[:, 1])).mean():.0%} (n={len(same)}), "
      f"ночной гэп t->t+1 {(np.sign(nxt.iloc[:, 0]) == np.sign(nxt.iloc[:, 1])).mean():.0%} (n={len(nxt)}); "
      f"corr t {same.corr().iloc[0, 1]:.2f}, t+1 {nxt.corr().iloc[0, 1]:.2f}")
yv = day.y1.reindex(big) - day.lrv.reindex(big)
print(f"   ln RV(t+1) - ln RV(t) в эти дни: среднее {yv.mean():+.3f} (по всем дням {(day.y1 - day.lrv).mean():+.3f})")
