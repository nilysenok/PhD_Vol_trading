"""М0 п.3: тест цели rv_target_h* и лаговых признаков на синтетике через тот же код построения.
Цель по спецификации: mean RV за [t+1, t+h] (в модели берётся ln). Признаки — только по t включительно.
Код цели: moex_volatility/results/final/code/00_prepare_data.py:47-48 (исполняется тот же оператор из исходника).
Код лагов: moex_volatility/src/data/intraday_features.py:336-418 (add_har_lags, add_extended_lags; polars, как в сборке)."""
import ast
import sys
import numpy as np
import pandas as pd
from m0_common import MV, load_prepared, calendar

sys.path.insert(0, str(MV))
from src.data.intraday_features import IntradayFeatureCalculator  # noqa: E402

PREP_SRC = MV / "results/final/code/00_prepare_data.py"
loop = next(n for n in ast.walk(ast.parse(PREP_SRC.read_text()))
            if isinstance(n, ast.For) and "rv_target_h" in ast.unparse(n))
print("оператор цели из исходника:", ast.unparse(loop).replace("\n", " "))


def build_target(df):
    ns = {"df": df}
    exec(compile(ast.Module([loop], []), str(PREP_SRC), "exec"), ns)
    return ns["df"]


def build_lags(df):
    import polars as pl
    calc = IntradayFeatureCalculator.__new__(IntradayFeatureCalculator)
    calc.use_polars = True
    out = calc.add_extended_lags(calc.add_har_lags(pl.from_pandas(df)), columns=["jump"], lags=[1, 5, 22])
    return out.to_pandas()


# синтетика: 2 тикера × 60 дней, rv_t = t (+1000 у B), jump_t = 10 t; у B выброшен день 30 (пропуск строки)
dates = pd.bdate_range("2020-01-01", periods=60)
rows = []
for k, tk in enumerate(["A", "B"]):
    for i, d in enumerate(dates):
        if tk == "B" and i == 30:
            continue
        rows.append({"date": d, "ticker": tk, "i": i, "rv": float(i + 1000 * k), "jump": 10.0 * i})
syn = pd.DataFrame(rows).sort_values(["ticker", "date"]).reset_index(drop=True)
res = build_target(build_lags(syn.copy())).sort_values(["ticker", "date"]).reset_index(drop=True)

ok_all = True
for h in (1, 5, 22):
    a = res[res.ticker == "A"].reset_index(drop=True)
    exp = a.rv[::-1].rolling(h).mean()[::-1].shift(-1)  # mean RV(t+1..t+h)
    got = a[f"rv_target_h{h}"]
    m = exp.notna() & got.notna()
    ok = bool(np.allclose(got[m], exp[m]))
    ok_all &= ok
    print(f"h={h}: цель = mean RV[t+1,t+h]? {'ПРОХОДИТ' if ok else 'НЕ ПРОХОДИТ'}; при t=10: код {got[10]:.1f}, "
          f"ожидалось {exp[10]:.1f} (код = RV(t+{h}) = {a.rv[10 + h]:.1f})")

b = res[res.ticker == "B"].reset_index(drop=True)
r = b[b.i == 29].iloc[0]
print(f"пропуск строки (B, день 30 выброшен): при t=29 rv_target_h1 = {r.rv_target_h1:.0f} = RV дня 31, "
      f"а не дня 30 -> сдвиг по строкам, не по календарю")

# признаки только по t включительно: возмущаем будущее и проверяем неизменность признаков
cols = ["rv_d", "rv_w", "rv_2w", "rv_m", "rv_q", "jump_lag1", "jump_lag5", "jump_lag22"]
base = build_lags(syn.copy()).sort_values(["ticker", "date"]).reset_index(drop=True)
leak = 0
for t0 in (10, 25, 40):
    s2 = syn.copy()
    s2.loc[s2.i > t0, ["rv", "jump"]] *= 7.0
    p = build_lags(s2).sort_values(["ticker", "date"]).reset_index(drop=True)
    m = base.i <= t0
    leak += int((~np.isclose(base.loc[m, cols], p.loc[m, cols], equal_nan=True)).sum())
a = base[base.ticker == "A"].reset_index(drop=True)
lag_ok = (np.allclose(a.rv_d[1:], a.rv.shift(1)[1:]) and np.allclose(a.rv_w[5:], a.rv.shift(1).rolling(5).mean()[5:])
          and np.allclose(a.rv_m[22:], a.rv.shift(1).rolling(22).mean()[22:]))
print(f"лаги: возмущение будущего меняет признаки прошлого в {leak} ячейках; rv_d=RV(t-1), rv_w=mean RV(t-5..t-1), "
      f"rv_m=mean RV(t-22..t-1): {'ПРОХОДИТ' if (leak == 0 and lag_ok) else 'НЕ ПРОХОДИТ'}")

# реальная матрица: цель совпадает с RV(t+h) по строкам; сколько строк через пропуск календаря
pr = load_prepared().sort_values(["ticker", "date"])
cal = pd.Series(np.arange(len(calendar())), index=calendar())
pos = pr.date.map(cal)
for h in (1, 5, 22):
    pt = pr.groupby("ticker")["rv"].shift(-h)
    mean_t = pr.groupby("ticker")["rv"].transform(lambda x: x[::-1].rolling(h).mean()[::-1].shift(-1))
    nxt = pr.groupby("ticker")["date"].shift(-h).map(cal)
    gap = (nxt - pos) != h
    tgt = pr[f"rv_target_h{h}"]
    m = tgt.notna()
    mc = m & pos.notna() & nxt.notna()  # только дни общего календаря (с 26.08.2014)
    print(f"матрица h={h}: target==RV(t+h) по строкам {np.isclose(tgt[m], pt[m]).mean():.4f}; "
          f"==mean[t+1,t+h] {np.isclose(tgt[m], mean_t[m]).mean():.4f}; строк через пропуск календаря {gap[mc].mean():.2%} ({int(gap[mc].sum())}); "
          f"corr(ln точечная, ln среднее) {np.corrcoef(np.log(tgt[m & mean_t.notna()] + 1e-10), np.log(mean_t[m & mean_t.notna()] + 1e-10))[0, 1]:.3f}")
