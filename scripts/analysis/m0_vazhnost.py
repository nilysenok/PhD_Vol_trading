"""М0 п.6: gain-важность бустинга по h = 1, 5, 22 (LightGBM П1 = gbdt + ранняя остановка 50, XGBoost как в regen_boosting.py).
Схема шагов — regen_boosting.py (annual, валидация = последний год обучающего окна). Топ-20 — шаг 2025 (val 2024).
Нулевой gain — сумма total gain по всем 9 шагам 2017-2025 (валидационные годы 2016-2024), отдельно для каждой модели.
Сверка: прогнозы шага 2025 против сохранённых data/regen/P1_h*_test.parquet и xgb_h*_test.parquet (основная копия).
Плюс gain сохранённых моделей moex_volatility/models/{lightgbm,xgboost}/model_h*.
Потоки: n_jobs=2. Выход: docs/otchety/m0/vazhnost_*.csv, stdout."""
import json
import os
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from m0_common import MAIN, MV, OUT, feature_cols, load_prepared, qlike

os.environ.setdefault("OMP_NUM_THREADS", "2")
FC = feature_cols()
OWN = FC[:23]
GROUP = {c: ("лаги RV" if c in ("rv", "rv_annualized", "rv_d", "rv_w", "rv_2w", "rv_m", "rv_q", "rsv_pos", "rsv_neg",
                                "rv_morning", "rv_midday", "rv_evening", "bv", "bv_annualized")
             else "скачки" if "jump" in c else "собств. прочие" if c in OWN
             else "Yahoo (глоб.)" if c.startswith("yahoo_") or c in ("brent", "vix", "sp500", "gold", "dxy", "eem", "wti",
                                                                      "tbill_3m", "treasury_10y", "treasury_5y")
             else "индексы MOEX" if ("_" in c and (c.startswith(("rv_", "close_", "idx_"))))
             else "макро") for c in FC}

data = load_prepared().sort_values("date").reset_index(drop=True)
data["year"] = pd.to_datetime(data.date).dt.year
X_all = np.nan_to_num(data[FC].values, nan=0, posinf=0, neginf=0)  # = clean_features из walk_forward_full.py


def fit_step(h, ty):
    target = f"rv_target_h{h}"
    tr = data.year < ty
    vy = data.loc[tr, "year"].max()
    ok = data[target].notna()
    itp, ivi, itd = tr & (data.year != vy) & ok, tr & (data.year == vy) & ok, (data.year == ty) & ok
    y = np.log(data[target].values + 1e-10)
    lp = json.load(open(MV / f"models/lightgbm/params_h{h}.json"))
    lp.update(verbosity=-1, random_state=42, n_jobs=2, boosting_type="gbdt")
    m1 = lgb.LGBMRegressor(**lp).fit(X_all[itp], y[itp], eval_set=[(X_all[ivi], y[ivi])], eval_metric="l2",
                                     callbacks=[lgb.early_stopping(50, verbose=False)])
    g1 = m1.booster_.feature_importance("gain", iteration=m1.best_iteration_)
    xp = json.load(open(MV / f"models/xgboost/params_h{h}.json"))
    xp.update(tree_method="hist", verbosity=0, random_state=42, n_jobs=2, early_stopping_rounds=50)
    m2 = xgb.XGBRegressor(**xp).fit(X_all[itp], y[itp], eval_set=[(X_all[ivi], y[ivi])], verbose=False)
    sc = m2.get_booster().get_score(importance_type="total_gain")
    g2 = np.array([sc.get(f"f{i}", 0.0) for i in range(len(FC))])
    pred = {"P1": np.exp(m1.predict(X_all[itd])), "xgb": np.exp(m2.predict(X_all[itd]))}
    return g1, g2, pred, data.loc[itd, ["date", "ticker", target]]


def top(g, k=20):
    s = pd.Series(g, index=FC)
    s = 100 * s / s.sum()
    t = s.sort_values(ascending=False).head(k)
    return pd.DataFrame({"признак": t.index, "gain_%": t.values.round(2), "группа": [GROUP[c] for c in t.index]})


for h in (1, 5, 22):
    tot = {"P1": np.zeros(len(FC)), "xgb": np.zeros(len(FC))}
    for ty in range(2017, 2026):
        g1, g2, pred, td = fit_step(h, ty)
        tot["P1"] += g1
        tot["xgb"] += g2
    # шаг 2025 = последний ty; g1, g2, pred относятся к нему
    for name, g in (("P1", g1), ("xgb", g2)):
        t = top(g)
        t.to_csv(OUT / f"vazhnost_{name}_h{h}_2025.csv", index=False)
        share = pd.Series(g, index=FC).groupby(pd.Series(GROUP)).sum()
        share = (100 * share / share.sum()).round(1).sort_values(ascending=False)
        ref = pd.read_parquet(MAIN / f"data/regen/{name}_h{h}_test.parquet")
        ref = ref[ref.year == 2025].merge(td.assign(p=pred[name])[["date", "ticker", "p"]], on=["date", "ticker"])
        print(f"\n### h={h} {name}, шаг 2025: сверка с сохранённым прогнозом: строк {len(ref)}, "
              f"макс |отн. разн.| {np.max(np.abs(ref.p / ref.rv_pred - 1)):.2e}, QLIKE наш {qlike(ref.rv_actual.values, ref.p.values):.4f} "
              f"/ сохр. {qlike(ref.rv_actual.values, ref.rv_pred.values):.4f}")
        print("доли gain по группам, %:", share.to_dict())
        print(t.to_string(index=False))
    zero = {k: [FC[i] for i in np.where(v == 0)[0]] for k, v in tot.items()}
    both = sorted(set(zero["P1"]) & set(zero["xgb"]), key=FC.index)
    pd.DataFrame({"признак": FC, "gain_P1_2017_2025": tot["P1"], "gain_xgb_2017_2025": tot["xgb"]}).to_csv(
        OUT / f"vazhnost_sum_h{h}.csv", index=False)
    print(f"\nh={h}: нулевой суммарный gain (9 шагов 2017-2025): P1 {len(zero['P1'])}, xgb {len(zero['xgb'])}, оба {len(both)}")
    print("оба:", both)
    print("только P1:", [c for c in zero["P1"] if c not in both])
    print("только xgb:", [c for c in zero["xgb"] if c not in both])

# сохранённые модели (обучены на фикс. разбиении, см. models/*/params)
for h in (1, 5, 22):
    b = lgb.Booster(model_file=str(MV / f"models/lightgbm/model_h{h}.txt"))
    gl = pd.Series(b.feature_importance("gain"), index=b.feature_name())
    bx = xgb.Booster()
    bx.load_model(str(MV / f"models/xgboost/model_h{h}.json"))
    gx = pd.Series(bx.get_score(importance_type="total_gain"))
    for nm, s in (("lightgbm", gl), ("xgboost", gx)):
        s = (100 * s / s.sum()).sort_values(ascending=False).head(10).round(1)
        print(f"сохранённая {nm} h={h}, топ-10 gain %: {s.to_dict()}")
