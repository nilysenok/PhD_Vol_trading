"""М0 v2: data/regen/features_pit_v2.parquet (+ .config.json с feature_cols) и таблица лагов всех признаков.
Собственные: rv, bv, jump, rsv_* из rv_1850 (ISS, срез 18:50), остальные — тот же калькулятор на свечах до 18:50,
лаги/цели — те же функции сборки (m0_pit_lib). Внешние — как v1; макро as-of + {ряд}_days_since_upd.
Формат: строки и колонки как у склейки data/prepared (+ новые), NaN признаков = медианы train <= 2017.
Выход: data/regen/features_pit_v2.parquet, data/regen/features_pit_v2.config.json,
docs/tables/prilozhenie_priznaki_lagi.csv, docs/otchety/m0/pit_v2_svodka.csv. Запуск: --rv1850 <путь> (по умолчанию ISS).
v3 (владелец 25.10, 20:40): --fill asof -> data/regen/features_pit_v3.parquet. Вместо медианы train <= 2017:
внешние признаки (EXT, по дате) протягиваются последним известным значением общего календаря на дату <= t (только прошлое),
строки акций вне календаря (до 26.08.2014 — NaN; выходные сессии 2025-2026) получают значение последнего дня календаря <= t;
*_days_since_upd — дни от фактического обновления, отражённого в значении (вне календаря: дни дня t_prev + (t - t_prev));
собственные признаки (OWN) — протягивание по строкам тикера (только прошлое). До первого наблюдения ряда — NaN
(строки не исключаются; в обучении clean_features переводит NaN в 0, как inf). Сайд-таблицы v2 в режиме asof не пишутся."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from m0_common import ROOT, OUT, REGEN, YAHOO, calendar, load_prepared, load_features_all, md_table
from m0_pit_lib import FC, OWN, EXT, MONTHLY, RV1850, MACRO_RAW, build_ext, external_inputs, own_features, assemble

ap = argparse.ArgumentParser()
ap.add_argument("--rv1850", default=str(RV1850))
ap.add_argument("--out", default=None)
ap.add_argument("--fill", choices=("median", "asof"), default="median")
ap.add_argument("--regen", default="/Users/nilysenok/Desktop/Старое/PhD_Vol_trading/data/regen")
a = ap.parse_args()
a.out = a.out or str(Path(a.regen) / ("features_pit_v2.parquet" if a.fill == "median" else "features_pit_v3.parquet"))
cal = calendar()
prep = load_prepared()
fa = load_features_all()

r18 = pd.read_parquet(a.rv1850)
r18["date"] = pd.to_datetime(r18["date"])
out, v2, NEWF, DAYS, lost, pre_nan = assemble(cal, prep, r18, a.fill)
out.to_parquet(a.out, index=False)
json.dump({"feature_cols": NEWF, "targets": ["rv_target_h1", "rv_target_h5", "rv_target_h22", "rv_target_mean_h1",
                                              "rv_target_mean_h5", "rv_target_mean_h22"]},
          open(a.out.replace(".parquet", ".config.json"), "w"), ensure_ascii=False, indent=1)
print("записан", a.out, "строк", len(out), flush=True)
if a.fill == "asof":
    raise SystemExit(0)


# --- таблица лагов: признак -> лаг в старом коде -> лаг в features_pit_v2 ---
def desc(c):
    hl = {"rv_d": "RV(t−1)", "rv_w": "среднее RV(t−5..t−1)", "rv_2w": "среднее RV(t−10..t−1)",
          "rv_m": "среднее RV(t−22..t−1)", "rv_q": "среднее RV(t−66..t−1)"}
    if c in hl:
        return "собств. лаг", "intraday_features.py:354-386", hl[c] + ", RV по всем барам", hl[c] + ", RV по срезу 18:50"
    if c.startswith("jump_lag"):
        k = c[8:]
        return "собств. лаг", "intraday_features.py:390-418", f"J(t−{k}), все бары", f"J(t−{k}), срез 18:50"
    if c in ("rv_morning", "rv_midday", "rv_evening"):
        return ("собств.", "intraday_features.py:129-155", "день t, своё окно ≤ 18:50; 1-я доходность от закрытия ~23:40",
                "день t, то же окно; 1-я доходность от закрытия 18:50")
    if c in OWN:
        src = "ISS rv_1850" if c in ("rv", "bv", "jump", "rsv_pos", "rsv_neg", "rv_annualized", "bv_annualized",
                                     "jump_annualized") else "intraday_features.py:95-203"
        return "собств.", src, "день t, все бары (с 22.06.2020 до 23:40)", "день t, бары до 18:50"
    if c.endswith("_days_since_upd"):
        return "макро (новый)", "m0_pit_lib.macro_pit", "—", "дней с последнего обновления (по дате доступности)"
    base = c.replace("yahoo_", "")
    for y in YAHOO:
        if base == y or base.startswith(y + "_"):
            if base.endswith("_lag1"):
                return "Yahoo", YAHOO[y], "закрытие США t−1", "закрытие США t−2"
            return "Yahoo", YAHOO[y], "закрытие США дня t (лаг 0)", "последнее закрытие США до t (лаг 1)"
    for m in MACRO_RAW:
        if c == m or c.startswith(m + "_"):
            sh = {"lag1": " (+1 торг. день)", "diff1": " (разность за 1 торг. день)", "diff22": " (разность за 22)"}.get(
                c[len(m) + 1:], "")
            if m in MONTHLY:
                return ("макро", MACRO_RAW[m], "с метки периода, лаг 0; метки в неторговые дни теряются" + sh,
                        "as-of: конец периода + 30 дн (даты публикации нет — пометка)" + sh)
            o, n = {"key_rate_cbr": ("дата вступления, лаг 0", "дата вступления, лаг 0 (as-of)"),
                    "fed_rate": ("DFF дня t, лаг 0", "DFF до t, лаг 1"),
                    "si_fut": ("закрытие 23:50 дня t, лаг 0", "закрытие t−1, лаг 1")}[m]
            return "макро", MACRO_RAW[m], o + sh, n + sh
    t = next(t for t in ("CNYRUB_TOM", "RVI", "RGBI", "MOEXMM", "IMOEX", "MOEXTL", "MOEXEU", "MOEXCH", "MOEXFN",
                         "MOEXCN", "MOEXOG") if t in c)
    sfx = " (по t−1)" if c.endswith("_lag1") else ""
    if t in ("CNYRUB_TOM", "RVI", "RGBI"):
        return "индекс MOEX", "rv_moex.parquet", f"день t, бары до {'19:00' if t == 'RGBI' else '23:50'}{sfx}", \
            f"день t, бары до 18:50{sfx}"
    return "индекс MOEX", "rv_moex.parquet", f"день t, 18:50{sfx}", f"день t, 18:50{sfx} (без изменений)"


rows = []
for c in NEWF + ["rv_target_h1", "rv_target_h5", "rv_target_h22", "rv_target_mean_h5", "rv_target_mean_h22"]:
    if c.startswith("rv_target"):
        h = c.split("_h")[-1]
        g, s = "цель", "00_prepare_data.py:47-48"
        o = f"RV(t+{h}) по строкам тикера, все бары" if "mean" not in c else "—"
        n = (f"RV(t+{h}) по строкам, срез 18:50 (как в коде)" if "mean" not in c else
             f"среднее RV(t+1..t+{h}), срез 18:50 (как в тексте)")
    else:
        g, s, o, n = desc(c)
    ch = (float(((out[c] - prep[c]).abs() > 1e-9 * (1 + prep[c].abs())).mean())
          if c in prep.columns and c in FC + ["rv_target_h1", "rv_target_h5", "rv_target_h22"] else np.nan)
    rows.append({"признак": c, "группа": g, "источник": s, "лаг_старый_код": o, "лаг_features_pit_v2": n,
                 "доля_строк_изменено": round(ch, 4) if ch == ch else ""})
lt = pd.DataFrame(rows)
(ROOT / "docs/tables").mkdir(parents=True, exist_ok=True)
lt.to_csv(ROOT / "docs/tables/prilozhenie_priznaki_lagi.csv", index=False)

# --- сводка ---
old_nan = fa[[c for c in FC if c in MACRO_RAW]].isna().mean()
sv = pd.DataFrame([{"ряд": m, "потеряно_периодов_до": lost[m][0] if m in MONTHLY else "дневной",
                    "потеряно_периодов_после": lost[m][1] if m in MONTHLY else "дневной",
                    "NaN_до_%": round(100 * old_nan[m], 2), "NaN_после_%": round(100 * pre_nan[m], 2)} for m in MACRO_RAW])
sv.to_csv(OUT / "pit_v2_svodka.csv", index=False)
print(md_table(sv))
grp = lt.groupby("группа").agg(признаков=("признак", "size"))
chg = lt[lt["доля_строк_изменено"] != ""].assign(x=lambda d: d["доля_строк_изменено"].astype(float) > 0)
print(grp.to_dict()["признаков"], "| изменено признаков:", int(chg.x.sum()), "из", len(chg),
      "| строк:", len(out), "| признаков в config v2:", len(NEWF))
# --- сверка собственных признаков старые/новые до и после 22.06.2020 (до заполнения медианами) ---
fo = fa.merge(prep[["date", "ticker", "rv_target_h1", "rv_target_h5", "rv_target_h22"]], on=["date", "ticker"])
cmp = v2.merge(fo, on=["date", "ticker"], suffixes=("", "_old"))
rr = []
for c in ["rv", "bv", "jump", "rsv_pos", "rsv_neg", "rv_morning", "rv_evening", "rskew", "close", "volume", "rv_d",
          "rv_w", "rv_m", "rv_q", "jump_lag1", "rv_target_h1", "rv_target_h5", "rv_target_h22"]:
    for lab, m in (("до", cmp.date < "2020-06-22"), ("с", cmp.date >= "2020-06-22")):
        x, y = cmp.loc[m, c], cmp.loc[m, f"{c}_old"]
        ok = x.notna() & y.notna() & (y != 0)
        rr.append({"признак": c, "период": lab, "строк": int(ok.sum()),
                   "совпало_%": round(100 * float(np.isclose(x[ok], y[ok], rtol=1e-6, atol=1e-14).mean()), 2),
                   "медиана_новый/старый": round(float((x[ok] / y[ok]).median()), 4)})
rc = pd.DataFrame(rr)
rc.to_csv(OUT / "pit_v2_sverka_rv.csv", index=False)
w = rc.pivot(index="признак", columns="период", values=["совпало_%", "медиана_новый/старый"])
w.columns = [f"{a} {b} 22.06.2020" for a, b in w.columns]
print(md_table(w.reindex(rc.признак.unique()).reset_index()))
both = out[["rv_target_h5", "rv_target_mean_h5"]].dropna()
print("цель h1: доля строк изменена", lt.set_index("признак").loc["rv_target_h1", "доля_строк_изменено"],
      "| corr ln(h5 код, h5 среднее):", round(np.corrcoef(np.log(both.values.T + 1e-10))[0, 1], 3))
