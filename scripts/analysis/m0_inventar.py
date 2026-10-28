"""М0 п.1, 2 (код), 4: инвентаризация внешних рядов, фактический лаг в матрице, доля протянутых значений, IMOEX.
Выход: docs/otchety/m0/inventar.csv, docs/otchety/m0/imoex.csv; таблицы в stdout (markdown)."""
import datetime as dt
import pandas as pd
from m0_common import (DF, INDICES, YAHOO, MACRO_RAW, ISS_IMOEX, OUT, calendar, load_features_all, yahoo_raw,
                       macro_raw, index_bars, md_table)

OUT.mkdir(parents=True, exist_ok=True)
cal = calendar()
fa = load_features_all()
sb = fa[fa.ticker == "SBER"].set_index("date").sort_index()  # внешние признаки одинаковы для всех тикеров
rows = []


def code_lag(col, src):
    """Сдвиг k (в днях календаря MOEX), при котором признак в features_all совпадает с источником."""
    s = sb[col].reindex(cal)
    for k in (0, 1, 2):
        sh = src.reindex(cal).shift(k)
        ok = (abs(s - sh) <= 1e-9 * (1 + abs(sh))) | (s.isna() & sh.isna())
        if ok[s.notna()].mean() > 0.99:
            return k, ok[s.notna()].mean()
    return None, 0.0


def nan_share(col):
    return fa[col].isna().mean()  # в prepared эти строки заполнены медианой train


# --- индексы MOEX (11 рядов × 9 признаков) ---
rvm = pd.read_parquet(DF / "rv_moex.parquet")
rvm["date"] = pd.to_datetime(rvm.date)
avail = {"CNYRUB_TOM": "вечерняя сессия: последний бар до 23:40–23:50", "RVI": "до 23:50",
         "RGBI": "до 18:40–19:00"}
for t in INDICES:
    x = rvm[rvm.ticker == t].set_index("date").sort_index()
    b = index_bars(t)
    late = b[b.begin.dt.time > dt.time(18, 50)]
    nb = b.groupby("date").size()
    k, share = code_lag(f"rv_{t}", x["rv_annualized"])
    rows.append(dict(
        ряд=t, группа="индекс MOEX", признаков=9, источник="MOEX ISS 10 мин, create_external_final.py:190-221",
        файл="dataset_final/02_external/rv_moex.parquet", частота="дневная (из 10-мин баров)",
        время_t=avail.get(t, "закрытие 18:50 (бар 18:50 до ~18:54)"),
        после_1850=f"{late.date.nunique()} дн. с барами >18:50 (с {late.date.min().date() if len(late) else '—'})",
        протяжка="source=filled/patched: ffill/дозаполнение; нет дня → NaN → медиана train",
        лаг_в_коде=k, совпадение=round(share, 4),
        доля_протянуто=round((x.source != "original").sum() / len(cal) + (1 - len(x) / len(cal)), 4),
        доля_NaN_в_матрице=round(nan_share(f"rv_{t}"), 4), дней_lt30_баров=int((nb < 30).sum())))

# --- Yahoo (10 рядов × 8 признаков) ---
ytime = {"brent": "ICE BZ=F: расчёт 19:30 Лондон (21:30–22:30 МСК), торги до 01:00 МСК",
         "gold": "COMEX GC=F: расчёт 13:30 ET (20:30–21:30 МСК), торги до 00:00–01:00 МСК",
         "wti": "NYMEX CL=F: расчёт 14:30 ET (21:30–22:30 МСК)", "vix": "CBOE 16:15 ET (23:15–00:15 МСК)",
         "dxy": "ICE 17:00 ET (00:00–01:00 МСК)"}
yd = pd.read_parquet(DF / "yahoo.parquet")
yd["date"] = pd.to_datetime(yd.date)
yd = yd.set_index("date")
for c in YAHOO:
    r = yahoo_raw(c)
    k, share = code_lag(c, yd[c])
    k_raw, _ = code_lag(c, r.set_index("date")["close"].reindex(cal).ffill())
    rows.append(dict(
        ряд=c, группа="Yahoo", признаков=8, источник=f"Yahoo Finance {YAHOO[c]}, download_yahoo_01.py",
        файл="dataset_final/02_external/yahoo.parquet", частота="дневная (дата биржи США)",
        время_t=ytime.get(c, "закрытие NYSE 16:00 ET (23:00–00:00 МСК)"), после_1850="все дни",
        протяжка="ffill на календарь MOEX (align_yahoo_to_common.py:41-46), bfill начала",
        лаг_в_коде=k_raw, совпадение=round(share, 4),
        доля_протянуто=round(1 - cal.isin(r.date).mean(), 4), доля_NaN_в_матрице=round(nan_share(c), 4),
        дней_lt30_баров=""))

# --- макро (8 рядов × 4 признака) ---
md = pd.read_parquet(DF / "macro.parquet")
md["date"] = pd.to_datetime(md.date)
md = md.set_index("date")
mtime = {"cpi_ru": ("месячная, метка = 1-е число отчётного месяца", "Росстат ~10–15 числа след. месяца, 19:00"),
         "cpi_us": ("месячная, метка = 1-е число", "BLS ~10–15 числа след. месяца (2025: до 18.12)"),
         "unemp_us": ("месячная, метка = 1-е число", "BLS 1-я пятница след. месяца"),
         "m2_ru": ("месячная, метка = последний пн месяца", "ЦБ, след. месяц; даты публикации нет"),
         "pmi_ru": ("месячная, метка = последний день месяца", "S&P Global, 1-й раб. день след. месяца"),
         "key_rate_cbr": ("событийная, метка = дата вступления в силу", "решение 13:30 МСК в пт, действует с пн"),
         "fed_rate": ("дневная (FRED DFF)", "DFF дня t публикуется t+1 ~ 16:00 МСК"),
         "si_fut": ("дневная (TradingView 4H)", "дневное закрытие = конец вечерней сессии 23:50")}
for c in MACRO_RAW:
    r = macro_raw(c)
    k, share = code_lag(c, md[c])
    # фактическая схема: точное соединение метки с днём MOEX, затем ffill (метка в неторговый день теряется)
    k_raw, sh_raw = code_lag(c, r[r.date.isin(cal)].set_index("date")["value"].reindex(cal).ffill())
    rr = r[(r.date >= cal[0]) & (r.date <= cal[-1])]
    lost = 1 - rr.date.isin(cal).mean()
    rows.append(dict(
        ряд=c, группа="макро", признаков=4, источник=f"raw/{MACRO_RAW[c]}.parquet (см. macro_processing_report.md)",
        файл="dataset_final/02_external/macro.parquet", частота=mtime[c][0], время_t=mtime[c][1], после_1850="",
        протяжка=f"точное соединение по дате + ffill; потеряно меток {lost:.0%}", лаг_в_коде=k_raw, совпадение=round(sh_raw, 4),
        доля_протянуто=round(1 - cal.isin(r.date).mean(), 4), доля_NaN_в_матрице=round(nan_share(c), 4),
        дней_lt30_баров=""))

inv = pd.DataFrame(rows)
inv.to_csv(OUT / "inventar.csv", index=False)
print(md_table(inv.drop(columns=["источник", "файл"])))
print("рядов:", len(inv), "внешних признаков:", inv["признаков"].sum())

# --- IMOEX: неполные дни и сверка с дневным ISS ---
b = index_bars("IMOEX")
nb = b.groupby("date").size()
low = nb[nb < 30]
iss = pd.read_parquet(ISS_IMOEX)
iss["date"] = pd.to_datetime(iss.date)
x = rvm[rvm.ticker == "IMOEX"].set_index("date")
j = x[["close", "rv_annualized", "n_bars"]].join(iss.set_index("date")[["close"]], rsuffix="_iss", how="inner")
j["rel"] = (j.close / j.close_iss - 1).abs()
j["low"] = j.n_bars < 30
pd.DataFrame({"date": low.index, "n_bars": low.values}).to_csv(OUT / "imoex_low_days.csv", index=False)
print(f"\nIMOEX: дней {len(nb)}, из них < 30 баров {len(low)} ({len(low)/len(nb):.1%}); по годам "
      f"{low.groupby(low.index.year).size().to_dict()}")
print(f"с 03.01.2022: дней {(nb.index >= '2022-01-03').sum()}, < 30 баров {(low.index >= '2022-01-03').sum()}")
print(f"close_IMOEX vs ISS (общих {len(j)}): расх >1e-6: {(j.rel > 1e-6).sum()}, из них в днях <30 баров "
      f"{((j.rel > 1e-6) & j.low).sum()}; макс {j.rel.max():.3%}")
print("rv_IMOEX медиана: полные дни", round(j.loc[~j.low, 'rv_annualized'].median(), 4), "| неполные",
      round(j.loc[j.low, 'rv_annualized'].median(), 4))
print("время последнего бара в неполные дни:", b[b.date.isin(low.index)].groupby("date").begin.max().dt.time
      .value_counts().head(3).to_dict())
