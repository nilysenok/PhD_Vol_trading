"""М0 п.5: матрица признаков point-in-time (срез 18:50 МСК дня t) — data/regen/features_pit.parquet.
Формат: все строки train+val+test+walkforward_* (как их склеивают walk_forward_full.py/regen_boosting.py),
те же 262 колонки в том же порядке, NaN = медианы train (<=2017, как 00_prepare_data.py:69-78);
доп. колонки (не в feature_cols): imoex_low_bars, rv_target_mean_h5, rv_target_mean_h22.
Внешние признаки пересобраны тем же ExternalFeatureBuilder (src/data/external_features.py, ветка pandas)
из исправленных входов. Самопроверка: на исходных входах сборка повторяет features_all.
Выход: data/regen/features_pit.parquet, docs/otchety/m0/pit_changes.csv."""
import sys
import numpy as np
import pandas as pd
from m0_common import (MV, DF, OUT, REGEN, YAHOO, MACRO_RAW, INDICES, calendar, feature_cols, load_prepared,
                       load_features_all, yahoo_raw, macro_raw, asof, rv_moex_pit, md_table)

sys.path.insert(0, str(MV))
from src.data.external_features import ExternalFeatureBuilder  # noqa: E402

cal = calendar()
FC = feature_cols()


class Builder(ExternalFeatureBuilder):
    def __init__(self, rvm, yahoo, macro):
        super().__init__(str(DF.parent), use_polars=False)
        self._r, self._y, self._m = rvm, yahoo, macro

    def load_moex_indices_rv(self):
        return self._r.drop(columns=["low_bars", "source"], errors="ignore")  # служебные, в признаки не входят

    def load_yahoo(self):
        return self._y.copy()

    def load_macro(self):
        return self._m.copy()


def build(rvm, yahoo, macro):
    e = Builder(rvm, yahoo, macro).build_all_external_features(verbose=False)
    e["date"] = pd.to_datetime(e["date"])
    return e.set_index("date")


def read(name):
    d = pd.read_parquet(DF / f"{name}.parquet")
    d["date"] = pd.to_datetime(d["date"])
    return d


EXT = [c for c in FC if c not in {"rv", "rsv_pos", "rsv_neg", "rv_morning", "rv_midday", "rv_evening", "volume",
                                   "close", "bv", "jump", "rskew", "rkurt", "rv_annualized", "bv_annualized",
                                   "jump_annualized", "rv_d", "rv_w", "rv_2w", "rv_m", "rv_q", "jump_lag1",
                                   "jump_lag5", "jump_lag22"}]
assert len(EXT) == 211, len(EXT)

# 0) самопроверка: исходные входы -> features_all
fa = load_features_all()
orig = build(read("rv_moex"), read("yahoo"), read("macro"))
sb = fa[fa.ticker == "SBER"].set_index("date")
chk = orig.reindex(sb.index)[EXT]
err = ((chk - sb[EXT]).abs() > 1e-7 + 1e-9 * sb[EXT].abs()) | (chk.isna() != sb[EXT].isna())  # 1e-7: шум rolling std
print(f"самопроверка сборки: ячеек с расхождением {int(err.values.sum())} из {err.size}")
assert err.values.sum() == 0

# 1) исправленные входы
yh = pd.DataFrame({"date": cal})
for c in YAHOO:  # последнее закрытие США строго до дня t (лаг 1 торговый день США)
    yh[c] = asof(cal, yahoo_raw(c), strict=True).values
mc = pd.DataFrame({"date": cal})
for c in MACRO_RAW:
    r = macro_raw(c)
    if c in ("cpi_ru", "cpi_us", "unemp_us", "m2_ru", "pmi_ru"):  # дата публикации неизвестна: конец периода + 30 дн
        r = r.assign(date=r.date + pd.offsets.MonthEnd(0) + pd.Timedelta(days=30))
        mc[c] = asof(cal, r, strict=False).values
    elif c == "key_rate_cbr":  # метка = дата вступления в силу (решение известно раньше) -> лаг 0
        mc[c] = asof(cal, r, strict=False).values
    else:  # fed_rate (DFF публикуется t+1), si_fut (закрытие 23:50) -> строго до t
        mc[c] = asof(cal, r, strict=True).values
rvp = rv_moex_pit()
pit = build(rvp, yh, mc)

# 2) на строки матрицы, медианы train
new = fa[["date", "ticker"]].join(pit[EXT], on="date")
train = fa.date <= "2017-12-31"
med = {c: (float(new.loc[train, c].median()) if pd.notna(new.loc[train, c].median()) else 0.0) for c in EXT}
new[EXT] = new[EXT].fillna(med)

prep = load_prepared()
key = prep[["date", "ticker"]].merge(new, on=["date", "ticker"], how="left")
assert len(key) == len(prep) and key[EXT].notna().all().all()
out = prep.copy()
old = prep[EXT].copy()
out[EXT] = key[EXT].values
im = rvp[rvp.ticker == "IMOEX"].set_index("date")["n_bars"]
out["imoex_low_bars"] = (out.date.map(im).fillna(0) < 30).astype("int8")
for h in (5, 22):
    g = out.sort_values(["ticker", "date"]).groupby("ticker")["rv"]
    out[f"rv_target_mean_h{h}"] = g.transform(lambda x: x[::-1].rolling(h).mean()[::-1].shift(-1))
REGEN.mkdir(parents=True, exist_ok=True)
out.to_parquet(REGEN / "features_pit.parquet", index=False)

# 3) таблица «что изменилось»
desc = {}
for c in YAHOO:
    for f in [c] + [f"yahoo_{c}_{s}" for s in ("lag1", "ret1", "ret5", "ret22", "ma5", "ma22", "vol22")]:
        desc[f] = ("закрытие США дня t (лаг 0)", "закрытие США строго до дня t (лаг 1)")
for c in ("cpi_ru", "cpi_us", "unemp_us", "m2_ru", "pmi_ru"):
    for f in (c, f"{c}_lag1", f"{c}_diff1", f"{c}_diff22"):
        desc[f] = ("с метки периода (начало/конец месяца), метки в неторговые дни теряются",
                   "с конца периода + 30 дн (даты публикации нет; консервативно)")
for c, (a, b) in {"fed_rate": ("DFF дня t", "DFF до t (лаг 1)"), "si_fut": ("закрытие 23:50 дня t", "закрытие t-1"),
                  "key_rate_cbr": ("дата вступления; точное соединение", "дата вступления; asof (без потерь)")}.items():
    for f in (c, f"{c}_lag1", f"{c}_diff1", f"{c}_diff22"):
        desc[f] = (a, b)
for t in ("CNYRUB_TOM", "RVI", "RGBI"):
    for f in (f"rv_{t}", f"close_{t}") + tuple(f"idx_{t}_rv_annualized_{s}" for s in
                                                ("lag1", "ret1", "ret5", "ret22", "ma5", "ma22", "vol22")):
        desc[f] = ("бары до 19:00 (с 20.08.2024)" if t == "RGBI" else "бары до 23:50 (вечерняя сессия)",
                   "бары с begin <= 18:50")
rows = []
for c in EXT:
    d = (out[c] - old[c]).abs() > 1e-9 * (1 + old[c].abs())
    a, b = desc.get(c, ("без изменений", "без изменений"))
    rows.append({"признак": c, "было": a, "стало": b, "доля_строк_изменено": round(float(d.mean()), 4)})
ch = pd.DataFrame(rows)
ch.to_csv(OUT / "pit_changes.csv", index=False)
ch["ряд"] = ch["признак"].str.replace(r"^(yahoo_|idx_|rv_|close_)", "", regex=True).str.replace(
    r"(_rv_annualized)?_(lag1|ret1|ret5|ret22|ma5|ma22|vol22|diff1|diff22)$", "", regex=True)
grp = ch.groupby(["ряд", "было", "стало"], sort=False).agg(признаков=("признак", "size"),
                                                           доля_строк_мин=("доля_строк_изменено", "min"),
                                                           доля_строк_макс=("доля_строк_изменено", "max")).reset_index()
print(md_table(grp[grp["было"] != "без изменений"]))
print("изменено признаков:", int((ch.доля_строк_изменено > 0).sum()), "из", len(ch), "| строк:", len(out),
      "| imoex_low_bars:", int(out.imoex_low_bars.sum()))
