"""М0 v2: функции сборки features_pit_v2 (внешние — как v1, макро с «днями с обновления», собственные — из rv_1850)."""
import ast
import datetime as dt
import sys
import numpy as np
import pandas as pd
from m0_common import (MAIN, MV, DF, YAHOO, MACRO_RAW, calendar, feature_cols, yahoo_raw, macro_raw, asof, rv_moex_pit)

sys.path.insert(0, str(MV))
from src.data.external_features import ExternalFeatureBuilder  # noqa: E402
from src.data.intraday_features import IntradayFeatureCalculator  # noqa: E402

FC = feature_cols()
OWN = FC[:23]
EXT = FC[23:]
MONTHLY = ("cpi_ru", "cpi_us", "unemp_us", "m2_ru", "pmi_ru")
CANDLES = MAIN / "moex_discovery/data/dataset_final/01_stocks/candles_10m"
RV1850 = MAIN / ".claude/worktrees/agent-a51b95b5ae61fb6bc/data/iss_check/rv_1850_noquar.parquet"  # begin в [09:50; 18:50), без баров карантина
PREP_SRC = MV / "results/final/code/00_prepare_data.py"


class ExtBuilder(ExternalFeatureBuilder):
    def __init__(self, rvm, yahoo, macro):
        super().__init__(str(DF.parent), use_polars=False)
        self._r, self._y, self._m = rvm, yahoo, macro

    def load_moex_indices_rv(self):
        return self._r.drop(columns=["low_bars", "source"], errors="ignore")

    def load_yahoo(self):
        return self._y.copy()

    def load_macro(self):
        return self._m.copy()


def build_ext(rvm, yahoo, macro):
    e = ExtBuilder(rvm, yahoo, macro).build_all_external_features(verbose=False)
    e["date"] = pd.to_datetime(e["date"])
    return e.set_index("date")


def macro_pit(cal):
    """Макро as-of: значение = последнее доступное к дню t; доступность — дата публикации (её нет) ->
    месячные: конец периода + 30 дн (пометка); key_rate: дата вступления; fed_rate, si_fut: строго до t.
    Возвращает (значения, дни с последнего обновления, доли потерянных периодов до/после)."""
    vals, days, lost = pd.DataFrame({"date": cal}), pd.DataFrame(index=cal), {}
    for c in MACRO_RAW:
        r = macro_raw(c).reset_index(drop=True)
        r["pid"] = np.arange(len(r))
        if c in MONTHLY:
            r["avail"] = r.date + pd.offsets.MonthEnd(0) + pd.Timedelta(days=30)
            strict = False
        elif c == "key_rate_cbr":
            r["avail"] = r.date
            strict = False
        else:
            r["avail"] = r.date
            strict = True
        o = r[["avail", "value", "pid"]].rename(columns={"avail": "date"}).sort_values("date")
        m = pd.merge_asof(pd.DataFrame({"date": cal}), o, on="date", allow_exact_matches=not strict)
        vals[c] = m["value"].values
        # «обновление»: новый период (месячные, fed_rate, si_fut) или новое значение ставки (key_rate)
        upd = r if c != "key_rate_cbr" else r[r.value.diff().fillna(1) != 0]
        u = pd.merge_asof(pd.DataFrame({"date": cal}), upd[["avail"]].assign(date=upd.avail).sort_values("date"),
                          on="date", allow_exact_matches=not strict)
        days[f"{c}_days_since_upd"] = (cal - pd.DatetimeIndex(u["avail"])).days.values
        rr = r[(r.date >= cal[0] - pd.Timedelta(days=62)) & (r.date <= cal[-1] - pd.Timedelta(days=62))]
        old_used = pd.merge_asof(pd.DataFrame({"date": cal}), r[r.date.isin(cal)][["date", "pid"]], on="date")["pid"]
        lost[c] = (round(1 - rr.pid.isin(old_used).mean(), 4), round(1 - rr.pid.isin(m["pid"]).mean(), 4))
    return vals, days, lost


def external_inputs(cal):
    yh = pd.DataFrame({"date": cal})
    for c in YAHOO:  # последнее закрытие США строго до дня t
        yh[c] = asof(cal, yahoo_raw(c), strict=True).values
    mc, days, lost = macro_pit(cal)
    return rv_moex_pit(), yh, mc, days, lost


class CutCalc(IntradayFeatureCalculator):
    """Тот же калькулятор (intraday_features.py:95-203), свечи — только begin в окне [t0; t1]."""
    def __init__(self, t0, t1):
        super().__init__(str(CANDLES), use_polars=True, annualize=True)
        self.t0, self.t1 = t0, t1

    def load_candles(self, ticker):
        import polars as pl
        df = super().load_candles(ticker)
        return df.filter((pl.col("begin").dt.time() >= self.t0) & (pl.col("begin").dt.time() <= self.t1))


def own_features(tickers, rv1850, windows=((dt.time(9, 50), dt.time(18, 40)), (dt.time(0, 0), dt.time(18, 50)))):
    """Собственные признаки по срезу 18:50: калькулятор на отфильтрованных свечах (окно подбирается по совпадению
    с rv_1850), затем rv, bv, jump, rsv_* берутся из rv_1850; лаги — add_har_lags/add_extended_lags (как в сборке)."""
    import polars as pl
    best = None
    for w in windows:
        c = CutCalc(*w)
        d = c.calculate_all_tickers(tickers, verbose=False).to_pandas()
        d["date"] = pd.to_datetime(d["date"])
        j = d.merge(rv1850[["date", "ticker", "rv"]], on=["date", "ticker"], suffixes=("", "_iss"))
        share = float(np.isclose(j.rv, j.rv_iss, rtol=1e-6, atol=1e-14).mean())
        if best is None or share > best[0]:
            best = (share, w, c, d)
    share, w, calc, d = best
    d = d.merge(rv1850[["date", "ticker", "rv", "bv", "j", "rsv_pos", "rsv_neg"]], on=["date", "ticker"],
                how="left", suffixes=("_cand", ""))
    miss = d.rv.isna()
    for a, b in (("rv", "rv_cand"), ("bv", "bv_cand"), ("rsv_pos", "rsv_pos_cand"), ("rsv_neg", "rsv_neg_cand")):
        d[a] = d[a].fillna(d[b])  # дней нет в rv_1850 -> значение калькулятора (учтено в отчёте)
    d["jump"] = d["j"].fillna(d["jump"])
    k = np.sqrt(252)
    d["rv_annualized"], d["bv_annualized"], d["jump_annualized"] = np.sqrt(d.rv) * k, np.sqrt(d.bv) * k, np.sqrt(d.jump) * k
    d = d.drop(columns=[c for c in d.columns if c.endswith("_cand")] + ["j"])
    p = calc.add_extended_lags(calc.add_har_lags(pl.from_pandas(d.sort_values(["ticker", "date"]))), columns=["jump"],
                               lags=[1, 5, 22]).to_pandas()
    return p, {"окно": f"{w[0]}–{w[1]}", "совпадение_rv_с_rv1850": round(share, 6), "дней_нет_в_rv1850": int(miss.sum())}


def add_targets(df):
    """rv_target_h{h} — тот же оператор 00_prepare_data.py:47-48 (RV строки t+h); rv_target_mean_h{h} — среднее t+1..t+h."""
    loop = next(n for n in ast.walk(ast.parse(PREP_SRC.read_text()))
                if isinstance(n, ast.For) and "rv_target_h" in ast.unparse(n))
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    ns = {"df": df}
    exec(compile(ast.Module([loop], []), str(PREP_SRC), "exec"), ns)
    df = ns["df"]
    for h in (1, 5, 22):
        df[f"rv_target_mean_h{h}"] = df.groupby("ticker")["rv"].transform(
            lambda x: x[::-1].rolling(h).mean()[::-1].shift(-1))
    return df


def asof_ext(ext, ext_cols, day_cols, dates):
    """v3: внешние признаки на даты строк акций as-of (только прошлое) вместо медианы train.
    ext — по дням общего календаря (индекс date). Значения: протягивание последнего наблюдения по календарю (ffill),
    затем для каждой даты строки d — строка календаря с датой <= d. До первого наблюдения ряда — NaN.
    day_cols («дни с обновления»): по календарю — как есть (от фактической даты обновления, отражённого в значении);
    для дат вне календаря — значение дня t_prev + (d - t_prev).days (то же обновление, что в протянутом значении)."""
    e = ext.sort_index()
    ff = e[list(ext_cols)].ffill()
    for c in day_cols:
        ff[c] = e[c]
    ff.index.name = "date"
    ff = ff.reset_index()
    ff["_cal"] = ff["date"]
    d = pd.DataFrame({"date": pd.DatetimeIndex(np.sort(pd.unique(pd.to_datetime(dates))))})
    m = pd.merge_asof(d, ff, on="date", direction="backward").set_index("date")
    gap = (m.index - pd.DatetimeIndex(m["_cal"])).days.astype(float)
    for c in day_cols:
        m[c] = m[c] + gap
    return m.drop(columns="_cal")


def assemble(cal, prep, r18, fill="median", log=print):
    """Сборка матрицы (m0_features_pit_v2.py): fill='median' — v2 (NaN = медианы train <= 2017, как 00_prepare_data.py),
    fill='asof' — v3 (asof_ext для внешних, протягивание по строкам тикера для собственных, до первого наблюдения — NaN).
    Возвращает (out, v2, NEWF, DAYS, lost, pre_nan)."""
    rvp, yh, mc, days, lost = external_inputs(cal)
    ext = build_ext(rvp, yh, mc)[EXT].join(days)
    DAYS = list(days.columns)
    own, info = own_features(sorted(prep.ticker.unique()), r18)
    log(f"собственные: {info}")
    keys = prep[["date", "ticker"]].copy()
    keys["_i"] = np.arange(len(keys))
    v2 = keys.merge(own[["date", "ticker", "n_bars"] + OWN], on=["date", "ticker"], how="left")
    NEWF = FC + DAYS
    v2 = v2.join(ext if fill == "median" else asof_ext(ext, EXT, DAYS, v2.date), on="date")
    v2 = add_targets(v2).sort_values("_i").reset_index(drop=True)
    pre_nan = v2[NEWF].isna().mean()
    if fill == "median":
        train = v2.date <= "2017-12-31"
        med = {c: (float(v2.loc[train, c].median()) if pd.notna(v2.loc[train, c].median()) else 0.0) for c in NEWF}
        v2[NEWF] = v2[NEWF].fillna(med)
    else:  # собственные: протягивание по строкам тикера в порядке дат (только прошлое), до первого наблюдения — NaN
        o = v2.sort_values(["ticker", "date"])
        v2.loc[o.index, OWN] = o.groupby("ticker")[OWN].ffill().values
        log(f"v3: доля NaN после as-of: OWN {v2[OWN].isna().mean().mean():.5f} EXT {v2[EXT].isna().mean().mean():.5f} "
            f"DAYS {v2[DAYS].isna().mean().mean():.5f}")
    out = prep.copy()
    for c in FC + ["n_bars", "rv_target_h1", "rv_target_h5", "rv_target_h22"]:
        out[c] = v2[c].values
    for c in DAYS + ["rv_target_mean_h1", "rv_target_mean_h5", "rv_target_mean_h22"]:
        out[c] = v2[c].values
    im = rvp[rvp.ticker == "IMOEX"].set_index("date")["n_bars"]
    out["imoex_low_bars"] = (out.date.map(im).fillna(0) < 30).astype("int8")
    return out, v2, NEWF, DAYS, lost, pre_nan
