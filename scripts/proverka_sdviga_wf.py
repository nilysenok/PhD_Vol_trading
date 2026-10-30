"""Д1 для счёта wf (v5/v6): проверка сдвига ВСЕХ входных рядов, одна таблица docs/tables/proverka_sdviga_wf.csv.

Для каждого ряда: corr(x_d, цель_{d+k}), k = −2…+1, медиана по тикерам; максимум |corr| обязан быть на k = 0.
Оконные ряды (σ̂, ATR) — и по приращениям. Плоский профиль (размах < ε = 0,01, testy.md п. 4) — вторая проверка через
связанную величину (ставка ЦБ: смена не раньше заседания; дивиденды: ценовой гэп в день события).
Код выхода 1, если хоть один ряд не прошёл или не прочитан (ряд без проверки в расчёт не входит).
--do 2021-12-31 (умолчание); позже 2021 — только при docs/resheniya/wf_zamok_snyat.json (period_v2.proverit_chtenie).
Запуск (≤ 2021, один процесс): python3 scripts/proverka_sdviga_wf.py; тесты — tests/test_proverka_sdviga_wf.py."""
from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
KS, EPS = (-2, -1, 0, 1), 0.01
OUT = ROOT / "docs/tables/proverka_sdviga_wf.csv"
IDX = "moex_discovery/data/final_v6/candles_10m/external/"
VPRED = "moex_volatility/results/final/strategies/data/vpred_aligned.parquet"


def profil(x: pd.DataFrame, y: pd.DataFrame, diff: bool = False) -> dict:
    """Медиана по столбцам (тикерам) corr(x_d, y_{d+k}); при diff — по приращениям обоих."""
    if diff:
        x, y = x.diff(), y.diff()
    cols = [c for c in x.columns if c in y.columns]
    return {k: float(np.nanmedian([x[c].corr(y[c].shift(-k)) for c in cols] or [np.nan])) for k in KS}


def verdikt(prof: dict, vtoraya: bool | None = None) -> tuple[bool, str]:
    """(ok, причина). Максимум |corr| на k = 0. vtoraya задана (ряд по природе ступенчатый/разреженный) — решает она;
    не задана, а профиль плоский (размах < ε) — не ок."""
    if vtoraya is not None:
        return bool(vtoraya), "вторая проверка (связанная величина)"
    v = {k: abs(c) for k, c in prof.items()}
    if not v or any(np.isnan(c) for c in v.values()):
        return False, "корреляция не определена"
    if max(v.values()) - min(v.values()) < EPS:
        return False, "плоский профиль без второй проверки"
    k = max(v, key=v.get)
    return (k == 0), f"максимум на k={k}"


def stroka(ryad: str, tip: str, cel: str, prof: dict, vtoraya: bool | None = None) -> dict:
    ok, pr = verdikt(prof, vtoraya)
    return {"ryad": ryad, "tip": tip, "cel": cel, **{f"corr_k{k:+d}": prof.get(k, np.nan) for k in KS}, "ok": ok, "primechanie": pr}


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()[:12]


def rets(c: pd.DataFrame) -> pd.DataFrame:
    return np.log(c).diff()


def indeks(main: Path, name: str, do: pd.Timestamp) -> pd.Series:
    """close индекса на 18:50 по дням; 10-минутки читаются фильтром begin ≤ do (vneshnie_1850.read_bars)."""
    from fastbt import daily_1850 as d18
    from fastbt.vneshnie_1850 import read_bars
    return d18.aggregate(read_bars(main / IDX / f"{name}.parquet", do + pd.Timedelta(days=1)), name).set_index("date")["close"].astype(float)


def zagruzit(main: Path, do: pd.Timestamp) -> dict:
    f = [("date", "<=", do.to_pydatetime())]
    d = pd.read_parquet(main / "data/fastbt/daily/ohlcv_daily_1850.parquet", filters=f)
    d["date"] = pd.to_datetime(d["date"])
    W = {k: d.pivot(index="date", columns="ticker", values=k) for k in ("close", "high", "low")}
    rv = pd.read_parquet(main / "data/iss_check/rv_1850_noquar.parquet", columns=["date", "ticker", "rv"], filters=f)
    rv["date"] = pd.to_datetime(rv["date"])
    W["rv"] = rv.pivot(index="date", columns="ticker", values="rv").clip(lower=1e-12)
    vp = pd.read_parquet(main / VPRED, filters=f)
    vp["date"] = pd.to_datetime(vp["date"])
    W.update({h: vp.pivot(index="date", columns="ticker", values=f"sigma_{h}") for h in ("h1", "h5", "h22")})
    return W


def ryady_aktsii(W: dict) -> list:
    lrv = np.log(W["rv"])
    rows = []
    for h in ("h1", "h5", "h22"):
        s = np.log(W[h].clip(lower=1e-12) ** 2)
        if h == "h1":                                       # неоконный: уровень (оконные — только по приращениям, их уровень персистентен)
            rows.append(stroka(f"sigma_{h}", "uroven", "ln RV_{d+k}", profil(s, lrv)))
        rows.append(stroka(f"sigma_{h}", "prirashcheniya", "ln RV_{d+k}", profil(s, lrv, True)))     # цель — дневная RV, как в ch4_dev/proverka_sdvig.csv
    rng = np.log(W["high"] / W["low"]).clip(lower=1e-6)
    tr = pd.concat([W["high"] - W["low"], (W["high"] - W["close"].shift()).abs(), (W["low"] - W["close"].shift()).abs()]).groupby(level=0).max()
    atr = np.log((tr / W["close"]).rolling(14).mean())
    lr2 = np.log(rng ** 2)
    rows += [stroka("rv", "uroven", "ln range²_{d+k}", profil(lrv, lr2)), stroka("rv", "prirashcheniya", "ln range²_{d+k}", profil(lrv, lr2, True)),
             stroka("atr14", "prirashcheniya", "ln RV_{d+k}", profil(atr, lrv, True)),
             stroka("bary_hl", "uroven", "sqrt RV_{d+k}", profil(rng, np.sqrt(W["rv"])))]
    return rows


def ryady_indeksov(main: Path, W: dict, do: pd.Timestamp) -> tuple[list, pd.Series]:
    """IMOEX, td_rel (r − r_IMOEX), сектор; цель — доходность тикера. Возврат (строки, доходность IMOEX на сетке)."""
    r = rets(W["close"])
    im = rets(indeks(main, "IMOEX", do).to_frame("IMOEX"))["IMOEX"].reindex(r.index)
    X = pd.DataFrame({c: im for c in r.columns})
    rows = [stroka("imoex", "doxodnost", "r тикера_{d+k}", profil(X, r)), stroka("td_rel", "otnositelnyj", "r тикера_{d+k}", profil(r.sub(X), r))]
    sk = pd.read_csv(ROOT / "docs/issledovaniya/2026-10-27_sektory.csv")
    Xs = {t: rets(indeks(main, n, do).to_frame(n))[n].reindex(r.index) for t, n in zip(sk["ticker"], sk["indeks"])
          if t in r.columns and (main / IDX / f"{n}.parquet").exists()}
    rows.append(stroka("sektor", "doxodnost", "r тикера_{d+k}", profil(pd.DataFrame(Xs), r)) if Xs else stroka("sektor", "net_dannykh", "", {}))
    return rows, im


def dx_usdrub_schet(main: Path, do: pd.Timestamp) -> pd.Series:
    """Доходность оси usdrub — тот же ряд, что читает счёт: USD до 11.06.2024, CNY после (vneshnie_usdrub, #795)."""
    from fastbt import vneshnie_1850 as v
    from fastbt import vneshnie_usdrub as vu
    gr = do + pd.Timedelta(days=1)
    pc = main / v.PATHS["cnyrub"]
    if not pc.exists() and do > vu.USD_LAST:      # Д4: хвост после стыка без CNY молча пуст — отказ
        raise FileNotFoundError(f"CNYRUB_TOM: нет {pc}; хвост usdrub после {vu.USD_LAST.date()} пуст")
    cny = v.read_bars(pc, gr) if pc.exists() else None
    return vu.faktor_usdrub(v.read_bars(main / v.PATHS["usdrub"], gr), cny)[0]


def ryady_vneshnie(main: Path, im: pd.Series, do: pd.Timestamp) -> list:
    from fastbt import vneshnie_1850 as v
    y = im.dropna().to_frame("x")
    rows = []
    for n in ("usdrub", "rvi"):
        if n == "usdrub":
            ch = dx_usdrub_schet(main, do)
        else:
            ch = v.change(n, v.close_1850(v.read_bars(main / v.PATHS[n], do + pd.Timedelta(days=1)), n))
        rows.append(stroka(f"vneshnie_{n}", "izmenenie", "r IMOEX_{d+k}", profil(ch.to_frame("x"), y)))
    b = v.load_brent(main).set_index("date")["ret"]
    rows.append(stroka("vneshnie_brent", "izmenenie", "r IMOEX_{d+k}", profil(b[b.index <= do].to_frame("x"), y)))
    return rows


def stavka_ok(st: pd.DataFrame, zasedaniya: np.ndarray) -> bool:
    """Вторая проверка ступенчатой ставки: каждая смена вступает в силу не раньше заседания и не позже чем через 10 дней."""
    for d in st["data_vstupleniya"].to_numpy()[1:]:
        prev = zasedaniya[zasedaniya <= d]
        if len(prev) == 0 or (d - prev.max()) / np.timedelta64(1, "D") > 10:
            return False
    return True


def ryady_sobytiya(do: pd.Timestamp, im: pd.Series, r: pd.DataFrame) -> list:
    from fastbt import dividendy
    ev = pd.read_csv(ROOT / "data/events.csv", parse_dates=["data_obyavleniya", "data_sobytiya"])
    ev = ev[ev["data_sobytiya"] <= do]
    pit = bool((ev["data_obyavleniya"] <= ev["data_sobytiya"]).all())
    zased = ev[ev["tip"].str.startswith("cbr_zasedanie")]["data_sobytiya"].to_numpy()
    st = pd.read_csv(ROOT / "fastbt/rates/key_rate_cbr_vladelec.csv", parse_dates=["data_vstupleniya"], comment="#")
    st = st[st["data_vstupleniya"] <= do]
    dr = st.set_index("data_vstupleniya")["stavka_pct"].reindex(im.index, method="ffill").diff().abs().fillna(0)
    rows = [stroka("stavka_cbr", "stupenchataya", "|r IMOEX|_{d+k}", profil(dr.to_frame("x"), im.abs().to_frame("x")), stavka_ok(st, zased)),
            {**stroka("sobytiya", "pit", "", {}), "ok": pit, "primechanie": "объявлено не позже события" if pit else "событие раньше объявления"}]
    D = pd.DataFrame(0.0, index=r.index, columns=r.columns)
    for t in r.columns:
        for cum in dividendy.events(t).query("last_cum_date <= @do")["last_cum_date"]:
            g = r.index.searchsorted(cum, side="right")     # первый бар после last_cum_date — день гэпа
            if 0 < g < len(r):
                D.iloc[g, D.columns.get_loc(t)] = 1.0
    pd_ = profil(D, -r.fillna(0))                           # связанная величина: гэп цены в день события — максимум на k = 0
    rows.append(stroka("dividendy", "gep_v_den_otsechki", "−r тикера_{d+k}", pd_, bool(D.to_numpy().sum() > 0 and max(pd_, key=lambda k: abs(pd_[k])) == 0)))
    return rows


def sobrat(main: Path, do: pd.Timestamp) -> pd.DataFrame:
    W = zagruzit(main, do)
    rows = ryady_aktsii(W)
    ri, im = ryady_indeksov(main, W, do)
    rows += ri
    for f in (lambda: ryady_vneshnie(main, im, do), lambda: ryady_sobytiya(do, im, rets(W["close"]))):
        try:
            rows += f()
        except Exception as e:                      # нечитаемый ряд = не прошёл (без проверки в расчёт не входит)
            rows.append({**stroka("nechitaemyj_ryad", "oshibka", str(e)[:80], {}), "ok": False})
    return pd.DataFrame(rows)


def shapka(main: Path, do: str, ok: bool) -> str:
    kod = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    vh = {p.name: sha(p) for p in (main / "data/iss_check/rv_1850_noquar.parquet", main / "data/fastbt/daily/ohlcv_daily_1850.parquet", ROOT / "data/events.csv")}
    st = "разработка 2018–2021, после аудита" if do <= "2021-12-31" else "wf, замок снят владельцем"
    return f"# статус: {st}; до {do}; Д1 {'пройдена' if ok else 'НЕ ПРОЙДЕНА'}\n# код: {kod}\n# хэши входов: {vh}\n"


def main(argv=None) -> int:
    a = argparse.ArgumentParser()
    a.add_argument("--do", default="2021-12-31")
    a.add_argument("--main", default=None)
    n = a.parse_args(argv)
    if pd.Timestamp(n.do).year >= 2022:
        from fastbt import period_v2
        period_v2.proverit_chtenie("wf", otmetka=period_v2.OTMETKA)
    from fastbt import daily_1850 as d18
    root = Path(n.main) if n.main else d18.default_main()
    t = sobrat(root, pd.Timestamp(n.do))
    ok = bool(t["ok"].all())
    OUT.write_text(shapka(root, n.do, ok) + t.to_csv(index=False, float_format="%.4f"))
    print(t[["ryad", "tip", "corr_k-2", "corr_k-1", "corr_k+0", "corr_k+1", "ok", "primechanie"]].to_string(index=False))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
