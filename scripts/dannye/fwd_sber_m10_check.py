#!/usr/bin/env python3
"""Проверки данных M10 SBER после 27.02.2026 (Ж-247 п. 3). Значения RV нового периода не печатаются и не агрегируются:
они только в запечатанном rv_fwd_SEALED.parquet (sha256 — в манифесте). `--manifest` — только манифест; без флага — все
проверки (пишет CHECKS.csv/.md). Первый прогон: полный (запечатывает RV, дописывает rv_* в манифест), затем --manifest."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import fwd_sber_m10 as L  # noqa: E402
from models_v2 import daily as D  # noqa: E402

R, OUT = L.ROOT, L.OUT
HAR = R / "data/regen/harj_panel_hybrid/har_SBER.parquet"
FIELDS = ["open", "high", "low", "close", "volume", "value"]
NEED = ["source", "request_from", "request_till", "request_time_start", "request_time_end", "pages", "raw_sha256", "out_sha256",
        "fwd_first", "fwd_last", "fwd_rows", "archive_sha256_before", "archive_sha256_after", "loader_sha256", "git_head",
        "rv_sealed_sha256", "rv_fwd_days"]
ARCH_SHA = "b451cb48473ac15c246e5d2df0dae64704dff718039e7f34ca0a05ab47d76b9f"
ROWS = []


def add(gate, check, item, value, ok=None):
    ROWS.append({"gate": gate, "check": check, "item": item, "value": value,
                 "status": "INFO" if ok is None else ("OK" if ok else "FAIL")})


def border_bad(fwd):
    return fwd[fwd["begin"] <= L.FWD_AFTER]


def dup_count(df):
    return int(df["begin"].duplicated().sum())


def units_bad_days(df, lo=0.9, hi=1.1):
    """Дни, где медиана (value/volume)/close вне lo…hi (объём в штуках, не лотах)."""
    d = df[df["volume"] > 0]
    r = (d["value"] / d["volume"] / d["close"]).groupby(d["begin"].dt.normalize()).median()
    return r[(r < lo) | (r > hi)]


def session_of(begin):
    t = begin.dt.strftime("%H:%M")
    s = np.where(t < "10:00", "utro", np.where(t <= "18:40", "osnov", "vecher"))
    return pd.Series(np.where(begin.dt.dayofweek >= 5, "vyhodnoy", s), index=begin.index)


def schedule(df):
    return df.groupby([df["begin"].dt.strftime("%Y-%m"), session_of(df["begin"])]).size()


def manifest_check(strict=True):
    man = json.loads((OUT / "MANIFEST.json").read_text(encoding="utf-8"))
    miss = [k for k in NEED if k not in man]
    add("F1", "manifest", "полнота ключей", f"нет: {miss}" if miss else "полон", not miss)
    a = L.sha256(L.ARCHIVE)
    add("F1", "manifest", "sha256 архива до=после=текущий=b451cb48", a[:8],
        man["archive_sha256_before"] == man["archive_sha256_after"] == a == ARCH_SHA)
    bad = [n for n, h in man["raw_sha256"].items() if L.sha256(OUT / "raw" / n) != h]
    bad += [n for n, h in man["out_sha256"].items() if L.sha256(OUT / n) != h]
    add("F1", "manifest", "sha256 сырых страниц и выходов", f"{len(man['raw_sha256'])} стр.; расхождений {len(bad)}", not bad)
    add("F1", "manifest", "страниц = файлов raw", len(man["raw_sha256"]), len(list((OUT / "raw").glob("*.json"))) == man["pages"])
    add("F1", "manifest", "sha256 кода загрузчика", L.sha256(L.__file__)[:8], man["loader_sha256"] == L.sha256(L.__file__))
    return man


def overlap_vs_archive(ov, arch):
    a = arch[(arch["begin"] >= L.OVERLAP_FROM) & (arch["begin"] <= L.FWD_AFTER)].set_index("begin")
    n = ov.set_index("begin")
    only_a, only_n = a.index.difference(n.index), n.index.difference(a.index)
    both = a.index.intersection(n.index)
    diffs = []
    for k in FIELDS:
        x, y = a.loc[both, k].to_numpy(float), n.loc[both, k].to_numpy(float)
        tol = 1e-6 * np.abs(x) if k == "value" else 0.0
        diffs += [(both[i], k, x[i], y[i]) for i in np.where(np.abs(x - y) > tol)[0]]
    add("F2", "граница", "бары перекрытия: архив / новые / общие; только в одном (архив | новые)",
        f"{len(a)} / {len(n)} / {len(both)}; {list(only_a)[:30]} | {list(only_n)[:30]}", not len(only_a) + len(only_n))
    add("F2", "граница", "расхождения полей (open/high/low/close/volume — 0; value — 1e-6 отн.); список", f"{len(diffs)}: {diffs[:50]}", not diffs)


def gaps(fwd):
    days = pd.Index(fwd["begin"].dt.normalize().unique())
    main = D.main_session(fwd)
    cnt = main.groupby(main["begin"].dt.normalize()).size()
    nomain = [d.date().isoformat() for d in pd.bdate_range(days.min(), days.max()) if d not in cnt.index]
    cal = pd.read_csv(R / "data/calendar/torgovye_dni_moex.csv")["date"].max()
    add("F3", "пропуски", f"будние дни без баров основной сессии (календарь torgovye_dni_moex.csv до {cal} период не покрывает)",
        f"{len(nomain)}: {' '.join(nomain)}")
    short = cnt[(cnt < 53) & (cnt.index.dayofweek < 5)]
    add("F3", "пропуски", "будние дни основной сессии < 53 баров (дата:баров)", f"{len(short)}: " + " ".join(f"{d.date()}:{c}" for d, c in short.items()))
    slots = pd.date_range("2000-01-01 10:00", "2000-01-01 18:40", freq="10min").strftime("%H:%M")
    inner = 0
    for d, g in main[main["begin"].dt.dayofweek < 5].groupby(main["begin"].dt.normalize()):
        miss = [s for s in slots if s not in set(g["begin"].dt.strftime("%H:%M"))]
        inner += bool(miss)
        if miss:
            add("F3", "пропуски внутри 10:00–18:40", d.date().isoformat(), f"нет {len(miss)}: {' '.join(miss[:12])}")
    add("F3", "пропуски внутри 10:00–18:40", "будних дней с дырой", inner)
    hol = [d.date().isoformat() for d in cnt.index if d.dayofweek < 5 and d not in proxy_days(fwd)]
    add("F3", "пропуски", "будние дни без вечерней сессии (праздничный формат 09:50–18:50)", f"{len(hol)}: {' '.join(hol)}")
    add("F3", "пропуски", "торговых дней (все сессии) / дней основной сессии на выходных", f"{len(days)} / {int((cnt.index.dayofweek >= 5).sum())}")


def schedule_rows(fwd, arch):
    last3 = arch[arch["begin"] >= "2025-12-01"]
    for name, df in (("fwd", fwd), ("архив 3 мес", last3)):
        for (m, s), c in schedule(df).items():
            add("F3", "расписание", f"{name} {m} {s}", int(c))


def units_rows(fwd, ov):
    for name, df in (("fwd", fwd), ("перекрытие", ov)):
        bad = units_bad_days(df)
        add("F3", "единицы объёма", f"{name}: дней вне 0,9…1,1 (медиана value/volume ÷ close)", len(bad), not len(bad))


def proxy_days(bars):
    """Замена «дней сессии IMOEX» (IMOEX-архив кончается 27.02.2026): будни с вечерней сессией (бар ≥ 19:00); праздничный формат (09:50–18:50) — нет."""
    ev = bars[(bars["begin"].dt.strftime("%H:%M") >= "19:00") & (bars["begin"].dt.dayofweek < 5)]
    return set(ev["begin"].dt.normalize())


def rv_rows(fwd, ov, arch, man):
    har = pd.read_parquet(HAR).set_index("target_date")["rv"]
    new = D.build_daily(ov)[0]["rv"]
    old = D.build_daily(arch[arch["begin"] >= "2026-02-20"])[0]["rv"]
    com = new.index.intersection(har.index)
    rel = lambda a, b: float((np.abs(a - b) / b).max())  # noqa: E731
    add("F4", "RV перекрытия", "дней в сверке (новые бары) / har_SBER", f"{len(new)} / {len(com)}", len(new) == len(com) and len(com) > 0)
    add("F4", "RV перекрытия", "max отн. |RV новых − har_SBER| ≤ 1e-12", f"{rel(new[com], har[com]):.3e}", rel(new[com], har[com]) <= 1e-12)
    add("F4", "RV перекрытия", "max отн. |RV новых − RV архивных баров| ≤ 1e-12", f"{rel(new, old.reindex(new.index)):.3e}",
        rel(new, old.reindex(new.index)) <= 1e-12)
    # отступление: IMOEX-архив кончается 27.02 — правило «нет сессии IMOEX» для нового периода заменено proxy_days; proxy сверен с IMOEX на архиве
    imx = D.index_session_days()
    a25 = arch[arch["begin"] >= D.WEEKEND_FROM]
    pa, ia = {x for x in proxy_days(a25)}, {x for x in imx if x >= D.WEEKEND_FROM and x.dayofweek < 5}
    add("F4", "proxy сессии IMOEX", "будни архива с 01.03.2025: proxy ≠ IMOEX (дней) / дней IMOEX", f"{len(pa ^ ia)} / {len(ia)}", pa == ia)
    d = D.build_daily(fwd, proxy_days(fwd))[0]
    p = OUT / "rv_fwd_SEALED.parquet"
    fresh = not p.exists()
    if fresh:
        d.to_parquet(p)
    same = pd.read_parquet(p).equals(d)
    h = L.sha256(p)
    if fresh:
        man.update(rv_sealed_sha256=h, rv_fwd_days=int(len(d)), rv_builder="models_v2.daily.build_daily, index_days=proxy_days (будни с вечерней сессией)")
        (OUT / "MANIFEST.json").write_text(json.dumps(man, ensure_ascii=False, indent=1), encoding="utf-8")
    add("F4", "RV нового периода", "построена тем же кодом без ошибок; запечатана; sha256 = манифест; дней", f"{len(d)}; {h[:8]}",
        same and h == man["rv_sealed_sha256"] and len(d) == man["rv_fwd_days"])


def events_rows(last):
    add("F3", "события", "corporate_actions.csv: строк SBER", int((pd.read_csv(R / "data/corporate_actions.csv")["ticker"] == "SBER").sum()))
    ev = pd.read_csv(R / "data/events.csv")
    e = ev[ev["ticker"].isin(["SBER", "*"])]
    cover = e["data_sobytiya"].max()
    new = e[e["data_sobytiya"] > "2026-02-27"]
    add("F3", "события", f"events.csv (SBER и *): покрытие до {cover}; строк в новом периоде", len(new), None)
    for _, r in new.iterrows():
        add("F3", "события", f"{r['data_sobytiya']} {r['tip']}", f"объявлено {r['data_obyavleniya']}", None)
    if cover < last:
        add("F3", "события", "дивидендная отсечка и корпоративные действия", f"нет PIT-данных после {cover} (поиск в сети не делался)", None)


def main():
    if "--manifest" in sys.argv:
        manifest_check()
    else:
        fwd, ov = pd.read_parquet(OUT / "SBER_m10_fwd.parquet"), pd.read_parquet(OUT / "SBER_m10_overlap.parquet")
        arch = pd.read_parquet(L.ARCHIVE)
        overlap_vs_archive(ov, arch)
        add("F2", "граница", "баров fwd ≤ 2026-02-27 23:59", len(border_bad(fwd)), not len(border_bad(fwd)))
        add("F2", "дубли", "повторы begin: fwd / перекрытие", f"{dup_count(fwd)} / {dup_count(ov)}", not dup_count(fwd) and not dup_count(ov))
        gaps(fwd)
        schedule_rows(fwd, arch)
        units_rows(fwd, ov)
        rv_rows(fwd, ov, arch, json.loads((OUT / "MANIFEST.json").read_text(encoding="utf-8")))
        events_rows(fwd["begin"].max().date().isoformat())
        manifest_check()
        t = pd.DataFrame(ROWS)
        t.to_csv(OUT / "CHECKS.csv", index=False)
        md = ["# CHECKS (данные M10 SBER после 27.02.2026; значения RV нового периода не показаны)", "",
              "| gate | check | item | value | status |", "|---|---|---|---|---|"]
        md += ["| " + " | ".join(str(r[k]).replace("|", "/") for k in ("gate", "check", "item", "value", "status")) + " |" for r in ROWS]
        (OUT / "CHECKS.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    bad = [r for r in ROWS if r["status"] == "FAIL"]
    print(f"проверок {len(ROWS)}; FAIL {len(bad)}")
    for r in bad[:40]:
        print("FAIL", r["check"], r["item"], r["value"])
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
