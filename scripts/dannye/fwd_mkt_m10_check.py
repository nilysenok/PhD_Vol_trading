#!/usr/bin/env python3
"""Проверки данных M10 16 бумаг + IMOEX после 27.02.2026 (Ж-248 п. 1). Только данные: признаки, RV и доходности нового периода
не считаются; у запечатанной RV SBER читается только индекс дат. `--manifest` — только манифест; без флага — все проверки
(пишет CHECKS.csv, CHECKS.md, AVAIL.csv в data/fwd/mkt_m10/)."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import fwd_mkt_m10 as L  # noqa: E402
import fwd_sber_m10_check as C  # noqa: E402
from models_v2 import daily as D  # noqa: E402

OUT, S = L.OUT, L.S
SBER_FWD, SEALED = S.OUT / "SBER_m10_fwd.parquet", S.OUT / "rv_fwd_SEALED.parquet"
FIELDS, ROWS = C.FIELDS, []
NEED = ["source", "request_from", "request_till", "instruments", "archive_sha256_before", "archive_sha256_after", "loader_sha256", "git_head"]
NEED_I = ["request_time_start", "request_time_end", "pages", "pages_contiguous", "raw_sha256", "out_sha256", "fwd_first", "fwd_last",
          "fwd_rows", "overlap_rows", "fwd_days"]


def add(gate, check, item, value, ok=None):
    ROWS.append({"gate": gate, "check": check, "item": item, "value": value, "status": "INFO" if ok is None else ("OK" if ok else "FAIL")})


def after_last(df):
    return df[df["begin"] > L.LAST_BAR]


def main_days(df):
    m = D.main_session(df)
    return m.groupby(m["begin"].dt.normalize()).size()


def manifest_check():
    man = json.loads((OUT / "MANIFEST.json").read_text(encoding="utf-8"))
    miss = [k for k in NEED if k not in man] + [f"{s}.{k}" for s, v in man["instruments"].items() for k in NEED_I if k not in v]
    add("M1", "manifest", "полнота ключей", f"нет: {miss}" if miss else "полон", not miss)
    ins = set(man["instruments"])
    add("M1", "manifest", "инструментов в манифесте (16 бумаг + IMOEX)", len(ins), ins == set(L.INSTRUMENTS))
    now = {s: S.sha256(L.archive_path(s)) for s in L.INSTRUMENTS}
    add("M1", "manifest", "sha256 архивных файлов до = после = текущий (17)", f"{sum(now[s] == man['archive_sha256_before'].get(s) for s in now)} из {len(now)}",
        now == man["archive_sha256_before"] == man["archive_sha256_after"])
    bad = [f"{s}/{n}" for s, v in man["instruments"].items() for n, h in v["raw_sha256"].items() if S.sha256(OUT / "raw" / s / n) != h]
    bad += [n for v in man["instruments"].values() for n, h in v["out_sha256"].items() if S.sha256(OUT / n) != h]
    add("M1", "manifest", "sha256 сырых страниц и выходов", f"страниц {sum(v['pages'] for v in man['instruments'].values())}; расхождений {len(bad)}", not bad)
    add("M1", "manifest", "страницы подряд, все кроме последней полные (нет потери страницы)",
        [s for s, v in man["instruments"].items() if not v["pages_contiguous"]] or "все", all(v["pages_contiguous"] for v in man["instruments"].values()))
    add("M1", "manifest", "sha256 кода загрузчика", S.sha256(L.__file__)[:8], man["loader_sha256"] == S.sha256(L.__file__))
    for s, v in man["instruments"].items():
        add("M1", "манифест", f"{s}: первая / последняя метка fwd; баров; дней", f"{v['fwd_first']} / {v['fwd_last']}; {v['fwd_rows']}; {v['fwd_days']}")
    return man


def overlap_vs_archive(s, ov, arch):
    a = arch[(arch["begin"] >= S.OVERLAP_FROM) & (arch["begin"] <= S.FWD_AFTER)].set_index("begin")
    n = ov.set_index("begin")
    only_a, only_n, both = a.index.difference(n.index), n.index.difference(a.index), a.index.intersection(n.index)
    diffs = []
    for k in FIELDS:
        x, y = a.loc[both, k].to_numpy(float), n.loc[both, k].to_numpy(float)
        tol = 1e-6 * np.abs(x) if k == "value" else 0.0
        diffs += [(str(both[i]), k, x[i], y[i]) for i in np.where(np.abs(x - y) > tol)[0]]
    add("M2", "граница", f"{s}: перекрытие 20–27.02: архив / новые / общие; только в одном (архив | новые)",
        f"{len(a)} / {len(n)} / {len(both)}; {[str(t) for t in only_a[:20]]} | {[str(t) for t in only_n[:20]]}", not len(only_a) + len(only_n))
    add("M2", "граница", f"{s}: расхождения полей (цены/объём — 0; value — 1e-6 отн.)", f"{len(diffs)}: {diffs[:30]}", not diffs)


def border_dups(s, fwd, ov):
    add("M2", "граница", f"{s}: баров fwd ≤ 2026-02-27 23:59 / > 2026-11-10 23:59", f"{len(C.border_bad(fwd))} / {len(after_last(fwd))}",
        not len(C.border_bad(fwd)) and not len(after_last(fwd)))
    add("M2", "дубли", f"{s}: повторы begin: fwd / перекрытие", f"{C.dup_count(fwd)} / {C.dup_count(ov)}", not C.dup_count(fwd) and not C.dup_count(ov))
    add("M2", "пустые значения", f"{s}: null в fwd", int(fwd.isna().sum().sum()), not fwd.isna().sum().sum())


def gaps(s, fwd, sdays):
    cnt = main_days(fwd)
    nomain = [d.date().isoformat() for d in sdays if d not in cnt.index]
    short = [f"{d.date()}:{cnt[d]}" for d in sdays if d in cnt.index and cnt[d] < D.MIN_BARS]
    add("M3", "пропуски", f"{s}: будние дни SBER без баров основной сессии инструмента", f"{len(nomain)}: {' '.join(nomain)}")
    add("M3", "пропуски", f"{s}: будние дни SBER с < {D.MIN_BARS} барами основной сессии (дата:баров)", f"{len(short)}: {' '.join(short)}")
    n_all = int(fwd["begin"].dt.normalize().nunique())
    add("M3", "пропуски", f"{s}: дней с барами (любая сессия) / будних дней SBER", f"{n_all} / {len(sdays)}")


def sessions(s, fwd, arch):
    last3 = arch[arch["begin"] >= "2025-12-01"]
    for name, df in (("fwd", fwd), ("архив 3 мес", last3)):
        t = C.schedule(df).unstack(fill_value=0)
        for m, r in t.iterrows():
            add("M3", "расписание", f"{s} {name} {m}", " ".join(f"{k}={int(v)}" for k, v in r.items() if v))
    fs, asl = set(fwd["begin"].dt.strftime("%H:%M")), set(last3["begin"].dt.strftime("%H:%M"))
    wk = fwd[fwd["begin"].dt.dayofweek < 5]["begin"].dt.strftime("%H:%M")
    new = sorted(set(wk) - set(last3[last3["begin"].dt.dayofweek < 5]["begin"].dt.strftime("%H:%M")))
    add("M3", "зона времени", f"{s}: метки без зоны (мск); слоты будней fwd вне слотов архива 3 мес {'(РАСХОЖДЕНИЕ)' if new else ''}", f"{len(new)}: {new[:20]}")
    add("M3", "зона времени", f"{s}: первый / последний слот будней fwd; архив 3 мес", f"{wk.min()} / {wk.max()}; "
        f"{min(asl)} / {max(asl)}")
    if s == "IMOEX":
        m = D.main_session(fwd)["begin"].dt.strftime("%H:%M")
        add("M3", "зона времени", "IMOEX: часы основной сессии (10:00–18:40) — первый / последний бар", f"{m.min()} / {m.max()}", m.min() == "10:00" and m.max() == "18:40")


def units(s, fwd, ov, arch):
    if s != "IMOEX":
        for name, df in (("fwd", fwd), ("перекрытие", ov)):
            bad = C.units_bad_days(df)
            add("M3", "единицы", f"{s} {name}: дней вне 0,9…1,1 (медиана value/volume ÷ close)", f"{len(bad)}: {[str(d.date()) for d in bad.index[:20]]}", not len(bad))
        return
    for name, df in (("fwd", fwd), ("архив 3 мес", arch[arch["begin"] >= "2025-12-01"])):
        add("M3", "единицы", f"IMOEX {name}: доля баров volume = 0 / медиана value / min close", f"{(df['volume'] == 0).mean():.3f} / {df['value'].median():.4g} / {df['close'].min():.2f}")
    add("M3", "единицы", "IMOEX: объём как в архиве (доля нулей fwd = архив 3 мес)", "см. строки выше",
        (fwd["volume"] == 0).mean() == (arch[arch["begin"] >= "2025-12-01"]["volume"] == 0).mean())


def sealed_dates():
    """Только индекс дат запечатанной RV SBER (значения не читаются)."""
    return set(pd.read_parquet(SEALED, columns=[]).index)


def calendar(imx_fwd, sber):
    """Правило IMOEX (исходное build_daily: дни сессий IMOEX из архива и нового периода) против замещающего (запечатанная RV)."""
    idays = set(D.index_session_days()) | set(main_days(imx_fwd).index)
    rule_imoex = set(D.build_daily(sber, index_days=idays)[0].index)
    rule_proxy = sealed_dates()
    only_i, only_p = sorted(rule_imoex - rule_proxy), sorted(rule_proxy - rule_imoex)
    f = lambda xs: " ".join(str(x.date()) for x in xs)  # noqa: E731
    add("M4", "календарь SBER", "дней по правилу IMOEX (build_daily, index_days = сессии IMOEX) / по замещающему (запечатанная RV)", f"{len(rule_imoex)} / {len(rule_proxy)}")
    add("M4", "календарь SBER", "даты только по правилу IMOEX (нет в запечатанной)", f"{len(only_i)}: {f(only_i)}", None)
    add("M4", "календарь SBER", "даты только по замещающему (нет по правилу IMOEX)", f"{len(only_p)}: {f(only_p)}", None)
    add("M4", "календарь SBER", "правила совпадают", "да" if not only_i and not only_p else "НЕТ (решение — дирижёр)", None)
    sdays = set(sber["begin"].dt.normalize())
    nosess = sorted(d for d in sdays if d.dayofweek < 5 and d not in idays)
    add("M4", "календарь SBER", "будние дни SBER без сессии IMOEX", f"{len(nosess)}: {f(nosess)}")


def availability(frames, sber):
    sm = D.main_session(sber)
    close = sm.groupby(sm["begin"].dt.normalize())["begin"].max()
    rows = []
    for s, df in frames.items():
        m = D.main_session(df)
        last = m.groupby(m["begin"].dt.normalize())["begin"].max().reindex(close.index)
        anyb = df.groupby(df["begin"].dt.normalize())["begin"].max().reindex(close.index)
        viol = last[last > close]
        rows += [{"date": d.date(), "instrument": s, "sber_close_begin": close[d], "last_main_begin": last[d], "last_any_begin": anyb[d]} for d in close.index]
        add("M4", "доступность", f"{s}: дней SBER / нет баров основной сессии / последний бар основной сессии позже SBER", f"{len(close)} / {int(last.isna().sum())} / {len(viol)}", not len(viol))
        add("M4", "доступность", f"{s}: максимум времени последнего бара основной сессии дня / любой сессии", f"{last.dropna().dt.strftime('%H:%M').max()} / {anyb.dropna().dt.strftime('%H:%M').max()}")
    pd.DataFrame(rows).to_csv(OUT / "AVAIL.csv", index=False)


def main():
    if "--manifest" in sys.argv:
        manifest_check()
    else:
        sber = pd.read_parquet(SBER_FWD)
        sdays = [d for d in sorted(set(sber["begin"].dt.normalize())) if d.dayofweek < 5]
        frames = {}
        for s in L.INSTRUMENTS:
            fwd, ov, arch = pd.read_parquet(OUT / f"{s}_m10_fwd.parquet"), pd.read_parquet(OUT / f"{s}_m10_overlap.parquet"), pd.read_parquet(L.archive_path(s))
            frames[s] = fwd
            overlap_vs_archive(s, ov, arch)
            border_dups(s, fwd, ov)
            gaps(s, fwd, sdays)
            sessions(s, fwd, arch)
            units(s, fwd, ov, arch)
        calendar(frames["IMOEX"], sber)
        availability(frames, sber)
        manifest_check()
        pd.DataFrame(ROWS).to_csv(OUT / "CHECKS.csv", index=False)
        md = ["# CHECKS (данные M10 16 бумаг + IMOEX после 27.02.2026; значения RV не показаны)", "", "| gate | check | item | value | status |", "|---|---|---|---|---|"]
        md += ["| " + " | ".join(str(r[k]).replace("|", "/") for k in ("gate", "check", "item", "value", "status")) + " |" for r in ROWS]
        (OUT / "CHECKS.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    bad = [r for r in ROWS if r["status"] == "FAIL"]
    print(f"проверок {len(ROWS)}; FAIL {len(bad)}")
    for r in bad[:40]:
        print("FAIL", r["check"], r["item"], str(r["value"])[:300])
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
