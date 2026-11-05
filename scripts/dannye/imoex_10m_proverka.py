"""Ж-201: сверка 10-мин баров IMOEX (ISS) с SBER и с дневными рядами IMOEX.

Вход: csv из `moex_iss index-days` (docs/tables/imoex_10m/days_*.csv, index_d_imoex.csv) и
data/iss_mix_2015_2019/IMOEX_2014.csv, IMOEX.csv (только чтение). Только стандартная библиотека.
Выход: docs/tables/imoex_10m/kalendar_vs_sber.csv, close_vs_dnevnoj.csv; сводка — в stdout.
Запуск из корня репозитория: python3 -I scripts/dannye/imoex_10m_proverka.py
"""
import csv
import json
from collections import Counter

T = "docs/tables/imoex_10m/"
MIX = ["data/iss_mix_2015_2019/IMOEX_2014.csv", "data/iss_mix_2015_2019/IMOEX.csv"]
FROM, TILL = "2014-06-01", "2026-02-27"
MIN_BARS = 30


def days(path):
    with open(path, newline="", encoding="utf-8") as f:
        return {r["date"]: r for r in csv.DictReader(f)}


def kalendar(im, sb):
    """Строки расхождений календаря и сводка по дням в [FROM; TILL]."""
    rows, c = [], Counter()
    for d in sorted(set(im) | set(sb)):
        if not FROM <= d <= TILL:
            continue
        a = int(im[d]["n_main"]) if d in im else None
        b = int(sb[d]["n_main"]) if d in sb else None
        flags = []
        if a is None:
            flags.append("нет у IMOEX")
        if b is None:
            flags.append("нет у SBER")
        if a is not None and a < MIN_BARS:
            flags.append(f"IMOEX < {MIN_BARS}")
        if a is not None and b is not None and a != b:
            flags.append("число баров разное")
        c["дней всего"] += 1
        if not flags:
            c["совпало"] += 1
        for fl in flags:
            c[fl] += 1
        if flags:
            rows.append({"date": d, "n_imoex": "" if a is None else a, "n_sber": "" if b is None else b,
                         "flags": ";".join(flags)})
    return rows, c


def mix_close():
    out = {}
    for p in MIX:
        with open(p, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r["CLOSE"]:
                    out[r["TRADEDATE"]] = float(r["CLOSE"])
    return out


def close_check(im, daily, name, col="close_last_main"):
    rows, bad = [], 0
    for d, ref in sorted(daily.items()):
        if d not in im or not im[d][col]:
            continue
        x = float(im[d][col])
        rel = (x - ref) / ref
        rows.append({"source": name, "column": col, "date": d, "close_10m": x, "close_daily": ref,
                     "rel_diff": f"{rel:.3e}"})
        bad += abs(rel) > 1e-4
    return rows, bad


def stat(rows):
    v = sorted(abs(float(r["rel_diff"])) for r in rows)
    if not v:
        return {}
    return {"n": len(v), "median": v[len(v) // 2], "p90": v[int(len(v) * 0.9)], "max": v[-1],
            "точно(<1e-6)": sum(x < 1e-6 for x in v), ">1e-4": sum(x > 1e-4 for x in v)}


def write(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["empty"])
        w.writeheader()
        w.writerows(rows)


def main():
    im, sb = days(T + "days_imoex.csv"), days(T + "days_sber.csv")
    rows, cal = kalendar(im, sb)
    write(T + "kalendar_vs_sber.csv", rows)
    daily = {d: float(r["close"]) for d, r in days(T + "index_d_imoex.csv").items()}
    mix = mix_close()
    mix = {d: v for d, v in mix.items() if d not in daily}
    allr, st = [], {}
    for col in ("close_last_main", "close_last_1850"):
        for nm, ref in (("index_d", daily), ("iss_mix", mix)):
            r, _ = close_check(im, ref, nm, col)
            allr += r
            st[f"{nm}/{col}"] = stat(r)
    write(T + "close_vs_dnevnoj.csv", allr)
    first = min(im)
    print(json.dumps({"first_day_imoex": first, "last_day_imoex": max(im), "calendar": cal, "close": st,
                      "bars_distribution": Counter(r["n_main"] for r in im.values()).most_common(6)},
                     ensure_ascii=False, indent=1))


def self_check():
    im = {"2020-01-09": {"n_main": "52"}, "2020-01-10": {"n_main": "20"}, "2020-01-13": {"n_main": "52"}}
    sb = {"2020-01-09": {"n_main": "52"}, "2020-01-10": {"n_main": "20"}, "2020-01-14": {"n_main": "52"}}
    rows, c = kalendar(im, sb)
    assert c["совпало"] == 1 and c["нет у SBER"] == 1 and c["нет у IMOEX"] == 1 and c[f"IMOEX < {MIN_BARS}"] == 1, c
    r, bad = close_check({"2020-01-09": {"close_last_main": "101"}}, {"2020-01-09": 100.0}, "t")
    assert bad == 1 and len(r) == 1


if __name__ == "__main__":
    self_check()
    main()
