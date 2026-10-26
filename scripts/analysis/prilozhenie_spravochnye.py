"""Таблица приложения «Справочные модели» (глава 3) из готовых CSV, без пересчёта моделей.

Входы (только разработочный набор, OOS, годы оценки 2017–2021; тестовые годы не читаются):
  data/regen/ref_nn_garch/REF_metrics_dev.csv  — LSTM, GRU, GARCH, GJR, HAR-J с поправкой (PR #97)
  data/regen/final3/designs_rank_{cand}_h{h}.csv — финал главы 3 (FINAL_DESIGN.md, поле 10)
  прежний справочный GRU 25.10 — числа из docs/otchety/2026-10-25_gru_sv.md (снят, протокол нечестный)

Выход: docs/tables/dlya_teksta/prilozhenie_spravochnye.csv («;», десятичная запятая, UTF-8 BOM).
Запуск: python3 scripts/analysis/prilozhenie_spravochnye.py [--data <корень с data/regen>]
"""
import argparse
import csv
import hashlib
import os

YEARS = [2017, 2018, 2019, 2020, 2021]
HS = [1, 5, 22]
REF = "data/regen/ref_nn_garch/REF_metrics_dev.csv"
REF_SHA = "2acad6efdb732566e941f6198c6c5fdf3f1ffc0ea3f17540c70b5f9be839e034"  # отчёт 2026-10-26_ref_nn_garch.md
FINAL = {1: ("D22_h1", "top5|V0_XGB|B-sm|L0"),
         5: ("D12_h5", "wrap2|V0_XGB|B-sm|L0"),
         22: ("HARJ_corr", "single|V0_HAR|B-sm|L0")}
GRU_2510 = {  # docs/otchety/2026-10-25_gru_sv.md, таблица «Цель (4)»
    1: ([0.2509, 0.3510, 0.3008, 0.3870, 0.2347], 0.3049, "B-sm"),
    5: ([0.1469, 0.2356, 0.2003, 0.3897, 0.1639], 0.2273, "B-sm"),
    22: ([0.0700, 0.1642, 0.1133, 0.6899, 0.1464], 0.2367, "B-smrob"),
}
# (имя в REF_metrics_dev.csv, подпись, статус)
ROWS = [
    ("LSTM", "LSTM (ансамбль seed 42/43/44, Б0.1)", "основной"),
    ("GRU", "GRU (ансамбль seed 42/43/44, Б0.1)", "основной"),
    ("GARCH", "GARCH(1,1) (калибровка a·F)", "основной"),
    ("GJR", "GJR-GARCH(1,1) (калибровка a·F)", "основной"),
    ("LSTM seed 42", "LSTM seed 42 (Б0.1)", "справочный, без выбора"),
    ("LSTM seed 43", "LSTM seed 43 (Б0.1)", "справочный, без выбора"),
    ("LSTM seed 44", "LSTM seed 44 (Б0.1)", "справочный, без выбора"),
    ("LSTM без Б0.1", "LSTM ансамбль без Б0.1", "справочный, без выбора"),
    ("GRU seed 42", "GRU seed 42 (Б0.1)", "справочный, без выбора"),
    ("GRU seed 43", "GRU seed 43 (Б0.1)", "справочный, без выбора"),
    ("GRU seed 44", "GRU seed 44 (Б0.1)", "справочный, без выбора"),
    ("GRU без Б0.1", "GRU ансамбль без Б0.1", "справочный, без выбора"),
    ("GARCH масштаб без F", "GARCH(1,1) масштаб a без F", "справочный, без выбора"),
    ("GARCH без калибровки", "GARCH(1,1) без калибровки", "справочный, без выбора"),
    ("GJR масштаб без F", "GJR-GARCH(1,1) масштаб a без F", "справочный, без выбора"),
    ("GJR без калибровки", "GJR-GARCH(1,1) без калибровки", "справочный, без выбора"),
]
HAR = "HAR-J с поправкой (sm)"


def num(x, d=4):
    return "" if x is None else f"{x:.{d}f}".replace(".", ",")


def pct(x, base):
    return f"{(x / base - 1) * 100:+.1f}".replace(".", ",")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=".", help="корень, где лежит data/regen (данные не в git)")
    ap.add_argument("--out", default="docs/tables/dlya_teksta/prilozhenie_spravochnye.csv")
    a = ap.parse_args()

    ref_path = os.path.join(a.data, REF)
    sha = hashlib.sha256(open(ref_path, "rb").read()).hexdigest()
    assert sha == REF_SHA, f"хэш {REF} не совпал с отчётом: {sha}"

    m = {}
    for r in csv.DictReader(open(ref_path, encoding="utf-8")):
        y = int(r["year"])
        assert y in YEARS, f"в {REF} есть год вне разработочного набора: {y}"
        m[(int(r["h"]), r["model"], y)] = (float(r["qlike"]), float(r["rmse"]), float(r["mae"]))

    def ref_row(h, name):
        q = [m[(h, name, y)][0] for y in YEARS]
        rm = sum(m[(h, name, y)][1] for y in YEARS) / 5
        ma = sum(m[(h, name, y)][2] for y in YEARS) / 5
        return q, sum(q) / 5, rm, ma

    src_ref = f"{REF} (PR #97)"
    out = [["h", "модель", "статус", *map(str, YEARS), "среднее 2017–2021",
            "к HAR-J с поправкой, %", "RMSE, RV×10⁴ (среднее лет)", "MAE, RV×10⁴ (среднее лет)", "источник"]]
    for h in HS:
        hq, hmean, hrm, hma = ref_row(h, HAR)
        out.append([h, "HAR-J с поправкой (s² sm), эталон", "эталон", *map(num, hq), num(hmean), "",
                    num(hrm, 3), num(hma, 3), src_ref])

        cand, design = FINAL[h]
        fp = f"data/regen/final3/designs_rank_{cand}_h{h}.csv"
        fr = next(r for r in csv.DictReader(open(os.path.join(a.data, fp), encoding="utf-8"))
                  if r["design"] == design)
        fq = [float(fr[f"q_{y}"]) for y in YEARS]
        fmean = float(fr["mean_1721"])
        assert abs(sum(fq) / 5 - fmean) < 1e-9
        assert abs(float(fr["har_1721"]) - hmean) < 5e-5, "эталон финала и PR #97 расходятся"
        out.append([h, f"финал главы 3: {cand} {design}", "финал (FINAL_DESIGN.md, поле 10)",
                    *map(num, fq), num(fmean), pct(fmean, hmean), "", "", fp])

        for name, label, status in ROWS:
            q, mean, rm, ma = ref_row(h, name)
            out.append([h, label, status, *map(num, q), num(mean), pct(mean, hmean),
                        num(rm, 3), num(ma, 3), src_ref])

        gq, gmean, s2 = GRU_2510[h]
        out.append([h, f"GRU справочный 25.10 (gru_sv.py --oos-params, {s2})",
                    "снят: число эпох — ранней остановкой по году оценки",
                    *map(num, gq), num(gmean), pct(gmean, hmean), "", "",
                    "docs/otchety/2026-10-25_gru_sv.md, таблица «Цель (4)»"])

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8-sig", newline="") as f:
        csv.writer(f, delimiter=";", lineterminator="\n").writerows(out)
    print("OK", a.out, len(out) - 1, "строк")


if __name__ == "__main__":
    main()
