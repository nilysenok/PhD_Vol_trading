"""Ж-041: S3–S6, лучший воспроизводимый против лучшего невоспроизводимого кандидата.

Диагностика, train 2018–2019. Выбор по min(tune, ctrl) при 0,04 % — это 2019, в
отличие от заморозки (по tune); заморозку не меняет. Источник — реестр отбора.
Допустимость по правилам 19:35: соседей >= 3, активность >= 0,5 базы (флаги реестра).
Запуск: python3 fastbt/zh041_vosp_vs_nevosp.py
"""
import csv

REESTR = "docs/otbor/reestr_otbora_A.csv"
OUT = "docs/tables/sweep_e1v2/S3_S6_vosp_vs_nevosp.csv"
PORIG = 0.2
HEAD = ("# разработка, train 2018–2019; validation не открыта; Ж-041. Диагностика: выбор по "
        "min(tune, ctrl) при 0,04 % (с участием 2019), не по tune, как в заморозке; "
        "заморозку не меняет. Допустимы по 19:35 (соседей >= 3, активность >= 0,5 базы); "
        "воспроизводимый = ekviv_s9 'da', невоспроизводимый = status otbroshena_net_etalona.\n")


def mekhanizm(x):
    """Механизм = причина отсутствия эталона из реестра: sqrt_rv / maska / trail_n."""
    t = x["prichina"]
    if "√RV" in t:
        return "sqrt_rv: " + t
    if "trail_n" in t:
        return "trail_n: " + t
    return "maska: " + t


def dopustim(x):
    try:
        return (x["aktivnost_ok"] == "1" and x["plato_ok"] == "1"
                and float(x["sosedey"]) >= 3 and x["vyrozhdena"] != "1")
    except ValueError:
        return False


def tselevaya(x):
    t, c = float(x["tune_004"]), float(x["ctrl_004"])
    return min(t, c), t, c


def luchshiy(rows):
    return max(rows, key=lambda x: tselevaya(x)[0]) if rows else None


def main():
    with open(REESTR, encoding="utf-8") as f:
        rows = list(csv.DictReader(l for l in f if not l.startswith("#")))
    out = []
    for s in ("S3", "S4", "S5", "S6"):
        mine = [x for x in rows if x["strategiya"] == s and dopustim(x)]
        v = luchshiy([x for x in mine if x["ekviv_s9"] == "da"])
        n = luchshiy([x for x in mine if x["status"] == "otbroshena_net_etalona"])
        mv, tv, cv = tselevaya(v)
        mn, tn, cn = tselevaya(n)
        out.append([s, v["klyuch"], tv, cv, mv, n["klyuch"], tn, cn, mn, mekhanizm(n),
                    mn - mv, int(mn - mv > PORIG)])
    with open(OUT, "w", encoding="utf-8", newline="") as f:
        f.write(HEAD)
        w = csv.writer(f)
        w.writerow(["strategiya", "vosp_klyuch", "vosp_tune", "vosp_ctrl", "vosp_min",
                    "nevosp_klyuch", "nevosp_tune", "nevosp_ctrl", "nevosp_min",
                    "nevosp_mekhanizm", "raznitsa_min", "flag_gt_0_2"])
        w.writerows(out)
    for r in out:
        print(r[0], *(f"{z:.3f}" if isinstance(z, float) else z for z in r[2:5] + r[6:9] + r[9:]))


if __name__ == "__main__":
    main()
