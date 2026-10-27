"""Реестр отбора A (Э1-в, Ж-039 усл. 3): строка на пробу S1–S6. Правила 19:35: соседей >= 3,
сторон/год >= 0,5 базы tune 2018. Только дополняется (otbor_statistika п. 4). Данные >= 2022 не читаются."""
import csv
import json
import os

NOVYE = "docs/tables/sweep_e1v2"
STARYE = "docs/tables/sweep_e1"
VYHOD = "docs/otbor/reestr_otbora_A.csv"
STRATEGII = [f"S{k}" for k in range(1, 7)]
POROG_AKT = 0.5
MIN_SOSEDEY = 3
TOP_K = 3
SHAPKA = "# реестр только дополняется (otbor_statistika п. 4); разработка, train 2018–2019; validation не открыта\n"
POLYA = ["strategiya", "klyuch", "rezhim", "istochnik", "tune_004", "ctrl_004", "tune_0035",
         "ctrl_0035", "storon_god", "plato", "sosedey", "aktivnost_ok", "plato_ok",
         "vyrozhdena", "status", "prichina"]
FAJLY = [("oat_0.04", "004", "oat"), ("oat_0.035", "0035", "oat"),
         ("factorial_0.04", "004", "factorial"), ("factorial_0.035", "0035", "factorial"),
         ("optuna_0.04", "004", "optuna"), ("optuna_0.035", "0035", "optuna"),
         ("plato_0.04", "004", "plato")]
PRIORITET = {"plato": 0, "factorial": 1, "optuna": 2, "oat": 3}


def chitat(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(l for l in f if not l.startswith("#")))


def norm(v):
    if v is None or v in ("", "null", "None"):
        return None
    if v in ("True", "False"):
        return float(v == "True")
    try:
        return float(v)
    except (TypeError, ValueError):
        return str(v)


def kanon_ledger(key):
    return tuple(sorted((n, norm(v)) for n, v in json.loads(key)))


def kanon_stroki(row):
    return tuple(sorted((c[2:], norm(v)) for c, v in row.items() if c.startswith("p.")))


def klyuch_str(kanon):
    return ";".join(f"{n}={v:g}" if isinstance(v, float) else f"{n}={v}"
                    for n, v in kanon if v is not None)


def num(row, *names):
    for n in names:
        v = norm(row.get(n))
        if isinstance(v, float):
            return v
    return None


def sobrat_tablicy(s, papka=NOVYE):
    """kanon -> сведения из таблиц S{k}_*.csv (метрики tune/ctrl, сторон/год, плато, соседи)."""
    res = {}
    for imya, kom, rezhim in FAJLY:
        for r in chitat(f"{papka}/{s}_{imya}.csv"):
            if not any(c.startswith("p.") for c in r):
                continue
            d = res.setdefault(kanon_stroki(r), {"rezhimy": set(), "prio": 9})
            d["rezhimy"].add(rezhim)
            d["tune_" + kom] = num(r, "tune_sharpe")
            d["ctrl_" + kom] = num(r, "ctrl_sharpe")
            if rezhim == "plato":
                d["tune_0035"] = num(r, "tune_sharpe_c0.035")
                d["ctrl_0035"] = num(r, "ctrl_sharpe_c0.035")
            if kom == "004" and PRIORITET[rezhim] < d["prio"]:
                d["prio"] = PRIORITET[rezhim]
                d["storon"] = num(r, "tune_sdelok_y", "sdelok_y")
                d["plato"] = num(r, "plateau")
                d["sos"] = num(r, "neighbors")
    return res


def baza_storon(s, papka=NOVYE):
    for r in chitat(f"{papka}/{s}_oat_0.04.csv"):
        if r.get("param") == "base":
            return num(r, "tune_sdelok_y")
    return None


def kandidaty(s, papka=NOVYE):
    rows = chitat(f"{papka}/{s}_kandidaty_plato.csv")
    return {kanon_stroki(r) for r in rows if int(r.get("rang") or r.get("rank")) <= TOP_K}


def ocenit(d, baza, kandidat, porog=POROG_AKT, min_sos=MIN_SOSEDEY):
    """Флаги и статус одной пробы. d: storon, sos (None = не измерено)."""
    st, sos = d.get("storon"), d.get("sos")
    akt = None if st is None or baza is None else st >= porog * baza
    pl = None if sos is None else sos >= min_sos
    vyr = akt is False or pl is False
    if kandidat and not vyr and akt and pl:
        status, why = "kandidat_zamorozki", "топ-3 по плато, активность и плато пройдены"
    elif vyr:
        parts = ([f"активность {st:g} < {porog:g}×базы {baza:g} сторон/год"] if akt is False else []) \
            + ([f"плато: соседей {sos:g} < {min_sos}"] if pl is False else [])
        status, why = "otbroshena_vyrozhdena", "; ".join(parts)
    elif akt is None or pl is None:
        status, why = "otbroshena_rang", "активность или плато не измерены (проба только в ledger/OAT)"
    else:
        status, why = "otbroshena_rang", "правила пройдены, но не вошла в топ-3 по плато"
    return {"aktivnost_ok": flag(akt), "plato_ok": flag(pl), "vyrozhdena": flag(vyr),
            "status": status, "prichina": why}


def godovoy(sr_dnevnoy):
    """Ledger хранит дневной Шарп; в таблицах — годовой (проверено: отношение = sqrt(252))."""
    return sr_dnevnoy * 252 ** 0.5


flag = lambda x: "" if x is None else str(int(x))  # noqa: E731
fmt = lambda x: "" if x is None else repr(x)  # noqa: E731


def stroki_novye(s):
    tab, baza, kand = sobrat_tablicy(s), baza_storon(s), kandidaty(s)
    led = json.load(open(f"{NOVYE}/_ledger/{s}/ledger.json", encoding="utf-8"))
    out = []
    for key, obj in led.items():
        k = kanon_ledger(key)
        d = tab.get(k, {"rezhimy": {"ledger"}})
        d.setdefault("tune_004", godovoy(obj))
        row = {"strategiya": s, "klyuch": klyuch_str(k), "istochnik": "sweep_e1v2",
               "rezhim": "|".join(sorted(d["rezhimy"], key=lambda m: PRIORITET.get(m, 9))),
               "tune_004": fmt(d.get("tune_004")), "ctrl_004": fmt(d.get("ctrl_004")),
               "tune_0035": fmt(d.get("tune_0035")), "ctrl_0035": fmt(d.get("ctrl_0035")),
               "storon_god": fmt(d.get("storon")), "plato": fmt(d.get("plato")),
               "sosedey": fmt(d.get("sos"))}
        row.update(ocenit(d, baza, k in kand))
        out.append(row)
    return out


def stroki_starye(s):
    path = f"{STARYE}/_ledger/{s}/ledger.json"
    if not os.path.exists(path):
        return []
    return [{"strategiya": s, "klyuch": klyuch_str(kanon_ledger(k)), "rezhim": "ledger",
             "istochnik": "sweep_e1", "tune_004": fmt(godovoy(v)), "status": "do_ispravleniya_znaka",
             "prichina": "до исправления знака, не используется (Ж-038)"}
            for k, v in json.load(open(path, encoding="utf-8")).items()]


kluch_strok = lambda r: (r["strategiya"], r["istochnik"], r["klyuch"])  # noqa: E731


def sliyanie(suschestvuyuschie, novye):
    """Только дополняет: старые строки остаются как есть, добавляются лишь неизвестные ключи."""
    est = {kluch_strok(r) for r in suschestvuyuschie}
    return list(suschestvuyuschie) + [r for r in novye if kluch_strok(r) not in est]


def zapisat(rows, path=VYHOD):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(SHAPKA)
        w = csv.DictWriter(f, POLYA, restval="")
        w.writeheader()
        w.writerows(rows)


def svodka(rows):
    """По стратегии: всего / вырожденных / кандидатов / вырожденных среди 10 пиков по tune."""
    res = {}
    for s in STRATEGII:
        mine = [r for r in rows if r["strategiya"] == s and r["istochnik"] == "sweep_e1v2"]
        izm = sorted((r for r in mine if r["storon_god"]), key=lambda r: -float(r["tune_004"]))
        kan = [r for r in mine if r["status"] == "kandidat_zamorozki"]
        luchshiy = max((float(r["tune_004"]) for r in kan), default=None)
        res[s] = {"vsego": len(mine), "vyrozhdeno": sum(r["vyrozhdena"] == "1" for r in mine),
                  "kandidatov": len(kan),
                  "pik_top10_vyrozhd": sum(r["vyrozhdena"] == "1" for r in izm[:10]),
                  "vyrozhd_vyshe_kandidata": sum(r["vyrozhdena"] == "1" and luchshiy is not None
                                                 and float(r["tune_004"]) > luchshiy for r in mine)}
    return res


def main():
    novye = [r for s in STRATEGII for r in stroki_novye(s) + stroki_starye(s)]
    rows = sliyanie(chitat(VYHOD), novye)
    zapisat(rows)
    for s, v in svodka(rows).items():
        print(s, v)


if __name__ == "__main__":
    main()
