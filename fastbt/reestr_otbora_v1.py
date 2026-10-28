"""Реестр отбора A на В1 (Ж-075 п. 2): годы — колонки по периоду, только дополнение (otbor_statistika п. 4).

Периоды: `train` — 2018, 2019 (старый реестр `reestr_otbora_A.csv`, только чтение; пишет `reestr_otbora.py`);
`train4` — 2016…2019 и итог train4 (новый `reestr_otbora_A_v1.csv`). Все кандидаты и все метрики — строка на
оценённую конфигурацию; пропуск кандидата — ошибка. Данные >= 2020 не читаются."""
import csv
import os

PERIODY = {"train": (2018, 2019), "train4": (2016, 2017, 2018, 2019)}
VYHOD = {"train": "docs/otbor/reestr_otbora_A.csv", "train4": "docs/otbor/reestr_otbora_A_v1.csv"}
SHAPKA = "# реестр только дополняется (otbor_statistika п. 4); разработка, {p}: {g}; validation не открыта\n"
BAZA_POLYA = ["strategiya", "klyuch", "rezhim", "istochnik"]
HVOST = ["storon_god", "plato", "sosedey", "aktivnost_ok", "plato_ok", "vyrozhdena", "status", "prichina"]


def kolonki(period):
    """Шапка периода: общие поля, затем tune/ctrl-Шарп на каждый год и итог периода, затем флаги."""
    god = [f"{k}_{y}" for y in PERIODY[period] for k in ("sharpe", "sdelok")]
    return BAZA_POLYA + god + [f"sharpe_{period}", "komissiya"] + HVOST


def kluch_strok(r):
    return (r["strategiya"], r["istochnik"], r["klyuch"])


def chitat(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(l for l in f if not l.startswith("#")))


def proverit(rows, period):
    """Каждая строка несёт ровно колонки периода (год 2016 не может выпасть молча)."""
    pol = set(kolonki(period))
    for r in rows:
        if set(r) != pol:
            raise ValueError(f"колонки строки {kluch_strok(r) if BAZA_POLYA[0] in r else r} "
                             f"не совпадают с периодом {period}: {sorted(set(r) ^ pol)}")


def dopisat(rows, period, n_ocenennyh, path=None):
    """Дописывает неизвестные ключи; len(rows) обязана равняться числу оценённых конфигураций.
    Существующие строки не меняются; шапка файла другого периода — отказ. Возвращает число добавленных."""
    path = path or VYHOD[period]
    if len(rows) != n_ocenennyh:
        raise ValueError(f"оценено {n_ocenennyh}, к записи {len(rows)}: кандидат пропущен")
    proverit(rows, period)
    est = chitat(path)
    cols = kolonki(period)
    if est and list(est[0]) != cols:
        raise ValueError(f"шапка {path} не совпадает с периодом {period}")
    known = {kluch_strok(r) for r in est}
    novye = [r for r in rows if kluch_strok(r) not in known]
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    nov_file = not os.path.exists(path)
    with open(path, "a", encoding="utf-8", newline="") as f:
        if nov_file:
            f.write(SHAPKA.format(p=period, g=", ".join(map(str, PERIODY[period]))))
        w = csv.DictWriter(f, cols, restval="")
        if nov_file:
            w.writeheader()
        w.writerows(novye)
    return len(novye)
