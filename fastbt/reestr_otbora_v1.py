"""Реестр отбора A на В1 (Ж-075 п. 2): годы — колонки по периоду, только дополнение (otbor_statistika п. 4).

Периоды: `train` — 2018, 2019 (старый реестр `reestr_otbora_A.csv`, только чтение; пишет `reestr_otbora.py`);
`train4` — 2016…2019 и итог train4 (новый `reestr_otbora_A_v1.csv`). Все кандидаты и все метрики — строка на
оценённую конфигурацию; пропуск кандидата — ошибка. Данные >= 2020 не читаются."""
import csv
import hashlib
import math
import os
import pickle
from pathlib import Path

LED_V1 = "docs/tables/zh051/_ledger_v1"

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


def _chitat_ledger(root, sk, it):
    """Pickle ledger_it{it}.pkl со сверкой sha256 по SHA256SUMS.txt (нет строки или хэш иной — отказ)."""
    p, key = Path(root) / sk / f"ledger_it{it}.pkl", f"{sk}/ledger_it{it}.pkl"
    sums = Path(root) / "SHA256SUMS.txt"
    strok = [ln.split() for ln in sums.read_text(encoding="utf-8").splitlines()] if sums.exists() else []
    h = next((s[0] for s in strok if len(s) > 1 and s[1] == key), None)
    if h is None or not p.exists():
        raise ValueError(f"{key}: нет файла или строки в SHA256SUMS")
    if hashlib.sha256(p.read_bytes()).hexdigest() != h:
        raise ValueError(f"{key}: sha256 не совпал")
    return pickle.loads(p.read_bytes())


def _stroka_ledger(sk, it, key, params, res, period):
    """Строка реестра из записи ledger: Шарп и сделки по годам периода; итог — min по годам (нет года -> пусто)."""
    by, sd = res.get("by_year") or {}, res.get("sdelok_y") or {}
    r = {c: "" for c in kolonki(period)}
    sh = []
    for y in PERIODY[period]:
        v, n = by.get(y, by.get(str(y))), sd.get(y, sd.get(str(y)))
        r[f"sharpe_{y}"] = "" if v is None else repr(float(v))
        r[f"sdelok_{y}"] = "" if n is None else repr(float(n))
        sh.append(v)
    ok = all(v is not None and math.isfinite(float(v)) for v in sh)
    r["sharpe_" + period] = repr(min(float(v) for v in sh)) if ok else ""
    if res.get("vyrozhdena") is not None:
        r["vyrozhdena"] = str(int(bool(res["vyrozhdena"])))
    kl = ";".join(f"{a}={b}" for a, b in sorted(params.items())) + f";period={key[1]}"
    r.update(strategiya=sk, klyuch=kl, rezhim="ledger", istochnik=f"zh051_v1_it{it}", status="ledger",
             prichina="строка ledger В1; ворота и плато в ledger не хранятся")
    return r


def chislo_konfiguracij(rows):
    """Число различных конфигураций (стратегия, ключ); повтор под другой итерацией не новый кандидат."""
    return len({(r["strategiya"], r["klyuch"]) for r in rows})


def iz_ledger(period, iteraciya, sk, root=LED_V1, path=None):
    """Дописывает ВСЕ записи ledger_it{iteraciya}.pkl стратегии sk; ключ — (итерация, sk, конфигурация);
    повтор ничего не добавляет. Возвращает число добавленных."""
    cache = _chitat_ledger(root, sk, iteraciya)
    rows = [_stroka_ledger(sk, iteraciya, k, par, res, period) for k, (par, res) in cache.items()]
    return dopisat(rows, period, len(cache), path)
