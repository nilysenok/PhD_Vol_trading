"""V6-14, часть 2: правило заморозки новых осей v6 и выход Морриса (Ж-132, решение дирижёра 29.10 13:20). Данных не читает.

Правило, объявленное до счёта: новая ось замораживается на «выкл.», если во КАЖДОМ из трёх окон (≤2016, ≤2017, ≤2018)
n_akt ≥ 20 и U95 < 0,05 (единицы Шарпа). n_akt < 20 в любом окне → «не измерено», не замораживается; NaN в U95 — не заморозка.
Оси v5 не замораживаются никогда. Замороженный родитель замораживает дочерние новые оси (инвентаризация §3.2). 2019 в решение не входит.
Выход: morris_S{k}.csv (колонки README `fastbt/designs/v6`, шапка `# …`) и zamorozka.yaml (`v: 1`, дата, sha входов, оси).
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import yaml

OKNA = ("<=2016", "<=2017", "<=2018")
PORIG_U, MIN_AKT = 0.05, 20
MERTVAYA, MALO = "мёртвая: не оценена", "мало шагов"
NET_SDELOK, DOLYA_NOL = "не оценена: нет сделок", 0.5      # Д4: Y целиком 0 при статусе ok; доля > 50 % у Sk — «ВНИМАНИЕ» в итоге драйвера
KOLONKI =("os", "gruppa", "novaya", "okno", "n_akt", "mu_star", "sigma", "U95", "zamorozit", "prichina")


def _nol(stat: dict, i: int) -> bool:
    """Ось i имела активные шаги, отброшенные как «обе точки без сделок» (stat[окно]['n_nol'], если задан)."""
    return any(int(stat[w].get("n_nol", np.zeros(i + 1, int))[i]) > 0 for w in OKNA)


def reshenie(osi: list[dict], stat: dict[str, dict]) -> dict[str, tuple[bool, str]]:
    """osi: [{os, gruppa, novaya, roditel?}]; stat[okno] = результат morris_v6.statistika (массивы по осям в порядке osi).
    Возвращает {ось: (замораживать, причина)}."""
    if tuple(stat) != OKNA:
        raise ValueError(f"нужны ровно окна {OKNA}, получено {tuple(stat)}")
    res: dict[str, tuple[bool, str]] = {}
    for i, o in enumerate(osi):
        n = [int(stat[w]["n_akt"][i]) for w in OKNA]
        u = [float(stat[w]["U95"][i]) for w in OKNA]
        if not o["novaya"]:
            res[o["os"]] = (False, "ось v5: не замораживается")
        elif max(n) == 0 and _nol(stat, i):
            res[o["os"]] = (False, f"не измерено, {NET_SDELOK}: n_akt {n}, шаги были, но обе точки пары без сделок (Д4)")
        elif max(n) == 0:
            res[o["os"]] = (False, f"не измерено, {MERTVAYA}: n_akt {n}, ось не заморожена и не значима")
        elif min(n) < MIN_AKT:
            res[o["os"]] = (False, f"не измерено, {MALO}: n_akt {n} < {MIN_AKT}")
        elif all(x < PORIG_U for x in u):  # NaN < 0,05 ложно
            res[o["os"]] = (True, f"U95 {[round(x, 4) for x in u]} < {PORIG_U} в трёх окнах")
        else:
            res[o["os"]] = (False, f"U95 {[round(x, 4) if np.isfinite(x) else 'nan' for x in u]}: эффект не нулевой")
    izmenilos = True
    while izmenilos:  # дочерние новые оси замороженного родителя; цепочки родитель → дочерняя → внучатая
        izmenilos = False
        for o in osi:
            r = o.get("roditel")
            if o["novaya"] and r and res.get(r, (False,))[0] and not res[o["os"]][0]:
                res[o["os"]] = (True, f"родитель {r} заморожен")
                izmenilos = True
    return res


def zapisat_csv(path: Path, shapka: dict, osi: list[dict], stat: dict[str, dict], res: dict) -> None:
    """morris_S{k}.csv: шапка `# ключ: значение`, затем строка на (ось, окно)."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        f.write('# "v": "1"\n# статус: разработка train4, до аудита\n')
        for k, v in shapka.items():
            f.write(f"# {k}: {v}\n")
        w = csv.writer(f)
        w.writerow(KOLONKI)
        for i, o in enumerate(osi):
            zam, pr = res[o["os"]]
            for okno in OKNA:
                s = stat[okno]
                w.writerow([o["os"], o["gruppa"], o["novaya"], okno, int(s["n_akt"][i]),
                            *(f"{float(s[c][i]):.6g}" for c in ("mu_star", "sigma", "U95")), zam, pr])


def pometki(osi: list[dict], stat: dict[str, dict]) -> dict:
    """{ось: {n_akt: [по окнам], pometka}} — активные шаги всех осей (и v5); метка «мёртвая: не оценена» (0 шагов во всех окнах),
    «мало шагов» (< MIN_AKT в каком-то окне) или None."""
    out = {}
    for i, o in enumerate(osi):
        n = [int(stat[w]["n_akt"][i]) for w in OKNA]
        mert = NET_SDELOK if _nol(stat, i) else MERTVAYA
        out[o["os"]] = {"n_akt": n, "pometka": mert if max(n) == 0 else MALO if min(n) < MIN_AKT else None}
    return out


def zapisat_yaml(path: Path, data: str, vhody: dict, res: dict, statusy: dict | None = None, osi_akt: dict | None = None,
                  nol_sdelok: dict | None = None) -> list[str]:
    """zamorozka.yaml: v, дата, sha входов (kub_sha, kod, csv), правило, список замороженных осей. Возвращает список.
    statusy — счётчик ok / ne_schitana / none_kanon / oshibka; osi_akt — `pometki`: активные шаги и метки «мёртвая»/«мало шагов»."""
    zam = sorted(k for k, (z, _) in res.items() if z)
    d = {"v": 1, "data": data, "vhody": vhody, "pravilo": {"okna": list(OKNA), "U95_lt": PORIG_U, "n_akt_ge": MIN_AKT},
         "osi": zam, "prichiny": {k: res[k][1] for k in zam}}
    if statusy is not None:
        d["statusy"] = statusy
    if nol_sdelok is not None:        # {chislo, ok, dolya}: точки ok с Y целиком 0 (0 сделок во всех окнах Y)
        d["nol_sdelok"] = nol_sdelok
    if osi_akt is not None:
        d["osi_aktivnye"] = osi_akt
        d["ne_ocenyeny_net_sdelok"] = sorted(k for k, v in osi_akt.items() if v["pometka"] == NET_SDELOK)
        d["mertvye_ne_ocenyeny"] = sorted(k for k, v in osi_akt.items() if v["pometka"] == MERTVAYA)
        d["malo_shagov"] = sorted(k for k, v in osi_akt.items() if v["pometka"] == MALO)
    path.write_text(yaml.safe_dump(d, allow_unicode=True, sort_keys=False), "utf-8")
    return zam
