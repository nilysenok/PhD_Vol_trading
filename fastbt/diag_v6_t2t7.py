"""Диагностика v6 train4 (Ж-145), лист Д-3: таблицы (2), (7) и оговорки по строкам ledger 2.2 (читатель без tik/ — Д-1).
Топ-10 — `diag_v6_pravila.top10` (по дневному `sharpe`, ворота ≥ 30 сделок за окно = Σ sdelok_y), пороги — `PRAVILA`.
Пул Sk = rol=sobol из train4_v6_s1 ∪ train4_v6_s2, дубль klyuch_n — один раз. Данных ≥ 2020 не читает.
Запуск: `FASTBT_MAIN=$PWD python3 -m fastbt.diag_v6_t2t7 <ledger/> <out/>`."""
from __future__ import annotations

import csv
import datetime as dt
import json
import math
import sys
from pathlib import Path

import numpy as np

from fastbt import diag_v6_pravila as P
from fastbt.diag_v6_chtenie import ChitatelPort

PROHODY = {"s1": "train4_v6_s1", "s2": "train4_v6_s2"}
POLYA = ("klyuch_n", "strategiya", "status", "rol", "msr", "sharpe", "sdelok_y", "prichiny", "sloy", "sobol_i", "skew", "kurt")
SLOVAR = ("stop", "take", "trail", "bu", "vremya", "signal", "lim")   # словарь причин `etap1_eval.v11`; всё вне него — «прочее»
BRENT = (dt.date(2021, 11, 24), dt.date(2021, 12, 31))               # пропуск ряда Brent (толкования [31.10 00:12])
NAN = float("nan")


def _med(x) -> float:
    x = [v for v in x if v is not None and math.isfinite(v)]
    return float(np.median(x)) if x else NAN


def _summa(r: dict) -> float:
    return float(sum((r.get("sdelok_y") or {}).values()))


def _nul(r: dict) -> bool:
    """Нулевая точка куба: status ok и ни одной сделки. Строки ne_schitana/oshibka — не нули (их доля отдельно)."""
    return r.get("status") == "ok" and _summa(r) == 0.0


def _v_god(r: dict, god: str | None = None) -> float:
    sd = r.get("sdelok_y") or {}
    return float(sd.get(god, NAN)) if god else (_summa(r) / len(sd) if sd else NAN)


def _prochee(r: dict) -> float:
    """Доля выходов вне словаря причин = 1 − Σ словаря (part, pir, razryv, sessiya, …); у точки без сделок — NaN."""
    if _summa(r) == 0.0:
        return NAN
    s = sum(float((r.get("prichiny") or {}).get(k, 0.0)) for k in SLOVAR)
    if s > 1 + 1e-6:
        raise ValueError(f"{r.get('klyuch_n')}: Σ причин {s} > 1")
    return max(0.0, 1.0 - s)


def dni_brent_v_okne(gody) -> int:
    d, n = BRENT[0], 0                      # дней пропуска Brent, попадающих в годы окна метрик
    while d <= BRENT[1]:
        n += d.year in gody
        d += dt.timedelta(days=1)
    return n


def szhat(r: dict, prohod: str) -> dict:
    """Поля строки, нужные таблицам, + признаки оговорок. Ось, не записанная в klyuch, стоит в «выкл.» (kub: vykl)."""
    k = r.get("klyuch") or {}
    vn = k.get("vn_uslovie", "net") != "net"
    return {**{f: r.get(f) for f in POLYA}, "prohod": prohod, "brent": vn and k.get("vn_faktor", "brent") == "brent",
            "stop_reentry": r.get("strategiya") in ("S3", "S4") and k.get("vhod_tip", "stop_zayavka") == "stop_zayavka"
            and k.get("reentry", "net") == "da"}


def chitat_pul(root, sids=None) -> tuple:
    """(pul {sid: [строки]}, meta {sid: {gody, N: {s1, s2}, chitano: {s1, s2}}}); чтение по одному Sk и выборке (память)."""
    pul, meta = {}, {}
    for sid in sids or sorted({p.name for d in PROHODY.values() if (Path(root) / d).exists()
                                                       for p in (Path(root) / d).iterdir() if p.is_dir()}):
        seen, pul[sid], meta[sid] = set(), [], {"gody": None, "N": {}, "chitano": {}}
        for pn, d in PROHODY.items():
            if not (Path(root) / d / sid).exists():
                continue
            c = ChitatelPort(Path(root) / d, sid)
            ck = Path(root) / d / sid / "checkpoint.json"
            rs = [r for r in c.rows(status=None) if r.get("rol") == "sobol"]
            meta[sid]["gody"] = list(c.gody)
            meta[sid]["N"][pn] = json.loads(ck.read_text())["N"] if ck.exists() else None
            meta[sid]["chitano"][pn] = len(rs)
            pul[sid] += [szhat(r, pn) for r in rs if r["klyuch_n"] not in seen]
            seen |= {r["klyuch_n"] for r in rs}
            del c, rs
    return pul, meta


def top10_po_sk(pul: dict, min_sdelok: int = 30, pole: str | None = None) -> dict:
    return {sid: P.top10(rows, min_sdelok, pole) for sid, rows in pul.items()}


def _vybrosheno(meta: dict, prohody) -> float:
    pr = [p for p in prohody if p in meta["chitano"]]
    n = [meta["N"].get(p) for p in pr]
    return NAN if not n or None in n else float(sum(n) - sum(meta["chitano"][p] for p in pr))


def tablica_2(pul: dict, meta: dict) -> list:
    """Строки (sid, prohod ∈ s1/s2/vse): число точек по статусам, выброшено каноном (N − прочитано), доля нулей = нулевые ok / ok,
    доля ядра (sloy=fast); у vse — сделок в год у топ-10 (медиана по точкам и по годам) и слабые места по PRAVILA."""
    top, out = top10_po_sk(pul), []
    for sid, rows in pul.items():
        for pr, prohody in (("s1", ["s1"]), ("s2", ["s2"]), ("vse", ["s1", "s2"])):
            rs = [r for r in rows if r["prohod"] in prohody]
            ok = [r for r in rs if r["status"] == "ok"]
            row = dict(sid=sid, prohod=pr, n_strok=len(rs), n_ok=len(ok), n_ne_schitana=sum(r["status"] == "ne_schitana" for r in rs),
                       n_oshibka=sum(r["status"] == "oshibka" for r in rs), vybrosheno=_vybrosheno(meta[sid], prohody),
                       n_nul=sum(_nul(r) for r in rs), nuli_dolya=sum(_nul(r) for r in rs) / len(ok) if ok else NAN,
                       sloy_fast_dolya=sum(r["sloy"] == "fast" for r in rs) / len(rs) if rs else NAN)
            if pr == "vse":
                row.update(_sdelki_top(top[sid]))
                row["slabye"] = ";".join(k for k, v in _slabye_t2(row).items() if v)
            out.append(row)
    return out


def _sdelki_top(top: list) -> dict:
    d = {"top10_n": len(top), "sdelok_v_god_mediana": _med([_v_god(r) for r in top])}
    d.update({f"sdelok_{g}": _med([_v_god(r, g) for r in top]) for g in ("2016", "2017", "2018", "2019")})
    return d


def _slabye_t2(row: dict) -> dict:
    m, nd = row["sdelok_v_god_mediana"], row["nuli_dolya"]
    return {"sdelok_malo": math.isfinite(m) and m < P.PRAVILA["sdelok_v_god_mediana_min"],
            "nuli_mnogo": math.isfinite(nd) and nd > P.PRAVILA["nuli_dolya_max"]}


def tablica_7(pul: dict) -> list:
    """Причины выходов топ-10: по точке (rang 1…10) и медиана по Sk (rang «med»); столбец «prochee» обязателен."""
    out = []
    for sid, top in top10_po_sk(pul).items():
        rows = [dict(sid=sid, rang=i, sobol_i=r["sobol_i"], klyuch_n=r["klyuch_n"],
                     **{k: (r.get("prichiny") or {}).get(k, NAN) for k in SLOVAR}, prochee=_prochee(r)) for i, r in enumerate(top, 1)]
        med = {k: _med([x[k] for x in rows]) for k in (*SLOVAR, "prochee")}
        pr = med["prochee"]
        sl = "prochee_mnogo" if math.isfinite(pr) and pr > P.PRAVILA["prochee_dolya_max"] else ""
        out += rows + [dict(sid=sid, rang="med", sobol_i="", klyuch_n="", **med, slabye=sl)]
    return out


def ogovorki(pul: dict, meta: dict) -> list:
    """Оговорки по Sk: ядро (sloy), Brent (дней в окне и точек), стоп + reentry (разрывы в прогреве; из них на эталонном пути slow)."""
    out, top = [], top10_po_sk(pul)
    for sid, rows in pul.items():
        ok = [r for r in rows if r["status"] == "ok"]
        fast = sum(r["sloy"] == "fast" for r in rows)
        out.append(dict(sid=sid, nazvanie="yadro_sloy_fast", chislo=fast, vsego=len(rows), tekst=f"slow: {len(rows) - fast}"))
        dni = dni_brent_v_okne(meta[sid]["gody"] or ())
        out.append(dict(sid=sid, nazvanie="brent_dni_v_okne", chislo=dni, vsego=dni_brent_v_okne(range(2000, 2100)),
                        tekst="не задевает" if dni == 0 else "ЗАДЕВАЕТ окно"))
        for nm, key in (("brent_tochki", "brent"), ("stop_reentry_tochki", "stop_reentry")):
            pl, tp = [r for r in ok if r[key]], [r for r in top[sid] if r[key]]
            out.append(dict(sid=sid, nazvanie=nm, chislo=len(tp), vsego=len(pl),
                            tekst=f"в топ-10 {len(tp)} из {len(pl)} в пуле; slow: топ-10 {sum(r['sloy'] == 'slow' for r in tp)}, "
                                  f"пул {sum(r['sloy'] == 'slow' for r in pl)}"))
    return out


def vyvod(rows: list, kind: str) -> str:
    """Строка вывода по PRAVILA: «слабое место: <что>, Sk: …» или «не слабое место»; везде пометка потолка."""
    sl = {}
    for r in rows:
        for k in filter(None, str(r.get("slabye", "")).split(";")):
            sl.setdefault(k, []).append(r["sid"])
    txt = "; ".join(f"{k}, Sk: {', '.join(v)}" for k, v in sl.items())
    return f"{kind}: " + (f"слабое место: {txt}" if txt else "не слабое место") + f" ({P.TOP10['pometka']})"


def zapisat(path: Path, rows: list, vyvod_str: str) -> None:
    kol = list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open("w", encoding="utf-8", newline="") as f:
        f.write(f"# статус: {P.STATUS}\n# вывод: {vyvod_str}\n# формат: \"v\": \"1\"\n")
        w = csv.DictWriter(f, kol, lineterminator="\n", restval="")
        w.writeheader()
        w.writerows(rows)


def main(root, out) -> None:
    (out := Path(out)).mkdir(parents=True, exist_ok=True)
    pul, meta = chitat_pul(root)
    t2, t7, og = tablica_2(pul, meta), tablica_7(pul), ogovorki(pul, meta)
    zapisat(out / "t2.csv", t2, vyvod(t2, "(2)"))
    zapisat(out / "t7.csv", t7, vyvod(t7, "(7)"))
    zapisat(out / "ogovorki.csv", og, "(оговорки) числа, порогов нет")
    for nm, g, pole in (("top10.csv", 30, "sharpe"), ("top10_msr_vorota.csv", 30, "msr"), ("top10_bez_vorot.csv", 0, "msr")):
        top = [dict(sid=s, rang=i, sobol_i=r["sobol_i"], prohod=r["prohod"], klyuch=r["klyuch_n"], sharpe=r["sharpe"], msr=r["msr"],
                    msr_plohoy=r["msr_plohoy"], sdelok_okno=P.sdelok_okno(r), pometka_sharpe=r["pometka_sharpe"])
               for s, t in top10_po_sk(pul, g, pole).items() for i, r in enumerate(t, 1)]
        zapisat(out / nm, top, f"(топ-10) ранг по {pole}, ворота: Σ sdelok_y ≥ {g}; вход для Д-4; msr_plohoy: |skew| > {P.SKEW_MAX} или kurt > {P.KURT_MAX}")
    print({s: sum(r["status"] == "ok" and P.sdelok_okno(r) >= 30 for r in rows) for s, rows in pul.items()}, "— прошли ворота ≥ 30 сделок")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
