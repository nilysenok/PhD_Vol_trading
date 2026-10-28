"""Реестр прогона Л-12 (docs/otchety/2026-10-28_progon_etap1.md §6): все строки ledger v2, только дополнение.

Файл — gzip с дописываемыми членами (один член на вызов, шапка один раз). Известная строка
(стратегия, istochnik, klyuch_sha) не дублируется. Число записанных + уже известных обязано равняться
числу строк ledger, иначе ValueError. Метрики 2020+ сюда не попадают: их отказывает читатель ledger_v2 (§1.6)."""
import csv
import gzip
import hashlib
import io
import json
import math
from pathlib import Path
from typing import Optional

from fastbt import ledger_v2 as L
from fastbt import reestr_otbora_v1 as V1
from fastbt.progon_osnovnaya import osnovnaya_stroki

VYHOD = "docs/otbor/reestr_otbora_progon_v2.csv.gz"
GODY = (2016, 2017, 2018, 2019)
KV = [f"{g}Q{q}" for g in GODY for q in range(1, 5)]
ZAEM = ("5", "10")
OTBOR = "otbor"
OTBOR_POLYA = ("region_id", "chlen_regiona", "kandidat", "wf_pravilo")
PRICHINA_OTBOR = "разметка Л-7/Л-8"


def kolonki() -> list:
    """Шапка реестра прогона: поля §1.2 плоско, 16 кварталов колонками, колонки отбора."""
    return (["strategiya", "istochnik", "klyuch_sha", "klyuch", "kod", "kod_slit", "sloy", "status", "prichina"]
            + [f"sharpe_{y}" for y in GODY] + [f"sdelok_{y}" for y in GODY]
            + [f"kv_{k}" for k in KV]
            + ["mediana_kv", "dolya_pol_kv", "zaem5_mediana_kv", "zaem10_mediana_kv"]
            + [f"msr_{y}" for y in GODY]
            + ["vyrozhdena", "kray", "region_id", "chlen_regiona", "kandidat", "wf_pravilo", "komissiya", "fdr", "osnovnaya"])


def _num(v) -> str:
    return "" if v is None or (isinstance(v, float) and math.isnan(v)) else repr(float(v))


def _bit(v) -> str:
    return "" if v is None else str(int(bool(v)))


def _godovoe(d, y):
    """Значение года: ключ — строка (контракт §1.2); int тоже допускается."""
    d = d or {}
    return d.get(str(y), d.get(y))


def _kvartaly(by_q) -> dict:
    """Пусто — допустимо (ne_schitana); иначе ровно 16 кварталов, 15 — отказ."""
    if not by_q:
        return {}
    if set(by_q) != set(KV):
        raise ValueError(f"by_quarter: ключей {len(by_q)}, ожидалось 16 кварталов 2016Q1…2019Q4")
    return by_q


def _stroka(r: dict) -> dict:
    """Строка реестра из строки ledger; nan и отсутствие — пустая ячейка."""
    out = {c: "" for c in kolonki()}
    kv, zaem = _kvartaly(r.get("by_quarter")), r.get("by_quarter_zaem") or {}
    out.update(strategiya=r["strategiya"], istochnik=r["istochnik"], klyuch_sha=r["klyuch_sha"],
               klyuch=json.dumps(r["klyuch"], sort_keys=True, ensure_ascii=False, separators=(",", ":")),
               kod=r["kod"], kod_slit=_bit(r.get("kod_slit")), sloy=r.get("sloy", ""), status=r["status"],
               prichina=r.get("prichina") or "", komissiya=_num(r["komissiya"]), fdr=_num(r.get("fdr")),
               mediana_kv=_num(r.get("mediana_kv")), dolya_pol_kv=_num(r.get("dolya_pol_kv")),
               osnovnaya=_bit(osnovnaya_stroki(r)))  # Ж-084 п. 2: пишутся все строки, признак — колонкой
    out.update({f"sharpe_{y}": _num(_godovoe(r.get("by_year"), y)) for y in GODY})
    out.update({f"sdelok_{y}": _num(_godovoe(r.get("sdelok_y"), y)) for y in GODY})
    out.update({f"msr_{y}": _num(_godovoe(r.get("msr_by_year"), y)) for y in GODY})
    out.update({f"kv_{k}": _num(v) for k, v in kv.items()})
    for z in ZAEM:
        q = _kvartaly(zaem.get(z))
        out[f"zaem{z}_mediana_kv"] = _num(L.tselevaya(q)[0]) if q else ""
    fl = r.get("flagi") or {}
    out.update(vyrozhdena=_bit(fl.get("vyrozhdena")), kray=";".join(fl.get("kray") or []))
    return out


def _chitat_gz(path) -> list:
    """Все члены gzip целиком; шапка — ровно kolonki(), иначе отказ."""
    if not Path(path).exists():
        return []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        if rd.fieldnames != kolonki():
            raise ValueError(f"{path}: шапка не совпадает с kolonki()")
        return list(rd)


def _dopisat_gz(path, rows: list) -> int:
    """Новый член gzip в конец файла; при существующем файле шапка не пишется, пустой член не пишется."""
    path = Path(path)
    if not rows and path.exists():
        return 0
    buf = io.StringIO()
    w = csv.DictWriter(buf, kolonki())
    if not path.exists():
        w.writeheader()
    w.writerows(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "ab") as f:
        f.write(gzip.compress(buf.getvalue().encode("utf-8")))
    return len(rows)


def _otbor_strok(vse: list, sid: str, otbor: dict, otb: set) -> list:
    """Строки разметки Л-7/Л-8: полная строка ledger, status = otbor; известная пара не меняется."""
    by = {r["klyuch_sha"]: r for r in vse}
    out = []
    for sha, meta in otbor.items():
        if set(meta) - set(OTBOR_POLYA):
            raise ValueError(f"разметка: лишние поля {sorted(set(meta) - set(OTBOR_POLYA))}")
        if sha not in by:
            raise ValueError(f"разметка для неизвестного klyuch_sha {sha[:12]}")
        if (sid, sha) in otb:
            continue
        r = _stroka(by[sha])
        r.update(status=OTBOR, prichina=PRICHINA_OTBOR, **{k: meta.get(k, "") for k in OTBOR_POLYA})
        out.append(r)
    return out


def dopisat_iz_ledger(root, sid: str, otbor: Optional[dict] = None) -> int:
    """Дописывает все строки ledger стратегии sid (ok, ne_schitana, oshibka; все istochnik).
    Известные (sid, istochnik, klyuch_sha) не дублируются. Возвращает число записанных ledger-строк.
    otbor: {klyuch_sha: {region_id, chlen_regiona, kandidat, wf_pravilo}} — новые строки status = otbor."""
    vse = L.Chitatel(Path(root), sid).rows(None)
    path = Path(VYHOD)
    led, otb = set(), set()
    for r in _chitat_gz(path):
        if r["status"] == OTBOR:
            otb.add((r["strategiya"], r["klyuch_sha"]))
        else:
            led.add((r["strategiya"], r["istochnik"], r["klyuch_sha"]))
    key = [(sid, r["istochnik"], r["klyuch_sha"]) for r in vse]
    if any(r["status"] == OTBOR for r in vse):
        raise ValueError("статус otbor в ledger не бывает")
    izvestnye = sum(k in led for k in key)
    out = [_stroka(r) for r, k in zip(vse, key) if k not in led]
    if len(out) + izvestnye != len(vse):
        raise ValueError(f"{sid}: записано {len(out)} + известно {izvestnye} != строк ledger {len(vse)}")
    n = _dopisat_gz(path, out)
    if otbor:
        _dopisat_gz(path, _otbor_strok(vse, sid, otbor, otb))
    return n


def ukazatel(sid: str) -> dict:
    """Строка-указатель в reestr_otbora_A_v1.csv (через V1.dopisat): istochnik = progon_v2, prichina — sha256 и число строк gz."""
    p = Path(VYHOD)
    sha, n = hashlib.sha256(p.read_bytes()).hexdigest(), len(_chitat_gz(p))
    row = {c: "" for c in V1.kolonki("train4")}
    row.update(strategiya=sid, klyuch=f"sha256={sha}", rezhim="progon_v2", istochnik="progon_v2",
               status="ukazatel", prichina=f"sha256 {sha}; строк {n}")
    V1.dopisat([row], "train4", 1)
    return row
