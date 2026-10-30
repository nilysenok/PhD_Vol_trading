"""Диагностика v6 train4 (Ж-145), лист Д-0: правила, объявленные ДО данных (otbor_statistika.md п. 1, 5).

Источник порогов и топ-10 — docs/resheniya/tolkovaniya.md, [31.10 00:12, класс B], пункты (а), (б), (в). Там, где план архитектора
(diag_v6/plan.md §3) говорит иное, действует толкование; МСП — плана, в толковании порогов для него нет.
Данных ledger модуль не читает. Запись gruppy_osej.csv: `FASTBT_MAIN=$PWD python3 -m fastbt.diag_v6_pravila`.
"""
from __future__ import annotations

import csv
import hashlib
import math
import subprocess
from pathlib import Path

from fastbt import canon_v6

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "docs/tables/diag_v6_train4/gruppy_osej.csv"
STATUS = "разработка train4 2016–2019, потолок (in-sample), после аудита"
GRUPPY = ("signal", "vyhod_risk", "kontekst", "ispolnenie")

# Пороги «слабое место» — толкование (в): «слабое место», если выполнено хоть одно. Единицы — доли (0,5 = 50 %).
PRAVILA = dict(
    sdelok_v_god_mediana_min=10,      # медиана сделок в год у топ-10 < 10
    nuli_dolya_max=0.5,             # доля нулевых точек куба > 50 %
    plato_dolya_min=0.3,              # плато: соседей с mSR > 0 < 30 %
    kontrol_percentil=95,             # топ-10 не лучше к2/плацебо на 95-м перцентиле
    odna_storona_dolya_max=0.8,       # доля одной стороны > 80 % ...
    odna_storona_let_min=2,           # ... в двух и более годах
    bez_luchshih_dnej_n=10,           # Sharpe без 10 лучших дней ...
    bez_luchshih_dnej_dolya_min=0.5,  # ... падает больше чем вдвое (остаток < 50 % полного)
    prochee_dolya_max=0.3,            # причина выхода «конец окна»/«прочее» > 30 %
)
PRAVILA_MSP = dict(delta_lambda_min=0.0, delta_aprim_min=0.0)   # план §3: Δ ≤ 0 — слабое место (в толковании порогов МСП нет)
TOP10 = dict(n=10, kluch="msr_ned", vorota=None, pul="status=ok, rol=sobol", pometka="потолок: ранжирование и метрики на одних данных")

# Фиксированный словарь «слабое место → предложение v7» (гипотеза для журнала, не решение).
PREDLOZHENIYA = {
    "sdelok_malo": "ослабить запрещающие фильтры контекста: сетка фильтров смещена к «выкл.»",
    "nuli_mnogo": "сузить куб: заморозить оси с нулём сделок, поднять P_FILTR до отказа от пустых сочетаний",
    "plato_net": "перейти от точек к плато: шаг осей крупнее, отбор по медиане соседей",
    "kontrol_ne_luchshe": "сигнал не отличим от случайного: пересмотреть семейство сигналов Sk, не выход",
    "odna_storona": "добавить симметрию сторон или раздельные пороги лонг/шорт",
    "zavisit_ot_dnej": "ограничить вклад лучших дней: тейк/трейл, потолок размера позиции",
    "prochee_mnogo": "объяснить причины «прочее»: добавить причины выхода в учёт",
}

_RAZDEL = {"signal": "signal", "filtr": "kontekst", "vhod": "ispolnenie", "vyhod": "vyhod_risk", "kontekst": "kontekst", "ispolnenie": "ispolnenie"}
# Явные исключения по имени оси (план §1): куда уходит ось независимо от раздела куба.
_ISKL = {**{n: "signal" for n in ("rel_signal", "dlit", "asim", "norm", "tf")},
         **{n: "vyhod_risk" for n in ("time_decay", "sigma_ist", "piv_buf", "bezubytok", "stop_v_plyus", "pir", "smena")},
         **{n: "kontekst" for n in ("pauza_porog", "pauza_okno")}}
# Спорные: ось и причина (группа выбрана по месту в оценщике / плану).
SPORNYE = {
    "T": "вход ли это (исполнение) или параметр тренда: в плане не названа; по разделу vhod — исполнение",
    "tf": "раздел куба — контекст, по плану — сигнал (таймфрейм сигнала)",
    "rel_signal": "раздел куба — контекст, по плану — сигнал",
    "dlit": "раздел куба — контекст, по плану — сигнал",
    "asim": "раздел куба — контекст, по плану — сигнал",
    "norm": "раздел куба — контекст, по плану — сигнал",
    "sigma_ist": "раздел куба — исполнение, по плану — выход-риск (источник σ для уровней)",
    "time_decay": "раздел куба — исполнение, по плану — выход-риск",
    "piv_buf": "раздел куба — исполнение, по плану — выход-риск",
    "bezubytok": "в S3/S4 лежит в разделе исполнения; везде — выход-риск",
    "stop_v_plyus": "в S3/S4 лежит в разделе исполнения; везде — выход-риск",
    "pir": "в S3/S4 лежит в разделе исполнения; везде — выход-риск",
    "smena": "в S3/S4 лежит в разделе исполнения; везде — выход-риск",
    "pauza_porog": "раздел куба — исполнение, по плану — контекст (пауза после серии)",
    "pauza_okno": "раздел куба — исполнение, по плану — контекст (пауза после серии)",
    "vhod_tip": "тип входа (стоп-заявка/подтверждение): исполнение, но влияет и на сигнал",
    "atr_filtr": "раздел signal в S5, по смыслу фильтр; оставлена в сигнале по разделу",
    "filter_set": "раздел signal в S5/S6, по смыслу фильтры; оставлена в сигнале по разделу",
    "t_vyhod": "время выхода: исполнение, но задаёт и риск удержания",
}


def gruppa_osi(os: str, razdel: str) -> str:
    """Группа владельца для оси: явное исключение по имени, иначе по разделу куба."""
    return _ISKL.get(os) or _RAZDEL[razdel]


def _razdely(kub: dict) -> dict:
    """{ось: раздел куба} по `canon_v6.osi(kub, gotovy=None)`: v5 — секция, новые — kontekst/ispolnenie."""
    from fastbt import progon_morris_v6 as pm
    return {r["os"]: r["gruppa"] for r in pm.spisok_osej(kub, canon_v6.osi(kub, gotovy=None))}


def stroki_osej() -> list:
    """[(sid, ось, раздел, группа, спорная, причина)] по всем осям 6 Sk."""
    out = []
    for sid in canon_v6.SIDS:
        for os, razdel in _razdely(canon_v6.load_kub(sid)).items():
            out.append((sid, os, razdel, gruppa_osi(os, razdel), os in SPORNYE, SPORNYE.get(os, "")))
    return out


def proverit_pokrytie(stroki: list) -> None:
    """Каждая ось каждого Sk (`canon_v6.osi(kub, gotovy=None)`) — ровно одна строка и группа из GRUPPY; иначе ValueError."""
    chisla = {}
    for s, o, _r, g, *_ in stroki:
        if g not in GRUPPY:
            raise ValueError(f"{s}.{o}: неизвестная группа {g!r}")
        chisla[(s, o)] = chisla.get((s, o), 0) + 1
    for sid in canon_v6.SIDS:
        for o in _razdely(canon_v6.load_kub(sid)):
            if chisla.get((sid, o), 0) != 1:
                raise ValueError(f"{sid}.{o}: строк {chisla.get((sid, o), 0)}, нужна ровно одна")
    lishnie = set(chisla) - {(s, o) for s in canon_v6.SIDS for o in _razdely(canon_v6.load_kub(s))}
    if lishnie:
        raise ValueError(f"лишние оси: {sorted(lishnie)}")


def sha_vhodov() -> str:
    h = hashlib.sha256()
    for p in sorted(canon_v6.KUB6.glob("kub_S*.yaml")) + [Path(canon_v6.__file__)]:
        h.update(p.read_bytes())
    return h.hexdigest()


def top10(rows: list) -> list:
    """Диагностический топ-10 (не отбор): status=ok, rol=sobol, конечный `msr_ned`; 10 лучших по msr_ned, при равенстве — меньший klyuch_n."""
    k = TOP10["kluch"]
    godnye = [r for r in rows if r.get("status") == "ok" and r.get("rol") == "sobol" and r.get(k) is not None and math.isfinite(r[k])]
    return sorted(godnye, key=lambda r: (-r[k], r["klyuch_n"]))[:TOP10["n"]]


def zapisat_csv(path: Path = CSV) -> None:
    kod = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        f.write(f"# статус: {STATUS}\n# kod: база {kod}, diag_v6_pravila.py sha256 {sha}\n# vhody: kub_S*.yaml + canon_v6.py sha256 {sha_vhodov()}\n")
        f.write('# формат: "v": "1"\n')
        w = csv.writer(f, lineterminator="\n")
        w.writerow(("sid", "os", "razdel_kuba", "gruppa", "spornaya", "prichina"))
        w.writerows((s, o, r, g, "da" if sp else "net", pr) for s, o, r, g, sp, pr in stroki_osej())


if __name__ == "__main__":
    zapisat_csv()
    stroki = stroki_osej()
    proverit_pokrytie(stroki)
    print(f"OK: {len(stroki)} строк (sid, ось) -> {CSV.relative_to(ROOT)}")
