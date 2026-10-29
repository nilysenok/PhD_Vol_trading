"""Таблица «число предзащиты → источник разрыва → честное значение», A, C, D, 2022–2024 (Ж-129 п. 1).

Модели не пересчитывает: собирает готовые CSV сверок #562, #564, #570. Шарп избыточный, МСП.
Знак: честное = печать + Σ колонок + остаток; остаток обязан быть < 1e-3. Колонки (лестница цепная):
  utechka_okna    — P1 (#570): маска окна [1] → [0];
  div_2224        — только дивиденды 2022–2024: шаг 6 (C, D) / A_zashch − A без них (A);
  div_progrev     — P6: дивиденды 2015–2021 в барах и PnL отбора (меняют выбор, не начисление);
  bary_dannye     — P0 и конец лестницы: смена ряда v4 → 18:50 и код после 26.10 (P0, E5, шаг 5б / A: A_net − E5);
  ispolnenie      — слой портфеля (ставка, Н2/Н5, остановка 2022, комиссия ряда 0,04 %), комиссия отбора,
                    фильтр оборота, σ̂/h, окна, корпсобытия, гистерезис (P2–P5, P7–P11, шаги 1–4 / A: E0 − печать).
Источники каждой ячейки — tabl_razryv_istochniki.csv. Запуск: python3 scripts/revers/r1_tabl_razryv.py
"""
from __future__ import annotations

import csv
import hashlib
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
T = ROOT / "docs" / "tables" / "revers_r1"
OUT = T / "tabl_razryv_predzashchita.csv"
OUT_SRC = T / "tabl_razryv_istochniki.csv"
PERIODS = ("2022", "2023", "2024", "2022–2024")
OSTATOK_MAX = 1e-3
COLS = ("utechka_okna", "div_2224", "div_progrev", "bary_dannye", "ispolnenie")
A5 = {
    "utechka_okna": ("P1",), "div_progrev": ("P6",), "bary_dannye": ("P0", "E5"),
    "ispolnenie": ("P2", "P3", "P4", "P5", "P7", "P8", "P9", "P10", "P11"),
}
CD_STEPS = {"ispolnenie": ("1", "2", "3", "4"), "bary_dannye": ("5б",), "div_2224": ("6",)}
IN5A, INA, INCD = "sverka_5a.csv", "sverka_predzashchita_A.csv", "sverka_predzashchita_CD.csv"
INTEST, INNET, INSH = "a_cd_test1.csv", "_net/a_cd_test1.csv", "a_plata_short5_test1.csv"
INPUTS = (IN5A, INA, INCD, INTEST, INNET, INSH)
PORTFEL = {"A": "A (s9", "C": "TEST_C", "D": "D:"}
SH5 = {"A": "A_zashch", "C": "TEST_C", "D": "D"}


def _read(name: str, delim: str = ",") -> list[dict]:
    with open(T / name, encoding="utf-8") as f:
        return list(csv.DictReader((x for x in f if not x.startswith("#")), delimiter=delim))


def _num(s: str, comma: bool = False) -> float:
    return float(s.replace(",", ".") if comma else s)


def _per(row: dict, base: str, comma: bool = False) -> dict[str, float]:
    """{период: значение} из колонок base, base_2022, base_2023, base_2024."""
    out = {"2022–2024": _num(row[base], comma)}
    for y in ("2022", "2023", "2024"):
        out[y] = _num(row[f"{base}_{y}"], comma)
    return out


def _lin(a: dict, b: dict, sign: float = 1.0) -> dict:
    return {p: a[p] + sign * b[p] for p in PERIODS}


def _tot(parts: list[dict]) -> dict:
    out = {p: 0.0 for p in PERIODS}
    for d in parts:
        out = _lin(out, d)
    return out


def _test(name: str, obj: str) -> dict[str, float]:
    """Шарп из a_cd_test1-подобного файла: «полные» → 2022–2024 и по годам."""
    res = {}
    for r in _read(name):
        if r["portfel"].startswith(PORTFEL[obj]):
            if r["stroka"] == "(а) полные":
                res["2022–2024"] = _num(r["sharpe"])
            elif r["stroka"].startswith("по году "):
                res[r["stroka"][-4:]] = _num(r["sharpe"])
    return res


def _sh5(obj: str) -> dict[str, float]:
    res = {}
    for r in _read(INSH):
        if r["obekt"].startswith(SH5[obj]) and r["plata_zaem"] == "0.05":
            res["2022–2024" if r["period"] == "полные" else r["period"]] = _num(r["sharpe"])
    return res


def razlozhit(pechat: dict, chestnoe: dict, cols: dict[str, list[dict]]) -> dict:
    """Ядро: честное − печать = Σ колонок + остаток; |остаток| ≥ 1e-3 — ошибка (пропущен или лишний шаг)."""
    out = {}
    for p in PERIODS:
        v = {c: _tot(cols.get(c, []))[p] for c in COLS}
        ost = chestnoe[p] - pechat[p] - sum(v.values())
        if abs(ost) >= OSTATOK_MAX:
            raise ValueError(f"остаток {ost:.6f} в {p}: сумма шагов не равна разрыву")
        out[p] = dict(pechat=pechat[p], ostatok=ost, chestnoe=chestnoe[p], **v)
    return out


def _chain(obj: str, skip: tuple[str, ...]) -> tuple[dict, dict, dict[str, list[tuple]]]:
    """(печать, честное, {колонка: [(файл, ключ строки, {период: Δ})]}) — из CSV, без пересчёта."""
    s5 = {(r["obekt"], r["shag"]): r for r in _read(IN5A)}
    honest = _test(INTEST, obj)
    cols: dict[str, list[tuple]] = {c: [] for c in COLS}
    for col, codes in A5.items():
        cols[col] += [(IN5A, f"{obj}/{c}", _per(s5[(obj, c)], "delta")) for c in codes if c not in skip]
    if obj == "A":
        a = {r["shag"]: r for r in _read(INA, ";")}
        pechat = _per(a["печать"], "sharpe", True)
        e0, e5, net = _per(s5[("A", "E0")], "sharpe"), _per(s5[("A", "E5")], "sharpe"), _test(INNET, "A")
        cols["ispolnenie"].append((f"{IN5A} (E0 − печать из {INA})", "A/E0 − A/печать", _lin(e0, pechat, -1.0)))
        cols["bary_dannye"].append((f"{INNET} − {IN5A}", "A (полные/по году) − A/E5", _lin(net, e5, -1.0)))
        cols["div_2224"].append((f"{INTEST} − {INNET}", "A (полные/по году)", _lin(honest, net, -1.0)))
        return pechat, honest, cols
    cd = {(r["obekt"], r["shag"]): r for r in _read(INCD)}
    for col, codes in CD_STEPS.items():
        cols[col] += [(INCD, f"{obj}/{c}", _per(cd[(obj, c)], "delta")) for c in codes if c not in skip]
    sverka = _per(cd[(obj, "6")], "sharpe")
    if any(abs(sverka[p] - honest[p]) > 1e-9 for p in PERIODS):
        raise ValueError("честное значение расходится: сверка CD ≠ a_cd_test1")
    return _per(cd[(obj, "0")], "sharpe"), honest, cols


def sobrat(obj: str, skip: tuple[str, ...] = ()) -> dict:
    """Строки таблицы объекта. skip — шаги, которые не включать (тест мутации «шаг пропущен»)."""
    pechat, honest, cols = _chain(obj, skip)
    return razlozhit(pechat, honest, {c: [d for _, _, d in v] for c, v in cols.items()})


def istochniki(obj: str) -> list[list]:
    """Строки tabl_razryv_istochniki.csv: из чего сложена каждая ячейка."""
    pechat, _, cols = _chain(obj, ())
    rows = [[obj, p, "pechat", INCD if obj != "A" else INA, f"{obj}/0" if obj != "A" else "A/печать", f"{pechat[p]:.6f}"] for p in PERIODS]
    for col, items in cols.items():
        for f, key, d in items:
            rows += [[obj, p, col, f, key, f"{d[p]:.6f}"] for p in PERIODS]
    rows += [[obj, p, "chestnoe", INTEST, f"{obj} (полные/по году)", f"{_test(INTEST, obj)[p]:.6f}"] for p in PERIODS]
    return rows


def _sha(name: str) -> str:
    return hashlib.sha256((T / name).read_bytes()).hexdigest()


def _commit() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def _head() -> list[str]:
    return [
        "# статус: реверс р.1, 2022–2024, справочно (Ж-129 п. 1); Шарп избыточный МСП; честное = печать + Σ колонок + остаток",
        f"# коммит (HEAD при сборке): {_commit()}",
        "# печать: период 2022–2024 пересчитан из печатного ряда MSP_*_inst.csv (< 2025); печать предзащиты давала 2022–2025 (A 1,32 / C 2,51 / D 2,54 — FINAL_RESULTS_CORRECTED_V2.md:10-26) — не сравнивать",
        "# честное: s9 сейчас (A_zashch / TEST_C / D), дивиденды 2022–2024, плата за шорт 0; chestnoe_plata_short5 — справочно, 5 %, в сумму не входит",
        "# утечка 2024: включает P0 (бары, воспроизведение): маске [1] нужен бар ≥ 2025, которого в зоне Z2 нет (sverka_5a.md, оговорки)",
        "# div_progrev — годовой Шарп при смене отбора (дивиденды 2015–2021 меняют, какие точки выбраны), не начисление в год Y; div_2224 — начисление",
        "# лестница цепная, вклад шагов зависит от порядка: вклады — не отдельные эффекты на фиксированной базе (#570 iDIV на базе P11 даёт иное число)",
    ] + [f"# вход: docs/tables/revers_r1/{n} sha256={_sha(n)}" for n in INPUTS]


def main() -> None:
    cols = ["obekt", "period", "pechat"] + list(COLS) + ["ostatok", "chestnoe", "chestnoe_plata_short5"]
    with open(OUT, "w", encoding="utf-8", newline="") as f:
        f.write("\n".join(_head()) + "\n")
        w = csv.writer(f)
        w.writerow(cols)
        for obj in ("A", "C", "D"):
            res, sh5 = sobrat(obj), _sh5(obj)
            for p in PERIODS:
                r = res[p]
                v = [r["pechat"]] + [r[c] for c in COLS] + [r["ostatok"], r["chestnoe"], sh5[p]]
                w.writerow([obj, p] + [f"{x:.6f}" for x in v])
    with open(OUT_SRC, "w", encoding="utf-8", newline="") as f:
        f.write("# источники ячеек tabl_razryv_predzashchita.csv: колонка, файл (docs/tables/revers_r1), ключ строки (obekt/shag), Δ или значение\n")
        w = csv.writer(f)
        w.writerow(["obekt", "period", "kolonka", "fajl", "klyuch_stroki", "znachenie"])
        for obj in ("A", "C", "D"):
            w.writerows(istochniki(obj))


if __name__ == "__main__":
    main()
