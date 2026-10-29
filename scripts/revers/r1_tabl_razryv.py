"""Таблица «число предзащиты → источник разрыва → честное значение», A, C, D, 2022–2024 (Ж-129 п. 1).

Модели не пересчитывает: собирает готовые CSV сверок #562, #564, #570. Шарп избыточный, МСП.
Знак: честное = печать + утечка + дивиденды + исполнение + остаток; остаток обязан быть < 1e-3.
  утечка     — шаг P1 лестницы 5а (#570), поставленный первым после воспроизведения;
  дивиденды  — P6 (2015–2021 в отборе) + дивиденды 2022–2024 (шаг 6 / A_zashch − A_net);
  исполнение — все остальные шаги: слой портфеля (ставка, Н2, Н5, комиссия 0,04 %), комиссия отбора,
               фильтр оборота, σ̂/h, бары и код, остановка 2022 (толкование дирижёра, класс B).
Запуск: python3 scripts/revers/r1_tabl_razryv.py
"""
from __future__ import annotations

import csv
import hashlib
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
T = ROOT / "docs" / "tables" / "revers_r1"
OUT = T / "tabl_razryv_predzashchita.csv"
PERIODS = ("2022", "2023", "2024", "2022–2024")
OSTATOK_MAX = 1e-3
LEAK = ("P1",)
DIV_5A = ("P6",)
EXEC_5A = ("P0", "P2", "P3", "P4", "P5", "P7", "P8", "P9", "P10", "P11", "E5")
CD_EXEC = ("1", "2", "3", "4", "5б")
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


def _steps5a() -> dict[tuple[str, str], dict]:
    return {(r["obekt"], r["shag"]): r for r in _read(IN5A)}


def razlozhit(pechat: dict, chestnoe: dict, leak: list, div: list, ispoln: list) -> dict:
    """Ядро: честное − печать = Σ шагов + остаток; |остаток| ≥ 1e-3 — ошибка (пропущен или лишний шаг)."""
    out = {}
    for p in PERIODS:
        u, d, i = _tot(leak)[p], _tot(div)[p], _tot(ispoln)[p]
        ost = chestnoe[p] - pechat[p] - u - d - i
        if abs(ost) >= OSTATOK_MAX:
            raise ValueError(f"остаток {ost:.6f} в {p}: сумма шагов не равна разрыву")
        out[p] = dict(pechat=pechat[p], utechka=u, dividendy=d, ispolnenie=i, ostatok=ost, chestnoe=chestnoe[p])
    return out


def sobrat(obj: str, skip: tuple[str, ...] = ()) -> dict:
    """Шаги объекта из CSV. skip — шаги, которые не включать (для теста мутации «шаг пропущен»)."""
    s5 = _steps5a()
    pick = lambda codes: [_per(s5[(obj, c)], "delta") for c in codes if c not in skip]  # noqa: E731
    honest = _test(INTEST, obj)
    leak, div, ex = pick(LEAK), pick(DIV_5A), pick(EXEC_5A)
    if obj == "A":
        a = {r["shag"]: r for r in _read(INA, ";")}
        pechat = _per(a["печать"], "sharpe", True)
        e0, e5, net = _per(s5[("A", "E0")], "sharpe"), _per(s5[("A", "E5")], "sharpe"), _test(INNET, "A")
        ex += [_lin(e0, pechat, -1.0), _lin(net, e5, -1.0)]
        div.append(_lin(honest, net, -1.0))
    else:
        cd = {(r["obekt"], r["shag"]): r for r in _read(INCD)}
        pechat = _per(cd[(obj, "0")], "sharpe")
        ex += [_per(cd[(obj, c)], "delta") for c in CD_EXEC if c not in skip]
        div.append(_per(cd[(obj, "6")], "delta"))
        sverka = _per(cd[(obj, "6")], "sharpe")
        if any(abs(sverka[p] - honest[p]) > 1e-9 for p in PERIODS):
            raise ValueError("честное значение расходится: сверка CD ≠ a_cd_test1")
    return razlozhit(pechat, honest, leak, div, ex)


def _sha(name: str) -> str:
    return hashlib.sha256((T / name).read_bytes()).hexdigest()


def _commit() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def main() -> None:
    head = [
        "# статус: реверс р.1, 2022–2024, справочно (Ж-129 п. 1); Шарп избыточный МСП; честное = печать + утечка + дивиденды + исполнение + остаток",
        f"# коммит (HEAD при сборке): {_commit()}",
        "# печать: период 2022–2024 пересчитан из печатного ряда MSP_*_inst.csv (< 2025); печать предзащиты давала 2022–2025 (A 1,32 / C 2,51 / D 2,54 — FINAL_RESULTS_CORRECTED_V2.md:10-26) — не сравнивать",
        "# честное: s9 сейчас (A_zashch / TEST_C / D), дивиденды 2022–2024, плата за шорт 0; chestnoe_plata_short5 — справочно, 5 %, в сумму не входит",
        "# утечка 2024: включает шаг P0 (бары, воспроизведение): маске [1] нужен бар ≥ 2025, которого в зоне Z2 нет (sverka_5a.md, оговорки)",
        "# лестница цепная, вклад шагов зависит от порядка; исполнение: слой портфеля, комиссии, фильтр оборота, σ̂/h, бары и код, ставка, Н2/Н5, остановка 2022",
    ] + [f"# вход: docs/tables/revers_r1/{n} sha256={_sha(n)}" for n in INPUTS]
    cols = ["obekt", "period", "pechat", "utechka_okna", "dividendy", "ispolnenie", "ostatok", "chestnoe", "chestnoe_plata_short5"]
    with open(OUT, "w", encoding="utf-8", newline="") as f:
        f.write("\n".join(head) + "\n")
        w = csv.writer(f)
        w.writerow(cols)
        for obj in ("A", "C", "D"):
            res, sh5 = sobrat(obj), _sh5(obj)
            for p in PERIODS:
                r = res[p]
                v = [r["pechat"], r["utechka"], r["dividendy"], r["ispolnenie"], r["ostatok"], r["chestnoe"], sh5[p]]
                w.writerow([obj, p] + [f"{x:.6f}" for x in v])


if __name__ == "__main__":
    main()
