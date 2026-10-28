"""Механическая сводка раунда 1 реверса (без выводов).

Вход: docs/otchety/revers_r1/{a,b,v,g,d,e,zh}.md, docs/tables/revers_r1/*.csv.
Выход: docs/otchety/2026-10-29_revers_r1.md; exit 1 при FAIL.
Шапка csv: строки на «#» в начале файла. Нужны маркер периода, строки
«# коммит: <sha>» и «# вход: <имя> sha256=<hex>». Таблица листа — файл с префиксом «<лист>_».
Самопроверка: python3 scripts/revers/r1_svodka.py --self-test
"""
import argparse
import csv
import hashlib
import re
import sys
import tempfile
from pathlib import Path

LISTY = ["a", "b", "v", "g", "d", "e", "zh"]
NAZV = {"a": "а", "b": "б", "v": "в", "g": "г", "d": "д", "e": "е", "zh": "ж"}
# список предрегистрации: лист -> что за таблица (docs/preregistraciya_revers_r1.md)
PREREG = {
    "a": "Sharpe/mSR/по годам/без 10 худших и лучших/стоп 2σ/размер 1/σ",
    "b": "лонг/шорт, дивиденды, комиссия, вклад ФДР",
    "v": "овернайт против внутридневной части",
    "g": "карта таймфреймов 6 ТФ × 3 семейства",
    "d": "кривые событий S1–S6 против к2 и плацебо",
    "e": "фазы рынка по месяцам и результат по фазам",
    "zh": "разрез по терцилям σ",
}
MARKER = "реверс раунд 1, 2022–2024, справочно, без выбора"
DATE_COLS = {"date", "data", "дата", "день", "day", "den"}
BPS = re.compile(r"bps|б\.\s?п\.", re.I)
YEAR = re.compile(r"^\s*(\d{4})")


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def razbor_csv(p):
    """-> (шапка, строки-данные, число строк, заголовок колонок)."""
    lines = p.read_text(encoding="utf-8").splitlines()
    head = [x for x in lines[: next((i for i, x in enumerate(lines) if not x.startswith("#")), len(lines))]]
    rows = list(csv.reader(x for x in lines[len(head):] if x.strip()))
    cols = rows[0] if rows else []
    return head, rows[1:], len(rows[1:]), cols


def proverka_tablicy(p):
    """-> (n_строк, шапка, список FAIL)."""
    fails = []
    head, rows, n, cols = razbor_csv(p)
    if MARKER not in " ".join(head).lower():
        fails.append(f"{p.name}: нет шапки «{MARKER}»")
    for j, c in enumerate(cols):
        if c.strip().lower() in DATE_COLS:
            for r in rows:
                m = YEAR.match(r[j]) if j < len(r) else None
                if m and int(m.group(1)) >= 2025:
                    fails.append(f"{p.name}: дата ≥ 2025 в колонке {c}: {r[j]}")
                    break
    return n, head, fails


def pervaya_stroka(text):
    for x in text.splitlines():
        if x.strip() and not x.lstrip().startswith("#"):
            return x.strip()
    return ""


def vyyti_hesh(head):
    kom = [m.group(1) for x in head for m in [re.search(r"коммит[^:]*:\s*(\S+)", x)] if m]
    vh = [m.group(1).strip() for x in head for m in [re.search(r"вход[^:]*:\s*(.+)", x)] if m]
    return kom, vh


def sobrat(root, out):
    md_dir, tb_dir = root / "docs/otchety/revers_r1", root / "docs/tables/revers_r1"
    csvs = sorted(tb_dir.glob("*.csv")) if tb_dir.is_dir() else []
    fails, info = [], {}
    for p in csvs:
        n, head, f = proverka_tablicy(p)
        fails += f
        info[p.name] = (p, n, head)
    texts = {k: (md_dir / f"{k}.md").read_text(encoding="utf-8") if (md_dir / f"{k}.md").exists() else None for k in LISTY}
    tabl = {k: [x for x in info if x.startswith(k + "_")] for k in LISTY}
    ozhid, gotovo = [], 0
    for k in LISTY:
        t = texts[k]
        if t is None:
            ozhid.append(f"{NAZV[k]}: нет абзаца")
        elif re.search(r"ABANDON|ждёт", t):
            ozhid.append(f"{NAZV[k]}: ABANDON/ждёт в абзаце")
        elif not tabl[k]:
            ozhid.append(f"{NAZV[k]}: нет таблиц")
        else:
            gotovo += 1
        if t and BPS.search(t):
            fails.append(f"{k}.md: «bps»/«б.п.» — владельцу проценты")
    L = [f"Реверс раунд 1, 2022–2024, справочно, без выбора; листов готово {gotovo} из 7; ABANDON/ждёт: "
         + ("; ".join(ozhid) or "нет"), ""]
    L += ["## Проверка сдвига входов", "", "| Лист | Первая строка абзаца |", "|---|---|"]
    L += [f"| {NAZV[k]} | {(pervaya_stroka(texts[k]) if texts[k] else '—').replace('|', '/')} |" for k in LISTY]
    for k in LISTY:
        L += ["", f"## Лист {NAZV[k]}", "", texts[k].strip() if texts[k] else "(абзаца нет)", "", "Таблицы:"]
        L += [f"- `{x}` sha256={sha(info[x][0])} строк={info[x][1]}" for x in tabl[k]] or ["- нет"]
    L += ["", "## Соответствие предрегистрации", ""]
    L += [f"- {NAZV[k]}) {PREREG[k]}: {'есть' if tabl[k] else 'нет'}" for k in LISTY]
    vne = [x for x in info if not any(x in v for v in tabl.values())]
    L += ["", "Вне предрегистрации (не удалены):"] + ([f"- `{x}`" for x in vne] or ["- нет"])
    L += ["", "## Хэши", "", "| Лист | Таблица | Коммит кода | Входы |", "|---|---|---|---|"]
    for k in LISTY:
        for x in tabl[k]:
            kom, vh = vyyti_hesh(info[x][2])
            L.append(f"| {NAZV[k]} | {x} | {', '.join(kom) or '—'} | {'; '.join(vh) or '—'} |")
    L += ["", "## Итог", ""] + [f"- FAIL: {f}" for f in fails]
    L.append("ИТОГ: " + ("FAIL" if fails else "OK"))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    return fails, L[0]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(Path(__file__).resolve().parents[2]))
    ap.add_argument("--out", default=None)
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args(argv)
    if a.self_test:
        return selftest()
    root = Path(a.root)
    out = Path(a.out) if a.out else root / "docs/otchety/2026-10-29_revers_r1.md"
    fails, first = sobrat(root, out)
    print(first)
    for f in fails:
        print("FAIL:", f)
    print("ИТОГ:", "FAIL" if fails else "OK")
    return 1 if fails else 0


def selftest():
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        (root / "docs/otchety/revers_r1").mkdir(parents=True)
        (root / "docs/tables/revers_r1").mkdir(parents=True)
        for k in LISTY:
            (root / f"docs/otchety/revers_r1/{k}.md").write_text("сдвиг ок\nтекст\n", encoding="utf-8")
            (root / f"docs/tables/revers_r1/{k}_t.csv").write_text(
                f"# {MARKER}\n# коммит: abc\n# вход: x sha256=00\ndate,v\n2022-01-03,1\n", encoding="utf-8")
        r = main(["--root", d])
    assert r == 0, "самопроверка: ожидался OK"
    print("самопроверка OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
