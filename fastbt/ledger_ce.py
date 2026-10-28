"""Формат `ce-ledger` v 1.0 для C/D/E (лист CE-4; Ж-088 п. 4). Данных не читает. Читатель отказывает, а не пропускает.

Основная строка — функция только параметров: `ispolnenie` (вход/выход/версия) не содержит `close`. Строка без `ispolnenie` —
ValueError, не «основная». Ключ строки включает `ispolnenie`, комиссию и ставку шорта (Д5).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

FORMAT, V = "ce-ledger", "1.0"
OKNO = ("2018-01-01", "2019-12-31")
POLYA = ("vhod", "vyhod", "v")


def osnovnaya_ce(ispolnenie) -> bool:
    if not isinstance(ispolnenie, dict) or not all(k in ispolnenie for k in POLYA):
        raise ValueError(f"ispolnenie без полей {POLYA}: основная строка не определена")
    return not any("close" in str(ispolnenie[k]) for k in ("vhod", "vyhod"))


def klyuch_sha(kl: dict, kod: str, vhody: str) -> str:
    osnovnaya_ce(kl.get("ispolnenie"))
    kan = json.dumps(kl, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(f"{kan}|{kod}|{vhody}|{V}".encode("utf-8")).hexdigest()


class Pisatel_ce:                                                    # noqa: N801
    """Дописывает строки в `rows.jsonl`; `osnovnaya` и `klyuch_sha` ставит сам из параметров."""

    def __init__(self, papka: Path, kod: str, vhody: str):
        self.p, self.kod, self.vhody = Path(papka), kod, vhody
        self.p.mkdir(parents=True, exist_ok=True)

    def zapisat(self, row: dict) -> dict:
        r = dict(row, v=V, format=FORMAT, kod=self.kod, vhody=self.vhody)
        r["osnovnaya"] = osnovnaya_ce(r["klyuch"].get("ispolnenie"))
        r["klyuch_sha"] = klyuch_sha(r["klyuch"], self.kod, self.vhody)
        Chitatel_ce().razobrat(r)
        with open(self.p / "rows.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n")
        return r


class Chitatel_ce:                                                   # noqa: N801
    def razobrat(self, row: dict) -> dict:
        if row.get("v") != V:
            raise ValueError(f"версия строки {row.get('v')!r} ≠ {V}")
        isp = row.get("klyuch", {}).get("ispolnenie")
        osn = osnovnaya_ce(isp)                                      # нет ispolnenie → ValueError
        if isp["v"] != V:
            raise ValueError(f"версия ispolnenie {isp['v']!r} ≠ {V}")
        if row.get("osnovnaya") and not osn:
            raise ValueError("основная строка с close в ispolnenie (Ж-088 п. 4)")
        if row.get("period") == "train_ce":
            bad = [y for k in ("by_year", "msr_by_year") for y in row.get(k, {}) if not 2018 <= int(y) <= 2019]
            if bad:
                raise ValueError(f"годы вне окна train_ce: {sorted(set(bad))}")
        return row

    def chitat(self, papka: Path) -> list:
        rows = [self.razobrat(json.loads(s)) for s in (Path(papka) / "rows.jsonl").read_text(encoding="utf-8").splitlines()]
        if len({(r.get("kod"), r.get("vhody")) for r in rows}) > 1:
            raise ValueError("разные kod/vhody в одной матрице")
        return rows
