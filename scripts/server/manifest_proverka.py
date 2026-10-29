#!/usr/bin/env python3
"""Данные переноса строго по манифесту (Ж-131 п. 2, Ж-133 п. 1): путь, sha256, байты, max_data, chitatel[, istochnik, sha256_istochnika].

Отказ всего переноса (выход 1, до копирования и до чтения файлов), если в манифесте: путь с test_locked или test1_2022_2024,
абсолютный путь или «..», max_data > 2021-12-31 или не дата, дубль, плохой sha256/байты. Затем по каждому файлу: размер и
sha256 в --koren совпадают с манифестом. --istochnik — сначала скопировать в --koren (источник тоже сверяется до копии).
--strogo — в --koren нет файлов вне манифеста. --spisok — записать список путей (для rsync --files-from).
    python3 manifest_proverka.py --manifest M --koren perenos/data [--istochnik <данные Мака>] [--strogo] [--spisok F]
Файлы читаются только для sha256 (содержимое не печатается)."""
import argparse
import hashlib
import os
import re
import shutil
import sys
from pathlib import Path

SHAPKA = ["put", "sha256", "bajty", "max_data", "chitatel"]
GRANICA = "2021-12-31"
ZAPRET = ("test_locked", "test1_2022_2024")


class Otkaz(Exception):
    pass


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _put_ok(put: str, n: int) -> None:
    if any(z in put for z in ZAPRET):
        raise Otkaz(f"строка {n}: путь {put} — запрещён ({'/'.join(ZAPRET)}); перенос отменён целиком")
    if put.startswith("/") or ".." in put.split("/") or put != os.path.normpath(put) or "\\" in put:
        raise Otkaz(f"строка {n}: путь {put!r} не относительный/не нормальный")


def prochitat(manifest: Path) -> list:
    """-> [(kuda, sha, bajty, max_data, otkuda)]; любое нарушение — Otkaz (весь перенос).
    Шапка 5 полей (put …) или 7 (… istochnik, sha256_istochnika): файл берётся на Маке по `put` (усечённая копия
    data/server_2021/…), а на сервере лежит по `istochnik` — пути, по которому его читает код."""
    stroki = manifest.read_text(encoding="utf-8").splitlines()
    shapka = stroki[0].split("\t") if stroki else []
    if shapka not in (SHAPKA, SHAPKA + ["istochnik", "sha256_istochnika"]):
        raise Otkaz(f"{manifest}: шапка ≠ {SHAPKA} [+ istochnik, sha256_istochnika]")
    out, vid = [], set()
    for n, s in enumerate(stroki[1:], 2):
        if not s.strip():
            continue
        f = s.split("\t")
        if len(f) != len(shapka):
            raise Otkaz(f"строка {n}: полей {len(f)}, нужно {len(shapka)}")
        otkuda, h, b, md = f[:4]
        put = f[5] if len(f) == 7 else otkuda
        _put_ok(otkuda, n)
        _put_ok(put, n)
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", md) or md > GRANICA:
            raise Otkaz(f"строка {n}: {put}: max_data {md!r} > {GRANICA} или не дата; перенос отменён целиком")
        if not re.fullmatch(r"[0-9a-f]{64}", h) or not b.isdigit():
            raise Otkaz(f"строка {n}: {put}: sha256/байты не в формате")
        if put in vid:
            raise Otkaz(f"строка {n}: дубль {put}")
        vid.add(put)
        out.append((put, h, int(b), md, otkuda))
    if not out:
        raise Otkaz(f"{manifest}: пустой манифест")
    return out


def sverit(p: Path, put: str, h: str, b: int, gde: str) -> None:
    if not p.is_file() or p.is_symlink():
        raise Otkaz(f"{gde}: нет файла {put}")
    if p.stat().st_size != b:
        raise Otkaz(f"{gde}: {put}: размер {p.stat().st_size} ≠ {b} по манифесту")
    if sha256(p) != h:
        raise Otkaz(f"{gde}: {put}: sha256 не совпадает с манифестом")


def _zamok(p: Path, koren: Path, vid: set) -> bool:
    """Пустой `.<файл>.lock` рядом с файлом манифеста — flock daily_1850.ensure при счёте, не данные."""
    imya = p.name[1:-5] if p.name.startswith(".") and p.name.endswith(".lock") else ""
    return bool(imya) and p.stat().st_size == 0 and str(p.relative_to(koren).with_name(imya)) in vid


def lishnie(koren: Path, vid: set) -> list:
    return sorted(str(p.relative_to(koren)) for p in koren.rglob("*")
                  if (p.is_file() or p.is_symlink()) and str(p.relative_to(koren)) not in vid and not _zamok(p, koren, vid))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--koren", type=Path, required=True)
    ap.add_argument("--istochnik", type=Path, action="append", help="корень данных; можно несколько — берётся первый, где файл есть")
    ap.add_argument("--strogo", action="store_true")
    ap.add_argument("--spisok", type=Path)
    a = ap.parse_args(argv)
    try:
        m = prochitat(a.manifest)
        if a.istochnik:
            for put, h, b, _, otkuda in m:
                src = next((r / otkuda for r in a.istochnik if (r / otkuda).exists()), a.istochnik[0] / otkuda)
                dst = a.koren / put
                sverit(src, otkuda, h, b, "источник")
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(src, dst.with_name(dst.name + ".tmp"))
                dst.with_name(dst.name + ".tmp").replace(dst)
        for put, h, b, _, _ in m:
            sverit(a.koren / put, put, h, b, str(a.koren))
        if a.strogo:
            lish = lishnie(a.koren, {x[0] for x in m})
            if lish:
                raise Otkaz(f"{a.koren}: {len(lish)} файлов вне манифеста, первый {lish[0]}")
    except Otkaz as e:
        print(f"FAIL manifest: {e}")
        return 1
    if a.spisok:
        a.spisok.write_text("".join(f"{x[0]}\n" for x in m), encoding="utf-8")
    print(f"OK manifest: {len(m)} файлов, {sum(x[2] for x in m) / 1e9:.3f} ГБ, max_data {max(x[3] for x in m)}, "
          f"sha256 и размеры совпадают ({a.koren})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
