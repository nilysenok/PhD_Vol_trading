"""Манифест переноса данных на сервер: только файлы, где max дата ВНУТРИ файла <= 2021-12-31 (Ж-131 п. 2).

Список: строки `шаблон<TAB>читатель[<TAB>без_дат: обоснование]`, шаблон — от корня `--koren`, `**` допустим.
Исключается ДО чтения: сегмент `test_locked`, каталог `data/regen/test1_2022_2024/`.
Имя с годом 2022-2029 не читается (хук) -> otvergnuto. Читается одна колонка дат; печатается только дата.
Файл без колонки дат принимается лишь с обоснованием в 3-м поле, иначе отвергается.
Выход: manifest_dannyh_2021.tsv (принято) и *_otvergnuto.tsv (причина).
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
import json_daty as jd  # noqa: E402

GRANICA = pd.Timestamp("2021-12-31")
IMYA_DAT = {"begin", "end", "time", "datetime", "timestamp", "ts", "dt", "day", "period", "index", "__index_level_0__"}
GOD_V_IMENI = re.compile(r"(?<!\d)(20\d\d)(?!\d)")
OTMETKA = Path("docs/resheniya/wf_zamok_snyat.json")     # отметку снятого замка ставит владелец (Ж-136), не агент
DO_WF = "2026-02-28"                                      # конец данных единого walk-forward


def zapreshen(p: str) -> str | None:
    """Причина исключения по пути (без обращения к файлу) или None."""
    for q in (p, os.path.realpath(p)):
        s = Path(q).parts
        if "test_locked" in s:
            return "test_locked"
        if "test1_2022_2024" in s and "regen" in s:
            return "data/regen/test1_2022_2024/"
    return None


def kolonka_dat(imya: str) -> bool:
    n = str(imya).lower()
    if n == "retrieved_at" or n.endswith("_status"):       # дата скачивания файла / текстовый статус даты, не дата данных
        return False
    return "date" in n or "дата" in n or n.startswith("data_") or n == "known_from" or n in IMYA_DAT


def max_data(ser: pd.Series) -> pd.Timestamp | None:
    """Max даты по одной колонке; дата не разобрана — ValueError (файл отвергается)."""
    ser = ser.dropna()
    if ser.empty:
        return None
    if pd.api.types.is_datetime64_any_dtype(ser):
        t = ser
    elif pd.api.types.is_integer_dtype(ser) and ser.between(19000101, 21001231).all():
        t = pd.to_datetime(ser.astype(str), format="%Y%m%d")
    elif ser.dtype == object or pd.api.types.is_string_dtype(ser):
        t = pd.to_datetime(ser.astype(str), format="ISO8601", errors="coerce")
        if t.isna().any():
            raise ValueError("дата не разобрана")
    else:
        raise ValueError("тип колонки дат не распознан")
    if getattr(t.dt, "tz", None) is not None:
        t = t.dt.tz_localize(None)
    return t.max()


def _max_cols(cols: list[pd.Series]) -> pd.Timestamp | None:
    m = [x for x in (max_data(c) for c in cols) if x is not None]
    return max(m) if m else None


def daty_parquet(p: str) -> pd.Timestamp | None | str:
    pf = pq.ParquetFile(p)
    sch = pf.schema_arrow
    names = [f.name for f in sch if kolonka_dat(f.name) or pd.api.types.is_datetime64_any_dtype(f.type.to_pandas_dtype())]
    if not names:
        return "NODATES"
    t = pf.read(columns=names)
    return _max_cols([t.column(n).to_pandas() for n in names])


def daty_csv(p: str) -> pd.Timestamp | None | str:
    cols = [c for c in pd.read_csv(p, nrows=0).columns if kolonka_dat(c)]
    if not cols:
        return "NODATES"
    d = pd.read_csv(p, usecols=cols)
    return _max_cols([d[c] for c in cols])


def daty_npz(p: str) -> pd.Timestamp | None | str:
    z = np.load(p, allow_pickle=False)
    out = [pd.Series(z[k].ravel()) for k in z.files if z[k].dtype.kind == "M" or kolonka_dat(k)]
    return _max_cols(out) if out else "NODATES"


CHITATELI = {".parquet": daty_parquet, ".csv": daty_csv, ".npz": daty_npz, ".json": jd.max_data}


def sha256(p: str) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for blok in iter(lambda: f.read(1 << 20), b""):
            h.update(blok)
    return h.hexdigest()


def proverit(p: str, bez_dat: str, do: pd.Timestamp = GRANICA) -> tuple[str, str]:
    """(статус, текст): 'ok' -> текст = max дата или 'без дат'; 'net' -> причина. `do` — верхняя дата (умолчание 2021-12-31)."""
    if any(int(g) > do.year for g in GOD_V_IMENI.findall(Path(p).name)):
        return "net", f"имя содержит 202x (год > {do.year}), не читался"
    f = CHITATELI.get(Path(p).suffix.lower())
    try:
        r = (jd.max_data(p, pit=do > GRANICA) if f is jd.max_data else f(p)) if f else "NODATES"
    except Exception as e:  # noqa: BLE001 — любой сбой чтения = отказ, не пропуск
        return "net", f"не прочитан ({type(e).__name__})"
    if isinstance(r, str):
        return ("ok", "без дат") if bez_dat else ("net", "нет колонки дат и нет обоснования «без дат»")
    if r is None:
        return "net", "нет значений дат"
    if r.normalize() > do:           # время внутри последнего дня допустимо
        return "net", f"max_data {r.date()} > {do.date()}"
    return "ok", str(r.date())


def razobrat(spisok: Path) -> list[tuple[str, str, str]]:
    out = []
    for s in spisok.read_text(encoding="utf-8").splitlines():
        if s.strip() and not s.startswith("#"):
            a = (s.split("\t") + ["", ""])[:3]
            out.append((a[0].strip(), a[1].strip(), a[2].strip()))
    return out


def _otmetka_godna(otmetka) -> bool:
    """Одна проверка отметки на всех: `fastbt.period_v2.otmetka_godna` (файл грузится по пути, без импорта пакета fastbt)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("_period_v2", Path(__file__).resolve().parents[2] / "fastbt/period_v2.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m          # dataclass в period_v2 ищет свой модуль в sys.modules
    try:
        spec.loader.exec_module(m)
        return m.otmetka_godna(otmetka)
    finally:
        sys.modules.pop(spec.name, None)


def verhnyaya_data(do: str, otmetka: Path = OTMETKA) -> pd.Timestamp:
    """Верхняя дата манифеста: > 2021-12-31 — только при отметке снятого замка (непустой файл), иначе PermissionError."""
    d = pd.Timestamp(do)
    if d > GRANICA and not _otmetka_godna(otmetka):
        raise PermissionError(f"--do {do} > 2021-12-31: нет отметки снятого замка {otmetka} (её ставит владелец) — манифест не собирается")
    return d


def sobrat(koren: str, zapisi, do: pd.Timestamp = GRANICA) -> dict:
    ok, net, isk, vid = [], [], [], set()
    for shablon, chit, bez in zapisi:
        for p in sorted(glob.glob(os.path.join(koren, shablon), recursive=True)):
            rel = os.path.relpath(p, koren)
            if rel in vid:
                continue
            vid.add(rel)
            if (pr := zapreshen(p)) is not None:
                isk.append((rel, pr))
            elif os.path.isfile(p):
                st, tx = proverit(p, bez, do)
                if st == "ok":
                    ok.append((rel, sha256(p), os.path.getsize(p), tx, chit))
                else:
                    net.append((rel, tx, chit))
    return {"prinyato": ok, "otvergnuto": net, "isklyucheno": isk}


def zapisat(r: dict, vyhod: Path) -> str:
    vyhod.parent.mkdir(parents=True, exist_ok=True)
    with open(vyhod, "w", encoding="utf-8") as f:
        f.write("put\tsha256\tbajty\tmax_data\tchitatel\n")
        f.writelines("\t".join(map(str, x)) + "\n" for x in r["prinyato"])
    nv = vyhod.with_name(vyhod.stem + "_otvergnuto.tsv")
    with open(nv, "w", encoding="utf-8") as f:
        f.write("put\tprichina\tchitatel\n")
        f.writelines("\t".join(x) + "\n" for x in r["otvergnuto"])
    dat = [x[3] for x in r["prinyato"] if x[3] != "без дат"]
    return (f"файлов {len(r['prinyato'])}, объём {sum(x[2] for x in r['prinyato']) / 1e6:.1f} МБ, "
            f"max дата {max(dat) if dat else '-'}, отвергнуто {len(r['otvergnuto'])}, исключено {len(r['isklyucheno'])}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--koren", required=True)
    ap.add_argument("--spisok", type=Path, required=True)
    ap.add_argument("--vyhod", type=Path, default=Path("docs/server/manifest_dannyh_2021.tsv"))
    ap.add_argument("--do", default="2021-12-31", help=f"верхняя дата; {DO_WF} (wf) — только при отметке снятого замка")
    ap.add_argument("--otmetka", type=Path, default=OTMETKA)
    a = ap.parse_args(argv)
    try:
        do = verhnyaya_data(a.do, a.otmetka)
    except PermissionError as e:
        print(f"FAIL manifest: {e}")
        return 1
    r = sobrat(a.koren, razobrat(a.spisok), do)
    for rel, pr in r["isklyucheno"]:
        print(f"ИСКЛЮЧЕНО до чтения: {rel} ({pr})")
    print(zapisat(r, a.vyhod))
    return 0


if __name__ == "__main__":
    sys.exit(main())
