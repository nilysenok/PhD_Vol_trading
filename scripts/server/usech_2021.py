"""Усечённые копии данных <= 2021-12-31 для сервера (Ж-131 п. 2; Д3 point-in-time).

Граница: `--do` (умолч. 2021-12-31; > неё — wf, только при отметке снятого замка, копии в data/server_wf). Фильтр НА
ЧТЕНИИ: parquet — пакетами iter_batches, csv — построчно; исходники не меняются. Копия: <выход>/<тот же путь>
(по умолч. <koren>/data/server_2021, вне git). Печатаются только пути, счётчики и даты <= границы.
Правило: строка входит, только если ВСЕ перечисленные даты (ПРАВИЛА) <= границы (parquet: < границы+1 день).
Календари: date и known_from — обе. Пустая/неразобранная дата — строка отброшена/ошибка. `retrieved_at` очищается.
ostanov=True — опция для отсортированных файлов; в прогоне выключена (файл дочитывается весь).
Копии заново проходят manifest_perenos.sobrat, манифест обновляется (istochnik).
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
import json_daty as jd  # noqa: E402
import manifest_perenos as mp  # noqa: E402

CUT = dt.date(2021, 12, 31)
PARQUET_BATCH = 65536
from usech_pravila import KAL, PRAVILA  # noqa: E402,F401  (правила — отдельный файл: ≤ 200 строк)


class Stat:
    def __init__(self):
        self.stroki = self.otbrosheno = self.paketov = 0
        self.max_data, self.ostanov = None, False

    def uchest(self, d: str) -> None:
        self.max_data = d if self.max_data is None or d > self.max_data else self.max_data


def _data(s: str) -> dt.date | None:
    s = s.strip()[:10]
    try:
        return dt.date.fromisoformat(s) if s else None
    except ValueError:
        raise ValueError("дата не разобрана") from None      # значение не печатаем


def usech_csv(src: Path, dst: Path, daty: tuple[str, ...], ochistit: tuple[str, ...] = (), ostanov: bool = False) -> Stat:
    st, prev, otsort = Stat(), None, True
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(src, newline="", encoding="utf-8-sig") as fi, open(dst, "w", newline="", encoding="utf-8") as fo:
        rd = csv.reader(ln for ln in fi if not ln.lstrip().startswith("#"))
        w = csv.writer(fo, lineterminator="\n")
        h = next(rd)
        w.writerow(h)
        idx, och = [h.index(c) for c in daty], [h.index(c) for c in ochistit if c in h]
        for r in rd:
            if not r:
                continue
            ds = [_data(r[i]) for i in idx]
            if None in ds:
                st.otbrosheno += 1
                continue
            if all(d <= CUT for d in ds):
                for i in och:
                    r[i] = ""
                w.writerow(r)
                st.stroki += 1
                if ds:                                   # без колонок дат (справочник): строки копируются как есть
                    st.uchest(max(ds).isoformat())
            else:
                st.otbrosheno += 1
            if not ds:
                continue
            if prev is not None and ds[0] < prev:
                otsort = False
            prev = ds[0]
            if ostanov and otsort and ds[0] > CUT:           # по умолчанию выкл.: дочитывается весь файл
                st.ostanov = True
                break
    return st


def _porog(t: pa.DataType) -> pa.Scalar:
    d1 = CUT + dt.timedelta(days=1)                  # строго меньше следующего дня после границы (2021-12-31 или --do)
    if pa.types.is_timestamp(t) and t.tz is None:
        return pa.scalar(dt.datetime.combine(d1, dt.time()), type=t)
    if pa.types.is_date32(t):
        return pa.scalar(d1, type=t)
    if pa.types.is_string(t):
        return pa.scalar(d1.isoformat())
    raise TypeError("тип колонки дат не поддержан")


def usech_pq(src: Path, dst: Path, kol: str, paket: int = PARQUET_BATCH, ostanov: bool = False) -> Stat:
    st, otsort, prev = Stat(), True, None
    pf = pq.ParquetFile(src)
    porog = _porog(pf.schema_arrow.field(kol).type)
    dst.parent.mkdir(parents=True, exist_ok=True)
    with pq.ParquetWriter(dst, pf.schema_arrow) as w:
        for b in pf.iter_batches(batch_size=paket, use_threads=False):
            st.paketov += 1
            col = b.column(kol)
            if col.null_count:
                raise ValueError("пустые даты в колонке")
            a = col.to_numpy(zero_copy_only=False)
            if len(a) > 1 and not bool(np.all(a[1:] >= a[:-1])) or (prev is not None and a[0] < prev):
                otsort = False
            prev = a[-1]
            keep = b.filter(pc.less(col, porog))
            if keep.num_rows:
                w.write_batch(keep)
                st.stroki += keep.num_rows
                st.uchest(str(pc.max(keep.column(kol)).as_py())[:10])
            st.otbrosheno += b.num_rows - keep.num_rows
            if ostanov and otsort and keep.num_rows < b.num_rows:
                st.ostanov = True
                break
    if st.stroki == 0:
        dst.unlink()
    return st


def usech_json(src: Path, dst: Path) -> Stat:
    st = Stat()
    st.stroki, st.otbrosheno, st.max_data = jd.usech(src, dst, CUT)
    return st


def usech(koren: Path, vyhod: Path, tolko: str = "") -> list[dict]:
    zap = []
    for shablon, vid, kol, och, chit in PRAVILA:
        if tolko and tolko not in shablon:
            continue
        for src in sorted(koren.glob(shablon)):
            rel = src.relative_to(koren).as_posix()
            if mp.zapreshen(str(src)) or rel.startswith(("data/server_2021/", "data/server_wf/")):
                continue
            # дата в начале имени справочной таблицы — день создания (2026), не данные; manifest_perenos не читает имена с 202x:
            # копия без префикса даты, в манифесте istochnik = исходное имя (его читает fastbt); свод Д8 без даты — имя не режется
            dst = vyhod / rel
            if rel.startswith("docs/issledovaniya/") and dst.name[:4].isdigit():
                dst = dst.with_name(dst.name.split("_", 1)[1])
            st = usech_pq(src, dst, kol[0]) if vid == "pq" else usech_json(src, dst) if vid == "json" else usech_csv(src, dst, kol, och)
            zap.append(dict(rel=rel, dst=dst, st=st, chit=chit, sha_src=mp.sha256(str(src)), bez=not kol))
            print(f"{rel}: строк {st.stroki}, отброшено {st.otbrosheno}, max {st.max_data}, остановка {st.ostanov}")
    return zap


def obnovit_manifest(koren: Path, zap: list[dict], vyhod: Path, tsv: Path) -> str:
    """Копии заново проверяет manifest_perenos (независимо от усечения) и дописывает в манифест."""
    vkoren = vyhod.parent.parent                 # <vkoren>/data/server_2021: put в манифесте = data/server_2021/<rel>
    dst_rel = {z["dst"].relative_to(vkoren).as_posix(): z for z in zap if z["dst"].exists()}
    bez = "справочник без дат (as-of в source_url ≤ 2021-12-30)"
    r = mp.sobrat(str(vkoren), [(p, z["chit"], bez if z["bez"] else "") for p, z in dst_rel.items()], pd.Timestamp(CUT))
    nov = [(p, sha, b, m, z["chit"], z["rel"], z["sha_src"]) for p, sha, b, m, _ in r["prinyato"] for z in [dst_rel[p]]]
    nv = tsv.with_name(tsv.stem + "_otvergnuto.tsv")
    star = [ln.rstrip("\n").split("\t") for ln in tsv.read_text(encoding="utf-8").splitlines()[1:]]
    star = [(s + [s[0], s[1]])[:7] if len(s) == 5 else s for s in star]
    nov = [x for x in nov if x[5] not in {s[0] for s in star}]       # оригинал ≤ 2021 принят целиком — копия не нужна (дубль пути)
    star = [s for s in star if s[0] not in {x[0] for x in nov}]
    srcs = {z["rel"] for z in zap}
    otv = [ln.split("\t") for ln in nv.read_text(encoding="utf-8").splitlines()[1:]]
    otv = [[o[0], o[1] + ("" if "заменён усечённой копией" in o[1] else f"; заменён усечённой копией ({tsv.name})"), o[2]] if o[0] in srcs else o for o in otv]
    otv += [[p, pr, ch] for p, pr, ch in r["otvergnuto"]]
    with open(tsv, "w", encoding="utf-8") as f:
        f.write("put\tsha256\tbajty\tmax_data\tchitatel\tistochnik\tsha256_istochnika\n")
        f.writelines("\t".join(map(str, x)) + "\n" for x in star + nov)
    with open(nv, "w", encoding="utf-8") as f:
        f.write("put\tprichina\tchitatel\n")
        f.writelines("\t".join(x) + "\n" for x in otv)
    vse = star + nov
    return f"файлов {len(vse)}, объём {sum(int(x[2]) for x in vse) / 1e6:.1f} МБ, max дата {max(x[3] for x in vse if x[3] != 'без дат')}, копий принято {len(nov)} из {len(zap)}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--koren", type=Path, required=True)
    ap.add_argument("--vyhod", type=Path, default=None)
    ap.add_argument("--tolko", default="", help="только правила, чей шаблон содержит подстроку (дополнение манифеста)")
    ap.add_argument("--tsv", type=Path, default=Path("docs/server/manifest_dannyh_2021.tsv"))
    ap.add_argument("--do", default="2021-12-31", help="граница; > 2021-12-31 (wf) — только при отметке снятого замка")
    ap.add_argument("--otmetka", type=Path, default=mp.OTMETKA)
    a = ap.parse_args(argv)
    global CUT
    try:
        CUT = mp.verhnyaya_data(a.do, a.otmetka).date()
    except PermissionError as e:
        return print(f"FAIL usech: {e}") or 1
    vyhod = a.vyhod or a.koren / "data" / ("server_2021" if CUT == dt.date(2021, 12, 31) else "server_wf")
    zap = usech(a.koren, vyhod, a.tolko)
    print(obnovit_manifest(a.koren, zap, vyhod, a.tsv))
    return 0


if __name__ == "__main__":
    sys.exit(main())
