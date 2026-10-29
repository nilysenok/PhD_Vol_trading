"""Усечённые копии данных <= 2021-12-31 для сервера (Ж-131 п. 2; Д3 point-in-time).

Фильтр НА ЧТЕНИИ: parquet — пакетами iter_batches (в памяти один пакет, значения >= 2022 не накапливаются и не
печатаются), csv — построчно. Исходники не меняются. Копия: <выход>/<тот же относительный путь>, по умолчанию
<koren>/data/server_2021 (вне git). Печатаются только пути, счётчики и даты <= 2021.

Правило даты по файлу (колонки в ПРАВИЛА): строка входит, только если ВСЕ перечисленные даты <= 2021-12-31
(для parquet: < 2022-01-01, т.е. внутридневное время 31.12 входит). Календари: date (событие) и known_from
(объявлено) — обе. Пустая/неразобранная дата — строка отброшена/ошибка (не подставляется). `retrieved_at` (дата
скачивания, не дата данных; читателями не используется) в копии очищается. Ранняя остановка (ostanov=True) —
опция для заведомо отсортированных файлов; в прогоне выключена: файл дочитывается весь (иначе строки <= 2021
после неотсортированной «2022» потерялись бы молча), в памяти — один пакет/одна строка.
После копирования копии заново прогоняются через manifest_perenos.sobrat, манифест обновляется (istochnik).
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import sys
from pathlib import Path

import numpy as np
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
        self.max_data: str | None = None
        self.ostanov = False

    def uchest(self, d: str) -> None:
        if self.max_data is None or d > self.max_data:
            self.max_data = d


def _data(s: str) -> dt.date | None:
    s = s.strip()[:10]
    if not s:
        return None
    try:
        return dt.date.fromisoformat(s)
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
    if pa.types.is_timestamp(t) and t.tz is None:
        return pa.scalar(dt.datetime(2022, 1, 1), type=t)
    if pa.types.is_date32(t):
        return pa.scalar(dt.date(2022, 1, 1), type=t)
    if pa.types.is_string(t):
        return pa.scalar("2022-01-01")
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
    st.stroki, st.otbrosheno, st.max_data = jd.usech(src, dst)
    return st


def usech(koren: Path, vyhod: Path, tolko: str = "") -> list[dict]:
    zap = []
    for shablon, vid, kol, och, chit in PRAVILA:
        if tolko and tolko not in shablon:
            continue
        for src in sorted(koren.glob(shablon)):
            rel = src.relative_to(koren).as_posix()
            if mp.zapreshen(str(src)) or rel.startswith("data/server_2021/"):
                continue
            # дата в начале имени справочной таблицы — день её создания (2026), не данные; manifest_perenos не читает файлы с
            # годом 202x в имени, поэтому копия лежит без префикса даты, а в манифесте istochnik = исходное имя (его читает fastbt)
            dst = vyhod / rel
            if rel.startswith("docs/issledovaniya/"):
                dst = dst.with_name(dst.name.split("_", 1)[1])
            st = usech_pq(src, dst, kol[0]) if vid == "pq" else usech_json(src, dst) if vid == "json" else usech_csv(src, dst, kol, och)
            zap.append(dict(rel=rel, dst=dst, st=st, chit=chit, sha_src=mp.sha256(str(src)), bez=not kol))
            print(f"{rel}: строк {st.stroki}, отброшено {st.otbrosheno}, max {st.max_data}, остановка {st.ostanov}")
    return zap


def obnovit_manifest(koren: Path, zap: list[dict], vyhod: Path, tsv: Path) -> str:
    """Копии заново проверяет manifest_perenos (независимо от усечения) и дописывает в манифест."""
    vkoren = vyhod.parent.parent                 # <vkoren>/data/server_2021: put в манифесте = data/server_2021/<rel>
    dst_rel = {z["dst"].relative_to(vkoren).as_posix(): z for z in zap if z["dst"].exists()}
    r = mp.sobrat(str(vkoren), [(p, z["chit"], "справочник без дат (as-of в source_url ≤ 2021-12-30)" if z["bez"] else "") for p, z in dst_rel.items()])
    nov = [(p, sha, b, m, z["chit"], z["rel"], z["sha_src"]) for p, sha, b, m, _ in r["prinyato"] for z in [dst_rel[p]]]
    nv = tsv.with_name(tsv.stem + "_otvergnuto.tsv")
    star = [ln.rstrip("\n").split("\t") for ln in tsv.read_text(encoding="utf-8").splitlines()[1:]]
    star = [(s + [s[0], s[1]])[:7] if len(s) == 5 else s for s in star]
    nov = [x for x in nov if x[5] not in {s[0] for s in star}]       # оригинал ≤ 2021 принят целиком — копия не нужна (дубль пути)
    puti = {x[0] for x in nov}
    star = [s for s in star if s[0] not in puti]
    srcs = {z["rel"] for z in zap}
    otv = [ln.split("\t") for ln in nv.read_text(encoding="utf-8").splitlines()[1:]]
    otv = [[o[0], o[1] + ("" if "заменён усечённой копией" in o[1] else "; заменён усечённой копией (manifest_dannyh_2021.tsv)"), o[2]] if o[0] in srcs else o for o in otv]
    otv += [[p, pr, ch] for p, pr, ch in r["otvergnuto"]]
    with open(tsv, "w", encoding="utf-8") as f:
        f.write("put\tsha256\tbajty\tmax_data\tchitatel\tistochnik\tsha256_istochnika\n")
        f.writelines("\t".join(map(str, x)) + "\n" for x in star + nov)
    with open(nv, "w", encoding="utf-8") as f:
        f.write("put\tprichina\tchitatel\n")
        f.writelines("\t".join(x) + "\n" for x in otv)
    vse = star + nov
    dat = [x[3] for x in vse if x[3] != "без дат"]
    return f"файлов {len(vse)}, объём {sum(int(x[2]) for x in vse) / 1e6:.1f} МБ, max дата {max(dat)}, копий принято {len(nov)} из {len(zap)}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--koren", type=Path, required=True)
    ap.add_argument("--vyhod", type=Path, default=None)
    ap.add_argument("--tolko", default="", help="только правила, чей шаблон содержит подстроку (дополнение манифеста)")
    ap.add_argument("--tsv", type=Path, default=Path("docs/server/manifest_dannyh_2021.tsv"))
    a = ap.parse_args(argv)
    vyhod = a.vyhod or a.koren / "data" / "server_2021"
    zap = usech(a.koren, vyhod, a.tolko)
    print(obnovit_manifest(a.koren, zap, vyhod, a.tsv))
    return 0


if __name__ == "__main__":
    sys.exit(main())
